"""Runtime adapters that drive official coding CLIs inside the runtime container."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from ...models.content import TextBlock
from ...models.message import Message
from ...models.trace import TokenUsage
from ..protocol import StartSessionRequest, StepRequest, StepResponse
from ..registry import AgentSpec
from .base import RuntimeAdapter


def _collect_text(message: Message) -> str:
    parts: list[str] = []
    for block in message.content:
        if block.type == "text":
            parts.append(block.text)
        elif block.type == "tool_result":
            text = "\n".join(item.text for item in block.content if item.type == "text")
            parts.append(f"[tool_result {block.tool_use_id}]\n{text}")
        elif block.type == "tool_use":
            parts.append(f"[tool_use {block.name}] {json.dumps(block.input, ensure_ascii=True)}")
        elif block.type == "image":
            parts.append("[image attachment omitted by CLI adapter]")
        elif block.type == "audio":
            parts.append("[audio attachment omitted by CLI adapter]")
        elif block.type == "video":
            parts.append("[video attachment omitted by CLI adapter]")
    return "\n".join(part for part in parts if part)


def _build_http_tool_section(request: StartSessionRequest) -> str:
    if not request.tools:
        return "No external HTTP tools are available for this task."

    endpoint_map = {endpoint.tool_name: endpoint for endpoint in request.tool_endpoints}
    missing = [tool.name for tool in request.tools if tool.name not in endpoint_map]
    if missing:
        joined = ", ".join(sorted(missing))
        raise RuntimeError(
            "This CLI adapter currently supports only HTTP-backed tools. "
            f"Missing endpoints for: {joined}"
        )

    lines = [
        "Available HTTP tools. Use Bash plus curl or Python stdlib to call the exact endpoints below.",
        "Do not invent endpoints, and do not ask the user to run commands for you.",
    ]
    for tool in request.tools:
        endpoint = endpoint_map[tool.name]
        schema = json.dumps(tool.input_schema or {"type": "object", "properties": {}}, ensure_ascii=True)
        lines.extend(
            [
                f"- {tool.name}",
                f"  Description: {tool.description}",
                f"  HTTP: {endpoint.method.upper()} {endpoint.url}",
                f"  JSON Schema: {schema}",
            ]
        )
    return "\n".join(lines)


def _build_cli_prompt(request: StartSessionRequest) -> str:
    conversation = []
    for message in request.initial_messages:
        text = _collect_text(message)
        if text:
            conversation.append(f"[{message.role.upper()}]\n{text}")

    instructions = [
        "You are running inside the past-bench runtime container as a real coding agent CLI.",
        "Complete the benchmark task autonomously and return the final answer only.",
        "If HTTP tools are listed below, call them yourself from Bash using the exact URLs.",
        "Do not ask the user for clarification or for manual command execution.",
    ]
    return "\n\n".join(
        [
            "\n".join(instructions),
            f"Task: {request.task_id} - {request.task_name}",
            _build_http_tool_section(request),
            "Conversation:\n" + "\n\n".join(conversation),
        ]
    )


def _with_no_proxy(env: dict[str, str], host: str) -> dict[str, str]:
    updated = dict(env)
    existing = [value for key in ("no_proxy", "NO_PROXY") for value in [updated.get(key)] if value]
    merged = ",".join(part for part in [*existing, host] if part)
    updated["no_proxy"] = merged
    updated["NO_PROXY"] = merged
    return updated


def _usage_from_payload(payload: dict[str, Any] | None) -> TokenUsage:
    if not isinstance(payload, dict):
        return TokenUsage()
    return TokenUsage(
        input_tokens=int(payload.get("input_tokens") or 0),
        output_tokens=int(payload.get("output_tokens") or 0),
    )


class _OneShotCliAdapter(RuntimeAdapter):
    """Base class for official CLI integrations that complete the task in one subprocess run."""

    cli_name = ""

    def __init__(self, spec: AgentSpec, request: StartSessionRequest) -> None:
        super().__init__(spec, request)
        self._completed_response: StepResponse | None = None
        self._temp_root = Path(tempfile.mkdtemp(prefix=f"claw-{self.cli_name or 'cli'}-"))
        self._home_dir = self._temp_root / "home"
        self._work_dir = self._temp_root / "workspace"
        self._home_dir.mkdir(parents=True, exist_ok=True)
        self._work_dir.mkdir(parents=True, exist_ok=True)

    def step(self, request: StepRequest) -> StepResponse:
        if self._completed_response is not None:
            return self._completed_response
        if request.step_id > 0 or request.tool_results:
            return StepResponse(
                status="error",
                error=(
                    f"{self.spec.name} currently runs the official CLI in one-shot mode and "
                    "does not support bench-managed follow-up tool turns yet."
                ),
            )

        started = time.monotonic()
        try:
            prompt = _build_cli_prompt(self.request)
            response = self._run_cli(prompt)
        except subprocess.TimeoutExpired:
            response = StepResponse(
                status="error",
                error=f"{self.spec.name} CLI timed out after {self.request.timeout_seconds}s",
            )
        except Exception as exc:
            response = StepResponse(
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )
        response.model_time_s = time.monotonic() - started
        self._completed_response = response
        return response

    def close(self, reason: str = "") -> None:
        shutil.rmtree(self._temp_root, ignore_errors=True)

    @property
    def timeout_seconds(self) -> int:
        return max(30, self.request.timeout_seconds)

    def _base_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["HOME"] = str(self._home_dir)
        env["XDG_CONFIG_HOME"] = str(self._home_dir / ".config")
        return _with_no_proxy(env, "host.docker.internal")

    def _run_cli(self, prompt: str) -> StepResponse:
        raise NotImplementedError


class CodexCliAdapter(_OneShotCliAdapter):
    """Real adapter backed by the official Codex CLI."""

    cli_name = "codex"

    def _login_with_api_key(self, env: dict[str, str]) -> None:
        api_key = self.request.model.api_key
        if not api_key:
            raise RuntimeError("Codex CLI requires an OpenAI API key")
        env["OPENAI_API_KEY"] = api_key
        login = subprocess.run(
            ["codex", "login", "--with-api-key"],
            input=f"{api_key}\n",
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            cwd=self._work_dir,
        )
        if login.returncode != 0:
            raise RuntimeError(f"codex login failed: {login.stderr.strip() or login.stdout.strip()}")

    def _run_cli(self, prompt: str) -> StepResponse:
        env = self._base_env()
        self._login_with_api_key(env)
        command = [
            "codex",
            "--dangerously-bypass-approvals-and-sandbox",
            "exec",
            "--json",
            "--skip-git-repo-check",
            "-C",
            str(self._work_dir),
            "-m",
            self.request.model.model_id,
            "-",
        ]
        proc = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            env=env,
            cwd=self._work_dir,
        )

        usage = TokenUsage()
        final_text = ""
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("type") == "item.completed":
                item = payload.get("item") or {}
                if item.get("type") == "agent_message" and item.get("text"):
                    final_text = str(item["text"]).strip()
            elif payload.get("type") == "turn.completed":
                raw_usage = payload.get("usage") or {}
                usage = TokenUsage(
                    input_tokens=int(raw_usage.get("input_tokens") or 0),
                    output_tokens=int(raw_usage.get("output_tokens") or 0),
                )

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            return StepResponse(status="error", error=detail or "codex exec failed")

        if not final_text:
            final_text = (proc.stderr or proc.stdout).strip()
        assistant = Message(role="assistant", content=[TextBlock(text=final_text)])
        return StepResponse(
            status="finished",
            assistant_message=assistant,
            usage=usage,
            final_output=final_text,
        )


class ClaudeCodeCliAdapter(_OneShotCliAdapter):
    """Real adapter backed by the official Claude Code CLI."""

    cli_name = "claude"

    def __init__(self, spec: AgentSpec, request: StartSessionRequest) -> None:
        super().__init__(spec, request)
        # Claude Code requires a minimal ~/.claude/settings.json to start without errors.
        claude_dir = self._home_dir / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)
        (claude_dir / "settings.json").write_text("{}")

    @staticmethod
    def _claude_code_settings(extra_body: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(extra_body, dict):
            return {}
        settings = extra_body.get("claude_code")
        return settings if isinstance(settings, dict) else {}

    @staticmethod
    def _cli_model(settings: dict[str, Any], fallback: str) -> str:
        configured = settings.get("cli_model")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
        return fallback

    @staticmethod
    def _should_set_model_env(settings: dict[str, Any]) -> bool:
        raw = settings.get("set_model_env")
        if isinstance(raw, bool):
            return raw
        return True

    def _run_cli(self, prompt: str) -> StepResponse:
        env = self._base_env()
        for key in (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_MODEL",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "ANTHROPIC_DEFAULT_SONNET_MODEL",
        ):
            env.pop(key, None)

        settings = self._claude_code_settings(self.request.model.extra_body)
        auth_env = str(settings.get("auth_env") or "ANTHROPIC_API_KEY")
        configured_env = settings.get("env") or {}
        cli_model = self._cli_model(settings, self.request.model.model_id)
        set_model_env = self._should_set_model_env(settings)
        if isinstance(configured_env, dict):
            for key, value in configured_env.items():
                env[str(key)] = str(value)

        base_url = self.request.model.base_url
        # Skip setting ANTHROPIC_BASE_URL for the default Anthropic API endpoint; setting it
        # explicitly confuses the Claude Code CLI and causes spurious model-not-found errors.
        _DEFAULT_ANTHROPIC_URL = "https://api.anthropic.com/v1"
        if base_url and base_url.rstrip("/") != _DEFAULT_ANTHROPIC_URL.rstrip("/"):
            env.setdefault("ANTHROPIC_BASE_URL", base_url)
        if self.request.model.model_id and set_model_env:
            env.setdefault("ANTHROPIC_MODEL", self.request.model.model_id)
        if self.request.model.api_key:
            env[auth_env] = self.request.model.api_key

        command = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--model",
            cli_model,
            "--allowedTools",
            "Bash",
        ]
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            env=env,
            cwd=self._work_dir,
        )

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            return StepResponse(status="error", error=detail or "claude failed")

        raw = (proc.stdout or "").strip()
        try:
            payload: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"type": "result", "subtype": "success", "result": raw, "is_error": False}

        if payload.get("is_error"):
            return StepResponse(
                status="error",
                error=str(payload.get("result") or payload.get("error") or "claude returned error"),
            )

        final_text = str(payload.get("result") or "").strip()
        assistant = Message(role="assistant", content=[TextBlock(text=final_text)])
        return StepResponse(
            status="finished",
            assistant_message=assistant,
            usage=_usage_from_payload(payload.get("usage")),
            final_output=final_text,
        )
