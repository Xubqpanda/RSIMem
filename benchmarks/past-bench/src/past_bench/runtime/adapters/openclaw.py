"""OpenClaw adapter backed by the local OpenClaw Gateway OpenResponses API."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from ...models.content import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock
from ...models.message import Message
from ...models.tool import ToolSpec
from ...models.trace import TokenUsage
from ..protocol import StartSessionRequest, StepRequest, StepResponse, ToolCallAction
from ..registry import AgentSpec
from .base import RuntimeAdapter
from .cli_agent import _build_cli_prompt, _with_no_proxy


def _tail_text(path: Path, *, limit: int = 4000) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    return data[-limit:]


def _tool_to_openclaw_definition(tool: ToolSpec) -> dict[str, Any]:
    parameters = tool.input_schema or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        },
    }


def _collect_text(blocks: list[TextBlock]) -> str:
    return "\n".join(block.text for block in blocks if block.text)


def _message_to_openclaw_items(message: Message) -> list[dict[str, Any]]:
    text_parts: list[str] = []
    image_items: list[dict[str, Any]] = []

    for block in message.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "image":
            image_block = block if isinstance(block, ImageBlock) else ImageBlock.model_validate(block)
            image_items.append(
                {
                    "type": "input_image",
                    "source": {
                        "type": "base64",
                        "media_type": image_block.mime_type,
                        "data": image_block.data,
                    },
                }
            )
        elif block.type in {"audio", "video"}:
            raise RuntimeError(
                f"OpenClaw adapter does not support {block.type} inputs yet. "
                f"Task media must be text/image for the current adapter."
            )

    items: list[dict[str, Any]] = [
        {
            "type": "message",
            "role": message.role,
            "content": "\n".join(part for part in text_parts if part),
        }
    ]
    items.extend(image_items)
    return items


def _tool_results_to_openclaw_items(tool_results: list[ToolResultBlock]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for result in tool_results:
        output = _collect_text(result.content)
        if result.is_error and output:
            output = f"ERROR: {output}"
        items.append(
            {
                "type": "function_call_output",
                "call_id": result.tool_use_id,
                "output": output,
            }
        )
    return items


def _parse_function_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw in (None, ""):
        return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _extract_text_from_message_item(item: dict[str, Any]) -> tuple[list[TextBlock], str | None]:
    content = item.get("content")
    blocks: list[TextBlock] = []
    reasoning_chunks: list[str] = []

    if isinstance(content, str):
        if content:
            blocks.append(TextBlock(text=content))
        return blocks, None

    if not isinstance(content, list):
        return blocks, None

    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        text = part.get("text") or part.get("content") or ""
        if part_type in {"output_text", "text"} and text:
            blocks.append(TextBlock(text=text))
        elif part_type in {"reasoning", "summary_text"} and text:
            reasoning_chunks.append(text)

    reasoning = "\n".join(reasoning_chunks) if reasoning_chunks else None
    return blocks, reasoning


def _usage_from_openclaw_response(payload: dict[str, Any]) -> TokenUsage:
    usage = payload.get("usage") or {}
    return TokenUsage(
        input_tokens=int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
    )


def _response_to_step_response(payload: dict[str, Any], *, model_time_s: float) -> StepResponse:
    content_blocks: list[TextBlock | ToolUseBlock] = []
    tool_calls: list[ToolCallAction] = []
    reasoning_chunks: list[str] = []

    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type == "function_call":
            call_id = item.get("call_id") or item.get("id") or f"call_{uuid4().hex[:8]}"
            name = item.get("name") or item.get("function", {}).get("name")
            if not name:
                continue
            arguments = _parse_function_arguments(item.get("arguments"))
            content_blocks.append(ToolUseBlock(id=str(call_id), name=str(name), input=arguments))
            tool_calls.append(
                ToolCallAction(
                    tool_use_id=str(call_id),
                    name=str(name),
                    arguments=arguments,
                )
            )
            continue

        if item_type == "message":
            text_blocks, reasoning = _extract_text_from_message_item(item)
            content_blocks.extend(text_blocks)
            if reasoning:
                reasoning_chunks.append(reasoning)
            continue

        if item_type == "reasoning":
            summary = item.get("summary") or ""
            if isinstance(summary, str) and summary:
                reasoning_chunks.append(summary)

    if not content_blocks and payload.get("output_text"):
        content_blocks.append(TextBlock(text=str(payload["output_text"])))

    assistant_message = Message(
        role="assistant",
        content=content_blocks,
        reasoning_content="\n".join(reasoning_chunks) if reasoning_chunks else None,
    )
    if tool_calls:
        return StepResponse(
            status="acting",
            assistant_message=assistant_message,
            usage=_usage_from_openclaw_response(payload),
            tool_calls=tool_calls,
            model_time_s=model_time_s,
        )
    return StepResponse(
        status="finished",
        assistant_message=assistant_message,
        usage=_usage_from_openclaw_response(payload),
        final_output=assistant_message.text,
        model_time_s=model_time_s,
    )


class OpenClawResponsesAdapter(RuntimeAdapter):
    """Runtime adapter backed by the local OpenClaw CLI in embedded mode."""

    def __init__(self, spec: AgentSpec, request: StartSessionRequest) -> None:
        super().__init__(spec, request)
        self._completed_response: StepResponse | None = None
        self._temp_root = Path(tempfile.mkdtemp(prefix="claw-openclaw-"))
        self._workspace_dir = self._temp_root / "workspace"
        self._state_dir = self._temp_root / "state"
        self._config_path = self._temp_root / "openclaw.json"
        self._workspace_dir.mkdir(parents=True, exist_ok=True)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._write_config()

    def step(self, request: StepRequest) -> StepResponse:
        started = time.monotonic()
        if self._completed_response is not None:
            return self._completed_response
        if request.step_id > 0 or request.tool_results:
            return StepResponse(
                status="error",
                error=(
                    "openclaw currently runs the embedded CLI in one-shot mode and "
                    "does not support bench-managed follow-up tool turns yet."
                ),
            )

        try:
            prompt = _build_cli_prompt(self.request)
            response = self._run_cli(prompt)
        except subprocess.TimeoutExpired:
            response = StepResponse(
                status="error",
                error=f"openclaw CLI timed out after {max(30, self.request.timeout_seconds)}s",
            )
        except Exception as exc:
            response = StepResponse(status="error", error=f"{type(exc).__name__}: {exc}")
        response.model_time_s = time.monotonic() - started
        self._completed_response = response
        return response

    def interrupt(self, reason: str = "") -> None:
        return None

    def close(self, reason: str = "") -> None:
        shutil.rmtree(self._temp_root, ignore_errors=True)

    def _base_env(self) -> dict[str, str]:
        env = _with_no_proxy(os.environ.copy(), "host.docker.internal")
        env["HOME"] = str(self._temp_root / "home")
        env["XDG_CONFIG_HOME"] = str(Path(env["HOME"]) / ".config")
        env["OPENCLAW_CONFIG_PATH"] = str(self._config_path)
        env["OPENCLAW_STATE_DIR"] = str(self._state_dir)
        if self.request.model.api_key:
            env["OPENCLAW_UPSTREAM_API_KEY"] = self.request.model.api_key
        return env

    def _run_cli(self, prompt: str) -> StepResponse:
        proc = subprocess.run(
            [
                "openclaw",
                "agent",
                "--local",
                "--agent",
                "default",
                "--message",
                prompt,
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=max(30, self.request.timeout_seconds),
            env=self._base_env(),
            cwd=self._workspace_dir,
        )
        parsed = _response_from_openclaw_agent_output(proc.stdout, proc.stderr)
        if parsed.status == "finished":
            return parsed
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            return StepResponse(status="error", error=detail or "openclaw agent failed")
        return parsed

    def _write_config(self) -> None:
        extra = self.request.model.extra_body or {}
        openclaw_cfg = extra.get("openclaw", {}) if isinstance(extra, dict) else {}
        if not isinstance(openclaw_cfg, dict):
            openclaw_cfg = {}

        model_id = self.request.model.model_id
        base_url = self.request.model.base_url or "https://api.openai.com/v1"
        provider_id = str(openclaw_cfg.get("provider_id") or "bench")
        provider_api = str(openclaw_cfg.get("provider_api") or "openai-completions")
        input_modalities = openclaw_cfg.get("input_modalities") or ["text", "image"]
        if not isinstance(input_modalities, list) or not input_modalities:
            input_modalities = ["text", "image"]
        context_window = int(
            openclaw_cfg.get("context_window")
            or max(32768, self.request.runtime_config.max_output_tokens * 16)
        )
        max_tokens = int(
            openclaw_cfg.get("max_tokens")
            or max(1024, self.request.runtime_config.max_output_tokens)
        )
        model_ref = f"{provider_id}/{model_id}"
        provider: dict[str, Any] = {
            "api": provider_api,
            "baseUrl": base_url,
            "models": [
                {
                    "id": model_id,
                    "name": model_id,
                    "input": input_modalities,
                    "contextWindow": context_window,
                    "maxTokens": max_tokens,
                }
            ],
        }
        if self.request.model.api_key:
            provider["apiKey"] = "${OPENCLAW_UPSTREAM_API_KEY}"
        provider_overrides = openclaw_cfg.get("provider")
        if isinstance(provider_overrides, dict):
            provider.update(provider_overrides)

        config = {
            "gateway": {
                "mode": "local",
                "port": 18789,
                "bind": "loopback",
                "auth": {"mode": "none"},
                "controlUi": {"enabled": False},
                "http": {
                    "endpoints": {
                        "responses": {"enabled": True},
                    }
                },
            },
            "tools": {
                "profile": "minimal",
                "deny": ["session_status"],
            },
            "agents": {
                "defaults": {
                    "skipBootstrap": True,
                    "workspace": str(self._workspace_dir),
                    "sandbox": {"mode": "off"},
                    "params": {"temperature": self.request.runtime_config.temperature},
                    "model": {"primary": model_ref},
                },
                "list": [
                    {
                        "id": "default",
                        "workspace": str(self._workspace_dir),
                        "params": {"temperature": self.request.runtime_config.temperature},
                        "model": {"primary": model_ref},
                    }
                ],
            },
            "models": {
                "mode": "merge",
                "providers": {
                    provider_id: provider,
                },
            },
        }
        self._config_path.write_text(json.dumps(config, ensure_ascii=True, indent=2), encoding="utf-8")


def _response_from_openclaw_agent_output(stdout: str, stderr: str) -> StepResponse:
    payload = None
    decoder = json.JSONDecoder()
    candidates = [stdout, stderr]
    if stdout or stderr:
        candidates.extend(
            [
                "\n".join(part for part in [stdout, stderr] if part),
                "\n".join(part for part in [stderr, stdout] if part),
            ]
        )

    for stream in candidates:
        if not stream:
            continue
        json_start = stream.find("{")
        while json_start >= 0:
            try:
                payload, _ = decoder.raw_decode(stream[json_start:])
                break
            except json.JSONDecodeError:
                json_start = stream.find("{", json_start + 1)
        if isinstance(payload, dict):
            break

    if not isinstance(payload, dict):
        detail = (stderr or stdout).strip()
        return StepResponse(status="error", error=detail or "openclaw agent produced no JSON result")

    final_chunks: list[str] = []
    for item in payload.get("payloads") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if text:
            final_chunks.append(text)
    final_text = "\n\n".join(final_chunks).strip()
    if not final_text:
        final_text = str(payload.get("meta", {}).get("error") or "").strip() or (stderr or stdout).strip()

    raw_usage = (
        payload.get("meta", {}).get("agentMeta", {}).get("lastCallUsage")
        or payload.get("meta", {}).get("agentMeta", {}).get("usage")
        or {}
    )
    usage = TokenUsage(
        input_tokens=int(raw_usage.get("input") or raw_usage.get("prompt") or 0),
        output_tokens=int(raw_usage.get("output") or raw_usage.get("completion") or 0),
    )

    assistant = Message(role="assistant", content=[TextBlock(text=final_text)])
    return StepResponse(
        status="finished",
        assistant_message=assistant,
        usage=usage,
        final_output=final_text,
    )
