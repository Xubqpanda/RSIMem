"""Runtime adapter that runs a nanobot agent in-process."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from ...models.content import TextBlock
from ...models.message import Message
from ...models.trace import TokenUsage
from ..protocol import StartSessionRequest, StepRequest, StepResponse
from ..registry import AgentSpec
from .base import RuntimeAdapter
from .cli_agent import _build_cli_prompt
from .http_task_tools import invoke_http_tool, matching_task_tools


class NanobotAdapter(RuntimeAdapter):
    """In-process adapter for the nanobot AI agent framework.

    Runs the nanobot AgentLoop directly in-process in one-shot mode,
    passing the task prompt via process_direct() and returning the
    final response.
    """

    def __init__(self, spec: AgentSpec, request: StartSessionRequest) -> None:
        super().__init__(spec, request)
        self._completed_response: StepResponse | None = None

    def step(self, request: StepRequest) -> StepResponse:
        if self._completed_response is not None:
            return self._completed_response

        if request.step_id > 0 or request.tool_results:
            return StepResponse(
                status="error",
                error="Nanobot adapter runs in one-shot mode; follow-up turns not supported.",
            )

        started = time.monotonic()
        try:
            response = self._run_agent()
        except Exception as exc:
            response = StepResponse(
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )
        response.model_time_s = time.monotonic() - started
        self._completed_response = response
        return response

    def _make_provider(self) -> Any:
        """Create the appropriate nanobot LLM provider from the request model config."""
        model_id = self.request.model.model_id or ""
        api_key = self.request.model.api_key or ""
        base_url = self.request.model.base_url or ""
        extra = self.request.model.extra_body or {}
        nanobot_cfg = extra.get("nanobot", {}) if isinstance(extra, dict) else {}

        provider_type = nanobot_cfg.get("provider", "")

        # Auto-detect provider from model_id or base_url
        if not provider_type:
            if "anthropic" in base_url or model_id.startswith("claude"):
                provider_type = "anthropic"
            else:
                provider_type = "openai_compat"

        from nanobot.providers.base import GenerationSettings

        temperature = float(self.request.runtime_config.temperature)
        max_tokens = int(nanobot_cfg.get("max_tokens", 4096))

        if provider_type == "anthropic":
            from nanobot.providers.anthropic_provider import AnthropicProvider

            # Only skip api_base for the real Anthropic API; third-party
            # anthropic-compatible endpoints (GLM, MiniMax) need it set.
            is_real_anthropic = base_url and "api.anthropic.com" in base_url
            provider = AnthropicProvider(
                api_key=api_key,
                api_base=None if is_real_anthropic or not base_url else base_url,
                default_model=model_id,
            )
        else:
            from nanobot.providers.openai_compat_provider import OpenAICompatProvider

            provider = OpenAICompatProvider(
                api_key=api_key,
                api_base=base_url or "https://openrouter.ai/api/v1",
                default_model=model_id,
            )

        provider.generation = GenerationSettings(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return provider

    def _run_agent(self) -> StepResponse:
        # Ensure nanobot is importable
        nanobot_path = Path(__file__).resolve().parent.parent.parent.parent.parent / "agents" / "nanobot"
        if str(nanobot_path) not in sys.path:
            sys.path.insert(0, str(nanobot_path))

        from nanobot.agent.loop import AgentLoop
        from nanobot.agent.tools.base import Tool as NanobotTool
        from nanobot.bus.queue import MessageBus

        extra = self.request.model.extra_body or {}
        nanobot_cfg = extra.get("nanobot", {}) if isinstance(extra, dict) else {}

        provider = self._make_provider()

        workspace_dir = str(nanobot_cfg.get("workspace_dir") or "").strip()
        if workspace_dir:
            workspace = Path(workspace_dir).expanduser().resolve()
            workspace.mkdir(parents=True, exist_ok=True)
        else:
            import tempfile

            workspace = Path(tempfile.mkdtemp(prefix="past-bench-nanobot-"))

        session_key = str(nanobot_cfg.get("session_key") or "claw:eval")
        artifacts_dir = str(nanobot_cfg.get("artifacts_dir") or "").strip()

        agent = AgentLoop(
            bus=MessageBus(),
            provider=provider,
            workspace=workspace,
            model=self.request.model.model_id or "claude-sonnet-4-6",
            max_iterations=int(nanobot_cfg.get("max_iterations", 30)),
            context_window_tokens=int(nanobot_cfg.get("context_window", 65536)),
            restrict_to_workspace=bool(nanobot_cfg.get("restrict_to_workspace", False)),
        )
        self._register_task_tools(agent, NanobotTool)

        session = agent.sessions.get_or_create(session_key)
        prior_history = bool(session.messages)
        forced_consolidation = False
        forced_consolidation_error = ""

        prompt = _build_cli_prompt(self.request)
        timeout = max(30, self.request.timeout_seconds)

        response = asyncio.run(
            asyncio.wait_for(
                agent.process_direct(
                    content=prompt,
                    session_key=session_key,
                    channel="cli",
                    chat_id="eval",
                ),
                timeout=timeout,
            )
        )

        if bool(nanobot_cfg.get("persistence_enabled", False)):
            try:
                forced_consolidation = asyncio.run(
                    self._force_memory_consolidation(agent, session_key)
                )
            except Exception as exc:
                forced_consolidation_error = f"{type(exc).__name__}: {exc}"

        # Cleanup MCP connections
        try:
            asyncio.run(agent.close_mcp())
        except Exception:
            pass

        self._capture_artifacts(
            workspace=workspace,
            artifacts_dir=Path(artifacts_dir).expanduser().resolve() if artifacts_dir else None,
            metadata={
                "session_key": session_key,
                "prior_history": prior_history,
                "forced_consolidation": forced_consolidation,
                "forced_consolidation_error": forced_consolidation_error,
            },
        )

        final_text = ""
        if response:
            final_text = response.content or ""

        assistant = Message(role="assistant", content=[TextBlock(text=final_text)])
        return StepResponse(
            status="finished",
            assistant_message=assistant,
            usage=TokenUsage(),
            final_output=final_text,
        )

    def close(self, reason: str = "") -> None:
        pass

    @staticmethod
    async def _force_memory_consolidation(agent: Any, session_key: str) -> bool:
        """Force nanobot's native memory consolidation at the episode boundary."""
        consolidator = getattr(agent, "memory_consolidator", None)
        if consolidator is None:
            return False
        session = agent.sessions.get_or_create(session_key)
        if session.last_consolidated >= len(session.messages):
            return False

        lock = consolidator.get_lock(session.key)
        async with lock:
            pending = session.messages[session.last_consolidated:]
            if not pending:
                return False
            archived = await consolidator.archive_messages(pending)
            if archived:
                session.last_consolidated = len(session.messages)
                agent.sessions.save(session)
            return archived

    def _register_task_tools(self, agent: Any, nanobot_tool_base: type[Any]) -> None:
        for tool, endpoint in matching_task_tools(
            self.request.tools or [],
            self.request.tool_endpoints or [],
        ):
            registry_tool = self._build_task_tool(
                nanobot_tool_base,
                name=tool.name,
                description=tool.description,
                schema=tool.input_schema or {"type": "object", "properties": {}},
                url=endpoint.url,
                method=endpoint.method or "POST",
            )
            agent.tools.register(registry_tool)

    @staticmethod
    def _build_task_tool(
        nanobot_tool_base: type[Any],
        *,
        name: str,
        description: str,
        schema: dict[str, Any],
        url: str,
        method: str,
    ) -> Any:
        class _BenchTaskTool(nanobot_tool_base):
            @property
            def name(self) -> str:
                return name

            @property
            def description(self) -> str:
                return description

            @property
            def parameters(self) -> dict[str, Any]:
                return schema

            async def execute(self, **kwargs: Any) -> Any:
                return invoke_http_tool(
                    url=url,
                    method=method,
                    arguments=kwargs,
                    timeout=30,
                )

        return _BenchTaskTool()

    @staticmethod
    def _capture_artifacts(
        *,
        workspace: Path,
        artifacts_dir: Path | None,
        metadata: dict[str, Any],
    ) -> None:
        if artifacts_dir is None:
            return
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        for name in ("memory", "skills", "sessions", "AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"):
            src = workspace / name
            dst = artifacts_dir / name
            if not src.exists():
                continue
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        (artifacts_dir / "nanobot_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
