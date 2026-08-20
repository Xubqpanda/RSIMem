"""Runtime adapter that runs an Agent Zero agent in-process."""

from __future__ import annotations

import os
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
from .http_task_tools import build_agent_zero_tool_module_source, matching_task_tools

_AGENT_ZERO_ROOT = Path(__file__).resolve().parents[4] / "agents" / "agent-zero"


class AgentZeroAdapter(RuntimeAdapter):
    """In-process adapter for the Agent Zero framework.

    Runs Agent Zero directly in-process via its initialize/communicate API,
    keeping evaluation of the agent framework separate from task definitions.
    """

    def __init__(self, spec: AgentSpec, request: StartSessionRequest) -> None:
        super().__init__(spec, request)
        self._completed_response: StepResponse | None = None
        self._written_tool_paths: list[Path] = []

    def step(self, request: StepRequest) -> StepResponse:
        if self._completed_response is not None:
            return self._completed_response

        if request.step_id > 0 or request.tool_results:
            return StepResponse(
                status="error",
                error="Agent Zero adapter runs in one-shot mode; follow-up turns not supported.",
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

    def _run_agent(self) -> StepResponse:
        agent_zero_root = str(_AGENT_ZERO_ROOT)
        if agent_zero_root not in sys.path:
            sys.path.insert(0, agent_zero_root)

        # Configure Agent Zero via environment variables before importing
        extra = self.request.model.extra_body or {}
        a0_cfg = extra.get("agent_zero", {}) if isinstance(extra, dict) else {}

        # Set API key for the provider via Agent Zero's env var convention
        model_id = self.request.model.model_id or ""
        api_key = self.request.model.api_key or ""

        if "/" in model_id:
            provider = model_id.split("/")[0]
            # Strip provider prefix from model_id — Agent Zero's models.py
            # re-prepends it as f"{provider}/{model}" for LiteLLM.
            model_name = model_id.split("/", 1)[1]
        else:
            provider = a0_cfg.get("provider", "openai")
            model_name = model_id

        # Set API key in Agent Zero's expected format AND standard env vars
        # (LiteLLM reads standard vars like OPENAI_API_KEY, ANTHROPIC_API_KEY)
        os.environ[f"API_KEY_{provider.upper()}"] = api_key
        provider_env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "zai_anthropic": "ZAI_API_KEY",
            "zai": "ZAI_API_KEY",
            "kimi": "KIMI_CODE_API_KEY",
            "minimax": "MINIMAX_API_KEY",
            "minimax_global": "MINIMAX_API_KEY",
        }
        std_env = provider_env_map.get(provider.lower())
        if std_env:
            os.environ[std_env] = api_key

        # Set RFC password (required by Agent Zero's plugin system)
        os.environ["RFC_PASSWORD"] = a0_cfg.get("rfc_password", "past-bench")

        # Ensure usr/ directory exists (Agent Zero requires it)
        usr_dir = _AGENT_ZERO_ROOT / "usr"
        usr_dir.mkdir(parents=True, exist_ok=True)
        (usr_dir / ".env").touch(exist_ok=True)

        # Write _model_config plugin override so Agent Zero uses the
        # correct provider/model instead of its default (openrouter).
        plugin_cfg_dir = usr_dir / "plugins" / "_model_config"
        plugin_cfg_dir.mkdir(parents=True, exist_ok=True)
        base_url = self.request.model.base_url or ""
        chat_model_cfg: dict[str, Any] = {
            "provider": provider,
            "name": model_name,
            "api_key": api_key,
            "ctx_length": 128000,
            "ctx_history": 0.7,
            "vision": True,
        }
        if base_url:
            chat_model_cfg["api_base"] = base_url
        utility_model_cfg: dict[str, Any] = {
            "provider": provider,
            "name": a0_cfg.get("utility_model", model_name),
            "api_key": api_key,
            "ctx_length": 128000,
            "ctx_input": 0.7,
        }
        if base_url:
            utility_model_cfg["api_base"] = base_url
        model_cfg = {
            "chat_model": chat_model_cfg,
            "utility_model": utility_model_cfg,
            "embedding_model": {
                "provider": "huggingface",
                "name": "sentence-transformers/all-MiniLM-L6-v2",
            },
        }
        import json as _json
        (plugin_cfg_dir / "config.json").write_text(
            _json.dumps(model_cfg, indent=2), encoding="utf-8"
        )

        # Import Agent Zero runtime and mark as "dockerized" so that
        # call_development_function() executes functions directly instead of
        # making RFC HTTP calls to a non-existent web UI server (port 55080).
        from helpers import runtime as a0_runtime
        a0_runtime.args["dockerized"] = True

        # Override files functions so /a0/ paths resolve to the real
        # agent-zero directory (not a Docker-only path).
        from helpers import files as a0_files
        a0_files._base_dir = agent_zero_root
        # Prevent conversion to /a0/ paths — keep real absolute paths
        a0_files.normalize_a0_path = lambda path: path
        a0_files.get_abs_path_dockerized = a0_files.get_abs_path

        # Import Agent Zero modules
        from initialize import initialize_agent
        from agent import Agent as A0Agent, AgentContext, UserMessage

        # --- past_bench task-tool wiring ---------------------------------
        # Agent Zero discovers tools by importing files under usr/tools/
        # (and a few other locations). Write one file per task-defined
        # tool so the agent can actually invoke them via its normal
        # tool-call protocol; each file is a thin Tool subclass that
        # POSTs the JSON args to the matching mock-service URL from
        # task.tool_endpoints.
        a0_tools_dir = usr_dir / "tools"
        a0_tools_dir.mkdir(parents=True, exist_ok=True)
        self._written_tool_paths = []
        for tool, endpoint in matching_task_tools(
            self.request.tools or [],
            self.request.tool_endpoints or [],
        ):
            tool_file = a0_tools_dir / f"{tool.name}.py"
            tool_source = build_agent_zero_tool_module_source(
                url=endpoint.url,
                method=endpoint.method or "POST",
            )
            tool_file.write_text(tool_source, encoding="utf-8")
            self._written_tool_paths.append(tool_file)

        # Agent Zero's stock validate_tool_request rejects empty
        # tool_args ({} fails `not tool_args`), which breaks tools whose
        # JSON schema marks all params optional (e.g.
        # helpdesk_list_tickets). Relax the check so an empty dict is
        # accepted; non-dict values still raise.
        async def _lenient_validate(self, tool_request):  # type: ignore[no-redef]
            if not isinstance(tool_request, dict):
                raise ValueError("Tool request must be a dictionary")
            tool_name = tool_request.get("tool_name", tool_request.get("tool"))
            if not tool_name or not isinstance(tool_name, str):
                raise ValueError("Tool request must have a tool_name (type string) field")
            # Normalize multiple possible arg-field names. Stock Agent Zero
            # only checks tool_args / args / arguments; the MiniMax model
            # often emits "parameters" (OpenAI / Anthropic convention),
            # which would otherwise silently get dropped here and arrive at
            # the Tool.execute() as {} — making every parameterised tool
            # call fail server-side with 422.
            tool_args = None
            for key in ("tool_args", "args", "arguments", "parameters", "input"):
                candidate = tool_request.get(key)
                if isinstance(candidate, dict) and candidate:
                    tool_args = candidate
                    break
            if tool_args is None:
                tool_request["tool_args"] = {}
            elif not isinstance(tool_args, dict):
                raise ValueError("Tool request tool_args must be a dictionary")
            else:
                tool_request["tool_args"] = tool_args

        A0Agent.validate_tool_request = _lenient_validate  # type: ignore[method-assign]
        # ------------------------------------------------------------

        config = initialize_agent()
        context = AgentContext(config=config)

        prompt = _build_cli_prompt(self.request)

        # Run the agent conversation
        task = context.communicate(UserMessage(message=prompt))

        timeout = max(30, self.request.timeout_seconds)
        # DeferredTask runs on its own EventLoopThread; use the
        # synchronous blocking helper to wait for the result.
        result = task.result_sync(timeout=timeout)

        final_text = str(result) if result else "(no response)"

        assistant = Message(
            role="assistant", content=[TextBlock(text=final_text)]
        )
        return StepResponse(
            status="finished",
            assistant_message=assistant,
            usage=TokenUsage(),
            final_output=final_text,
        )

    def close(self, reason: str = "") -> None:
        # Remove the per-task tool files written into usr/tools/ so
        # they don't leak across sessions or pollute the agent-zero
        # workspace between runs.
        for path in self._written_tool_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
        self._written_tool_paths = []
