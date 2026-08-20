"""Runtime adapter that runs a Hermes agent in-process."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from ...models.content import TextBlock
from ...models.message import Message
from ...models.trace import TokenUsage
from ..protocol import StartSessionRequest, StepRequest, StepResponse
from ..registry import AgentSpec
from .base import RuntimeAdapter
from .cli_agent import _collect_text
from .sandbox_workspace import SandboxWorkspaceMirror

# Ensure the hermes-agent package is importable.
_HERMES_ROOT_DEFAULT = Path(__file__).resolve().parents[4] / "agents" / "hermes-agent"
_PAST_BENCH_TOOLSET = "past_bench_runtime"
_MISSING = object()


def _get_hermes_root(agent_name: str = "hermes") -> Path:
    """Return the Hermes agent root directory based on agent registry name.

    Supports both 'hermes' (default) and 'hermes-plus' forks so that
    benchmark runs can compare the original framework against an improved
    variant without modifying the original source tree.
    """
    if agent_name == "hermes-plus":
        return Path(__file__).resolve().parents[4] / "agents" / "hermes-plus"
    return _HERMES_ROOT_DEFAULT


def _build_hermes_prompt(request: StartSessionRequest) -> str:
    conversation = []
    for message in request.initial_messages:
        text = _collect_text(message)
        if text:
            conversation.append(f"[{message.role.upper()}]\n{text}")

    instructions = [
        "You are running inside the past-bench runtime container as Hermes Agent.",
        "Complete the benchmark task autonomously using your available tools.",
        "You MUST use tool calls to actually perform actions. Do NOT just describe what you would do — actually call the tools.",
        "Do not ask the user for clarification or for manual command execution.",
        "Your final answer must be plain text only. Never emit raw tool-call markup or XML-like invoke blocks in the final response.",
    ]
    return "\n\n".join(
        [
            "\n".join(instructions),
            f"Task: {request.task_id} - {request.task_name}",
            "Conversation:\n" + "\n\n".join(conversation),
        ]
    )


class HermesAdapter(RuntimeAdapter):
    """In-process adapter for the Hermes Agent framework.

    Runs the AIAgent from hermes-agent directly in-process, keeping
    evaluation of the agent framework separate from task definitions.
    """

    def __init__(self, spec: AgentSpec, request: StartSessionRequest) -> None:
        super().__init__(spec, request)
        self._completed_response: StepResponse | None = None
        self._session_db: Any | None = None
        self._past_bench_tool_backups: dict[str, Any] = {}
        self._sandbox_workspace = SandboxWorkspaceMirror(request)
        self._sandbox_workspace.prepare()

    def step(self, request: StepRequest) -> StepResponse:
        if self._completed_response is not None:
            return self._completed_response

        if request.step_id > 0 or request.tool_results:
            return StepResponse(
                status="error",
                error="Hermes adapter runs in one-shot mode; follow-up turns not supported.",
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
        # Deferred import so the module loads even when hermes is not installed.
        hermes_root = str(_get_hermes_root(self.spec.name))
        if hermes_root not in sys.path:
            sys.path.insert(0, hermes_root)

        extra = self.request.model.extra_body or {}
        hermes_cfg = extra.get("hermes", {}) if isinstance(extra, dict) else {}
        persistence_enabled = bool(hermes_cfg.get("persistence_enabled", False))
        hermes_home = None
        if hermes_cfg.get("home_dir"):
            hermes_home = Path(str(hermes_cfg["home_dir"])).expanduser().resolve()
            self._activate_hermes_home(hermes_home)
            hermes_home = self._prepare_hermes_home(hermes_cfg)
            self._activate_hermes_home(hermes_home)

        self._register_past_bench_tools()

        from run_agent import AIAgent
        from hermes_state import SessionDB

        # Auto-detect provider hint for anthropic-compatible endpoints so
        # Hermes uses the correct API mode (Messages API vs chat completions).
        # Only set provider="anthropic" for native Anthropic endpoints (anthropic.com).
        # Third-party /anthropic endpoints (MiniMax, DashScope, etc.) must NOT get
        # provider="anthropic" - run_agent.py already handles those via URL detection
        # and setting provider="anthropic" causes credential refresh to replace the
        # third-party key with ANTHROPIC_API_KEY on auth errors.
        provider = hermes_cfg.get("provider")
        base_url = self.request.model.base_url or ""
        if not provider and base_url:
            url_lower = base_url.lower()
            if "anthropic.com" in url_lower:
                provider = "anthropic"

        enabled_toolsets = hermes_cfg.get("enabled_toolsets")
        if self.request.tools:
            if enabled_toolsets is None:
                enabled_toolsets = [_PAST_BENCH_TOOLSET]
            else:
                enabled_toolsets = list(enabled_toolsets)
                if _PAST_BENCH_TOOLSET not in enabled_toolsets:
                    enabled_toolsets.append(_PAST_BENCH_TOOLSET)

        memory_cfg = hermes_cfg.get("config_overrides", {}).get("memory", {})
        memory_active = bool(memory_cfg.get("memory_enabled", True) or memory_cfg.get("user_profile_enabled", True))
        session_db = None
        if hermes_home is not None and hermes_cfg.get("session_search_enabled", False):
            session_db = SessionDB()
            self._session_db = session_db

        agent = AIAgent(
            model=self.request.model.model_id,
            api_key=self.request.model.api_key,
            base_url=self.request.model.base_url,
            provider=provider,
            temperature=self.request.runtime_config.temperature,
            max_iterations=int(hermes_cfg.get("max_iterations", 50)),
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=hermes_cfg.get("disabled_toolsets"),
            skip_context_files=bool(hermes_cfg.get("skip_context_files", True)),
            skip_memory=bool(hermes_cfg.get("skip_memory", (not persistence_enabled) or (not memory_active))),
            quiet_mode=True,
            session_db=session_db,
        )

        prompt = _build_hermes_prompt(self.request)

        result = agent.run_conversation(
            user_message=prompt,
            conversation_history=[],
            task_id=self.request.session_id,
        )

        self._set_session_title_if_missing(agent)

        review_wait_s = float(
            hermes_cfg.get("background_review_wait_s", 1.0 if persistence_enabled else 0.0)
        )
        if review_wait_s > 0:
            time.sleep(review_wait_s)

        self._capture_hermes_artifacts(
            hermes_cfg,
            hermes_home,
            session_log_file=getattr(agent, "session_log_file", None),
        )

        final_text = result.get("final_response") or ""
        if not isinstance(final_text, str):
            final_text = str(final_text)
        input_tokens = result.get("input_tokens")
        if input_tokens is None:
            input_tokens = result.get("prompt_tokens")
        output_tokens = result.get("output_tokens")
        if output_tokens is None:
            output_tokens = result.get("completion_tokens")
        if input_tokens is None or output_tokens is None:
            usage_data = result.get("usage") or {}
            if input_tokens is None:
                input_tokens = usage_data.get("input_tokens", usage_data.get("prompt_tokens", 0))
            if output_tokens is None:
                output_tokens = usage_data.get("output_tokens", usage_data.get("completion_tokens", 0))
        usage = TokenUsage(
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
        )

        assistant = Message(role="assistant", content=[TextBlock(text=final_text or "(no response)")])
        return StepResponse(
            status="finished",
            assistant_message=assistant,
            usage=usage,
            final_output=final_text,
        )

    def close(self, reason: str = "") -> None:
        self._restore_past_bench_tools()
        if self._session_db is not None:
            try:
                self._session_db.close()
            except Exception:
                pass
        self._session_db = None
        self._sandbox_workspace.close()

    @staticmethod
    def _activate_hermes_home(hermes_home: Path) -> None:
        os.environ["HERMES_HOME"] = str(hermes_home)
        HermesAdapter._reload_hermes_modules_if_needed(hermes_home)

    @staticmethod
    def _reload_hermes_modules_if_needed(hermes_home: Path) -> None:
        target_home = str(hermes_home.resolve())
        target_db_path = str((hermes_home / "state.db").resolve())

        run_agent_module = sys.modules.get("run_agent")
        hermes_state_module = sys.modules.get("hermes_state")
        current_home = getattr(run_agent_module, "_hermes_home", None)
        current_db_path = getattr(hermes_state_module, "DEFAULT_DB_PATH", None)

        loaded_modules = {
            module_name: sys.modules[module_name]
            for module_name in (
                "hermes_constants",
                "hermes_state",
                "hermes_cli.config",
                "tools.memory_tool",
                "tools.skill_manager_tool",
                "tools.skills_tool",
                "run_agent",
            )
            if module_name in sys.modules
        }
        if not loaded_modules:
            return

        if (
            current_home is not None
            and str(Path(current_home).resolve()) == target_home
            and current_db_path is not None
            and str(Path(current_db_path).resolve()) == target_db_path
        ):
            return

        # hermes-plus: also clear the 'agent' package and its submodules so that
        # switching between hermes-agent and hermes-plus in the same process
        # correctly loads the new agent/ package.
        modules_to_remove = [
            k for k in list(sys.modules.keys())
            if k == "agent" or k.startswith("agent.")
        ]
        for module_name in modules_to_remove:
            del sys.modules[module_name]

        for module_name in (
            "hermes_constants",
            "hermes_state",
            "hermes_cli.config",
            "tools.memory_tool",
            "tools.skill_manager_tool",
            "tools.skills_tool",
            "run_agent",
        ):
            module = loaded_modules.get(module_name)
            if module is not None:
                importlib.reload(module)

    @staticmethod
    def _prepare_hermes_home(hermes_cfg: dict[str, Any]) -> Path:
        home_dir = Path(str(hermes_cfg["home_dir"])).expanduser().resolve()
        home_dir.mkdir(parents=True, exist_ok=True)

        config_overrides = hermes_cfg.get("config_overrides") or {}
        config_path = home_dir / "config.yaml"
        current = {}
        if config_path.exists():
            with open(config_path, encoding="utf-8") as fh:
                current = yaml.safe_load(fh) or {}

        merged = (
            HermesAdapter._merge_dicts(current, config_overrides)
            if isinstance(config_overrides, dict)
            else current
        )
        with open(config_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(merged, fh, allow_unicode=True, sort_keys=True)

        fixture_dir = str(hermes_cfg.get("initial_home_fixture_dir") or "").strip()
        if fixture_dir:
            HermesAdapter._copy_initial_home_fixture(Path(fixture_dir).expanduser().resolve(), home_dir)

        preseed_dir = str(hermes_cfg.get("preseed_artifacts_dir") or "").strip()
        if preseed_dir:
            HermesAdapter._copy_preseed_artifacts(Path(preseed_dir).expanduser().resolve(), home_dir)
            HermesAdapter._import_session_seed_if_needed(home_dir)
        return home_dir

    @staticmethod
    def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            current = merged.get(key)
            if isinstance(current, dict) and isinstance(value, dict):
                merged[key] = HermesAdapter._merge_dicts(current, value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _capture_hermes_artifacts(
        hermes_cfg: dict[str, Any],
        hermes_home: Path | None,
        *,
        session_log_file: str | Path | None = None,
    ) -> None:
        capture_dir = hermes_cfg.get("capture_artifacts_dir")
        if not capture_dir or hermes_home is None:
            return

        target_dir = Path(str(capture_dir)).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        for name in ("config.yaml", "memories", "skills"):
            src = hermes_home / name
            dst = target_dir / name
            if not src.exists():
                continue
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        state_db = hermes_home / "state.db"
        if state_db.exists():
            shutil.copy2(state_db, target_dir / "state.db")

        sessions_dir = hermes_home / "sessions"
        if not sessions_dir.exists():
            return

        current_session = None
        if session_log_file:
            candidate = Path(str(session_log_file)).expanduser().resolve()
            if candidate.exists():
                current_session = candidate

        latest_session = None
        if current_session is None:
            for candidate in sorted(sessions_dir.glob("session_*.json")):
                latest_session = candidate
            current_session = latest_session

        if current_session is not None:
            shutil.copy2(current_session, target_dir / "session_current.json")
            shutil.copy2(current_session, target_dir / "session_latest.json")

    @staticmethod
    def _copy_preseed_artifacts(src_dir: Path, hermes_home: Path) -> None:
        if not src_dir.exists():
            return

        for name in ("memories", "skills", "sessions"):
            src = src_dir / name
            dst = hermes_home / name
            if not src.exists():
                continue
            HermesAdapter._overlay_path(src, dst)

    @staticmethod
    def _copy_initial_home_fixture(src_dir: Path, hermes_home: Path) -> None:
        if not src_dir.exists():
            return

        for name in ("memories", "skills", "sessions", "state.db", "state.db-wal", "state.db-shm"):
            src = src_dir / name
            dst = hermes_home / name
            if not src.exists() or dst.exists():
                continue
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    @staticmethod
    def _overlay_path(src: Path, dst: Path) -> None:
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            for child in src.iterdir():
                HermesAdapter._overlay_path(child, dst / child.name)
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    @staticmethod
    def _import_session_seed_if_needed(hermes_home: Path) -> None:
        session_seed_path = hermes_home / "session_seed.json"
        state_db_path = hermes_home / "state.db"
        if not session_seed_path.exists() or state_db_path.exists():
            return

        from hermes_state import SessionDB

        try:
            payload = json.loads(session_seed_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        sessions = payload.get("sessions", payload if isinstance(payload, list) else [])
        if not isinstance(sessions, list):
            return

        db = SessionDB(db_path=state_db_path)
        try:
            for session in sessions:
                if not isinstance(session, dict):
                    continue
                session_id = str(session.get("id") or "").strip()
                if not session_id:
                    continue
                db.create_session(
                    session_id=session_id,
                    source=str(session.get("source") or "cli"),
                    model=str(session.get("model") or "seed"),
                    model_config=session.get("model_config") if isinstance(session.get("model_config"), dict) else None,
                    system_prompt=str(session.get("system_prompt") or "") or None,
                    user_id=str(session.get("user_id") or "") or None,
                    parent_session_id=str(session.get("parent_session_id") or "") or None,
                )
                title = str(session.get("title") or "").strip()
                if title:
                    try:
                        db.set_session_title(session_id, title)
                    except Exception:
                        pass
                for message in session.get("messages", []) or []:
                    if not isinstance(message, dict):
                        continue
                    db.append_message(
                        session_id=session_id,
                        role=str(message.get("role") or "user"),
                        content=str(message.get("content") or "") or None,
                        tool_name=str(message.get("tool_name") or "") or None,
                        tool_calls=message.get("tool_calls"),
                        tool_call_id=str(message.get("tool_call_id") or "") or None,
                        token_count=message.get("token_count"),
                        finish_reason=str(message.get("finish_reason") or "") or None,
                        reasoning=str(message.get("reasoning") or "") or None,
                        reasoning_details=message.get("reasoning_details"),
                        codex_reasoning_items=message.get("codex_reasoning_items"),
                    )
        finally:
            db.close()

    def _register_past_bench_tools(self) -> None:
        if not self.request.tools:
            return

        hermes_root = str(_get_hermes_root(self.spec.name))
        if hermes_root not in sys.path:
            sys.path.insert(0, hermes_root)

        from tools.registry import registry

        tool_store = getattr(registry, "_tools", None)
        endpoint_map = {endpoint.tool_name: endpoint for endpoint in self.request.tool_endpoints}
        for tool in self.request.tools:
            endpoint = endpoint_map.get(tool.name)
            if endpoint is None:
                continue

            if tool.name not in self._past_bench_tool_backups:
                if tool_store is not None:
                    self._past_bench_tool_backups[tool.name] = tool_store.get(tool.name, _MISSING)
                else:
                    self._past_bench_tool_backups[tool.name] = _MISSING

            registry.register(
                name=tool.name,
                toolset=_PAST_BENCH_TOOLSET,
                schema={
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema or {"type": "object", "properties": {}},
                },
                handler=self._make_past_bench_tool_handler(endpoint),
                check_fn=lambda: True,
                emoji="🧰",
            )

    @staticmethod
    def _make_past_bench_tool_handler(endpoint):
        def _handler(args, **kwargs):
            import httpx

            method = endpoint.method.upper()
            request_kwargs: dict[str, Any] = {"timeout": 60.0}
            if method == "GET":
                request_kwargs["params"] = args
            else:
                request_kwargs["json"] = args

            try:
                response = httpx.request(method, endpoint.url, **request_kwargs)
                try:
                    payload = response.json()
                except ValueError:
                    payload = {"text": response.text}
                if response.status_code >= 400 and isinstance(payload, dict) and "error" not in payload:
                    payload["error"] = f"HTTP {response.status_code}"
                return json.dumps(payload, ensure_ascii=False)
            except Exception as exc:
                return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)

        return _handler

    def _restore_past_bench_tools(self) -> None:
        if not self._past_bench_tool_backups:
            return

        hermes_root = str(_get_hermes_root(self.spec.name))
        if hermes_root not in sys.path:
            sys.path.insert(0, hermes_root)

        from tools.registry import registry

        tool_store = getattr(registry, "_tools", None)
        if tool_store is None:
            self._past_bench_tool_backups.clear()
            return

        for tool_name, original in self._past_bench_tool_backups.items():
            if original is _MISSING:
                tool_store.pop(tool_name, None)
            else:
                tool_store[tool_name] = original
        self._past_bench_tool_backups.clear()

    def _set_session_title_if_missing(self, agent: Any) -> None:
        if self._session_db is None:
            return

        session_id = getattr(agent, "session_id", None)
        if not session_id:
            return

        try:
            existing = self._session_db.get_session_title(session_id)
        except Exception:
            return

        if existing:
            return

        title = f"{self.request.task_id} - {self.request.task_name}".strip()
        try:
            self._session_db.set_session_title(session_id, title)
        except Exception:
            pass
