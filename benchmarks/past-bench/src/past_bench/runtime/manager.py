"""In-process implementation of the runtime session manager."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .adapters import (
    AgentZeroAdapter,
    ClaudeCodeCliAdapter,
    CodexCliAdapter,
    NanobotAdapter,
    OpenAICompatChatAdapter,
    OpenClawResponsesAdapter,
    RuntimeAdapter,
    ZeroClawAdapter,
    HermesAdapter,
)
from .protocol import (
    BootstrapRequest,
    BootstrapResponse,
    CloseSessionRequest,
    HealthResponse,
    InterruptRequest,
    StartSessionRequest,
    StartSessionResponse,
    StepRequest,
    StepResponse,
)
from .registry import AgentSpec, get_agent_spec, load_agent_registry


_ADAPTERS: dict[str, type[RuntimeAdapter]] = {
    "openai_compat_chat": OpenAICompatChatAdapter,
    "openclaw_responses": OpenClawResponsesAdapter,
    "codex_cli": CodexCliAdapter,
    "claude_code_cli": ClaudeCodeCliAdapter,
    "zeroclaw": ZeroClawAdapter,
    "hermes": HermesAdapter,
    "agent_zero": AgentZeroAdapter,
    "nanobot": NanobotAdapter,
}


def _normalize_bootstrap_command(command: str) -> str:
    stripped = command.lstrip()
    indent = command[: len(command) - len(stripped)]
    if stripped == "pip" or stripped.startswith("pip "):
        suffix = stripped[3:]
        return f'{indent}"{sys.executable}" -m pip{suffix}'
    return command


def _bootstrap_marker_payload(spec: AgentSpec) -> dict[str, str]:
    return {
        "install_policy": spec.install_policy,
        "python_executable": sys.executable,
    }


def _bootstrap_marker_is_current(marker: Path, spec: AgentSpec) -> bool:
    if not marker.exists():
        return False
    raw = marker.read_text(encoding="utf-8").strip()
    if not raw:
        return False
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Legacy marker contents were a plain install_policy string.
        return payload_matches_legacy(raw, spec)
    if not isinstance(payload, dict):
        return False
    if payload.get("install_policy") != spec.install_policy:
        return False
    if spec.install_policy == "pip" and payload.get("python_executable") != sys.executable:
        return False
    return True


def payload_matches_legacy(raw: str, spec: AgentSpec) -> bool:
    if raw != spec.install_policy:
        return False
    # Force a one-time re-bootstrap for legacy pip markers so installs land in
    # the current interpreter environment.
    return spec.install_policy != "pip"


class RuntimeSessionManager:
    """Owns agent registry resolution, bootstrapping, and session state."""

    def __init__(
        self,
        *,
        registry_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self._registry_path = Path(registry_path).resolve() if registry_path else None
        self._cache_dir = Path(cache_dir or ".runtime_cache").resolve()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._registry = load_agent_registry(self._registry_path)
        self._sessions: dict[str, RuntimeAdapter] = {}

    def health(self) -> HealthResponse:
        return HealthResponse(agents=sorted(self._registry))

    def bootstrap(self, request: BootstrapRequest) -> BootstrapResponse:
        spec = get_agent_spec(request.agent_name, self._registry)
        marker = self._cache_dir / f"{request.agent_name}.ready"
        if _bootstrap_marker_is_current(marker, spec) and not request.force:
            return BootstrapResponse(
                agent_name=request.agent_name,
                already_present=True,
            )
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        commands_run: list[str] = []
        for command in spec.bootstrap_commands:
            normalized_command = _normalize_bootstrap_command(command)
            proc = subprocess.run(
                normalized_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=900,
            )
            commands_run.append(normalized_command)
            stdout_parts.append(proc.stdout)
            stderr_parts.append(proc.stderr)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Bootstrap failed for {request.agent_name}: {normalized_command}\n{proc.stderr}"
                )
        marker.write_text(
            json.dumps(_bootstrap_marker_payload(spec), ensure_ascii=True),
            encoding="utf-8",
        )
        return BootstrapResponse(
            agent_name=request.agent_name,
            commands_run=commands_run,
            stdout="".join(stdout_parts),
            stderr="".join(stderr_parts),
        )

    def start_session(self, request: StartSessionRequest) -> StartSessionResponse:
        spec = get_agent_spec(request.agent_name, self._registry)
        adapter_cls = _ADAPTERS.get(spec.adapter)
        if adapter_cls is None:
            raise KeyError(f"Unsupported adapter type: {spec.adapter}")
        self._sessions[request.session_id] = adapter_cls(spec, request)
        return StartSessionResponse(
            session_id=request.session_id,
            agent_name=request.agent_name,
            adapter=spec.adapter,
        )

    def step(self, request: StepRequest) -> StepResponse:
        adapter = self._sessions.get(request.session_id)
        if adapter is None:
            raise KeyError(f"Unknown session_id: {request.session_id}")
        return adapter.step(request)

    def interrupt(self, request: InterruptRequest) -> None:
        adapter = self._sessions.get(request.session_id)
        if adapter is not None:
            adapter.interrupt(request.reason)

    def close_session(self, request: CloseSessionRequest) -> None:
        adapter = self._sessions.pop(request.session_id, None)
        if adapter is not None:
            adapter.close(request.reason)

    @property
    def registry(self) -> dict[str, AgentSpec]:
        return self._registry
