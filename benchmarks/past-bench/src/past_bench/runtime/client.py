"""Bench-side clients for local and container runtime transports."""

from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

from ..config import RuntimeConfig
from .container import RuntimeContainerHandle, RuntimeContainerManager
from .manager import RuntimeSessionManager
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
from .registry import AgentSpec


class BaseRuntimeClient(ABC):
    """Abstract runtime transport."""

    @abstractmethod
    def start(self, *, run_id: str, agent_spec: AgentSpec) -> None:
        """Prepare transport resources."""

    @abstractmethod
    def stop(self) -> None:
        """Release transport resources."""

    @abstractmethod
    def health(self) -> HealthResponse:
        """Probe runtime health."""

    @abstractmethod
    def bootstrap(self, request: BootstrapRequest) -> BootstrapResponse:
        """Ensure the target agent is installed and ready."""

    @abstractmethod
    def start_session(self, request: StartSessionRequest) -> StartSessionResponse:
        """Create a new runtime session."""

    @abstractmethod
    def step(self, request: StepRequest) -> StepResponse:
        """Advance the runtime by one step."""

    @abstractmethod
    def interrupt(self, request: InterruptRequest) -> None:
        """Interrupt a session."""

    @abstractmethod
    def close_session(self, request: CloseSessionRequest) -> None:
        """Terminate a session."""


class LocalRuntimeClient(BaseRuntimeClient):
    """In-process runtime transport used for local execution and tests."""

    def __init__(self, *, registry_path: str | None = None, cache_dir: str | None = None) -> None:
        self._manager = RuntimeSessionManager(registry_path=registry_path, cache_dir=cache_dir)

    def start(self, *, run_id: str, agent_spec: AgentSpec) -> None:
        return None

    def stop(self) -> None:
        return None

    def health(self) -> HealthResponse:
        return self._manager.health()

    def bootstrap(self, request: BootstrapRequest) -> BootstrapResponse:
        return self._manager.bootstrap(request)

    def start_session(self, request: StartSessionRequest) -> StartSessionResponse:
        return self._manager.start_session(request)

    def step(self, request: StepRequest) -> StepResponse:
        return self._manager.step(request)

    def interrupt(self, request: InterruptRequest) -> None:
        self._manager.interrupt(request)

    def close_session(self, request: CloseSessionRequest) -> None:
        self._manager.close_session(request)


class ContainerRuntimeClient(BaseRuntimeClient):
    """HTTP client for the generic runtime container."""

    def __init__(
        self,
        runtime_config: RuntimeConfig,
        *,
        registry_path: str | None = None,
        image: str | None = None,
    ) -> None:
        self._runtime_config = runtime_config
        self._registry_path = registry_path
        self._manager = RuntimeContainerManager(runtime_config, image=image)
        self._handle: RuntimeContainerHandle | None = None
        self._client: httpx.Client | None = None
        self._step_timeout_s = 120.0

    def start(self, *, run_id: str, agent_spec: AgentSpec) -> None:
        self._handle = self._manager.start_container(
            run_id=run_id,
            agent_spec=agent_spec,
            registry_path=self._registry_path,
        )
        self._client = httpx.Client(base_url=self._handle.base_url, timeout=120.0)
        self._step_timeout_s = 120.0

    def stop(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._handle is not None:
            self._manager.stop_container(self._handle)
            self._handle = None

    def _http(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        *,
        timeout: float | None = None,
    ) -> dict:
        if self._client is None:
            raise RuntimeError("Runtime client is not started")
        resp = self._client.request(method, path, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> HealthResponse:
        return HealthResponse.model_validate(self._http("GET", "/health"))

    def bootstrap(self, request: BootstrapRequest) -> BootstrapResponse:
        return BootstrapResponse.model_validate(
            self._http("POST", "/bootstrap", request.model_dump())
        )

    def start_session(self, request: StartSessionRequest) -> StartSessionResponse:
        self._step_timeout_s = max(120.0, float(request.timeout_seconds) + 30.0)
        return StartSessionResponse.model_validate(
            self._http("POST", "/start_session", request.model_dump(mode="json"))
        )

    def step(self, request: StepRequest) -> StepResponse:
        return StepResponse.model_validate(
            self._http(
                "POST",
                "/step",
                request.model_dump(mode="json"),
                timeout=self._step_timeout_s,
            )
        )

    def interrupt(self, request: InterruptRequest) -> None:
        self._http("POST", "/interrupt", request.model_dump())

    def close_session(self, request: CloseSessionRequest) -> None:
        self._http("POST", "/close_session", request.model_dump())


def create_runtime_client(
    runtime_config: RuntimeConfig,
    *,
    mode: str,
    registry_path: str | None = None,
    image: str | None = None,
) -> BaseRuntimeClient:
    """Factory for runtime transports."""

    if mode == "container":
        return ContainerRuntimeClient(runtime_config, registry_path=registry_path, image=image)
    if mode == "local":
        return LocalRuntimeClient(registry_path=registry_path, cache_dir=runtime_config.cache_dir)
    raise ValueError(f"Unsupported runtime mode: {mode}")
