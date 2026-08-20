"""Base classes for runtime adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..protocol import StartSessionRequest, StepRequest, StepResponse
from ..registry import AgentSpec


class RuntimeAdapter(ABC):
    """Stateful runtime adapter for one agent session."""

    def __init__(self, spec: AgentSpec, request: StartSessionRequest) -> None:
        self.spec = spec
        self.request = request

    @abstractmethod
    def step(self, request: StepRequest) -> StepResponse:
        """Advance the session by one assistant turn."""

    def interrupt(self, reason: str = "") -> None:
        """Interrupt the session if the adapter supports it."""

    def close(self, reason: str = "") -> None:
        """Release adapter resources."""

