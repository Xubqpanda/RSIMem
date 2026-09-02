"""Launcher-facing binding from a PAST runtime response to a method adapter.

The coordinator consumes only the content-free terminal fields exported by the
vendored runtime protocol.  It deliberately does not import PAST runtime
classes so an external/container runner can use the same boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

from .adapter_contracts import (
    AdapterResult,
    AdapterStatus,
    BenchmarkAdapter,
    BenchmarkTaskRequest,
    CanonicalHostEvent,
    HostAdapter,
    HostCapabilities,
    MemoryMethodAdapter,
    MethodRunIdentity,
    MethodStateSnapshot,
)
from .adapter_harness import AdapterHarness, AdapterRuntimeBindingResult
from .memory.contracts import MemoryKind
from .past_bench_adapter import PastExecutionTrace


class ObservedTerminalHostAdapter:
    """Replay one runtime-observed terminal event as an immutable host state."""

    def __init__(
        self,
        event: CanonicalHostEvent,
        trace: PastExecutionTrace,
    ) -> None:
        if trace.host_event_count != 1 or trace.host_state_digest is None:
            raise ValueError("observed terminal host requires one host event and state")
        self._event = event
        self._trace = trace
        self._run: MethodRunIdentity | None = None
        self._observed = False

    @property
    def capabilities(self) -> HostCapabilities:
        return HostCapabilities(
            memory_kinds=tuple(MemoryKind),
            tool_call_result_closure=True,
            usage_accounting=True,
            restart=False,
            context_snapshot=True,
            native_bypass=False,
        )

    def prepare_session(self, run: MethodRunIdentity) -> AdapterResult:
        expected = (self._event.session_id, self._event.task_id, self._event.revision)
        if (run.session_id, run.task_id, run.state_revision) != expected:
            return AdapterResult(
                AdapterStatus.REJECTED,
                "operation.observed-host.prepare",
                "identity_mismatch",
            )
        if self._run is not None and self._run != run:
            return AdapterResult(
                AdapterStatus.REJECTED,
                "operation.observed-host.prepare",
                "active_run",
            )
        self._run = run
        return AdapterResult(AdapterStatus.SUPPORTED, "operation.observed-host.prepare")

    def observe_event(self, event: CanonicalHostEvent) -> AdapterResult:
        if self._run is None:
            return AdapterResult(
                AdapterStatus.REJECTED,
                "operation.observed-host.observe",
                "run_not_prepared",
            )
        if event != self._event:
            return AdapterResult(
                AdapterStatus.REJECTED,
                "operation.observed-host.observe",
                "event_mismatch",
            )
        if self._observed:
            return AdapterResult(
                AdapterStatus.REJECTED,
                "operation.observed-host.observe",
                "duplicate_event",
            )
        self._observed = True
        return AdapterResult(AdapterStatus.SUPPORTED, "operation.observed-host.observe")

    def snapshot_state(self) -> MethodStateSnapshot:
        return MethodStateSnapshot(
            state_id=f"state.observed-host.{self._event.event_id}",
            revision=self._event.revision,
            state_schema="past.runtime.observed-terminal.v1",
            state_digest=self._trace.host_state_digest or "0" * 64,
            active=self._run is not None,
        )

    def restart(self, run: MethodRunIdentity) -> AdapterResult:
        return AdapterResult(
            AdapterStatus.UNSUPPORTED,
            "operation.observed-host.restart",
            "runtime_trace_is_immutable",
        )


@dataclass(frozen=True, slots=True)
class PastRuntimeTerminalBinding:
    """Content-free result of one launcher-side terminal method binding."""

    trace: PastExecutionTrace
    event: CanonicalHostEvent
    result: AdapterRuntimeBindingResult


class PastRuntimeTerminalCoordinator:
    """Bind a finished PAST response to a fresh method adapter instance."""

    def __init__(
        self,
        benchmark: BenchmarkAdapter,
        method: MemoryMethodAdapter,
    ) -> None:
        self._benchmark = benchmark
        self._method = method

    def bind(
        self,
        request: BenchmarkTaskRequest,
        response: object,
        *,
        run_id: str,
    ) -> PastRuntimeTerminalBinding:
        trace = PastExecutionTrace.from_runtime_response(request, response)
        raw_events = getattr(response, "host_events", ())
        raw_ids = getattr(response, "host_event_ids", ())
        if not isinstance(raw_events, (list, tuple)) or not isinstance(raw_ids, (list, tuple)):
            raise TypeError("PAST runtime host evidence must be collections")
        events = tuple(CanonicalHostEvent.from_payload(value) for value in raw_events)
        event_ids = tuple(raw_ids)
        if (
            len(events) != trace.host_event_count
            or len(events) != 1
            or event_ids != tuple(event.event_id for event in events)
        ):
            raise ValueError("PAST runtime host event evidence is inconsistent")
        event = events[0]
        if event.task_id != request.case_id:
            raise ValueError("runtime terminal event is not bound to the opaque case ID")
        run = MethodRunIdentity(run_id, event.session_id, event.task_id, event.revision)
        host: HostAdapter = ObservedTerminalHostAdapter(event, trace)
        result = AdapterHarness(self._benchmark, host, self._method).bind_runtime_terminal(
            request,
            run,
            event,
            trace,
        )
        return PastRuntimeTerminalBinding(trace, event, result)


__all__ = [
    "ObservedTerminalHostAdapter",
    "PastRuntimeTerminalBinding",
    "PastRuntimeTerminalCoordinator",
]
