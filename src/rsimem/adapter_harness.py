"""Composable execution harness for the Stage 2 adapter boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .adapter_contracts import (
    AdapterResult,
    AdapterStatus,
    BenchmarkAdapter,
    BenchmarkTaskRequest,
    CanonicalHostEvent,
    HostAdapter,
    HostEventKind,
    MemoryMethodAdapter,
    MethodRunIdentity,
)
from .past_bench_adapter import PastExecutionTrace


@dataclass(frozen=True, slots=True)
class AdapterHarnessResult:
    case_id: str
    run_id: str
    status: AdapterStatus
    benchmark_result: AdapterResult
    host_result: AdapterResult
    method_result: AdapterResult
    event_id: str
    host_state_digest: str
    method_state_digest: str

    def __post_init__(self) -> None:
        for value, name in ((self.case_id, "case ID"), (self.run_id, "run ID"), (self.event_id, "event ID"), (self.host_state_digest, "host state digest"), (self.method_state_digest, "method state digest")):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"harness {name} must not be empty")
        object.__setattr__(self, "status", AdapterStatus(self.status))

    def payload(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "run_id": self.run_id,
            "status": self.status.value,
            "benchmark": self.benchmark_result.status.value,
            "host": self.host_result.status.value,
            "method": self.method_result.status.value,
            "event_id": self.event_id,
            "host_state_digest": self.host_state_digest,
            "method_state_digest": self.method_state_digest,
        }


@dataclass(frozen=True, slots=True)
class AdapterRuntimeBindingResult:
    """Result of binding one real terminal runtime trace to a method adapter."""

    case_id: str
    run_id: str
    status: AdapterStatus
    host_result: AdapterResult
    method_result: AdapterResult
    trace_id: str
    host_state_digest: str
    method_state_digest: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.case_id, "case ID"),
            (self.run_id, "run ID"),
            (self.trace_id, "runtime trace ID"),
            (self.host_state_digest, "host state digest"),
            (self.method_state_digest, "method state digest"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"runtime binding {name} must not be empty")
        object.__setattr__(self, "status", AdapterStatus(self.status))


class AdapterHarness:
    """Run one benchmark case through independent host and method adapters."""

    def __init__(
        self,
        benchmark: BenchmarkAdapter,
        host: HostAdapter,
        method: MemoryMethodAdapter,
    ) -> None:
        self.benchmark = benchmark
        self.host = host
        self.method = method

    @staticmethod
    def _first_failure(*results: AdapterResult) -> AdapterResult | None:
        for result in results:
            if result.status not in {AdapterStatus.SUPPORTED, AdapterStatus.ACCEPTED}:
                return result
        return None

    def run_case(
        self,
        request: BenchmarkTaskRequest,
        run: MethodRunIdentity,
    ) -> AdapterHarnessResult:
        if request.case_id != run.task_id:
            raise ValueError("benchmark request and method run task identity disagree")
        benchmark_reset = self.benchmark.reset(request)
        if benchmark_reset.status is not AdapterStatus.SUPPORTED:
            return AdapterHarnessResult(
                request.case_id, run.run_id, benchmark_reset.status,
                benchmark_reset, AdapterResult(AdapterStatus.UNSUPPORTED, "operation.host.not_run", "benchmark_reset_failed"),
                AdapterResult(AdapterStatus.UNSUPPORTED, "operation.method.not_run", "benchmark_reset_failed"),
                f"event.not-run.{request.case_id}", "0" * 64, "0" * 64,
            )
        host_prepare = self.host.prepare_session(run)
        method_prepare = self.method.prepare_run(run)
        failure = self._first_failure(host_prepare, method_prepare)
        if failure is not None:
            return AdapterHarnessResult(
                request.case_id, run.run_id, failure.status,
                benchmark_reset, host_prepare, method_prepare,
                f"event.not-started.{request.case_id}", "0" * 64, "0" * 64,
            )
        method_start = self.method.start_episode(run)
        if method_start.status not in {AdapterStatus.SUPPORTED, AdapterStatus.ACCEPTED}:
            return AdapterHarnessResult(
                request.case_id, run.run_id, method_start.status,
                benchmark_reset, host_prepare, method_start,
                f"event.not-started.{request.case_id}", "0" * 64, "0" * 64,
            )
        benchmark_step, public_event = self.benchmark.step(request)
        if benchmark_step.status not in {AdapterStatus.SUPPORTED, AdapterStatus.ACCEPTED}:
            return AdapterHarnessResult(
                request.case_id, run.run_id, benchmark_step.status,
                benchmark_step, host_prepare, method_start,
                f"event.not-stepped.{request.case_id}", "0" * 64, "0" * 64,
            )
        if public_event.case_id != request.case_id:
            raise ValueError("benchmark event case identity mismatch")
        host_event = CanonicalHostEvent(
            event_id=f"host.{public_event.event_id}",
            session_id=run.session_id,
            task_id=run.task_id,
            kind=HostEventKind.TURN_COMPLETED,
            revision=run.state_revision,
            attributes=dict(public_event.attributes),
        )
        host_result = self.host.observe_event(host_event)
        method_result = self.method.observe_event(host_event)
        finalize = self.method.finalize_episode(run)
        host_state = self.host.snapshot_state()
        method_state = self.method.snapshot_state()
        final_status = self._first_failure(benchmark_step, host_result, method_result, finalize)
        status = final_status.status if final_status is not None else AdapterStatus.ACCEPTED
        return AdapterHarnessResult(
            request.case_id,
            run.run_id,
            status,
            benchmark_step,
            host_result,
            method_result,
            host_event.event_id,
            host_state.state_digest,
            method_state.state_digest,
        )

    def bind_runtime_terminal(
        self,
        request: BenchmarkTaskRequest,
        run: MethodRunIdentity,
        event: CanonicalHostEvent,
        trace: PastExecutionTrace,
    ) -> AdapterRuntimeBindingResult:
        """Feed an observed PAST terminal event into the configured method.

        The real Hermes bridge has already observed the host event by the time
        the runner returns.  A duplicate host observation is therefore normal
        and is treated as an accepted idempotent join; all other host rejects,
        trace drift, and method failures remain explicit.
        """

        if trace.case_id != request.case_id:
            raise ValueError("runtime trace and benchmark request disagree")
        if event.session_id != run.session_id or event.task_id != run.task_id:
            raise ValueError("runtime terminal event identity disagrees with method run")
        if event.revision != run.state_revision:
            raise ValueError("runtime terminal event revision disagrees with method run")
        expected_kind = (
            HostEventKind.TASK_COMPLETED
            if trace.terminal_status == "finished"
            else HostEventKind.TASK_FAILED
            if trace.terminal_status == "error"
            else None
        )
        if expected_kind is None or event.kind is not expected_kind:
            raise ValueError("runtime terminal event status disagrees with execution trace")
        if trace.host_event_count < 1 or trace.host_state_digest is None:
            raise ValueError("runtime terminal trace has no host state evidence")

        host_prepare = self.host.prepare_session(run)
        method_prepare = self.method.prepare_run(run)
        failure = self._first_failure(host_prepare, method_prepare)
        if failure is not None:
            return AdapterRuntimeBindingResult(
                request.case_id,
                run.run_id,
                failure.status,
                host_prepare,
                method_prepare,
                trace.trace_id,
                "0" * 64,
                "0" * 64,
            )
        method_start = self.method.start_episode(run)
        if method_start.status not in {AdapterStatus.SUPPORTED, AdapterStatus.ACCEPTED}:
            return AdapterRuntimeBindingResult(
                request.case_id,
                run.run_id,
                method_start.status,
                host_prepare,
                method_start,
                trace.trace_id,
                "0" * 64,
                "0" * 64,
            )
        host_result = self.host.observe_event(event)
        if (
            host_result.status is AdapterStatus.REJECTED
            and host_result.reason_code == "duplicate_event"
        ):
            host_result = AdapterResult(
                AdapterStatus.ACCEPTED,
                "operation.host.observe_runtime",
                "duplicate_event",
            )
        host_state = self.host.snapshot_state()
        if host_state.state_digest != trace.host_state_digest:
            return AdapterRuntimeBindingResult(
                request.case_id,
                run.run_id,
                AdapterStatus.STALE,
                AdapterResult(
                    AdapterStatus.STALE,
                    "operation.host.runtime_trace",
                    "state_digest_mismatch",
                ),
                AdapterResult(
                    AdapterStatus.UNSUPPORTED,
                    "operation.method.not_run",
                    "host_state_mismatch",
                ),
                trace.trace_id,
                host_state.state_digest,
                "0" * 64,
            )
        method_result = self.method.observe_event(event)
        finalize = self.method.finalize_episode(run)
        method_state = self.method.snapshot_state()
        failure = self._first_failure(host_result, method_result, finalize)
        return AdapterRuntimeBindingResult(
            request.case_id,
            run.run_id,
            failure.status if failure is not None else AdapterStatus.ACCEPTED,
            host_result,
            method_result,
            trace.trace_id,
            host_state.state_digest,
            method_state.state_digest,
        )


__all__ = [
    "AdapterHarness",
    "AdapterHarnessResult",
    "AdapterRuntimeBindingResult",
]
