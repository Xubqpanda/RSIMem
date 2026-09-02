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


__all__ = ["AdapterHarness", "AdapterHarnessResult"]
