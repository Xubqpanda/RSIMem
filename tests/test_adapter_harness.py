from __future__ import annotations

from dataclasses import replace

import pytest

from rsimem.adapter_contracts import (
    AdapterResult,
    AdapterStatus,
    BenchmarkPublicEvent,
    BenchmarkSplit,
    BenchmarkTaskRequest,
    DeterministicHostAdapter,
    DeterministicMemoryMethodAdapter,
    HostCapabilities,
    MemoryMethodAdapter,
    MethodCapabilities,
    MethodRunIdentity,
)
from rsimem.adapter_harness import AdapterHarness
from rsimem.memory import MemoryKind
from rsimem.memory.lifecycle_surfaces import MemoryLifecycleSurface


class _Benchmark:
    def reset(self, request):
        return AdapterResult(AdapterStatus.SUPPORTED, "operation.benchmark.reset")

    def step(self, request):
        digest = "a" * 64
        return (
            AdapterResult(AdapterStatus.SUPPORTED, "operation.benchmark.step", output_digest=digest),
            BenchmarkPublicEvent(
                event_id="benchmark-event.fixture.v1",
                case_id=request.case_id,
                stage="task",
                event_type="task.ready",
                public_state_digest=digest,
            ),
        )


class _FailedBenchmark(_Benchmark):
    def reset(self, request):
        return AdapterResult(AdapterStatus.UNSUPPORTED, "operation.benchmark.reset", "runner_unavailable")


def _host() -> DeterministicHostAdapter:
    return DeterministicHostAdapter(HostCapabilities(
        memory_kinds=tuple(MemoryKind),
        tool_call_result_closure=True,
        usage_accounting=True,
        restart=True,
        context_snapshot=True,
        native_bypass=True,
    ))


def _method() -> DeterministicMemoryMethodAdapter:
    return DeterministicMemoryMethodAdapter(MethodCapabilities(
        method_id="method.semantic.harness.v1",
        primary_kind=MemoryKind.SEMANTIC,
        secondary_kind=None,
        transform=None,
        owned_surfaces=(MemoryLifecycleSurface.CONSTRUCTION,),
        required_feedback=(),
        required_host_capabilities=(),
        state_schema="method.state.harness.v1",
        lineage_schema="method.lineage.harness.v1",
        online_update=False,
        validation=False,
        rollback=False,
    ))


def _request() -> BenchmarkTaskRequest:
    return BenchmarkTaskRequest(
        case_id="case.harness.v1",
        split=BenchmarkSplit.TRAIN,
        task_template_id="template.harness.v1",
        seed="seed.harness.v1",
        tool_budget=2,
        max_turns=2,
    )


def test_harness_runs_benchmark_host_method_without_final_score() -> None:
    request = _request()
    run = MethodRunIdentity("run.harness.v1", "session.harness.v1", request.case_id, "revision.initial")
    result = AdapterHarness(_Benchmark(), _host(), _method()).run_case(request, run)
    assert result.status is AdapterStatus.ACCEPTED
    assert result.benchmark_result.status is AdapterStatus.SUPPORTED
    assert result.host_result.status is AdapterStatus.SUPPORTED
    assert result.method_result.status is AdapterStatus.SUPPORTED
    assert result.event_id.startswith("host.benchmark-event")
    assert result.host_state_digest != "0" * 64
    assert result.method_state_digest != "0" * 64
    assert "score" not in str(result.payload())


def test_harness_stops_before_host_or_method_when_benchmark_reset_fails() -> None:
    request = _request()
    run = MethodRunIdentity("run.harness.failed.v1", "session.harness.failed.v1", request.case_id, "revision.initial")
    result = AdapterHarness(_FailedBenchmark(), _host(), _method()).run_case(request, run)
    assert result.status is AdapterStatus.UNSUPPORTED
    assert result.benchmark_result.reason_code == "runner_unavailable"
    assert result.host_result.reason_code == "benchmark_reset_failed"
    assert result.method_result.reason_code == "benchmark_reset_failed"


def test_harness_rejects_case_task_identity_mismatch() -> None:
    request = _request()
    run = MethodRunIdentity("run.harness.mismatch.v1", "session.harness.v1", "task.other.v1", "revision.initial")
    with pytest.raises(ValueError, match="identity disagree"):
        AdapterHarness(_Benchmark(), _host(), _method()).run_case(request, run)
