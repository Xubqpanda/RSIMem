from __future__ import annotations

from types import SimpleNamespace

import pytest

from rsimem.adapter_contracts import (
    BenchmarkSplit,
    BenchmarkTaskRequest,
    CanonicalHostEvent,
    DeterministicMemoryMethodAdapter,
    FeedbackCondition,
    HostEventKind,
    MethodCapabilities,
)
from rsimem.memory import MemoryKind
from rsimem.memory.lifecycle_surfaces import MemoryLifecycleSurface
from rsimem.past_runtime_coordinator import PastRuntimeTerminalCoordinator


def _request() -> BenchmarkTaskRequest:
    return BenchmarkTaskRequest(
        case_id="case.runtime-coordinator.v1",
        split=BenchmarkSplit.TRAIN,
        task_template_id="template.runtime-coordinator.v1",
        seed="seed.runtime-coordinator.v1",
        tool_budget=2,
        max_turns=2,
    )


def _method() -> DeterministicMemoryMethodAdapter:
    return DeterministicMemoryMethodAdapter(MethodCapabilities(
        method_id="method.semantic.runtime-coordinator.v1",
        primary_kind=MemoryKind.SEMANTIC,
        secondary_kind=None,
        transform=None,
        owned_surfaces=(MemoryLifecycleSurface.RETRIEVAL_EXPOSURE,),
        required_feedback=(FeedbackCondition.F3,),
        required_host_capabilities=("context_snapshot",),
        state_schema="method.state.runtime-coordinator.v1",
        lineage_schema="method.lineage.runtime-coordinator.v1",
        online_update=False,
        validation=False,
        rollback=False,
    ))


def _response(event: CanonicalHostEvent) -> SimpleNamespace:
    return SimpleNamespace(
        status="finished",
        final_output="private response text",
        usage=SimpleNamespace(
            input_tokens=4,
            output_tokens=2,
            cache_read_tokens=0,
            cache_write_tokens=0,
            reasoning_tokens=1,
            request_count=1,
            retry_count=0,
            usage_complete=True,
        ),
        process_feedback_event_ids=("process-event.runtime-coordinator.v1",),
        process_feedback_digest="a" * 64,
        host_event_ids=(event.event_id,),
        host_events=(event.payload(),),
        host_state_digest="b" * 64,
        host_projection_digest="c" * 64,
    )


def test_coordinator_binds_content_free_past_terminal_response() -> None:
    request = _request()
    event = CanonicalHostEvent(
        event_id="host-event.runtime-coordinator.v1",
        session_id="session.runtime-coordinator.v1",
        task_id=request.case_id,
        kind=HostEventKind.TASK_COMPLETED,
        revision="revision.runtime-coordinator.v1",
        attributes={"completed": True},
    )
    binding = PastRuntimeTerminalCoordinator(object(), _method()).bind(
        request,
        _response(event),
        run_id="run.runtime-coordinator.v1",
    )
    assert binding.trace.case_id == request.case_id
    assert binding.event == event
    assert binding.result.status.value == "accepted"
    assert "private response text" not in str(binding.trace.identity_payload())


def test_coordinator_rejects_mismatched_runtime_host_payload() -> None:
    event = CanonicalHostEvent(
        event_id="host-event.runtime-coordinator.bad.v1",
        session_id="session.runtime-coordinator.bad.v1",
        task_id=_request().case_id,
        kind=HostEventKind.TASK_COMPLETED,
        revision="revision.runtime-coordinator.bad.v1",
        attributes={"completed": True},
    )
    response = _response(event)
    response.host_event_ids = ("host-event.other.v1",)
    with pytest.raises(ValueError, match="host event evidence"):
        PastRuntimeTerminalCoordinator(object(), _method()).bind(
            _request(),
            response,
            run_id="run.runtime-coordinator.bad.v1",
        )


def test_coordinator_rejects_nonopaque_runtime_task_id() -> None:
    event = CanonicalHostEvent(
        event_id="host-event.runtime-coordinator.leak.v1",
        session_id="session.runtime-coordinator.leak.v1",
        task_id="SM01_benchmark-task-leak",
        kind=HostEventKind.TASK_COMPLETED,
        revision="revision.runtime-coordinator.leak.v1",
        attributes={"completed": True},
    )
    with pytest.raises(ValueError, match="opaque case ID"):
        PastRuntimeTerminalCoordinator(object(), _method()).bind(
            _request(),
            _response(event),
            run_id="run.runtime-coordinator.leak.v1",
        )
