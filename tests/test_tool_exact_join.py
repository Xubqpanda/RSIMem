from __future__ import annotations

import json
import hashlib

import pytest

from rsimem.memory.tool_exact_join import (
    ToolCallResultJoin,
    ToolJoinResolutionStatus,
    resolve_tool_call_result,
)
from rsimem.memory.pure_process import PureProcessCorpus
from rsimem.memory.process_feedback import (
    ProcessEvent,
    ProcessEventKind,
    ProcessEventStatus,
    audit_process_events,
)


def _join(**overrides: object) -> ToolCallResultJoin:
    values: dict[str, object] = {
        "call_id": "call.notes.v1",
        "result_id": "result.notes.v1",
        "tool_name_digest": "a" * 64,
        "success": True,
        "retry_identity": "retry.notes.0",
        "run_id": "run.fixture.v1",
        "variant": "native",
        "trace_id": "trace.fixture.v1",
        "episode_id": "episode.fixture.v1",
        "session_id": "session.fixture.v1",
        "task_id": "task.fixture.v1",
        "source_revision": "revision.fixture.v1",
        "host_event_id": "event.tool.v1",
        "policy_lineage_id": "lineage.exposure.v1",
        "memory_use_operation_id": "op.use.v1",
        "call_receipt_id": "receipt.call.v1",
        "result_receipt_id": "receipt.result.v1",
    }
    values.update(overrides)
    return ToolCallResultJoin.create(**values)


def test_exact_call_result_join_replays_and_projects_two_events() -> None:
    join = _join()
    assert resolve_tool_call_result(join).status == ToolJoinResolutionStatus.COMPLETE
    events = join.process_events()
    assert len(events) == 2
    assert events[0].kind.value == "tool_call"
    assert events[1].kind.value == "tool_result"
    assert events[0].tool_call_id == join.call_id
    assert events[0].tool_name_digest == join.tool_name_digest
    assert events[0].retry_identity == join.retry_identity
    assert events[1].tool_result_id == join.result_id
    assert events[1].tool_success is True
    pure = PureProcessCorpus.create(events)
    projected_call = next(item for item in pure.events if item.kind.value == "tool_call")
    projected_result = next(item for item in pure.events if item.kind.value == "tool_result")
    assert projected_call.tool_call_id == join.call_id
    assert projected_result.tool_result_id == join.result_id
    serialized = json.dumps(events[0].payload(), ensure_ascii=True)
    assert "arguments" not in serialized
    assert ToolCallResultJoin.from_payload(json.loads(json.dumps(join.payload()))) == join


def test_projection_scope_is_attached_to_process_events_only() -> None:
    join = _join()
    events = join.process_events(
        family_id="SM02_constraint_retention",
        stage="learn_a",
    )

    assert len(events) == 2
    assert {event.family_id for event in events} == {"SM02_constraint_retention"}
    assert {event.stage for event in events} == {"learn_a"}
    # The host-neutral join contract itself remains unchanged.
    assert "family_id" not in join.payload()
    assert "stage" not in join.payload()


def test_exact_join_is_not_tied_to_notes_protocol() -> None:
    tool_name = "calendar.create_event"
    join = _join(
        call_id="call.calendar.v1",
        result_id="result.calendar.v1",
        tool_name_digest=hashlib.sha256(tool_name.encode()).hexdigest(),
        success=False,
        retry_identity="retry.calendar.0",
        call_receipt_id="receipt.calendar.call.v1",
        result_receipt_id="receipt.calendar.result.v1",
    )
    events = join.process_events(
        family_id="application.calendar",
        stage="tool_boundary",
    )
    assert [event.kind.value for event in events] == ["tool_call", "tool_result"]
    assert events[1].status.value == "failed"
    assert events[1].reason_codes == ("tool_failure",)


def test_missing_result_projects_only_call_and_resolves_non_exact() -> None:
    join = _join(result_present=False, result_id=None)
    events = join.process_events()
    assert len(events) == 1
    assert events[0].kind.value == "tool_call"
    assert resolve_tool_call_result(join).status == ToolJoinResolutionStatus.MISSING


@pytest.mark.parametrize(
    ("field", "status"),
    (
        ("result_present", ToolJoinResolutionStatus.MISSING),
        ("duplicate_call", ToolJoinResolutionStatus.DUPLICATE),
        ("duplicate_result", ToolJoinResolutionStatus.DUPLICATE),
        ("orphan_result", ToolJoinResolutionStatus.ORPHANED),
        ("cross_task", ToolJoinResolutionStatus.CROSS_TASK),
        ("type_mismatch", ToolJoinResolutionStatus.TYPE_MISMATCH),
    ),
)
def test_non_closed_tool_join_fails_closed(field: str, status: ToolJoinResolutionStatus) -> None:
    values: dict[str, object] = {field: True}
    if field == "result_present":
        values["result_present"] = False
        values["result_id"] = None
    result = resolve_tool_call_result(_join(**values))
    assert result.status == status
    assert result.exact is False


def test_censored_observation_does_not_claim_closure() -> None:
    result = resolve_tool_call_result(_join(observation_complete=False))
    assert result.status == ToolJoinResolutionStatus.CENSORED
    assert result.exact is False


def test_missing_receipt_identity_is_not_exact() -> None:
    result = resolve_tool_call_result(_join(result_receipt_id=None))
    assert result.status == ToolJoinResolutionStatus.MISSING
    assert result.reason_code == "missing_receipt_identity"


def test_process_audit_rechecks_projected_tool_closure() -> None:
    complete = _join().process_events()
    assert audit_process_events(complete) == ()

    missing = _join(result_present=False, result_id=None).process_events()
    errors = audit_process_events(missing)
    assert any("lacks matching result" in error for error in errors)

    orphan = _join(call_present=False, call_id=None, orphan_result=True).process_events()
    errors = audit_process_events(orphan)
    assert any("orphan tool result" in error for error in errors)


def test_process_audit_rejects_tool_type_mismatch_and_cross_task_join() -> None:
    common = dict(
        run_id="run.audit-tool.v1",
        variant="native",
        trace_id="trace.audit-tool.v1",
        episode_id="episode.audit-tool.v1",
        session_id="session.audit-tool.v1",
        host_event_id="event.audit-tool.v1",
        source_revision="revision.audit-tool.v1",
        input_payload={},
        output_payload={},
        execution_receipt_ids=("receipt.audit-tool.v1",),
        tool_call_id="call.audit-tool.v1",
        retry_identity="retry.audit-tool.v1",
    )
    call = ProcessEvent.create(
        kind=ProcessEventKind.TOOL_CALL,
        status=ProcessEventStatus.EXECUTED,
        task_id="task.audit-tool.v1",
        tool_name_digest="a" * 64,
        **common,
    )
    mismatched_result = ProcessEvent.create(
        kind=ProcessEventKind.TOOL_RESULT,
        status=ProcessEventStatus.SUCCESS,
        task_id="task.audit-tool.v1",
        tool_result_id="result.audit-tool.v1",
        tool_name_digest="b" * 64,
        tool_success=True,
        **common,
    )
    errors = audit_process_events((call, mismatched_result))
    assert any("type mismatch" in error for error in errors)

    cross_task_result = ProcessEvent.create(
        kind=ProcessEventKind.TOOL_RESULT,
        status=ProcessEventStatus.SUCCESS,
        task_id="task.audit-tool.v2",
        tool_result_id="result.audit-tool.v2",
        tool_name_digest="a" * 64,
        tool_success=True,
        **common,
    )
    errors = audit_process_events((call, cross_task_result))
    assert any("crosses task boundary" in error for error in errors)


def test_process_audit_rejects_duplicate_call_id_across_retries() -> None:
    events = tuple(
        ProcessEvent.create(
            kind=ProcessEventKind.TOOL_CALL,
            status=ProcessEventStatus.EXECUTED,
            run_id="run.audit-duplicate.v1",
            variant="native",
            trace_id="trace.audit-duplicate.v1",
            episode_id="episode.audit-duplicate.v1",
            session_id="session.audit-duplicate.v1",
            task_id="task.audit-duplicate.v1",
            host_event_id=f"event.audit-duplicate.{retry}",
            source_revision="revision.audit-duplicate.v1",
            input_payload={},
            output_payload={},
            execution_receipt_ids=(f"receipt.audit-duplicate.{retry}",),
            tool_call_id="call.audit-duplicate.v1",
            tool_name_digest="a" * 64,
            retry_identity=retry,
        )
        for retry in ("retry.audit-duplicate.0", "retry.audit-duplicate.1")
    )
    errors = audit_process_events(events)
    assert any("duplicate tool call ID" in error for error in errors)


def test_process_audit_rejects_duplicate_result_id_across_tasks() -> None:
    common = dict(
        run_id="run.audit-result-duplicate.v1",
        variant="native",
        trace_id="trace.audit-result-duplicate.v1",
        episode_id="episode.audit-result-duplicate.v1",
        session_id="session.audit-result-duplicate.v1",
        host_event_id="event.audit-result-duplicate.v1",
        source_revision="revision.audit-result-duplicate.v1",
        input_payload={},
        output_payload={},
        execution_receipt_ids=("receipt.audit-result-duplicate.v1",),
        tool_name_digest="a" * 64,
        tool_result_id="result.audit-result-duplicate.v1",
        tool_success=True,
        retry_identity="retry.audit-result-duplicate.v1",
    )
    first = ProcessEvent.create(
        kind=ProcessEventKind.TOOL_RESULT,
        status=ProcessEventStatus.SUCCESS,
        task_id="task.audit-result-duplicate.one",
        tool_call_id="call.audit-result-duplicate.one",
        **common,
    )
    second = ProcessEvent.create(
        kind=ProcessEventKind.TOOL_RESULT,
        status=ProcessEventStatus.SUCCESS,
        task_id="task.audit-result-duplicate.two",
        tool_call_id="call.audit-result-duplicate.two",
        **common,
    )
    errors = audit_process_events((first, second))
    assert any("duplicate tool result ID across task boundary" in error for error in errors)


@pytest.mark.parametrize(
    "overrides",
    (
        {"result_present": False, "result_id": "result.present-when-absent"},
        {"call_present": False, "call_id": "call.present-when-absent"},
    ),
)
def test_presence_flags_cannot_disagree_with_tool_ids(overrides) -> None:
    with pytest.raises(ValueError, match="cannot carry"):
        _join(**overrides)


def test_orphan_result_must_be_explicitly_marked() -> None:
    with pytest.raises(ValueError, match="marked orphan"):
        _join(call_present=False, call_id=None)


def test_unknown_result_success_is_not_reported_as_failure() -> None:
    events = _join(success=None).process_events()
    assert events[-1].status.value == "unknown"
    resolution = resolve_tool_call_result(_join(success=None))
    assert resolution.status == ToolJoinResolutionStatus.TYPE_MISMATCH
    assert resolution.reason_code == "tool_success_unknown"


def test_strict_success_flag() -> None:
    with pytest.raises(TypeError, match="success must be bool"):
        _join(success="true")
