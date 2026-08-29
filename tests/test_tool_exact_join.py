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


def test_unknown_result_success_is_not_reported_as_failure() -> None:
    events = _join(success=None).process_events()
    assert events[-1].status.value == "unknown"


def test_strict_success_flag() -> None:
    with pytest.raises(TypeError, match="success must be bool"):
        _join(success="true")
