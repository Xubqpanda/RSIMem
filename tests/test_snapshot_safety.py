from __future__ import annotations

import pytest

from rsimem.lifecycle import (
    ContextAction,
    ContextEvaluation,
    EvaluationSignal,
    EvaluationTrigger,
    HermesMessage,
    HermesSnapshotCollector,
    HermesStateSnapshotCollector,
    TaskLifecycleState,
    WritebackAction,
    WritebackCoordinator,
    snapshot_to_evaluation_request,
)


def test_current_turn_must_exist_or_be_none() -> None:
    collector = HermesSnapshotCollector()
    messages = (
        HermesMessage("message", "user", "Completed input.", "turn-1", 2, completed=True),
    )
    kwargs = {
        "run_id": "run",
        "episode_id": "episode",
        "session_id": "session",
        "task_id": "task",
        "task_state": TaskLifecycleState.COMPLETED,
        "lifecycle_state": "task_completed",
        "source_ref": "fixture:current-turn",
    }

    with pytest.raises(ValueError, match="identify a turn"):
        collector.collect(messages, current_turn_id="missing-turn", **kwargs)
    with pytest.raises(ValueError, match="None or a non-empty"):
        collector.collect(messages, current_turn_id="", **kwargs)

    snapshot = collector.collect(messages, current_turn_id=None, **kwargs)
    assert snapshot.current_turn_id is None


def test_old_unresolved_segment_is_protected_during_active_task() -> None:
    snapshot = HermesSnapshotCollector().collect(
        (
            HermesMessage(
                "old-unresolved",
                "assistant",
                "Waiting for an external tool result.",
                "turn-1",
                6,
                completed=False,
            ),
            HermesMessage(
                "current",
                "user",
                "Continue the active task.",
                "turn-2",
                5,
                completed=False,
            ),
        ),
        run_id="run",
        episode_id="episode",
        session_id="session",
        task_id="task",
        current_turn_id="turn-2",
        task_state=TaskLifecycleState.ACTIVE,
        lifecycle_state="task_active",
        source_ref="fixture:active-task",
        active_message_ids=("current",),
    )
    old_segment = snapshot.segments[0]
    assert old_segment.segment_id in snapshot.protected_segment_ids

    request = snapshot_to_evaluation_request(snapshot, evaluation_id="unsafe-unresolved")
    evaluation = ContextEvaluation(
        evaluation_id=request.evaluation_id,
        evaluator="unsafe",
        trigger=EvaluationTrigger.CONTEXT_PRESSURE,
        signals=tuple(
            EvaluationSignal(
                segment_id=segment.segment_id,
                context_action=(
                    ContextAction.EVICT
                    if segment.segment_id == old_segment.segment_id
                    else ContextAction.RETAIN
                ),
                writeback_action=(
                    WritebackAction.DISCARD
                    if segment.segment_id == old_segment.segment_id
                    else WritebackAction.DEFER
                ),
                utility_estimate=0.0,
                confidence=1.0,
            )
            for segment in request.segments
        ),
    )
    assert WritebackCoordinator().create_plans(snapshot, evaluation) == ()


def test_persisted_hermes_rows_have_stable_ids_and_closed_tools() -> None:
    rows = (
        {"id": 11, "role": "user", "content": "Use TSV output.", "token_count": None},
        {
            "id": 12,
            "role": "assistant",
            "content": "I will inspect the task.",
            "token_count": 6,
            "tool_calls": [{
                "id": "call-1",
                "function": {"name": "read_file", "arguments": '{"path":"task"}'},
            }],
        },
        {
            "id": 13,
            "role": "tool",
            "content": "task contents",
            "tool_call_id": "call-1",
            "token_count": None,
        },
        {"id": 14, "role": "assistant", "content": "Done.", "token_count": 2},
    )
    collector = HermesStateSnapshotCollector()
    kwargs = {
        "run_id": "run",
        "episode_id": "episode",
        "session_id": "session",
        "task_id": "task",
        "task_state": TaskLifecycleState.COMPLETED,
        "lifecycle_state": "task_completed",
        "source_ref": "hermes_state:session:test",
    }

    first = collector.collect(rows, **kwargs)
    replay = collector.collect(rows, **kwargs)
    extended = collector.collect(
        (*rows, {"id": 15, "role": "user", "content": "Thanks.", "token_count": 1}),
        **kwargs,
    )

    assert first.snapshot_id == replay.snapshot_id
    assert first.context_revision == replay.context_revision
    assert [item.segment_id for item in first.segments] == [
        item.segment_id for item in replay.segments
    ]
    assert [item.segment_id for item in first.segments] == [
        item.segment_id for item in extended.segments[: len(first.segments)]
    ]
    assert first.context_revision != extended.context_revision
    assert first.current_turn_id is None
    assert first.total_token_count == sum(item.token_count for item in first.segments)
    assert first.segments[0].metadata["token_count_source"] == "deterministic_estimate"
    assert len(first.tool_closures) == 1
    assert first.tool_closures[0].closed


@pytest.mark.parametrize(
    ("rows", "error"),
    [
        (
            ({"id": 1, "role": "tool", "content": "orphan", "tool_call_id": "x"},),
            "no matching call",
        ),
        (
            ({
                "id": 1,
                "role": "assistant",
                "content": "calling",
                "tool_calls": [{"id": "x", "function": {"name": "tool", "arguments": "{}"}}],
            },),
            "open tool call",
        ),
        (
            (
                {"id": 1, "role": "user", "content": "one"},
                {"id": 1, "role": "assistant", "content": "duplicate"},
            ),
            "row IDs must be unique",
        ),
    ],
)
def test_persisted_hermes_rows_fail_closed(rows, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        HermesStateSnapshotCollector().collect(
            rows,
            run_id="run",
            episode_id="episode",
            session_id="session",
            task_id="task",
            task_state=TaskLifecycleState.COMPLETED,
            lifecycle_state="task_completed",
            source_ref="hermes_state:session:test",
        )
