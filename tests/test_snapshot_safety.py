from __future__ import annotations

import pytest

from rsimem.lifecycle import (
    ContextAction,
    ContextEvaluation,
    EvaluationSignal,
    EvaluationTrigger,
    HermesMessage,
    HermesSnapshotCollector,
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
