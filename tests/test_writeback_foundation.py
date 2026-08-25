from __future__ import annotations

from dataclasses import replace

import pytest

from rsimem.lifecycle import (
    ContextAction,
    ContextEvaluation,
    DeterministicPreferenceEvaluator,
    EvaluationSignal,
    EvaluationTrigger,
    HermesMessage,
    HermesSnapshotCollector,
    PlanValidationStatus,
    SegmentKind,
    TaskLifecycleState,
    WritebackAction,
    WritebackCoordinator,
    run_sm01_preference_fixture,
    snapshot_to_evaluation_request,
)
from rsimem.memory import MemoryKind


def test_sm01_replay_has_stable_identity_and_content_free_events() -> None:
    first = run_sm01_preference_fixture()
    second = run_sm01_preference_fixture()

    assert [item.segment_id for item in first.snapshot.segments] == [
        item.segment_id for item in second.snapshot.segments
    ]
    assert first.snapshot.context_revision == second.snapshot.context_revision
    assert first.snapshot.provenance == second.snapshot.provenance
    assert first.plans[0].idempotency_key == second.plans[0].idempotency_key
    serialized = repr([event.to_dict() for event in first.events])
    assert "Use TSV with owner" not in serialized
    assert "current task is complete" not in serialized


def test_active_and_current_segments_cannot_be_evicted() -> None:
    result = run_sm01_preference_fixture()
    snapshot = result.snapshot
    request = snapshot_to_evaluation_request(snapshot, evaluation_id="unsafe")
    signals = tuple(
        replace(
            signal,
            context_action=ContextAction.EVICT,
            writeback_action=WritebackAction.ADD,
            memory_kind=MemoryKind.SEMANTIC,
        )
        for signal in result.evaluation.signals
    )
    evaluation = ContextEvaluation(
        evaluation_id=request.evaluation_id,
        evaluator="unsafe",
        trigger=EvaluationTrigger.TASK_COMPLETED,
        signals=signals,
    )
    coordinator = WritebackCoordinator()
    assert coordinator.create_plans(snapshot, evaluation) == ()


def test_tool_call_and_result_are_one_writeback_unit() -> None:
    collector = HermesSnapshotCollector()
    snapshot = collector.collect(
        (
            HermesMessage(
                "call", "assistant", "call search", "turn-1", 2,
                kind=SegmentKind.TOOL_CALL, tool_call_id="tool-1", completed=True,
            ),
            HermesMessage(
                "result", "tool", "result", "turn-1", 1,
                kind=SegmentKind.TOOL_RESULT, tool_call_id="tool-1", completed=True,
            ),
        ),
        run_id="run", episode_id="episode", session_id="session", task_id="task",
        current_turn_id="completion-turn", task_state=TaskLifecycleState.COMPLETED,
        lifecycle_state="task_completed", source_ref="fixture:tool",
    )
    request = snapshot_to_evaluation_request(snapshot, evaluation_id="evaluation")
    signals = tuple(
        EvaluationSignal(
            segment_id=segment.segment_id,
            context_action=ContextAction.EVICT,
            writeback_action=WritebackAction.ADD,
            memory_kind=MemoryKind.SEMANTIC,
            utility_estimate=1.0,
            confidence=1.0,
        )
        for segment in request.segments
    )
    evaluation = ContextEvaluation(
        "evaluation", "fixture", EvaluationTrigger.TASK_COMPLETED, signals,
    )
    plan = WritebackCoordinator().create_plans(snapshot, evaluation)
    assert len(plan) == 1
    assert set(plan[0].source_segment_ids) == {
        segment.segment_id for segment in snapshot.segments
    }


def test_changed_snapshot_marks_old_plan_stale() -> None:
    result = run_sm01_preference_fixture()
    plan = result.plans[0]
    changed = HermesSnapshotCollector().collect(
        (
            HermesMessage(
                "sm01-learn-preference", "user", "Use CSV instead.", "learn-turn", 4,
                completed=True,
            ),
            HermesMessage(
                "sm01-preference-ack", "assistant", "Acknowledged.", "learn-turn", 2,
                completed=True,
            ),
            HermesMessage(
                "sm01-task-complete", "assistant", "The current task is complete.",
                "completion-turn", 6, completed=True,
            ),
        ),
        run_id="run-sm01-fixture", episode_id="episode-sm01-learn",
        session_id="session-sm01-learn", task_id="SM01_preference_adoption",
        current_turn_id="completion-turn", task_state=TaskLifecycleState.COMPLETED,
        lifecycle_state="task_completed", source_ref="fixture:sm01_preference_adoption:v1",
        active_message_ids=("sm01-task-complete",),
    )
    receipt = WritebackCoordinator().dry_run(plan, changed)
    assert changed.snapshot_id == result.snapshot.snapshot_id
    assert changed.context_revision != result.snapshot.context_revision
    assert receipt.validation.status == PlanValidationStatus.STALE
    assert receipt.status.value == "stale"


def test_retrying_plan_is_idempotent() -> None:
    result = run_sm01_preference_fixture()
    coordinator = WritebackCoordinator()
    first = coordinator.dry_run(result.plans[0], result.snapshot)
    second = coordinator.dry_run(result.plans[0], result.snapshot)
    assert first.status.value == "accepted"
    assert second.status.value == "duplicate"
    assert first.mutation_id == second.mutation_id
    assert len(coordinator.dry_run_mutations) == 1


def test_incomplete_evaluation_creates_no_plan() -> None:
    result = run_sm01_preference_fixture()
    incomplete = replace(result.evaluation, signals=result.evaluation.signals[:1])
    coordinator = WritebackCoordinator()
    assert coordinator.create_plans(result.snapshot, incomplete) == ()
