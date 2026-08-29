from __future__ import annotations

from rsimem.memory.commit_scheduler import (
    CommitScheduleStatus,
    CommitScheduler,
    InMemoryCommitScheduleStore,
    JsonCommitScheduleStore,
)
from rsimem.memory.policy_contracts import CommitDecision


def _decision(*, revision: str = "backend.rev.1") -> CommitDecision:
    return CommitDecision.create(
        policy_version="fixed.commit.parent.v1",
        source_revision="snapshot.rev.1",
        input_payload={"admission": "decision.admission"},
        output_payload={"mutation_ids": ["mutation.1"]},
        action="RUN",
        execution_status="pending",
        reason_codes=("immediate_parent",),
        lineage_id="lineage.fixture",
        trigger_event_id="event.fixture",
        mutation_ids=("mutation.1",),
        expected_revision=revision,
        commit_mode="deferred",
        execution_boundary="session_end",
    )


def test_deferred_schedule_survives_restart_and_is_idempotent(tmp_path) -> None:
    path = tmp_path / "commit-schedules.json"
    first = CommitScheduler(JsonCommitScheduleStore(path)).schedule(_decision(), boundary="session_end")
    assert first is not None and first.status is CommitScheduleStatus.PENDING
    assert first.trigger_event_id == "event.fixture"
    restarted = CommitScheduler(JsonCommitScheduleStore(path))
    replay = restarted.schedule(_decision(), boundary="session_end")
    assert replay == first
    assert len(restarted.store.all()) == 1


def test_stale_revision_fails_without_apply() -> None:
    scheduler = CommitScheduler(InMemoryCommitScheduleStore())
    schedule = scheduler.schedule(_decision(), boundary="session_end")
    called: list[tuple[str, ...]] = []
    result = scheduler.execute(schedule.schedule_id, current_revision="backend.rev.2", apply=lambda ids: called.append(ids) or "receipt.1")
    assert result.status is CommitScheduleStatus.FAILED
    assert result.failure_reason == "stale_revision"
    assert called == []


def test_apply_failure_and_retry_terminal_state_are_recorded() -> None:
    scheduler = CommitScheduler(
        InMemoryCommitScheduleStore(),
        mutation_validator=lambda schedule: None,
    )
    schedule = scheduler.schedule(_decision(), boundary="session_end")
    failed = scheduler.execute(schedule.schedule_id, current_revision="backend.rev.1", apply=lambda ids: (_ for _ in ()).throw(RuntimeError("boom")))
    assert failed.status is CommitScheduleStatus.FAILED
    assert scheduler.execute(schedule.schedule_id, current_revision="backend.rev.1", apply=lambda ids: "receipt.ignored") == failed


def test_missing_or_failed_mutation_validator_cannot_call_apply() -> None:
    scheduler = CommitScheduler(InMemoryCommitScheduleStore())
    schedule = scheduler.schedule(_decision(), boundary="session_end")
    called: list[tuple[str, ...]] = []
    failed = scheduler.execute(
        schedule.schedule_id,
        current_revision="backend.rev.1",
        apply=lambda ids: called.append(ids) or "receipt.unsafe",
    )
    assert failed.status is CommitScheduleStatus.FAILED
    assert failed.failure_reason == "mutation_validator_missing"
    assert called == []

    scheduler = CommitScheduler(
        InMemoryCommitScheduleStore(),
        mutation_validator=lambda schedule: (_ for _ in ()).throw(
            ValueError("invalid mutation")
        ),
    )
    schedule = scheduler.schedule(_decision(), boundary="session_end")
    failed = scheduler.execute(
        schedule.schedule_id,
        current_revision="backend.rev.1",
        apply=lambda ids: called.append(ids) or "receipt.unsafe",
    )
    assert failed.status is CommitScheduleStatus.FAILED
    assert failed.failure_reason == "ValueError"
    assert called == []


def test_commit_can_be_cancelled_without_mutation() -> None:
    scheduler = CommitScheduler()
    schedule = scheduler.schedule(_decision(), boundary="session_end")
    cancelled = scheduler.cancel(schedule.schedule_id)
    assert cancelled.status is CommitScheduleStatus.CANCELLED
    assert scheduler.execute(schedule.schedule_id, current_revision="backend.rev.1", apply=lambda ids: "receipt.bad") == cancelled
