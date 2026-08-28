from __future__ import annotations

import pytest

from rsimem.memory.policy_contracts import DecisionAction, ExecutionStatus
from rsimem.memory.trigger_policy import (
    DeterministicTriggerPolicy,
    HermesTriggerEventAdapter,
    HostTriggerAdapter,
    TriggerPolicyConfig,
)


def _event(adapter: HostTriggerAdapter, kind: str, revision: str, turn: int, *, supported: bool | None = None):
    return adapter.event(
        kind,
        source_revision=revision,
        payload={"kind": kind, "revision": revision, "turn": turn},
        session_id="session.fixture",
        task_id="task.fixture",
        turn_id=f"turn.{turn}",
        turn_index=turn,
        supported=supported,
    )


def test_task_completion_is_run_and_shadow_events_are_recorded_without_run() -> None:
    adapter = HostTriggerAdapter()
    policy = DeterministicTriggerPolicy()
    completed = policy.decide(_event(adapter, "task_completed", "rev.1", 1))
    session_end = policy.decide(_event(adapter, "session_end", "rev.1", 2))
    assert completed.decision.action is DecisionAction.RUN
    assert completed.decision.execution_status is ExecutionStatus.PENDING
    assert not completed.shadow_only
    assert session_end.decision.action is DecisionAction.SKIP
    assert session_end.decision.reason_codes == ("shadow_only",)
    assert session_end.shadow_only
    assert len(policy.observations) == 2


def test_duplicate_event_and_revision_are_suppressed() -> None:
    adapter = HostTriggerAdapter()
    first_policy = DeterministicTriggerPolicy()
    event = _event(adapter, "task_completed", "rev.1", 1)
    first = first_policy.decide(event)
    duplicate_event = first_policy.decide(event)
    assert first.decision.action is DecisionAction.RUN
    assert duplicate_event.decision.action is DecisionAction.SKIP
    assert duplicate_event.decision.reason_codes == ("duplicate_event",)

    second = first_policy.decide(_event(adapter, "task_completed", "rev.1", 2))
    assert second.decision.reason_codes == ("duplicate_source_revision",)


def test_minimum_interval_defers_and_declares_next_boundary() -> None:
    adapter = HostTriggerAdapter()
    policy = DeterministicTriggerPolicy(TriggerPolicyConfig(min_turns_between_runs=3))
    assert policy.decide(_event(adapter, "task_completed", "rev.1", 1)).decision.action is DecisionAction.RUN
    deferred = policy.decide(_event(adapter, "task_completed", "rev.2", 2)).decision
    assert deferred.action is DecisionAction.DEFER
    assert deferred.execution_status is ExecutionStatus.DEFERRED
    assert deferred.next_eligible_boundary == "turn:4"


def test_unsupported_host_event_is_explicit_and_fail_closed() -> None:
    adapter = HostTriggerAdapter()
    event = adapter.event(
        "host_magic_boundary",
        source_revision="rev.1",
        payload={"raw": True},
    )
    assert event.supported is False
    assert event.metadata["unsupported_reason"] == "unknown_host_event"
    observation = DeterministicTriggerPolicy().decide(event)
    assert observation.decision.action is DecisionAction.SKIP
    assert observation.decision.reason_codes == ("unsupported_trigger",)


@pytest.mark.parametrize("kind", ["turn_interval", "tool_boundary", "context_pressure", "manual", "session_end"])
def test_all_non_parent_boundaries_are_shadow_only(kind: str) -> None:
    adapter = HostTriggerAdapter()
    policy = DeterministicTriggerPolicy()
    event = _event(adapter, kind, "rev.1", 2)
    if kind == "context_pressure":
        event = adapter.event(
            kind,
            source_revision="rev.1",
            payload={"tokens": 4096},
            turn_index=2,
            supported=True,
        )
    observation = policy.decide(event)
    assert observation.shadow_only is True
    assert observation.decision.action is DecisionAction.SKIP
    assert observation.decision.reason_codes == ("shadow_only",)


def test_adapter_rejects_empty_event_type() -> None:
    with pytest.raises(ValueError, match="event type"):
        HostTriggerAdapter().event("", source_revision="rev.1", payload={})


def test_hermes_snapshot_adapter_preserves_revision_and_marks_missing_pressure() -> None:
    class Snapshot:
        context_revision = "snapshot.rev.1"
        snapshot_id = "snapshot.fixture"
        session_id = "session.fixture"
        task_id = "task.fixture"
        task_state = "completed"
        current_turn_id = "turn.2"

    adapter = HermesTriggerEventAdapter()
    event = adapter.from_snapshot(Snapshot(), "task_completed", turn_index=2)
    assert event.source_revision == "snapshot.rev.1"
    assert event.metadata["snapshot_id"] == "snapshot.fixture"
    assert DeterministicTriggerPolicy().decide(event).decision.action is DecisionAction.RUN

    pressure = adapter.from_snapshot(Snapshot(), "context_pressure")
    assert pressure.supported is False
    assert pressure.metadata["unsupported_reason"] == "context_tokens_unobserved"
    assert DeterministicTriggerPolicy().decide(pressure).decision.reason_codes == ("unsupported_trigger",)
