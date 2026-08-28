from __future__ import annotations

from rsimem.lifecycle.snapshot import (
    ContextSnapshot,
    ProvenanceRef,
    SnapshotSegment,
    TaskLifecycleState,
)
from rsimem.memory.contracts import MemoryAccessMode, MemoryBackendDescriptor, MemoryKind, MemoryKindCapability
from rsimem.memory.policy_replay import DeterministicPolicyReplay
from rsimem.memory.trigger_policy import HostTriggerAdapter


def _snapshot() -> ContextSnapshot:
    snapshot_id = "snapshot.replay"
    return ContextSnapshot(
        "run.replay",
        "episode.replay",
        "session.replay",
        "task.replay",
        snapshot_id,
        "revision.replay",
        (SnapshotSegment("segment.done", "message.done", "user", "durable preference", "turn.1", 2, completed=True),),
        (),
        None,
        TaskLifecycleState.COMPLETED,
        "task_completed",
        (),
        2,
        ProvenanceRef("run.replay", "episode.replay", "session.replay", "task.replay", snapshot_id, "fixture"),
    )


def _backend() -> MemoryBackendDescriptor:
    return MemoryBackendDescriptor(
        "backend.replay",
        (MemoryKindCapability(MemoryKind.SEMANTIC, MemoryAccessMode.EAGER),),
    )


def test_six_layer_replay_has_stable_lineage_and_receipt_join() -> None:
    snapshot = _snapshot()
    event = HostTriggerAdapter().event(
        "task_completed",
        source_revision=snapshot.context_revision,
        payload={"snapshot_id": snapshot.snapshot_id},
        session_id=snapshot.session_id,
        task_id=snapshot.task_id,
        turn_index=1,
    )
    kwargs = {
        "backend": _backend(),
        "candidate_fact_ids": ("fact.1",),
        "artifact_ids": ("artifact.1",),
        "mutation_ids": ("mutation.1",),
    }
    first = DeterministicPolicyReplay().run(snapshot, event, **kwargs)
    replay = DeterministicPolicyReplay().run(snapshot, event, **kwargs)
    assert first.decisions == replay.decisions
    assert first.lineage == replay.lineage
    assert first.audit.ok
    assert {decision.layer.value for decision in first.decisions} == {
        "trigger", "source_selection", "extraction", "admission", "commit", "exposure",
    }
    assert first.lineage.mutation_receipt_ids
    assert first.lineage.injection_receipt_ids


def test_replay_records_explicit_skip_without_downstream_mutation() -> None:
    snapshot = _snapshot()
    event = HostTriggerAdapter().event(
        "session_end",
        source_revision=snapshot.context_revision,
        payload={"snapshot_id": snapshot.snapshot_id},
        session_id=snapshot.session_id,
        task_id=snapshot.task_id,
    )
    result = DeterministicPolicyReplay().run(
        snapshot,
        event,
        backend=_backend(),
        candidate_fact_ids=("fact.1",),
    )
    assert result.decisions[0].action.value == "SKIP"
    assert len(result.decisions) == 2
    assert result.lineage.mutation_receipt_ids == ()
