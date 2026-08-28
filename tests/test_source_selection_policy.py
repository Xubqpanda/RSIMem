from __future__ import annotations

import pytest

from rsimem.lifecycle.snapshot import (
    ContextSnapshot,
    ProvenanceRef,
    SegmentKind,
    SnapshotSegment,
    TaskLifecycleState,
    ToolClosure,
)
from rsimem.memory.policy_contracts import ProjectionMode
from rsimem.memory.source_selection_policy import (
    DeterministicSourceSelectionPolicy,
    SourceSelectionConfig,
)
from rsimem.memory.extraction_source import ExtractionSourceProjector
from rsimem.memory.trigger_policy import HostTriggerAdapter


def _snapshot(*, revision: str = "revision.fixture", current: str | None = None, state: TaskLifecycleState = TaskLifecycleState.COMPLETED) -> ContextSnapshot:
    segments = (
        SnapshotSegment("seg.user", "msg.user", "user", "remember TSV", "turn.1", 2, completed=True),
        SnapshotSegment("seg.call", "msg.call", "assistant", "notes_share", "turn.1", 2, SegmentKind.TOOL_CALL, True, "tool.1"),
        SnapshotSegment("seg.result", "msg.result", "tool", "ok", "turn.1", 1, SegmentKind.TOOL_RESULT, True, "tool.1"),
        SnapshotSegment("seg.open", "msg.open", "assistant", "pending", "turn.2", 1, SegmentKind.TOOL_CALL, False, "tool.2"),
        SnapshotSegment("seg.current", "msg.current", "user", "active question", "turn.3", 2, completed=True),
    )
    closures = (
        ToolClosure("closure.1", "tool.1", "seg.call", "seg.result"),
        ToolClosure("closure.2", "tool.2", "seg.open", None),
    )
    provenance = ProvenanceRef("run.fixture", "episode.fixture", "session.fixture", "task.fixture", "snapshot.fixture", "fixture")
    return ContextSnapshot(
        "run.fixture", "episode.fixture", "session.fixture", "task.fixture", "snapshot.fixture", revision,
        segments, ("seg.current",) if current else (), current, state, "task_completed", closures, 8, provenance,
    )


def test_whole_task_selection_excludes_unresolved_current_and_preserves_closure() -> None:
    snapshot = _snapshot(current="turn.3")
    event = HostTriggerAdapter().event("task_completed", source_revision=snapshot.context_revision, payload={"snapshot": snapshot.snapshot_id}, turn_index=3)
    decision = DeterministicSourceSelectionPolicy().select(snapshot, event)
    assert decision.selected_segment_ids == ("seg.user", "seg.call", "seg.result")
    assert "seg.open" in decision.rejected_segment_ids
    assert "seg.current" in decision.rejected_segment_ids


def test_budget_rejects_whole_tool_closure_deterministically() -> None:
    snapshot = _snapshot(current="turn.3")
    event = HostTriggerAdapter().event("task_completed", source_revision=snapshot.context_revision, payload={"snapshot": snapshot.snapshot_id})
    decision = DeterministicSourceSelectionPolicy(SourceSelectionConfig(max_tokens=3)).select(snapshot, event)
    assert decision.selected_segment_ids == ("seg.user",)
    assert set(decision.rejected_segment_ids) >= {"seg.call", "seg.result"}
    assert decision.source_digest


def test_selected_and_incremental_modes_are_distinct_and_replayable() -> None:
    snapshot = _snapshot(current="turn.3")
    event = HostTriggerAdapter().event("task_completed", source_revision=snapshot.context_revision, payload={"snapshot": snapshot.snapshot_id})
    selected = DeterministicSourceSelectionPolicy(SourceSelectionConfig(
        projection_mode=ProjectionMode.SELECTED_COMPLETED_SEGMENTS,
        selected_segment_ids=("seg.user",),
    )).select(snapshot, event)
    incremental = DeterministicSourceSelectionPolicy(SourceSelectionConfig(
        projection_mode=ProjectionMode.INCREMENTAL_REVISION,
    )).select(snapshot, event, previous_segment_ids=("seg.user",))
    replay = DeterministicSourceSelectionPolicy(SourceSelectionConfig(
        projection_mode=ProjectionMode.SELECTED_COMPLETED_SEGMENTS,
        selected_segment_ids=("seg.user",),
    )).select(snapshot, event)
    assert selected.selected_segment_ids == ("seg.user",)
    assert incremental.selected_segment_ids == ("seg.call", "seg.result")
    assert selected.decision_id == replay.decision_id
    assert selected.source_digest != incremental.source_digest


def test_non_completed_snapshot_skips_without_source() -> None:
    snapshot = _snapshot(current="turn.3", state=TaskLifecycleState.ACTIVE)
    event = HostTriggerAdapter().event("task_completed", source_revision=snapshot.context_revision, payload={"snapshot": snapshot.snapshot_id})
    decision = DeterministicSourceSelectionPolicy().select(snapshot, event)
    assert decision.action.value == "SKIP"
    assert not decision.selected_segment_ids


def test_revision_mismatch_fails_closed() -> None:
    snapshot = _snapshot(current="turn.3")
    event = HostTriggerAdapter().event("task_completed", source_revision="revision.other", payload={})
    with pytest.raises(ValueError, match="revision"):
        DeterministicSourceSelectionPolicy().select(snapshot, event)


def test_non_projectable_host_roles_are_rejected_before_extraction() -> None:
    base = _snapshot(current="turn.3")
    system = SnapshotSegment(
        "seg.system", "msg.system", "system", "host-only", "turn.0", 1, completed=True,
    )
    snapshot = ContextSnapshot(
        base.run_id, base.episode_id, base.session_id, base.task_id, base.snapshot_id,
        base.context_revision, (system, *base.segments), base.active_segment_ids,
        base.current_turn_id, base.task_state, base.lifecycle_state, base.tool_closures,
        base.total_token_count + 1, base.provenance,
    )
    event = HostTriggerAdapter().event(
        "task_completed", source_revision=snapshot.context_revision, payload={"snapshot": snapshot.snapshot_id}
    )
    decision = DeterministicSourceSelectionPolicy().select(snapshot, event)
    assert "seg.system" in decision.rejected_segment_ids


def test_skip_records_no_selected_source() -> None:
    snapshot = _snapshot(current="turn.3")
    event = HostTriggerAdapter().event(
        "session_end", source_revision=snapshot.context_revision, payload={"snapshot": snapshot.snapshot_id}
    )
    decision = DeterministicSourceSelectionPolicy().skip(snapshot, event)
    assert decision.action.value == "SKIP"
    assert decision.selected_segment_ids == ()
    assert decision.reason_codes == ("trigger_not_run",)


def test_projection_can_apply_audited_selection_without_splitting_closure() -> None:
    base = _snapshot(current=None)
    segments = tuple(item for item in base.segments if item.segment_id in {"seg.user", "seg.call", "seg.result"})
    snapshot = ContextSnapshot(
        base.run_id, base.episode_id, base.session_id, base.task_id, base.snapshot_id,
        base.context_revision, segments, (), None, base.task_state, base.lifecycle_state,
        (base.tool_closures[0],), sum(item.token_count for item in segments), base.provenance,
    )
    projection = ExtractionSourceProjector().project(
        snapshot,
        selected_segment_ids=("seg.user", "seg.call", "seg.result"),
    )
    assert projection.source_segment_ids == ("seg.user", "seg.call", "seg.result")
    assert projection.omitted_segment_ids == ()
    with pytest.raises(ValueError, match="splits a tool closure"):
        ExtractionSourceProjector().project(
            snapshot,
            selected_segment_ids=("seg.user", "seg.call"),
        )
