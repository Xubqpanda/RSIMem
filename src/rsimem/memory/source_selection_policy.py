"""Deterministic source-selection policy over a host-neutral context snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..lifecycle.snapshot import ContextSnapshot, SegmentKind, TaskLifecycleState
from .policy_contracts import (
    DecisionAction,
    ExecutionStatus,
    PolicyArtifactIdentity,
    PolicyArtifactKind,
    PolicyLayer,
    ProjectionMode,
    SafetyBoundary,
    SourceSelectionDecision,
    TriggerEvent,
    content_digest,
)


@dataclass(frozen=True, slots=True)
class SourceSelectionConfig:
    policy_version: str = "fixed.source.parent.v1"
    projection_mode: ProjectionMode = ProjectionMode.WHOLE_COMPLETED_TASK
    max_tokens: int | None = None
    selected_segment_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "projection_mode", ProjectionMode(self.projection_mode))
        if not self.policy_version.strip():
            raise ValueError("source selection policy version must not be empty")
        if self.max_tokens is not None and (type(self.max_tokens) is not int or self.max_tokens < 1):
            raise ValueError("source selection token budget must be positive")
        if len(self.selected_segment_ids) != len(set(self.selected_segment_ids)) or any(not item.strip() for item in self.selected_segment_ids):
            raise ValueError("selected segment IDs must be unique non-empty strings")


class DeterministicSourceSelectionPolicy:
    """Whole-task parent with auditable selected/incremental variants."""

    layer = PolicyLayer.SOURCE_SELECTION

    def __init__(self, config: SourceSelectionConfig | None = None) -> None:
        self.config = config or SourceSelectionConfig()

    @property
    def artifact_identity(self) -> PolicyArtifactIdentity:
        return PolicyArtifactIdentity.create(
            policy_version=self.config.policy_version,
            kind=PolicyArtifactKind.FIXED,
            layers=(PolicyLayer.SOURCE_SELECTION,),
        )

    def select(
        self,
        snapshot: ContextSnapshot,
        event: TriggerEvent,
        *,
        previous_segment_ids: Sequence[str] = (),
    ) -> SourceSelectionDecision:
        if event.source_revision != snapshot.context_revision:
            raise ValueError("source selection event revision does not match snapshot")
        safety = SafetyBoundary(
            active_segment_ids=snapshot.active_segment_ids,
            current_turn_id=snapshot.current_turn_id,
            current_turn_segment_ids=tuple(
                segment.segment_id
                for segment in snapshot.segments
                if snapshot.current_turn_id is not None and segment.turn_id == snapshot.current_turn_id
            ),
            tool_closures=tuple(closure.segment_ids for closure in snapshot.tool_closures),
        )
        by_id = {segment.segment_id: segment for segment in snapshot.segments}
        previous = set(previous_segment_ids)
        if not previous.issubset(by_id):
            raise ValueError("previous source segment IDs are not in snapshot")
        if snapshot.task_state != TaskLifecycleState.COMPLETED:
            return self._decision(
                snapshot,
                event,
                safety,
                selected=(),
                skipped=tuple(segment.segment_id for segment in snapshot.segments),
                rejected=(),
                reason="task_not_completed",
            )

        candidates = []
        rejected: list[str] = []
        protected = set(snapshot.protected_segment_ids)
        for segment in snapshot.segments:
            if not segment.completed:
                rejected.append(segment.segment_id)
                continue
            if segment.segment_id in protected:
                rejected.append(segment.segment_id)
                continue
            if segment.role not in {"user", "assistant", "tool"}:
                rejected.append(segment.segment_id)
                continue
            if self.config.projection_mode == ProjectionMode.SELECTED_COMPLETED_SEGMENTS and segment.segment_id not in set(self.config.selected_segment_ids):
                continue
            if self.config.projection_mode == ProjectionMode.INCREMENTAL_REVISION and segment.segment_id in previous:
                continue
            candidates.append(segment)

        closures = {item for closure in snapshot.tool_closures for item in closure.segment_ids}
        selected: list[str] = []
        skipped: list[str] = []
        consumed = 0
        seen_closure: set[str] = set()
        closure_by_segment = {
            segment_id: closure.segment_ids
            for closure in snapshot.tool_closures
            for segment_id in closure.segment_ids
        }
        for segment in candidates:
            group = closure_by_segment.get(segment.segment_id, (segment.segment_id,))
            group_key = ":".join(group)
            if group_key in seen_closure:
                continue
            seen_closure.add(group_key)
            group_segments = [by_id[item] for item in group]
            if any(item.segment_id in protected for item in group_segments):
                rejected.extend(item.segment_id for item in group_segments)
                continue
            if self.config.max_tokens is not None and consumed + sum(item.token_count for item in group_segments) > self.config.max_tokens:
                rejected.extend(item.segment_id for item in group_segments)
                continue
            selected.extend(item.segment_id for item in group_segments)
            consumed += sum(item.token_count for item in group_segments)

        selected_set = set(selected)
        candidate_set = {item.segment_id for item in candidates}
        rejected_set = set(rejected)
        skipped.extend(sorted(candidate_set.difference(selected_set).difference(rejected)))
        skipped.extend(
            segment.segment_id
            for segment in snapshot.segments
            if segment.segment_id not in candidate_set and segment.segment_id not in protected and segment.segment_id not in closures
            and segment.segment_id not in rejected_set
        )
        reason = "source_selected" if selected else "no_eligible_source"
        return self._decision(snapshot, event, safety, selected=tuple(selected), skipped=tuple(dict.fromkeys(skipped)), rejected=tuple(dict.fromkeys(rejected)), reason=reason)

    def select_source(self, snapshot: ContextSnapshot, event: TriggerEvent, *, previous_segment_ids: Sequence[str] = ()) -> SourceSelectionDecision:
        return self.select(snapshot, event, previous_segment_ids=previous_segment_ids)

    def skip(
        self,
        snapshot: ContextSnapshot,
        event: TriggerEvent,
        *,
        reason: str = "trigger_not_run",
    ) -> SourceSelectionDecision:
        """Record an explicit non-selection when trigger policy did not RUN."""

        if event.source_revision != snapshot.context_revision:
            raise ValueError("source selection event revision does not match snapshot")
        safety = SafetyBoundary(
            active_segment_ids=snapshot.active_segment_ids,
            current_turn_id=snapshot.current_turn_id,
            current_turn_segment_ids=tuple(
                segment.segment_id
                for segment in snapshot.segments
                if snapshot.current_turn_id is not None and segment.turn_id == snapshot.current_turn_id
            ),
            tool_closures=tuple(closure.segment_ids for closure in snapshot.tool_closures),
        )
        return self._decision(
            snapshot,
            event,
            safety,
            selected=(),
            skipped=tuple(segment.segment_id for segment in snapshot.segments),
            rejected=(),
            reason=reason,
        )

    def _decision(self, snapshot: ContextSnapshot, event: TriggerEvent, safety: SafetyBoundary, *, selected: tuple[str, ...], skipped: tuple[str, ...], rejected: tuple[str, ...], reason: str) -> SourceSelectionDecision:
        source_payload = {
            "snapshot_id": snapshot.snapshot_id,
            "context_revision": snapshot.context_revision,
            "segments": [
                {
                    "segment_id": segment.segment_id,
                    "content_digest": content_digest(segment.content),
                    "token_count": segment.token_count,
                }
                for segment in snapshot.segments
                if segment.segment_id in selected
            ],
        }
        action = DecisionAction.RUN if selected else DecisionAction.SKIP
        return SourceSelectionDecision.create(
            policy_version=self.config.policy_version,
            source_revision=snapshot.context_revision,
            input_payload={"event_id": event.event_id, "snapshot_id": snapshot.snapshot_id},
            output_payload={"selected": list(selected), "skipped": list(skipped), "rejected": list(rejected), "reason": reason},
            action=action,
            execution_status=ExecutionStatus.PENDING if action == DecisionAction.RUN else ExecutionStatus.SKIPPED,
            reason_codes=(reason,),
            lineage_id=f"lineage.{event.event_id}",
            trigger_event_id=event.event_id,
            selected_segment_ids=selected,
            skipped_segment_ids=skipped,
            rejected_segment_ids=rejected,
            projection_mode=self.config.projection_mode,
            source_payload=source_payload,
            safety=safety,
            truncation=bool(rejected and self.config.max_tokens is not None),
        )


__all__ = ["SourceSelectionConfig", "DeterministicSourceSelectionPolicy"]
