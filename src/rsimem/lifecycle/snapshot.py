"""Stable, host-neutral context snapshots for lifecycle evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping


def _frozen_metadata(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


class TaskLifecycleState(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SegmentKind(StrEnum):
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


@dataclass(frozen=True, slots=True)
class ProvenanceRef:
    """Stable identifiers joining one lifecycle object to its origin."""

    run_id: str
    episode_id: str
    session_id: str
    task_id: str
    snapshot_id: str
    source_ref: str
    segment_ids: tuple[str, ...] = ()
    evaluation_id: str | None = None
    mutation_id: str | None = None

    def __post_init__(self) -> None:
        required = (
            self.run_id,
            self.episode_id,
            self.session_id,
            self.task_id,
            self.snapshot_id,
            self.source_ref,
        )
        if any(not value.strip() for value in required):
            raise ValueError("provenance identifiers must not be empty")
        if len(self.segment_ids) != len(set(self.segment_ids)):
            raise ValueError("provenance segment IDs must be unique")


@dataclass(frozen=True, slots=True)
class SnapshotSegment:
    """A stable context unit. Raw content is confined to the snapshot layer."""

    segment_id: str
    source_message_id: str
    role: str
    content: str
    turn_id: str
    token_count: int
    kind: SegmentKind = SegmentKind.MESSAGE
    completed: bool = False
    tool_call_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (self.segment_id, self.source_message_id, self.role, self.turn_id)
        ):
            raise ValueError("snapshot segment identifiers and role must not be empty")
        if not self.content.strip():
            raise ValueError("snapshot segment content must not be empty")
        if self.token_count < 0:
            raise ValueError("snapshot segment token_count must not be negative")
        object.__setattr__(self, "kind", SegmentKind(self.kind))
        if self.kind in {SegmentKind.TOOL_CALL, SegmentKind.TOOL_RESULT}:
            if self.tool_call_id is None or not self.tool_call_id.strip():
                raise ValueError("tool segments require tool_call_id")
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ToolClosure:
    """Association between one tool call and its optional result."""

    closure_id: str
    tool_call_id: str
    call_segment_id: str
    result_segment_id: str | None = None

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (self.closure_id, self.tool_call_id, self.call_segment_id)
        ):
            raise ValueError("tool closure identifiers must not be empty")
        if self.result_segment_id is not None and not self.result_segment_id.strip():
            raise ValueError("tool result segment ID must not be empty")

    @property
    def closed(self) -> bool:
        return self.result_segment_id is not None

    @property
    def segment_ids(self) -> tuple[str, ...]:
        if self.result_segment_id is None:
            return (self.call_segment_id,)
        return (self.call_segment_id, self.result_segment_id)


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    """Immutable context captured at one host lifecycle boundary."""

    run_id: str
    episode_id: str
    session_id: str
    task_id: str
    snapshot_id: str
    context_revision: str
    segments: tuple[SnapshotSegment, ...]
    active_segment_ids: tuple[str, ...]
    current_turn_id: str
    task_state: TaskLifecycleState
    lifecycle_state: str
    tool_closures: tuple[ToolClosure, ...]
    total_token_count: int
    provenance: ProvenanceRef

    def __post_init__(self) -> None:
        required = (
            self.run_id,
            self.episode_id,
            self.session_id,
            self.task_id,
            self.snapshot_id,
            self.context_revision,
            self.current_turn_id,
            self.lifecycle_state,
        )
        if any(not value.strip() for value in required):
            raise ValueError("snapshot identifiers and state must not be empty")
        if not self.segments:
            raise ValueError("context snapshot requires at least one segment")
        object.__setattr__(self, "task_state", TaskLifecycleState(self.task_state))

        ids = [segment.segment_id for segment in self.segments]
        id_set = set(ids)
        if len(ids) != len(id_set):
            raise ValueError("snapshot segment IDs must be unique")
        if len(self.active_segment_ids) != len(set(self.active_segment_ids)):
            raise ValueError("active segment IDs must be unique")
        if not set(self.active_segment_ids).issubset(id_set):
            raise ValueError("active segments must be present in the snapshot")
        expected_tokens = sum(segment.token_count for segment in self.segments)
        if self.total_token_count != expected_tokens:
            raise ValueError("snapshot total_token_count must equal the segment total")

        closure_ids: set[str] = set()
        closure_segments: set[str] = set()
        by_id = {segment.segment_id: segment for segment in self.segments}
        for closure in self.tool_closures:
            if closure.closure_id in closure_ids:
                raise ValueError("tool closure IDs must be unique")
            closure_ids.add(closure.closure_id)
            if not set(closure.segment_ids).issubset(id_set):
                raise ValueError("tool closure segments must be present in the snapshot")
            if closure_segments.intersection(closure.segment_ids):
                raise ValueError("a tool segment cannot belong to multiple closures")
            closure_segments.update(closure.segment_ids)
            call = by_id[closure.call_segment_id]
            if call.kind != SegmentKind.TOOL_CALL or call.tool_call_id != closure.tool_call_id:
                raise ValueError("tool closure call does not match its segment")
            if closure.result_segment_id is not None:
                result = by_id[closure.result_segment_id]
                if result.kind != SegmentKind.TOOL_RESULT or result.tool_call_id != closure.tool_call_id:
                    raise ValueError("tool closure result does not match its segment")

        expected_provenance = (
            self.run_id,
            self.episode_id,
            self.session_id,
            self.task_id,
            self.snapshot_id,
        )
        actual_provenance = (
            self.provenance.run_id,
            self.provenance.episode_id,
            self.provenance.session_id,
            self.provenance.task_id,
            self.provenance.snapshot_id,
        )
        if actual_provenance != expected_provenance:
            raise ValueError("snapshot provenance does not match snapshot identity")

    @property
    def protected_segment_ids(self) -> frozenset[str]:
        current = {
            segment.segment_id
            for segment in self.segments
            if segment.turn_id == self.current_turn_id
        }
        open_tools = {
            closure.call_segment_id for closure in self.tool_closures if not closure.closed
        }
        return frozenset(self.active_segment_ids) | current | open_tools
