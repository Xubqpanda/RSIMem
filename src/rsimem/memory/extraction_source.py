"""Canonical completed-task source projection for semantic extraction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:
    from ..lifecycle import ContextSnapshot, SegmentKind, TaskLifecycleState
from .contracts import MemoryExperience, MemoryMessage


EXTRACTION_SOURCE_SCHEMA_VERSION = 1
EXTRACTION_SOURCE_SCHEMA = "completed-task-extraction-source-v1"
DEFAULT_EXTRACTION_SOURCE_MAX_CHARS = 32_000
EXTRACTION_SOURCE_ALLOWED_ROLES = ("user", "assistant", "tool")
EXTRACTION_SOURCE_METADATA_ALLOWLIST = (
    "segment_id",
    "source_message_id",
    "segment_kind",
    "tool_call_id",
)


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ExtractionSourceMessage:
    segment_id: str
    source_message_id: str
    role: str
    content: str
    segment_kind: SegmentKind
    tool_call_id: str | None = None
    content_truncated: bool = False

    def __post_init__(self) -> None:
        from ..lifecycle import SegmentKind as _SegmentKind

        object.__setattr__(self, "segment_kind", _SegmentKind(self.segment_kind))
        if any(not value.strip() for value in (
            self.segment_id,
            self.source_message_id,
            self.role,
            self.content,
        )):
            raise ValueError("extraction source message is incomplete")
        if self.role not in EXTRACTION_SOURCE_ALLOWED_ROLES:
            raise ValueError("extraction source message role is not allowed")
        if type(self.content_truncated) is not bool:
            raise TypeError("content_truncated must be bool")
        tool_kind = self.segment_kind in {
            _SegmentKind.TOOL_CALL,
            _SegmentKind.TOOL_RESULT,
        }
        if tool_kind != bool(self.tool_call_id):
            raise ValueError("extraction tool message identity is inconsistent")

    def prompt_payload(self) -> dict[str, object]:
        return {
            "segment_id": self.segment_id,
            "source_message_id": self.source_message_id,
            "role": self.role,
            "content": self.content,
            "segment_kind": self.segment_kind.value,
            "tool_call_id": self.tool_call_id,
            "content_truncated": self.content_truncated,
        }

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionSourceMessage":
        fields = {
            "segment_id",
            "source_message_id",
            "role",
            "content",
            "segment_kind",
            "tool_call_id",
            "content_truncated",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("malformed extraction source message")
        try:
            from ..lifecycle import SegmentKind as _SegmentKind

            return cls(
                segment_id=value["segment_id"],
                source_message_id=value["source_message_id"],
                role=value["role"],
                content=value["content"],
                segment_kind=_SegmentKind(value["segment_kind"]),
                tool_call_id=value["tool_call_id"],
                content_truncated=value["content_truncated"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed extraction source message") from exc


@dataclass(frozen=True, slots=True)
class ExtractionSourceProjection:
    projection_id: str
    snapshot_id: str
    task_id: str
    context_revision: str
    messages: tuple[ExtractionSourceMessage, ...]
    source_message_ids: tuple[str, ...]
    source_segment_ids: tuple[str, ...]
    omitted_segment_ids: tuple[str, ...]
    truncated_segment_ids: tuple[str, ...]
    max_content_chars: int
    projected_content_chars: int
    projection_digest: str
    schema: str = EXTRACTION_SOURCE_SCHEMA
    schema_version: int = EXTRACTION_SOURCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != EXTRACTION_SOURCE_SCHEMA_VERSION
            or self.schema != EXTRACTION_SOURCE_SCHEMA
        ):
            raise ValueError("unsupported extraction source projection schema")
        if not self.messages:
            raise ValueError("extraction source projection requires messages")
        if type(self.max_content_chars) is not int or self.max_content_chars < 2:
            raise ValueError("extraction source content budget is invalid")
        if self.projected_content_chars != sum(
            len(message.content) for message in self.messages
        ) or self.projected_content_chars > self.max_content_chars:
            raise ValueError("extraction source content accounting is invalid")
        if self.source_message_ids != tuple(
            message.source_message_id for message in self.messages
        ) or self.source_segment_ids != tuple(
            message.segment_id for message in self.messages
        ):
            raise ValueError("extraction source IDs must exactly match messages")
        for values in (
            self.source_message_ids,
            self.source_segment_ids,
            self.omitted_segment_ids,
            self.truncated_segment_ids,
        ):
            if len(values) != len(set(values)):
                raise ValueError("extraction source IDs must be unique")
        if not set(self.truncated_segment_ids).issubset(self.source_segment_ids):
            raise ValueError("truncated extraction segments must remain projected")
        expected = _digest(self.identity_payload())
        if self.projection_digest != expected:
            raise ValueError("extraction source projection digest mismatch")
        if self.projection_id != f"extraction-source.{expected[:40]}":
            raise ValueError("extraction source projection ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "schema": self.schema,
            "snapshot_id": self.snapshot_id,
            "task_id": self.task_id,
            "context_revision": self.context_revision,
            "messages": [message.prompt_payload() for message in self.messages],
            "source_message_ids": list(self.source_message_ids),
            "source_segment_ids": list(self.source_segment_ids),
            "omitted_segment_ids": list(self.omitted_segment_ids),
            "truncated_segment_ids": list(self.truncated_segment_ids),
            "max_content_chars": self.max_content_chars,
            "projected_content_chars": self.projected_content_chars,
        }

    def prompt_messages(self) -> list[dict[str, object]]:
        return [message.prompt_payload() for message in self.messages]

    def payload(self) -> dict[str, object]:
        return {
            "projection_id": self.projection_id,
            "projection_digest": self.projection_digest,
            **self.identity_payload(),
        }

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionSourceProjection":
        fields = {
            "projection_id",
            "projection_digest",
            "schema_version",
            "schema",
            "snapshot_id",
            "task_id",
            "context_revision",
            "messages",
            "source_message_ids",
            "source_segment_ids",
            "omitted_segment_ids",
            "truncated_segment_ids",
            "max_content_chars",
            "projected_content_chars",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("malformed extraction source projection")
        list_fields = (
            "messages",
            "source_message_ids",
            "source_segment_ids",
            "omitted_segment_ids",
            "truncated_segment_ids",
        )
        if any(not isinstance(value[field], list) for field in list_fields):
            raise ValueError("malformed extraction source projection collections")
        try:
            return cls(
                projection_id=value["projection_id"],
                snapshot_id=value["snapshot_id"],
                task_id=value["task_id"],
                context_revision=value["context_revision"],
                messages=tuple(
                    ExtractionSourceMessage.from_payload(item)
                    for item in value["messages"]
                ),
                source_message_ids=tuple(value["source_message_ids"]),
                source_segment_ids=tuple(value["source_segment_ids"]),
                omitted_segment_ids=tuple(value["omitted_segment_ids"]),
                truncated_segment_ids=tuple(value["truncated_segment_ids"]),
                max_content_chars=value["max_content_chars"],
                projected_content_chars=value["projected_content_chars"],
                projection_digest=value["projection_digest"],
                schema=value["schema"],
                schema_version=value["schema_version"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed extraction source projection") from exc

    def to_experience(self, snapshot: ContextSnapshot) -> MemoryExperience:
        if (
            snapshot.snapshot_id != self.snapshot_id
            or snapshot.context_revision != self.context_revision
            or snapshot.task_id != self.task_id
        ):
            raise ValueError("extraction projection belongs to another snapshot")
        return MemoryExperience(
            experience_id=f"experience.{self.projection_id}",
            session_id=snapshot.session_id,
            task_id=snapshot.task_id,
            outcome="completed",
            messages=tuple(
                MemoryMessage(
                    message.role,
                    message.content,
                    metadata={
                        "segment_id": message.segment_id,
                        "source_message_id": message.source_message_id,
                        "segment_kind": message.segment_kind.value,
                        "tool_call_id": message.tool_call_id,
                    },
                )
                for message in self.messages
            ),
        )


class ExtractionSourceProjector:
    def __init__(
        self,
        *,
        max_content_chars: int = DEFAULT_EXTRACTION_SOURCE_MAX_CHARS,
    ) -> None:
        if type(max_content_chars) is not int or max_content_chars < 2:
            raise ValueError("extraction source content budget is invalid")
        self.max_content_chars = max_content_chars

    def project(
        self,
        snapshot: ContextSnapshot,
        *,
        selected_segment_ids: Sequence[str] | None = None,
    ) -> ExtractionSourceProjection:
        from ..lifecycle import TaskLifecycleState as _TaskLifecycleState

        if snapshot.task_state != _TaskLifecycleState.COMPLETED:
            raise ValueError("extraction source requires completed task")
        if snapshot.current_turn_id is not None or snapshot.active_segment_ids:
            raise ValueError("active/current context cannot enter extraction source")
        if any(not segment.completed for segment in snapshot.segments):
            raise ValueError("unresolved segment cannot enter extraction source")
        if any(not closure.closed for closure in snapshot.tool_closures):
            raise ValueError("open tool closure cannot enter extraction source")

        selected: set[str] | None = None
        if selected_segment_ids is not None:
            selected = set(selected_segment_ids)
            if len(selected) != len(tuple(selected_segment_ids)):
                raise ValueError("selected extraction source segment IDs must be unique")
            all_ids = {segment.segment_id for segment in snapshot.segments}
            if not selected.issubset(all_ids):
                raise ValueError("selected extraction source segment is not in snapshot")
            if selected.intersection(snapshot.protected_segment_ids):
                raise ValueError("selected extraction source includes protected segment")
            for closure in snapshot.tool_closures:
                overlap = selected.intersection(closure.segment_ids)
                if overlap and overlap != set(closure.segment_ids):
                    raise ValueError("selected extraction source splits a tool closure")
            selected_roles = {
                segment.role
                for segment in snapshot.segments
                if segment.segment_id in selected
            }
            if selected_roles.difference(EXTRACTION_SOURCE_ALLOWED_ROLES):
                raise ValueError("selected extraction source role is not allowed")
        allowed = tuple(
            segment for segment in snapshot.segments
            if segment.role in EXTRACTION_SOURCE_ALLOWED_ROLES
            and (selected is None or segment.segment_id in selected)
        )
        if not allowed:
            raise ValueError("snapshot has no allowed extraction source messages")
        allowed_ids = {segment.segment_id for segment in allowed}
        closure_by_segment = {}
        for closure in snapshot.tool_closures:
            members = tuple(
                segment_id for segment_id in closure.segment_ids
                if segment_id in allowed_ids
            )
            if members and len(members) != len(closure.segment_ids):
                raise ValueError("role filtering cannot split a tool closure")
            for segment_id in members:
                closure_by_segment[segment_id] = members

        units = []
        consumed = set()
        by_id = {segment.segment_id: segment for segment in allowed}
        positions = {segment.segment_id: index for index, segment in enumerate(allowed)}
        for segment in allowed:
            if segment.segment_id in consumed:
                continue
            ids = closure_by_segment.get(segment.segment_id, (segment.segment_id,))
            unit = tuple(sorted(
                (by_id[value] for value in ids),
                key=lambda item: positions[item.segment_id],
            ))
            consumed.update(ids)
            units.append(unit)

        selected_ids: set[str] = set()
        projected_content: dict[str, str] = {}
        remaining = self.max_content_chars
        truncated_ids: list[str] = []
        for unit in reversed(units):
            unit_chars = sum(len(segment.content) for segment in unit)
            if unit_chars <= remaining:
                for segment in unit:
                    selected_ids.add(segment.segment_id)
                    projected_content[segment.segment_id] = segment.content
                remaining -= unit_chars
                continue
            if remaining < len(unit):
                break
            allocations = [1] * len(unit)
            extra = remaining - len(unit)
            for index, segment in enumerate(reversed(unit)):
                take = min(len(segment.content) - 1, extra)
                allocations[len(unit) - index - 1] += take
                extra -= take
            for segment, size in zip(unit, allocations):
                selected_ids.add(segment.segment_id)
                projected_content[segment.segment_id] = segment.content[-size:]
                truncated_ids.append(segment.segment_id)
            remaining = 0
            break

        messages = []
        for segment in allowed:
            if segment.segment_id not in selected_ids:
                continue
            messages.append(ExtractionSourceMessage(
                segment.segment_id,
                segment.source_message_id,
                segment.role,
                projected_content[segment.segment_id],
                segment.kind,
                segment.tool_call_id,
                segment.segment_id in truncated_ids,
            ))
        omitted = tuple(
            segment.segment_id for segment in snapshot.segments
            if segment.segment_id not in selected_ids
        )
        core = {
            "schema_version": EXTRACTION_SOURCE_SCHEMA_VERSION,
            "schema": EXTRACTION_SOURCE_SCHEMA,
            "snapshot_id": snapshot.snapshot_id,
            "task_id": snapshot.task_id,
            "context_revision": snapshot.context_revision,
            "messages": [message.prompt_payload() for message in messages],
            "source_message_ids": [message.source_message_id for message in messages],
            "source_segment_ids": [message.segment_id for message in messages],
            "omitted_segment_ids": list(omitted),
            "truncated_segment_ids": list(truncated_ids),
            "max_content_chars": self.max_content_chars,
            "projected_content_chars": sum(len(message.content) for message in messages),
        }
        digest = _digest(core)
        return ExtractionSourceProjection(
            projection_id=f"extraction-source.{digest[:40]}",
            snapshot_id=snapshot.snapshot_id,
            task_id=snapshot.task_id,
            context_revision=snapshot.context_revision,
            messages=tuple(messages),
            source_message_ids=tuple(
                message.source_message_id for message in messages
            ),
            source_segment_ids=tuple(message.segment_id for message in messages),
            omitted_segment_ids=omitted,
            truncated_segment_ids=tuple(truncated_ids),
            max_content_chars=self.max_content_chars,
            projected_content_chars=sum(
                len(message.content) for message in messages
            ),
            projection_digest=digest,
        )
