"""Deterministic Hermes transcript adapter and the first SM01 fixture."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..memory.contracts import MemoryKind
from .contracts import (
    CompletionStatus,
    ContextAction,
    ContextEvaluation,
    ContextEvaluationRequest,
    ContextSegment,
    EvaluationSignal,
    EvaluationTrigger,
    MemoryScope,
    TemporalValidity,
    WritebackAction,
    LIFECYCLE_CONTRACT_SCHEMA_VERSION,
)
from .snapshot import (
    ContextSnapshot,
    ProvenanceRef,
    SegmentKind,
    SnapshotSegment,
    TaskLifecycleState,
    ToolClosure,
)
from .writeback import DryRunReceipt, WritebackCoordinator, WritebackEvent, WritebackPlan


def _canonical_hash(prefix: str, value: object, *, length: int = 24) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


@dataclass(frozen=True, slots=True)
class HermesMessage:
    """Minimal host-neutral view of a Hermes transcript message."""

    message_id: str
    role: str
    content: str
    turn_id: str
    token_count: int
    kind: SegmentKind = SegmentKind.MESSAGE
    completed: bool = False
    tool_call_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(not value.strip() for value in (self.message_id, self.role, self.turn_id)):
            raise ValueError("Hermes message identifiers and role must not be empty")
        if not self.content.strip():
            raise ValueError("Hermes message content must not be empty")
        if self.token_count < 0:
            raise ValueError("Hermes message token_count must not be negative")
        object.__setattr__(self, "kind", SegmentKind(self.kind))
        if self.kind in {SegmentKind.TOOL_CALL, SegmentKind.TOOL_RESULT}:
            if self.tool_call_id is None or not self.tool_call_id.strip():
                raise ValueError("Hermes tool messages require tool_call_id")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class HermesSnapshotCollector:
    """Convert stable Hermes message identities into a lifecycle snapshot."""

    def collect(
        self,
        messages: Sequence[HermesMessage],
        *,
        run_id: str,
        episode_id: str,
        session_id: str,
        task_id: str,
        current_turn_id: str | None,
        task_state: TaskLifecycleState,
        lifecycle_state: str,
        source_ref: str,
        active_message_ids: Sequence[str] = (),
    ) -> ContextSnapshot:
        if not messages:
            raise ValueError("Hermes snapshot requires messages")
        source_ids = [message.message_id for message in messages]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Hermes message IDs must be unique")
        source_to_segment = {
            message.message_id: _canonical_hash(
                "seg",
                {
                    "session_id": session_id,
                    "task_id": task_id,
                    "message_id": message.message_id,
                    "role": message.role,
                    "kind": message.kind.value,
                    "tool_call_id": message.tool_call_id,
                },
            )
            for message in messages
        }
        unknown_active = set(active_message_ids) - set(source_to_segment)
        if unknown_active:
            raise ValueError("active Hermes messages must be present in the transcript")

        segments = tuple(
            SnapshotSegment(
                segment_id=source_to_segment[message.message_id],
                source_message_id=message.message_id,
                role=message.role,
                content=message.content,
                turn_id=message.turn_id,
                token_count=message.token_count,
                kind=message.kind,
                completed=message.completed,
                tool_call_id=message.tool_call_id,
                metadata=message.metadata,
            )
            for message in messages
        )
        closures = self._build_tool_closures(messages, source_to_segment, session_id)
        active_segment_ids = tuple(source_to_segment[item] for item in active_message_ids)
        revision_payload = {
            "schema_version": LIFECYCLE_CONTRACT_SCHEMA_VERSION,
            "segments": [
                {
                    "segment_id": segment.segment_id,
                    "content_sha256": hashlib.sha256(segment.content.encode("utf-8")).hexdigest(),
                    "turn_id": segment.turn_id,
                    "token_count": segment.token_count,
                    "completed": segment.completed,
                }
                for segment in segments
            ],
            "active_segment_ids": active_segment_ids,
            "current_turn_id": current_turn_id,
            "task_state": TaskLifecycleState(task_state).value,
            "lifecycle_state": lifecycle_state,
            "tool_closures": [
                {
                    "closure_id": closure.closure_id,
                    "segment_ids": closure.segment_ids,
                    "closed": closure.closed,
                }
                for closure in closures
            ],
        }
        context_revision = _canonical_hash("rev", revision_payload, length=40)
        snapshot_id = _canonical_hash(
            "snapshot",
            {
                "schema_version": LIFECYCLE_CONTRACT_SCHEMA_VERSION,
                "run_id": run_id,
                "episode_id": episode_id,
                "session_id": session_id,
                "task_id": task_id,
                "source_ref": source_ref,
                "lifecycle_state": lifecycle_state,
            },
        )
        provenance = ProvenanceRef(
            run_id=run_id,
            episode_id=episode_id,
            session_id=session_id,
            task_id=task_id,
            snapshot_id=snapshot_id,
            source_ref=source_ref,
        )
        return ContextSnapshot(
            run_id=run_id,
            episode_id=episode_id,
            session_id=session_id,
            task_id=task_id,
            snapshot_id=snapshot_id,
            context_revision=context_revision,
            segments=segments,
            active_segment_ids=active_segment_ids,
            current_turn_id=current_turn_id,
            task_state=task_state,
            lifecycle_state=lifecycle_state,
            tool_closures=closures,
            total_token_count=sum(segment.token_count for segment in segments),
            provenance=provenance,
        )

    @staticmethod
    def _build_tool_closures(
        messages: Sequence[HermesMessage],
        source_to_segment: Mapping[str, str],
        session_id: str,
    ) -> tuple[ToolClosure, ...]:
        grouped: dict[str, dict[SegmentKind, HermesMessage]] = {}
        order: list[str] = []
        for message in messages:
            if message.kind not in {SegmentKind.TOOL_CALL, SegmentKind.TOOL_RESULT}:
                continue
            assert message.tool_call_id is not None
            if message.tool_call_id not in grouped:
                grouped[message.tool_call_id] = {}
                order.append(message.tool_call_id)
            kind_group = grouped[message.tool_call_id]
            if message.kind in kind_group:
                raise ValueError("a tool call ID may have only one call and one result")
            kind_group[message.kind] = message

        closures: list[ToolClosure] = []
        for tool_call_id in order:
            group = grouped[tool_call_id]
            call = group.get(SegmentKind.TOOL_CALL)
            if call is None:
                raise ValueError("Hermes tool result has no matching call")
            result = group.get(SegmentKind.TOOL_RESULT)
            closures.append(ToolClosure(
                closure_id=_canonical_hash(
                    "closure",
                    {"session_id": session_id, "tool_call_id": tool_call_id},
                ),
                tool_call_id=tool_call_id,
                call_segment_id=source_to_segment[call.message_id],
                result_segment_id=(
                    source_to_segment[result.message_id] if result is not None else None
                ),
            ))
        return tuple(closures)


class HermesStateSnapshotCollector:
    """Project persisted Hermes ``state.db`` message rows into a snapshot.

    Hermes' in-memory conversation dictionaries do not carry stable message
    identities. The persisted row ID is therefore the authoritative source
    identity at lifecycle boundaries. Structured tool calls are expanded into
    their own segments so call/result closure can be validated explicitly.
    """

    def __init__(self, collector: HermesSnapshotCollector | None = None) -> None:
        self._collector = collector or HermesSnapshotCollector()

    def collect(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        run_id: str,
        episode_id: str,
        session_id: str,
        task_id: str,
        task_state: TaskLifecycleState,
        lifecycle_state: str,
        source_ref: str,
    ) -> ContextSnapshot:
        messages = self._project_rows(rows)
        return self._collector.collect(
            messages,
            run_id=run_id,
            episode_id=episode_id,
            session_id=session_id,
            task_id=task_id,
            current_turn_id=None,
            task_state=task_state,
            lifecycle_state=lifecycle_state,
            source_ref=source_ref,
        )

    @classmethod
    def _project_rows(
        cls,
        rows: Sequence[Mapping[str, Any]],
    ) -> tuple[HermesMessage, ...]:
        if not rows:
            raise ValueError("Hermes state snapshot requires persisted messages")

        messages: list[HermesMessage] = []
        seen_row_ids: set[int] = set()
        seen_tool_calls: set[str] = set()
        seen_tool_results: set[str] = set()
        current_turn_id: str | None = None

        for row in rows:
            row_id = row.get("id")
            if type(row_id) is not int or row_id < 1:
                raise ValueError("Hermes state message requires a positive integer row ID")
            if row_id in seen_row_ids:
                raise ValueError("Hermes state message row IDs must be unique")
            seen_row_ids.add(row_id)

            role = row.get("role")
            if not isinstance(role, str) or not role.strip():
                raise ValueError("Hermes state message role must not be empty")
            role = role.strip()
            if role == "user":
                current_turn_id = f"turn_db_{row_id}"
            if current_turn_id is None:
                current_turn_id = f"turn_preamble_{row_id}"

            content = row.get("content")
            if content is not None and not isinstance(content, str):
                raise ValueError("Hermes state message content must be text or null")
            content = content or ""
            tool_call_id = row.get("tool_call_id")
            if tool_call_id is not None and (
                not isinstance(tool_call_id, str) or not tool_call_id.strip()
            ):
                raise ValueError("Hermes tool result ID must be non-empty text")

            if role == "tool":
                if not tool_call_id:
                    raise ValueError("Hermes tool result has no matching call ID")
                if tool_call_id in seen_tool_results:
                    raise ValueError("Hermes tool call may have only one result")
                seen_tool_results.add(tool_call_id)
                projected_content = content if content.strip() else "[empty tool result]"
                messages.append(HermesMessage(
                    message_id=f"db:{row_id}:tool-result:{tool_call_id}",
                    role=role,
                    content=projected_content,
                    turn_id=current_turn_id,
                    token_count=cls._token_count(row, projected_content),
                    kind=SegmentKind.TOOL_RESULT,
                    completed=True,
                    tool_call_id=tool_call_id,
                    metadata=cls._token_metadata(row),
                ))
            elif content.strip():
                messages.append(HermesMessage(
                    message_id=f"db:{row_id}:message",
                    role=role,
                    content=content,
                    turn_id=current_turn_id,
                    token_count=cls._token_count(row, content),
                    completed=True,
                    metadata=cls._token_metadata(row),
                ))

            for call in cls._tool_calls(row):
                call_id = call.get("id")
                function = call.get("function")
                if (
                    not isinstance(call_id, str)
                    or not call_id.strip()
                    or not isinstance(function, Mapping)
                ):
                    raise ValueError("Hermes tool call requires stable ID and function")
                if call_id in seen_tool_calls:
                    raise ValueError("Hermes tool call IDs must be unique")
                seen_tool_calls.add(call_id)
                call_content = json.dumps(
                    {"function": dict(function)},
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                messages.append(HermesMessage(
                    message_id=f"db:{row_id}:tool-call:{call_id}",
                    role="assistant",
                    content=call_content,
                    turn_id=current_turn_id,
                    token_count=cls._estimate_tokens(call_content),
                    kind=SegmentKind.TOOL_CALL,
                    completed=True,
                    tool_call_id=call_id,
                    metadata={"token_count_source": "deterministic_estimate"},
                ))

        orphan_results = seen_tool_results - seen_tool_calls
        open_calls = seen_tool_calls - seen_tool_results
        if orphan_results:
            raise ValueError("Hermes tool result has no matching call")
        if open_calls:
            raise ValueError("Hermes snapshot contains an open tool call")
        if not messages:
            raise ValueError("Hermes state snapshot has no projectable messages")
        return tuple(messages)

    @staticmethod
    def _tool_calls(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        value = row.get("tool_calls")
        if value is None or value == "":
            return ()
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("Hermes tool_calls JSON is malformed") from exc
        if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
            raise ValueError("Hermes tool_calls must be a list of objects")
        return tuple(value)

    @classmethod
    def _token_count(cls, row: Mapping[str, Any], content: str) -> int:
        value = row.get("token_count")
        if value is None:
            return cls._estimate_tokens(content)
        if type(value) is not int or value < 0:
            raise ValueError("Hermes message token_count must be a non-negative integer")
        return value

    @staticmethod
    def _token_metadata(row: Mapping[str, Any]) -> Mapping[str, str]:
        return {
            "token_count_source": (
                "hermes_state" if row.get("token_count") is not None
                else "deterministic_estimate"
            )
        }

    @staticmethod
    def _estimate_tokens(content: str) -> int:
        # Hermes persists no per-message usage by default. Keep the fallback
        # explicit and deterministic instead of representing unknown as zero.
        return max(1, (len(content.encode("utf-8")) + 3) // 4)


def snapshot_to_evaluation_request(
    snapshot: ContextSnapshot,
    *,
    evaluation_id: str,
    trigger: EvaluationTrigger = EvaluationTrigger.TASK_COMPLETED,
    turn_index: int = 0,
    policy_version: str | None = None,
) -> ContextEvaluationRequest:
    if policy_version is not None and not policy_version.strip():
        raise ValueError("policy_version must be non-empty when present")
    return ContextEvaluationRequest(
        evaluation_id=evaluation_id,
        session_id=snapshot.session_id,
        task_id=snapshot.task_id,
        trigger=trigger,
        turn_index=turn_index,
        context_revision=snapshot.context_revision,
        active_segment_ids=tuple(sorted(snapshot.protected_segment_ids)),
        context_tokens=snapshot.total_token_count,
        segments=tuple(
            ContextSegment(
                segment_id=segment.segment_id,
                role=segment.role,
                content=segment.content,
                token_count=segment.token_count,
                completed=segment.completed,
                metadata={
                    "source_message_id": segment.source_message_id,
                    "turn_id": segment.turn_id,
                    "segment_kind": segment.kind.value,
                },
            )
            for segment in snapshot.segments
        ),
        metadata={
            "snapshot_id": snapshot.snapshot_id,
            "policy_version": policy_version,
        },
    )


class DeterministicPreferenceEvaluator:
    """Fixture evaluator selecting one known preference segment without an LLM."""

    name = "deterministic-sm01-preference"

    def __init__(self, preference_segment_id: str, *, policy_version: str = "static-v0") -> None:
        self.preference_segment_id = preference_segment_id
        self.policy_version = policy_version

    def evaluate(self, request: ContextEvaluationRequest) -> ContextEvaluation:
        if self.preference_segment_id not in {item.segment_id for item in request.segments}:
            raise ValueError("preference segment is absent from the evaluation request")
        signals = tuple(
            EvaluationSignal(
                segment_id=segment.segment_id,
                context_action=(
                    ContextAction.EVICT
                    if segment.segment_id == self.preference_segment_id
                    else ContextAction.RETAIN
                ),
                writeback_action=(
                    WritebackAction.ADD
                    if segment.segment_id == self.preference_segment_id
                    else WritebackAction.DEFER
                ),
                memory_kind=(
                    MemoryKind.SEMANTIC
                    if segment.segment_id == self.preference_segment_id
                    else None
                ),
                utility_estimate=1.0 if segment.segment_id == self.preference_segment_id else 0.0,
                confidence=1.0,
                completion_status=(
                    CompletionStatus.COMPLETED
                    if segment.completed
                    else CompletionStatus.IN_PROGRESS
                ),
                completion_evidence=(
                    ("host_segment_completed",)
                    if segment.completed
                    else ("host_segment_unresolved",)
                ),
                safe_to_evict=(
                    segment.completed
                    and segment.segment_id not in request.active_segment_ids
                ),
                unresolved_state=None if segment.completed else "host_unresolved",
                scope=(
                    MemoryScope.USER
                    if segment.segment_id == self.preference_segment_id
                    else MemoryScope.TASK
                ),
                temporal_validity=(
                    TemporalValidity.DURABLE
                    if segment.segment_id == self.preference_segment_id
                    else TemporalValidity.CURRENT
                ),
                provenance=(
                    str(request.metadata.get("snapshot_id") or "snapshot_unknown"),
                    segment.segment_id,
                ),
                reason_codes=(
                    ("preference_candidate",)
                    if segment.segment_id == self.preference_segment_id
                    else ("no_action",)
                ),
            )
            for segment in request.segments
        )
        return ContextEvaluation(
            evaluation_id=request.evaluation_id,
            evaluator=self.name,
            trigger=request.trigger,
            signals=signals,
            policy_version=self.policy_version,
            input_chars=sum(len(segment.content) for segment in request.segments),
        )


@dataclass(frozen=True, slots=True)
class Sm01FixtureResult:
    snapshot: ContextSnapshot
    evaluation: ContextEvaluation
    plans: tuple[WritebackPlan, ...]
    receipts: tuple[DryRunReceipt, ...]
    events: tuple[WritebackEvent, ...]


def run_sm01_preference_fixture() -> Sm01FixtureResult:
    """Run the first learn -> snapshot -> plan -> dry-run semantic path."""

    messages = (
        HermesMessage(
            message_id="sm01-learn-preference",
            role="user",
            content="Use TSV with owner, priority, task, and due_date.",
            turn_id="learn-turn",
            token_count=12,
            completed=True,
        ),
        HermesMessage(
            message_id="sm01-preference-ack",
            role="assistant",
            content="I will use that table format for future task lists.",
            turn_id="learn-turn",
            token_count=11,
            completed=True,
        ),
        HermesMessage(
            message_id="sm01-task-complete",
            role="assistant",
            content="The current task is complete.",
            turn_id="completion-turn",
            token_count=6,
            completed=True,
        ),
    )
    collector = HermesSnapshotCollector()
    snapshot = collector.collect(
        messages,
        run_id="run-sm01-fixture",
        episode_id="episode-sm01-learn",
        session_id="session-sm01-learn",
        task_id="SM01_preference_adoption",
        current_turn_id="completion-turn",
        task_state=TaskLifecycleState.COMPLETED,
        lifecycle_state="task_completed",
        source_ref="fixture:sm01_preference_adoption:v1",
        active_message_ids=("sm01-task-complete",),
    )
    preference_segment_id = snapshot.segments[0].segment_id
    request = snapshot_to_evaluation_request(
        snapshot,
        evaluation_id="evaluation-sm01-static-v0",
        turn_index=2,
    )
    evaluation = DeterministicPreferenceEvaluator(preference_segment_id).evaluate(request)
    events: list[WritebackEvent] = []

    class _Observer:
        def record(self, event: WritebackEvent) -> None:
            events.append(event)

    coordinator = WritebackCoordinator(observers=(_Observer(),))
    plans = coordinator.create_plans(snapshot, evaluation)
    receipts = tuple(coordinator.dry_run(plan, snapshot) for plan in plans)
    return Sm01FixtureResult(snapshot, evaluation, plans, receipts, tuple(events))
