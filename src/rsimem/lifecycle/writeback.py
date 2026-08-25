"""Validated, content-free writeback plans and dry-run coordination."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Iterable, Protocol, runtime_checkable

from ..memory.contracts import MemoryKind
from .contracts import (
    ContextAction,
    ContextEvaluation,
    EvaluationSignal,
    WritebackAction,
)
from .snapshot import ContextSnapshot, ProvenanceRef


_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _stable_hash(prefix: str, value: object, *, length: int = 24) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


class PlanContextAction(StrEnum):
    KEEP = "keep"
    EVICT = "evict"


class PlanMemoryAction(StrEnum):
    DISCARD = "discard"
    ADD = "add"
    UPDATE = "update"


class PlanValidationStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    STALE = "stale"


class DryRunStatus(StrEnum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    STALE = "stale"
    REJECTED = "rejected"


class WritebackEventKind(StrEnum):
    PLAN_CREATED = "plan_created"
    PLAN_REJECTED = "plan_rejected"
    PLAN_VALIDATED = "plan_validated"
    DRY_RUN_MUTATION = "dry_run_mutation"
    DRY_RUN_DUPLICATE = "dry_run_duplicate"


@dataclass(frozen=True, slots=True)
class RawResourceUsage:
    """Raw quantities only; provider prices and derived objectives live elsewhere."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    model_requests: int = 0
    duration_ms: int | None = None
    storage_bytes: int = 0

    def __post_init__(self) -> None:
        values = (
            self.input_tokens,
            self.output_tokens,
            self.model_requests,
            self.duration_ms,
            self.storage_bytes,
        )
        if any(value is not None and value < 0 for value in values):
            raise ValueError("raw resource quantities must not be negative")

    def to_dict(self) -> dict[str, int | None]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model_requests": self.model_requests,
            "duration_ms": self.duration_ms,
            "storage_bytes": self.storage_bytes,
        }


@dataclass(frozen=True, slots=True)
class WritebackPlan:
    """One atomic context and memory decision over stable source IDs."""

    plan_id: str
    context_action: PlanContextAction
    memory_action: PlanMemoryAction
    memory_kind: MemoryKind | None
    source_segment_ids: tuple[str, ...]
    base_revision: str
    policy_version: str
    evaluation_id: str
    provenance: ProvenanceRef
    idempotency_key: str
    summary: str
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        required = (
            self.plan_id,
            self.base_revision,
            self.policy_version,
            self.evaluation_id,
            self.idempotency_key,
            self.summary,
        )
        if any(not value.strip() for value in required):
            raise ValueError("writeback plan identifiers and summary must not be empty")
        object.__setattr__(self, "context_action", PlanContextAction(self.context_action))
        object.__setattr__(self, "memory_action", PlanMemoryAction(self.memory_action))
        if self.memory_kind is not None:
            object.__setattr__(self, "memory_kind", MemoryKind(self.memory_kind))
        if not self.source_segment_ids:
            raise ValueError("writeback plan requires source segment IDs")
        if len(self.source_segment_ids) != len(set(self.source_segment_ids)):
            raise ValueError("writeback source segment IDs must be unique")
        if self.memory_action in {PlanMemoryAction.ADD, PlanMemoryAction.UPDATE}:
            if self.memory_kind is None:
                raise ValueError("add/update plans require memory_kind")
        elif self.memory_kind is not None:
            raise ValueError("discard plans must not declare memory_kind")
        if self.provenance.segment_ids != self.source_segment_ids:
            raise ValueError("plan provenance must identify its source segments")
        if self.provenance.evaluation_id != self.evaluation_id:
            raise ValueError("plan provenance must identify its evaluation")
        if any(not _REASON_CODE.fullmatch(code) for code in self.reason_codes):
            raise ValueError("writeback reason codes must be machine-readable")


@dataclass(frozen=True, slots=True)
class PlanValidationResult:
    status: PlanValidationStatus
    reason_codes: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.status == PlanValidationStatus.VALID


@dataclass(frozen=True, slots=True)
class DryRunMutation:
    mutation_id: str
    plan_id: str
    action: PlanMemoryAction
    memory_kind: MemoryKind | None
    source_segment_ids: tuple[str, ...]
    provenance: ProvenanceRef


@dataclass(frozen=True, slots=True)
class DryRunReceipt:
    plan_id: str
    status: DryRunStatus
    validation: PlanValidationResult
    mutation_id: str | None = None


@dataclass(frozen=True, slots=True)
class WritebackEvent:
    """Observer event deliberately incapable of carrying source content."""

    kind: WritebackEventKind
    run_id: str
    episode_id: str
    session_id: str
    task_id: str
    snapshot_id: str
    evaluation_id: str
    plan_id: str | None
    mutation_id: str | None
    context_action: PlanContextAction | None
    memory_action: PlanMemoryAction | None
    memory_kind: MemoryKind | None
    source_segment_count: int
    status: str
    reason_codes: tuple[str, ...] = ()
    resources: RawResourceUsage = RawResourceUsage()

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "snapshot_id": self.snapshot_id,
            "evaluation_id": self.evaluation_id,
            "plan_id": self.plan_id,
            "mutation_id": self.mutation_id,
            "context_action": self.context_action.value if self.context_action else None,
            "memory_action": self.memory_action.value if self.memory_action else None,
            "memory_kind": self.memory_kind.value if self.memory_kind else None,
            "source_segment_count": self.source_segment_count,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "resources": self.resources.to_dict(),
        }


@runtime_checkable
class WritebackObserver(Protocol):
    def record(self, event: WritebackEvent) -> None: ...


class WritebackPlanValidator:
    """Deterministic safety checks shared by plan creation and execution."""

    @staticmethod
    def validate_evaluation(
        snapshot: ContextSnapshot,
        evaluation: ContextEvaluation,
    ) -> PlanValidationResult:
        reasons: list[str] = []
        if not evaluation.evaluation_id.strip():
            reasons.append("missing_evaluation_id")
        expected = {segment.segment_id for segment in snapshot.segments}
        actual = {signal.segment_id for signal in evaluation.signals}
        if len(evaluation.signals) != len(actual):
            reasons.append("duplicate_signal")
        if actual != expected:
            reasons.append("incomplete_evaluation")
        if any(not _REASON_CODE.fullmatch(code) for signal in evaluation.signals for code in signal.reason_codes):
            reasons.append("invalid_reason_code")

        protected = snapshot.protected_segment_ids
        if any(
            signal.segment_id in protected and signal.context_action == ContextAction.EVICT
            for signal in evaluation.signals
        ):
            reasons.append("protected_segment_eviction")
        if any(
            signal.context_action == ContextAction.EVICT
            and signal.writeback_action == WritebackAction.DEFER
            for signal in evaluation.signals
        ):
            reasons.append("eviction_without_memory_resolution")

        by_id = {signal.segment_id: signal for signal in evaluation.signals}
        for closure in snapshot.tool_closures:
            if not closure.closed:
                continue
            signals = [by_id.get(segment_id) for segment_id in closure.segment_ids]
            if any(signal is None for signal in signals):
                continue
            actions = {
                (signal.context_action, signal.writeback_action, signal.memory_kind)
                for signal in signals
                if signal is not None
            }
            if len(actions) != 1:
                reasons.append("split_tool_closure")

        if reasons:
            return PlanValidationResult(
                PlanValidationStatus.INVALID,
                tuple(dict.fromkeys(reasons)),
            )
        return PlanValidationResult(PlanValidationStatus.VALID)

    @staticmethod
    def validate_plan(
        plan: WritebackPlan,
        current_snapshot: ContextSnapshot,
    ) -> PlanValidationResult:
        if plan.base_revision != current_snapshot.context_revision:
            return PlanValidationResult(PlanValidationStatus.STALE, ("revision_mismatch",))
        identity = plan.provenance
        if (
            identity.run_id,
            identity.episode_id,
            identity.session_id,
            identity.task_id,
            identity.snapshot_id,
        ) != (
            current_snapshot.run_id,
            current_snapshot.episode_id,
            current_snapshot.session_id,
            current_snapshot.task_id,
            current_snapshot.snapshot_id,
        ):
            return PlanValidationResult(PlanValidationStatus.INVALID, ("provenance_mismatch",))

        source_ids = set(plan.source_segment_ids)
        snapshot_ids = {segment.segment_id for segment in current_snapshot.segments}
        if not source_ids.issubset(snapshot_ids):
            return PlanValidationResult(PlanValidationStatus.INVALID, ("unknown_source_segment",))
        if plan.context_action == PlanContextAction.EVICT:
            if source_ids.intersection(current_snapshot.protected_segment_ids):
                return PlanValidationResult(
                    PlanValidationStatus.INVALID,
                    ("protected_segment_eviction",),
                )
            for closure in current_snapshot.tool_closures:
                members = set(closure.segment_ids)
                if source_ids.intersection(members) and not members.issubset(source_ids):
                    return PlanValidationResult(
                        PlanValidationStatus.INVALID,
                        ("split_tool_closure",),
                    )
        return PlanValidationResult(PlanValidationStatus.VALID)


class WritebackCoordinator:
    """Create validated plans and simulate idempotent mutations without writes."""

    def __init__(
        self,
        *,
        validator: WritebackPlanValidator | None = None,
        observers: Iterable[WritebackObserver] = (),
    ) -> None:
        self.validator = validator or WritebackPlanValidator()
        self.observers = tuple(observers)
        self._mutations_by_key: dict[str, DryRunMutation] = {}

    @property
    def dry_run_mutations(self) -> tuple[DryRunMutation, ...]:
        return tuple(self._mutations_by_key.values())

    def _record(self, event: WritebackEvent) -> None:
        for observer in self.observers:
            observer.record(event)

    def _event(
        self,
        kind: WritebackEventKind,
        snapshot: ContextSnapshot,
        evaluation_id: str,
        *,
        plan: WritebackPlan | None = None,
        mutation_id: str | None = None,
        status: str,
        reason_codes: tuple[str, ...] = (),
        resources: RawResourceUsage = RawResourceUsage(),
    ) -> WritebackEvent:
        return WritebackEvent(
            kind=kind,
            run_id=snapshot.run_id,
            episode_id=snapshot.episode_id,
            session_id=snapshot.session_id,
            task_id=snapshot.task_id,
            snapshot_id=snapshot.snapshot_id,
            evaluation_id=evaluation_id,
            plan_id=plan.plan_id if plan else None,
            mutation_id=mutation_id,
            context_action=plan.context_action if plan else None,
            memory_action=plan.memory_action if plan else None,
            memory_kind=plan.memory_kind if plan else None,
            source_segment_count=len(plan.source_segment_ids) if plan else 0,
            status=status,
            reason_codes=reason_codes,
            resources=resources,
        )

    def create_plans(
        self,
        snapshot: ContextSnapshot,
        evaluation: ContextEvaluation,
        *,
        resources: RawResourceUsage = RawResourceUsage(),
    ) -> tuple[WritebackPlan, ...]:
        validation = self.validator.validate_evaluation(snapshot, evaluation)
        if not validation.valid:
            self._record(self._event(
                WritebackEventKind.PLAN_REJECTED,
                snapshot,
                evaluation.evaluation_id,
                status=validation.status.value,
                reason_codes=validation.reason_codes,
                resources=resources,
            ))
            return ()

        by_id = {signal.segment_id: signal for signal in evaluation.signals}
        groups: list[tuple[str, ...]] = []
        grouped: set[str] = set()
        for closure in snapshot.tool_closures:
            groups.append(closure.segment_ids)
            grouped.update(closure.segment_ids)
        groups.extend(
            (segment.segment_id,)
            for segment in snapshot.segments
            if segment.segment_id not in grouped
        )

        plans: list[WritebackPlan] = []
        for source_ids in groups:
            signal = by_id[source_ids[0]]
            if (
                signal.context_action == ContextAction.RETAIN
                and signal.writeback_action == WritebackAction.DEFER
            ):
                continue
            plan = self._build_plan(snapshot, evaluation, source_ids, signal)
            plan_validation = self.validator.validate_plan(plan, snapshot)
            if not plan_validation.valid:
                self._record(self._event(
                    WritebackEventKind.PLAN_REJECTED,
                    snapshot,
                    evaluation.evaluation_id,
                    plan=plan,
                    status=plan_validation.status.value,
                    reason_codes=plan_validation.reason_codes,
                    resources=resources,
                ))
                continue
            plans.append(plan)
            self._record(self._event(
                WritebackEventKind.PLAN_CREATED,
                snapshot,
                evaluation.evaluation_id,
                plan=plan,
                status="created",
                reason_codes=plan.reason_codes,
                resources=resources,
            ))
        return tuple(plans)

    @staticmethod
    def _build_plan(
        snapshot: ContextSnapshot,
        evaluation: ContextEvaluation,
        source_ids: tuple[str, ...],
        signal: EvaluationSignal,
    ) -> WritebackPlan:
        context_action = (
            PlanContextAction.EVICT
            if signal.context_action == ContextAction.EVICT
            else PlanContextAction.KEEP
        )
        memory_action = PlanMemoryAction(signal.writeback_action.value)
        memory_kind = signal.memory_kind if memory_action != PlanMemoryAction.DISCARD else None
        key_payload = {
            "source_segment_ids": source_ids,
            "policy_version": evaluation.policy_version,
            "context_action": context_action.value,
            "memory_action": memory_action.value,
            "memory_kind": memory_kind.value if memory_kind else None,
            "base_revision": snapshot.context_revision,
        }
        idempotency_key = _stable_hash("idem", key_payload, length=40)
        plan_id = _stable_hash(
            "plan",
            {"evaluation_id": evaluation.evaluation_id, "idempotency_key": idempotency_key},
        )
        provenance = replace(
            snapshot.provenance,
            segment_ids=source_ids,
            evaluation_id=evaluation.evaluation_id,
        )
        summary = ":".join((
            context_action.value,
            memory_action.value,
            memory_kind.value if memory_kind else "none",
        ))
        return WritebackPlan(
            plan_id=plan_id,
            context_action=context_action,
            memory_action=memory_action,
            memory_kind=memory_kind,
            source_segment_ids=source_ids,
            base_revision=snapshot.context_revision,
            policy_version=evaluation.policy_version,
            evaluation_id=evaluation.evaluation_id,
            provenance=provenance,
            idempotency_key=idempotency_key,
            summary=summary,
            reason_codes=signal.reason_codes,
        )

    def dry_run(
        self,
        plan: WritebackPlan,
        current_snapshot: ContextSnapshot,
    ) -> DryRunReceipt:
        validation = self.validator.validate_plan(plan, current_snapshot)
        self._record(self._event(
            WritebackEventKind.PLAN_VALIDATED,
            current_snapshot,
            plan.evaluation_id,
            plan=plan,
            status=validation.status.value,
            reason_codes=validation.reason_codes,
        ))
        if validation.status == PlanValidationStatus.STALE:
            return DryRunReceipt(plan.plan_id, DryRunStatus.STALE, validation)
        if not validation.valid:
            return DryRunReceipt(plan.plan_id, DryRunStatus.REJECTED, validation)

        existing = self._mutations_by_key.get(plan.idempotency_key)
        if existing is not None:
            self._record(self._event(
                WritebackEventKind.DRY_RUN_DUPLICATE,
                current_snapshot,
                plan.evaluation_id,
                plan=plan,
                mutation_id=existing.mutation_id,
                status=DryRunStatus.DUPLICATE.value,
                reason_codes=("idempotent_replay",),
            ))
            return DryRunReceipt(
                plan.plan_id,
                DryRunStatus.DUPLICATE,
                validation,
                existing.mutation_id,
            )

        mutation_id = _stable_hash("mutation", {"idempotency_key": plan.idempotency_key})
        mutation = DryRunMutation(
            mutation_id=mutation_id,
            plan_id=plan.plan_id,
            action=plan.memory_action,
            memory_kind=plan.memory_kind,
            source_segment_ids=plan.source_segment_ids,
            provenance=replace(plan.provenance, mutation_id=mutation_id),
        )
        self._mutations_by_key[plan.idempotency_key] = mutation
        self._record(self._event(
            WritebackEventKind.DRY_RUN_MUTATION,
            current_snapshot,
            plan.evaluation_id,
            plan=plan,
            mutation_id=mutation_id,
            status=DryRunStatus.ACCEPTED.value,
        ))
        return DryRunReceipt(
            plan.plan_id,
            DryRunStatus.ACCEPTED,
            validation,
            mutation_id,
        )
