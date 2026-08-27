"""Validated, content-free writeback plans and dry-run coordination."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol, runtime_checkable

from ..memory.contracts import MemoryKind
from ..memory.runtime import MemoryBackendRegistry
from .contracts import (
    CompletionStatus,
    ContextAction,
    ContextEvaluation,
    EvaluationSignal,
    MemoryScope,
    TemporalValidity,
    WritebackAction,
    LIFECYCLE_CONTRACT_SCHEMA_VERSION,
    _require_current_schema,
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
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    model_requests: int = 0
    retry_count: int = 0
    duration_ms: int | None = None
    storage_bytes: int = 0
    schema_version: int = LIFECYCLE_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_current_schema(self.schema_version, "raw resource usage")
        values = (
            self.input_tokens,
            self.output_tokens,
            self.cache_read_tokens,
            self.cache_write_tokens,
            self.reasoning_tokens,
            self.model_requests,
            self.retry_count,
            self.duration_ms,
            self.storage_bytes,
        )
        if any(value is not None and value < 0 for value in values):
            raise ValueError("raw resource quantities must not be negative")

    def to_dict(self) -> dict[str, int | None]:
        return {
            "schema_version": self.schema_version,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "model_requests": self.model_requests,
            "retry_count": self.retry_count,
            "duration_ms": self.duration_ms,
            "storage_bytes": self.storage_bytes,
        }


@dataclass(frozen=True, slots=True)
class ExitEvidence:
    """Resolved analysis carried from context evaluation into compilation."""

    completion_status: CompletionStatus
    completion_evidence: tuple[str, ...]
    safe_to_evict: bool
    unresolved_state: str | None
    scope: MemoryScope | None
    temporal_validity: TemporalValidity | None
    provenance: tuple[str, ...]
    reusable_facts: tuple[str, ...]
    reusable_procedures: tuple[str, ...]
    update_hints: tuple[str, ...]
    utility_estimate: float = 0.0
    confidence: float = 0.0
    schema_version: int = LIFECYCLE_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_current_schema(self.schema_version, "exit evidence")
        if type(self.safe_to_evict) is not bool:
            raise TypeError("safe_to_evict must be bool")
        object.__setattr__(self, "completion_status", CompletionStatus(self.completion_status))
        if self.scope is not None:
            object.__setattr__(self, "scope", MemoryScope(self.scope))
        if self.temporal_validity is not None:
            object.__setattr__(self, "temporal_validity", TemporalValidity(self.temporal_validity))
        if self.unresolved_state is not None and not self.unresolved_state.strip():
            raise ValueError("unresolved_state must be non-empty when present")
        if not 0.0 <= self.utility_estimate <= 1.0:
            raise ValueError("exit evidence utility_estimate must be in [0,1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("exit evidence confidence must be in [0,1]")
        if not self.provenance:
            raise ValueError("exit evidence requires deterministic provenance")
        sequences = (
            self.completion_evidence,
            self.provenance,
            self.reusable_facts,
            self.reusable_procedures,
            self.update_hints,
        )
        if any(not value.strip() for values in sequences for value in values):
            raise ValueError("exit evidence values must not be empty")

    def compiler_input_payload(self) -> dict[str, object]:
        """Canonical evidence that can change compiler-produced memory content."""

        return {
            "schema_version": self.schema_version,
            "completion_status": self.completion_status.value,
            "completion_evidence": self.completion_evidence,
            "unresolved_state": self.unresolved_state,
            "scope": self.scope.value if self.scope is not None else None,
            "temporal_validity": (
                self.temporal_validity.value
                if self.temporal_validity is not None
                else None
            ),
            "reusable_facts": self.reusable_facts,
            "reusable_procedures": self.reusable_procedures,
            "update_hints": self.update_hints,
            "utility_estimate": self.utility_estimate,
            "confidence": self.confidence,
        }

@dataclass(frozen=True, slots=True)
class UpdateTarget:
    backend: str
    artifact_id: str
    expected_revision: str
    memory_kind: MemoryKind

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (self.backend, self.artifact_id, self.expected_revision)
        ):
            raise ValueError("update target identifiers and revision must not be empty")
        object.__setattr__(self, "memory_kind", MemoryKind(self.memory_kind))


@runtime_checkable
class UpdateTargetResolver(Protocol):
    def resolve(
        self,
        snapshot: ContextSnapshot,
        signal: EvaluationSignal,
    ) -> UpdateTarget | None: ...


class AllowlistedUpdateTargetResolver:
    """Bind an update suggestion to an artifact owned by the selected backend."""

    def __init__(
        self,
        registry: MemoryBackendRegistry,
        search: Callable[[ContextSnapshot, EvaluationSignal], Iterable[str]],
        *,
        allowed_backends: Mapping[str, frozenset[MemoryKind]],
    ) -> None:
        self.registry = registry
        self.search = search
        self.allowed_backends = {
            backend: frozenset(MemoryKind(kind) for kind in kinds)
            for backend, kinds in allowed_backends.items()
        }

    def resolve(
        self,
        snapshot: ContextSnapshot,
        signal: EvaluationSignal,
    ) -> UpdateTarget | None:
        if signal.memory_kind is None:
            return None
        try:
            backend = self.registry.resolve(signal.memory_kind)
        except KeyError:
            return None
        descriptor = backend.descriptor
        capability = descriptor.capability_for(signal.memory_kind)
        if capability is None or not capability.updatable:
            return None
        if signal.memory_kind not in self.allowed_backends.get(
            descriptor.name,
            frozenset(),
        ):
            return None

        artifact_ids = tuple(dict.fromkeys(
            artifact_id
            for artifact_id in self.search(snapshot, signal)
            if isinstance(artifact_id, str) and artifact_id.strip()
        ))
        resolved = []
        for artifact_id in artifact_ids:
            artifact = backend.get(artifact_id)
            if artifact is None:
                continue
            if artifact.artifact_id != artifact_id or artifact.kind != signal.memory_kind:
                continue
            if not isinstance(artifact.revision, str) or not artifact.revision.strip():
                continue
            resolved.append(UpdateTarget(
                backend=descriptor.name,
                artifact_id=artifact.artifact_id,
                expected_revision=artifact.revision,
                memory_kind=artifact.kind,
            ))
        return resolved[0] if len(resolved) == 1 else None


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
    exit_evidence: ExitEvidence
    target_backend: str | None = None
    target_artifact_id: str | None = None
    expected_memory_revision: str | None = None
    update_mode: str | None = None
    compiler_version: str = "uncompiled-v0"
    reason_codes: tuple[str, ...] = ()
    schema_version: int = LIFECYCLE_CONTRACT_SCHEMA_VERSION

    @property
    def update_hints(self) -> tuple[str, ...]:
        """Expose the complete compiler hint tuple without duplicating storage."""

        return self.exit_evidence.update_hints

    def __post_init__(self) -> None:
        _require_current_schema(self.schema_version, "writeback plan")
        if (
            self.provenance.schema_version != self.schema_version
            or self.exit_evidence.schema_version != self.schema_version
        ):
            raise ValueError("plan contract schema versions must match")
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
            if not self.compiler_version.strip():
                raise ValueError("add/update plans require compiler_version")
        elif self.memory_kind is not None:
            raise ValueError("discard plans must not declare memory_kind")
        optional_values = (
            self.target_backend,
            self.target_artifact_id,
            self.expected_memory_revision,
            self.update_mode,
        )
        if any(value is not None and not value.strip() for value in optional_values):
            raise ValueError("optional writeback target fields must be non-empty")
        if self.update_mode is not None and not _REASON_CODE.fullmatch(self.update_mode):
            raise ValueError("writeback update_mode must be machine-readable")
        has_existing_target = any((
            self.target_artifact_id,
            self.expected_memory_revision,
            self.update_mode,
        ))
        if self.memory_action == PlanMemoryAction.ADD and any((
            self.target_backend,
            has_existing_target,
        )):
            raise ValueError("add plans must not carry an existing memory target")
        if self.memory_action == PlanMemoryAction.UPDATE:
            if not all((
                self.target_backend,
                self.target_artifact_id,
                self.expected_memory_revision,
            )):
                raise ValueError("update plans require backend, artifact, and revision")
            if not self.exit_evidence.update_hints and not self.update_mode:
                raise ValueError("update plans require update_hint or update_mode")
        if self.memory_action == PlanMemoryAction.DISCARD and any(optional_values):
            raise ValueError("discard plans must not carry a memory target")
        if self.provenance.segment_ids != self.source_segment_ids:
            raise ValueError("plan provenance must identify its source segments")
        if self.provenance.evaluation_id != self.evaluation_id:
            raise ValueError("plan provenance must identify its evaluation")
        if self.context_action == PlanContextAction.EVICT:
            if not self.exit_evidence.safe_to_evict:
                raise ValueError("eviction plans require resolved safe_to_evict=true")
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
    target_backend: str | None = None
    target_artifact_id: str | None = None
    expected_memory_revision: str | None = None


@dataclass(frozen=True, slots=True)
class DryRunReceipt:
    plan_id: str
    status: DryRunStatus
    validation: PlanValidationResult
    mutation_id: str | None = None


@dataclass(frozen=True, slots=True)
class IdempotencyReceipt:
    idempotency_key: str
    plan_id: str
    mutation_id: str

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (self.idempotency_key, self.plan_id, self.mutation_id)
        ):
            raise ValueError("idempotency receipt identifiers must not be empty")


@runtime_checkable
class IdempotencyReceiptStore(Protocol):
    def get(self, idempotency_key: str) -> IdempotencyReceipt | None: ...

    def reserve_if_absent(
        self,
        receipt: IdempotencyReceipt,
    ) -> tuple[IdempotencyReceipt, bool]: ...


class InMemoryIdempotencyReceiptStore:
    def __init__(self) -> None:
        self._receipts: dict[str, IdempotencyReceipt] = {}
        self._lock = threading.Lock()

    def get(self, idempotency_key: str) -> IdempotencyReceipt | None:
        with self._lock:
            return self._receipts.get(idempotency_key)

    def reserve_if_absent(
        self,
        receipt: IdempotencyReceipt,
    ) -> tuple[IdempotencyReceipt, bool]:
        with self._lock:
            existing = self._receipts.get(receipt.idempotency_key)
            if existing is not None:
                if existing != receipt:
                    raise ValueError("idempotency key already has a different receipt")
                return existing, False
            self._receipts[receipt.idempotency_key] = receipt
            return receipt, True


class JsonIdempotencyReceiptStore:
    """Small persistent receipt store for replay-safe coordinator execution."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()

    @contextmanager
    def _lock(self, operation: int):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("w", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _read(self) -> dict[str, IdempotencyReceipt]:
        if not self.path.exists():
            return {}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("idempotency receipt file must contain an object")
        receipts: dict[str, IdempotencyReceipt] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("idempotency receipt keys must be non-empty strings")
            if not isinstance(item, dict):
                raise ValueError(f"malformed idempotency receipt: {key}")
            if set(item) != {"plan_id", "mutation_id"}:
                raise ValueError(f"malformed idempotency receipt: {key}")
            if not all(isinstance(item[field], str) for field in item):
                raise ValueError(f"malformed idempotency receipt: {key}")
            receipts[key] = IdempotencyReceipt(
                idempotency_key=key,
                plan_id=str(item["plan_id"]),
                mutation_id=str(item["mutation_id"]),
            )
        return receipts

    def get(self, idempotency_key: str) -> IdempotencyReceipt | None:
        with self._lock(fcntl.LOCK_SH):
            return self._read().get(idempotency_key)

    def reserve_if_absent(
        self,
        receipt: IdempotencyReceipt,
    ) -> tuple[IdempotencyReceipt, bool]:
        with self._lock(fcntl.LOCK_EX):
            receipts = self._read()
            existing = receipts.get(receipt.idempotency_key)
            if existing is not None:
                if existing != receipt:
                    raise ValueError("idempotency key already has a different receipt")
                return existing, False
            receipts[receipt.idempotency_key] = receipt
            payload = {
                key: {
                    "plan_id": value.plan_id,
                    "mutation_id": value.mutation_id,
                }
                for key, value in sorted(receipts.items())
            }
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                dir=self.path.parent,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=True, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_name, self.path)
            except BaseException:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass
                raise
            return receipt, True


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
    target_backend: str | None
    target_artifact_id: str | None
    compiler_version: str | None
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
            "target_backend": self.target_backend,
            "target_artifact_id": self.target_artifact_id,
            "compiler_version": self.compiler_version,
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

    def __init__(
        self,
        *,
        updatable_backends: Mapping[str, frozenset[MemoryKind]] | None = None,
    ) -> None:
        self.updatable_backends = {
            backend: frozenset(MemoryKind(kind) for kind in kinds)
            for backend, kinds in (updatable_backends or {}).items()
        }

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
        if any(
            not _REASON_CODE.fullmatch(hint)
            for signal in evaluation.signals
            for hint in signal.update_hints
        ) or any(
            signal.update_mode is not None
            and not _REASON_CODE.fullmatch(signal.update_mode)
            for signal in evaluation.signals
        ):
            reasons.append("invalid_update_hint")

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
                (
                    signal.context_action,
                    signal.writeback_action,
                    signal.memory_kind,
                    signal.target_backend,
                    signal.target_artifact_id,
                    signal.expected_memory_revision,
                    signal.update_mode,
                    signal.update_hints,
                    signal.compiler_version,
                )
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

    def validate_plan(
        self,
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
        if plan.memory_action == PlanMemoryAction.UPDATE:
            supported_kinds = self.updatable_backends.get(plan.target_backend or "", frozenset())
            if plan.memory_kind not in supported_kinds:
                return PlanValidationResult(
                    PlanValidationStatus.INVALID,
                    ("backend_update_not_supported",),
                )
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
        target_resolver: UpdateTargetResolver | None = None,
        receipt_store: IdempotencyReceiptStore | None = None,
        observers: Iterable[WritebackObserver] = (),
    ) -> None:
        self.validator = validator or WritebackPlanValidator()
        self.target_resolver = target_resolver
        self.receipt_store = receipt_store or InMemoryIdempotencyReceiptStore()
        self.observers = tuple(observers)
        self._dry_run_mutations: dict[str, DryRunMutation] = {}

    @property
    def dry_run_mutations(self) -> tuple[DryRunMutation, ...]:
        return tuple(self._dry_run_mutations.values())

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
            target_backend=plan.target_backend if plan else None,
            target_artifact_id=plan.target_artifact_id if plan else None,
            compiler_version=plan.compiler_version if plan else None,
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
            signals = tuple(by_id[source_id] for source_id in source_ids)
            signal = signals[0]
            if (
                signal.context_action == ContextAction.RETAIN
                and signal.writeback_action == WritebackAction.DEFER
            ):
                continue
            target: UpdateTarget | None = None
            if signal.writeback_action == WritebackAction.UPDATE:
                targets = tuple(
                    self.target_resolver.resolve(snapshot, item)
                    for item in signals
                ) if self.target_resolver is not None else ()
                if not targets or any(item is None for item in targets):
                    self._record(self._event(
                        WritebackEventKind.PLAN_REJECTED,
                        snapshot,
                        evaluation.evaluation_id,
                        status=PlanValidationStatus.INVALID.value,
                        reason_codes=("update_target_unresolved",),
                        resources=resources,
                    ))
                    continue
                resolved_targets = {item for item in targets if item is not None}
                if len(resolved_targets) != 1:
                    self._record(self._event(
                        WritebackEventKind.PLAN_REJECTED,
                        snapshot,
                        evaluation.evaluation_id,
                        status=PlanValidationStatus.INVALID.value,
                        reason_codes=("split_tool_update_target",),
                        resources=resources,
                    ))
                    continue
                target = resolved_targets.pop()
            plan = self._build_plan(
                snapshot,
                evaluation,
                source_ids,
                signals,
                target=target,
            )
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
        signals: tuple[EvaluationSignal, ...],
        *,
        target: UpdateTarget | None,
    ) -> WritebackPlan:
        signal = signals[0]

        def combined(values: Iterable[tuple[str, ...]]) -> tuple[str, ...]:
            return tuple(dict.fromkeys(value for group in values for value in group))

        context_action = (
            PlanContextAction.EVICT
            if signal.context_action == ContextAction.EVICT
            else PlanContextAction.KEEP
        )
        memory_action = PlanMemoryAction(signal.writeback_action.value)
        memory_kind = signal.memory_kind if memory_action != PlanMemoryAction.DISCARD else None
        protected = snapshot.protected_segment_ids
        safe_to_evict = not bool(set(source_ids).intersection(protected))
        source_segments = {
            segment.segment_id: segment for segment in snapshot.segments
        }
        resolved_unresolved_state = (
            "host_unresolved"
            if any(not source_segments[source_id].completed for source_id in source_ids)
            else None
        )
        completion_statuses = {item.completion_status for item in signals}
        scopes = {item.scope for item in signals}
        temporal_validities = {item.temporal_validity for item in signals}
        exit_evidence = ExitEvidence(
            completion_status=(
                completion_statuses.pop()
                if len(completion_statuses) == 1
                else CompletionStatus.UNKNOWN
            ),
            completion_evidence=combined(
                item.completion_evidence for item in signals
            ),
            safe_to_evict=safe_to_evict,
            unresolved_state=resolved_unresolved_state,
            scope=scopes.pop() if len(scopes) == 1 else None,
            temporal_validity=(
                temporal_validities.pop()
                if len(temporal_validities) == 1
                else None
            ),
            provenance=(
                snapshot.run_id,
                snapshot.episode_id,
                snapshot.session_id,
                snapshot.task_id,
                snapshot.snapshot_id,
                evaluation.evaluation_id,
                *source_ids,
            ),
            reusable_facts=combined(item.reusable_facts for item in signals),
            reusable_procedures=combined(
                item.reusable_procedures for item in signals
            ),
            update_hints=combined(item.update_hints for item in signals),
            utility_estimate=min(item.utility_estimate for item in signals),
            confidence=min(item.confidence for item in signals),
        )
        key_payload = {
            "schema_version": LIFECYCLE_CONTRACT_SCHEMA_VERSION,
            "source_segment_ids": source_ids,
            "policy_version": evaluation.policy_version,
            "context_action": context_action.value,
            "memory_action": memory_action.value,
            "memory_kind": memory_kind.value if memory_kind else None,
            "base_revision": snapshot.context_revision,
            "target_backend": target.backend if target else None,
            "target_artifact_id": target.artifact_id if target else None,
            "expected_memory_revision": target.expected_revision if target else None,
            "compiler_input": exit_evidence.compiler_input_payload(),
            "update_mode": signal.update_mode,
            "compiler_version": signal.compiler_version,
        }
        idempotency_key = _stable_hash("idem", key_payload, length=40)
        plan_id = _stable_hash("plan", {"idempotency_key": idempotency_key})
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
            exit_evidence=exit_evidence,
            target_backend=target.backend if target else None,
            target_artifact_id=target.artifact_id if target else None,
            expected_memory_revision=target.expected_revision if target else None,
            update_mode=signal.update_mode,
            compiler_version=signal.compiler_version,
            reason_codes=combined(item.reason_codes for item in signals),
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

        mutation_id = _stable_hash("mutation", {"idempotency_key": plan.idempotency_key})
        receipt, reserved = self.receipt_store.reserve_if_absent(IdempotencyReceipt(
            idempotency_key=plan.idempotency_key,
            plan_id=plan.plan_id,
            mutation_id=mutation_id,
        ))
        if not reserved:
            self._record(self._event(
                WritebackEventKind.DRY_RUN_DUPLICATE,
                current_snapshot,
                plan.evaluation_id,
                plan=plan,
                mutation_id=receipt.mutation_id,
                status=DryRunStatus.DUPLICATE.value,
                reason_codes=("idempotent_replay",),
            ))
            return DryRunReceipt(
                plan.plan_id,
                DryRunStatus.DUPLICATE,
                validation,
                receipt.mutation_id,
            )

        mutation = DryRunMutation(
            mutation_id=mutation_id,
            plan_id=plan.plan_id,
            action=plan.memory_action,
            memory_kind=plan.memory_kind,
            source_segment_ids=plan.source_segment_ids,
            provenance=replace(plan.provenance, mutation_id=mutation_id),
            target_backend=plan.target_backend,
            target_artifact_id=plan.target_artifact_id,
            expected_memory_revision=plan.expected_memory_revision,
        )
        self._dry_run_mutations[plan.idempotency_key] = mutation
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
