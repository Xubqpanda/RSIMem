"""Transactional semantic mutation execution, verification, and recovery."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Mapping

from ..lifecycle import EvaluationTrigger
from .contracts import (
    MemoryArtifact,
    MemoryKind,
    MemoryMutation,
    MemoryMutationAction,
    MemoryQuery,
)
from .ingestion import InternalMemoryAction, MemoryIngestResult
from .receipts import (
    JsonMutationReceiptStore,
    MutationReceipt,
    MutationReceiptPhase,
    MutationReceiptStatus,
    SemanticMutationWriter,
)
from .runtime import MemoryBackendRegistry
from .operation_graph import (
    ArtifactKind,
    ArtifactNode,
    AtomicOperationRecorder,
    MutationEdge,
    OperationContext,
    OperationKind,
    OperationSpec,
    OperationStatus,
    build_artifact_id,
    build_operation_id,
)
from .validation import (
    MutationValidator,
    TrustedValidationContext,
    UntrustedMemoryCandidate,
    ValidationResult,
    fingerprint_memory_candidate,
)


MUTATION_EXECUTOR_SCHEMA_VERSION = 1
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha(value: str | bytes) -> str:
    encoded = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_id(prefix: str, value: object) -> str:
    return f"{prefix}.{_sha(_canonical_json(value))[:40]}"


class CrashPoint(StrEnum):
    AFTER_RESERVE = "after_reserve"
    BEFORE_BACKEND_CALL = "before_backend_call"
    AFTER_BACKEND_WRITE = "after_backend_write"
    BEFORE_VERIFICATION = "before_verification"
    BEFORE_RECEIPT_COMMIT = "before_receipt_commit"


class InjectedMutationCrash(BaseException):
    def __init__(self, point: CrashPoint) -> None:
        self.point = CrashPoint(point)
        super().__init__(self.point.value)


class MutationExecutionStatus(StrEnum):
    DISABLED = "disabled"
    COMMITTED = "committed"
    DUPLICATE = "duplicate"
    FAILED = "failed"
    BLOCKED = "blocked"
    ROLLED_BACK = "rolled_back"


class RecoveryMode(StrEnum):
    CONTINUE = "continue"
    ROLLBACK_IF_SAFE = "rollback_if_safe"


class ProbeState(StrEnum):
    PRE_STATE = "pre_state"
    DESIRED_STATE = "desired_state"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class MutationExecutionRequest:
    candidate: UntrustedMemoryCandidate
    ingest_result: MemoryIngestResult
    trusted_context: TrustedValidationContext
    current_source_digest: str
    trigger: EvaluationTrigger
    evidence_input_artifact_ids: tuple[str, ...] = ()
    evidence_proposal_operation_ids: tuple[str, ...] = ()
    schema_version: int = MUTATION_EXECUTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MUTATION_EXECUTOR_SCHEMA_VERSION:
            raise ValueError("unsupported mutation execution request schema version")
        object.__setattr__(self, "trigger", EvaluationTrigger(self.trigger))
        if not _DIGEST.fullmatch(self.current_source_digest):
            raise ValueError("mutation execution current_source_digest must be sha256")
        for name, values in (
            ("input artifact", self.evidence_input_artifact_ids),
            ("proposal operation", self.evidence_proposal_operation_ids),
        ):
            if len(values) != len(set(values)) or any(
                not _IDENTIFIER.fullmatch(value) for value in values
            ):
                raise ValueError(f"mutation evidence {name} IDs must be unique identifiers")


@dataclass(frozen=True, slots=True)
class ContextExitReport:
    natural_exit: bool
    logical_exit: bool
    physical_rewrite: bool
    source_retained: bool
    reason_code: str
    saved_tokens: int | None = None

    def __post_init__(self) -> None:
        values = (
            self.natural_exit,
            self.logical_exit,
            self.physical_rewrite,
            self.source_retained,
        )
        if any(type(value) is not bool for value in values):
            raise TypeError("context exit report flags must be bool")
        if not _REASON_CODE.fullmatch(self.reason_code):
            raise ValueError("context exit reason_code must be machine-readable")
        if self.physical_rewrite:
            raise ValueError("physical context rewrite is disabled")
        if self.saved_tokens is not None:
            raise ValueError("disabled physical rewrite cannot report saved tokens")
        if self.logical_exit and self.source_retained:
            raise ValueError("logical exit cannot retain the same source")


@dataclass(frozen=True, slots=True)
class MutationExecutionResult:
    status: MutationExecutionStatus
    mutation_id: str
    receipt_id: str | None
    reason_code: str
    validation: ValidationResult | None
    artifact_id: str | None
    revision: str | None
    storage_bytes: int
    context_exit: ContextExitReport
    writer_identity: SemanticMutationWriter | None = None
    schema_version: int = MUTATION_EXECUTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MUTATION_EXECUTOR_SCHEMA_VERSION:
            raise ValueError("unsupported mutation execution result schema version")
        object.__setattr__(self, "status", MutationExecutionStatus(self.status))
        if self.writer_identity is not None:
            object.__setattr__(
                self,
                "writer_identity",
                SemanticMutationWriter(self.writer_identity),
            )
        if not self.mutation_id.strip():
            raise ValueError("mutation execution result requires mutation_id")
        if self.receipt_id is not None and not self.receipt_id.strip():
            raise ValueError("mutation execution receipt_id must be non-empty")
        if not _REASON_CODE.fullmatch(self.reason_code):
            raise ValueError("mutation execution reason_code must be machine-readable")
        if self.storage_bytes < 0:
            raise ValueError("mutation execution storage_bytes must not be negative")

    def observer_evidence(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "mutation_id": self.mutation_id,
            "receipt_id": self.receipt_id,
            "reason_code": self.reason_code,
            "validation_status": (
                self.validation.status.value if self.validation is not None else None
            ),
            "artifact_id": self.artifact_id,
            "revision": self.revision,
            "storage_bytes": self.storage_bytes,
            "writer_identity": (
                self.writer_identity.value if self.writer_identity is not None else None
            ),
            "context_exit": {
                "natural_exit": self.context_exit.natural_exit,
                "logical_exit": self.context_exit.logical_exit,
                "physical_rewrite": self.context_exit.physical_rewrite,
                "source_retained": self.context_exit.source_retained,
                "reason_code": self.context_exit.reason_code,
                "saved_tokens": self.context_exit.saved_tokens,
            },
        }


@dataclass(frozen=True, slots=True)
class ProbeResult:
    state: ProbeState
    artifact: MemoryArtifact | None = None


class ContextExitGate:
    _NATURAL_TRIGGERS = {
        EvaluationTrigger.TASK_COMPLETED,
        EvaluationTrigger.SESSION_END,
    }

    def evaluate(
        self,
        trigger: EvaluationTrigger,
        receipt: MutationReceipt | None,
        *,
        failure_reason: str | None = None,
    ) -> ContextExitReport:
        trigger = EvaluationTrigger(trigger)
        natural = trigger in self._NATURAL_TRIGGERS
        logical = bool(
            natural
            and receipt is not None
            and receipt.status == MutationReceiptStatus.COMMITTED
            and receipt.verified
            and receipt.action != InternalMemoryAction.NONE
        )
        if not natural:
            reason = "non_natural_boundary"
        elif logical:
            reason = "verified_memory_committed"
        elif failure_reason is not None and _REASON_CODE.fullmatch(failure_reason):
            reason = failure_reason
        elif receipt is not None and receipt.action == InternalMemoryAction.NONE:
            reason = "no_memory_commit"
        else:
            reason = "memory_not_committed"
        return ContextExitReport(
            natural_exit=natural,
            logical_exit=logical,
            physical_rewrite=False,
            source_retained=not logical,
            reason_code=reason,
        )


class TransactionalMutationExecutor:
    """Execute isolated semantic mutations behind validation and durable receipts."""

    def __init__(
        self,
        registry: MemoryBackendRegistry,
        validator: MutationValidator,
        receipt_store: JsonMutationReceiptStore,
        *,
        enabled: bool = False,
        isolated_fixture: bool = False,
        exit_gate: ContextExitGate | None = None,
        operation_recorder: AtomicOperationRecorder | None = None,
    ) -> None:
        if enabled and not isolated_fixture:
            raise ValueError("real mutation requires explicit isolated_fixture=true")
        self.registry = registry
        self.validator = validator
        self.receipt_store = receipt_store
        self.enabled = enabled
        self.isolated_fixture = isolated_fixture
        self.exit_gate = exit_gate or ContextExitGate()
        self.operation_recorder = operation_recorder
        if self.validator.target_resolver is None:
            self.validator.target_resolver = receipt_store

    def execute(
        self,
        request: MutationExecutionRequest,
        *,
        crash_at: CrashPoint | None = None,
    ) -> MutationExecutionResult:
        candidate_digest = fingerprint_memory_candidate(request.candidate)
        identity = self._mutation_identity(request, candidate_digest)
        mutation_id = _stable_id("mutation", identity)
        if not self.enabled:
            return self._unpersisted_result(
                MutationExecutionStatus.DISABLED,
                mutation_id,
                request,
                reason="mutation_disabled",
            )
        if request.trigger not in ContextExitGate._NATURAL_TRIGGERS:
            return self._unpersisted_result(
                MutationExecutionStatus.FAILED,
                mutation_id,
                request,
                reason="non_natural_boundary",
            )

        idempotency_key = _stable_id("mutation-key", identity)
        existing = self.receipt_store.get(idempotency_key)
        if existing is not None:
            expected = self._build_receipt(
                request,
                candidate_digest,
                identity=identity,
                pre_artifact_ids=existing.pre_artifact_ids,
            )
            self._require_same_core(existing, expected)
            return self._existing_result(existing, request)

        context = self._operation_context(request)
        validation_spec = self._operation_spec(
            OperationKind.VALIDATION,
            request,
            context,
            "validation",
            parent_ids=(request.trusted_context.provenance.operation_id,),
            input_ids=request.evidence_input_artifact_ids,
        )
        validation_artifact_id = None
        if self.operation_recorder is None:
            validation = self.validator.validate(
                request.candidate,
                request.ingest_result,
                current_source_digest=request.current_source_digest,
                trusted_context=request.trusted_context,
            )
        else:
            with self.operation_recorder.operation_scope(validation_spec) as scope:
                validation = self.validator.validate(
                    request.candidate,
                    request.ingest_result,
                    current_source_digest=request.current_source_digest,
                    trusted_context=request.trusted_context,
                )
                validation_artifact = self._validation_artifact(
                    request,
                    context,
                    validation,
                )
                self.operation_recorder.record_artifact(validation_artifact)
                validation_artifact_id = validation_artifact.artifact_id
                scope.complete(
                    output_artifact_ids=(validation_artifact_id,),
                    status=(
                        OperationStatus.SUCCESS
                        if validation.accepted
                        else OperationStatus.REJECTED
                    ),
                    reason_code=(
                        None if validation.accepted else "validation_rejected"
                    ),
                )
        if not validation.accepted:
            return self._unpersisted_result(
                MutationExecutionStatus.FAILED,
                mutation_id,
                request,
                reason="validation_rejected",
                validation=validation,
            )
        if validation.candidate_digest != candidate_digest:
            return self._unpersisted_result(
                MutationExecutionStatus.FAILED,
                mutation_id,
                request,
                reason="validation_digest_mismatch",
                validation=validation,
            )

        expected = self._build_receipt(request, candidate_digest, identity=identity)
        receipt, reserved = self.receipt_store.reserve_pending(expected)
        if not reserved:
            self._require_same_core(receipt, expected)
            return self._existing_result(receipt, request, validation=validation)
        self._inject(crash_at, CrashPoint.AFTER_RESERVE)
        return self._apply_reserved(
            receipt,
            request,
            validation,
            crash_at=crash_at,
            parent_operation_ids=(validation_spec.operation_id,),
            validation_artifact_id=validation_artifact_id,
        )

    def recover(
        self,
        request: MutationExecutionRequest,
        *,
        mode: RecoveryMode = RecoveryMode.CONTINUE,
    ) -> MutationExecutionResult:
        if not self.enabled:
            candidate_digest = fingerprint_memory_candidate(request.candidate)
            identity = self._mutation_identity(request, candidate_digest)
            return self._unpersisted_result(
                MutationExecutionStatus.DISABLED,
                _stable_id("mutation", identity),
                request,
                reason="mutation_disabled",
            )
        mode = RecoveryMode(mode)
        candidate_digest = fingerprint_memory_candidate(request.candidate)
        identity = self._mutation_identity(request, candidate_digest)
        idempotency_key = _stable_id("mutation-key", identity)
        receipt = self.receipt_store.get(idempotency_key)
        if receipt is None:
            expected = self._build_receipt(request, candidate_digest, identity=identity)
            return self._result(
                MutationExecutionStatus.FAILED,
                expected,
                request,
                reason="receipt_not_found",
            )
        expected = self._build_receipt(
            request,
            candidate_digest,
            identity=identity,
            pre_artifact_ids=receipt.pre_artifact_ids,
        )
        self._require_same_core(receipt, expected)
        if receipt.status != MutationReceiptStatus.PENDING:
            return self._existing_result(receipt, request)
        if receipt.target_blocked:
            return self._result(
                MutationExecutionStatus.BLOCKED,
                receipt,
                request,
                reason=receipt.reason_code or "target_state_unknown",
            )

        if receipt.phase == MutationReceiptPhase.RESERVED:
            if mode == RecoveryMode.ROLLBACK_IF_SAFE:
                rolled = self._terminal(
                    receipt,
                    MutationReceiptStatus.ROLLED_BACK,
                    "reserved_mutation_rolled_back",
                )
                return self._result(
                    MutationExecutionStatus.ROLLED_BACK,
                    rolled,
                    request,
                    reason="reserved_mutation_rolled_back",
                )
            return self._apply_reserved(
                receipt,
                request,
                None,
                crash_at=None,
                writer_identity=SemanticMutationWriter.OPERATOR_RECOVERY,
            )

        probe = self._probe(receipt)
        if mode == RecoveryMode.ROLLBACK_IF_SAFE:
            if probe.state == ProbeState.PRE_STATE:
                rolled = self._terminal(
                    receipt,
                    MutationReceiptStatus.ROLLED_BACK,
                    "pre_state_verified_rollback",
                )
                return self._result(
                    MutationExecutionStatus.ROLLED_BACK,
                    rolled,
                    request,
                    reason="pre_state_verified_rollback",
                )
            blocked = self._block(receipt, "rollback_state_unknown")
            return self._result(
                MutationExecutionStatus.BLOCKED,
                blocked,
                request,
                reason="rollback_state_unknown",
            )

        if receipt.phase == MutationReceiptPhase.APPLYING:
            if probe.state == ProbeState.PRE_STATE:
                return self._call_backend(
                    receipt,
                    request,
                    None,
                    crash_at=None,
                    writer_identity=SemanticMutationWriter.OPERATOR_RECOVERY,
                )
            if (
                probe.state == ProbeState.DESIRED_STATE
                and receipt.action != InternalMemoryAction.DELETE
            ):
                applied = self._mark_applied_from_probe(
                    receipt,
                    probe,
                    writer_identity=SemanticMutationWriter.RSIMEM_EXECUTOR,
                )
                return self._verify_and_commit(applied, request, None, crash_at=None)
            blocked = self._block(receipt, "applying_state_unknown")
            return self._result(
                MutationExecutionStatus.BLOCKED,
                blocked,
                request,
                reason="applying_state_unknown",
            )

        if receipt.phase in {
            MutationReceiptPhase.APPLIED,
            MutationReceiptPhase.VERIFYING,
            MutationReceiptPhase.VERIFIED,
        }:
            if probe.state != ProbeState.DESIRED_STATE:
                blocked = self._block(receipt, "post_apply_state_unknown")
                return self._result(
                    MutationExecutionStatus.BLOCKED,
                    blocked,
                    request,
                    reason="post_apply_state_unknown",
                )
            if receipt.phase == MutationReceiptPhase.VERIFIED:
                committed = self._terminal(
                    receipt,
                    MutationReceiptStatus.COMMITTED,
                    None,
                )
                return self._result(
                    MutationExecutionStatus.COMMITTED,
                    committed,
                    request,
                    reason="recovery_committed",
                )
            if receipt.phase == MutationReceiptPhase.APPLIED:
                return self._verify_and_commit(receipt, request, None, crash_at=None)
            return self._verify_and_commit(receipt, request, None, crash_at=None)

        blocked = self._block(receipt, "unknown_receipt_phase")
        return self._result(
            MutationExecutionStatus.BLOCKED,
            blocked,
            request,
            reason="unknown_receipt_phase",
        )

    @staticmethod
    def _inject(configured: CrashPoint | None, point: CrashPoint) -> None:
        if configured is not None and CrashPoint(configured) == point:
            raise InjectedMutationCrash(point)

    def _build_receipt(
        self,
        request: MutationExecutionRequest,
        candidate_digest: str,
        *,
        identity: dict[str, object] | None = None,
        pre_artifact_ids: tuple[str, ...] | None = None,
    ) -> MutationReceipt:
        mutation_identity = identity or self._mutation_identity(request, candidate_digest)
        provenance = request.trusted_context.provenance
        matching = [
            operation
            for operation in request.ingest_result.operations
            if operation.operation_id == provenance.operation_id
        ]
        if len(matching) != 1:
            raise ValueError("mutation execution requires one trusted ingest operation")
        operation = matching[0]
        action = operation.action
        candidate = request.candidate
        backend = request.ingest_result.fixed_route.backend
        namespace = str(candidate.namespace)
        if pre_artifact_ids is None:
            pre_artifact_ids = self._namespace_artifact_ids(
                request.ingest_result.fixed_route.kind,
                namespace,
            )
        mutation_digest = _sha(_canonical_json(mutation_identity))
        mutation_id = _stable_id("mutation", mutation_identity)
        idempotency_key = _stable_id("mutation-key", mutation_identity)
        lock_key = (
            _stable_id("target", {
                "backend": backend,
                "artifact_id": operation.target_artifact_id,
            })
            if operation.target_artifact_id is not None
            else _stable_id("add", {
                "backend": backend,
                "namespace": namespace,
                "after_content_digest": operation.new_content_digest,
                "action": action.value,
            })
        )
        return MutationReceipt(
            receipt_id=_stable_id("receipt", {"idempotency_key": idempotency_key}),
            idempotency_key=idempotency_key,
            mutation_id=mutation_id,
            attempt=1,
            backend=backend,
            lock_key=lock_key,
            target_artifact_id=operation.target_artifact_id,
            pre_revision=operation.expected_revision,
            pre_content_digest=operation.old_content_digest,
            mutation_digest=mutation_digest,
            candidate_digest=candidate_digest,
            after_content_digest=operation.new_content_digest,
            namespace=namespace,
            action=action,
            kind=request.ingest_result.fixed_route.kind,
            provenance=provenance,
            pre_artifact_ids=pre_artifact_ids,
        )

    @staticmethod
    def _mutation_identity(
        request: MutationExecutionRequest,
        candidate_digest: str,
    ) -> dict[str, object]:
        provenance = request.trusted_context.provenance
        matching = [
            operation
            for operation in request.ingest_result.operations
            if operation.operation_id == provenance.operation_id
        ]
        if len(matching) != 1:
            raise ValueError("mutation execution requires one trusted ingest operation")
        operation = matching[0]
        return {
            "schema_version": MUTATION_EXECUTOR_SCHEMA_VERSION,
            "ingest_idempotency_key": request.ingest_result.idempotency_key,
            "execution_id": request.ingest_result.execution_id,
            "operation_id": operation.operation_id,
            "candidate_digest": candidate_digest,
            "action": operation.action.value,
            "backend": request.ingest_result.fixed_route.backend,
            "target_artifact_id": operation.target_artifact_id,
            "expected_revision": operation.expected_revision,
            "old_content_digest": operation.old_content_digest,
            "new_content_digest": operation.new_content_digest,
        }

    @staticmethod
    def _require_same_core(actual: MutationReceipt, expected: MutationReceipt) -> None:
        if actual.core_payload() != expected.core_payload():
            raise ValueError("mutation request conflicts with persisted receipt")

    def _apply_reserved(
        self,
        receipt: MutationReceipt,
        request: MutationExecutionRequest,
        validation: ValidationResult | None,
        *,
        crash_at: CrashPoint | None,
        writer_identity: SemanticMutationWriter = SemanticMutationWriter.RSIMEM_EXECUTOR,
        parent_operation_ids: tuple[str, ...] = (),
        validation_artifact_id: str | None = None,
    ) -> MutationExecutionResult:
        applying = self._transition(receipt, phase=MutationReceiptPhase.APPLYING)
        self._inject(crash_at, CrashPoint.BEFORE_BACKEND_CALL)
        return self._call_backend(
            applying,
            request,
            validation,
            crash_at=crash_at,
            writer_identity=writer_identity,
            parent_operation_ids=parent_operation_ids,
            validation_artifact_id=validation_artifact_id,
        )

    def _call_backend(
        self,
        receipt: MutationReceipt,
        request: MutationExecutionRequest,
        validation: ValidationResult | None,
        *,
        crash_at: CrashPoint | None,
        writer_identity: SemanticMutationWriter,
        parent_operation_ids: tuple[str, ...] = (),
        validation_artifact_id: str | None = None,
    ) -> MutationExecutionResult:
        context = self._operation_context(request)
        mutation_spec = self._operation_spec(
            OperationKind.MUTATION,
            request,
            context,
            "mutation",
            parent_ids=parent_operation_ids,
            input_ids=tuple(dict.fromkeys((
                *request.evidence_input_artifact_ids,
                *(
                    (validation_artifact_id,)
                    if validation_artifact_id is not None
                    else ()
                ),
            ))),
        )
        if receipt.action == InternalMemoryAction.NONE:
            if self.operation_recorder is not None:
                with self.operation_recorder.operation_scope(mutation_spec) as scope:
                    scope.complete(
                        status=OperationStatus.NONE,
                        reason_code="no_memory_change",
                    )
                self.operation_recorder.record_mutation(MutationEdge(
                    mutation_id=receipt.mutation_id,
                    operation_id=mutation_spec.operation_id,
                    proposal_operation_ids=(
                        request.evidence_proposal_operation_ids
                        or (request.trusted_context.provenance.operation_id,)
                    ),
                    action=receipt.action,
                    target_artifact_id=None,
                    expected_revision=None,
                    before_digest=None,
                    after_digest=None,
                    receipt_id=None,
                ))
            applied = self._transition(receipt, phase=MutationReceiptPhase.APPLIED)
            self._inject(crash_at, CrashPoint.AFTER_BACKEND_WRITE)
            return self._verify_and_commit(
                applied,
                request,
                validation,
                crash_at=crash_at,
                parent_operation_ids=(mutation_spec.operation_id,),
            )
        backend = self.registry.resolve(receipt.kind)
        try:
            if self.operation_recorder is None:
                result = backend.mutate(self._memory_mutation(receipt, request.candidate))
            else:
                with self.operation_recorder.operation_scope(mutation_spec) as scope:
                    result = backend.mutate(
                        self._memory_mutation(receipt, request.candidate)
                    )
                    scope.complete(
                        status=(
                            OperationStatus.REJECTED
                            if result.accepted
                            and receipt.action == InternalMemoryAction.ADD
                            and result.reason_code == "already_present"
                            else
                            OperationStatus.SUCCESS
                            if result.accepted
                            else OperationStatus.REJECTED
                        ),
                        reason_code=(
                            "add_ownership_ambiguous"
                            if result.accepted
                            and receipt.action == InternalMemoryAction.ADD
                            and result.reason_code == "already_present"
                            else
                            None
                            if result.accepted
                            else result.reason_code or "backend_rejected"
                        ),
                    )
        except Exception:
            return self._backend_failure(
                receipt,
                request,
                validation,
                "backend_exception",
            )
        if result.backend != receipt.backend or result.action.value != receipt.action.value:
            blocked = self._block(receipt, "backend_result_mismatch")
            return self._result(
                MutationExecutionStatus.BLOCKED,
                blocked,
                request,
                reason="backend_result_mismatch",
                validation=validation,
            )
        if not result.accepted:
            reason = (
                result.reason_code
                if isinstance(result.reason_code, str)
                and _REASON_CODE.fullmatch(result.reason_code)
                else "backend_rejected"
            )
            return self._backend_failure(receipt, request, validation, reason)
        if (
            receipt.action == InternalMemoryAction.ADD
            and result.reason_code == "already_present"
        ):
            blocked = self._block(receipt, "add_ownership_ambiguous")
            return self._result(
                MutationExecutionStatus.BLOCKED,
                blocked,
                request,
                reason="add_ownership_ambiguous",
                validation=validation,
            )
        applied_artifact_id = (
            receipt.target_artifact_id
            if receipt.action == InternalMemoryAction.DELETE
            else result.artifact_id
        )
        if applied_artifact_id is None:
            blocked = self._block(receipt, "backend_result_incomplete")
            return self._result(
                MutationExecutionStatus.BLOCKED,
                blocked,
                request,
                reason="backend_result_incomplete",
                validation=validation,
            )
        self._inject(crash_at, CrashPoint.AFTER_BACKEND_WRITE)
        applied = self._transition(
            receipt,
            phase=MutationReceiptPhase.APPLIED,
            applied_artifact_id=applied_artifact_id,
            applied_revision=result.revision,
            writer_identity=writer_identity,
        )
        if self.operation_recorder is not None:
            self.operation_recorder.record_mutation(MutationEdge(
                mutation_id=receipt.mutation_id,
                operation_id=mutation_spec.operation_id,
                proposal_operation_ids=(
                    request.evidence_proposal_operation_ids
                    or (request.trusted_context.provenance.operation_id,)
                ),
                action=receipt.action,
                target_artifact_id=(
                    receipt.target_artifact_id or applied_artifact_id
                ),
                expected_revision=receipt.pre_revision,
                before_digest=receipt.pre_content_digest,
                after_digest=receipt.after_content_digest,
                receipt_id=receipt.receipt_id,
            ))
        return self._verify_and_commit(
            applied,
            request,
            validation,
            crash_at=crash_at,
            parent_operation_ids=(mutation_spec.operation_id,),
        )

    def _backend_failure(
        self,
        receipt: MutationReceipt,
        request: MutationExecutionRequest,
        validation: ValidationResult | None,
        reason: str,
    ) -> MutationExecutionResult:
        probe = self._probe(receipt)
        if probe.state == ProbeState.PRE_STATE:
            failed = self._terminal(receipt, MutationReceiptStatus.FAILED, reason)
            return self._result(
                MutationExecutionStatus.FAILED,
                failed,
                request,
                reason=reason,
                validation=validation,
            )
        blocked = self._block(receipt, "backend_failure_state_unknown")
        return self._result(
            MutationExecutionStatus.BLOCKED,
            blocked,
            request,
            reason="backend_failure_state_unknown",
            validation=validation,
        )

    def _verify_and_commit(
        self,
        receipt: MutationReceipt,
        request: MutationExecutionRequest,
        validation: ValidationResult | None,
        *,
        crash_at: CrashPoint | None,
        parent_operation_ids: tuple[str, ...] = (),
    ) -> MutationExecutionResult:
        current = receipt
        if current.phase == MutationReceiptPhase.APPLIED:
            current = self._transition(current, phase=MutationReceiptPhase.VERIFYING)
        self._inject(crash_at, CrashPoint.BEFORE_VERIFICATION)
        verification_spec = self._operation_spec(
            OperationKind.REREAD_VERIFICATION,
            request,
            self._operation_context(request),
            "verification",
            parent_ids=parent_operation_ids,
            input_ids=(
                (current.applied_artifact_id,)
                if current.action != InternalMemoryAction.DELETE
                and current.applied_artifact_id is not None
                else ()
            ),
        )
        if self.operation_recorder is None:
            verified, reason, artifact, storage_bytes = self._verify(
                current,
                request.candidate,
            )
        else:
            with self.operation_recorder.operation_scope(verification_spec) as scope:
                verified, reason, artifact, storage_bytes = self._verify(
                    current,
                    request.candidate,
                )
                output_artifact_ids = ()
                if artifact is not None:
                    memory_artifact = self._memory_artifact(
                        request,
                        artifact.artifact_id,
                        artifact.content,
                        artifact.revision,
                    )
                    self.operation_recorder.record_artifact(memory_artifact)
                    output_artifact_ids = (memory_artifact.artifact_id,)
                scope.complete(
                    output_artifact_ids=output_artifact_ids,
                    status=(
                        OperationStatus.SUCCESS
                        if verified
                        else OperationStatus.FAILED
                    ),
                    reason_code=None if verified else reason,
                )
        if not verified:
            blocked = self._block(current, reason)
            return self._result(
                MutationExecutionStatus.BLOCKED,
                blocked,
                request,
                reason=reason,
                validation=validation,
            )
        verified_receipt = self._transition(
            current,
            phase=MutationReceiptPhase.VERIFIED,
            applied_artifact_id=(
                artifact.artifact_id if artifact is not None else current.applied_artifact_id
            ),
            applied_revision=(artifact.revision if artifact is not None else None),
            applied_content_digest=(
                _sha(artifact.content) if artifact is not None else None
            ),
            verified=True,
            storage_bytes=storage_bytes,
        )
        self._inject(crash_at, CrashPoint.BEFORE_RECEIPT_COMMIT)
        committed = self._terminal(
            verified_receipt,
            MutationReceiptStatus.COMMITTED,
            None,
        )
        return self._result(
            MutationExecutionStatus.COMMITTED,
            committed,
            request,
            reason="mutation_committed",
            validation=validation,
        )

    def _verify(
        self,
        receipt: MutationReceipt,
        candidate: UntrustedMemoryCandidate,
    ) -> tuple[bool, str, MemoryArtifact | None, int]:
        backend = self.registry.resolve(receipt.kind)
        try:
            if receipt.action in {InternalMemoryAction.ADD, InternalMemoryAction.UPDATE}:
                if receipt.applied_artifact_id is None:
                    return False, "missing_applied_artifact", None, 0
                artifact = backend.get(receipt.applied_artifact_id)
                if artifact is None:
                    return False, "reread_missing", None, 0
                if artifact.kind != receipt.kind or artifact.namespace != receipt.namespace:
                    return False, "reread_identity_mismatch", artifact, 0
                if _sha(artifact.content) != receipt.after_content_digest:
                    return False, "reread_digest_mismatch", artifact, 0
                if not isinstance(artifact.revision, str) or not artifact.revision:
                    return False, "reread_revision_missing", artifact, 0
                if artifact.resources:
                    return False, "reread_resource_mismatch", artifact, 0
                storage_bytes = len(artifact.content.encode("utf-8")) + sum(
                    len(resource.content) for resource in artifact.resources
                )
                return True, "verified", artifact, storage_bytes
            if receipt.action == InternalMemoryAction.DELETE:
                if receipt.target_artifact_id is None:
                    return False, "missing_delete_target", None, 0
                if backend.get(receipt.target_artifact_id) is not None:
                    return False, "reread_delete_mismatch", None, 0
                return True, "verified", None, 0
            return True, "verified", None, 0
        except Exception:
            return False, "reread_exception", None, 0

    @staticmethod
    def _memory_mutation(
        receipt: MutationReceipt,
        candidate: UntrustedMemoryCandidate,
    ) -> MemoryMutation:
        action = {
            InternalMemoryAction.ADD: MemoryMutationAction.ADD,
            InternalMemoryAction.UPDATE: MemoryMutationAction.UPDATE,
            InternalMemoryAction.DELETE: MemoryMutationAction.DELETE,
        }[receipt.action]
        artifact = None
        if receipt.action in {InternalMemoryAction.ADD, InternalMemoryAction.UPDATE}:
            if not isinstance(candidate.content, str) or not isinstance(
                candidate.metadata,
                Mapping,
            ):
                raise ValueError("validated mutation candidate content is invalid")
            artifact = MemoryArtifact(
                artifact_id=(
                    str(candidate.candidate_id)
                    if receipt.action == InternalMemoryAction.ADD
                    else receipt.target_artifact_id or "missing-target"
                ),
                kind=receipt.kind,
                content=candidate.content,
                namespace=receipt.namespace,
                metadata=dict(candidate.metadata),
            )
        return MemoryMutation(
            action=action,
            kind=receipt.kind,
            artifact=artifact,
            artifact_id=receipt.target_artifact_id,
            expected_revision=receipt.pre_revision,
        )

    def _probe(self, receipt: MutationReceipt) -> ProbeResult:
        try:
            backend = self.registry.resolve(receipt.kind)
            if receipt.action == InternalMemoryAction.NONE:
                return ProbeResult(ProbeState.DESIRED_STATE)
            if receipt.action == InternalMemoryAction.ADD:
                matches = self._find_digest(backend, receipt)
                if len(matches) == 1:
                    return ProbeResult(ProbeState.DESIRED_STATE, matches[0])
                if matches:
                    return ProbeResult(ProbeState.UNKNOWN)
                current_ids = tuple(
                    artifact.artifact_id
                    for artifact in self._namespace_artifacts(backend, receipt)
                )
                return ProbeResult(
                    ProbeState.PRE_STATE
                    if current_ids == receipt.pre_artifact_ids
                    else ProbeState.UNKNOWN
                )
            if receipt.target_artifact_id is None:
                return ProbeResult(ProbeState.UNKNOWN)
            target = backend.get(receipt.target_artifact_id)
            if target is not None:
                if (
                    target.revision == receipt.pre_revision
                    and _sha(target.content) == receipt.pre_content_digest
                ):
                    return ProbeResult(ProbeState.PRE_STATE, target)
                return ProbeResult(ProbeState.UNKNOWN, target)
            if receipt.action == InternalMemoryAction.DELETE:
                return ProbeResult(ProbeState.DESIRED_STATE)
            matches = self._find_digest(backend, receipt)
            if len(matches) == 1:
                return ProbeResult(ProbeState.DESIRED_STATE, matches[0])
            return ProbeResult(ProbeState.UNKNOWN)
        except Exception:
            return ProbeResult(ProbeState.UNKNOWN)

    @staticmethod
    def _find_digest(backend: Any, receipt: MutationReceipt) -> tuple[MemoryArtifact, ...]:
        if receipt.after_content_digest is None:
            return ()
        hits = backend.query(MemoryQuery(
            receipt.kind,
            "",
            namespace=receipt.namespace,
            limit=10_000,
        ))
        return tuple(
            hit.artifact
            for hit in hits
            if _sha(hit.artifact.content) == receipt.after_content_digest
        )

    def _namespace_artifact_ids(
        self,
        kind: MemoryKind,
        namespace: str,
    ) -> tuple[str, ...]:
        backend = self.registry.resolve(kind)
        artifacts = self._namespace_artifacts_for(kind, namespace, backend)
        return tuple(artifact.artifact_id for artifact in artifacts)

    @staticmethod
    def _namespace_artifacts(
        backend: Any,
        receipt: MutationReceipt,
    ) -> tuple[MemoryArtifact, ...]:
        return TransactionalMutationExecutor._namespace_artifacts_for(
            receipt.kind,
            receipt.namespace,
            backend,
        )

    @staticmethod
    def _namespace_artifacts_for(
        kind: MemoryKind,
        namespace: str,
        backend: Any,
    ) -> tuple[MemoryArtifact, ...]:
        hits = backend.query(MemoryQuery(
            kind,
            "",
            namespace=namespace,
            limit=10_000,
        ))
        return tuple(
            sorted(
                (hit.artifact for hit in hits),
                key=lambda artifact: artifact.artifact_id,
            )
        )

    def _mark_applied_from_probe(
        self,
        receipt: MutationReceipt,
        probe: ProbeResult,
        *,
        writer_identity: SemanticMutationWriter,
    ) -> MutationReceipt:
        artifact = probe.artifact
        if receipt.action in {InternalMemoryAction.ADD, InternalMemoryAction.UPDATE}:
            if artifact is None:
                return self._block(receipt, "recovery_artifact_missing")
            return self._transition(
                receipt,
                phase=MutationReceiptPhase.APPLIED,
                applied_artifact_id=artifact.artifact_id,
                applied_revision=artifact.revision,
                writer_identity=writer_identity,
            )
        return self._transition(
            receipt,
            phase=MutationReceiptPhase.APPLIED,
            applied_artifact_id=receipt.target_artifact_id,
            writer_identity=writer_identity,
        )

    def _transition(self, receipt: MutationReceipt, **changes: object) -> MutationReceipt:
        updated = replace(
            receipt,
            **changes,
            store_revision=receipt.store_revision + 1,
        )
        return self.receipt_store.transition(
            updated,
            expected_store_revision=receipt.store_revision,
        )

    def _terminal(
        self,
        receipt: MutationReceipt,
        status: MutationReceiptStatus,
        reason: str | None,
    ) -> MutationReceipt:
        return self._transition(
            receipt,
            status=status,
            phase=MutationReceiptPhase.TERMINAL,
            verified=(receipt.verified if status == MutationReceiptStatus.COMMITTED else False),
            target_blocked=False,
            reason_code=reason,
        )

    def _block(self, receipt: MutationReceipt, reason: str) -> MutationReceipt:
        if receipt.target_blocked and receipt.reason_code == reason:
            return receipt
        return self._transition(
            receipt,
            target_blocked=True,
            reason_code=reason,
        )

    def _existing_result(
        self,
        receipt: MutationReceipt,
        request: MutationExecutionRequest,
        *,
        validation: ValidationResult | None = None,
    ) -> MutationExecutionResult:
        if receipt.status == MutationReceiptStatus.COMMITTED:
            return self._result(
                MutationExecutionStatus.DUPLICATE,
                receipt,
                request,
                reason="duplicate_committed_mutation",
                validation=validation,
            )
        if receipt.status == MutationReceiptStatus.FAILED:
            status = MutationExecutionStatus.FAILED
        elif receipt.status == MutationReceiptStatus.ROLLED_BACK:
            status = MutationExecutionStatus.ROLLED_BACK
        else:
            status = MutationExecutionStatus.BLOCKED
        return self._result(
            status,
            receipt,
            request,
            reason=receipt.reason_code or "mutation_in_progress",
            validation=validation,
        )

    def _result(
        self,
        status: MutationExecutionStatus,
        receipt: MutationReceipt,
        request: MutationExecutionRequest,
        *,
        reason: str,
        validation: ValidationResult | None = None,
    ) -> MutationExecutionResult:
        reason = reason if _REASON_CODE.fullmatch(reason) else "mutation_failed"
        return MutationExecutionResult(
            status=status,
            mutation_id=receipt.mutation_id,
            receipt_id=receipt.receipt_id,
            reason_code=reason,
            validation=validation,
            artifact_id=receipt.applied_artifact_id,
            revision=receipt.applied_revision,
            storage_bytes=receipt.storage_bytes,
            writer_identity=receipt.writer_identity,
            context_exit=self.exit_gate.evaluate(
                request.trigger,
                receipt if status in {
                    MutationExecutionStatus.COMMITTED,
                    MutationExecutionStatus.DUPLICATE,
                } else None,
                failure_reason=reason,
            ),
        )

    def _unpersisted_result(
        self,
        status: MutationExecutionStatus,
        mutation_id: str,
        request: MutationExecutionRequest,
        *,
        reason: str,
        validation: ValidationResult | None = None,
    ) -> MutationExecutionResult:
        reason = reason if _REASON_CODE.fullmatch(reason) else "mutation_failed"
        return MutationExecutionResult(
            status=status,
            mutation_id=mutation_id,
            receipt_id=None,
            reason_code=reason,
            validation=validation,
            artifact_id=None,
            revision=None,
            storage_bytes=0,
            writer_identity=None,
            context_exit=self.exit_gate.evaluate(
                request.trigger,
                None,
                failure_reason=reason,
            ),
        )

    @staticmethod
    def _validation_artifact(
        request: MutationExecutionRequest,
        context: OperationContext,
        validation: ValidationResult,
    ) -> ArtifactNode:
        payload = _canonical_json(validation.observer_evidence())
        digest = _sha(payload)
        return ArtifactNode(
            build_artifact_id(
                ArtifactKind.VALIDATION_RESULT,
                context,
                logical_name=f"{request.ingest_result.execution_id}.validation",
                content_digest=digest,
            ),
            ArtifactKind.VALIDATION_RESULT,
            "semantic-validation-result-v1",
            digest,
            len(payload.encode("utf-8")),
            None,
            None,
            validation.operation_id,
        )

    @staticmethod
    def _memory_artifact(
        request: MutationExecutionRequest,
        artifact_id: str,
        content: str,
        revision: str | None,
    ) -> ArtifactNode:
        return ArtifactNode(
            artifact_id,
            ArtifactKind.MEMORY_ARTIFACT,
            "hermes-semantic-artifact-v1",
            _sha(content),
            len(content.encode("utf-8")),
            None,
            revision,
            request.ingest_result.fixed_route.backend,
        )

    @staticmethod
    def _operation_context(request: MutationExecutionRequest) -> OperationContext:
        provenance = request.trusted_context.provenance
        return OperationContext(
            run_id=provenance.run_id,
            episode_id=provenance.episode_id,
            session_id=provenance.session_id,
            task_id=provenance.task_id,
            policy_version=request.ingest_result.policy_version,
            prompt_version=request.ingest_result.prompt_version,
            framework_version=request.ingest_result.framework_version,
        )

    @staticmethod
    def _operation_spec(
        kind: OperationKind,
        request: MutationExecutionRequest,
        context: OperationContext,
        step: str,
        *,
        parent_ids: tuple[str, ...] = (),
        input_ids: tuple[str, ...] = (),
    ) -> OperationSpec:
        step_id = f"{request.ingest_result.execution_id}.{step}"
        operation_id = build_operation_id(
            kind,
            context,
            step_id=step_id,
            parent_operation_ids=parent_ids,
            input_artifact_ids=input_ids,
        )
        return OperationSpec(
            operation_id,
            kind,
            context,
            parent_ids,
            input_ids,
        )
