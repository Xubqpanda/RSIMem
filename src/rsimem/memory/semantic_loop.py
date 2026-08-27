"""Semantic ingestion-to-transaction orchestration for isolated end-to-end gates."""

from __future__ import annotations

from dataclasses import dataclass

from .executor import (
    MutationExecutionRequest,
    MutationExecutionResult,
    MutationExecutionStatus,
    TransactionalMutationExecutor,
)
from .contracts import MemoryEvent, MemoryEventKind, MemoryKind, MemoryObserver
from .ingestion import (
    InternalMemoryAction,
    MemoryIngestResult,
    MemoryIngestStatus,
    SemanticIngestRequest,
    SemanticIngestionCoordinator,
)
from .operation_graph import (
    AtomicOperationRecorder,
    OperationKind,
    OperationSpec,
)
from .validation import TrustedValidationContext, ValidationProvenance
from ..memory_systems.mem0_flat.policy import (
    FlatSemanticCandidateReader,
    Mem0FlatSemanticPolicy,
    build_validation_candidate,
)


SEMANTIC_LOOP_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SemanticWritebackLoopResult:
    ingestion: MemoryIngestResult | None
    executions: tuple[MutationExecutionResult, ...]
    logical_exit: bool
    source_retained: bool
    reason_code: str
    schema_version: int = SEMANTIC_LOOP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEMANTIC_LOOP_SCHEMA_VERSION:
            raise ValueError("unsupported semantic loop result schema version")
        if type(self.logical_exit) is not bool or type(self.source_retained) is not bool:
            raise TypeError("semantic loop exit flags must be bool")
        if self.logical_exit == self.source_retained:
            raise ValueError("semantic loop must either exit or retain source")
        if not self.reason_code.strip():
            raise ValueError("semantic loop reason_code must not be empty")

    def observer_evidence(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "ingestion_execution_id": (
                self.ingestion.execution_id if self.ingestion is not None else None
            ),
            "ingestion_status": (
                self.ingestion.status.value if self.ingestion is not None else None
            ),
            "mutation_ids": [item.mutation_id for item in self.executions],
            "mutation_statuses": [item.status.value for item in self.executions],
            "logical_exit": self.logical_exit,
            "source_retained": self.source_retained,
            "reason_code": self.reason_code,
        }


class SemanticWritebackLoop:
    """Run Mem0-flat planning then validated transaction execution."""

    def __init__(
        self,
        coordinator: SemanticIngestionCoordinator,
        policy: Mem0FlatSemanticPolicy,
        candidates: FlatSemanticCandidateReader,
        executor: TransactionalMutationExecutor,
        observer: MemoryObserver | None = None,
        operation_recorder: AtomicOperationRecorder | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.policy = policy
        self.candidates = candidates
        self.executor = executor
        self.observer = observer
        self.operation_recorder = operation_recorder
        if operation_recorder is not None:
            if policy.operation_recorder not in {None, operation_recorder}:
                raise ValueError("semantic loop policy uses a different operation recorder")
            if executor.operation_recorder not in {None, operation_recorder}:
                raise ValueError("semantic loop executor uses a different operation recorder")
            policy.operation_recorder = operation_recorder
            executor.operation_recorder = operation_recorder

    def run(
        self,
        request: SemanticIngestRequest,
        *,
        current_source_revision: str,
    ) -> SemanticWritebackLoopResult:
        if not current_source_revision.strip():
            raise ValueError("semantic loop current_source_revision must not be empty")
        ingestion = self.coordinator.ingest(
            request,
            self.candidates,
            current_source_revision=current_source_revision,
        )
        if ingestion is None:
            return SemanticWritebackLoopResult(
                None,
                (),
                False,
                True,
                "ingestion_disabled",
            )
        if ingestion.status != MemoryIngestStatus.SUCCESS:
            return SemanticWritebackLoopResult(
                ingestion,
                (),
                False,
                True,
                "ingestion_not_successful",
            )

        executions = []
        operation_trace = self.policy.operation_trace(request.idempotency_key)
        for index, operation in enumerate(ingestion.operations):
            source = request.provenance.source
            provenance = ValidationProvenance(
                run_id=source.run_id,
                episode_id=source.episode_id,
                session_id=source.session_id,
                task_id=source.task_id,
                snapshot_id=source.snapshot_id,
                execution_id=ingestion.execution_id,
                operation_id=operation.operation_id,
                source_digest=ingestion.source_digest,
            )
            candidate = build_validation_candidate(
                ingestion,
                index,
                self.policy,
                provenance,
            )
            evidence_input_ids: tuple[str, ...] = ()
            evidence_proposal_ids: tuple[str, ...] = ()
            if self.operation_recorder is not None and operation_trace is not None:
                proposal_id = (
                    operation_trace.proposal_artifact_ids[index]
                    if index < len(operation_trace.proposal_artifact_ids)
                    else None
                )
                values = []
                if proposal_id is not None:
                    values.append(proposal_id)
                if (
                    operation.target_artifact_id is not None
                    and operation.target_artifact_id
                    in operation_trace.related_artifact_ids
                ):
                    values.append(operation.target_artifact_id)
                evidence_input_ids = tuple(values)
                parents = (
                    (operation_trace.decision_operation_id,)
                    if operation_trace.decision_operation_id is not None
                    else (operation_trace.extraction_operation_id,)
                )
                resolution_spec = OperationSpec(
                    operation.operation_id,
                    OperationKind.TARGET_RESOLUTION,
                    operation_trace.context,
                    parents,
                    evidence_input_ids,
                )
                with self.operation_recorder.operation_scope(resolution_spec) as scope:
                    scope.complete()
                evidence_proposal_ids = (
                    (operation_trace.decision_operation_id,)
                    if operation_trace.decision_operation_id is not None
                    else (operation.operation_id,)
                )
            if self.observer is not None:
                self.observer.record(MemoryEvent(
                    MemoryEventKind.MUTATION_REQUESTED,
                    MemoryKind.SEMANTIC,
                    ingestion.fixed_route.backend,
                    artifact_ids=(
                        (operation.target_artifact_id,)
                        if operation.target_artifact_id is not None
                        else ()
                    ),
                    attributes={
                        "action": operation.action.value,
                        "execution_id": ingestion.execution_id,
                        "operation_id": operation.operation_id,
                        "snapshot_id": source.snapshot_id,
                    },
                ))
            result = self.executor.execute(MutationExecutionRequest(
                candidate=candidate,
                ingest_result=ingestion,
                trusted_context=TrustedValidationContext(
                    provenance,
                    request.scope,
                    request.validity,
                ),
                current_source_digest=ingestion.source_digest,
                trigger=request.trigger,
                evidence_input_artifact_ids=evidence_input_ids,
                evidence_proposal_operation_ids=evidence_proposal_ids,
            ))
            executions.append(result)
            if self.observer is not None:
                self.observer.record(MemoryEvent(
                    (
                        MemoryEventKind.MUTATION_COMMITTED
                        if result.status in {
                            MutationExecutionStatus.COMMITTED,
                            MutationExecutionStatus.DUPLICATE,
                        }
                        else MemoryEventKind.MUTATION_REJECTED
                    ),
                    MemoryKind.SEMANTIC,
                    ingestion.fixed_route.backend,
                    artifact_ids=(result.artifact_id,) if result.artifact_id else (),
                    reason_code=result.reason_code,
                    attributes={
                        "action": operation.action.value,
                        "execution_id": ingestion.execution_id,
                        "operation_id": operation.operation_id,
                        "mutation_id": result.mutation_id,
                        "receipt_id": result.receipt_id,
                        "snapshot_id": source.snapshot_id,
                    },
                ))
            if result.status not in {
                MutationExecutionStatus.COMMITTED,
                MutationExecutionStatus.DUPLICATE,
            }:
                break

        complete = len(executions) == len(ingestion.operations) and all(
            result.status in {
                MutationExecutionStatus.COMMITTED,
                MutationExecutionStatus.DUPLICATE,
            }
            for result in executions
        )
        has_mutation = any(
            operation.action != InternalMemoryAction.NONE
            for operation in ingestion.operations
        )
        logical_exit = complete and has_mutation
        reason = (
            "all_mutations_committed"
            if logical_exit
            else "no_memory_mutation"
            if complete
            else "mutation_not_committed"
        )
        return SemanticWritebackLoopResult(
            ingestion,
            tuple(executions),
            logical_exit,
            not logical_exit,
            reason,
        )
