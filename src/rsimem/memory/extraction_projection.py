"""Project real Mem0-flat compilation results into extraction feedback evidence."""

from __future__ import annotations

from .executor import MutationExecutionStatus
from .extraction_feedback import (
    ExtractedFactEvidence,
    ExtractionQualityIssue,
    ExtractionSetStatus,
    ExtractionSourceEvidence,
    FactDisposition,
    FeedbackContractRegistry,
    default_feedback_contract_registry,
    detect_extracted_fact_semantic_keys,
)
from .ingestion import InternalMemoryAction, MemoryIngestStatus
from .live_writeback import StaticSemanticBoundaryResult
from ..memory_systems.mem0_flat.policy import Mem0FlatSemanticPolicy


class Mem0FlatExtractionSourceProjector:
    """Build content-free source evidence from the actual policy and executor trace."""

    def __init__(self, registry: FeedbackContractRegistry | None = None) -> None:
        self.registry = registry or default_feedback_contract_registry()

    def project(
        self,
        boundary: StaticSemanticBoundaryResult,
        policy: Mem0FlatSemanticPolicy,
        *,
        family_id: str,
        available_semantic_keys: tuple[str, ...],
    ) -> ExtractionSourceEvidence:
        if boundary.duplicate or boundary.writeback is None:
            raise ValueError("extraction projection requires an original writeback result")
        ingestion = boundary.writeback.ingestion
        if ingestion is None:
            raise ValueError("extraction projection requires an ingestion result")
        trace = policy.operation_trace(ingestion.idempotency_key)
        if trace is None:
            raise ValueError("extraction projection requires a Mem0-flat operation trace")
        contract = self.registry.resolver(family_id).contract
        allowed_keys = set(contract.opportunity.memory_scope_keys)
        available_keys = tuple(available_semantic_keys)
        if len(available_keys) != len(set(available_keys)):
            raise ValueError("available extraction semantic keys must be unique")
        if set(available_keys) - allowed_keys:
            raise ValueError("available extraction semantic keys escape family contract")
        if (
            boundary.receipt is not None
            and boundary.receipt.source_projection_digest != ingestion.source_digest
        ):
            raise ValueError("compilation receipt and ingestion source digest disagree")

        operations = ingestion.operations
        executions = boundary.writeback.executions
        accepted_index = 0
        facts = []
        for extraction in trace.fact_extractions:
            fact = policy.fact_for_digest(extraction.content_digest)
            if fact is None or fact.fact_id != extraction.fact_id:
                raise ValueError("policy fact owner disagrees with extraction trace")
            semantic_keys = detect_extracted_fact_semantic_keys(
                family_id,
                fact.content,
            )
            if set(semantic_keys) - allowed_keys:
                raise ValueError("extracted fact semantic keys escape family contract")
            quality_issue = (
                ExtractionQualityIssue.UNSUPPORTED
                if semantic_keys and not set(semantic_keys).issubset(available_keys)
                else None
            )
            artifact_id = None
            if not extraction.accepted:
                disposition = FactDisposition.FILTERED
            else:
                operation = (
                    operations[accepted_index]
                    if accepted_index < len(operations)
                    else None
                )
                execution = (
                    executions[accepted_index]
                    if accepted_index < len(executions)
                    else None
                )
                accepted_index += 1
                if operation is None or ingestion.status != MemoryIngestStatus.SUCCESS:
                    disposition = FactDisposition.MUTATION_FAILED
                elif operation.action in {
                    InternalMemoryAction.NONE,
                    InternalMemoryAction.DELETE,
                }:
                    disposition = FactDisposition.NONE
                elif (
                    execution is not None
                    and execution.status in {
                        MutationExecutionStatus.COMMITTED,
                        MutationExecutionStatus.DUPLICATE,
                    }
                    and execution.artifact_id is not None
                ):
                    disposition = FactDisposition.PERSISTED
                    artifact_id = execution.artifact_id
                else:
                    disposition = FactDisposition.MUTATION_FAILED
            facts.append(ExtractedFactEvidence(
                extraction.fact_id,
                semantic_keys,
                disposition,
                artifact_id=artifact_id,
                quality_issue=quality_issue,
            ))

        dispositions = {fact.disposition for fact in facts}
        if not facts:
            status = (
                ExtractionSetStatus.EMPTY
                if ingestion.status == MemoryIngestStatus.SUCCESS
                and any(
                    operation.action == InternalMemoryAction.NONE
                    for operation in operations
                )
                else ExtractionSetStatus.NONE
            )
        elif FactDisposition.MUTATION_FAILED in dispositions:
            status = ExtractionSetStatus.MUTATION_FAILED
        elif FactDisposition.PERSISTED in dispositions:
            status = ExtractionSetStatus.NONEMPTY
        elif dispositions == {FactDisposition.FILTERED}:
            status = ExtractionSetStatus.FILTERED
        else:
            status = ExtractionSetStatus.NONE
        return ExtractionSourceEvidence(
            trace.source_artifact_id,
            ingestion.source_digest,
            trace.extraction_operation_id,
            status,
            available_keys,
            tuple(facts),
        )
