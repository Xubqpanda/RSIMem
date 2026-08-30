"""Exact joins from deployment evidence into the optimizer-only corpus."""

from __future__ import annotations

from dataclasses import dataclass

from .extraction_feedback import (
    DeploymentObservation,
    ExtractionFeedbackLabel,
    FactDisposition,
)
from .evidence_planes import EvidencePlane, require_optimizer_plane
from .extraction_optimizer_corpus import (
    ExtractionOptimizerCorpusExample,
    OptimizerArtifactLineage,
    OptimizerAuditJoin,
    OptimizerComponentOwnership,
    OptimizerDelayedEvidence,
    OptimizerExtractedFact,
    OptimizerSourceMessage,
)
from .extraction_projection import ExtractionSourceRecord, LiveExtractionFeedbackRecord
from .pure_extraction import (
    PureExtractionFeedbackRecord,
    PureExtractionOptimizerExample,
    PureExtractionSourceRecord,
)
from .extraction_source import ExtractionSourceProjection
from .operation_graph import ArtifactKind, OperationGraph, OperationKind
from .optimizer_content_boundary import OptimizerSecretBoundary
from .prompt_components import canonical_json, content_digest, text_digest


@dataclass(frozen=True, slots=True)
class ExtractionFactContent:
    fact_id: str
    content: str
    accepted: bool
    reason_code: str | None

    def __post_init__(self) -> None:
        if not self.fact_id.strip() or not isinstance(self.content, str):
            raise ValueError("optimizer fact content input is incomplete")
        if type(self.accepted) is not bool:
            raise TypeError("optimizer fact accepted flag must be bool")
        if self.accepted == (self.reason_code is not None):
            raise ValueError("optimizer fact rejection reason is inconsistent")

    def trace_payload(self) -> dict[str, object]:
        return {
            "fact_id": self.fact_id,
            "content_digest": text_digest(self.content),
            "accepted": self.accepted,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True, slots=True)
class DelayedEvidenceContent:
    source_completed_at: str
    observed_at: str
    current_input: str
    observation_window: str = "window.unbound"


class ExtractionOptimizerCorpusBuilder:
    def __init__(
        self,
        boundary: OptimizerSecretBoundary | None = None,
        *,
        evidence_plane: EvidencePlane | None = None,
    ) -> None:
        self.boundary = boundary or OptimizerSecretBoundary()
        self.evidence_plane = (
            require_optimizer_plane(evidence_plane)
            if evidence_plane is not None else None
        )

    def build_examples(
        self,
        *,
        projection: ExtractionSourceProjection,
        source_record: ExtractionSourceRecord,
        feedback_record: LiveExtractionFeedbackRecord,
        observation: DeploymentObservation,
        operation_graph: OperationGraph,
        fact_contents: tuple[ExtractionFactContent, ...],
        delayed_content: DelayedEvidenceContent,
        forbidden_values: tuple[str, ...] = (),
    ) -> tuple[ExtractionOptimizerCorpusExample, ...]:
        # This builder materializes the legacy family-bound audit path.  A
        # deployment-only projection must use a dedicated pure-process
        # builder; accepting it here would silently reintroduce family/stage
        # labels into the optimizer corpus.
        if not isinstance(source_record, ExtractionSourceRecord) or not isinstance(
            feedback_record, LiveExtractionFeedbackRecord
        ):
            raise TypeError("family-bound optimizer builder requires family-bound source/feedback records")
        self._validate_record_join(projection, source_record, feedback_record)
        evidence_plane = self.evidence_plane or EvidencePlane(
            feedback_record.dataset.evidence_plane
        )
        self._validate_observation(feedback_record, observation, delayed_content)
        facts = self._project_facts(
            source_record,
            operation_graph,
            fact_contents,
            forbidden_values,
        )
        source_messages = tuple(
            OptimizerSourceMessage(
                message.segment_id,
                message.source_message_id,
                message.role,
                message.segment_kind.value,
                message.tool_call_id,
                message.content_truncated,
                self.boundary.project(
                    message.content,
                    forbidden_values=forbidden_values,
                ),
            )
            for message in projection.messages
        )
        operations = self._operation_join(
            source_record,
            feedback_record,
            operation_graph,
        )
        lineages = self._artifact_lineages(
            source_record,
            operation_graph,
        )
        delayed = OptimizerDelayedEvidence(
            feedback_record.deployment_observation_id,
            delayed_content.source_completed_at,
            delayed_content.observed_at,
            feedback_record.dataset.examples[0].future_opportunity_id,
            feedback_record.opportunity_operation_id,
            feedback_record.use_operation_id,
            feedback_record.outcome_operation_id,
            self.boundary.project(
                delayed_content.current_input,
                forbidden_values=forbidden_values,
            ),
            self.boundary.project(
                observation.final_response,
                forbidden_values=forbidden_values,
            ),
            self.boundary.project(
                canonical_json({
                    "completed": observation.completed,
                    "observation_complete": observation.observation_complete,
                    "censor_reason": observation.censor_reason,
                    "tool_events": [{
                        "event_id": event.event_id,
                        "tool_name": event.tool_name,
                        "success": event.success,
                        "subject_ids": list(event.subject_ids),
                        "recipient_ids": list(event.recipient_ids),
                    } for event in observation.tool_events],
                }),
                forbidden_values=forbidden_values,
            ),
        )
        examples = []
        for feedback in feedback_record.dataset.examples:
            join = OptimizerAuditJoin(
                source_record.family_id,
                source_record.record_id,
                source_record.content_digest,
                source_record.stage,
                source_record.run_id,
                source_record.episode_id,
                source_record.session_id,
                source_record.task_id,
                projection.projection_id,
                projection.projection_digest,
                feedback_record.record_id,
                feedback_record.dataset.dataset_id,
                feedback.example_id,
                feedback_record.stage,
                feedback_record.run_id,
                feedback_record.trace_id,
                feedback_record.episode_id,
                feedback_record.session_id,
                feedback_record.task_id,
                source_record.extraction_artifact_id,
                source_record.extraction_artifact_digest,
                source_record.extraction_output_digest,
                operations,
                lineages,
                observation_window=delayed_content.observation_window,
            )
            examples.append(ExtractionOptimizerCorpusExample.create(
                primary_unit_id=feedback.primary_unit_id,
                level=feedback.level,
                primary=feedback.primary,
                feedback_fact_id=feedback.fact_id,
                feedback_semantic_key=feedback.semantic_key,
                feedback_artifact_ids=feedback.artifact_ids,
                exposure_mode=feedback.exposure_mode,
                label=feedback.label,
                attribution_confidence=feedback.attribution_confidence,
                reason_codes=feedback.reason_codes,
                component_ownership=self._ownership(feedback.label, feedback.reason_codes),
                audit_join=join,
                source_messages=source_messages,
                extracted_facts=facts,
                delayed_evidence=delayed,
                evidence_plane=evidence_plane,
            ))
        return tuple(examples)

    def build_pure_process_example(
        self,
        *,
        source: PureExtractionSourceRecord,
        feedback: PureExtractionFeedbackRecord,
    ) -> PureExtractionOptimizerExample:
        """Explicit entry point for the deployment-only optimizer path."""

        return PureExtractionOptimizerBuilder().build_example(
            source=source,
            feedback=feedback,
        )


    @staticmethod
    def _validate_record_join(
        projection: ExtractionSourceProjection,
        source: ExtractionSourceRecord,
        feedback: LiveExtractionFeedbackRecord,
    ) -> None:
        if (
            projection.projection_digest != source.source.source_projection_digest
            or feedback.dataset.source_projection_digest != projection.projection_digest
        ):
            raise ValueError("optimizer source projection join mismatch")
        if projection.task_id != source.task_id:
            raise ValueError("optimizer source task join mismatch")
        if feedback.source_record_id != source.record_id:
            raise ValueError("optimizer feedback source record join mismatch")
        if feedback.family_id != source.family_id:
            raise ValueError("optimizer feedback family join mismatch")
        for example in feedback.dataset.examples:
            if (
                example.source_id != source.source.source_id
                or example.extraction_set_id != source.source.extraction_set_id
                or example.contract_digest != feedback.dataset.contract_digest
                or example.opportunity_operation_id
                != feedback.opportunity_operation_id
                or example.use_operation_id != feedback.use_operation_id
                or example.outcome_operation_id != feedback.outcome_operation_id
            ):
                raise ValueError("optimizer feedback example join mismatch")
        opportunity_ids = {
            example.future_opportunity_id for example in feedback.dataset.examples
        }
        if len(opportunity_ids) != 1:
            raise ValueError("optimizer feedback opportunity join mismatch")

    @staticmethod
    def _validate_observation(
        feedback: LiveExtractionFeedbackRecord,
        observation: DeploymentObservation,
        content: DelayedEvidenceContent,
    ) -> None:
        if (
            observation.observation_id != feedback.deployment_observation_id
            or observation.family_id != feedback.family_id
            or observation.stage != feedback.stage
            or observation.task_id != feedback.task_id
        ):
            raise ValueError("optimizer deployment observation join mismatch")
        if text_digest(content.current_input) != (
            observation.current_input_projection_digest
        ):
            raise ValueError("optimizer current input projection digest mismatch")

    def _project_facts(
        self,
        source: ExtractionSourceRecord,
        graph: OperationGraph,
        inputs: tuple[ExtractionFactContent, ...],
        forbidden_values: tuple[str, ...],
    ) -> tuple[OptimizerExtractedFact, ...]:
        input_ids = tuple(value.fact_id for value in inputs)
        if len(input_ids) != len(set(input_ids)):
            raise ValueError("optimizer fact content IDs must be unique")
        evidence_by_id = {value.fact_id: value for value in source.source.facts}
        if set(input_ids) != set(evidence_by_id):
            raise ValueError("optimizer fact content set differs from source evidence")
        if source.extraction_output_digest != content_digest(
            [value.trace_payload() for value in inputs]
        ):
            raise ValueError("optimizer extraction output digest mismatch")
        artifact_by_id = {value.artifact_id: value for value in graph.artifacts}
        result = []
        for value in inputs:
            evidence = evidence_by_id[value.fact_id]
            if value.accepted != (evidence.disposition != FactDisposition.FILTERED):
                raise ValueError("optimizer fact acceptance differs from source evidence")
            artifact = artifact_by_id.get(value.fact_id)
            if value.accepted:
                if (
                    artifact is None
                    or artifact.kind != ArtifactKind.EXTRACTED_FACT
                    or artifact.content_digest != text_digest(value.content)
                ):
                    raise ValueError("optimizer fact artifact join mismatch")
            elif artifact is not None:
                raise ValueError("rejected optimizer fact has an artifact node")
            result.append(OptimizerExtractedFact(
                value.fact_id,
                self.boundary.project(
                    value.content,
                    forbidden_values=forbidden_values,
                ),
                text_digest(value.content),
                value.accepted,
                value.reason_code,
                evidence.semantic_keys,
                evidence.disposition,
                evidence.artifact_id,
            ))
        return tuple(result)

    @staticmethod
    def _operation_join(
        source: ExtractionSourceRecord,
        feedback: LiveExtractionFeedbackRecord,
        graph: OperationGraph,
    ) -> tuple[str, ...]:
        by_id = {value.operation_id: value for value in graph.operations}
        if len(by_id) != len(graph.operations):
            raise ValueError("optimizer operation graph has duplicate operations")
        artifact_by_id = {value.artifact_id: value for value in graph.artifacts}
        if len(artifact_by_id) != len(graph.artifacts):
            raise ValueError("optimizer operation graph has duplicate artifacts")
        source_artifact = artifact_by_id.get(source.source.source_id)
        if (
            source_artifact is None
            or source_artifact.kind != ArtifactKind.SOURCE_OBSERVATION
            or source_artifact.content_digest
            != source.source.source_projection_digest
        ):
            raise ValueError("optimizer source artifact join mismatch")
        required = (
            source.source.extraction_set_id,
            feedback.opportunity_operation_id,
            feedback.use_operation_id,
            feedback.outcome_operation_id,
        )
        if any(value not in by_id for value in required):
            raise ValueError("optimizer operation join is incomplete")
        extraction = by_id[source.source.extraction_set_id]
        if (
            extraction.kind != OperationKind.FACT_EXTRACTION
            or extraction.context.run_id != source.run_id
            or extraction.context.episode_id != source.episode_id
            or extraction.context.session_id != source.session_id
            or extraction.context.task_id != source.task_id
            or source.source.source_id not in extraction.input_artifact_ids
        ):
            raise ValueError("optimizer extraction operation context mismatch")
        expected_kinds = (
            OperationKind.FUTURE_QUERY,
            OperationKind.USE,
            OperationKind.DOWNSTREAM_OUTCOME,
        )
        for operation_id, expected_kind in zip(required[1:], expected_kinds):
            operation = by_id[operation_id]
            context = operation.context
            if (
                operation.kind != expected_kind
                or context.run_id != feedback.run_id
                or context.episode_id != feedback.episode_id
                or context.session_id != feedback.session_id
                or context.task_id != feedback.task_id
            ):
                raise ValueError("optimizer feedback operation context mismatch")
        lineage_operations = {
            value.operation_id for value in graph.mutations
            if value.target_artifact_id in set(source.artifact_ids)
        }
        lineage_operations.update(
            parent
            for value in graph.mutations
            if value.target_artifact_id in set(source.artifact_ids)
            for parent in value.proposal_operation_ids
        )
        return tuple(dict.fromkeys((*required, *sorted(lineage_operations))))

    @staticmethod
    def _artifact_lineages(
        source: ExtractionSourceRecord,
        graph: OperationGraph,
    ) -> tuple[OptimizerArtifactLineage, ...]:
        results = []
        graph_operation_ids = {value.operation_id for value in graph.operations}
        mutation_ids = tuple(value.mutation_id for value in graph.mutations)
        if len(mutation_ids) != len(set(mutation_ids)):
            raise ValueError("optimizer operation graph has duplicate mutations")
        for artifact_id in source.artifact_ids:
            mutations = tuple(
                value for value in graph.mutations
                if value.target_artifact_id == artifact_id
            )
            if not mutations:
                raise ValueError("optimizer persisted artifact has no mutation lineage")
            if any(
                value.operation_id not in graph_operation_ids
                or not set(value.proposal_operation_ids).issubset(
                    graph_operation_ids
                )
                for value in mutations
            ):
                raise ValueError("optimizer mutation operation join is incomplete")
            digests = {
                value.after_digest or value.before_digest for value in mutations
            }
            if None in digests or len(digests) != 1:
                raise ValueError("optimizer persisted artifact digest is ambiguous")
            lineage_operation_ids = tuple(dict.fromkeys(
                value
                for mutation in mutations
                for value in (mutation.operation_id, *mutation.proposal_operation_ids)
            ))
            results.append(OptimizerArtifactLineage(
                artifact_id,
                next(iter(digests)),
                lineage_operation_ids,
                tuple(value.mutation_id for value in mutations),
            ))
        return tuple(results)

    @staticmethod
    def _ownership(
        label: ExtractionFeedbackLabel,
        reason_codes: tuple[str, ...],
    ) -> OptimizerComponentOwnership:
        if label in {
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.HARMFUL,
            ExtractionFeedbackLabel.MISSED,
        }:
            return OptimizerComponentOwnership.EXTRACTION
        reasons = set(reason_codes)
        if reasons & {"injected_not_used", "application_failure"}:
            return OptimizerComponentOwnership.APPLICATION
        if reasons & {"retrieval_failure", "not_exposed"}:
            return OptimizerComponentOwnership.RETRIEVAL
        if reasons & {"outcome_unknown", "task_incomplete"}:
            return OptimizerComponentOwnership.OUTCOME
        return OptimizerComponentOwnership.UNRESOLVED


class PureExtractionOptimizerBuilder:
    """Build generic optimizer identities from pure-process projections only."""

    def build_example(
        self,
        *,
        source: PureExtractionSourceRecord,
        feedback: PureExtractionFeedbackRecord,
    ) -> PureExtractionOptimizerExample:
        if not isinstance(source, PureExtractionSourceRecord) or not isinstance(
            feedback, PureExtractionFeedbackRecord
        ):
            raise TypeError("pure optimizer builder requires pure-process source/feedback records")
        return PureExtractionOptimizerExample.from_records(source, feedback)

    def build_examples(
        self,
        *,
        sources: tuple[PureExtractionSourceRecord, ...],
        feedback: tuple[PureExtractionFeedbackRecord, ...],
    ) -> tuple[PureExtractionOptimizerExample, ...]:
        source_by_id = {value.record_id: value for value in sources}
        if len(source_by_id) != len(sources):
            raise ValueError("pure optimizer source identities must be unique")
        result = []
        for record in feedback:
            source = source_by_id.get(record.source_record_id)
            if source is None:
                raise ValueError("pure optimizer source/feedback join is incomplete")
            result.append(self.build_example(source=source, feedback=record))
        return tuple(result)
