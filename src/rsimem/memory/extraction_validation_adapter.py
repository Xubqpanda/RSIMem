"""Mem0-flat/Hermes evidence adapter for host-neutral extraction validation."""

from __future__ import annotations

from .extraction_feedback import ExtractionFeedbackLabel
from .extraction_projection import (
    ExtractionSourceRecord,
    LiveExtractionFeedbackRecord,
)
from .extraction_prompt_validation import (
    ExtractionValidationObservation,
    ExtractionValidationSafetyEvidence,
    ExtractionValidationVariant,
)


class ExtractionValidationObservationAssembler:
    """Join immutable live/source evidence into one validation observation."""

    def assemble(
        self,
        *,
        live_feedback: LiveExtractionFeedbackRecord,
        source: ExtractionSourceRecord,
        safety: ExtractionValidationSafetyEvidence,
        variant: ExtractionValidationVariant,
        replicate: int,
        task_template_group_id: str,
        task_manifest_digest: str,
        model_profile_digest: str,
        budget_id: str,
        persistence_state_digest: str,
    ) -> ExtractionValidationObservation:
        if live_feedback.source_record_id != source.record_id:
            raise ValueError("live feedback and extraction source record differ")
        if live_feedback.family_id != source.family_id:
            raise ValueError("live feedback and extraction source family differ")
        if (
            live_feedback.dataset.source_projection_digest
            != source.source.source_projection_digest
        ):
            raise ValueError("live feedback and source projection digest differ")
        if safety.live_feedback_record_id != live_feedback.record_id or (
            safety.source_record_id != source.record_id
        ):
            raise ValueError("validation safety evidence join mismatch")
        if not safety.complete:
            raise ValueError("validation safety audit is incomplete")
        examples = live_feedback.dataset.examples
        if any(
            example.source_id != source.source.source_id
            or example.extraction_set_id != source.source.extraction_set_id
            for example in examples
        ):
            raise ValueError("live feedback examples do not belong to source record")
        primary = next((example for example in examples if example.primary), None)
        if primary is None:
            raise ValueError("live feedback dataset has no primary example")
        return ExtractionValidationObservation.create(
            variant=variant,
            replicate=replicate,
            family_id=live_feedback.family_id,
            task_template_group_id=task_template_group_id,
            task_id=live_feedback.task_id,
            run_id=live_feedback.run_id,
            episode_id=live_feedback.episode_id,
            extraction_set_id=source.source.extraction_set_id,
            task_manifest_digest=task_manifest_digest,
            model_profile_digest=model_profile_digest,
            budget_id=budget_id,
            persistence_state_digest=persistence_state_digest,
            extraction_artifact_id=source.extraction_artifact_id,
            extraction_artifact_digest=source.extraction_artifact_digest,
            extraction_output_digest=source.extraction_output_digest,
            label=primary.label,
            extraction_status=source.source.status,
            missed_assessable=(
                True
                if primary.label == ExtractionFeedbackLabel.MISSED
                else None
            ),
            failure_counts=safety.failure_counts,
        )
