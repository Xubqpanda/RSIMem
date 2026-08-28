from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rsimem.memory.extraction_feedback import (
    AttributionConfidence,
    ExtractionFeedbackLabel,
    ExtractionFeedbackLevel,
    FactDisposition,
)
from rsimem.memory.extraction_optimizer_corpus import (
    ExtractionOptimizerCorpus,
    ExtractionOptimizerCorpusExample,
    OptimizerArtifactLineage,
    OptimizerAuditJoin,
    OptimizerComponentOwnership,
    OptimizerCorpusRetention,
    OptimizerCorpusSplit,
    OptimizerDelayedEvidence,
    OptimizerExtractedFact,
    OptimizerSourceMessage,
)
from rsimem.memory.optimizer_content_boundary import OptimizerSecretBoundary
from rsimem.memory.prompt_components import content_digest, text_digest


def _join() -> OptimizerAuditJoin:
    return OptimizerAuditJoin(
        source_record_id="compilation.source-v1",
        source_record_digest="1" * 64,
        source_projection_id="extraction-source.source-v1",
        source_projection_digest="2" * 64,
        feedback_record_id="live-extraction-feedback.feedback-v1",
        feedback_dataset_id="extraction-feedback.dataset-v1",
        feedback_example_id="feedback-example.example-v1",
        extraction_artifact_id="extraction-prompt.parent-v1",
        extraction_artifact_digest="3" * 64,
        extraction_output_digest="4" * 64,
        operation_ids=("op.extract-v1", "op.opportunity-v1", "op.use-v1", "op.outcome-v1"),
        artifacts=(OptimizerArtifactLineage(
            "artifact.memory-v1",
            "5" * 64,
            ("op.mutate-v1",),
            ("mutation.persist-v1",),
        ),),
    )


def _source(boundary: OptimizerSecretBoundary) -> tuple[OptimizerSourceMessage, ...]:
    return (OptimizerSourceMessage(
        "segment.user-v1",
        "message.user-v1",
        "user",
        "message",
        None,
        False,
        boundary.project("I prefer concise durable status summaries."),
    ),)


def _facts(boundary: OptimizerSecretBoundary) -> tuple[OptimizerExtractedFact, ...]:
    text = "The user prefers concise durable status summaries."
    return (OptimizerExtractedFact(
        "fact.preference-v1",
        boundary.project(text),
        text_digest(text),
        True,
        None,
        ("preference.status.concise",),
        FactDisposition.PERSISTED,
        "artifact.memory-v1",
    ),)


def _delayed(
    boundary: OptimizerSecretBoundary,
    *,
    observed_at: str = "2026-08-20T12:00:00Z",
) -> OptimizerDelayedEvidence:
    return OptimizerDelayedEvidence(
        "observation.future-v1",
        "2026-08-19T12:00:00Z",
        observed_at,
        "opportunity.future-v1",
        "op.opportunity-v1",
        "op.use-v1",
        "op.outcome-v1",
        boundary.project("A future task required the stored status preference."),
        boundary.project("The response applied the concise status preference."),
        boundary.project("The future task completed successfully."),
    )


def _example(
    *,
    delayed: OptimizerDelayedEvidence | None = None,
    label: ExtractionFeedbackLabel = ExtractionFeedbackLabel.USEFUL,
    ownership: OptimizerComponentOwnership = OptimizerComponentOwnership.EXTRACTION,
) -> ExtractionOptimizerCorpusExample:
    boundary = OptimizerSecretBoundary()
    return ExtractionOptimizerCorpusExample.create(
        primary_unit_id="feedback-unit.primary-v1",
        level=ExtractionFeedbackLevel.EXTRACTION_SET,
        primary=True,
        label=label,
        attribution_confidence=AttributionConfidence.HIGH,
        reason_codes=("explicit_memory_use", "successful_outcome"),
        component_ownership=ownership,
        audit_join=_join(),
        source_messages=_source(boundary),
        extracted_facts=_facts(boundary),
        delayed_evidence=delayed or _delayed(boundary),
    )


def test_corpus_identity_is_canonical_and_covers_content_and_split() -> None:
    first = ExtractionOptimizerCorpus.create(
        batch_id="batch.train-v1",
        attempt_id="attempt.001",
        split=OptimizerCorpusSplit.TRAIN,
        observation_cutoff="2026-08-21T00:00:00Z",
        retention=OptimizerCorpusRetention.DELETE_AFTER_POLICY_DECISION,
        examples=(_example(),),
    )
    second = ExtractionOptimizerCorpus.create(
        batch_id="batch.train-v1",
        attempt_id="attempt.001",
        split=OptimizerCorpusSplit.TRAIN,
        observation_cutoff="2026-08-21T00:00:00Z",
        retention=OptimizerCorpusRetention.DELETE_AFTER_POLICY_DECISION,
        examples=(_example(),),
    )

    assert first == second
    assert first.corpus_digest == content_digest(first.identity_payload())
    assert first.examples[0].audit_join.feedback_example_id == (
        "feedback-example.example-v1"
    )
    assert first.examples[0].source_messages[0].content.trust == "untrusted_data"
    changed = ExtractionOptimizerCorpus.create(
        batch_id="batch.validation-v1",
        attempt_id="attempt.001",
        split=OptimizerCorpusSplit.VALIDATION,
        observation_cutoff="2026-08-21T00:00:00Z",
        retention=OptimizerCorpusRetention.DELETE_AFTER_POLICY_DECISION,
        examples=(_example(),),
    )
    assert changed.corpus_id != first.corpus_id


def test_resolved_labels_and_useful_three_stage_evidence_fail_closed() -> None:
    boundary = OptimizerSecretBoundary()
    with pytest.raises(ValueError, match="extraction-owned"):
        _example(ownership=OptimizerComponentOwnership.APPLICATION)
    incomplete = replace(
        _delayed(boundary),
        use_operation_id=None,
        use=boundary.project(""),
    )
    with pytest.raises(ValueError, match="three-stage evidence"):
        _example(delayed=incomplete)


def test_future_dates_split_activation_and_tampering_fail_closed() -> None:
    with pytest.raises(ValueError, match="future-dated"):
        ExtractionOptimizerCorpus.create(
            batch_id="batch.train-v1",
            attempt_id="attempt.001",
            split=OptimizerCorpusSplit.TRAIN,
            observation_cutoff="2026-08-20T00:00:00Z",
            retention=OptimizerCorpusRetention.DELETE_AFTER_POLICY_DECISION,
            examples=(_example(),),
        )
    with pytest.raises(ValueError, match="activation identity"):
        ExtractionOptimizerCorpus.create(
            batch_id="batch.future-v1",
            attempt_id="attempt.001",
            split=OptimizerCorpusSplit.FUTURE_TEST,
            observation_cutoff="2026-08-21T00:00:00Z",
            retention=OptimizerCorpusRetention.DELETE_AFTER_EXPERIMENT,
            examples=(_example(),),
        )
    future = ExtractionOptimizerCorpus.create(
        batch_id="batch.future-v1",
        attempt_id="attempt.001",
        split=OptimizerCorpusSplit.FUTURE_TEST,
        observation_cutoff="2026-08-21T00:00:00Z",
        retention=OptimizerCorpusRetention.DELETE_AFTER_EXPERIMENT,
        activation_artifact_id="extraction-prompt.candidate-v2",
        examples=(_example(),),
    )
    with pytest.raises(ValueError, match="corpus digest"):
        replace(future, corpus_digest="f" * 64)
    serialized = json.dumps(future.payload(), ensure_ascii=True, sort_keys=True)
    assert "I prefer concise durable status summaries" in serialized
    assert "official_grader" not in serialized
    assert "answer_key" not in serialized
    assert "hidden_expectation" not in serialized
