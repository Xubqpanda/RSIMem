from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rsimem.memory.extraction_feedback import (
    AttributionConfidence,
    ExposureMode,
    ExtractionFeedbackLabel,
    ExtractionFeedbackLevel,
    FactDisposition,
)
from rsimem.memory.evidence_planes import EvidencePlane
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
from rsimem.memory.extraction_optimizer_store import JsonExtractionOptimizerCorpusStore
from rsimem.memory.prompt_components import content_digest, text_digest


def _join() -> OptimizerAuditJoin:
    return OptimizerAuditJoin(
        family_id="family.fixture-v1",
        source_record_id="compilation.source-v1",
        source_record_digest="1" * 64,
        source_stage="learn",
        source_run_id="run.source-v1",
        source_episode_id="episode.source-v1",
        source_session_id="session.source-v1",
        source_task_id="task.source-v1",
        source_projection_id="extraction-source.source-v1",
        source_projection_digest="2" * 64,
        feedback_record_id="live-extraction-feedback.feedback-v1",
        feedback_dataset_id="extraction-feedback.dataset-v1",
        feedback_example_id="feedback-example.example-v1",
        feedback_stage="eval",
        feedback_run_id="run.future-v1",
        feedback_trace_id="trace.future-v1",
        feedback_episode_id="episode.future-v1",
        feedback_session_id="session.future-v1",
        feedback_task_id="task.future-v1",
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
        feedback_fact_id=None,
        feedback_semantic_key=None,
        feedback_artifact_ids=("artifact.memory-v1",),
        exposure_mode=ExposureMode.EAGER_SYSTEM_PROMPT,
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
    assert ExtractionOptimizerCorpus.from_payload(
        json.loads(json.dumps(first.payload()))
    ) == first
    legacy = first.payload()
    legacy["schema_version"] = 1
    with pytest.raises(ValueError, match="malformed extraction optimizer corpus"):
        ExtractionOptimizerCorpus.from_payload(legacy)
    changed = ExtractionOptimizerCorpus.create(
        batch_id="batch.validation-v1",
        attempt_id="attempt.001",
        split=OptimizerCorpusSplit.VALIDATION,
        observation_cutoff="2026-08-21T00:00:00Z",
        retention=OptimizerCorpusRetention.DELETE_AFTER_POLICY_DECISION,
        examples=(_example(),),
    )
    assert changed.corpus_id != first.corpus_id


def test_optimizer_example_rejects_benchmark_and_final_evidence_planes() -> None:
    baseline = _example()
    for plane in (EvidencePlane.BENCHMARK_AUDIT, EvidencePlane.FINAL_EVALUATION):
        with pytest.raises(ValueError, match="optimizer requires pure_process"):
            replace(baseline, evidence_plane=plane)


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
    harmful = _example(label=ExtractionFeedbackLabel.HARMFUL)
    missed = _example(label=ExtractionFeedbackLabel.MISSED)
    assert harmful.component_ownership == OptimizerComponentOwnership.EXTRACTION
    assert missed.component_ownership == OptimizerComponentOwnership.EXTRACTION
    without_outcome = replace(
        _delayed(boundary),
        outcome_operation_id=None,
        outcome=boundary.project(""),
    )
    with pytest.raises(ValueError, match="attribution evidence"):
        _example(
            label=ExtractionFeedbackLabel.MISSED,
            delayed=without_outcome,
        )


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


def test_private_store_is_restart_safe_split_gated_and_least_privilege(tmp_path) -> None:
    corpus = ExtractionOptimizerCorpus.create(
        batch_id="batch.train-v1",
        attempt_id="attempt.001",
        split=OptimizerCorpusSplit.TRAIN,
        observation_cutoff="2026-08-21T00:00:00Z",
        retention=OptimizerCorpusRetention.DELETE_AFTER_POLICY_DECISION,
        examples=(_example(),),
    )
    attempt = tmp_path / "outputs" / "batch" / "attempt.001"
    store = JsonExtractionOptimizerCorpusStore(
        attempt,
        owner_controlled_root=tmp_path / "outputs",
        attempt_id="attempt.001",
        split=OptimizerCorpusSplit.TRAIN,
    )

    assert store.write(corpus) is True
    assert store.write(corpus) is False
    restarted = JsonExtractionOptimizerCorpusStore(
        attempt,
        owner_controlled_root=tmp_path / "outputs",
        attempt_id="attempt.001",
        split=OptimizerCorpusSplit.TRAIN,
    )
    assert restarted.read_for_optimizer() == corpus
    assert store.path.relative_to(attempt).as_posix() == (
        "private/optimizer-corpus/train.json"
    )
    assert store.path.stat().st_mode & 0o777 == 0o600
    assert store.private_root.stat().st_mode & 0o777 == 0o700
    with pytest.raises(PermissionError, match="validator"):
        restarted.read_for_validation()
    with pytest.raises(ValueError, match="retention"):
        restarted.purge(retention=OptimizerCorpusRetention.DELETE_AFTER_EXPERIMENT)
    assert restarted.purge(
        retention=OptimizerCorpusRetention.DELETE_AFTER_POLICY_DECISION
    ) is True
    assert not store.path.exists()


def test_future_store_requires_matching_activation_and_detects_corruption(tmp_path) -> None:
    future = ExtractionOptimizerCorpus.create(
        batch_id="batch.future-v1",
        attempt_id="attempt.002",
        split=OptimizerCorpusSplit.FUTURE_TEST,
        observation_cutoff="2026-08-21T00:00:00Z",
        retention=OptimizerCorpusRetention.DELETE_AFTER_EXPERIMENT,
        activation_artifact_id="extraction-prompt.candidate-v2",
        examples=(_example(),),
    )
    store = JsonExtractionOptimizerCorpusStore(
        tmp_path / "outputs" / "attempt.002",
        owner_controlled_root=tmp_path / "outputs",
        attempt_id="attempt.002",
        split=OptimizerCorpusSplit.FUTURE_TEST,
    )
    store.write(future)
    with pytest.raises(PermissionError, match="before activation"):
        store.read_for_future_evaluation(active_artifact_id=None)
    with pytest.raises(PermissionError, match="mismatch"):
        store.read_for_future_evaluation(
            active_artifact_id="extraction-prompt.other-v2"
        )
    assert store.read_for_future_evaluation(
        active_artifact_id="extraction-prompt.candidate-v2"
    ) == future

    store.path.chmod(0o644)
    with pytest.raises(PermissionError, match="permissions"):
        store.read_for_future_evaluation(
            active_artifact_id="extraction-prompt.candidate-v2"
        )
    store.path.chmod(0o600)

    payload = json.loads(store.path.read_text(encoding="utf-8"))
    payload["corpus"]["examples"][0]["source_messages"][0]["content"][
        "text"
    ] = "tampered"
    store.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed optimizer corpus store"):
        store.read_for_future_evaluation(
            active_artifact_id="extraction-prompt.candidate-v2"
        )


def test_private_store_rejects_path_outside_owner_controlled_root(tmp_path) -> None:
    with pytest.raises(ValueError, match="owner-controlled root"):
        JsonExtractionOptimizerCorpusStore(
            tmp_path / "tracked" / "attempt.003",
            owner_controlled_root=tmp_path / "outputs",
            attempt_id="attempt.003",
            split=OptimizerCorpusSplit.TRAIN,
        )
    with pytest.raises(ValueError, match="below"):
        JsonExtractionOptimizerCorpusStore(
            tmp_path / "outputs",
            owner_controlled_root=tmp_path / "outputs",
            attempt_id="attempt.003",
            split=OptimizerCorpusSplit.TRAIN,
        )
