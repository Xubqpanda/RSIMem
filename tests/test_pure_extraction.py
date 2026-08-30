from __future__ import annotations

import json

import pytest

from rsimem.memory.extraction_optimizer_builder import (
    ExtractionOptimizerCorpusBuilder,
    PureExtractionOptimizerBuilder,
)
from rsimem.memory.pure_extraction import (
    JsonPureExtractionFeedbackRecordStore,
    JsonPureExtractionOptimizerCorpusStore,
    JsonPureExtractionSourceRecordStore,
    PureExtractionAttribution,
    PureExtractionFeedbackRecord,
    PureExtractionOptimizerExample,
    PureExtractionOptimizerCorpus,
    PureExtractionSourceRecord,
)
from rsimem.memory.opportunity import OpportunityEvidence, OpportunitySurface
from rsimem.memory.use_attribution import MemoryUseEvidence, OutcomeEvidenceKind
from test_extraction_optimizer_builder import _fixture


def test_family_projection_strips_benchmark_scope_and_replays() -> None:
    projection, source, *_ = _fixture()
    pure = PureExtractionSourceRecord.create(
        source_projection_id=projection.projection_id,
        source_projection_digest=projection.projection_digest,
        context_revision=projection.context_revision,
        extraction_set_id=source.source.extraction_set_id,
        extraction_artifact_id=source.extraction_artifact_id,
        extraction_artifact_digest=source.extraction_artifact_digest,
        extraction_output_digest=source.extraction_output_digest,
        source=source.source,
        activation=source.activation,
        provenance_id="provenance.pure-v1",
    )
    projected = PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id=projection.projection_id,
        provenance_id="provenance.pure-v1",
    )
    assert projected.source_projection_digest == pure.source_projection_digest
    assert projected.extraction_set_id == pure.extraction_set_id
    assert "family_id" not in pure.payload()
    assert "stage" not in pure.payload()
    assert PureExtractionSourceRecord.from_payload(pure.payload()) == pure

    # The projection's identity is independent of the family/stage fields;
    # re-projecting the same captured record is replay-stable.
    assert PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id=projection.projection_id,
        provenance_id="provenance.pure-v1",
    ) == projected


def test_pure_feedback_requires_censored_status_for_incomplete_observation() -> None:
    _, source, *_ = _fixture()
    kwargs = dict(
        source_record_id="pure-source.v1",
        source_projection_digest=source.source.source_projection_digest,
        extraction_set_id=source.source.extraction_set_id,
        opportunity=None,
        memory_use=None,
        observation_window="window.completed-v1",
        provenance_id="provenance.pure-v1",
    )
    with pytest.raises(ValueError, match="must be censored"):
        PureExtractionFeedbackRecord.create(
            **kwargs,
            observation_complete=False,
        )
    censored = PureExtractionFeedbackRecord.create(
        **kwargs,
        attribution=PureExtractionAttribution.CENSORED,
        observation_complete=False,
    )
    assert PureExtractionFeedbackRecord.from_payload(censored.payload()) == censored


def test_pure_feedback_store_is_restart_safe_and_rejects_benchmark_fields(tmp_path) -> None:
    _, source, *_ = _fixture()
    record = PureExtractionFeedbackRecord.create(
        source_record_id="pure-source.v1",
        source_projection_digest=source.source.source_projection_digest,
        extraction_set_id=source.source.extraction_set_id,
        opportunity=None,
        memory_use=None,
        observation_window="window.completed-v1",
        provenance_id="provenance.pure-v1",
    )
    path = tmp_path / "pure-feedback.jsonl"
    store = JsonPureExtractionFeedbackRecordStore(path)
    assert store.append(record) is True
    assert store.append(record) is False
    assert JsonPureExtractionFeedbackRecordStore(path).records() == (record,)

    payload = record.payload()
    payload["family_id"] = "SM01_forbidden"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed pure extraction"):
        JsonPureExtractionFeedbackRecordStore(path).records()


def test_pure_optimizer_rejects_unbound_observation_window() -> None:
    projection, source, *_ = _fixture()
    pure_source = PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id=projection.projection_id,
    )
    feedback = PureExtractionFeedbackRecord.create(
        source_record_id=pure_source.record_id,
        source_projection_digest=pure_source.source_projection_digest,
        extraction_set_id=pure_source.extraction_set_id,
        opportunity=None,
        memory_use=None,
        observation_window="window.unbound",
        provenance_id=pure_source.provenance_id,
    )
    with pytest.raises(ValueError, match="observation window is unbound"):
        PureExtractionOptimizerExample.from_records(pure_source, feedback)


def test_pure_optimizer_corpus_is_sorted_and_restart_safe(tmp_path) -> None:
    projection, source, *_ = _fixture()
    pure_source = PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id=projection.projection_id,
        provenance_id="provenance.corpus-v1",
    )
    feedback = PureExtractionFeedbackRecord.create(
        source_record_id=pure_source.record_id,
        source_projection_digest=pure_source.source_projection_digest,
        extraction_set_id=pure_source.extraction_set_id,
        opportunity=None,
        memory_use=None,
        observation_window="window.completed-v1",
        provenance_id="provenance.corpus-v1",
    )
    example = PureExtractionOptimizerExample.from_records(pure_source, feedback)
    corpus = PureExtractionOptimizerCorpus.create(
        split="train",
        observation_cutoff="2026-08-24T00:00:00Z",
        examples=(example,),
    )
    path = tmp_path / "pure-optimizer.json"
    store = JsonPureExtractionOptimizerCorpusStore(path)
    assert store.write(corpus) is True
    assert store.write(corpus) is False
    assert store.read_for_optimizer() == corpus

    payload = corpus.payload()
    payload["schema_version"] = 0
    with pytest.raises(ValueError, match="malformed|unsupported"):
        PureExtractionOptimizerCorpus.from_payload(payload)

    validation = PureExtractionOptimizerCorpus.create(
        split="validation",
        observation_cutoff="2026-08-24T00:00:00Z",
        examples=(example,),
    )
    validation_path = tmp_path / "pure-validation.json"
    JsonPureExtractionOptimizerCorpusStore(validation_path).write(validation)
    with pytest.raises(PermissionError, match="training"):
        JsonPureExtractionOptimizerCorpusStore(validation_path).read_for_optimizer()


def test_family_bound_optimizer_builder_does_not_accept_pure_projection() -> None:
    projection, source, feedback, observation, graph, facts, delayed = _fixture()
    pure = PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id=projection.projection_id,
    )
    with pytest.raises(TypeError, match="family-bound"):
        ExtractionOptimizerCorpusBuilder().build_examples(
            projection=projection,
            source_record=pure,
            feedback_record=feedback,
            observation=observation,
            operation_graph=graph,
            fact_contents=facts,
            delayed_content=delayed,
        )


def test_family_bound_builder_cannot_relabel_audit_evidence_as_pure() -> None:
    projection, source, feedback, observation, graph, facts, delayed = _fixture()
    with pytest.raises(ValueError, match="cannot be relabeled"):
        ExtractionOptimizerCorpusBuilder(
            evidence_plane="pure_process",
        ).build_examples(
            projection=projection,
            source_record=source,
            feedback_record=feedback,
            observation=observation,
            operation_graph=graph,
            fact_contents=facts,
            delayed_content=delayed,
        )


def test_pure_optimizer_builder_joins_only_pure_records() -> None:
    projection, source, *_ = _fixture()
    pure_source = PureExtractionSourceRecord.create(
        source_projection_id=projection.projection_id,
        source_projection_digest=projection.projection_digest,
        context_revision=projection.context_revision,
        extraction_set_id=source.source.extraction_set_id,
        extraction_artifact_id=source.extraction_artifact_id,
        extraction_artifact_digest=source.extraction_artifact_digest,
        extraction_output_digest=source.extraction_output_digest,
        source=source.source,
        activation=source.activation,
        provenance_id="provenance.pure-v2",
    )
    pure_feedback = PureExtractionFeedbackRecord.create(
        source_record_id=pure_source.record_id,
        source_projection_digest=pure_source.source_projection_digest,
        extraction_set_id=pure_source.extraction_set_id,
        opportunity=None,
        memory_use=None,
        observation_window="window.completed-v2",
        provenance_id="provenance.pure-v2",
    )
    example = PureExtractionOptimizerBuilder().build_example(
        source=pure_source,
        feedback=pure_feedback,
    )
    assert example.evidence_plane.value == "pure_process"
    assert "family_id" not in example.payload()
    assert "stage" not in example.payload()
    assert PureExtractionOptimizerExample.from_payload(example.payload()) == example

    with pytest.raises(TypeError, match="pure-process"):
        PureExtractionOptimizerBuilder().build_example(
            source=source,  # type: ignore[arg-type]
            feedback=pure_feedback,
        )


def test_pure_feedback_derivation_is_conservative_and_replay_stable() -> None:
    projection, source, *_ = _fixture()
    pure_source = PureExtractionSourceRecord.create(
        source_projection_id=projection.projection_id,
        source_projection_digest=projection.projection_digest,
        context_revision=projection.context_revision,
        extraction_set_id=source.source.extraction_set_id,
        extraction_artifact_id=source.extraction_artifact_id,
        extraction_artifact_digest=source.extraction_artifact_digest,
        extraction_output_digest=source.extraction_output_digest,
        source=source.source,
        activation=source.activation,
        provenance_id="provenance.pure-v3",
    )
    opportunity = OpportunityEvidence.create(
        source_surface=OpportunitySurface.TOOL_SCHEMA,
        semantic_requirement="preference.summary.tsv",
        observation_time="2026-08-22T00:00:00Z",
        operation_id="op.opportunity.pure-v3",
        provenance_id="provenance.pure-v3",
        source_payload={"tool": "render_table", "schema": "tsv"},
    )
    use = MemoryUseEvidence.create(
        artifact_ids=("artifact.memory-v1",),
        retrieval_operation_id="op.retrieval.pure-v3",
        retrieved_artifact_ids=("artifact.memory-v1",),
        injection_operation_id="op.injection.pure-v3",
        injected_artifact_ids=("artifact.memory-v1",),
        downstream_operation_id="op.use.pure-v3",
        used_artifact_ids=("artifact.memory-v1",),
        outcome_operation_id="op.outcome.pure-v3",
        outcome_kind=OutcomeEvidenceKind.STATE_TRANSITION,
        outcome_success=True,
        observation_cutoff="2026-08-23T00:00:00Z",
        provenance_id="provenance.pure-v3",
    )
    derived = PureExtractionFeedbackRecord.derive_from_evidence(
        source=pure_source,
        opportunity=opportunity,
        memory_use=use,
        observation_window="window.completed-v3",
        provenance_id="provenance.pure-v3",
    )
    assert derived.attribution is PureExtractionAttribution.ATTRIBUTABLE_SUCCESS
    assert PureExtractionFeedbackRecord.from_payload(derived.payload()) == derived

    confounded = PureExtractionFeedbackRecord.derive_from_evidence(
        source=pure_source,
        opportunity=opportunity,
        memory_use=use,
        observation_window="window.completed-v3",
        provenance_id="provenance.pure-v3",
        current_input_requirements=("preference.summary.tsv",),
    )
    assert confounded.attribution is PureExtractionAttribution.UNRESOLVED
