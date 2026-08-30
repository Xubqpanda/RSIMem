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
    PureExtractionSourceProjector,
    prepare_pure_extraction_corpus,
    PureExtractionSourceRecord,
)
from rsimem.memory.opportunity import OpportunityEvidence, OpportunitySurface
from rsimem.memory.artifact_set import ArtifactSetSemanticBinding
from rsimem.memory.use_attribution import MemoryUseEvidence, OutcomeEvidenceKind
from rsimem.memory.process_signal import ProcessSignalCaseCensus
from rsimem.memory.revocation import JsonRevocationRegistry, RevocationEntry
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
    assert projected.source.available_semantic_keys == ()
    assert "family_id" not in pure.payload()
    assert "stage" not in pure.payload()
    assert PureExtractionSourceRecord.from_payload(pure.payload()) == pure

    visible = PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id=projection.projection_id,
        provenance_id="provenance.pure-v1",
        visible_semantic_keys=("preference.summary.tsv",),
    )
    assert visible.source.available_semantic_keys == ("preference.summary.tsv",)

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


def test_pure_source_and_feedback_logs_return_canonical_identity_order(tmp_path) -> None:
    projection, source, *_ = _fixture()
    first = PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id=projection.projection_id,
        provenance_id="provenance.order-a",
    )
    second = PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id="projection.order-b",
        provenance_id="provenance.order-b",
    )
    path = tmp_path / "pure-sources.jsonl"
    store = JsonPureExtractionSourceRecordStore(path)
    store.append(second)
    store.append(first)
    assert store.records() == tuple(sorted((first, second), key=lambda value: value.record_id))


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
        process_signal_gate="ready",
        process_signal_protocol_id="protocol.pure-v1",
        process_signal_case_digest="a" * 64,
        process_signal_case_count=2,
        process_signal_optimization_count=2,
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
        process_signal_gate="no_signal",
    )
    validation_path = tmp_path / "pure-validation.json"
    JsonPureExtractionOptimizerCorpusStore(validation_path).write(validation)
    with pytest.raises(PermissionError, match="training"):
        JsonPureExtractionOptimizerCorpusStore(validation_path).read_for_optimizer()

    blocked = PureExtractionOptimizerCorpus.create(
        split="train",
        observation_cutoff="2026-08-24T00:00:00Z",
        examples=(example,),
        process_signal_gate="no_signal",
    )
    blocked_path = tmp_path / "pure-blocked.json"
    JsonPureExtractionOptimizerCorpusStore(blocked_path).write(blocked)
    with pytest.raises(PermissionError, match="process-signal gate"):
        JsonPureExtractionOptimizerCorpusStore(blocked_path).read_for_optimizer()

    with pytest.raises(ValueError, match="bound optimization signal"):
        PureExtractionOptimizerCorpus.create(
            split="train",
            observation_cutoff="2026-08-24T00:00:00Z",
            examples=(example,),
            process_signal_gate="ready",
        )


def test_pure_corpus_factory_uses_census_conflicts_as_no_signal() -> None:
    projection, source, *_ = _fixture()
    pure_source = PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id=projection.projection_id,
        provenance_id="provenance.census-v1",
    )
    feedback = PureExtractionFeedbackRecord.create(
        source_record_id=pure_source.record_id,
        source_projection_digest=pure_source.source_projection_digest,
        extraction_set_id=pure_source.extraction_set_id,
        opportunity=None,
        memory_use=None,
        observation_window="window.completed-census-v1",
        provenance_id="provenance.census-v1",
    )
    example = PureExtractionOptimizerExample.from_records(pure_source, feedback)
    census = ProcessSignalCaseCensus(
        physical_observation_count=2,
        logical_case_count=1,
        status_counts={"optimization_signal": 1},
        conflict_case_count=1,
    )
    corpus = PureExtractionOptimizerCorpus.create_from_process_signal_census(
        split="train",
        observation_cutoff="2026-08-24T00:00:00Z",
        examples=(example,),
        process_signal_protocol_id="protocol.census-v1",
        process_signal_case_digest=None,
        census=census,
    )
    assert corpus.process_signal_gate == "no_signal"
    with pytest.raises(ValueError, match="case digest mismatch"):
        PureExtractionOptimizerCorpus.create_from_process_signal_census(
            split="train",
            observation_cutoff="2026-08-24T00:00:00Z",
            examples=(example,),
            process_signal_protocol_id="protocol.census-v1",
            process_signal_case_digest="f" * 64,
            census=census,
        )


def test_pure_corpus_factory_rejects_single_optimization_case() -> None:
    projection, source, *_ = _fixture()
    pure_source = PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id=projection.projection_id,
        provenance_id="provenance.census-single-v1",
    )
    feedback = PureExtractionFeedbackRecord.create(
        source_record_id=pure_source.record_id,
        source_projection_digest=pure_source.source_projection_digest,
        extraction_set_id=pure_source.extraction_set_id,
        opportunity=None,
        memory_use=None,
        observation_window="window.completed-census-single-v1",
        provenance_id="provenance.census-single-v1",
    )
    example = PureExtractionOptimizerExample.from_records(pure_source, feedback)
    census = ProcessSignalCaseCensus(
        physical_observation_count=1,
        logical_case_count=1,
        status_counts={"optimization_signal": 1},
        conflict_case_count=0,
    )
    corpus = PureExtractionOptimizerCorpus.create_from_process_signal_census(
        split="train",
        observation_cutoff="2026-08-24T00:00:00Z",
        examples=(example,),
        process_signal_protocol_id="protocol.census-single-v1",
        process_signal_case_digest=None,
        census=census,
    )
    assert corpus.process_signal_gate == "no_signal"


def test_pure_optimizer_store_honors_shared_revocation_registry(tmp_path) -> None:
    projection, source, *_ = _fixture()
    pure_source = PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id=projection.projection_id,
        provenance_id="provenance.revoke-v1",
    )
    feedback = PureExtractionFeedbackRecord.create(
        source_record_id=pure_source.record_id,
        source_projection_digest=pure_source.source_projection_digest,
        extraction_set_id=pure_source.extraction_set_id,
        opportunity=None,
        memory_use=None,
        observation_window="window.completed-revoke-v1",
        provenance_id="provenance.revoke-v1",
    )
    example = PureExtractionOptimizerExample.from_records(pure_source, feedback)
    corpus = PureExtractionOptimizerCorpus.create(
        split="train",
        observation_cutoff="2026-08-24T00:00:00Z",
        examples=(example,),
        process_signal_gate="ready",
        process_signal_protocol_id="protocol.revoke-v1",
        process_signal_case_digest="a" * 64,
        process_signal_case_count=2,
        process_signal_optimization_count=2,
    )
    corpus_store = JsonPureExtractionOptimizerCorpusStore(tmp_path / "corpus.json")
    corpus_store.write(corpus)
    registry = JsonRevocationRegistry(tmp_path / "revocations.jsonl")
    registry.initialize()
    assert corpus_store.read_for_optimizer(revocation_registry=registry) == corpus
    registry.append(RevocationEntry.create(
        artifact_id=corpus.corpus_id,
        artifact_schema_version=2,
        artifact_digest=corpus.corpus_digest,
        evidence_plane="pure_process",
        evidence_source="runtime_observation",
        revoked_at="2026-08-25T00:00:00Z",
        reason_code="stale_evidence",
    ))
    with pytest.raises(ValueError, match="revoked"):
        corpus_store.read_for_optimizer(revocation_registry=registry)


def test_prepare_pure_extraction_corpus_is_single_join_entrypoint() -> None:
    projection, source, *_ = _fixture()
    pure_source = PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id=projection.projection_id,
        provenance_id="provenance.prepare-v1",
    )
    feedback = PureExtractionFeedbackRecord.create(
        source_record_id=pure_source.record_id,
        source_projection_digest=pure_source.source_projection_digest,
        extraction_set_id=pure_source.extraction_set_id,
        opportunity=None,
        memory_use=None,
        observation_window="window.completed-prepare-v1",
        provenance_id="provenance.prepare-v1",
    )
    census = ProcessSignalCaseCensus(
        physical_observation_count=1,
        logical_case_count=1,
        status_counts={"observable_only": 1},
        conflict_case_count=0,
    )
    corpus = prepare_pure_extraction_corpus(
        sources=(pure_source,),
        feedback=(feedback,),
        split="train",
        observation_cutoff="2026-08-24T00:00:00Z",
        process_signal_protocol_id="protocol.prepare-v1",
        process_signal_case_digest=None,
        process_signal_census=census,
    )
    assert corpus.process_signal_gate == "no_signal"
    with pytest.raises(TypeError, match="tuple"):
        prepare_pure_extraction_corpus(
            sources=[pure_source],  # type: ignore[arg-type]
            feedback=(feedback,),
            split="train",
            observation_cutoff="2026-08-24T00:00:00Z",
            process_signal_protocol_id="protocol.prepare-v1",
            process_signal_case_digest=None,
            process_signal_census=census,
        )


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


def test_complete_set_binding_can_attribute_once_but_partial_set_stays_unresolved() -> None:
    projection, source, *_ = _fixture()
    pure_source = PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id=projection.projection_id,
        provenance_id="provenance.set-v1",
        visible_semantic_keys=("preference.summary.tsv",),
    )
    opportunity = OpportunityEvidence.create(
        source_surface=OpportunitySurface.TOOL_SCHEMA,
        semantic_requirement="preference.summary.tsv",
        observation_time="2026-08-22T00:00:00Z",
        operation_id="op.opportunity.set-v1",
        provenance_id="provenance.set-v1",
        source_payload={"tool": "render_table", "schema": "tsv"},
    )
    binding = ArtifactSetSemanticBinding.create(
        semantic_unit_id="semantic-unit.preference.tsv.v1",
        semantic_key="preference.summary.tsv",
        member_artifact_ids=("artifact.memory-v1",),
        member_fact_ids=("fact.preference-v1",),
        complete=True,
        source_digest=pure_source.source_projection_digest,
        provenance_id="provenance.set-v1",
    )
    use = MemoryUseEvidence.create(
        artifact_set_id=binding.binding_id,
        retrieved_artifact_ids=("artifact.memory-v1",),
        injection_operation_id="op.injection.set-v1",
        injected_artifact_ids=("artifact.memory-v1",),
        retrieval_operation_id="op.retrieval.set-v1",
        downstream_operation_id="op.use.set-v1",
        used_artifact_ids=("artifact.memory-v1",),
        outcome_operation_id="op.outcome.set-v1",
        outcome_kind=OutcomeEvidenceKind.STATE_TRANSITION,
        outcome_success=True,
        observation_cutoff="2026-08-23T00:00:00Z",
        provenance_id="provenance.set-v1",
    )
    complete = PureExtractionFeedbackRecord.derive_from_evidence(
        source=pure_source,
        opportunity=opportunity,
        memory_use=use,
        artifact_set_binding=binding,
        observation_window="window.completed-set-v1",
        provenance_id="provenance.set-v1",
    )
    assert complete.attribution is PureExtractionAttribution.ATTRIBUTABLE_SUCCESS

    partial_binding = ArtifactSetSemanticBinding.create(
        semantic_unit_id="semantic-unit.preference.tsv.partial.v1",
        semantic_key="preference.summary.tsv",
        member_artifact_ids=("artifact.memory-v1",),
        member_fact_ids=("fact.preference-v1",),
        complete=False,
        source_digest=pure_source.source_projection_digest,
        provenance_id="provenance.set-v1",
    )
    partial_use = MemoryUseEvidence.create(
        artifact_set_id=partial_binding.binding_id,
        retrieved_artifact_ids=("artifact.memory-v1",),
        injection_operation_id="op.injection.set-v1",
        injected_artifact_ids=("artifact.memory-v1",),
        retrieval_operation_id="op.retrieval.set-v1",
        downstream_operation_id="op.use.set-v1",
        used_artifact_ids=("artifact.memory-v1",),
        outcome_operation_id="op.outcome.set-v1",
        outcome_kind=OutcomeEvidenceKind.STATE_TRANSITION,
        outcome_success=True,
        observation_cutoff="2026-08-23T00:00:00Z",
        provenance_id="provenance.set-v1",
    )
    unresolved = PureExtractionFeedbackRecord.derive_from_evidence(
        source=pure_source,
        opportunity=opportunity,
        memory_use=partial_use,
        artifact_set_binding=partial_binding,
        observation_window="window.completed-set-v1",
        provenance_id="provenance.set-v1",
    )
    assert unresolved.attribution is PureExtractionAttribution.UNRESOLVED


def test_live_pure_projector_does_not_consult_family_parser(tmp_path) -> None:
    from test_extraction_projection import _compile

    runtime, boundary = _compile(tmp_path, facts=("A durable user preference.",))
    projected = PureExtractionSourceProjector().project_record(
        boundary,
        runtime.policy,
        runtime.extraction_runtime_binding,
        source_projection_id="projection.runtime-pure-v1",
        context_revision="revision.runtime-pure-v1",
        provenance_id="provenance.runtime-pure-v1",
        visible_semantic_keys=(),
    )
    assert projected.source.available_semantic_keys == ()
    assert projected.source.facts[0].semantic_keys == ()
    assert projected.source.facts[0].artifact_id is not None

    with pytest.raises(TypeError, match="mapping"):
        PureExtractionSourceProjector().project_record(
            boundary,
            runtime.policy,
            runtime.extraction_runtime_binding,
            source_projection_id="projection.runtime-pure-v1",
            context_revision="revision.runtime-pure-v1",
            provenance_id="provenance.runtime-pure-v1",
            fact_semantic_keys="not-a-map",  # type: ignore[arg-type]
        )
