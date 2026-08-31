from __future__ import annotations

import json
from dataclasses import replace

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
from rsimem.memory.opportunity import (
    ApplicationOpportunitySchema,
    JsonApplicationOpportunitySchemaRegistry,
    OpportunityEvidence,
    OpportunitySurface,
)
from rsimem.memory.artifact_set import ArtifactSetSemanticBinding
from rsimem.memory.use_attribution import MemoryUseEvidence, OutcomeEvidenceKind
from rsimem.memory.tool_exact_join import ToolCallResultJoin
from rsimem.memory.process_signal import ProcessSignalCaseCensus
from rsimem.memory.revocation import JsonRevocationRegistry, RevocationEntry
from test_extraction_optimizer_builder import _fixture


TSV_KEY = "preference.summary.tsv"


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
    assert all(not fact.semantic_keys for fact in projected.source.facts)
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
    assert visible.source.facts[0].semantic_keys == ("preference.summary.tsv",)

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


def test_pure_feedback_rejects_malformed_tool_join_collection() -> None:
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
    with pytest.raises(TypeError, match="tool joins must be a tuple"):
        replace(record, tool_joins=[])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="wrong type"):
        replace(record, tool_joins=(object(),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tool joins must be a tuple"):
        PureExtractionFeedbackRecord.create(
            source_record_id=record.source_record_id,
            source_projection_digest=record.source_projection_digest,
            extraction_set_id=record.extraction_set_id,
            opportunity=None,
            memory_use=None,
            tool_joins=[],  # type: ignore[arg-type]
            observation_window=record.observation_window,
            provenance_id=record.provenance_id,
        )


def test_pure_feedback_requires_one_provenance_across_evidence_joins() -> None:
    projection, source, *_ = _fixture()
    pure_source = PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id=projection.projection_id,
        provenance_id="provenance.join-source-v1",
    )
    opportunity = OpportunityEvidence.create(
        source_surface=OpportunitySurface.USER_REQUEST,
        semantic_requirement="preference.summary.tsv",
        observation_time="2026-08-22T00:00:00Z",
        operation_id="op.opportunity.join-v1",
        provenance_id="provenance.join-other-v1",
        source_payload={"request": "summary"},
    )
    with pytest.raises(ValueError, match="source provenance"):
        PureExtractionFeedbackRecord.derive_from_evidence(
            source=pure_source,
            opportunity=None,
            memory_use=None,
            observation_window="window.join-v1",
            provenance_id="provenance.join-other-v1",
        )
    with pytest.raises(ValueError, match="opportunity provenance"):
        PureExtractionFeedbackRecord.derive_from_evidence(
            source=pure_source,
            opportunity=opportunity,
            memory_use=None,
            observation_window="window.join-v1",
            provenance_id=pure_source.provenance_id,
        )
    with pytest.raises(ValueError, match="opportunity provenance"):
        PureExtractionFeedbackRecord.create(
            source_record_id=pure_source.record_id,
            source_projection_digest=pure_source.source_projection_digest,
            extraction_set_id=pure_source.extraction_set_id,
            opportunity=opportunity,
            memory_use=None,
            observation_window="window.join-v1",
            provenance_id=pure_source.provenance_id,
        )


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


def test_pure_feedback_store_replays_application_schema_opportunity(tmp_path) -> None:
    _, source, *_ = _fixture()
    schema = ApplicationOpportunitySchema.create(
        schema_id="application.notes",
        version="v1",
        requirement_ids=("application.notes.share",),
    )
    opportunity = OpportunityEvidence.create(
        source_surface=OpportunitySurface.APPLICATION_SCHEMA,
        semantic_requirement="application.notes.share",
        observation_time="2026-08-30T01:02:03Z",
        operation_id="op.application-notes.share",
        provenance_id="provenance.pure-v1",
        source_payload={"schema_event": "published"},
        application_schema=schema,
    )
    record = PureExtractionFeedbackRecord.create(
        source_record_id="pure-source.application-schema",
        source_projection_digest=source.source.source_projection_digest,
        extraction_set_id=source.source.extraction_set_id,
        opportunity=opportunity,
        memory_use=None,
        observation_window="window.completed-v1",
        provenance_id="provenance.pure-v1",
    )
    registry = JsonApplicationOpportunitySchemaRegistry(
        tmp_path / "application-schemas.jsonl"
    )
    assert registry.register(schema) is True
    path = tmp_path / "pure-feedback.jsonl"
    assert JsonPureExtractionFeedbackRecordStore(
        path,
        schema_registry=registry,
    ).append(record) is True
    restarted = JsonPureExtractionFeedbackRecordStore(
        path,
        schema_registry=JsonApplicationOpportunitySchemaRegistry(
            tmp_path / "application-schemas.jsonl"
        ),
    )
    assert restarted.records() == (record,)
    with pytest.raises(ValueError, match="schema registry"):
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


def test_pure_source_store_atomic_append_survives_restart_and_clone(tmp_path) -> None:
    """A copied runtime home must never inherit a half-written source line."""

    projection, source, *_ = _fixture()
    first = PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id=projection.projection_id,
        provenance_id="provenance.atomic-a",
    )
    second = PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id="projection.atomic-b",
        provenance_id="provenance.atomic-b",
    )
    path = tmp_path / "home" / ".rsimem" / "pure-sources.jsonl"
    store = JsonPureExtractionSourceRecordStore(path)
    assert store.append(first) is True

    # A fresh coordinator can continue the append-only log and a runtime
    # clone can read the result immediately, without relying on in-memory
    # state or a shared file descriptor.
    assert JsonPureExtractionSourceRecordStore(path).append(second) is True
    clone = tmp_path / "clone" / ".rsimem" / "pure-sources.jsonl"
    clone.parent.mkdir(parents=True)
    clone.write_bytes(path.read_bytes())
    assert JsonPureExtractionSourceRecordStore(clone).records() == tuple(
        sorted((first, second), key=lambda value: value.record_id)
    )

    # Replaying either record remains idempotent after the restart/clone
    # boundary and does not create duplicate lines.
    assert JsonPureExtractionSourceRecordStore(path).append(first) is False
    assert len(JsonPureExtractionSourceRecordStore(path).records()) == 2


@pytest.mark.parametrize(
    "store_type",
    (
        JsonPureExtractionSourceRecordStore,
        JsonPureExtractionFeedbackRecordStore,
        JsonPureExtractionOptimizerCorpusStore,
    ),
)
def test_pure_stores_reject_symlinked_final_paths(tmp_path, store_type) -> None:
    target = tmp_path / "target"
    target.write_text("", encoding="utf-8")
    path = tmp_path / "store"
    path.symlink_to(target)
    store = store_type(path)
    with pytest.raises(ValueError, match="symlink"):
        if store_type is JsonPureExtractionOptimizerCorpusStore:
            store.read()
        else:
            store.records()


@pytest.mark.parametrize(
    "store_type",
    (
        JsonPureExtractionSourceRecordStore,
        JsonPureExtractionFeedbackRecordStore,
        JsonPureExtractionOptimizerCorpusStore,
    ),
)
def test_pure_stores_reject_symlinked_lock_paths(tmp_path, store_type) -> None:
    path = tmp_path / "store"
    lock_target = tmp_path / "lock-target"
    lock_target.write_text("", encoding="utf-8")
    path.with_name(path.name + ".lock").symlink_to(lock_target)
    store = store_type(path)
    with pytest.raises(ValueError, match="lock.*symlink"):
        if store_type is JsonPureExtractionOptimizerCorpusStore:
            store.read()
        else:
            store.records()


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
        process_signal_hypothesis_digest="b" * 64,
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


def test_pure_corpus_factory_requires_shared_hypothesis_and_actionable_examples() -> None:
    projection, source, *_ = _fixture()
    pure_source = PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id=projection.projection_id,
        provenance_id="provenance.census-shared-v1",
    )
    feedback = PureExtractionFeedbackRecord.create(
        source_record_id=pure_source.record_id,
        source_projection_digest=pure_source.source_projection_digest,
        extraction_set_id=pure_source.extraction_set_id,
        opportunity=None,
        memory_use=None,
        observation_window="window.completed-census-shared-v1",
        provenance_id="provenance.census-shared-v1",
    )
    example = PureExtractionOptimizerExample.from_records(pure_source, feedback)
    hypothesis = "a" * 64
    census = ProcessSignalCaseCensus(
        physical_observation_count=2,
        logical_case_count=2,
        status_counts={"optimization_signal": 2},
        conflict_case_count=0,
        optimization_hypothesis_case_counts={hypothesis: 2},
    )
    corpus = PureExtractionOptimizerCorpus.create_from_process_signal_census(
        split="train",
        observation_cutoff="2026-08-24T00:00:00Z",
        examples=(example,),
        process_signal_protocol_id="protocol.census-shared-v1",
        process_signal_case_digest=None,
        census=census,
    )
    # Census metadata is not sufficient to unlock optimization.  The
    # examples themselves must contain extraction-owned attributable joins;
    # this fixture intentionally carries only unresolved feedback.
    assert corpus.process_signal_gate == "no_signal"
    assert corpus.process_signal_hypothesis_digest is None


def test_pure_corpus_factory_does_not_count_duplicate_physical_actionable_rows() -> None:
    """Repeated feedback for one source/window cannot unlock a second case."""

    projection, source, *_ = _fixture()
    pure_source = PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id=projection.projection_id,
        provenance_id="provenance.census-duplicate-actionable-v1",
        visible_semantic_keys=(TSV_KEY,),
    )
    artifact_ids = tuple(
        fact.artifact_id
        for fact in pure_source.source.facts
        if fact.artifact_id is not None
    )

    def example(tag: str) -> PureExtractionOptimizerExample:
        opportunity = OpportunityEvidence.create(
            source_surface=OpportunitySurface.TOOL_SCHEMA,
            semantic_requirement=TSV_KEY,
            observation_time="2026-08-31T00:00:00Z",
            operation_id=f"op.opportunity.duplicate-actionable.{tag}",
            provenance_id=pure_source.provenance_id,
            source_payload={"tool": "fixture_apply"},
        )
        memory_use = MemoryUseEvidence.create(
            artifact_ids=artifact_ids,
            retrieval_operation_id=f"op.retrieval.duplicate-actionable.{tag}",
            retrieved_artifact_ids=artifact_ids,
            injection_operation_id=f"op.injection.duplicate-actionable.{tag}",
            injected_artifact_ids=artifact_ids,
            downstream_operation_id=f"op.use.duplicate-actionable.{tag}",
            used_artifact_ids=artifact_ids,
            outcome_operation_id=f"op.outcome.duplicate-actionable.{tag}",
            outcome_kind=OutcomeEvidenceKind.STATE_TRANSITION,
            outcome_success=True,
            observation_cutoff="2026-08-31T00:01:00Z",
            provenance_id=pure_source.provenance_id,
        )
        feedback = PureExtractionFeedbackRecord.derive_from_evidence(
            source=pure_source,
            opportunity=opportunity,
            memory_use=memory_use,
            # Same logical source/future boundary; only physical operation
            # identities differ as they would across a replay/retry.
            observation_window="window.duplicate-actionable.v1",
            provenance_id=pure_source.provenance_id,
        )
        return PureExtractionOptimizerExample.from_records(pure_source, feedback)

    first, replay = example("first"), example("replay")
    assert first.example_id != replay.example_id
    hypothesis = "a" * 64
    census = ProcessSignalCaseCensus(
        physical_observation_count=2,
        logical_case_count=2,
        status_counts={"optimization_signal": 2},
        conflict_case_count=0,
        optimization_hypothesis_case_counts={hypothesis: 2},
    )
    corpus = PureExtractionOptimizerCorpus.create_from_process_signal_census(
        split="train",
        observation_cutoff="2026-08-31T00:00:00Z",
        examples=(first, replay),
        process_signal_protocol_id="protocol.census-duplicate-actionable-v1",
        process_signal_case_digest=None,
        census=census,
    )
    assert corpus.process_signal_gate == "no_signal"
    assert corpus.process_signal_hypothesis_digest is None


def test_pure_corpus_factory_accepts_two_independent_actionable_cases() -> None:
    """Two independently formed source sets still satisfy the ready gate."""

    def actionable_example(tag: str) -> PureExtractionOptimizerExample:
        projection, source, *_ = _fixture()
        pure_source = PureExtractionSourceRecord.from_family_record(
            source,
            source_projection_id=projection.projection_id,
            provenance_id=f"provenance.census-independent.{tag}",
            visible_semantic_keys=(TSV_KEY,),
        )
        artifact_ids = tuple(
            fact.artifact_id
            for fact in pure_source.source.facts
            if fact.artifact_id is not None
        )
        opportunity = OpportunityEvidence.create(
            source_surface=OpportunitySurface.TOOL_SCHEMA,
            semantic_requirement=TSV_KEY,
            observation_time="2026-08-31T00:00:00Z",
            operation_id=f"op.opportunity.independent.{tag}",
            provenance_id=pure_source.provenance_id,
            source_payload={"tool": "fixture_apply"},
        )
        memory_use = MemoryUseEvidence.create(
            artifact_ids=artifact_ids,
            retrieval_operation_id=f"op.retrieval.independent.{tag}",
            retrieved_artifact_ids=artifact_ids,
            injection_operation_id=f"op.injection.independent.{tag}",
            injected_artifact_ids=artifact_ids,
            downstream_operation_id=f"op.use.independent.{tag}",
            used_artifact_ids=artifact_ids,
            outcome_operation_id=f"op.outcome.independent.{tag}",
            outcome_kind=OutcomeEvidenceKind.STATE_TRANSITION,
            outcome_success=True,
            observation_cutoff="2026-08-31T00:01:00Z",
            provenance_id=pure_source.provenance_id,
        )
        feedback = PureExtractionFeedbackRecord.derive_from_evidence(
            source=pure_source,
            opportunity=opportunity,
            memory_use=memory_use,
            observation_window=f"window.independent.{tag}",
            provenance_id=pure_source.provenance_id,
        )
        return PureExtractionOptimizerExample.from_records(pure_source, feedback)

    examples = (actionable_example("one"), actionable_example("two"))
    hypothesis = "a" * 64
    census = ProcessSignalCaseCensus(
        physical_observation_count=2,
        logical_case_count=2,
        status_counts={"optimization_signal": 2},
        conflict_case_count=0,
        optimization_hypothesis_case_counts={hypothesis: 2},
    )
    corpus = PureExtractionOptimizerCorpus.create_from_process_signal_census(
        split="train",
        observation_cutoff="2026-08-31T00:00:00Z",
        examples=examples,
        process_signal_protocol_id="protocol.census-independent-v1",
        process_signal_case_digest=None,
        census=census,
    )
    assert corpus.process_signal_gate == "ready"
    assert corpus.process_signal_hypothesis_digest == hypothesis


def test_pure_corpus_factory_rejects_multiple_shared_hypotheses() -> None:
    projection, source, *_ = _fixture()
    pure_source = PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id=projection.projection_id,
        provenance_id="provenance.census-multi-v1",
    )
    feedback = PureExtractionFeedbackRecord.create(
        source_record_id=pure_source.record_id,
        source_projection_digest=pure_source.source_projection_digest,
        extraction_set_id=pure_source.extraction_set_id,
        opportunity=None,
        memory_use=None,
        observation_window="window.completed-census-multi-v1",
        provenance_id="provenance.census-multi-v1",
    )
    example = PureExtractionOptimizerExample.from_records(pure_source, feedback)
    census = ProcessSignalCaseCensus(
        physical_observation_count=4,
        logical_case_count=4,
        status_counts={"optimization_signal": 4},
        conflict_case_count=0,
        optimization_hypothesis_case_counts={"a" * 64: 2, "b" * 64: 2},
    )
    corpus = PureExtractionOptimizerCorpus.create_from_process_signal_census(
        split="train",
        observation_cutoff="2026-08-24T00:00:00Z",
        examples=(example,),
        process_signal_protocol_id="protocol.census-multi-v1",
        process_signal_case_digest=None,
        census=census,
    )
    assert corpus.process_signal_gate == "no_signal"
    assert corpus.process_signal_hypothesis_digest is None


def test_pure_corpus_rejects_inconsistent_process_signal_counts() -> None:
    projection, source, *_ = _fixture()
    pure_source = PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id=projection.projection_id,
        provenance_id="provenance.census-invalid-v1",
    )
    feedback = PureExtractionFeedbackRecord.create(
        source_record_id=pure_source.record_id,
        source_projection_digest=pure_source.source_projection_digest,
        extraction_set_id=pure_source.extraction_set_id,
        opportunity=None,
        memory_use=None,
        observation_window="window.completed-census-invalid-v1",
        provenance_id="provenance.census-invalid-v1",
    )
    example = PureExtractionOptimizerExample.from_records(pure_source, feedback)
    with pytest.raises(ValueError, match="exceeds case count"):
        PureExtractionOptimizerCorpus.create(
            split="train",
            observation_cutoff="2026-08-24T00:00:00Z",
            examples=(example,),
            process_signal_gate="no_signal",
            process_signal_protocol_id="protocol.census-invalid-v1",
            process_signal_case_digest="a" * 64,
            process_signal_case_count=1,
            process_signal_optimization_count=2,
        )
    with pytest.raises(ValueError, match="unbound.*cannot carry"):
        PureExtractionOptimizerCorpus.create(
            split="train",
            observation_cutoff="2026-08-24T00:00:00Z",
            examples=(example,),
            process_signal_gate="not_bound",
            process_signal_protocol_id="protocol.census-invalid-v1",
            process_signal_case_digest="a" * 64,
            process_signal_case_count=1,
        )


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
        process_signal_hypothesis_digest="b" * 64,
    )
    corpus_store = JsonPureExtractionOptimizerCorpusStore(tmp_path / "corpus.json")
    corpus_store.write(corpus)
    registry = JsonRevocationRegistry(tmp_path / "revocations.jsonl")
    registry.initialize()
    assert corpus_store.read_for_optimizer(revocation_registry=registry) == corpus
    registry.append(RevocationEntry.create(
        artifact_id=corpus.corpus_id,
        artifact_schema_version=3,
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


def test_public_feedback_constructor_cannot_bypass_memory_use_join() -> None:
    """Hand-built attributable labels must still satisfy the deterministic join."""

    projection, source, *_ = _fixture()
    provenance = "provenance.constructor-gate-v1"
    pure_source = PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id=projection.projection_id,
        provenance_id=provenance,
        visible_semantic_keys=(TSV_KEY,),
    )
    opportunity = OpportunityEvidence.create(
        source_surface=OpportunitySurface.TOOL_SCHEMA,
        semantic_requirement=TSV_KEY,
        observation_time="2026-08-22T00:00:00Z",
        operation_id="op.opportunity.constructor-gate-v1",
        provenance_id=provenance,
        source_payload={"tool": "render_table", "schema": "tsv"},
    )
    # This evidence claims a downstream use and successful outcome but omits
    # retrieval and injection closure.  The public constructor must reject it
    # instead of accepting a caller-supplied ATTRIBUTABLE_SUCCESS label.
    incomplete_use = MemoryUseEvidence.create(
        artifact_ids=("artifact.memory-v1",),
        downstream_operation_id="op.use.constructor-gate-v1",
        used_artifact_ids=("artifact.memory-v1",),
        outcome_operation_id="op.outcome.constructor-gate-v1",
        outcome_kind=OutcomeEvidenceKind.STATE_TRANSITION,
        outcome_success=True,
        observation_cutoff="2026-08-23T00:00:00Z",
        provenance_id=provenance,
    )
    with pytest.raises(ValueError, match="complete memory-use join"):
        PureExtractionFeedbackRecord.create(
            source_record_id=pure_source.record_id,
            source_projection_digest=pure_source.source_projection_digest,
            extraction_set_id=pure_source.extraction_set_id,
            opportunity=opportunity,
            memory_use=incomplete_use,
            observation_window="window.completed-constructor-gate-v1",
            provenance_id=provenance,
            attribution=PureExtractionAttribution.ATTRIBUTABLE_SUCCESS,
        )


def test_bound_tool_closure_is_required_for_extraction_attribution() -> None:
    """A memory-use claim cannot receive credit without an exact tool closure."""

    projection, source, *_ = _fixture()
    provenance = "provenance.tool-gate-v1"
    pure_source = PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id=projection.projection_id,
        provenance_id=provenance,
        visible_semantic_keys=(TSV_KEY,),
    )
    opportunity = OpportunityEvidence.create(
        source_surface=OpportunitySurface.TOOL_SCHEMA,
        semantic_requirement=TSV_KEY,
        observation_time="2026-08-22T00:00:00Z",
        operation_id="op.opportunity.tool-gate-v1",
        provenance_id=provenance,
        source_payload={"tool": "render_table", "schema": "tsv"},
    )
    memory_use = MemoryUseEvidence.create(
        artifact_ids=("artifact.memory-v1",),
        retrieved_artifact_ids=("artifact.memory-v1",),
        retrieval_operation_id="op.retrieval.tool-gate-v1",
        injected_artifact_ids=("artifact.memory-v1",),
        injection_operation_id="op.injection.tool-gate-v1",
        used_artifact_ids=("artifact.memory-v1",),
        downstream_operation_id="op.use.tool-gate-v1",
        outcome_operation_id="op.outcome.tool-gate-v1",
        outcome_kind=OutcomeEvidenceKind.STATE_TRANSITION,
        outcome_success=True,
        observation_cutoff="2026-08-23T00:00:00Z",
        provenance_id=provenance,
    )

    common_join = {
        "call_id": "call.tool-gate-v1",
        "result_id": "result.tool-gate-v1",
        "tool_name_digest": "a" * 64,
        "success": True,
        "retry_identity": "retry.tool-gate-v1",
        "run_id": "run.tool-gate-v1",
        "variant": "native",
        "trace_id": "trace.tool-gate-v1",
        "episode_id": "episode.tool-gate-v1",
        "session_id": "session.tool-gate-v1",
        "task_id": "task.tool-gate-v1",
        "source_revision": "revision.tool-gate-v1",
        "host_event_id": "event.tool-gate-v1",
        "memory_use_operation_id": memory_use.downstream_operation_id,
        "call_receipt_id": "receipt.call.tool-gate-v1",
        "result_receipt_id": "receipt.result.tool-gate-v1",
    }
    complete_join = ToolCallResultJoin.create(**common_join)
    complete = PureExtractionFeedbackRecord.derive_from_evidence(
        source=pure_source,
        opportunity=opportunity,
        memory_use=memory_use,
        tool_joins=(complete_join,),
        observation_window="window.completed-tool-gate-v1",
        provenance_id=provenance,
    )
    assert complete.attribution is PureExtractionAttribution.ATTRIBUTABLE_SUCCESS

    with pytest.raises(ValueError, match="complete successful tool joins"):
        PureExtractionFeedbackRecord.create(
            source_record_id=pure_source.record_id,
            source_projection_digest=pure_source.source_projection_digest,
            extraction_set_id=pure_source.extraction_set_id,
            opportunity=opportunity,
            memory_use=memory_use,
            tool_joins=(ToolCallResultJoin.create(**{**common_join, "success": False}),),
            observation_window="window.completed-tool-gate-v1",
            provenance_id=provenance,
            attribution=PureExtractionAttribution.ATTRIBUTABLE_SUCCESS,
        )

    failed_tool = ToolCallResultJoin.create(**{**common_join, "success": False})
    failed = PureExtractionFeedbackRecord.derive_from_evidence(
        source=pure_source,
        opportunity=opportunity,
        memory_use=memory_use,
        tool_joins=(failed_tool,),
        observation_window="window.completed-tool-gate-v1",
        provenance_id=provenance,
    )
    assert failed.attribution is PureExtractionAttribution.UNRESOLVED
    assert "tool_join_tool_failure" in failed.reason_codes

    censored_tool = ToolCallResultJoin.create(
        **{**common_join, "observation_complete": False}
    )
    censored = PureExtractionFeedbackRecord.derive_from_evidence(
        source=pure_source,
        opportunity=opportunity,
        memory_use=memory_use,
        tool_joins=(censored_tool,),
        observation_window="window.completed-tool-gate-v1",
        provenance_id=provenance,
    )
    assert censored.attribution is PureExtractionAttribution.CENSORED
    assert censored.observation_complete is False
    assert censored.reason_codes == ("observation_censored",)

    malformed = (
        {"result_present": False, "result_id": None, "result_receipt_id": None},
        {"orphan_result": True},
        {"duplicate_call": True},
        {"duplicate_result": True},
        {"type_mismatch": True},
        {"cross_task": True},
    )
    for overrides in malformed:
        values = dict(common_join)
        values.update(overrides)
        join = ToolCallResultJoin.create(**values)
        derived = PureExtractionFeedbackRecord.derive_from_evidence(
            source=pure_source,
            opportunity=opportunity,
            memory_use=memory_use,
            tool_joins=(join,),
            observation_window="window.completed-tool-gate-v1",
            provenance_id=provenance,
        )
        assert derived.attribution is PureExtractionAttribution.UNRESOLVED
        assert any(reason.startswith("tool_join_") for reason in derived.reason_codes)


def test_unrelated_tool_closure_does_not_gate_extraction_attribution() -> None:
    projection, source, *_ = _fixture()
    provenance = "provenance.tool-unrelated-v1"
    pure_source = PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id=projection.projection_id,
        provenance_id=provenance,
        visible_semantic_keys=(TSV_KEY,),
    )
    opportunity = OpportunityEvidence.create(
        source_surface=OpportunitySurface.TOOL_SCHEMA,
        semantic_requirement=TSV_KEY,
        observation_time="2026-08-22T00:00:00Z",
        operation_id="op.opportunity.tool-unrelated-v1",
        provenance_id=provenance,
        source_payload={"tool": "render_table", "schema": "tsv"},
    )
    memory_use = MemoryUseEvidence.create(
        artifact_ids=("artifact.memory-v1",),
        retrieved_artifact_ids=("artifact.memory-v1",),
        retrieval_operation_id="op.retrieval.tool-unrelated-v1",
        injected_artifact_ids=("artifact.memory-v1",),
        injection_operation_id="op.injection.tool-unrelated-v1",
        used_artifact_ids=("artifact.memory-v1",),
        downstream_operation_id="op.use.tool-unrelated-v1",
        outcome_operation_id="op.outcome.tool-unrelated-v1",
        outcome_kind=OutcomeEvidenceKind.STATE_TRANSITION,
        outcome_success=True,
        observation_cutoff="2026-08-23T00:00:00Z",
        provenance_id=provenance,
    )
    unrelated = ToolCallResultJoin.create(
        call_id="call.tool-unrelated-v1",
        result_id=None,
        tool_name_digest="a" * 64,
        success=None,
        retry_identity="retry.tool-unrelated-v1",
        run_id="run.tool-unrelated-v1",
        variant="native",
        trace_id="trace.tool-unrelated-v1",
        episode_id="episode.tool-unrelated-v1",
        session_id="session.tool-unrelated-v1",
        task_id="task.tool-unrelated-v1",
        source_revision="revision.tool-unrelated-v1",
        host_event_id="event.tool-unrelated-v1",
        call_receipt_id="receipt.call.tool-unrelated-v1",
        result_present=False,
    )
    derived = PureExtractionFeedbackRecord.derive_from_evidence(
        source=pure_source,
        opportunity=opportunity,
        memory_use=memory_use,
        tool_joins=(unrelated,),
        observation_window="window.completed-tool-unrelated-v1",
        provenance_id=provenance,
    )
    assert derived.attribution is PureExtractionAttribution.ATTRIBUTABLE_SUCCESS


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


def test_set_binding_must_belong_to_source_before_attribution() -> None:
    projection, source, *_ = _fixture()
    pure_source = PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id=projection.projection_id,
        provenance_id="provenance.set-ownership-v1",
        visible_semantic_keys=("preference.summary.tsv",),
    )
    opportunity = OpportunityEvidence.create(
        source_surface=OpportunitySurface.TOOL_SCHEMA,
        semantic_requirement="preference.summary.tsv",
        observation_time="2026-08-22T00:00:00Z",
        operation_id="op.opportunity.set-ownership-v1",
        provenance_id="provenance.set-ownership-v1",
        source_payload={"tool": "render_table", "schema": "tsv"},
    )
    foreign = ArtifactSetSemanticBinding.create(
        semantic_unit_id="semantic.foreign-set.v1",
        semantic_key="preference.summary.tsv",
        member_artifact_ids=("artifact.foreign-set.v1",),
        member_fact_ids=("fact.foreign-set.v1",),
        complete=True,
        source_digest=pure_source.source_projection_digest,
        provenance_id="provenance.set-ownership-v1",
    )
    use = MemoryUseEvidence.create(
        artifact_set_id=foreign.binding_id,
        retrieval_operation_id="op.retrieval.set-ownership-v1",
        retrieved_artifact_ids=foreign.member_artifact_ids,
        injection_operation_id="op.injection.set-ownership-v1",
        injected_artifact_ids=foreign.member_artifact_ids,
        downstream_operation_id="op.use.set-ownership-v1",
        used_artifact_ids=foreign.member_artifact_ids,
        outcome_operation_id="op.outcome.set-ownership-v1",
        outcome_kind=OutcomeEvidenceKind.STATE_TRANSITION,
        outcome_success=True,
        observation_cutoff="2026-08-23T00:00:00Z",
        provenance_id="provenance.set-ownership-v1",
    )
    derived = PureExtractionFeedbackRecord.derive_from_evidence(
        source=pure_source,
        opportunity=opportunity,
        memory_use=use,
        artifact_set_binding=foreign,
        observation_window="window.completed-set-ownership-v1",
        provenance_id="provenance.set-ownership-v1",
    )
    assert derived.attribution is PureExtractionAttribution.UNRESOLVED
    assert "artifact_set_member_foreign" in derived.reason_codes

    mismatched_digest = ArtifactSetSemanticBinding.create(
        semantic_unit_id="semantic.mismatched-set.v1",
        semantic_key="preference.summary.tsv",
        member_artifact_ids=("artifact.memory-v1",),
        member_fact_ids=("fact.preference-v1",),
        complete=True,
        source_digest="b" * 64,
        provenance_id="provenance.set-ownership-v1",
    )
    with pytest.raises(ValueError, match="source digest mismatch"):
        PureExtractionFeedbackRecord.derive_from_evidence(
            source=pure_source,
            opportunity=opportunity,
            memory_use=None,
            artifact_set_binding=mismatched_digest,
            observation_window="window.completed-set-ownership-v1",
            provenance_id="provenance.set-ownership-v1",
        )


def test_plain_memory_use_artifact_ownership_is_checked_before_attribution() -> None:
    projection, source, *_ = _fixture()
    provenance = "provenance.plain-ownership-v1"
    pure_source = PureExtractionSourceRecord.from_family_record(
        source,
        source_projection_id=projection.projection_id,
        provenance_id=provenance,
        visible_semantic_keys=(TSV_KEY,),
    )
    opportunity = OpportunityEvidence.create(
        source_surface=OpportunitySurface.TOOL_SCHEMA,
        semantic_requirement=TSV_KEY,
        observation_time="2026-08-22T00:00:00Z",
        operation_id="op.opportunity.plain-ownership-v1",
        provenance_id=provenance,
        source_payload={"tool": "render_table"},
    )
    foreign = MemoryUseEvidence.create(
        artifact_ids=("artifact.foreign-plain.v1",),
        retrieval_operation_id="op.retrieval.plain-ownership-v1",
        retrieved_artifact_ids=("artifact.foreign-plain.v1",),
        injection_operation_id="op.injection.plain-ownership-v1",
        injected_artifact_ids=("artifact.foreign-plain.v1",),
        downstream_operation_id="op.use.plain-ownership-v1",
        used_artifact_ids=("artifact.foreign-plain.v1",),
        outcome_operation_id="op.outcome.plain-ownership-v1",
        outcome_kind=OutcomeEvidenceKind.STATE_TRANSITION,
        outcome_success=True,
        observation_cutoff="2026-08-23T00:00:00Z",
        provenance_id=provenance,
    )
    derived = PureExtractionFeedbackRecord.derive_from_evidence(
        source=pure_source,
        opportunity=opportunity,
        memory_use=foreign,
        observation_window="window.completed-plain-ownership-v1",
        provenance_id=provenance,
    )
    assert derived.attribution is PureExtractionAttribution.UNRESOLVED
    assert derived.reason_codes == ("memory_artifact_foreign",)


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


def test_live_pure_projector_accepts_only_explicit_visible_fact_bindings(tmp_path) -> None:
    """Set-level semantics require an owner-provided per-fact mapping."""

    from test_extraction_projection import _compile

    runtime, boundary = _compile(
        tmp_path,
        facts=("A durable preference.", "A second member of the same rule."),
    )
    try:
        assert boundary.writeback is not None
        trace = runtime.policy.operation_trace(
            boundary.writeback.ingestion.idempotency_key
        )
        assert trace is not None
        fact_ids = tuple(item.fact_id for item in trace.fact_extractions)
        projected = PureExtractionSourceProjector().project_record(
            boundary,
            runtime.policy,
            runtime.extraction_runtime_binding,
            source_projection_id="projection.runtime-set-v1",
            context_revision="revision.runtime-set-v1",
            provenance_id="provenance.runtime-set-v1",
            visible_semantic_keys=(TSV_KEY,),
            fact_semantic_keys={fact_id: (TSV_KEY,) for fact_id in fact_ids},
        )
        assert tuple(
            fact.semantic_keys for fact in projected.source.facts
        ) == ((TSV_KEY,), (TSV_KEY,))

        with pytest.raises(ValueError, match="unknown fact"):
            PureExtractionSourceProjector().project_record(
                boundary,
                runtime.policy,
                runtime.extraction_runtime_binding,
                source_projection_id="projection.runtime-set-v1",
                context_revision="revision.runtime-set-v1",
                provenance_id="provenance.runtime-set-v1",
                visible_semantic_keys=(TSV_KEY,),
                fact_semantic_keys={"fact.unknown": (TSV_KEY,)},
            )
        with pytest.raises(ValueError, match="not visible"):
            PureExtractionSourceProjector().project_record(
                boundary,
                runtime.policy,
                runtime.extraction_runtime_binding,
                source_projection_id="projection.runtime-set-v1",
                context_revision="revision.runtime-set-v1",
                provenance_id="provenance.runtime-set-v1",
                visible_semantic_keys=(TSV_KEY,),
                fact_semantic_keys={fact_ids[0]: ("preference.hidden",)},
            )
    finally:
        runtime.close()
