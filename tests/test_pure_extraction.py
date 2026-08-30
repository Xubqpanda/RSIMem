from __future__ import annotations

import json

import pytest

from rsimem.memory.extraction_optimizer_builder import (
    ExtractionOptimizerCorpusBuilder,
    PureExtractionOptimizerBuilder,
)
from rsimem.memory.pure_extraction import (
    JsonPureExtractionFeedbackRecordStore,
    JsonPureExtractionSourceRecordStore,
    PureExtractionAttribution,
    PureExtractionFeedbackRecord,
    PureExtractionOptimizerExample,
    PureExtractionSourceRecord,
)
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
