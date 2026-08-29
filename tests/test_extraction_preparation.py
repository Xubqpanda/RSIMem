from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from rsimem.extraction_preparation import (
    audit_extraction_feedback_batch,
    build_extraction_optimizer_corpus,
)
from rsimem.memory.extraction_optimizer_capture import (
    ExtractionOptimizerFeedbackCapture,
    ExtractionOptimizerSourceCapture,
    JsonExtractionOptimizerCaptureLog,
)
from rsimem.memory.extraction_optimizer_corpus import OptimizerCorpusSplit
from rsimem.memory.operation_graph import (
    AppendOnlyOperationEvidenceLog,
    EvidenceKind,
    TracingLevel,
)
from rsimem.memory.extraction_optimizer_store import JsonExtractionOptimizerCorpusStore
from rsimem.memory.extraction_projection import (
    JsonExtractionSourceRecordStore,
    JsonLiveExtractionFeedbackRecordLog,
)
from test_extraction_optimizer_builder import _fixture


def _write_graph(path: Path, graph) -> None:
    log = AppendOnlyOperationEvidenceLog(path)
    ordinal = 0
    for kind, values, payload in (
        (EvidenceKind.ARTIFACT, graph.artifacts, lambda value: value.to_payload()),
        (EvidenceKind.OPERATION, graph.operations, lambda value: value.to_payload()),
        (EvidenceKind.MUTATION, graph.mutations, lambda value: value.to_payload()),
    ):
        for value in values:
            ordinal += 1
            log.append({
                "schemaVersion": 1,
                "eventId": f"oev.preparation-{ordinal}",
                "evidenceKind": kind.value,
                "tracingLevel": TracingLevel.MINIMAL.value,
                "payload": payload(value),
            })


def test_stage_one_style_legacy_batch_is_audited_without_fabricating_signal(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "home" / ".rsimem" / "extraction_sources.jsonl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(json.dumps({
        "schema_version": 2,
        "record_id": "source.legacy-v2",
    }) + "\n", encoding="utf-8")
    feedback_path = tmp_path / "eval" / "rsimem_extraction_feedback.jsonl"
    feedback_path.parent.mkdir(parents=True)
    feedback_path.write_text(json.dumps({
        "schema_version": 1,
        "record_id": "feedback.legacy-v1",
        "source_record_id": "source.legacy-v2",
        "dataset": {"examples": [{
            "primary": True,
            "label": "unresolved",
            "attribution_confidence": "low",
        }]},
    }) + "\n", encoding="utf-8")

    audit = audit_extraction_feedback_batch(tmp_path, batch_id="batch.legacy-v1")

    assert audit.corpus_ready is False
    assert audit.optimizer_signal_ready is False
    assert audit.actionable_primary_count == 0
    assert audit.primary_label_counts["unresolved"] == 1
    assert audit.reason_codes == (
        "source_schema_not_current",
        "feedback_schema_not_current",
        "source_optimizer_capture_missing",
        "feedback_optimizer_capture_missing",
    )


def test_build_corpus_exactly_joins_private_and_public_live_evidence(
    tmp_path: Path,
) -> None:
    projection, source, feedback, observation, graph, facts, delayed = _fixture()
    batch = tmp_path / "batch"
    source_path = batch / "home" / ".rsimem" / "extraction_sources.jsonl"
    JsonExtractionSourceRecordStore(source_path).append(source)
    feedback_path = batch / "eval" / "rsimem_extraction_feedback.jsonl"
    JsonLiveExtractionFeedbackRecordLog(feedback_path).append(feedback)
    capture_path = (
        batch / "home" / ".rsimem" / "extraction_optimizer_capture.jsonl"
    )
    captures = JsonExtractionOptimizerCaptureLog(capture_path)
    captures.append(ExtractionOptimizerSourceCapture.create(
        captured_at=delayed.source_completed_at,
        source_record_id=source.record_id,
        source_record_digest=source.content_digest,
        projection=projection,
        fact_contents=facts,
    ))
    captures.append(ExtractionOptimizerFeedbackCapture.create(
        captured_at=delayed.observed_at,
        feedback_record_id=feedback.record_id,
        source_record_id=source.record_id,
        observation=observation,
        current_input=delayed.current_input,
    ))
    _write_graph(batch / "eval" / "rsimem_semantic_operations.jsonl", graph)
    owner = tmp_path / "owner"
    attempt = owner / "attempt.train-v1"

    audit, corpus, store = build_extraction_optimizer_corpus(
        batch,
        batch_id="batch.train-v1",
        attempt_id="attempt.train-v1",
        observation_cutoff="2026-08-22T00:00:00Z",
        owner_controlled_root=owner,
        attempt_root=attempt,
    )

    assert audit.corpus_ready is True
    assert audit.optimizer_signal_ready is False
    assert audit.reason_codes == ("insufficient_actionable_extraction_signal",)
    assert audit.actionable_primary_count == 0
    assert corpus.examples
    assert sum(value.primary for value in corpus.examples) == 1
    assert store.path.stat().st_mode & 0o777 == 0o600
    replay = JsonExtractionOptimizerCorpusStore(
        attempt,
        owner_controlled_root=owner,
        attempt_id="attempt.train-v1",
        split=OptimizerCorpusSplit.TRAIN,
    ).read_for_optimizer()
    assert replay == corpus


def test_audit_rejects_incomplete_operation_join(tmp_path: Path) -> None:
    projection, source, feedback, observation, graph, facts, delayed = _fixture()
    batch = tmp_path / "batch"
    JsonExtractionSourceRecordStore(
        batch / "home" / ".rsimem" / "extraction_sources.jsonl"
    ).append(source)
    JsonLiveExtractionFeedbackRecordLog(
        batch / "eval" / "rsimem_extraction_feedback.jsonl"
    ).append(feedback)
    captures = JsonExtractionOptimizerCaptureLog(
        batch / "eval" / "extraction_optimizer_capture.jsonl"
    )
    captures.append(ExtractionOptimizerSourceCapture.create(
        captured_at=delayed.source_completed_at,
        source_record_id=source.record_id,
        source_record_digest=source.content_digest,
        projection=projection,
        fact_contents=facts,
    ))
    captures.append(ExtractionOptimizerFeedbackCapture.create(
        captured_at=delayed.observed_at,
        feedback_record_id=feedback.record_id,
        source_record_id=source.record_id,
        observation=observation,
        current_input=delayed.current_input,
    ))
    broken = replace(
        graph,
        operations=tuple(
            value for value in graph.operations
            if value.operation_id != "op.mutation-v1"
        ),
    )
    _write_graph(batch / "eval" / "rsimem_semantic_operations.jsonl", broken)

    audit = audit_extraction_feedback_batch(
        batch,
        batch_id="batch.broken-join-v1",
    )

    assert audit.corpus_ready is False
    assert audit.optimizer_signal_ready is False
    assert audit.reason_codes == ("optimizer_corpus_join_invalid",)
