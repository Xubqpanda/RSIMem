from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from rsimem.extraction_preparation import (
    _process_signal_gate,
    _optimizer_signal_is_ready,
    ExtractionFeedbackBatchAudit,
    audit_extraction_feedback_batch,
    build_extraction_optimizer_corpus,
)
from rsimem.memory.extraction_optimizer_corpus import PROCESS_SIGNAL_GATE_NO_SIGNAL


def test_optimizer_signal_requires_ready_pure_process_gate() -> None:
    assert not _optimizer_signal_is_ready(
        corpus_ready=True,
        actionable_primary_count=99,
        process_signal_gate="not_bound",
    )


def test_batch_audit_cannot_claim_optimizer_ready_without_ready_gate() -> None:
    with pytest.raises(ValueError, match="requires a ready process-signal gate"):
        ExtractionFeedbackBatchAudit(
            audit_id="audit.invalid-gate",
            batch_id="batch.invalid-gate",
            source_count=1,
            feedback_count=1,
            source_schema_versions=(4,),
            feedback_schema_versions=(2,),
            source_capture_count=1,
            feedback_capture_count=1,
            primary_label_counts={"unresolved": 0},
            actionable_primary_count=2,
            corpus_ready=True,
            optimizer_signal_ready=True,
            reason_codes=("test",),
        )
    assert not _optimizer_signal_is_ready(
        corpus_ready=True,
        actionable_primary_count=99,
        process_signal_gate=PROCESS_SIGNAL_GATE_NO_SIGNAL,
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
from rsimem.memory.process_signal import (
    JsonProcessSignalCaseStore,
    ProcessSignalCase,
    ProcessSignalCaseStatus,
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


def test_process_signal_case_store_without_signal_blocks_optimizer_gate(
    tmp_path: Path,
) -> None:
    case_path = tmp_path / "run" / "process_signal_cases.jsonl"
    case_path.parent.mkdir(parents=True)
    case_path.write_text("", encoding="utf-8")

    audit = audit_extraction_feedback_batch(tmp_path, batch_id="batch.no-signal-v1")

    assert audit.process_signal_gate == PROCESS_SIGNAL_GATE_NO_SIGNAL
    assert audit.process_signal_case_count == 0
    assert audit.process_signal_optimization_count == 0
    assert "no_optimization_process_signal" in audit.reason_codes
    assert audit.optimizer_signal_ready is False


def test_process_signal_gate_requires_two_logical_cases_for_one_hypothesis(
    tmp_path: Path,
) -> None:
    store = JsonProcessSignalCaseStore(tmp_path / "run" / "process_signal_cases.jsonl")

    def case(logical_case_id: str, physical_id: str, hypothesis: str) -> ProcessSignalCase:
        return ProcessSignalCase.create(
            logical_case_id=logical_case_id,
            physical_observation_ids=(physical_id,),
            source_observed=True,
            extraction_observed=True,
            persistence_observed=True,
            retrieval_observed=True,
            exposure_observed=True,
            outcome_observed=True,
            extraction_attributable=True,
            abstract_hypothesis_digest=hypothesis,
            observation_complete=True,
            analysis_protocol_id="signal-protocol.fixture",
            replicate_id=physical_id.replace("physical-observation", "replicate"),
            observation_window="window.fixture",
        )

    first = case("logical-case.signal.1", "physical-observation.signal.1", "a" * 64)
    second = case("logical-case.signal.2", "physical-observation.signal.2", "a" * 64)
    assert store.append(first) is True
    gate, protocol_id, case_digest, case_count, optimization_count, hypothesis = _process_signal_gate(tmp_path)
    assert gate == PROCESS_SIGNAL_GATE_NO_SIGNAL
    assert protocol_id == "signal-protocol.fixture"
    assert isinstance(case_digest, str) and len(case_digest) == 64
    assert (case_count, optimization_count, hypothesis) == (1, 1, None)
    audit = audit_extraction_feedback_batch(tmp_path, batch_id="batch.single-signal-v1")
    assert audit.process_signal_gate == PROCESS_SIGNAL_GATE_NO_SIGNAL
    assert audit.process_signal_optimization_count == 1
    assert store.append(second) is True
    ready = _process_signal_gate(tmp_path)
    assert ready[0] == "ready"
    assert ready[-1] == "a" * 64

    distinct = case("logical-case.signal.3", "physical-observation.signal.3", "b" * 64)
    assert store.append(distinct) is True
    gate, _, _, case_count, optimization_count, _ = _process_signal_gate(tmp_path)
    assert gate == "ready"
    assert case_count == 3
    assert optimization_count == 3


def test_process_signal_gate_deduplicates_replicates_by_logical_case(
    tmp_path: Path,
) -> None:
    store = JsonProcessSignalCaseStore(tmp_path / "run" / "process_signal_cases.jsonl")
    for index in range(3):
        case = ProcessSignalCase.create(
            logical_case_id="logical-case.same-replicate",
            physical_observation_ids=(f"physical-observation.same.{index}",),
            source_observed=True,
            extraction_observed=True,
            persistence_observed=True,
            retrieval_observed=True,
            exposure_observed=True,
            outcome_observed=True,
            extraction_attributable=True,
            abstract_hypothesis_digest="a" * 64,
            observation_complete=True,
            analysis_protocol_id="signal-protocol.replicates",
            replicate_id=f"replicate.same.{index}",
            observation_window="window.replicates",
        )
        assert store.append(case) is True
    gate, _, _, case_count, optimization_count, hypothesis = _process_signal_gate(
        tmp_path
    )
    assert gate == PROCESS_SIGNAL_GATE_NO_SIGNAL
    assert (case_count, optimization_count, hypothesis) == (1, 1, None)


def test_process_signal_gate_rejects_unbound_or_mixed_protocol_cases(
    tmp_path: Path,
) -> None:
    def case(
        logical_case_id: str,
        physical_id: str,
        protocol_id: str | None,
        replicate_id: str | None,
        window: str | None,
    ) -> ProcessSignalCase:
        return ProcessSignalCase.create(
            logical_case_id=logical_case_id,
            physical_observation_ids=(physical_id,),
            source_observed=True,
            extraction_observed=True,
            persistence_observed=True,
            retrieval_observed=True,
            exposure_observed=True,
            outcome_observed=True,
            extraction_attributable=True,
            abstract_hypothesis_digest="a" * 64,
            observation_complete=True,
            analysis_protocol_id=protocol_id,
            replicate_id=replicate_id,
            observation_window=window,
        )

    unbound_store = JsonProcessSignalCaseStore(
        tmp_path / "unbound" / "process_signal_cases.jsonl"
    )
    unbound_store.append(case(
        "logical-case.unbound",
        "physical-observation.unbound",
        None,
        None,
        None,
    ))
    with pytest.raises(ValueError, match="fully protocol bound"):
        _process_signal_gate(tmp_path / "unbound")

    mixed_store = JsonProcessSignalCaseStore(
        tmp_path / "mixed" / "process_signal_cases.jsonl"
    )
    mixed_store.append(case(
        "logical-case.mixed.1",
        "physical-observation.mixed.1",
        "signal-protocol.one",
        "replicate.1",
        "window.v1",
    ))
    mixed_store.append(case(
        "logical-case.mixed.2",
        "physical-observation.mixed.2",
        "signal-protocol.two",
        "replicate.2",
        "window.v1",
    ))
    with pytest.raises(ValueError, match="mix frozen protocols"):
        _process_signal_gate(tmp_path / "mixed")


def test_process_signal_gate_does_not_vote_over_conflicting_replicates(
    tmp_path: Path,
) -> None:
    store = JsonProcessSignalCaseStore(
        tmp_path / "conflict" / "process_signal_cases.jsonl"
    )

    def case(physical_id: str, *, attributable: bool) -> ProcessSignalCase:
        return ProcessSignalCase.create(
            logical_case_id="logical-case.conflict",
            physical_observation_ids=(physical_id,),
            source_observed=True,
            extraction_observed=True,
            persistence_observed=True,
            retrieval_observed=True,
            exposure_observed=True,
            outcome_observed=True,
            extraction_attributable=attributable,
            abstract_hypothesis_digest=("a" * 64 if attributable else None),
            observation_complete=True,
            analysis_protocol_id="signal-protocol.conflict",
            replicate_id=physical_id.replace(
                "physical-observation", "replicate"
            ),
            observation_window="window.conflict",
        )

    assert store.append(case("physical-observation.conflict.1", attributable=True))
    assert store.append(case("physical-observation.conflict.2", attributable=False))

    gate, protocol_id, case_digest, case_count, optimization_count, hypothesis = (
        _process_signal_gate(tmp_path / "conflict")
    )
    assert gate == PROCESS_SIGNAL_GATE_NO_SIGNAL
    assert protocol_id == "signal-protocol.conflict"
    assert isinstance(case_digest, str) and len(case_digest) == 64
    assert case_count == 1
    assert optimization_count == 0
    assert hypothesis is None


def test_process_signal_gate_rejects_hypothesis_conflict_within_logical_case(
    tmp_path: Path,
) -> None:
    store = JsonProcessSignalCaseStore(
        tmp_path / "hypothesis-conflict" / "process_signal_cases.jsonl"
    )

    def case(physical_id: str, hypothesis: str) -> ProcessSignalCase:
        return ProcessSignalCase.create(
            logical_case_id="logical-case.hypothesis-conflict",
            physical_observation_ids=(physical_id,),
            source_observed=True,
            extraction_observed=True,
            persistence_observed=True,
            retrieval_observed=True,
            exposure_observed=True,
            outcome_observed=True,
            extraction_attributable=True,
            abstract_hypothesis_digest=hypothesis,
            observation_complete=True,
            analysis_protocol_id="signal-protocol.hypothesis-conflict",
            replicate_id=physical_id.replace(
                "physical-observation", "replicate"
            ),
            observation_window="window.hypothesis-conflict",
        )

    first = case("physical-observation.hypothesis.1", "a" * 64)
    second = case("physical-observation.hypothesis.2", "b" * 64)
    assert store.append(first)
    assert store.append(second)
    assert second.status is ProcessSignalCaseStatus.OPTIMIZATION_SIGNAL

    gate, protocol_id, case_digest, case_count, optimization_count, hypothesis = (
        _process_signal_gate(tmp_path / "hypothesis-conflict")
    )
    assert gate == PROCESS_SIGNAL_GATE_NO_SIGNAL
    assert protocol_id == "signal-protocol.hypothesis-conflict"
    assert isinstance(case_digest, str) and len(case_digest) == 64
    assert case_count == 1
    assert optimization_count == 0
    assert hypothesis is None


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
    # The fixture intentionally carries benchmark-audit attribution.  It may
    # be persisted for diagnosis, but the optimizer read boundary must reject
    # it instead of silently treating audit labels as pure deployment signal.
    with pytest.raises(ValueError, match="optimizer requires pure_process"):
        JsonExtractionOptimizerCorpusStore(
            attempt,
            owner_controlled_root=owner,
            attempt_id="attempt.train-v1",
            split=OptimizerCorpusSplit.TRAIN,
        ).read_for_optimizer()


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
