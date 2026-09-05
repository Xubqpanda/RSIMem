from __future__ import annotations

import json
import subprocess
import sys

import pytest

from rsimem.native_attribution import attribute_native_observation
from rsimem.native_attribution_corpus import NativeAttributionCorpus, NativeAttributionCorpusStore
from rsimem.native_attribution_report import build_attribution_report, main as report_main
from rsimem.native_attribution_review import (
    NativeAttributionReviewRecord,
    NativeAttributionReviewStore,
    ReviewDecision,
    build_review_packet,
    build_review_summary,
    validate_review_record,
)
from rsimem.native_observation import extract_native_observations
from test_native_execution_audit import _fixture


def _corpus(tmp_path):
    run = _fixture(tmp_path)
    observations = extract_native_observations(run=run, output_root=tmp_path)
    candidates = tuple(attribute_native_observation(value, None) for value in observations)
    return NativeAttributionCorpus.create(
        protocol_id="native-attribution-repair-v1.fixture",
        accepted_run_ids=(run.run_id,),
        observations=observations,
        candidates=candidates,
        excluded_runs=({"run_id": "native-run.excluded", "reason": "usage_incomplete"},),
    )


def test_corpus_is_deterministic_content_free_and_append_once(tmp_path) -> None:
    corpus = _corpus(tmp_path)
    assert corpus.actionable_count == 0
    assert corpus.unresolved_count == len(corpus.candidates)
    serialized = json.dumps(corpus.payload())
    assert "final_response_text" not in serialized
    store = NativeAttributionCorpusStore(tmp_path / "corpus.json")
    assert store.put(corpus) is True
    assert store.put(corpus) is False
    loaded = store.load()
    assert loaded.payload() == corpus.payload()


def test_corpus_rejects_forbidden_evaluation_fields(tmp_path) -> None:
    corpus = _corpus(tmp_path)
    with pytest.raises(ValueError, match="forbidden field"):
        NativeAttributionCorpus.create(
            protocol_id=corpus.protocol_id,
            accepted_run_ids=corpus.accepted_run_ids,
            observations=corpus.observations,
            candidates=corpus.candidates,
            excluded_runs=({"run_id": "native-run.bad", "task_score": 0.1},),
        )


def test_corpus_rejects_observation_candidate_mismatch(tmp_path) -> None:
    corpus = _corpus(tmp_path)
    with pytest.raises(ValueError, match="coverage mismatch"):
        NativeAttributionCorpus(
            corpus_id=corpus.corpus_id,
            protocol_id=corpus.protocol_id,
            accepted_run_ids=corpus.accepted_run_ids,
            observations=corpus.observations,
            candidates=corpus.candidates[:-1],
            excluded_runs=corpus.excluded_runs,
        )


def test_attribution_report_is_content_free_and_reconstructible(tmp_path) -> None:
    corpus = _corpus(tmp_path)
    report = build_attribution_report(corpus)
    assert report["observation_count"] == 5
    assert report["candidate_count"] == 5
    assert report["unresolved_count"] == 5
    assert report["actionable_count"] == 0
    assert report["unresolved_rate"] == 1.0
    assert report["actionability_rate"] == 0.0
    assert report["non_memory_exclusion_rate"] == 0.0
    assert report["attribution_coverage"] == 1.0
    assert report["surface_counts"] == {"unresolved": 5}
    assert report["memory_kind_counts"] == {"semantic": 5}
    assert report["panel_counts"] == {"semantic": 5}
    assert report["cross_family_consistency"] is True
    assert report["excluded_reasons"] == {"usage_incomplete": 1}
    assert "final_response_text" not in json.dumps(report)
    assert report["report_id"].startswith("native-attribution-report.")


def test_attribution_report_cli_reads_frozen_corpus(tmp_path, capsys) -> None:
    corpus = _corpus(tmp_path)
    store = NativeAttributionCorpusStore(tmp_path / "corpus.json")
    store.put(corpus)
    assert report_main([str(store.path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["corpus_id"] == corpus.corpus_id
    assert output["actionable_count"] == 0


def test_attribution_report_module_entrypoint(tmp_path) -> None:
    corpus = _corpus(tmp_path)
    store = NativeAttributionCorpusStore(tmp_path / "corpus.json")
    store.put(corpus)
    result = subprocess.run(
        [sys.executable, "-m", "rsimem.native_attribution_report", str(store.path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["corpus_id"] == corpus.corpus_id


def test_review_packet_is_content_free_and_review_store_is_append_once(tmp_path) -> None:
    corpus = _corpus(tmp_path)
    packet = build_review_packet(corpus)
    assert len(packet) == len(corpus.candidates)
    assert all("final_response_text" not in json.dumps(item) for item in packet)
    candidate = corpus.candidates[0]
    record = NativeAttributionReviewRecord.create(
        corpus_id=corpus.corpus_id,
        candidate_id=candidate.attribution_id,
        reviewer_id="reviewer-a",
        decision=ReviewDecision.ESCALATE,
        reviewed_evidence_refs=candidate.evidence_refs or ("no_evidence",),
        rationale_codes=("insufficient_process_evidence",),
    )
    store = NativeAttributionReviewStore(tmp_path / "reviews.jsonl", corpus=corpus)
    assert store.append(record) is True
    assert store.append(record) is False
    assert NativeAttributionReviewRecord.from_payload(record.payload()) == record
    validate_review_record(record, corpus)
    assert build_review_summary(corpus, (record,))["review_coverage"] == 0.2
    assert build_review_summary(corpus, (record,))["reviewer_count"] == 1
    assert store.load_all(corpus=corpus) == (record,)

    with pytest.raises(ValueError, match="unknown evidence"):
        validate_review_record(
            NativeAttributionReviewRecord.create(
                corpus_id=corpus.corpus_id,
                candidate_id=candidate.attribution_id,
                reviewer_id="reviewer-b",
                decision=ReviewDecision.ESCALATE,
                reviewed_evidence_refs=("made-up-ref",),
                rationale_codes=("insufficient_process_evidence",),
            ),
            corpus,
        )

    with pytest.raises(ValueError, match="not in corpus"):
        validate_review_record(
            NativeAttributionReviewRecord.create(
                corpus_id=corpus.corpus_id,
                candidate_id="native-attribution.missing",
                reviewer_id="reviewer-b",
                decision=ReviewDecision.ESCALATE,
                reviewed_evidence_refs=("no_evidence",),
                rationale_codes=("missing_candidate",),
            ),
            corpus,
        )


def test_review_record_rejects_tampered_id(tmp_path) -> None:
    corpus = _corpus(tmp_path)
    candidate = corpus.candidates[0]
    record = NativeAttributionReviewRecord.create(
        corpus_id=corpus.corpus_id,
        candidate_id=candidate.attribution_id,
        reviewer_id="reviewer-a",
        decision=ReviewDecision.CONFIRM,
        reviewed_evidence_refs=("ref",),
        rationale_codes=("evidence_complete",),
    )
    payload = dict(record.payload(), decision="reject")
    with pytest.raises(ValueError, match="ID mismatch"):
        NativeAttributionReviewRecord.from_payload(payload)


def test_corpus_store_load_fails_closed_on_tampering(tmp_path) -> None:
    corpus = _corpus(tmp_path)
    store = NativeAttributionCorpusStore(tmp_path / "corpus.json")
    store.put(corpus)
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    payload["candidates"][0]["confidence"] = "high"
    store.path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ID mismatch|canonical"):
        store.load()


def test_corpus_store_load_rejects_identity_type_coercion(tmp_path) -> None:
    corpus = _corpus(tmp_path)
    store = NativeAttributionCorpusStore(tmp_path / "corpus.json")
    store.put(corpus)
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    payload["observations"][0]["task_id"] = 42
    store.path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="scalar types"):
        store.load()
