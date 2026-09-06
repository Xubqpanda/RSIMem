from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace

import pytest

from rsimem.native_attribution import attribute_native_observation
from rsimem.native_attribution import _digest as attribution_digest
from rsimem.native_attribution import NativeAttributionCandidate
from rsimem.native_repair_selection import (
    NativeRepairCaseList,
    NativeRepairCaseListStore,
    select_native_repair_cases,
    build_case_list_payload,
    freeze_native_repair_case_list,
)
from rsimem.native_attribution_corpus import (
    NativeAttributionCorpus, NativeAttributionCorpusStore, merge_native_attribution_corpora,
)
from rsimem.native_attribution_report import (
    assess_stage2_gate,
    build_attribution_report,
    main as report_main,
)
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


def test_corpus_merge_is_deterministic_and_rejects_overlap(tmp_path) -> None:
    corpus = _corpus(tmp_path)
    merged = merge_native_attribution_corpora((corpus,))
    assert merged.payload() == corpus.payload()
    with pytest.raises(ValueError, match="duplicate accepted runs"):
        merge_native_attribution_corpora((corpus, corpus))


def test_corpus_rejects_malformed_or_duplicate_exclusions(tmp_path) -> None:
    corpus = _corpus(tmp_path)
    with pytest.raises(ValueError, match="exclusion fields"):
        NativeAttributionCorpus.create(
            protocol_id=corpus.protocol_id,
            accepted_run_ids=corpus.accepted_run_ids,
            observations=corpus.observations,
            candidates=corpus.candidates,
            excluded_runs=({"run_id": "native-run.bad", "reason": "x", "extra": "y"},),
        )
    with pytest.raises(ValueError, match="exclusion identity"):
        NativeAttributionCorpus.create(
            protocol_id=corpus.protocol_id,
            accepted_run_ids=corpus.accepted_run_ids,
            observations=corpus.observations,
            candidates=corpus.candidates,
            excluded_runs=(
                {"run_id": "native-run.bad", "reason": "x"},
                {"run_id": "native-run.bad", "reason": "y"},
            ),
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
    assert report["surface_counts"]["unresolved"] == 5
    assert report["surface_counts"]["formation_missing"] == 0
    assert report["surface_counts"]["non_memory_failure"] == 0
    assert report["memory_kind_counts"] == {"semantic": 5}
    assert report["panel_counts"] == {"semantic": 5}
    assert len(report["case_ids_by_surface"]["unresolved"]) == 5
    assert report["case_ids_by_surface"]["formation_missing"] == []
    assert all(
        isinstance(case_id, str) and case_id
        for case_id in report["case_ids_by_surface"]["unresolved"]
    )
    assert set(report["evidence_refs_by_case"]) == {
        f"{candidate.case_id}@replicate-{candidate.replicate_id if candidate.replicate_id is not None else 'unknown'}"
        for candidate in corpus.candidates
    }
    assert report["cross_family_consistency"] is True
    assert report["excluded_reasons"] == {"usage_incomplete": 1}
    assert "final_response_text" not in json.dumps(report)
    assert report["report_id"].startswith("native-attribution-report.")
    gate = assess_stage2_gate(corpus)
    assert gate["decision"] == "STOP_NO_ACTIONABLE_SIGNAL"
    assert "unresolved_only" in gate["reasons"]
    assert "no_two_reviewer_case" in gate["reasons"]
    assert "no_high_confidence_actionable_candidate" in gate["reasons"]
    assert gate["invalid_repair_axis_count"] == 0
    assert "incomplete_evidence" not in gate["reasons"]
    assert gate["evidence_contract_valid"] is True


def test_stage2_gate_requires_two_reviewer_coverage_for_all_actionable_cases(tmp_path) -> None:
    corpus = _corpus(tmp_path)
    def make_actionable(base, surface, axis):
        values = {**base.identity_payload(), "primary_failure_surface": surface,
                  "candidate_repair_axis": axis, "is_actionable": True,
                  "confidence": "high"}
        return NativeAttributionCandidate(
            attribution_id="native-attribution." + attribution_digest(values)[:40],
            observation_id=base.observation_id, case_id=base.case_id,
            family_id=base.family_id, memory_kind=base.memory_kind,
            primary_failure_surface=surface, secondary_observations=base.secondary_observations,
            evidence_refs=base.evidence_refs, confidence="high",
            candidate_repair_axis=axis, is_actionable=True,
            review_status=base.review_status, expectation_contract_id=base.expectation_contract_id,
            replicate_id=base.replicate_id,
        )
    actionable = make_actionable(corpus.candidates[0], "formation_missing", "formation")
    second = make_actionable(corpus.candidates[1], "retrieval_missed", "retrieval")
    mixed = NativeAttributionCorpus.create(
        protocol_id=corpus.protocol_id,
        accepted_run_ids=corpus.accepted_run_ids,
        observations=corpus.observations,
        candidates=(actionable, second, *corpus.candidates[2:]),
        excluded_runs=corpus.excluded_runs,
    )
    gate = assess_stage2_gate(mixed, reviewer_two_reviewer_count=0)
    assert gate["decision"] == "STOP_NO_ACTIONABLE_SIGNAL"
    gate = assess_stage2_gate(mixed, reviewer_two_reviewer_count=1)
    assert gate["decision"] == "STOP_NO_ACTIONABLE_SIGNAL"
    assert "insufficient_two_reviewer_coverage" in gate["reasons"]
    gate = assess_stage2_gate(mixed, reviewer_two_reviewer_count=2)
    assert gate["decision"] == "OPEN_STAGE2"


def test_repair_case_selection_is_fail_closed_and_corpus_bound(tmp_path) -> None:
    corpus = _corpus(tmp_path)
    with pytest.raises(ValueError, match="no eligible"):
        select_native_repair_cases(corpus, two_reviewer_candidate_ids=())
    base = corpus.candidates[0]
    values = {**base.identity_payload(), "primary_failure_surface": "formation_missing",
              "candidate_repair_axis": "formation", "is_actionable": True, "confidence": "high",
              "evidence_refs": ["event.fixture"]}
    candidate = NativeAttributionCandidate(
        attribution_id="native-attribution." + attribution_digest(values)[:40],
        observation_id=base.observation_id, case_id=base.case_id, family_id=base.family_id,
        memory_kind=base.memory_kind, primary_failure_surface="formation_missing",
        secondary_observations=base.secondary_observations, evidence_refs=("event.fixture",),
        confidence="high", candidate_repair_axis="formation", is_actionable=True,
        review_status=base.review_status, expectation_contract_id=base.expectation_contract_id,
        replicate_id=base.replicate_id,
    )
    reviewed = NativeAttributionCorpus.create(
        protocol_id=corpus.protocol_id, accepted_run_ids=corpus.accepted_run_ids,
        observations=corpus.observations, candidates=(candidate, *corpus.candidates[1:]),
        excluded_runs=corpus.excluded_runs,
    )
    cases = select_native_repair_cases(reviewed, two_reviewer_candidate_ids=(candidate.attribution_id,))
    payload = build_case_list_payload(reviewed, cases)
    assert payload["corpus_id"] == reviewed.corpus_id
    assert payload["cases"][0]["repair_axis"] == "formation"
    case_list = NativeRepairCaseList.from_payload({
        **payload, "schema": payload["schema"],
    })
    store = NativeRepairCaseListStore(tmp_path / "repair-cases.json")
    assert store.put(case_list) is True
    assert store.put(case_list) is False
    assert store.load().payload() == case_list.payload()
    malformed = dict(case_list.payload())
    malformed["cases"] = [dict(malformed["cases"][0]), dict(malformed["cases"][0])]
    with pytest.raises(ValueError, match="duplicate"):
        NativeRepairCaseList.from_payload(malformed)


def test_freeze_native_repair_cases_requires_reviewer_store(tmp_path) -> None:
    corpus = _corpus(tmp_path)
    with pytest.raises(ValueError, match="no eligible"):
        freeze_native_repair_case_list(
            corpus=corpus,
            review_store_path=tmp_path / "missing-review.jsonl",
            output_path=tmp_path / "cases.json",
        )


def test_stage2_gate_positive_contract_requires_two_reviewer_case(tmp_path) -> None:
    corpus = _corpus(tmp_path)
    original = corpus.candidates[0]
    values = dict(original.identity_payload())
    values.update(
        primary_failure_surface="formation_missing",
        confidence="high",
        candidate_repair_axis="formation",
        is_actionable=True,
        evidence_refs=["native-lifecycle-event.fixture"],
    )
    actionable = replace(
        original,
        attribution_id="native-attribution." + attribution_digest(values)[:40],
        primary_failure_surface="formation_missing",
        confidence="high",
        candidate_repair_axis="formation",
        is_actionable=True,
        evidence_refs=("native-lifecycle-event.fixture",),
    )
    candidates = (actionable, *corpus.candidates[1:])
    positive = NativeAttributionCorpus.create(
        protocol_id=corpus.protocol_id,
        accepted_run_ids=corpus.accepted_run_ids,
        observations=corpus.observations,
        candidates=candidates,
        excluded_runs=(),
    )
    gate = assess_stage2_gate(positive, reviewer_two_reviewer_count=1)
    assert gate["decision"] == "OPEN_STAGE2"
    assert gate["high_confidence_actionable_count"] == 1


@pytest.mark.parametrize("count", (True, -1, 1.5, "1"))
def test_stage2_gate_rejects_malformed_reviewer_count(tmp_path, count) -> None:
    with pytest.raises(ValueError, match="two-reviewer candidate count"):
        assess_stage2_gate(_corpus(tmp_path), reviewer_two_reviewer_count=count)


def test_attribution_report_cli_reads_frozen_corpus(tmp_path, capsys) -> None:
    corpus = _corpus(tmp_path)
    store = NativeAttributionCorpusStore(tmp_path / "corpus.json")
    store.put(corpus)
    assert report_main([str(store.path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["corpus_id"] == corpus.corpus_id
    assert output["actionable_count"] == 0
    assert output["stage2_gate"]["decision"] == "STOP_NO_ACTIONABLE_SIGNAL"


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


def test_attribution_report_cli_binds_review_store_to_gate(tmp_path, capsys) -> None:
    corpus = _corpus(tmp_path)
    corpus_store = NativeAttributionCorpusStore(tmp_path / "corpus.json")
    corpus_store.put(corpus)
    candidate = corpus.candidates[0]
    record = NativeAttributionReviewRecord.create(
        corpus_id=corpus.corpus_id,
        candidate_id=candidate.attribution_id,
        reviewer_id="reviewer-a",
        decision=ReviewDecision.ESCALATE,
        reviewed_evidence_refs=candidate.evidence_refs or ("no_evidence",),
        rationale_codes=("insufficient_process_evidence",),
    )
    review_store = NativeAttributionReviewStore(tmp_path / "reviews.jsonl", corpus=corpus)
    review_store.append(record)
    assert report_main([str(corpus_store.path), "--review-store", str(review_store.path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["review_summary"]["reviewer_count"] == 1
    assert output["stage2_gate"]["decision"] == "STOP_NO_ACTIONABLE_SIGNAL"
    assert "no_two_reviewer_case" in output["stage2_gate"]["reasons"]


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


def test_review_packet_module_entrypoint(tmp_path) -> None:
    corpus = _corpus(tmp_path)
    store = NativeAttributionCorpusStore(tmp_path / "corpus.json")
    store.put(corpus)
    result = subprocess.run(
        [sys.executable, "-m", "rsimem.native_attribution_review", str(store.path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["corpus_id"] == corpus.corpus_id
    assert len(payload["candidates"]) == len(corpus.candidates)
    assert "final_response_text" not in result.stdout

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
