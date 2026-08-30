from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from rsimem.memory.evidence_planes import EvidencePlane, EvidenceSourceKind
from rsimem.memory.policy_contracts import TriggerDecision
from rsimem.memory.policy_evidence import JsonPolicyDecisionLedger, PolicyDecisionEvidence


def _decision() -> TriggerDecision:
    return TriggerDecision.create(
        policy_version="fixed.trigger.parent.v1",
        source_revision="snapshot.rev.1",
        input_payload={"event": "event.fixture"},
        output_payload={"action": "RUN"},
        action="RUN",
        execution_status="pending",
        reason_codes=("task_completed_parent",),
        lineage_id="lineage.fixture",
        trigger_event_id="event.fixture",
    )


def _record(ledger: JsonPolicyDecisionLedger):
    return ledger.record_decision(
        _decision(),
        run_id="run.fixture",
        episode_id="episode.fixture",
        session_id="session.fixture",
        task_id="task.fixture",
        snapshot_id="snapshot.fixture",
    )


def test_policy_evidence_is_content_free_and_idempotent_across_restart(tmp_path) -> None:
    path = tmp_path / "policy.jsonl"
    first = _record(JsonPolicyDecisionLedger(path))
    second = _record(JsonPolicyDecisionLedger(path))
    assert first == second
    payload = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(payload) == 1
    assert "memory content" not in json.dumps(payload)
    assert payload[0]["decisionId"] == first.decision_id


def test_policy_evidence_conflict_and_malformed_entry_fail_closed(tmp_path) -> None:
    path = tmp_path / "policy.jsonl"
    evidence = _record(JsonPolicyDecisionLedger(path))
    altered = dict(evidence.payload())
    altered["outputDigest"] = "0" * 64
    path.write_text(json.dumps(altered) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed|conflicting"):
        JsonPolicyDecisionLedger(path)


def test_evidence_payload_round_trips() -> None:
    decision = _decision()
    evidence = PolicyDecisionEvidence.from_decision(
        decision,
        run_id="run.fixture", episode_id="episode.fixture", session_id="session.fixture",
        task_id="task.fixture", snapshot_id="snapshot.fixture",
        mutation_receipt_ids=("receipt.mutation",),
    )
    assert PolicyDecisionEvidence.from_payload(evidence.payload()) == evidence


def test_policy_evidence_plane_and_source_are_explicit(tmp_path) -> None:
    pure = _record(JsonPolicyDecisionLedger(tmp_path / "policy-plane.jsonl"))
    assert pure.evidence_plane is EvidencePlane.PURE_PROCESS
    assert pure.evidence_source is EvidenceSourceKind.RUNTIME_OBSERVATION

    audited = JsonPolicyDecisionLedger(
        tmp_path / "policy-audit-plane.jsonl",
        family_id="SM02_constraint_retention",
        stage="eval_near",
    ).record_decision(
        _decision(),
        run_id="run.audit-plane",
        episode_id="episode.audit-plane",
        session_id="session.audit-plane",
        task_id="task.audit-plane",
        snapshot_id="snapshot.audit-plane",
    )
    assert audited.evidence_plane is EvidencePlane.BENCHMARK_AUDIT
    assert audited.evidence_source is EvidenceSourceKind.BENCHMARK_CONTRACT

    with pytest.raises(ValueError, match="plane and source identity"):
        replace(pure, evidence_plane=EvidencePlane.BENCHMARK_AUDIT)


def test_policy_evidence_ledger_reserves_one_concurrent_writer(tmp_path) -> None:
    path = tmp_path / "policy-concurrent.jsonl"

    def record_once() -> PolicyDecisionEvidence:
        return _record(JsonPolicyDecisionLedger(path))

    with ThreadPoolExecutor(max_workers=8) as executor:
        records = tuple(executor.map(lambda _: record_once(), range(8)))
    assert len({item.event_id for item in records}) == 1
    assert len(JsonPolicyDecisionLedger(path).events) == 1


def test_policy_evidence_ledger_rejects_symlinked_paths(tmp_path) -> None:
    target = tmp_path / "target.jsonl"
    target.write_text("sentinel\n", encoding="utf-8")
    path = tmp_path / "policy.jsonl"
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        JsonPolicyDecisionLedger(path)
    assert target.read_text(encoding="utf-8") == "sentinel\n"


def test_policy_evidence_ledger_rejects_symlinked_lock(tmp_path) -> None:
    path = tmp_path / "policy.jsonl"
    lock_target = tmp_path / "lock-target"
    lock_target.write_text("", encoding="utf-8")
    path.with_name(path.name + ".lock").symlink_to(lock_target)
    with pytest.raises(ValueError, match="lock.*symlink"):
        JsonPolicyDecisionLedger(path)
