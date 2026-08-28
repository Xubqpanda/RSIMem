from __future__ import annotations

import json

import pytest

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
