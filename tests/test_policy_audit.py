from __future__ import annotations

import json

from rsimem.memory.policy_audit import audit_policy_evidence
from rsimem.memory.policy_contracts import TriggerDecision
from rsimem.memory.policy_evidence import JsonPolicyDecisionLedger


def _write(path, *, variant="native+ledger", trace_id="trace.fixture"):
    decision = TriggerDecision.create(
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
    JsonPolicyDecisionLedger(path, variant=variant, trace_id=trace_id).record_decision(
        decision,
        run_id="run.fixture",
        episode_id="episode.fixture",
        session_id="session.fixture",
        task_id="task.fixture",
        snapshot_id="snapshot.fixture",
    )


def test_policy_audit_checks_full_host_identity_and_layers(tmp_path) -> None:
    path = tmp_path / "policy.jsonl"
    _write(path)
    ok = audit_policy_evidence(
        path,
        run_id="run.fixture",
        variant="native+ledger",
        trace_id="trace.fixture",
        episode_id="episode.fixture",
        session_id="session.fixture",
        task_id="task.fixture",
        required_layers=("trigger",),
    )
    assert ok.ok
    assert ok.layers == ("trigger",)

    mismatch = audit_policy_evidence(path, variant="native+adapter+ledger")
    assert not mismatch.ok
    assert any("variant" in error for error in mismatch.errors)


def test_policy_audit_joins_lifecycle_snapshot_and_rejects_missing_snapshot(tmp_path) -> None:
    path = tmp_path / "policy.jsonl"
    _write(path)
    lifecycle = tmp_path / "lifecycle.jsonl"
    lifecycle.write_text(json.dumps({"kind": "context_snapshot", "snapshotId": "other.snapshot"}) + "\n", encoding="utf-8")
    report = audit_policy_evidence(path, lifecycle_events=lifecycle)
    assert not report.ok
    assert any("snapshot" in error for error in report.errors)
