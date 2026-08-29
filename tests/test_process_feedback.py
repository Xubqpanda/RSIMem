from __future__ import annotations

import json

import pytest

from rsimem.memory.policy_contracts import PolicyLayer
from rsimem.memory.policy_feasibility_fixture import (
    build_fixture_backend,
    build_fixture_snapshot,
)
from rsimem.memory.policy_replay import DeterministicPolicyReplay
from rsimem.memory.process_feedback import (
    JsonProcessFeedbackLedger,
    ProcessEvent,
    ProcessEventKind,
    ProcessEventStatus,
    audit_process_events,
)
from rsimem.memory.trigger_policy import HostTriggerAdapter


def _replay():
    snapshot = build_fixture_snapshot()
    event = HostTriggerAdapter().event(
        "task_completed",
        source_revision=snapshot.context_revision,
        payload={"snapshot_id": snapshot.snapshot_id},
        session_id=snapshot.session_id,
        task_id=snapshot.task_id,
        turn_index=2,
    )
    result = DeterministicPolicyReplay().run(
        snapshot,
        event,
        backend=build_fixture_backend(),
        candidate_fact_ids=("fact.tsv",),
        artifact_ids=("artifact.tsv",),
        mutation_ids=("mutation.tsv",),
    )
    return snapshot, event, result


def test_replay_emits_process_events_bound_to_policy_and_host() -> None:
    snapshot, event, replay = _replay()
    assert replay.process_events
    assert len(replay.process_events) == len(replay.decisions)
    for process, decision in zip(replay.process_events, replay.decisions):
        assert process.host_event_id == event.event_id
        assert process.policy_decision_id == decision.decision_id
        assert process.policy_layer is decision.layer
        assert process.source_revision == snapshot.context_revision
        assert process.lineage_id == decision.lineage_id
        assert process.event_id == ProcessEvent.from_payload(process.payload()).event_id


def test_process_ledger_is_restart_safe_and_conflict_checked(tmp_path) -> None:
    _, _, replay = _replay()
    path = tmp_path / "process.jsonl"
    first = JsonProcessFeedbackLedger(path)
    event = replay.process_events[0]
    assert first.record(event)[1] is True
    assert JsonProcessFeedbackLedger(path).record(event)[1] is False
    assert JsonProcessFeedbackLedger(path).events == (event,)

    payload = event.payload()
    payload["output_digest"] = "0" * 64
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed|conflicting"):
        JsonProcessFeedbackLedger(path).events


def test_process_reason_codes_keep_failure_stages_distinct() -> None:
    common = dict(
        run_id="run.process",
        variant="native+ledger",
        trace_id="trace.process",
        episode_id="episode.process",
        session_id="session.process",
        task_id="task.process",
        host_event_id="event.process",
        source_revision="revision.process",
        input_payload={"query": "digest-only"},
        output_payload={"count": 0},
    )
    retrieval = ProcessEvent.create(
        kind=ProcessEventKind.RETRIEVAL,
        status=ProcessEventStatus.FAILED,
        reason_codes=("retrieval_miss",),
        execution_receipt_ids=("receipt.retrieval",),
        **common,
    )
    tool = ProcessEvent.create(
        kind=ProcessEventKind.TOOL_RESULT,
        status=ProcessEventStatus.FAILED,
        reason_codes=("tool_failure",),
        execution_receipt_ids=("receipt.tool",),
        **common,
    )
    exposure = ProcessEvent.create(
        kind=ProcessEventKind.EXPOSURE,
        status=ProcessEventStatus.FAILED,
        reason_codes=("injection_failure",),
        execution_receipt_ids=("receipt.injection",),
        **common,
    )
    assert audit_process_events((retrieval, tool, exposure)) == ()

    # Constructing a new event is the public way to probe the validator's
    # stage-specific semantics because IDs are content-derived.
    bad = ProcessEvent.create(
        kind=ProcessEventKind.RETRIEVAL,
        status=ProcessEventStatus.SUCCESS,
        reason_codes=("retrieval_miss",),
        execution_receipt_ids=("receipt.retrieval",),
        **common,
    )
    errors = audit_process_events((bad,))
    assert any("retrieval miss" in error for error in errors)


def test_process_audit_checks_policy_and_host_revision_joins() -> None:
    _, event, replay = _replay()
    process = replay.process_events[0]
    errors = audit_process_events(
        (process,),
        policy_decision_ids=(),
        source_revisions={event.event_id: "revision.other"},
    )
    assert any("source revision" in error for error in errors)
