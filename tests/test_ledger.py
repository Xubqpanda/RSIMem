from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsimem.audit import audit_run
from rsimem.ledger import (
    LifecycleLedgerObserver,
    MemoryLedgerObserver,
    build_events,
    load_episode_lifecycle_events,
    write_ledger,
)
from rsimem.lifecycle import (
    HermesMessage,
    HermesSnapshotCollector,
    TaskLifecycleState,
    RawResourceUsage,
    run_sm01_preference_fixture,
)
from rsimem.memory import MemoryEvent, MemoryEventKind, MemoryKind
from rsimem.memory.policy_contracts import TriggerDecision
from rsimem.memory.policy_evidence import JsonPolicyDecisionLedger
from rsimem.memory.process_feedback import ProcessEvent, ProcessEventKind, ProcessEventStatus
from rsimem.memory.ingestion import (
    InternalMemoryAction,
    MemoryIngestOutcome,
    MemoryIngestStatus,
)


MEMORY = "Use TSV with owner, priority, task, and due_date."


def _runtime_evidence_path(comparison: Path) -> Path:
    data = json.loads(comparison.read_text(encoding="utf-8"))
    trace = Path(data["with_persistence"]["episodes"][0]["trace"])
    return trace.parent / "artifacts" / "rsimem_memory_events.jsonl"


def _lifecycle_evidence_path(comparison: Path) -> Path:
    payload = json.loads(comparison.read_text(encoding="utf-8"))
    trace = Path(payload["with_persistence"]["episodes"][0]["trace"])
    return trace.parent / "artifacts" / "rsimem_lifecycle_events.jsonl"


def _fixture(tmp_path: Path) -> Path:
    run = tmp_path / "run-1"
    episode_dir = run / "with_persistence" / "learn"
    artifacts = episode_dir / "artifacts"
    artifacts.mkdir(parents=True)
    trace = episode_dir / "trace.jsonl"
    trace_events = [{
        "type": "trace_start",
        "trace_id": "trace-1",
        "task_id": "task-1",
        "model": "gpt-test",
    }, {
        "type": "model_call_usage",
        "trace_id": "trace-1",
        "call_id": "model-call-0001",
        "sequence": 1,
        "component": "agent",
        "purpose": "task_execution",
        "provider": "test-provider",
        "model": "gpt-test",
        "api_mode": "codex_responses",
        "attempt": 1,
        "status": "success",
        "usage": {
            "input_tokens": 80,
            "output_tokens": 20,
            "cache_read_tokens": 20,
            "cache_write_tokens": 0,
            "reasoning_tokens": 5,
        },
        "usage_available": True,
        "duration_ms": 125.0,
        "http_status": None,
        "error_category": None,
    }, {
        "type": "trace_end",
        "trace_id": "trace-1",
        "model_input_tokens": 80,
        "model_output_tokens": 20,
        "input_tokens": 80,
        "output_tokens": 20,
        "total_tokens": 100,
        "cache_read_tokens": 20,
        "cache_write_tokens": 0,
        "reasoning_tokens": 5,
        "model_request_count": 1,
        "model_retry_count": 0,
        "model_usage_complete": True,
    }]
    trace.write_text(
        "".join(json.dumps(event) + "\n" for event in trace_events),
        encoding="utf-8",
    )
    session = artifacts / "session_current.json"
    session.write_text(json.dumps({
        "model": "gpt-test",
        "system_prompt": f"Memory:\n{MEMORY}",
        "messages": [
            {"role": "user", "content": "task"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "memory"}}]},
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ],
    }), encoding="utf-8")
    memories = artifacts / "memories"
    memories.mkdir()
    (memories / "MEMORY.md").write_text(MEMORY, encoding="utf-8")
    episode = {
        "trace": str(trace),
        "trace_id": "trace-1",
        "task_id": "task-1",
        "family_id": "family-1",
        "stage": "learn",
        "bucket": "learn",
        "task_score": 1.0,
        "passed": True,
        "judge_score": 0.0,
        "token_usage": {
            "input_tokens": 80,
            "output_tokens": 20,
            "total_tokens": 100,
            "cache_read_tokens": 20,
            "cache_write_tokens": 0,
            "reasoning_tokens": 5,
            "model_request_count": 1,
            "model_retry_count": 0,
            "model_usage_complete": True,
        },
        "timing": {"model_time_s": 2.0, "wall_time_s": 3.0},
        "internal_tools": {
            "session_file": str(session),
            "calls": [{
                "name": "memory",
                "args": {"action": "add", "target": "memory", "content": MEMORY},
                "message_index": 1,
            }],
        },
        "retrieval_signals": {"memory_injection_count": 1},
        "artifacts": {
            "memory_chars": len(MEMORY),
            "user_chars": 0,
            "memory_entries": [MEMORY],
            "skill_count": 0,
        },
    }
    comparison = run / "sequence_comparison.json"
    comparison.write_text(json.dumps({
        "with_persistence": {"episodes": [episode]},
        "without_persistence": {"episodes": []},
        "delta": {},
    }), encoding="utf-8")
    return comparison


def _write_runtime_evidence(
    comparison: Path,
    *,
    overrides: dict | None = None,
    repeat: int = 1,
) -> dict:
    observer = MemoryLedgerObserver(
        run_id=comparison.parent.name,
        variant="with_persistence",
        trace_id="trace-1",
        episode_id="learn",
        session_id="session-1",
        task_id="task-1",
        family_id="family-1",
        stage="learn",
        execution_mode="native+adapter+ledger",
    )
    observer.record(MemoryEvent(
        MemoryEventKind.QUERY,
        MemoryKind.SEMANTIC,
        "hermes-native-semantic",
        query_chars=12,
        attributes={"limit": 5, "namespace": "default"},
    ))
    event = json.loads(json.dumps(observer.events[0]))
    event.update(overrides or {})
    _runtime_evidence_path(comparison).write_text(
        "".join(json.dumps(event) + "\n" for _ in range(repeat)),
        encoding="utf-8",
    )
    return event


def _write_static_utility_evidence(
    comparison: Path,
    *,
    execution_id: str = "ingest.utility.1",
    gate_digest: str = "a" * 64,
) -> tuple[dict, dict]:
    source = SimpleNamespace(
        run_id=comparison.parent.name,
        episode_id="learn",
        session_id="session-1",
        task_id="task-1",
        snapshot_id="snapshot-1",
    )
    request = SimpleNamespace(
        idempotency_key=f"semantic_request.{execution_id}",
        provenance=SimpleNamespace(source=source),
    )
    operation = SimpleNamespace(
        operation_id=f"operation.{execution_id}",
        action=InternalMemoryAction.ADD,
    )
    result = SimpleNamespace(
        idempotency_key=request.idempotency_key,
        execution_id=execution_id,
        status=MemoryIngestStatus.SUCCESS,
        outcome=MemoryIngestOutcome.PLANNED_MUTATION,
        fixed_route=SimpleNamespace(
            backend="hermes-native-semantic",
            kind=MemoryKind.SEMANTIC,
        ),
        policy_provider="mem0_flat",
        policy_version="mem0-flat.utility.fixture",
        framework_version="mem0-flat-framework-v1",
        prompt_version="mem0-flat-prompts-v1",
        feature_schema_version="semantic-static-utility-features-v1",
        operations=(operation,),
        source_digest="b" * 64,
        content_digests=("c" * 64,),
        reason_codes=(),
        usage=RawResourceUsage(model_requests=2),
    )
    evidence = {
        "schema_version": 1,
        "gate_version": "mem0-flat-static-utility-gate-v1",
        "gate_digest": gate_digest,
        "feature_schema": "semantic-static-utility-features-v1",
        "request_id": request.idempotency_key,
        "decisions": [{
            "schema_version": 1,
            "target": "generation",
            "disposition": "accept",
            "score": 0.5,
            "predicted_benefit": 0.7,
            "lifecycle_cost": 0.1,
            "risk": 0.1,
            "contributions": {"benefit.scope": 0.05},
            "reason_codes": ["utility_accepted"],
            "feature_digest": "d" * 64,
            "cost_digest": "e" * 64,
            "feature_schema": "semantic-static-utility-features-v1",
            "cost_schema": "semantic-lifecycle-cost-v1",
            "policy_version": "semantic-static-utility-policy-v1",
            "cutoff": 0,
        }],
    }
    observer = LifecycleLedgerObserver(
        variant="with_persistence",
        trace_id="trace-1",
        family_id="family-1",
        stage="learn",
        output_path=_lifecycle_evidence_path(comparison),
    )
    observer.record_ingestion(request, result)
    observer.record_utility_decisions(request, result, evidence)
    return observer.events[-2], observer.events[-1]


def test_auto_loads_content_free_episode_runtime_evidence(tmp_path: Path) -> None:
    comparison = _fixture(tmp_path)
    runtime_event = _write_runtime_evidence(comparison)

    loaded = load_episode_lifecycle_events(comparison)
    events = build_events(comparison)

    assert loaded == (runtime_event,)
    assert events[-1] == runtime_event
    assert events[-1]["data"]["executionMode"] == "native+adapter+ledger"
    assert MEMORY not in json.dumps(events)


def test_auto_loads_content_free_static_mutation_identities(tmp_path: Path) -> None:
    comparison = _fixture(tmp_path)
    observer = MemoryLedgerObserver(
        run_id=comparison.parent.name,
        variant="with_persistence",
        trace_id="trace-1",
        episode_id="learn",
        session_id="session-1",
        task_id="task-1",
        family_id="family-1",
        stage="learn",
        execution_mode="native+ledger",
    )
    observer.record(MemoryEvent(
        MemoryEventKind.MUTATION_COMMITTED,
        MemoryKind.SEMANTIC,
        "hermes-native-semantic",
        artifact_ids=("artifact.user.1",),
        attributes={
            "action": "add",
            "execution_id": "ingest.1",
            "operation_id": "operation.1",
            "snapshot_id": "snapshot_1",
            "mutation_id": "mutation.1",
            "receipt_id": "receipt.1",
            "writer_identity": "rsimem_executor",
        },
    ))
    event = json.loads(json.dumps(observer.events[0]))
    _runtime_evidence_path(comparison).write_text(
        json.dumps(event) + "\n",
        encoding="utf-8",
    )
    assert load_episode_lifecycle_events(comparison) == (event,)

    event["data"]["attributes"]["action"] = "none"
    event["data"]["attributes"]["writer_identity"] = None
    _runtime_evidence_path(comparison).write_text(
        json.dumps(event) + "\n",
        encoding="utf-8",
    )
    assert load_episode_lifecycle_events(comparison) == (event,)

    event["data"]["attributes"]["writer_identity"] = "private writer text"
    _runtime_evidence_path(comparison).write_text(
        json.dumps(event) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="operation identity"):
        load_episode_lifecycle_events(comparison)


def test_auto_loads_strict_projection_check_evidence(tmp_path: Path) -> None:
    comparison = _fixture(tmp_path)
    observer = MemoryLedgerObserver(
        run_id=comparison.parent.name,
        variant="with_persistence",
        trace_id="trace-1",
        episode_id="learn",
        session_id="session-1",
        task_id="task-1",
        family_id="family-1",
        stage="learn",
        execution_mode="native+adapter+ledger",
    )
    observer.record(MemoryEvent(
        MemoryEventKind.PROJECTION_CHECK,
        MemoryKind.SEMANTIC,
        "hermes-native-semantic",
        content_chars=12,
        attributes={"surface": "system_prompt", "equivalent": True},
    ))
    event = json.loads(json.dumps(observer.events[0]))
    _runtime_evidence_path(comparison).write_text(
        json.dumps(event) + "\n",
        encoding="utf-8",
    )

    assert load_episode_lifecycle_events(comparison) == (event,)

    event["data"]["attributes"]["equivalent"] = "false"
    _runtime_evidence_path(comparison).write_text(
        json.dumps(event) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="projection result"):
        load_episode_lifecycle_events(comparison)


def test_auto_loads_strict_lifecycle_contract_evidence(tmp_path: Path) -> None:
    comparison = _fixture(tmp_path)
    observer = LifecycleLedgerObserver(
        variant="with_persistence",
        trace_id="trace-1",
        family_id="family-1",
        stage="learn",
    )
    fixture = run_sm01_preference_fixture()
    snapshot = replace(
        fixture.snapshot,
        run_id=comparison.parent.name,
        episode_id="learn",
        session_id="session-1",
        task_id="task-1",
        provenance=replace(
            fixture.snapshot.provenance,
            run_id=comparison.parent.name,
            episode_id="learn",
            session_id="session-1",
            task_id="task-1",
        ),
    )
    observer.record_snapshot(snapshot)
    event = json.loads(json.dumps(observer.events[0]))
    path = _lifecycle_evidence_path(comparison)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")

    assert load_episode_lifecycle_events(comparison) == (event,)
    assert build_events(comparison)[-1] == event

    event["data"]["rawContext"] = "private text"
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lifecycle event data fields"):
        load_episode_lifecycle_events(comparison)


def test_static_utility_evidence_joins_ingestion_and_audits_frozen_policy(
    tmp_path: Path,
) -> None:
    comparison = _fixture(tmp_path)
    ingestion, utility = _write_static_utility_evidence(comparison)

    assert load_episode_lifecycle_events(comparison) == (ingestion, utility)
    write_ledger(
        comparison,
        comparison.parent / "ledger.jsonl",
        judge_enabled=False,
    )
    report = audit_run(comparison.parent)

    assert report["ok"] is True
    assert report["staticUtility"] == {
        "events": 1,
        "uniqueExecutions": 1,
        "duplicateViews": 0,
        "decisionCount": 1,
        "targets": {"generation": 1},
        "dispositions": {"accept": 1},
        "gateDigests": ["a" * 64],
        "gateVersions": ["mem0-flat-static-utility-gate-v1"],
        "featureSchemas": ["semantic-static-utility-features-v1"],
        "policyVersions": ["semantic-static-utility-policy-v1"],
    }


def test_audit_run_reports_policy_evidence_and_identity(tmp_path: Path) -> None:
    comparison = _fixture(tmp_path)
    policy_path = _runtime_evidence_path(comparison).with_name(
        "rsimem_policy_decisions.jsonl"
    )
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
    JsonPolicyDecisionLedger(
        policy_path,
        variant="with_persistence",
        trace_id="trace-1",
        family_id="family-1",
        stage="learn",
    ).record_decision(
        decision,
        run_id="run-1",
        episode_id="learn",
        session_id="session-1",
        task_id="task-1",
        snapshot_id="snapshot.fixture",
    )
    write_ledger(
        comparison,
        comparison.parent / "ledger.jsonl",
        judge_enabled=False,
    )
    report = audit_run(comparison.parent)
    assert report["policyEvidence"]["files"] == 1
    assert report["policyEvidence"]["events"] == 1
    assert report["policyEvidence"]["reports"][0]["ok"] is True


def test_audit_run_reports_and_validates_process_evidence(tmp_path: Path) -> None:
    comparison = _fixture(tmp_path)
    process_path = _runtime_evidence_path(comparison).with_name(
        "rsimem_process_feedback.jsonl"
    )
    event = ProcessEvent.create(
        kind=ProcessEventKind.RETRIEVAL,
        status=ProcessEventStatus.SUCCESS,
        run_id="run-1",
        variant="with_persistence",
        trace_id="trace-1",
        episode_id="episode-1",
        session_id="session-1",
        task_id="task-1",
        host_event_id="event-retrieval-1",
        source_revision="revision-1",
        input_payload={"query": "digest-only"},
        output_payload={"count": 1},
        execution_receipt_ids=("receipt-retrieval-1",),
        family_id="family-1",
        stage="learn",
    )
    process_path.write_text(json.dumps(event.payload()) + "\n", encoding="utf-8")
    write_ledger(comparison, comparison.parent / "ledger.jsonl", judge_enabled=False)

    report = audit_run(comparison.parent)
    assert report["ok"] is True
    assert report["processEvidence"]["files"] == 1
    assert report["processEvidence"]["events"] == 1
    assert report["processEvidence"]["reports"][0]["ok"] is True

    invalid = ProcessEvent.create(
        kind=ProcessEventKind.RETRIEVAL,
        status=ProcessEventStatus.SUCCESS,
        run_id="run-1",
        variant="with_persistence",
        trace_id="trace-other",
        episode_id="episode-1",
        session_id="session-1",
        task_id="task-1",
        host_event_id="event-retrieval-2",
        source_revision="revision-1",
        input_payload={"query": "digest-only"},
        output_payload={"count": 0},
        reason_codes=("retrieval_miss",),
        policy_decision_id="decision.missing",
        policy_layer="exposure",
        execution_receipt_ids=("receipt-retrieval-2",),
        family_id="family-1",
        stage="learn",
    )
    process_path.write_text(json.dumps(invalid.payload()) + "\n", encoding="utf-8")
    report = audit_run(comparison.parent)
    assert report["ok"] is False
    assert any(issue["kind"] == "process_feedback_audit_failed" for issue in report["issues"])
    assert any(
        "trace_id does not match expected identity" in error
        for row in report["processEvidence"]["reports"]
        for error in row["errors"]
    )


def test_static_utility_evidence_rejects_content_and_policy_drift(
    tmp_path: Path,
) -> None:
    comparison = _fixture(tmp_path)
    _write_static_utility_evidence(comparison)
    path = _lifecycle_evidence_path(comparison)
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    events[-1]["data"]["decisions"][0]["raw_content"] = MEMORY
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="utility decision fields"):
        load_episode_lifecycle_events(comparison)

    path.unlink()
    _write_static_utility_evidence(comparison)
    _write_static_utility_evidence(
        comparison,
        execution_id="ingest.utility.2",
        gate_digest="f" * 64,
    )
    write_ledger(
        comparison,
        comparison.parent / "ledger.jsonl",
        judge_enabled=False,
    )
    report = audit_run(comparison.parent)
    assert report["ok"] is False
    assert {issue["kind"] for issue in report["issues"]} == {
        "static_utility_gate_changed_within_run"
    }


def test_run_anchored_relative_paths_are_cwd_independent(tmp_path: Path) -> None:
    comparison = _fixture(tmp_path)
    runtime_event = _write_runtime_evidence(comparison)
    data = json.loads(comparison.read_text(encoding="utf-8"))
    run = comparison.parent
    episode = data["with_persistence"]["episodes"][0]
    trace = Path(episode["trace"])
    session = Path(episode["internal_tools"]["session_file"])
    prefix = Path("..") / ".." / "outputs" / run.name
    episode["trace"] = str(prefix / trace.relative_to(run))
    episode["internal_tools"]["session_file"] = str(
        prefix / session.relative_to(run)
    )
    comparison.write_text(json.dumps(data), encoding="utf-8")

    events = build_events(comparison)
    report_path = run / "ledger.jsonl"
    write_ledger(comparison, report_path)
    report = audit_run(run)

    assert runtime_event in events
    assert any(event["kind"] == "model_call_usage" for event in events)
    assert any(event["kind"] == "memory_injection" for event in events)
    assert report["ok"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("runId", "another-run"),
        ("variant", "without_persistence"),
        ("traceId", "another-trace"),
        ("taskId", "another-task"),
        ("familyId", "another-family"),
        ("stage", "eval_near"),
    ),
)
def test_auto_loaded_runtime_evidence_must_match_owning_episode(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    comparison = _fixture(tmp_path)
    _write_runtime_evidence(comparison, overrides={field: value})

    with pytest.raises(ValueError, match="evidence identity"):
        build_events(comparison)


@pytest.mark.parametrize("serialized", ("not-json\n", "[]\n"))
def test_auto_loaded_runtime_evidence_rejects_malformed_jsonl(
    tmp_path: Path,
    serialized: str,
) -> None:
    comparison = _fixture(tmp_path)
    _runtime_evidence_path(comparison).write_text(serialized, encoding="utf-8")

    with pytest.raises(ValueError, match="malformed|must be an object"):
        build_events(comparison)


def test_auto_loaded_runtime_evidence_rejects_content_fields(tmp_path: Path) -> None:
    comparison = _fixture(tmp_path)
    event = _write_runtime_evidence(comparison)
    event["data"]["content"] = MEMORY
    _runtime_evidence_path(comparison).write_text(
        json.dumps(event) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="runtime event data fields"):
        build_events(comparison)


def test_infra_blocked_episode_with_empty_trace_keeps_unknown_usage(
    tmp_path: Path,
) -> None:
    comparison = _fixture(tmp_path)
    payload = json.loads(comparison.read_text(encoding="utf-8"))
    episode = payload["with_persistence"]["episodes"][0]
    episode["trace"] = ""
    episode["infra_blocked"] = True
    episode["token_usage"]["model_request_count"] = None
    episode["token_usage"]["model_usage_complete"] = False
    comparison.write_text(json.dumps(payload), encoding="utf-8")

    events = build_events(comparison)

    outcome = next(event for event in events if event["kind"] == "episode_outcome")
    usage = next(event for event in events if event["kind"] == "model_usage")
    assert outcome["data"]["infraBlocked"] is True
    assert usage["data"]["detailedUsageAvailable"] is False
    assert not any(event["kind"] == "model_call_usage" for event in events)


def test_auto_loaded_runtime_event_duplicates_are_idempotent_but_conflicts_fail(
    tmp_path: Path,
) -> None:
    comparison = _fixture(tmp_path)
    event = _write_runtime_evidence(comparison, repeat=2)
    loaded = load_episode_lifecycle_events(comparison)
    assert loaded == (event,)

    conflicting = json.loads(json.dumps(event))
    conflicting["data"]["queryChars"] = 99
    _runtime_evidence_path(comparison).write_text(
        json.dumps(event) + "\n" + json.dumps(conflicting) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conflicting ledger eventId"):
        load_episode_lifecycle_events(comparison)


def test_builds_evidence_backed_events_without_memory_text(tmp_path: Path) -> None:
    comparison = _fixture(tmp_path)
    events = build_events(comparison)
    kinds = [event["kind"] for event in events]
    assert kinds == [
        "episode_outcome",
        "model_usage",
        "model_call_usage",
        "tool_call",
        "memory_operation",
        "memory_injection",
        "storage_snapshot",
    ]
    usage = next(event for event in events if event["kind"] == "model_usage")
    outcome = next(event for event in events if event["kind"] == "episode_outcome")
    assert usage["data"]["requestCount"] == 1
    assert usage["data"]["requestCountEvidence"] == "model_call_usage_events"
    assert usage["data"]["retryCount"] == 0
    assert usage["data"]["cacheReadTokens"] == 20
    assert usage["data"]["reasoningTokens"] == 5
    assert usage["data"]["detailedUsageAvailable"] is True
    model_call = next(event for event in events if event["kind"] == "model_call_usage")
    assert model_call["data"]["billingExecutionId"] == "trace-1:model-call-0001"
    assert outcome["data"]["judgeEnabled"] is None
    assert outcome["data"]["judgeConfigurationEvidence"] == "unavailable"
    operation = next(event for event in events if event["kind"] == "memory_operation")
    injection = next(event for event in events if event["kind"] == "memory_injection")
    assert operation["data"]["recordId"] == injection["data"]["recordId"]
    old_content_fingerprint = hashlib.sha256(" ".join(MEMORY.split()).encode()).hexdigest()[:24]
    assert old_content_fingerprint not in operation["data"]["recordId"]
    assert MEMORY not in json.dumps(events)


def test_ledger_is_deterministic_and_marks_unmatched_injection(tmp_path: Path) -> None:
    comparison = _fixture(tmp_path)
    data = json.loads(comparison.read_text())
    session_path = Path(data["with_persistence"]["episodes"][0]["internal_tools"]["session_file"])
    session = json.loads(session_path.read_text())
    session["system_prompt"] = "No memory here."
    session_path.write_text(json.dumps(session), encoding="utf-8")

    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first_events = write_ledger(comparison, first, judge_enabled=False)
    second_events = write_ledger(comparison, second, judge_enabled=False)
    assert first.read_bytes() == second.read_bytes()
    assert [event["eventId"] for event in first_events] == [event["eventId"] for event in second_events]
    unresolved = next(event for event in first_events if event["kind"] == "memory_injection_unresolved")
    assert unresolved["data"] == {"reportedCount": 1, "matchedCount": 0}
    outcome = next(event for event in first_events if event["kind"] == "episode_outcome")
    assert outcome["data"]["judgeEnabled"] is False
    assert outcome["data"]["judgeConfigurationEvidence"] == "launcher_explicit"

    write_ledger(
        comparison,
        comparison.parent / "ledger.jsonl",
        judge_enabled=False,
    )
    report = audit_run(comparison.parent)
    assert report["ok"] is False
    assert {issue["kind"] for issue in report["issues"]} == {
        "unresolved_memory_injection"
    }


def test_generated_event_ids_follow_logical_ids_when_evidence_is_reordered(
    tmp_path: Path,
) -> None:
    """Replaying a trace in a different order must not rename its events."""

    comparison = _fixture(tmp_path)
    payload = json.loads(comparison.read_text(encoding="utf-8"))
    episode = payload["with_persistence"]["episodes"][0]
    trace_path = Path(episode["trace"])
    trace_events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    second_call = dict(
        next(item for item in trace_events if item.get("type") == "model_call_usage")
    )
    second_call.update({"call_id": "model-call-0002", "sequence": 2, "attempt": 1})
    trace_events.insert(
        next(
            index
            for index, item in enumerate(trace_events)
            if item.get("type") == "trace_end"
        ),
        second_call,
    )
    trace_path.write_text(
        "".join(json.dumps(item) + "\n" for item in trace_events),
        encoding="utf-8",
    )

    first = build_events(comparison)
    model_ids_first = {
        event["data"]["callId"]: event["eventId"]
        for event in first
        if event["kind"] == "model_call_usage"
    }

    model_calls = [
        item for item in trace_events if item.get("type") == "model_call_usage"
    ]
    remaining = [
        item for item in trace_events if item.get("type") != "model_call_usage"
    ]
    reordered = remaining[:1] + list(reversed(model_calls)) + remaining[1:]
    trace_path.write_text(
        "".join(json.dumps(item) + "\n" for item in reordered),
        encoding="utf-8",
    )
    second = build_events(comparison)
    model_ids_second = {
        event["data"]["callId"]: event["eventId"]
        for event in second
        if event["kind"] == "model_call_usage"
    }

    assert model_ids_first == model_ids_second

    # Tool calls use their Hermes message index as the logical identity, not
    # their position in the calls array.
    episode["internal_tools"]["calls"] = [
        {
            "name": "memory",
            "args": {"action": "add", "target": "memory", "content": MEMORY},
            "message_index": 1,
        },
        {
            "name": "memory",
            "args": {
                "action": "update",
                "target": "memory",
                "new_content": "updated",
            },
            "message_index": 2,
        },
    ]
    comparison.write_text(json.dumps(payload), encoding="utf-8")
    first = build_events(comparison)
    tool_ids_first = {
        (event["kind"], event["source"]["messageIndex"]): event["eventId"]
        for event in first
        if event["kind"] in {"tool_call", "memory_operation"}
    }
    episode["internal_tools"]["calls"].reverse()
    comparison.write_text(json.dumps(payload), encoding="utf-8")
    second = build_events(comparison)
    tool_ids_second = {
        (event["kind"], event["source"]["messageIndex"]): event["eventId"]
        for event in second
        if event["kind"] in {"tool_call", "memory_operation"}
    }
    assert tool_ids_first == tool_ids_second


def test_generated_event_id_conflict_is_not_silently_overwritten(tmp_path: Path) -> None:
    comparison = _fixture(tmp_path)
    payload = json.loads(comparison.read_text(encoding="utf-8"))
    original = payload["with_persistence"]["episodes"][0]
    conflicting = json.loads(json.dumps(original))
    conflicting["task_score"] = 0.25
    payload["with_persistence"]["episodes"].append(conflicting)
    comparison.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="conflicting ledger eventId"):
        build_events(comparison)


def test_user_profile_injection_matches_and_is_privacy_audited(tmp_path: Path) -> None:
    comparison = _fixture(tmp_path)
    data = json.loads(comparison.read_text(encoding="utf-8"))
    episode = data["with_persistence"]["episodes"][0]
    session_path = Path(episode["internal_tools"]["session_file"])
    session = json.loads(session_path.read_text(encoding="utf-8"))
    private_user_entry = "The user prefers a four-column TSV."
    session["system_prompt"] = f"User profile:\n{private_user_entry}"
    session_path.write_text(json.dumps(session), encoding="utf-8")
    episode["artifacts"]["memory_entries"] = []
    episode["artifacts"]["user_entries"] = [private_user_entry]
    comparison.write_text(json.dumps(data), encoding="utf-8")

    output = comparison.parent / "ledger.jsonl"
    events = write_ledger(comparison, output, judge_enabled=False)
    injections = [event for event in events if event["kind"] == "memory_injection"]

    assert len(injections) == 1
    assert not any(event["kind"] == "memory_injection_unresolved" for event in events)
    assert private_user_entry not in output.read_text(encoding="utf-8")
    report = audit_run(comparison.parent)
    assert report["ok"] is True
    assert report["privacy"]["memoryTextLeaks"] == 0


def test_failed_model_call_keeps_unknown_tokens_null(tmp_path: Path) -> None:
    comparison = _fixture(tmp_path)
    data = json.loads(comparison.read_text())
    trace_path = Path(data["with_persistence"]["episodes"][0]["trace"])
    failed_call = {
        "type": "model_call_usage",
        "trace_id": "trace-1",
        "call_id": "model-call-0002",
        "sequence": 2,
        "component": "agent",
        "purpose": "task_execution",
        "provider": "test-provider",
        "model": "gpt-test",
        "api_mode": "codex_responses",
        "attempt": 2,
        "status": "error",
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "usage_available": False,
        "duration_ms": 10.0,
        "http_status": 503,
        "error_category": "ServiceUnavailableError",
    }
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(failed_call) + "\n")

    events = build_events(comparison)
    failed = next(
        event for event in events
        if event["kind"] == "model_call_usage" and event["data"]["status"] == "error"
    )
    assert failed["data"]["usageAvailable"] is False
    assert failed["data"]["inputTokens"] is None
    assert failed["data"]["outputTokens"] is None


def test_audit_reconciles_request_usage_and_ledger(tmp_path: Path) -> None:
    comparison = _fixture(tmp_path)
    run_dir = comparison.parent
    write_ledger(comparison, run_dir / "ledger.jsonl", judge_enabled=False)

    report = audit_run(run_dir)

    assert report["ok"] is True
    assert report["issues"] == []
    assert report["uniqueTraceCount"] == 1
    assert report["uniquePhysicalUsage"]["requests"] == 1
    assert report["uniquePhysicalUsage"]["cacheReadTokens"] == 20
    assert report["ledgerUniqueBillingCalls"] == 1
    assert report["privacy"]["memoryTextLeaks"] == 0


def test_write_ledger_joins_validated_lifecycle_events(tmp_path: Path) -> None:
    comparison = _fixture(tmp_path)
    snapshot = HermesSnapshotCollector().collect(
        (
            HermesMessage(
                "message-1",
                "user",
                MEMORY,
                "turn-1",
                10,
                completed=True,
            ),
        ),
        run_id=comparison.parent.name,
        episode_id="episode-1",
        session_id="session-1",
        task_id="task-1",
        current_turn_id=None,
        task_state=TaskLifecycleState.COMPLETED,
        lifecycle_state="task_completed",
        source_ref="fixture:ledger-join",
    )
    observer = LifecycleLedgerObserver(
        variant="with_persistence",
        trace_id="trace-1",
        family_id="family-1",
        stage="learn",
    )
    observer.record_snapshot(snapshot)

    output = comparison.parent / "ledger.jsonl"
    events = write_ledger(
        comparison,
        output,
        lifecycle_events=observer.events,
    )
    joined = events[-1]
    assert joined["kind"] == "context_snapshot"
    assert joined["episodeId"] == "episode-1"
    assert joined["sessionId"] == "session-1"
    assert joined["data"]["segmentCount"] == 1
    assert MEMORY not in output.read_text(encoding="utf-8")

    mismatches = {
        "runId": "another-run",
        "variant": "another-variant",
        "traceId": "another-trace",
        "taskId": "another-task",
        "familyId": "another-family",
        "stage": "eval_near",
    }
    for index, (field, value) in enumerate(mismatches.items()):
        invalid = dict(joined, **{field: value}, eventId=f"evt_other_{index}")
        with pytest.raises(ValueError, match=field):
            build_events(comparison, lifecycle_events=(invalid,))


def test_identical_lifecycle_event_is_idempotent_but_payload_conflict_fails(
    tmp_path: Path,
) -> None:
    comparison = _fixture(tmp_path)
    snapshot = HermesSnapshotCollector().collect(
        (
            HermesMessage("message-1", "user", MEMORY, "turn-1", 10, completed=True),
        ),
        run_id=comparison.parent.name,
        episode_id="episode-1",
        session_id="session-1",
        task_id="task-1",
        current_turn_id=None,
        task_state=TaskLifecycleState.COMPLETED,
        lifecycle_state="task_completed",
        source_ref="fixture:ledger-conflict",
    )
    observer = LifecycleLedgerObserver(
        variant="with_persistence",
        trace_id="trace-1",
        family_id="family-1",
        stage="learn",
    )
    observer.record_snapshot(snapshot)
    same = observer.events[0]
    assert len(build_events(comparison, lifecycle_events=(same, same))) == 8

    conflicting = json.loads(json.dumps(same))
    conflicting["data"]["totalTokens"] = 999
    conflicting["eventId"] = same["eventId"]
    with pytest.raises(ValueError, match="conflicting ledger eventId"):
        build_events(comparison, lifecycle_events=(same, conflicting))


def test_lifecycle_event_identity_excludes_mutable_status_and_reasons(tmp_path: Path) -> None:
    comparison = _fixture(tmp_path)
    observer = LifecycleLedgerObserver(
        variant="with_persistence",
        trace_id="trace-1",
        family_id="family-1",
        stage="learn",
    )
    snapshot = HermesSnapshotCollector().collect(
        (HermesMessage("message-1", "user", MEMORY, "turn-1", 10, completed=True),),
        run_id=comparison.parent.name,
        episode_id="episode-1",
        session_id="session-1",
        task_id="task-1",
        current_turn_id=None,
        task_state=TaskLifecycleState.COMPLETED,
        lifecycle_state="task_completed",
        source_ref="fixture:lifecycle-identity",
    )
    observer.record_evaluation(
        snapshot,
        evaluation_id="evaluation-1",
        trigger="task_completed",
        evaluator="deterministic",
        policy_version="policy-v1",
        status="accepted",
        reason_codes=("decision_observed",),
    )
    first = observer.events[-1]
    with pytest.raises(ValueError, match="conflicting lifecycle ledger event"):
        observer.record_evaluation(
            snapshot,
            evaluation_id="evaluation-1",
            trigger="task_completed",
            evaluator="deterministic",
            policy_version="policy-v1",
            status="accepted",
            reason_codes=("absence",),
        )
    assert observer.events[-1] == first
