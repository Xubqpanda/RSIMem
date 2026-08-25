from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rsimem.audit import audit_run
from rsimem.ledger import LifecycleLedgerObserver, build_events, write_ledger
from rsimem.lifecycle import (
    HermesMessage,
    HermesSnapshotCollector,
    TaskLifecycleState,
)


MEMORY = "Use TSV with owner, priority, task, and due_date."


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
