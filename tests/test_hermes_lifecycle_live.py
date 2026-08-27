from __future__ import annotations

import json
from copy import deepcopy

import pytest

from rsimem.lifecycle import (
    DryRunStatus,
    EvaluationTrigger,
    HermesLifecycleConfig,
    HermesLifecycleDryRunRuntime,
    TaskLifecycleState,
)


def _rows():
    return (
        {"id": 1, "role": "user", "content": "Always use TSV output."},
        {"id": 2, "role": "assistant", "content": "Understood.", "token_count": 3},
    )


def _runtime(tmp_path, config=None, *, complete=None):
    return HermesLifecycleDryRunRuntime(
        config or HermesLifecycleConfig(evaluator_mode="deterministic"),
        run_id="run-live",
        episode_id="episode-live",
        session_id="session-live",
        task_id="SM01-live",
        variant="native+adapter+ledger",
        trace_id="trace-live",
        receipt_path=tmp_path / "receipts.json",
        evidence_path=tmp_path / "lifecycle.jsonl",
        family_id="SM01",
        stage="learn",
        injected_complete=complete,
    )


def test_deterministic_live_runtime_reserves_once_per_boundary(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    first = runtime.process(
        _rows(),
        trigger=EvaluationTrigger.TASK_COMPLETED,
        task_state=TaskLifecycleState.COMPLETED,
        source_ref="hermes_state:session:live",
    )
    duplicate = runtime.process(
        _rows(),
        trigger=EvaluationTrigger.TASK_COMPLETED,
        task_state=TaskLifecycleState.COMPLETED,
        source_ref="hermes_state:session:live",
    )
    session_end = runtime.process(
        _rows(),
        trigger=EvaluationTrigger.SESSION_END,
        task_state=TaskLifecycleState.COMPLETED,
        source_ref="hermes_state:session:live",
    )

    assert duplicate is first
    assert len(first.plans) == len(first.receipts) == 1
    assert first.receipts[0].status == DryRunStatus.ACCEPTED
    assert session_end.receipts[0].status == DryRunStatus.ACCEPTED
    assert first.snapshot.snapshot_id != session_end.snapshot.snapshot_id
    kinds = [
        json.loads(line)["kind"]
        for line in (tmp_path / "lifecycle.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert kinds.count("context_snapshot") == 2
    assert kinds.count("evaluation_accepted") == 2
    assert kinds.count("dry_run_mutation") == 2


def test_live_runtime_restart_returns_persistent_duplicate(tmp_path) -> None:
    first = _runtime(tmp_path).process(
        _rows(),
        trigger=EvaluationTrigger.TASK_COMPLETED,
        task_state=TaskLifecycleState.COMPLETED,
        source_ref="hermes_state:session:live",
    )
    replay = _runtime(tmp_path).process(
        _rows(),
        trigger=EvaluationTrigger.TASK_COMPLETED,
        task_state=TaskLifecycleState.COMPLETED,
        source_ref="hermes_state:session:live",
    )

    assert first.plans[0].plan_id == replay.plans[0].plan_id
    assert replay.receipts[0].status == DryRunStatus.DUPLICATE
    assert replay.receipts[0].mutation_id == first.receipts[0].mutation_id


def test_injected_json_cannot_override_host_policy_version(tmp_path) -> None:
    def complete(prompt: str) -> str:
        request = json.loads(prompt)
        assert request["context_revision"].startswith("rev_")
        assert request["turn_index"] == 1
        assert request["protected_segment_ids"] == []
        assert request["host_policy_version"] == "host-policy-v3"
        return json.dumps({
            "policy_version": "model-controlled",
            "signals": [
                {
                    "segment_id": segment_id,
                    "context_action": "retain",
                    "writeback_action": "defer",
                    "utility_estimate": 0,
                    "confidence": 1,
                }
                for segment_id in (
                    segment["segment_id"] for segment in request["segments"]
                )
            ],
        })

    runtime = _runtime(
        tmp_path,
        HermesLifecycleConfig(
            evaluator_mode="injected_json",
            policy_version="host-policy-v3",
        ),
        complete=complete,
    )
    result = runtime.process(
        _rows(),
        trigger=EvaluationTrigger.TASK_COMPLETED,
        task_state=TaskLifecycleState.COMPLETED,
        source_ref="hermes_state:session:live",
    )
    assert result.evaluation.policy_version == "host-policy-v3"
    assert result.plans == ()


def test_disabled_and_unknown_lifecycle_configuration_fail_closed(tmp_path) -> None:
    with pytest.raises(ValueError, match="disabled lifecycle mode"):
        _runtime(tmp_path, HermesLifecycleConfig())
    with pytest.raises(ValueError, match="unknown Hermes lifecycle"):
        HermesLifecycleConfig.from_mapping({"turn_interval": 1})
    with pytest.raises(ValueError, match="requires an injected client"):
        _runtime(tmp_path, HermesLifecycleConfig(evaluator_mode="injected_json"))


def test_evaluator_failure_is_audited_and_same_boundary_can_retry(tmp_path) -> None:
    calls = 0

    def complete(prompt: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("fixture timeout with private response text")
        request = json.loads(prompt)
        return json.dumps({
            "policy_version": "ignored",
            "signals": [
                {
                    "segment_id": segment["segment_id"],
                    "context_action": "retain",
                    "writeback_action": "defer",
                    "utility_estimate": 0,
                    "confidence": 1,
                }
                for segment in request["segments"]
            ],
        })

    runtime = _runtime(
        tmp_path,
        HermesLifecycleConfig(
            evaluator_mode="injected_json",
            policy_version="host-retry-v1",
        ),
        complete=complete,
    )
    kwargs = {
        "trigger": EvaluationTrigger.TASK_COMPLETED,
        "task_state": TaskLifecycleState.COMPLETED,
        "source_ref": "hermes_state:session:live",
    }
    with pytest.raises(TimeoutError, match="fixture timeout"):
        runtime.process(_rows(), **kwargs)
    result = runtime.process(_rows(), **kwargs)

    assert calls == 2
    assert result.evaluation.policy_version == "host-retry-v1"
    events = [
        json.loads(line)
        for line in (tmp_path / "lifecycle.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["kind"] for event in events] == [
        "context_snapshot",
        "evaluation_rejected",
        "evaluation_accepted",
    ]
    serialized = json.dumps(events, ensure_ascii=True)
    assert "private response text" not in serialized
    assert not (tmp_path / "receipts.json").exists()


def test_live_runtime_rejects_stale_host_task_state(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    with pytest.raises(ValueError, match="requires completed host task state"):
        runtime.process(
            _rows(),
            trigger=EvaluationTrigger.TASK_COMPLETED,
            task_state=TaskLifecycleState.ACTIVE,
            source_ref="hermes_state:session:live",
        )
    with pytest.raises(ValueError, match="cannot carry active host task state"):
        runtime.process(
            _rows(),
            trigger=EvaluationTrigger.SESSION_END,
            task_state=TaskLifecycleState.ACTIVE,
            source_ref="hermes_state:session:live",
        )


def test_dry_run_leaves_hermes_state_and_input_rows_byte_identical(tmp_path) -> None:
    hermes_home = tmp_path / "hermes-home"
    files = {
        hermes_home / "state.db": b"fixture-state-db-bytes",
        hermes_home / "memories" / "MEMORY.md": b"private semantic memory\n",
        hermes_home / "memories" / "USER.md": b"private user profile\n",
        hermes_home / "skills" / "fixture" / "SKILL.md": b"fixture procedure\n",
    }
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    before = {path: path.read_bytes() for path in files}
    rows = [dict(row) for row in _rows()]
    rows_before = deepcopy(rows)

    result = _runtime(tmp_path / "control-plane").process(
        rows,
        trigger=EvaluationTrigger.TASK_COMPLETED,
        task_state=TaskLifecycleState.COMPLETED,
        source_ref="hermes_state:session:live",
    )

    assert result.receipts[0].status == DryRunStatus.ACCEPTED
    assert rows == rows_before
    assert {path: path.read_bytes() for path in files} == before
