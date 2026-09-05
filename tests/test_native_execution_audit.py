from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rsimem.native_attribution_protocol import NativeAttributionRepairProtocol
from rsimem.native_attribution_run import build_native_attribution_manifest
from rsimem.native_execution_audit import NativeExecutionAuditStore, audit_native_execution


ROOT = Path(__file__).resolve().parents[1]


def _fixture(tmp_path: Path, family_id: str | None = None):
    manifest = build_native_attribution_manifest(
        batch_id="execution-audit-fixture",
        protocol=NativeAttributionRepairProtocol.create(),
        past_bench_root=ROOT / "benchmarks" / "past-bench",
        rsimem_commit="commit.rsimem.fixture",
        past_bench_commit="commit.past.fixture",
        replicate_count=1,
        port_base_offset=100,
    )
    run = next(
        value for value in manifest.runs
        if family_id is None or value.family_id == family_id
    )
    for relative in (
        run.state_directory, run.hermes_home_directory, run.session_directory,
        run.artifact_directory,
    ):
        (tmp_path / relative).mkdir(parents=True)
    trace_root = tmp_path / run.trace_directory
    result_episodes = []
    services_by_task = {}
    for value in run.service_identities:
        services_by_task.setdefault(value.task_id, []).append(value)
    for index, task_id in enumerate(run.native_task_ids, start=1):
        service_specs = services_by_task.get(task_id, [])
        episode_dir = trace_root / f"{index:02d}_episode"
        episode_dir.mkdir(parents=True)
        trace_id = f"trace-fixture-{index}"
        trace_path = episode_dir / f"task_{index}.jsonl"
        events = [
            {"type": "trace_start", "trace_id": trace_id, "task_id": task_id, "model": run.model_id},
            {
                "type": "model_call_usage", "trace_id": trace_id, "attempt": 1,
                "usage_available": True,
                "usage": {
                    "input_tokens": 10, "output_tokens": 2,
                    "cache_read_tokens": 1, "cache_write_tokens": 0,
                    "reasoning_tokens": 1, "usage_complete": True,
                },
            },
            {
                "type": "trace_end", "trace_id": trace_id,
                "model_input_tokens": 10, "model_output_tokens": 2,
                "cache_read_tokens": 1, "cache_write_tokens": 0,
                "reasoning_tokens": 1, "model_request_count": 1,
                "model_retry_count": 0, "model_usage_complete": True,
            },
        ]
        trace_path.write_text("\n".join(json.dumps(item) for item in events) + "\n", encoding="utf-8")
        services = [{
            "schema": "past-bench-verified-service-identity-v1",
            "service": value.service,
            "port": value.port,
            "fixture_digest": value.fixture_digest,
        } for value in service_specs]
        (episode_dir / "service_identity.json").write_text(json.dumps({
            "schema": "past-bench-episode-service-identity-v1",
            "task_id": task_id,
            "services": services,
        }), encoding="utf-8")
        artifact_dir = episode_dir / "artifacts"
        artifact_dir.mkdir()
        memory_events = [
            {"eventId": f"memory-query-{index}", "kind": "query"},
            {"eventId": f"memory-retrieved-{index}", "kind": "retrieved"},
        ]
        (artifact_dir / "rsimem_memory_events.jsonl").write_text(
            "\n".join(json.dumps(item) for item in memory_events) + "\n",
            encoding="utf-8",
        )
        process_events = [
            {"event_id": f"tool-call-{index}", "kind": "tool_call"},
            {"event_id": f"tool-result-{index}", "kind": "tool_result"},
            {"event_id": f"outcome-{index}", "kind": "task_outcome"},
        ]
        (artifact_dir / "pure_process_event_archive.jsonl").write_text(
            "\n".join(json.dumps(item) for item in process_events) + "\n",
            encoding="utf-8",
        )
        empty_artifacts = {
            "artifact_ids": [], "memory_entry_count": 0,
            "user_entry_count": 0, "skill_count": 0, "digest": "1" * 64,
        }
        (episode_dir / "native_episode_identity.json").write_text(json.dumps({
            "schema": "past-bench-native-episode-identity-v1",
            "task_id": task_id,
            "family_id": run.family_id,
            "stage": "fixture",
            "trace_id": trace_id,
            "state_before_digest": "2" * 64,
            "state_after_digest": "3" * 64,
            "artifact_before": empty_artifacts,
            "artifact_after": empty_artifacts,
        }), encoding="utf-8")
        result_episodes.append({
            "trace_id": trace_id,
            "task_id": task_id,
            "family_id": run.family_id,
            "trace": str(trace_path),
            "internal_tools": {},
            "token_usage": {
                "input_tokens": 10, "output_tokens": 2,
                "cache_read_tokens": 1, "cache_write_tokens": 0,
                "reasoning_tokens": 1, "model_request_count": 1,
                "model_retry_count": 0, "model_usage_complete": True,
            },
        })
    (trace_root / "sequence_results.json").write_text(json.dumps({
        "sequence": f"native-attribution.{run.run_id}",
        "variant": "with_persistence",
        "episodes": result_episodes,
    }), encoding="utf-8")
    (trace_root / "native_runtime_identity.json").write_text(json.dumps({
        "schema": "past-bench-native-runtime-identity-v1",
        "sequence": f"native-attribution.{run.run_id}",
        "method_task_id": run.method_case_id,
        "agent": "hermes-luna",
        "model_id": run.model_id,
        "base_url": f"https://{run.provider_id}",
        "port_offset": run.port_offset,
        "state_directory": str(tmp_path / run.state_directory),
        "hermes_home_directory": str(tmp_path / run.hermes_home_directory),
        "session_directory": str(tmp_path / run.session_directory),
        "artifact_directory": str(tmp_path / run.artifact_directory),
        "trace_directory": str(trace_root),
        "initial_home_digest": run.initial_home_digest,
    }), encoding="utf-8")
    return run


def test_native_execution_audit_reconstructs_content_free_identity(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    audit = audit_native_execution(run=run, output_root=tmp_path)
    assert audit.run_id == run.run_id
    assert audit.provider_id == run.provider_id
    assert audit.model_id == run.model_id
    assert len(audit.trace_ids) == len({value.episode_id for value in run.service_identities})
    assert audit.request_count == len(audit.trace_ids)
    assert audit.input_tokens == 10 * len(audit.trace_ids)
    assert audit.payload()["schema"] == "rsimem-native-execution-audit-v1"


def test_native_execution_audit_accepts_manifest_declared_no_service_tasks(tmp_path: Path) -> None:
    run = _fixture(tmp_path, "PG01_release_decision_followup")
    audit = audit_native_execution(run=run, output_root=tmp_path)
    assert len(audit.trace_ids) == len(run.native_task_ids)


def test_native_execution_audit_fails_closed_for_incomplete_usage(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    path = tmp_path / run.trace_directory / "sequence_results.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result["episodes"][0]["token_usage"]["model_usage_complete"] = False
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="usage is incomplete"):
        audit_native_execution(run=run, output_root=tmp_path)


def test_native_execution_audit_rejects_trace_and_service_drift(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    service_path = next((tmp_path / run.trace_directory).glob("**/service_identity.json"))
    service = json.loads(service_path.read_text(encoding="utf-8"))
    service["services"][0]["port"] += 1
    service_path.write_text(json.dumps(service), encoding="utf-8")
    with pytest.raises(ValueError, match="service fixture identity"):
        audit_native_execution(run=run, output_root=tmp_path)


def test_native_execution_audit_rejects_request_usage_mismatch(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    trace_path = next((tmp_path / run.trace_directory).glob("**/*.jsonl"))
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    next(item for item in events if item["type"] == "model_call_usage")["usage"]["input_tokens"] = 11
    trace_path.write_text("\n".join(json.dumps(item) for item in events) + "\n")
    with pytest.raises(ValueError, match="usage total mismatch: input_tokens"):
        audit_native_execution(run=run, output_root=tmp_path)


def test_native_execution_audit_store_is_append_once(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    audit = audit_native_execution(run=run, output_root=tmp_path)
    store = NativeExecutionAuditStore(tmp_path / "receipts")
    assert store.put(audit) is True
    assert store.put(audit) is False
    changed = replace(audit, state_digest="f" * 64)
    with pytest.raises(ValueError, match="different execution audit"):
        store.put(changed)
