from __future__ import annotations

import json
from pathlib import Path

from rsimem.experiment_manifest import (
    EXECUTION_MODES,
    execution_order,
    initialize_batch_manifest,
    record_attempt,
)
from rsimem.matched_analysis import analyze_batch


def _batch(tmp_path: Path) -> Path:
    root = tmp_path / "batch"
    initialize_batch_manifest(
        root / "batch_manifest.json",
        replicates=3,
        task_family="memory_ability/SM01",
        agent="hermes-luna",
        runtime="local",
        model={
            "profile": "hermes-luna/default_model",
            "modelId": "test-model",
            "providerBaseUrl": "https://provider.invalid/v1",
            "temperature": 0.0,
        },
        judge={"enabled": False, "profile": "disabled", "modelId": None},
        budget={
            "source": "task_manifest",
            "taskTimeoutOverrideSeconds": None,
            "tasks": [{
                "episode": "eval",
                "taskId": "task-eval",
                "maxTurns": 20,
                "timeoutSeconds": 300,
            }],
            "taskManifestDigest": "digest",
        },
        environment={
            "pythonVersion": "3.11.0",
            "distributions": {
                "rsimem": "0.1.0",
                "past-bench": "1.0.0",
                "hermes-agent": "0.4.0",
            },
        },
        persistence_isolation={
            "strategy": "per_attempt_trace_directory",
            "compareNoPersistence": True,
        },
        adapter_projection_verification=True,
        rsimem_commit="head",
        rsimem_working_tree_dirty=False,
        past_bench_commit="past",
        past_bench_tree="tree",
        past_bench_dirty=False,
    )
    for replicate in range(1, 4):
        for ordinal, mode in enumerate(execution_order(replicate), 1):
            run_name = f"r{replicate}_{mode.replace('+', '_')}"
            record_attempt(
                root / "batch_manifest.json",
                replicate=replicate,
                ordinal=ordinal,
                mode=mode,
                run_name=run_name,
                status="running",
            )
            run = root / run_name
            run.mkdir(parents=True)
            episode = {
                "task_id": "task-eval",
                "trace_id": f"trace-{replicate}-{mode}",
                "index": 1,
                "stage": "eval_near",
                "bucket": "evaluation",
                "history_mode": "fresh",
                "history_load_anchor": None,
                "history_save_anchor": None,
                "persistence_allowed": True,
                "timing": {"wall_time_s": 2.0},
                "artifacts": {
                    "memory_chars": 0,
                    "user_chars": 0,
                    "memory_entries": [],
                    "user_entries": [],
                    "skill_count": 0,
                },
            }
            comparison = {
                "with_persistence": {
                    "episodes": [episode],
                    "bucket_summary": {"evaluation": {
                        "avg_task_score": 1.0,
                        "pass_rate": 1.0,
                    }},
                },
                "without_persistence": {
                    "episodes": [episode],
                    "bucket_summary": {"evaluation": {
                        "avg_task_score": 0.4,
                        "pass_rate": 0.0,
                    }},
                },
                "delta": {},
            }
            (run / "sequence_comparison.json").write_text(
                json.dumps(comparison),
                encoding="utf-8",
            )
            projection_checks = 2 if mode == "native+adapter+ledger" else 0
            audit = {
                "ok": True,
                "issues": [],
                "projectionChecks": projection_checks,
                "projectionMismatches": 0,
                "adapterNativeBypasses": 0,
                "unresolvedMemoryInjections": 0,
                "uniquePhysicalUsage": {
                    "requests": 2,
                    "inputTokens": 10,
                    "outputTokens": 5,
                    "cacheReadTokens": 0,
                    "cacheWriteTokens": 0,
                    "reasoningTokens": 1,
                    "retries": 0,
                },
            }
            (run / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
            events = [
                {"kind": "tool_call", "data": {}},
                {"kind": "memory_injection", "data": {"contentChars": 12}},
                {"kind": "storage_snapshot", "data": {
                    "memoryFilesBytes": 12,
                    "skillFilesBytes": 0,
                    "stateDbBytes": 4,
                }},
            ]
            if projection_checks:
                events.append({"kind": "projection_check", "data": {
                    "attributes": {"equivalent": True},
                }})
            (run / "ledger.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            record_attempt(
                root / "batch_manifest.json",
                replicate=replicate,
                ordinal=ordinal,
                mode=mode,
                run_name=run_name,
                status="completed",
            )
    return root


def test_matched_analysis_accepts_complete_content_free_batch(tmp_path: Path) -> None:
    report = analyze_batch(_batch(tmp_path))

    assert report["stageGatePassed"] is True
    assert report["issues"] == []
    for mode in EXECUTION_MODES:
        assert report["summaryByMode"][mode]["sampleSize"] == 3
        assert report["summaryByMode"][mode]["metrics"]["persistenceScoreGap"]["median"] == 0.6


def test_matched_analysis_rejects_missing_adapter_projection_checks(tmp_path: Path) -> None:
    root = _batch(tmp_path)
    adapter_run = next(root.glob("r1_native_adapter_ledger"))
    audit_path = adapter_run / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["projectionChecks"] = 0
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    report = analyze_batch(root)

    assert report["stageGatePassed"] is False
    assert {issue["kind"] for issue in report["issues"]} == {
        "missing_projection_checks"
    }
