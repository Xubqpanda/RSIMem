from __future__ import annotations

import json
from pathlib import Path

from rsimem.experiment_manifest import (
    STATIC_UTILITY_METHOD_VARIANTS,
    execution_order,
    initialize_batch_manifest,
    record_attempt,
)
from rsimem.static_utility_analysis import analyze_static_utility_batch


PRIVATE_MEMORY = "Always use a private formatting preference."


def _batch(tmp_path: Path) -> Path:
    root = tmp_path / "batch"
    initialize_batch_manifest(
        root / "batch_manifest.json",
        replicates=3,
        task_family="memory_ability/SM01_preference_adoption",
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
            "compareNoPersistence": False,
        },
        adapter_projection_verification=True,
        rsimem_commit="head",
        rsimem_working_tree_dirty=False,
        past_bench_commit="past",
        past_bench_tree="tree",
        past_bench_dirty=False,
        execution_modes=STATIC_UTILITY_METHOD_VARIANTS,
    )
    for replicate in range(1, 4):
        for ordinal, mode in enumerate(
            execution_order(replicate, STATIC_UTILITY_METHOD_VARIANTS),
            1,
        ):
            if replicate == ordinal == 1:
                failed_name = "r1_static_failed"
                record_attempt(
                    root / "batch_manifest.json",
                    replicate=replicate,
                    ordinal=ordinal,
                    mode=mode,
                    run_name=failed_name,
                    status="running",
                )
                record_attempt(
                    root / "batch_manifest.json",
                    replicate=replicate,
                    ordinal=ordinal,
                    mode=mode,
                    run_name=failed_name,
                    status="failed",
                    failure_stage="launcher_timeout",
                )
            run_name = f"r{replicate}_{mode.replace('-', '_')}"
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
            episodes = [{
                "task_id": "task-cold",
                "trace_id": f"trace-{replicate}-{mode}-cold",
                "index": 1,
                "stage": "cold",
                "bucket": "baseline",
                "history_mode": "fresh",
                "history_load_anchor": None,
                "history_save_anchor": None,
                "persistence_allowed": True,
                "task_score": 0.25,
                "passed": False,
                "timing": {"wall_time_s": 1.0},
            }, {
                "task_id": "task-eval",
                "trace_id": f"trace-{replicate}-{mode}-eval",
                "index": 2,
                "stage": "eval",
                "bucket": "evaluation",
                "history_mode": "fresh",
                "history_load_anchor": "post-learn",
                "history_save_anchor": None,
                "persistence_allowed": True,
                "task_score": 0.75 if mode == "static-utility-rsimem" else 0.5,
                "passed": mode == "static-utility-rsimem",
                "timing": {"wall_time_s": 2.0},
            }]
            (run / "sequence_comparison.json").write_text(
                json.dumps({"with_persistence": {"episodes": episodes}}),
                encoding="utf-8",
            )
            utility_enabled = mode == "static-utility-rsimem"
            utility = {
                "events": 1 if utility_enabled else 0,
                "uniqueExecutions": 1 if utility_enabled else 0,
                "decisionCount": 2 if utility_enabled else 0,
                "targets": (
                    {"generation": 1, "internal_operation": 1}
                    if utility_enabled
                    else {}
                ),
                "gateDigests": ["a" * 64] if utility_enabled else [],
                "gateVersions": ["gate-v1"] if utility_enabled else [],
                "featureSchemas": ["features-v1"] if utility_enabled else [],
                "policyVersions": ["policy-v1"] if utility_enabled else [],
            }
            audit = {
                "ok": True,
                "issues": [],
                "uniquePhysicalUsage": {
                    "requests": 4,
                    "inputTokens": 100,
                    "outputTokens": 20,
                    "cacheReadTokens": 10,
                    "reasoningTokens": 3,
                    "retries": 0,
                },
                "ingestionUsage": {
                    "uniqueExecutions": 1,
                    "modelRequests": 2,
                    "inputTokens": 40,
                    "outputTokens": 8,
                    "durationMs": 12,
                    "outcomes": {"planned_mutation": 1},
                },
                "staticUtility": utility,
            }
            (run / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
            events = [{
                "kind": "storage_snapshot",
                "data": {"memoryFilesBytes": 24},
            }]
            if utility_enabled:
                events.append({"kind": "memory_injection", "data": {}})
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


def test_static_utility_analysis_accepts_complete_rotated_batch(tmp_path: Path) -> None:
    report = analyze_static_utility_batch(_batch(tmp_path))

    assert report["stageGatePassed"] is True
    assert report["issues"] == []
    assert len(report["failedAttempts"]) == 1
    assert report["failedAttempts"][0]["failureStage"] == "launcher_timeout"
    for mode in STATIC_UTILITY_METHOD_VARIANTS:
        assert report["summaryByMode"][mode]["sampleSize"] == 3
    utility = report["summaryByMode"]["static-utility-rsimem"]["metrics"]
    assert utility["utilityExecutions"]["mean"] == 1
    assert utility["primaryScore"]["mean"] == 0.5
    assert PRIVATE_MEMORY not in json.dumps(report)


def test_static_utility_analysis_rejects_cross_replicate_policy_drift(
    tmp_path: Path,
) -> None:
    root = _batch(tmp_path)
    path = root / "r2_static_utility_rsimem" / "audit.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    audit["staticUtility"]["gateDigests"] = ["b" * 64]
    path.write_text(json.dumps(audit), encoding="utf-8")

    report = analyze_static_utility_batch(root)

    assert report["stageGatePassed"] is False
    assert {issue["kind"] for issue in report["issues"]} == {
        "utility_identity_changed_across_replicates"
    }
