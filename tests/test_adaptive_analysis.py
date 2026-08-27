from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsimem.adaptive_analysis import analyze_adaptive_batch
from rsimem.experiment_manifest import (
    ADAPTIVE_METHOD_VARIANTS,
    execution_order,
    initialize_batch_manifest,
    record_attempt,
)
from rsimem.memory.operation_graph import (
    AppendOnlyOperationEvidenceLog,
    AtomicOperationRecorder,
)
from test_experiment_manifest import _adaptive_profile, _manifest_kwargs
from test_feedback_dataset import _graph


PRIVATE_MEMORY = "Use a private four-column preference."


def _write_operations(run: Path, policy_version: str) -> None:
    graph = _graph("used", policy_version=policy_version)
    sink = AppendOnlyOperationEvidenceLog()
    recorder = AtomicOperationRecorder(sink)
    for artifact in graph.artifacts:
        recorder.record_artifact(artifact)
    for operation in graph.operations:
        recorder.record_operation(operation)
    for mutation in graph.mutations:
        recorder.record_mutation(mutation)
    path = run / "01_fixture" / "artifacts" / "rsimem_semantic_operations.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in sink.events),
        encoding="utf-8",
    )


def _batch(tmp_path: Path) -> Path:
    root = tmp_path / "batch"
    kwargs = _manifest_kwargs()
    kwargs["replicates"] = 3
    profile = _adaptive_profile()
    initialize_batch_manifest(
        root / "batch_manifest.json",
        **kwargs,
        execution_modes=ADAPTIVE_METHOD_VARIANTS,
        adaptive_policy=profile,
        semantic_feedback_contract="sm01_tsv_v1",
    )
    quality = {
        "no-persistence": 0.3,
        "native-hermes": 0.5,
        "native-ledger": 0.5,
        "static-rsimem": 0.7,
        "adaptive-rsimem": 0.8,
    }
    for replicate in range(1, 4):
        for ordinal, mode in enumerate(
            execution_order(replicate, ADAPTIVE_METHOD_VARIANTS),
            1,
        ):
            if replicate == ordinal == 1:
                failed = "r01_failed"
                record_attempt(
                    root / "batch_manifest.json",
                    replicate=replicate,
                    ordinal=ordinal,
                    mode=mode,
                    run_name=failed,
                    status="running",
                )
                record_attempt(
                    root / "batch_manifest.json",
                    replicate=replicate,
                    ordinal=ordinal,
                    mode=mode,
                    run_name=failed,
                    status="failed",
                    failure_stage="provider",
                )
            run_name = f"r{replicate:02d}_{mode.replace('-', '_')}"
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
            score = quality[mode]
            episodes = [{
                "task_id": "task-cold",
                "index": 1,
                "stage": "cold",
                "bucket": "baseline",
                "history_mode": "continue",
                "history_load_anchor": None,
                "history_save_anchor": None,
                "task_score": 0.2,
                "passed": False,
                "hard_pass": False,
                "timing": {"wall_time_s": 1.0},
            }, {
                "task_id": "task-eval",
                "index": 2,
                "stage": "eval_near",
                "bucket": "evaluation",
                "history_mode": "from_anchor",
                "history_load_anchor": "post-learn",
                "history_save_anchor": None,
                "task_score": score,
                "passed": score >= 0.7,
                "hard_pass": score >= 0.8,
                "timing": {"wall_time_s": 2.0},
            }]
            variant = (
                "without_persistence"
                if mode == "no-persistence"
                else "with_persistence"
            )
            (run / "sequence_comparison.json").write_text(
                json.dumps({variant: {"episodes": episodes}}),
                encoding="utf-8",
            )
            policy_version = (
                profile["activePolicyVersion"]
                if mode == "adaptive-rsimem"
                else "static-policy-v1"
            )
            policy_enabled = mode in {"static-rsimem", "adaptive-rsimem"}
            if policy_enabled:
                _write_operations(run, policy_version)
            audit = {
                "ok": True,
                "issues": [],
                "privacy": {
                    "absoluteSourcePaths": 0,
                    "memoryTextLeaks": 0,
                    "credentialPatternHits": {
                        "aws": 0,
                        "bearer": 0,
                        "github": 0,
                        "openai_style": 0,
                    },
                },
                "uniquePhysicalUsage": {
                    "requests": 4,
                    "inputTokens": 100,
                    "outputTokens": 20,
                    "cacheReadTokens": 10,
                    "cacheWriteTokens": 2,
                    "reasoningTokens": 3,
                    "retries": 0,
                },
                "ingestionUsage": ({
                    "uniqueExecutions": 2,
                    "modelRequests": 2,
                    "inputTokens": 40,
                    "outputTokens": 8,
                    "cacheReadTokens": 0,
                    "cacheWriteTokens": 0,
                    "reasoningTokens": 1,
                    "retries": 0,
                    "durationMs": 12,
                } if policy_enabled else None),
                "staticUtility": ({
                    "events": 2,
                    "uniqueExecutions": 2,
                    "decisionCount": 2,
                    "policyVersions": [policy_version],
                } if policy_enabled else {
                    "events": 0,
                    "uniqueExecutions": 0,
                    "decisionCount": 0,
                    "policyVersions": [],
                }),
            }
            (run / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
            ledger = [
                {"kind": "tool_call", "data": {}},
                {"kind": "retrieved", "data": {"attributes": {"count": 1}}},
                {"kind": "memory_injection", "data": {"contentChars": 12}},
                {"kind": "storage_snapshot", "data": {
                    "memoryFilesBytes": 20,
                    "skillFilesBytes": 0,
                    "stateDbBytes": 4,
                }},
            ]
            (run / "ledger.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in ledger),
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


def test_adaptive_analysis_rebuilds_five_method_raw_and_derived_metrics(
    tmp_path: Path,
) -> None:
    report = analyze_adaptive_batch(_batch(tmp_path))

    assert report["stageGatePassed"] is True
    assert report["issues"] == []
    assert len(report["failedAttempts"]) == 1
    assert report["failedAttempts"][0]["failureStage"] == "provider"
    for mode in ADAPTIVE_METHOD_VARIANTS:
        assert report["summaryByMode"][mode]["sampleSize"] == 3
    adaptive = report["summaryByMode"]["adaptive-rsimem"]["metrics"]
    assert adaptive["primaryScore"]["mean"] == 0.5
    assert adaptive["cacheWriteTokens"]["mean"] == 2
    assert adaptive["futureQueries"]["mean"] == 1
    assert adaptive["retrievalOperations"]["mean"] == 1
    assert adaptive["futureUtilityPerCost"]["missingCount"] == 0
    assert report["providerPricing"] is None
    assert len(report["costQualityFrontier"]) == 5
    paired = report["pairedStaticAdaptiveDelta"]
    assert paired["primaryScore"]["mean"] == pytest.approx(0.05)
    assert paired["requests"]["mean"] == 0
    assert report["claimGate"]["memoryMediatedSelfImprovement"][
        "eligible"
    ] is True
    assert report["claimGate"]["recursiveSelfImprovement"]["eligible"] is False
    assert report["claimGate"]["pastBenchGeneralization"]["eligible"] is False
    assert report["claimGate"]["qualitySuperiority"]["eligible"] is False
    assert PRIVATE_MEMORY not in json.dumps(report)


def test_adaptive_analysis_rejects_policy_drift_and_incomplete_replicates(
    tmp_path: Path,
) -> None:
    root = _batch(tmp_path)
    audit_path = root / "r02_adaptive_rsimem" / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["staticUtility"]["policyVersions"] = ["wrong-policy-v1"]
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    report = analyze_adaptive_batch(root, required_replicates=4)

    kinds = {issue["kind"] for issue in report["issues"]}
    assert "adaptive_policy_identity_mismatch" in kinds
    assert "incomplete_completed_replicates" in kinds


def test_adaptive_analysis_rejects_missing_raw_usage_bucket(tmp_path: Path) -> None:
    root = _batch(tmp_path)
    audit_path = root / "r01_static_rsimem" / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    del audit["uniquePhysicalUsage"]["cacheWriteTokens"]
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    try:
        analyze_adaptive_batch(root)
    except ValueError as exc:
        assert "usage accounting is incomplete" in str(exc)
    else:
        raise AssertionError("missing raw usage bucket was accepted")
