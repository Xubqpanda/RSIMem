"""Rebuild a historical five-method threshold-infrastructure PAST batch.

The heterogeneous lifecycle cost scalar is retained only to replay the
superseded experiment. Current extraction-prompt reports use
``extraction_experiment_analysis`` and never consume this scalar.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

from .experiment_manifest import ADAPTIVE_METHOD_VARIANTS, load_manifest
from .memory.operation_graph import (
    AppendOnlyOperationEvidenceLog,
    OperationKind,
    materialize_operation_graph,
)


ADAPTIVE_ANALYSIS_SCHEMA_VERSION = 1
FUTURE_UTILITY_COST_SCHEMA = "adaptive-future-utility-raw-cost-v1"
_USAGE_FIELDS = (
    "requests",
    "inputTokens",
    "outputTokens",
    "cacheReadTokens",
    "cacheWriteTokens",
    "reasoningTokens",
    "retries",
)
_METRIC_FIELDS = (
    "primaryScore",
    "passRate",
    "hardPassRate",
    "persistenceScoreGap",
    *_USAGE_FIELDS,
    "wallTimeSeconds",
    "toolCalls",
    "retrievedRecords",
    "memoryInjections",
    "injectedChars",
    "peakStoredBytes",
    "ingestionExecutions",
    "ingestionModelRequests",
    "ingestionInputTokens",
    "ingestionOutputTokens",
    "ingestionCacheReadTokens",
    "ingestionCacheWriteTokens",
    "ingestionReasoningTokens",
    "ingestionRetries",
    "ingestionDurationMs",
    "utilityExecutions",
    "utilityDecisions",
    "operationArtifacts",
    "memoryMutations",
    "futureQueries",
    "retrievalOperations",
    "injectionOperations",
    "useOperations",
    "supersessionOperations",
    "recoveryOperations",
    "recoveryDurationMs",
    "lifecycleCostUnits",
    "futureUtilityPerCost",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"required adaptive evidence is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"required adaptive evidence is not an object: {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"required adaptive evidence is unreadable: {path.name}") from exc
    events = []
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("adaptive ledger contains malformed JSON") from exc
        if not isinstance(event, dict):
            raise ValueError("adaptive ledger event must be an object")
        events.append(event)
    return events


def _episodes(comparison: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    variant = (
        "without_persistence"
        if mode == "no-persistence"
        else "with_persistence"
    )
    if set(comparison) != {variant}:
        raise ValueError("adaptive run has an unexpected persistence variant")
    episodes = comparison[variant].get("episodes")
    if not isinstance(episodes, list) or not episodes or any(
        not isinstance(episode, dict) for episode in episodes
    ):
        raise ValueError("adaptive run has no episode evidence")
    return episodes


def _episode_signature(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "taskId": episode.get("task_id"),
        "index": episode.get("index"),
        "stage": episode.get("stage"),
        "bucket": episode.get("bucket"),
        "historyMode": episode.get("history_mode"),
        "historyLoadAnchor": episode.get("history_load_anchor"),
        "historySaveAnchor": episode.get("history_save_anchor"),
    } for episode in episodes]


def _operation_metrics(run_dir: Path) -> dict[str, Any]:
    paths = tuple(sorted(
        run_dir.glob("[0-9][0-9]_*/*/rsimem_semantic_operations.jsonl")
    ))
    if not paths:
        return {
            "artifactCount": 0,
            "mutationCount": 0,
            "kindCounts": {kind.value: 0 for kind in OperationKind},
            "recoveryDurationMs": 0,
            "policyVersions": [],
        }
    merged = AppendOnlyOperationEvidenceLog()
    for path in paths:
        for event in AppendOnlyOperationEvidenceLog(path).events:
            merged.append(event)
    graph = materialize_operation_graph(merged.events)
    kinds = Counter(operation.kind for operation in graph.operations)
    return {
        "artifactCount": len(graph.artifacts),
        "mutationCount": len(graph.mutations),
        "kindCounts": {kind.value: kinds[kind] for kind in OperationKind},
        "recoveryDurationMs": sum(
            operation.latency_ms
            for operation in graph.operations
            if operation.kind == OperationKind.RECOVERY
        ),
        "policyVersions": sorted({
            operation.context.policy_version for operation in graph.operations
        }),
    }


def _known_nonnegative_ints(value: object, fields: Sequence[str]) -> bool:
    return isinstance(value, dict) and set(value) == set(fields) and all(
        type(value.get(field)) is int and value[field] >= 0 for field in fields
    )


def _run_metrics(
    episodes: list[dict[str, Any]],
    audit: dict[str, Any],
    ledger: list[dict[str, Any]],
    operations: dict[str, Any],
) -> dict[str, int | float | None]:
    primary = [episode for episode in episodes if episode.get("bucket") != "reflection"]
    baseline = [episode for episode in primary if episode.get("bucket") == "baseline"]
    evaluation = [episode for episode in primary if episode.get("bucket") == "evaluation"]
    if not primary or len(baseline) != 1 or not evaluation:
        raise ValueError("adaptive run has an invalid primary episode set")
    for episode in primary:
        if not isinstance(episode.get("task_score"), (int, float)) or (
            type(episode.get("passed")) is not bool
        ):
            raise ValueError("adaptive run quality evidence is incomplete")
    usage = audit.get("uniquePhysicalUsage")
    if not _known_nonnegative_ints(usage, _USAGE_FIELDS):
        raise ValueError("adaptive run usage accounting is incomplete")
    ingestion = audit.get("ingestionUsage")
    utility = audit.get("staticUtility")
    ledger_kinds = Counter(str(event.get("kind")) for event in ledger)
    retrieved_events = [event for event in ledger if event.get("kind") == "retrieved"]
    retrieved = (
        sum(
            int(event.get("data", {}).get("attributes", {}).get("count") or 0)
            for event in retrieved_events
        )
        if retrieved_events
        else None
    )
    injected_chars = sum(
        int(event.get("data", {}).get("contentChars") or 0)
        for event in ledger
        if event.get("kind") == "memory_injection"
    )
    stored = [
        sum(int(event.get("data", {}).get(field) or 0) for field in (
            "memoryFilesBytes",
            "skillFilesBytes",
            "stateDbBytes",
        ))
        for event in ledger
        if event.get("kind") == "storage_snapshot"
    ]
    evaluation_score = mean(
        float(episode["task_score"]) for episode in evaluation
    )
    persistence_gap = evaluation_score - float(baseline[0]["task_score"])
    ingestion_fields = {
        "ingestionExecutions": "uniqueExecutions",
        "ingestionModelRequests": "modelRequests",
        "ingestionInputTokens": "inputTokens",
        "ingestionOutputTokens": "outputTokens",
        "ingestionCacheReadTokens": "cacheReadTokens",
        "ingestionCacheWriteTokens": "cacheWriteTokens",
        "ingestionReasoningTokens": "reasoningTokens",
        "ingestionRetries": "retries",
        "ingestionDurationMs": "durationMs",
    }
    ingestion_values = {
        output: (
            value
            if isinstance(ingestion, dict)
            and (value := ingestion.get(source)) is not None
            else None
        )
        for output, source in ingestion_fields.items()
    }
    utility_executions = (
        utility.get("uniqueExecutions") if isinstance(utility, dict) else None
    )
    utility_decisions = (
        utility.get("decisionCount") if isinstance(utility, dict) else None
    )
    lifecycle_cost = (
        1
        + sum(usage[field] for field in _USAGE_FIELDS)
        + max(stored, default=0)
        + injected_chars
        + int(operations["recoveryDurationMs"])
    )
    kinds = operations["kindCounts"]
    return {
        "primaryScore": mean(float(episode["task_score"]) for episode in primary),
        "passRate": mean(bool(episode["passed"]) for episode in primary),
        "hardPassRate": mean(bool(episode.get("hard_pass")) for episode in primary),
        "persistenceScoreGap": persistence_gap,
        **{field: usage[field] for field in _USAGE_FIELDS},
        "wallTimeSeconds": round(sum(
            float(episode.get("timing", {}).get("wall_time_s") or 0.0)
            for episode in episodes
        ), 6),
        "toolCalls": ledger_kinds["tool_call"],
        "retrievedRecords": retrieved,
        "memoryInjections": ledger_kinds["memory_injection"],
        "injectedChars": injected_chars,
        "peakStoredBytes": max(stored, default=0),
        **ingestion_values,
        "utilityExecutions": utility_executions,
        "utilityDecisions": utility_decisions,
        "operationArtifacts": operations["artifactCount"],
        "memoryMutations": operations["mutationCount"],
        "futureQueries": kinds[OperationKind.FUTURE_QUERY.value],
        "retrievalOperations": kinds[OperationKind.RETRIEVAL.value],
        "injectionOperations": kinds[OperationKind.INJECTION.value],
        "useOperations": kinds[OperationKind.USE.value],
        "supersessionOperations": kinds[OperationKind.SUPERSESSION.value],
        "recoveryOperations": kinds[OperationKind.RECOVERY.value],
        "recoveryDurationMs": operations["recoveryDurationMs"],
        "lifecycleCostUnits": lifecycle_cost,
        "futureUtilityPerCost": persistence_gap / lifecycle_cost,
    }


def _summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for field in _METRIC_FIELDS:
        values = [row["metrics"][field] for row in rows]
        known = [
            value for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        result[field] = {
            "values": values,
            "mean": mean(known) if known else None,
            "median": median(known) if known else None,
            "min": min(known) if known else None,
            "max": max(known) if known else None,
            "missingCount": len(values) - len(known),
        }
    return result


def _paired_static_adaptive_deltas(
    by_mode: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    static = {row["replicate"]: row for row in by_mode["static-rsimem"]}
    adaptive = {row["replicate"]: row for row in by_mode["adaptive-rsimem"]}
    if set(static) != set(adaptive):
        return {}
    fields = (
        "primaryScore",
        "passRate",
        "hardPassRate",
        "persistenceScoreGap",
        *_USAGE_FIELDS,
        "wallTimeSeconds",
        "peakStoredBytes",
        "ingestionModelRequests",
        "ingestionInputTokens",
        "ingestionOutputTokens",
        "lifecycleCostUnits",
        "futureUtilityPerCost",
    )
    result = {}
    for field in fields:
        values = []
        for replicate in sorted(static):
            left = static[replicate]["metrics"][field]
            right = adaptive[replicate]["metrics"][field]
            values.append(
                right - left
                if isinstance(left, (int, float))
                and not isinstance(left, bool)
                and isinstance(right, (int, float))
                and not isinstance(right, bool)
                else None
            )
        known = [value for value in values if value is not None]
        result[field] = {
            "values": values,
            "mean": mean(known) if known else None,
            "median": median(known) if known else None,
            "min": min(known) if known else None,
            "max": max(known) if known else None,
            "missingCount": len(values) - len(known),
        }
    return result


def analyze_adaptive_batch(
    batch_root: Path,
    *,
    required_replicates: int = 3,
) -> dict[str, Any]:
    batch_root = batch_root.expanduser().resolve()
    manifest = load_manifest(batch_root / "batch_manifest.json")
    issues: list[dict[str, Any]] = []
    modes = tuple(manifest["configuration"]["executionModes"])
    if modes != ADAPTIVE_METHOD_VARIANTS:
        issues.append({"kind": "unexpected_method_set", "modes": list(modes)})
    running = [item for item in manifest["attempts"] if item["status"] == "running"]
    if running:
        issues.append({"kind": "running_attempts", "count": len(running)})
    failed = [{
        "replicate": item["replicate"],
        "mode": item["mode"],
        "attemptNumber": item["attemptNumber"],
        "failureStage": item["failureStage"],
        "outputDirectory": item["outputDirectory"],
    } for item in manifest["attempts"] if item["status"] == "failed"]
    completed = [item for item in manifest["attempts"] if item["status"] == "completed"]
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reference_signature = None
    static_policy_versions: set[str] = set()
    active_version = (
        manifest["configuration"].get("adaptivePolicy") or {}
    ).get("activePolicyVersion")

    for attempt in completed:
        run_dir = (batch_root / attempt["outputDirectory"]).resolve()
        if not run_dir.is_relative_to(batch_root):
            raise ValueError("adaptive run directory escapes batch")
        comparison = _read_json(run_dir / "sequence_comparison.json")
        episodes = _episodes(comparison, attempt["mode"])
        audit = _read_json(run_dir / "audit.json")
        ledger = _read_jsonl(run_dir / "ledger.jsonl")
        if audit.get("ok") is not True or audit.get("issues") != []:
            issues.append({"kind": "audit_failed", "runName": attempt["runName"]})
        privacy = audit.get("privacy")
        if (
            not isinstance(privacy, dict)
            or privacy.get("absoluteSourcePaths") != 0
            or privacy.get("memoryTextLeaks") != 0
            or any((privacy.get("credentialPatternHits") or {}).values())
        ):
            issues.append({"kind": "privacy_audit_failed", "runName": attempt["runName"]})
        signature = _episode_signature(episodes)
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            issues.append({
                "kind": "episode_signature_mismatch",
                "runName": attempt["runName"],
            })
        operations = _operation_metrics(run_dir)
        utility = audit.get("staticUtility") or {}
        if attempt["mode"] in {"static-rsimem", "adaptive-rsimem"}:
            ingestion = audit.get("ingestionUsage")
            required_ingestion = (
                "uniqueExecutions",
                "modelRequests",
                "inputTokens",
                "outputTokens",
                "cacheReadTokens",
                "cacheWriteTokens",
                "reasoningTokens",
                "retries",
                "durationMs",
            )
            if not isinstance(ingestion, dict) or any(
                type(ingestion.get(field)) is not int or ingestion[field] < 0
                for field in required_ingestion
            ):
                raise ValueError("adaptive ingestion accounting is incomplete")
            if (
                int(utility.get("events") or 0) < 1
                or int(utility.get("uniqueExecutions") or 0) < 1
                or len(utility.get("policyVersions") or ()) != 1
                or not operations["policyVersions"]
            ):
                issues.append({
                    "kind": "missing_policy_evidence",
                    "runName": attempt["runName"],
                })
            elif attempt["mode"] == "adaptive-rsimem":
                if (
                    utility["policyVersions"] != [active_version]
                    or operations["policyVersions"] != [active_version]
                ):
                    issues.append({
                        "kind": "adaptive_policy_identity_mismatch",
                        "runName": attempt["runName"],
                    })
            else:
                static_policy_versions.update(utility["policyVersions"])
                if operations["policyVersions"] != utility["policyVersions"]:
                    issues.append({
                        "kind": "static_policy_identity_mismatch",
                        "runName": attempt["runName"],
                    })
        elif int(utility.get("events") or 0) != 0 or operations["policyVersions"]:
            issues.append({
                "kind": "control_has_policy_evidence",
                "runName": attempt["runName"],
            })
        by_mode[attempt["mode"]].append({
            "replicate": attempt["replicate"],
            "runName": attempt["runName"],
            "actualOrdinal": attempt["actualOrdinal"],
            "metrics": _run_metrics(episodes, audit, ledger, operations),
            "operationKindCounts": operations["kindCounts"],
        })

    for mode in ADAPTIVE_METHOD_VARIANTS:
        by_mode[mode].sort(key=lambda item: item["replicate"])
        replicates = [item["replicate"] for item in by_mode[mode]]
        if replicates != list(range(1, required_replicates + 1)):
            issues.append({
                "kind": "incomplete_completed_replicates",
                "mode": mode,
                "replicates": replicates,
            })
    if len(static_policy_versions) != 1:
        issues.append({
            "kind": "static_policy_changed_across_replicates",
            "versions": sorted(static_policy_versions),
        })

    summaries = {
        mode: {
            "sampleSize": len(by_mode[mode]),
            "metrics": _summaries(by_mode[mode]) if by_mode[mode] else {},
        }
        for mode in ADAPTIVE_METHOD_VARIANTS
    }
    frontier = [{
        "mode": mode,
        "quality": summaries[mode]["metrics"].get("primaryScore", {}).get("mean"),
        "lifecycleCostUnits": summaries[mode]["metrics"].get(
            "lifecycleCostUnits", {}
        ).get("mean"),
        "futureUtilityPerCost": summaries[mode]["metrics"].get(
            "futureUtilityPerCost", {}
        ).get("mean"),
    } for mode in ADAPTIVE_METHOD_VARIANTS]
    paired_deltas = _paired_static_adaptive_deltas(by_mode)
    stage_passed = not issues
    quality_delta = paired_deltas.get("primaryScore", {}).get("mean")
    implementation_claim = stage_passed
    return {
        "schemaVersion": ADAPTIVE_ANALYSIS_SCHEMA_VERSION,
        "experimentId": manifest["experimentId"],
        "stageGatePassed": stage_passed,
        "issues": issues,
        "scheduledOrder": manifest["executionOrderByReplicate"],
        "revisions": manifest["revisions"],
        "modelProfile": manifest["configuration"]["model"],
        "judgeProfile": manifest["configuration"]["judge"],
        "budget": manifest["configuration"]["budget"],
        "failedAttempts": failed,
        "successfulRuns": {
            mode: by_mode[mode] for mode in ADAPTIVE_METHOD_VARIANTS
        },
        "summaryByMode": summaries,
        "costQualityFrontier": frontier,
        "pairedStaticAdaptiveDelta": paired_deltas,
        "futureUtilityCostSchema": FUTURE_UTILITY_COST_SCHEMA,
        "providerPricing": None,
        "policyIdentity": {
            "staticPolicyVersions": sorted(static_policy_versions),
            "adaptivePolicyVersion": active_version,
        },
        "adaptiveGainBudgetAudit": {
            "configuredBudgetMatched": True,
            "realizedRequestDelta": paired_deltas.get("requests", {}).get("mean"),
            "realizedInputTokenDelta": paired_deltas.get("inputTokens", {}).get("mean"),
            "realizedOutputTokenDelta": paired_deltas.get("outputTokens", {}).get("mean"),
            "realizedLifecycleCostDelta": paired_deltas.get(
                "lifecycleCostUnits", {}
            ).get("mean"),
        },
        "claimGate": {
            "fixedRouteSemanticMemoryOptimization": {
                "eligible": implementation_claim,
                "reasonCodes": [] if implementation_claim else ["batch_gate_failed"],
            },
            "unifiedMemoryPolicyObjective": {
                "eligible": implementation_claim,
                "reasonCodes": [] if implementation_claim else ["batch_gate_failed"],
            },
            "operationAttributedPolicyImprovement": {
                "eligible": implementation_claim,
                "reasonCodes": [] if implementation_claim else ["batch_gate_failed"],
            },
            "memoryMediatedSelfImprovement": {
                "eligible": bool(
                    implementation_claim
                    and isinstance(quality_delta, (int, float))
                    and quality_delta > 0
                ),
                "reasonCodes": (
                    []
                    if implementation_claim
                    and isinstance(quality_delta, (int, float))
                    and quality_delta > 0
                    else ["no_positive_paired_quality_delta"]
                    if implementation_claim
                    else ["batch_gate_failed"]
                ),
            },
            "recursiveSelfImprovement": {
                "eligible": False,
                "reasonCodes": ["second_replayable_iteration_missing"],
            },
            "pastBenchGeneralization": {
                "eligible": False,
                "reasonCodes": ["multiple_predeclared_families_missing"],
            },
            "qualitySuperiority": {
                "eligible": False,
                "reasonCodes": ["statistical_claim_gate_not_satisfied"],
            },
        },
        "claimBoundary": (
            "This report rebuilds one SM01 five-method batch. It does not by itself "
            "establish recursive iteration, cross-family generalization, or causality."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--required-replicates", type=int, default=3)
    args = parser.parse_args(argv)
    report = analyze_adaptive_batch(
        args.batch_root,
        required_replicates=args.required_replicates,
    )
    rendered = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["stageGatePassed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
