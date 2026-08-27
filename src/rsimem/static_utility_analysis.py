"""Content-free acceptance analysis for a static utility comparison batch."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

from .experiment_manifest import STATIC_UTILITY_METHOD_VARIANTS, load_manifest


_METRICS = (
    "primaryScore",
    "primaryPassRate",
    "persistenceScoreGap",
    "requests",
    "inputTokens",
    "outputTokens",
    "cacheReadTokens",
    "reasoningTokens",
    "retries",
    "wallTimeSeconds",
    "memoryInjections",
    "peakMemoryBytes",
    "ingestionExecutions",
    "ingestionModelRequests",
    "ingestionInputTokens",
    "ingestionOutputTokens",
    "ingestionDurationMs",
    "plannedMutations",
    "utilityExecutions",
    "utilityDecisions",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"required static utility evidence is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"required static utility evidence is not an object: {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"required static utility evidence is unreadable: {path.name}") from exc
    events = []
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("static utility ledger contains malformed JSON") from exc
        if not isinstance(event, dict):
            raise ValueError("static utility ledger event must be an object")
        events.append(event)
    return events


def _signature(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "taskId": episode.get("task_id"),
        "index": episode.get("index"),
        "stage": episode.get("stage"),
        "bucket": episode.get("bucket"),
        "historyMode": episode.get("history_mode"),
        "historyLoadAnchor": episode.get("history_load_anchor"),
        "historySaveAnchor": episode.get("history_save_anchor"),
        "persistenceAllowed": episode.get("persistence_allowed"),
    } for episode in episodes]


def _run_metrics(
    episodes: list[dict[str, Any]],
    audit: dict[str, Any],
    ledger: list[dict[str, Any]],
) -> dict[str, int | float]:
    primary = [episode for episode in episodes if episode.get("bucket") != "reflection"]
    baseline = [episode for episode in primary if episode.get("bucket") == "baseline"]
    evaluation = [episode for episode in primary if episode.get("bucket") == "evaluation"]
    if not primary or len(baseline) != 1 or not evaluation:
        raise ValueError("static utility comparison has an invalid primary episode set")
    scores = [float(episode.get("task_score")) for episode in primary]
    evaluation_score = mean(float(episode.get("task_score")) for episode in evaluation)
    usage = audit.get("uniquePhysicalUsage")
    ingestion = audit.get("ingestionUsage")
    utility = audit.get("staticUtility")
    if not all(isinstance(value, dict) for value in (usage, ingestion, utility)):
        raise ValueError("static utility audit summaries are incomplete")
    storage = [
        int(event.get("data", {}).get("memoryFilesBytes") or 0)
        for event in ledger
        if event.get("kind") == "storage_snapshot"
    ]
    outcomes = ingestion.get("outcomes") or {}
    return {
        "primaryScore": mean(scores),
        "primaryPassRate": mean(bool(episode.get("passed")) for episode in primary),
        "persistenceScoreGap": evaluation_score - float(baseline[0].get("task_score")),
        "requests": int(usage.get("requests") or 0),
        "inputTokens": int(usage.get("inputTokens") or 0),
        "outputTokens": int(usage.get("outputTokens") or 0),
        "cacheReadTokens": int(usage.get("cacheReadTokens") or 0),
        "reasoningTokens": int(usage.get("reasoningTokens") or 0),
        "retries": int(usage.get("retries") or 0),
        "wallTimeSeconds": sum(
            float(episode.get("timing", {}).get("wall_time_s") or 0.0)
            for episode in episodes
        ),
        "memoryInjections": sum(
            1 for event in ledger if event.get("kind") == "memory_injection"
        ),
        "peakMemoryBytes": max(storage, default=0),
        "ingestionExecutions": int(ingestion.get("uniqueExecutions") or 0),
        "ingestionModelRequests": int(ingestion.get("modelRequests") or 0),
        "ingestionInputTokens": int(ingestion.get("inputTokens") or 0),
        "ingestionOutputTokens": int(ingestion.get("outputTokens") or 0),
        "ingestionDurationMs": int(ingestion.get("durationMs") or 0),
        "plannedMutations": int(outcomes.get("planned_mutation") or 0),
        "utilityExecutions": int(utility.get("uniqueExecutions") or 0),
        "utilityDecisions": int(utility.get("decisionCount") or 0),
    }


def _summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for field in _METRICS:
        values = [row["metrics"][field] for row in rows]
        result[field] = {
            "values": values,
            "mean": mean(values),
            "median": median(values),
            "min": min(values),
            "max": max(values),
        }
    return result


def analyze_static_utility_batch(
    batch_root: Path,
    *,
    required_replicates: int = 3,
) -> dict[str, Any]:
    batch_root = batch_root.expanduser().resolve()
    manifest = load_manifest(batch_root / "batch_manifest.json")
    issues: list[dict[str, Any]] = []
    modes = tuple(manifest["configuration"]["executionModes"])
    if modes != STATIC_UTILITY_METHOD_VARIANTS:
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
    utility_identities: dict[str, set[str]] = defaultdict(set)

    for attempt in completed:
        run = batch_root / attempt["outputDirectory"]
        comparison = _read_json(run / "sequence_comparison.json")
        payload = comparison.get("with_persistence")
        episodes = payload.get("episodes") if isinstance(payload, dict) else None
        if not isinstance(episodes, list) or not all(
            isinstance(episode, dict) for episode in episodes
        ):
            raise ValueError("static utility comparison is missing episode evidence")
        audit = _read_json(run / "audit.json")
        ledger = _read_jsonl(run / "ledger.jsonl")
        if audit.get("ok") is not True or audit.get("issues") != []:
            issues.append({"kind": "audit_failed", "runName": attempt["runName"]})
        signature = _signature(episodes)
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            issues.append({
                "kind": "episode_signature_mismatch",
                "runName": attempt["runName"],
            })
        ingestion = audit.get("ingestionUsage") or {}
        utility = audit.get("staticUtility") or {}
        if attempt["mode"] == "static-utility-rsimem":
            if utility.get("uniqueExecutions") != ingestion.get("uniqueExecutions"):
                issues.append({
                    "kind": "utility_ingestion_count_mismatch",
                    "runName": attempt["runName"],
                })
            for field in (
                "gateDigests",
                "gateVersions",
                "featureSchemas",
                "policyVersions",
            ):
                values = utility.get(field)
                if not isinstance(values, list) or len(values) != 1:
                    issues.append({
                        "kind": "utility_identity_not_frozen",
                        "runName": attempt["runName"],
                        "field": field,
                    })
                else:
                    utility_identities[field].add(str(values[0]))
            targets = utility.get("targets") or {}
            if int(targets.get("generation") or 0) < 1 or int(
                targets.get("internal_operation") or 0
            ) < 1:
                issues.append({
                    "kind": "utility_objective_not_exercised",
                    "runName": attempt["runName"],
                })
        elif int(utility.get("events") or 0) != 0:
            issues.append({
                "kind": "baseline_has_utility_evidence",
                "runName": attempt["runName"],
            })
        by_mode[attempt["mode"]].append({
            "replicate": attempt["replicate"],
            "runName": attempt["runName"],
            "actualOrdinal": attempt["actualOrdinal"],
            "metrics": _run_metrics(episodes, audit, ledger),
        })

    for mode in STATIC_UTILITY_METHOD_VARIANTS:
        by_mode[mode].sort(key=lambda item: item["replicate"])
        replicates = [item["replicate"] for item in by_mode[mode]]
        if replicates != list(range(1, required_replicates + 1)):
            issues.append({
                "kind": "incomplete_completed_replicates",
                "mode": mode,
                "replicates": replicates,
            })
    for field, values in utility_identities.items():
        if len(values) > 1:
            issues.append({
                "kind": "utility_identity_changed_across_replicates",
                "field": field,
                "count": len(values),
            })

    return {
        "schemaVersion": 1,
        "experimentId": manifest["experimentId"],
        "stageGatePassed": not issues,
        "issues": issues,
        "scheduledOrder": manifest["executionOrderByReplicate"],
        "revisions": manifest["revisions"],
        "failedAttempts": failed,
        "successfulRuns": {
            mode: by_mode[mode] for mode in STATIC_UTILITY_METHOD_VARIANTS
        },
        "summaryByMode": {
            mode: {
                "sampleSize": len(by_mode[mode]),
                "metrics": _summaries(by_mode[mode]) if by_mode[mode] else {},
            }
            for mode in STATIC_UTILITY_METHOD_VARIANTS
        },
        "utilityIdentity": {
            field: sorted(values) for field, values in utility_identities.items()
        },
        "attributionRule": (
            "Fixed route/boundary/cadence is established by deterministic matched tests; "
            "live score/resource differences are independent-unseeded variation."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = analyze_static_utility_batch(args.batch_root)
    rendered = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["stageGatePassed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
