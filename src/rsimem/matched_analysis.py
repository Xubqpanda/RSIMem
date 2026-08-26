"""Content-free acceptance analysis for a matched RSIMem batch."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from .experiment_manifest import EXECUTION_MODES, load_manifest


_METRIC_FIELDS = (
    "withPersistenceScore",
    "withPersistencePassRate",
    "withoutPersistenceScore",
    "withoutPersistencePassRate",
    "persistenceScoreGap",
    "requests",
    "inputTokens",
    "outputTokens",
    "cacheReadTokens",
    "cacheWriteTokens",
    "reasoningTokens",
    "retries",
    "toolCallViews",
    "retrievedRecordViews",
    "injectedChars",
    "peakStoredBytes",
    "wallTimeSeconds",
    "ledgerEvents",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"required matched evidence is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"required matched evidence is not an object: {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"required matched evidence is unreadable: {path.name}") from exc
    events: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("matched ledger contains malformed JSON") from exc
        if not isinstance(event, dict):
            raise ValueError("matched ledger event must be an object")
        events.append(event)
    return events


def _episode_signature(comparison: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for variant in ("with_persistence", "without_persistence"):
        episodes = comparison.get(variant, {}).get("episodes")
        if not isinstance(episodes, list):
            raise ValueError("matched comparison is missing episode evidence")
        result[variant] = [{
            "taskId": episode.get("task_id"),
            "index": episode.get("index"),
            "stage": episode.get("stage"),
            "bucket": episode.get("bucket"),
            "historyMode": episode.get("history_mode"),
            "historyLoadAnchor": episode.get("history_load_anchor"),
            "historySaveAnchor": episode.get("history_save_anchor"),
            "persistenceAllowed": episode.get("persistence_allowed"),
        } for episode in episodes]
    return result


def _initial_state_signature(comparison: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for variant in ("with_persistence", "without_persistence"):
        episodes = comparison[variant]["episodes"]
        if not episodes:
            raise ValueError("matched comparison has no initial episode")
        artifacts = episodes[0].get("artifacts", {})
        result[variant] = {
            "memoryChars": artifacts.get("memory_chars"),
            "userChars": artifacts.get("user_chars"),
            "memoryEntries": len(artifacts.get("memory_entries") or ()),
            "userEntries": len(artifacts.get("user_entries") or ()),
            "skillCount": artifacts.get("skill_count"),
        }
    return result


def _evaluation(comparison: dict[str, Any], variant: str) -> tuple[float, float]:
    evaluation = comparison.get(variant, {}).get("bucket_summary", {}).get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("matched comparison is missing evaluation summary")
    score = evaluation.get("avg_task_score")
    pass_rate = evaluation.get("pass_rate")
    if not isinstance(score, (int, float)) or not isinstance(pass_rate, (int, float)):
        raise ValueError("matched evaluation summary is invalid")
    return float(score), float(pass_rate)


def _wall_time(comparison: dict[str, Any]) -> float:
    seen: set[str] = set()
    total = 0.0
    for variant in ("with_persistence", "without_persistence"):
        for episode in comparison[variant]["episodes"]:
            trace_id = str(episode.get("trace_id") or "")
            if not trace_id or trace_id in seen:
                continue
            seen.add(trace_id)
            total += float(episode.get("timing", {}).get("wall_time_s") or 0.0)
    return round(total, 6)


def _run_metrics(
    comparison: dict[str, Any],
    audit: dict[str, Any],
    ledger: list[dict[str, Any]],
) -> dict[str, int | float | None]:
    with_score, with_pass = _evaluation(comparison, "with_persistence")
    without_score, without_pass = _evaluation(comparison, "without_persistence")
    usage = audit.get("uniquePhysicalUsage", {})
    kinds = Counter(str(event.get("kind")) for event in ledger)
    retrieved_events = [event for event in ledger if event.get("kind") == "retrieved"]
    retrieved = sum(
        int(event.get("data", {}).get("attributes", {}).get("count") or 0)
        for event in retrieved_events
    ) if retrieved_events else None
    injected_chars = sum(
        int(event.get("data", {}).get("contentChars") or 0)
        for event in ledger
        if event.get("kind") == "memory_injection"
    )
    stored = [
        sum(int(event.get("data", {}).get(field) or 0) for field in (
            "memoryFilesBytes", "skillFilesBytes", "stateDbBytes"
        ))
        for event in ledger
        if event.get("kind") == "storage_snapshot"
    ]
    return {
        "withPersistenceScore": with_score,
        "withPersistencePassRate": with_pass,
        "withoutPersistenceScore": without_score,
        "withoutPersistencePassRate": without_pass,
        "persistenceScoreGap": with_score - without_score,
        "requests": usage.get("requests"),
        "inputTokens": usage.get("inputTokens"),
        "outputTokens": usage.get("outputTokens"),
        "cacheReadTokens": usage.get("cacheReadTokens"),
        "cacheWriteTokens": usage.get("cacheWriteTokens"),
        "reasoningTokens": usage.get("reasoningTokens"),
        "retries": usage.get("retries"),
        "toolCallViews": kinds["tool_call"],
        "retrievedRecordViews": retrieved,
        "injectedChars": injected_chars,
        "peakStoredBytes": max(stored, default=0),
        "wallTimeSeconds": _wall_time(comparison),
        "ledgerEvents": len(ledger),
    }


def _summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in _METRIC_FIELDS:
        values = [row["metrics"][field] for row in rows]
        known = [value for value in values if isinstance(value, (int, float))]
        result[field] = {
            "values": values,
            "median": median(known) if known else None,
            "min": min(known) if known else None,
            "max": max(known) if known else None,
        }
    return result


def analyze_batch(batch_root: Path, *, required_replicates: int = 3) -> dict[str, Any]:
    batch_root = batch_root.expanduser().resolve()
    manifest = load_manifest(batch_root / "batch_manifest.json")
    issues: list[dict[str, Any]] = []
    completed = [attempt for attempt in manifest["attempts"] if attempt["status"] == "completed"]
    failed = [
        {
            "replicate": attempt["replicate"],
            "mode": attempt["mode"],
            "attemptNumber": attempt["attemptNumber"],
            "failureStage": attempt["failureStage"],
            "outputDirectory": attempt["outputDirectory"],
        }
        for attempt in manifest["attempts"]
        if attempt["status"] == "failed"
    ]
    running = [attempt for attempt in manifest["attempts"] if attempt["status"] == "running"]
    if running:
        issues.append({"kind": "running_attempts", "count": len(running)})

    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reference_episode_signature: dict[str, list[dict[str, Any]]] | None = None
    reference_initial_state: dict[str, dict[str, Any]] | None = None
    for attempt in completed:
        run_dir = batch_root / attempt["outputDirectory"]
        comparison = _read_json(run_dir / "sequence_comparison.json")
        audit = _read_json(run_dir / "audit.json")
        ledger = _read_jsonl(run_dir / "ledger.jsonl")
        if audit.get("ok") is not True or audit.get("issues") != []:
            issues.append({
                "kind": "audit_failed",
                "runName": attempt["runName"],
                "issueCount": len(audit.get("issues") or ()),
            })
        for field in (
            "projectionMismatches",
            "adapterNativeBypasses",
            "unresolvedMemoryInjections",
        ):
            if int(audit.get(field) or 0) != 0:
                issues.append({
                    "kind": "nonzero_integrity_counter",
                    "runName": attempt["runName"],
                    "field": field,
                    "value": audit.get(field),
                })
        if attempt["mode"] == "native+adapter+ledger" and int(audit.get("projectionChecks") or 0) < 1:
            issues.append({
                "kind": "missing_projection_checks",
                "runName": attempt["runName"],
            })

        episode_signature = _episode_signature(comparison)
        initial_state = _initial_state_signature(comparison)
        if reference_episode_signature is None:
            reference_episode_signature = episode_signature
            reference_initial_state = initial_state
        else:
            if episode_signature != reference_episode_signature:
                issues.append({"kind": "episode_signature_mismatch", "runName": attempt["runName"]})
            if initial_state != reference_initial_state:
                issues.append({"kind": "initial_state_mismatch", "runName": attempt["runName"]})

        by_mode[attempt["mode"]].append({
            "replicate": attempt["replicate"],
            "runName": attempt["runName"],
            "actualOrdinal": attempt["actualOrdinal"],
            "projectionChecks": int(audit.get("projectionChecks") or 0),
            "metrics": _run_metrics(comparison, audit, ledger),
        })

    for mode in EXECUTION_MODES:
        rows = by_mode[mode]
        rows.sort(key=lambda row: row["replicate"])
        if len(rows) != required_replicates:
            issues.append({
                "kind": "insufficient_completed_replicates",
                "mode": mode,
                "required": required_replicates,
                "actual": len(rows),
            })

    return {
        "schemaVersion": 1,
        "experimentId": manifest["experimentId"],
        "stageGatePassed": not issues,
        "issues": issues,
        "scheduledOrder": manifest["executionOrderByReplicate"],
        "failedAttempts": failed,
        "successfulRuns": {
            mode: by_mode[mode]
            for mode in EXECUTION_MODES
        },
        "summaryByMode": {
            mode: {
                "sampleSize": len(by_mode[mode]),
                "metrics": _summaries(by_mode[mode]) if by_mode[mode] else {},
            }
            for mode in EXECUTION_MODES
        },
        "attributionRule": (
            "Exact same-call native shadow checks gate adapter causation; "
            "remaining output/resource differences are independent-unseeded variation."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = analyze_batch(args.batch_root)
    rendered = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["stageGatePassed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
