"""Aggregate the audited B0/B2 refactor validation without mixing B1."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Mapping

from .adamem_batch_audit import audit_batch
from .adamem_experiment import AdaMemCondition


B0 = AdaMemCondition.MEM0_STATIC
B2 = AdaMemCondition.ADAMEM_FULL_TRAJECTORY


def _load(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"aggregation evidence must be an object: {path}")
    return value


def _mean_sd(values: list[float]) -> dict[str, object]:
    return {
        "values": values,
        "mean": statistics.fmean(values) if values else None,
        "sd": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
    }


def _phase(condition: AdaMemCondition, root: Path) -> str:
    return "static" if condition is B0 and (root / "static").is_dir() else "suffix"


def _evaluation_scores(root: Path, condition: AdaMemCondition) -> dict[str, float]:
    result = _load(root / _phase(condition, root) / "sequence_results.json")
    episodes = result.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("sequence results lack episodes")
    scores: dict[str, float] = {}
    for episode in episodes:
        if not isinstance(episode, Mapping) or episode.get("bucket") != "evaluation":
            continue
        task_id = episode.get("task_id")
        score = episode.get("task_score")
        if not isinstance(task_id, str) or task_id in scores:
            raise ValueError("evaluation task IDs must be unique")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError("evaluation task score is invalid")
        scores[task_id] = float(score)
    if not scores:
        raise ValueError("no evaluation episodes found")
    return scores


def _complete_task_usage(root: Path, condition: AdaMemCondition) -> int:
    result = _load(root / _phase(condition, root) / "sequence_results.json")
    episodes = result.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("sequence results lack episodes")
    total = 0
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise ValueError("sequence episode is malformed")
        usage = episode.get("token_usage")
        if not isinstance(usage, Mapping) or usage.get("model_usage_complete") is not True:
            raise ValueError("accepted validation run has incomplete task usage")
        input_tokens = usage.get("input_tokens")
        if type(input_tokens) is not int or input_tokens < 0:
            raise ValueError("accepted validation run has invalid task usage")
        total += input_tokens
    return total


def _replicate_rows(root: Path, condition: AdaMemCondition, audit: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for item in audit.get("replicates", ()):
        if not isinstance(item, Mapping):
            raise ValueError("malformed replicate audit")
        run_id = str(item["run_id"])
        run_root = root / run_id
        outcome = item.get("outcome")
        if not isinstance(outcome, Mapping):
            raise ValueError("replicate lacks policy outcome")
        updater = None
        updater_path = run_root / "updater_usage.json"
        if condition is B2:
            updater = dict(_load(updater_path))
            if updater.get("usage_complete") is not True:
                raise ValueError("B2 updater usage is incomplete")
            for field in ("input_tokens", "output_tokens", "request_count", "latency_ms"):
                value = updater.get(field)
                if type(value) is not int or value < 0:
                    raise ValueError("B2 updater usage is malformed")
        rows.append({
            "condition": condition.value,
            "replicate": item.get("replicate"),
            "run_id": run_id,
            "policy_outcome": outcome.get("outcome"),
            "reason_code": outcome.get("reason_code"),
            "policy_version": outcome.get("candidate_policy_version"),
            "patch_digest": outcome.get("patch_digest"),
            "evaluation_scores": _evaluation_scores(run_root, condition),
            "task_input_tokens": _complete_task_usage(run_root, condition),
            "updater_usage": updater,
        })
    if len(rows) != 3 or {row["replicate"] for row in rows} != {1, 2, 3}:
        raise ValueError("validation aggregate requires three accepted replicates")
    return rows


def aggregate_validation_batches(
    *, family_roots: Mapping[str, Mapping[AdaMemCondition, Path]],
) -> dict[str, object]:
    if not family_roots:
        raise ValueError("validation aggregate requires at least one family")
    family_reports: dict[str, object] = {}
    all_rows: dict[AdaMemCondition, list[dict[str, object]]] = {B0: [], B2: []}
    for family_id, roots in family_roots.items():
        if set(roots) != {B0, B2}:
            raise ValueError("validation aggregate requires exactly B0 and B2")
        audits = {condition: audit_batch(root) for condition, root in roots.items()}
        rows = {condition: _replicate_rows(roots[condition], condition, audits[condition]) for condition in (B0, B2)}
        all_rows[B0].extend(rows[B0])
        all_rows[B2].extend(rows[B2])
        b0_by_rep = {row["replicate"]: row for row in rows[B0]}
        b2_by_rep = {row["replicate"]: row for row in rows[B2]}
        task_ids = sorted(set(b0_by_rep[1]["evaluation_scores"]) & set(b2_by_rep[1]["evaluation_scores"]))
        paired = {
            task: _mean_sd([
                b2_by_rep[rep]["evaluation_scores"][task] - b0_by_rep[rep]["evaluation_scores"][task]
                for rep in (1, 2, 3)
                if task in b0_by_rep[rep]["evaluation_scores"] and task in b2_by_rep[rep]["evaluation_scores"]
            ]) for task in task_ids
        }
        family_reports[family_id] = {
            "accepted": True,
            "conditions": {
                B0.value: rows[B0],
                B2.value: rows[B2],
            },
            "paired_deltas": paired,
        }

    condition_summary: dict[str, object] = {}
    for condition in (B0, B2):
        rows = all_rows[condition]
        scores = [score for row in rows for score in row["evaluation_scores"].values()]
        updater_rows = [row["updater_usage"] for row in rows if row["updater_usage"] is not None]
        outcomes = [row["policy_outcome"] for row in rows]
        updated = sum(value == "updated" for value in outcomes)
        condition_summary[condition.value] = {
            "accepted_replicate_family_runs": len(rows),
            "evaluation_score": _mean_sd([float(value) for value in scores]),
            "task_input_tokens": _mean_sd([float(row["task_input_tokens"]) for row in rows]),
            "update_rate": updated / len(rows) if condition is B2 else None,
            "abstention_rate": sum(value == "no_update" for value in outcomes) / len(rows) if condition is B2 else None,
            "rollback_rate": sum(row["reason_code"] == "rollback" for row in rows) / len(rows) if condition is B2 else None,
            "harmful_update_rate": None,
            "updater_input_tokens": _mean_sd([float(item["input_tokens"]) for item in updater_rows]) if updater_rows else None,
            "updater_output_tokens": _mean_sd([float(item["output_tokens"]) for item in updater_rows]) if updater_rows else None,
            "updater_latency_ms": _mean_sd([float(item["latency_ms"]) for item in updater_rows]) if updater_rows else None,
            "policy_diffs": [row["patch_digest"] for row in rows if row["patch_digest"] is not None],
        }

    b0_rows = {(row["run_id"], row["replicate"]): row for row in all_rows[B0]}
    harmful = 0
    updated_count = 0
    for row in all_rows[B2]:
        if row["policy_outcome"] == "updated":
            updated_count += 1
            b0 = b0_rows.get((row["run_id"], row["replicate"]))
            # Run IDs intentionally differ by condition; match by family/replicate below.
            if b0 is not None:
                common = set(row["evaluation_scores"]) & set(b0["evaluation_scores"])
                if common and statistics.fmean(row["evaluation_scores"][task] for task in common) < statistics.fmean(b0["evaluation_scores"][task] for task in common):
                    harmful += 1
    # The run ID differs across conditions, so family-level reports are the
    # authoritative paired source for harmful-update counting.
    harmful = 0
    for report in family_reports.values():
        b0 = {row["replicate"]: row for row in report["conditions"][B0.value]}
        for row in report["conditions"][B2.value]:
            if row["policy_outcome"] != "updated":
                continue
            common = set(row["evaluation_scores"]) & set(b0[row["replicate"]]["evaluation_scores"])
            if common and statistics.fmean(row["evaluation_scores"][task] for task in common) < statistics.fmean(b0[row["replicate"]]["evaluation_scores"][task] for task in common):
                harmful += 1
    condition_summary[B2.value]["harmful_update_rate"] = harmful / len(all_rows[B2])
    return {
        "schema": "rsimem-adamem-mem0-refactor-validation-aggregate-v1",
        "accepted": True,
        "conditions": condition_summary,
        "families": family_reports,
        "paired_unit": "family-replicate matched evaluation task intersection",
    }


__all__ = ["aggregate_validation_batches"]
