"""Fail-closed aggregation for audited AdaMem comparison batches."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Mapping

from .adamem_batch_audit import audit_batch
from .adamem_experiment import AdaMemCondition


def _load(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"unreadable aggregation evidence: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"aggregation evidence must be an object: {path}")
    return value


def _mean_sd(values: list[float]) -> dict[str, object]:
    if not values:
        return {"values": [], "mean": None, "sd": None}
    return {"values": values, "mean": statistics.fmean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0}


def _episodes(run_root: Path) -> dict[str, Mapping[str, object]]:
    result = _load(run_root / "suffix" / "sequence_results.json")
    episodes = result.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("suffix sequence results lack episodes")
    selected: dict[str, Mapping[str, object]] = {}
    for episode in episodes:
        if not isinstance(episode, Mapping) or episode.get("bucket") != "evaluation":
            continue
        task_id = episode.get("task_id")
        if not isinstance(task_id, str) or task_id in selected:
            raise ValueError("evaluation episodes must have unique task IDs")
        score = episode.get("task_score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError("evaluation task score is invalid")
        selected[task_id] = episode
    if not selected:
        raise ValueError("no evaluation episodes found")
    return selected


def aggregate_batches(batch_roots: Mapping[AdaMemCondition, Path]) -> dict[str, object]:
    if set(batch_roots) != set(AdaMemCondition):
        raise ValueError("aggregation requires exactly B0, B1, and B2 batches")
    audited: dict[AdaMemCondition, dict[str, object]] = {}
    for condition, root in batch_roots.items():
        report = audit_batch(root)
        if report.get("accepted") is not True or len(report.get("replicates", ())) != 3:
            raise ValueError(f"{condition.value} does not have three accepted replicates")
        audited[condition] = report
    rows: list[dict[str, object]] = []
    by_condition: dict[str, list[dict[str, object]]] = {}
    for condition, audit in audited.items():
        root = Path(batch_roots[condition]).resolve()
        condition_rows: list[dict[str, object]] = []
        for item in audit["replicates"]:
            if not isinstance(item, Mapping):
                raise ValueError("malformed replicate audit")
            run_id, replicate = item.get("run_id"), item.get("replicate")
            episodes = _episodes(root / str(run_id))
            outcome = item.get("outcome")
            if not isinstance(outcome, Mapping):
                raise ValueError("replicate lacks policy outcome")
            suffix_result = _load(root / str(run_id) / "suffix" / "sequence_results.json")
            usage = [e.get("token_usage") for e in suffix_result.get("episodes", ()) if isinstance(e, Mapping)]
            complete_usage = [u for u in usage if isinstance(u, Mapping) and u.get("model_usage_complete") is True]
            if len(complete_usage) != len(usage):
                raise ValueError("incomplete model usage in accepted replicate")
            suffix_input_tokens = sum(int(u.get("input_tokens", 0) or 0) for u in complete_usage)
            updater_usage = None
            updater_usage_path = root / str(run_id) / "updater_usage.json"
            if condition is not AdaMemCondition.MEM0_STATIC and updater_usage_path.exists():
                updater_usage = _load(updater_usage_path)
                if updater_usage.get("usage_complete") is not True:
                    raise ValueError("AdaMem updater usage is incomplete")
                for field in ("input_tokens", "output_tokens", "request_count"):
                    value = updater_usage.get(field)
                    if type(value) is not int or value < 0:
                        raise ValueError("AdaMem updater usage is malformed")
            row = {"condition": condition.value, "replicate": replicate, "run_id": run_id,
                   "policy_outcome": outcome.get("outcome"), "policy_version": outcome.get("candidate_policy_version"),
                   "evaluation_scores": {task: ep.get("task_score") for task, ep in sorted(episodes.items())},
                   "suffix_input_tokens": suffix_input_tokens,
                   # Older accepted runs only persist an updater request digest;
                   # never present suffix task usage as updater usage.
                   "updater_input_tokens": (updater_usage.get("input_tokens") if updater_usage else None)}
            condition_rows.append(row); rows.append(row)
        by_condition[condition.value] = condition_rows
    task_ids = sorted({task for row in rows for task in row["evaluation_scores"]})
    summary: dict[str, object] = {"schema": "rsimem-adamem-batch-aggregate-v1", "accepted": True,
        "conditions": {}, "paired_deltas": {}, "replicates": rows}
    for condition in AdaMemCondition:
        condition_rows = by_condition[condition.value]
        scores = {task: _mean_sd([float(row["evaluation_scores"][task]) for row in condition_rows]) for task in task_ids}
        outcomes = [row["policy_outcome"] for row in condition_rows]
        updater = condition is not AdaMemCondition.MEM0_STATIC
        updater_inputs = [row["updater_input_tokens"] for row in condition_rows]
        if updater and any(value is None for value in updater_inputs) and any(value is not None for value in updater_inputs):
            raise ValueError("AdaMem updater usage is only partially available")
        updated = sum(value == "updated" for value in outcomes)
        no_update = sum(value == "no_update" for value in outcomes)
        summary["conditions"][condition.value] = {"replicates": 3, "evaluation_scores": scores,
            "suffix_input_tokens": _mean_sd([float(row["suffix_input_tokens"]) for row in condition_rows]),
            "updater_input_tokens": (
                _mean_sd([float(value) for value in updater_inputs])
                if updater and all(value is not None for value in updater_inputs) else None
            ),
            "updater_attempted": updater, "proposal_rate": (updated + no_update) / 3 if updater else 0.0,
            "abstention_rate": no_update / 3 if updater else None, "acceptance_rate": updated / 3 if updater else None,
            "update_rate": updated / 3 if updater else 0.0, "rollback_rate": 0.0}
    for left, right in ((AdaMemCondition.ADAMEM_TERMINAL, AdaMemCondition.MEM0_STATIC),
                        (AdaMemCondition.ADAMEM_FULL_TRAJECTORY, AdaMemCondition.MEM0_STATIC),
                        (AdaMemCondition.ADAMEM_FULL_TRAJECTORY, AdaMemCondition.ADAMEM_TERMINAL)):
        left_rows = {row["replicate"]: row for row in by_condition[left.value]}
        right_rows = {row["replicate"]: row for row in by_condition[right.value]}
        if set(left_rows) != set(right_rows):
            raise ValueError("paired replicate identities do not match")
        summary["paired_deltas"][f"{left.value}-{right.value}"] = {
            task: _mean_sd([float(left_rows[r]["evaluation_scores"][task]) - float(right_rows[r]["evaluation_scores"][task]) for r in sorted(left_rows)])
            for task in task_ids}
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.output_root.resolve()
    names = {AdaMemCondition.MEM0_STATIC: "sm01-b0-formal-20260908-r1", AdaMemCondition.ADAMEM_TERMINAL: "sm01-b1-formal-20260908-r1", AdaMemCondition.ADAMEM_FULL_TRAJECTORY: "sm01-b2-formal-20260908-r1"}
    report = aggregate_batches({condition: root / name for condition, name in names.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
