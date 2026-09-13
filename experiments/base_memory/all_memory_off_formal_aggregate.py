"""Freeze an independently auditable AllMemoryOff result ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Mapping

from .all_memory_off_formal_launcher import (
    CONDITION,
    _accepted_batch_roots,
    _load,
    audit_replicate_batch,
)


SCHEMA = "rsimem-all-memory-off-result-ledger-v1"
GROUPS = {
    "SM": "Semantic",
    "EP": "Episodic",
    "PC": "Procedural",
    "PG": "Proactive retrieval",
}


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _mean_sd(values: list[float]) -> dict[str, object]:
    if not values:
        raise ValueError("cannot summarize an empty value list")
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _group(family_id: str) -> str:
    prefix = family_id.split("_", 1)[0][:2]
    try:
        return GROUPS[prefix]
    except KeyError as exc:
        raise ValueError(f"unknown AllMemoryOff family group: {family_id}") from exc


def _evaluation(run_root: Path, run_manifest: Mapping[str, object]) -> dict[str, object]:
    trace = run_manifest.get("trace_directory")
    if not isinstance(trace, str):
        raise ValueError("AllMemoryOff run manifest lacks trace_directory")
    sequence = _load(run_root / trace / "sequence_results.json")
    episodes = sequence.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("AllMemoryOff sequence has no episodes")
    rows: list[dict[str, object]] = []
    by_distance: dict[str, list[float]] = {"near": [], "far": []}
    for episode in episodes:
        if not isinstance(episode, Mapping) or episode.get("bucket") != "evaluation":
            continue
        task_id = episode.get("task_id")
        distance = episode.get("transfer_distance")
        score = episode.get("task_score")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("AllMemoryOff evaluation episode lacks task_id")
        if distance not in by_distance:
            raise ValueError(f"AllMemoryOff evaluation has invalid distance: {distance}")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError(f"AllMemoryOff evaluation score is invalid: {task_id}")
        value = float(score)
        by_distance[str(distance)].append(value)
        rows.append({
            "task_id": task_id,
            "task_name": episode.get("task_name"),
            "transfer_distance": distance,
            "task_score": value,
        })
    if not rows or any(not values for values in by_distance.values()):
        raise ValueError("AllMemoryOff family must contain both Near and Far evaluations")
    return {
        "tasks": rows,
        "near": statistics.fmean(by_distance["near"]),
        "far": statistics.fmean(by_distance["far"]),
        "evaluation_mean": statistics.fmean(by_distance["near"] + by_distance["far"]),
    }


def aggregate_formal_suite(*, formal_manifest: Path, output_root: Path) -> dict[str, object]:
    """Re-audit every accepted family and build the frozen ledger."""
    manifest = _load(formal_manifest)
    if manifest.get("condition") != CONDITION.value or manifest.get("replicate_count") != 3:
        raise ValueError("formal manifest is not the frozen AllMemoryOff protocol")
    expected = [
        str(record["family_id"])
        for record in manifest.get("families", ())
        if isinstance(record, Mapping) and isinstance(record.get("family_id"), str)
    ]
    if len(expected) != 26 or len(set(expected)) != 26:
        raise ValueError("formal manifest must contain 26 unique families")
    accepted = _accepted_batch_roots(
        formal_manifest_digest=str(manifest["formal_manifest_digest"]),
        output_root=output_root,
    )
    if set(accepted) != set(expected):
        missing = sorted(set(expected) - set(accepted))
        extra = sorted(set(accepted) - set(expected))
        raise ValueError(f"accepted AllMemoryOff families mismatch; missing={missing}, extra={extra}")

    replicates: list[dict[str, object]] = []
    family_summaries: list[dict[str, object]] = []
    for family_id in expected:
        batch_root = accepted[family_id]
        audit = audit_replicate_batch(batch_root)
        persisted_audit = _load(batch_root / "batch_audit.json")
        if persisted_audit != audit:
            raise ValueError(f"persisted AllMemoryOff audit differs: {family_id}")
        batch = _load(batch_root / "batch_manifest.json")
        family_rows: list[dict[str, object]] = []
        for item in audit["replicates"]:
            if not isinstance(item, Mapping):
                raise ValueError("malformed AllMemoryOff replicate audit")
            run_root = batch_root / str(item["run_root"])
            run_manifest = _load(run_root / "run_manifest.json")
            evaluation = _evaluation(run_root, run_manifest)
            usage = item.get("usage")
            if not isinstance(usage, Mapping):
                raise ValueError("AllMemoryOff replicate lacks audited usage")
            row = {
                "family_id": family_id,
                "group": _group(family_id),
                "category": batch.get("category"),
                "replicate": item.get("replicate"),
                "batch_id": batch.get("batch_id"),
                "run_root": str(run_root.resolve()),
                "near_score": evaluation["near"],
                "far_score": evaluation["far"],
                "evaluation_mean": evaluation["evaluation_mean"],
                "evaluation_tasks": evaluation["tasks"],
                "episodes": usage.get("episodes"),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "all_memory_off_evidence": item.get("all_memory_off_evidence"),
            }
            family_rows.append(row)
            replicates.append(row)
        family_summaries.append({
            "family_id": family_id,
            "group": _group(family_id),
            "replicates": family_rows,
            "near": _mean_sd([float(row["near_score"]) for row in family_rows]),
            "far": _mean_sd([float(row["far_score"]) for row in family_rows]),
            "evaluation_mean": _mean_sd([float(row["evaluation_mean"]) for row in family_rows]),
            "input_tokens": _mean_sd([float(row["input_tokens"]) for row in family_rows]),
            "output_tokens": _mean_sd([float(row["output_tokens"]) for row in family_rows]),
        })

    group_summaries: dict[str, object] = {}
    for group in GROUPS.values():
        families = [item for item in family_summaries if item["group"] == group]
        if not families:
            raise ValueError(f"AllMemoryOff group is empty: {group}")
        by_replicate: dict[int, list[float]] = {1: [], 2: [], 3: []}
        for family in families:
            for row in family["replicates"]:
                by_replicate[int(row["replicate"])].append(float(row["evaluation_mean"]))
        replicate_macros = [statistics.fmean(by_replicate[index]) for index in (1, 2, 3)]
        group_summaries[group] = {
            "family_count": len(families),
            "replicate_family_macros": replicate_macros,
            "evaluation_mean": _mean_sd(replicate_macros),
            "near": _mean_sd([
                statistics.fmean([float(row["near_score"]) for row in family["replicates"]])
                for family in families
            ]),
            "far": _mean_sd([
                statistics.fmean([float(row["far_score"]) for row in family["replicates"]])
                for family in families
            ]),
        }
    all_macros = [
        statistics.fmean([
            float(row["evaluation_mean"])
            for row in replicates
            if int(row["replicate"]) == replicate
        ])
        for replicate in (1, 2, 3)
    ]
    group_summaries["Overall"] = {
        "family_count": len(family_summaries),
        "replicate_family_macros": all_macros,
        "evaluation_mean": _mean_sd(all_macros),
        "near": _mean_sd([
            statistics.fmean([
                float(row["near_score"]) for row in replicates if int(row["replicate"]) == replicate
            ])
            for replicate in (1, 2, 3)
        ]),
        "far": _mean_sd([
            statistics.fmean([
                float(row["far_score"]) for row in replicates if int(row["replicate"]) == replicate
            ])
            for replicate in (1, 2, 3)
        ]),
    }

    payload: dict[str, object] = {
        "schema": SCHEMA,
        "accepted": True,
        "condition": CONDITION.value,
        "formal_manifest": str(formal_manifest.resolve()),
        "formal_manifest_digest": manifest["formal_manifest_digest"],
        "family_count": len(family_summaries),
        "accepted_replicate_count": len(replicates),
        "replicate_count_per_family": 3,
        "families": family_summaries,
        "groups": group_summaries,
        "replicates": replicates,
    }
    payload["ledger_digest"] = _digest(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--main-table-output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = aggregate_formal_suite(
        formal_manifest=args.formal_manifest.resolve(),
        output_root=args.output_root.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    table = {
        "schema": "rsimem-all-memory-off-main-table-input-v1",
        "accepted": True,
        "condition": report["condition"],
        "formal_manifest_digest": report["formal_manifest_digest"],
        "ledger_digest": report["ledger_digest"],
        "groups": report["groups"],
        "family_count": report["family_count"],
        "accepted_replicate_count": report["accepted_replicate_count"],
    }
    args.main_table_output.parent.mkdir(parents=True, exist_ok=True)
    args.main_table_output.write_text(json.dumps(table, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"accepted": True, "family_count": report["family_count"], "accepted_replicate_count": report["accepted_replicate_count"], "ledger_digest": report["ledger_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["aggregate_formal_suite"]
