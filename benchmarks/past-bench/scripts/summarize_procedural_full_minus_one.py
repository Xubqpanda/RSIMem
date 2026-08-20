#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


CONFIGS = (
    "all",
    "minus_planning",
    "minus_memory",
    "minus_skill",
    "minus_gate",
    "minus_closeout",
)


def load_family(path: Path) -> dict:
    comparison_path = path / "sequence_comparison.json"
    with_path = path / "with_persistence" / "sequence_summary.json"
    without_path = path / "without_persistence" / "sequence_summary.json"
    if not all(p.exists() for p in (comparison_path, with_path, without_path)):
        return {"status": "incomplete"}

    comparison = json.loads(comparison_path.read_text())
    with_summary = json.loads(with_path.read_text())
    without_summary = json.loads(without_path.read_text())
    zero_token_episodes = []
    for variant, summary in (
        ("with", with_summary),
        ("without", without_summary),
    ):
        for episode in summary.get("episodes", []):
            usage = episode.get("token_usage", {})
            if usage.get("input_tokens", 0) == 0 and usage.get("output_tokens", 0) == 0:
                zero_token_episodes.append(f"{variant}:{episode.get('task_id', episode.get('label', '?'))}")
    if zero_token_episodes:
        return {"status": "invalid", "zero_token_episodes": zero_token_episodes}

    with_eval = with_summary["bucket_summary"]["evaluation"]["avg_task_score"]
    without_eval = without_summary["bucket_summary"]["evaluation"]["avg_task_score"]
    with_mech = with_summary["benchmark_signal"]["avg_mechanism_score"]
    without_mech = without_summary["benchmark_signal"]["avg_mechanism_score"]
    return {
        "status": "valid",
        "without": without_eval,
        "with": with_eval,
        "delta": comparison["delta"]["evaluation_avg_task_score"],
        "mech_with": with_mech,
        "mech_without": without_mech,
        "mech_delta": comparison["delta"]["avg_mechanism_score"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        default="traces/ablation/procedural_full_minus_one_minimax_m27",
    )
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    aggregate_rows = []

    for config in CONFIGS:
        family_root = root / config / "procedural_ability"
        config_rows = []
        if family_root.exists():
            for path in sorted(family_root.iterdir()):
                if not path.is_dir() or ".invalid_" in path.name:
                    continue
                result = load_family(path)
                row = {"config": config, "family": path.name, **result}
                rows.append(row)
                if result["status"] == "valid":
                    config_rows.append(row)

        aggregate = {
            "config": config,
            "valid_families": len(config_rows),
            "complete": len(config_rows) == 8,
        }
        for field in ("without", "with", "delta", "mech_with", "mech_without", "mech_delta"):
            aggregate[field] = mean(row[field] for row in config_rows) if config_rows else None
        aggregate_rows.append(aggregate)

    root.mkdir(parents=True, exist_ok=True)
    detail_path = root / "procedural_full_minus_one_families.csv"
    with detail_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "config",
                "family",
                "status",
                "without",
                "with",
                "delta",
                "mech_with",
                "mech_without",
                "mech_delta",
                "zero_token_episodes",
            ),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    aggregate_path = root / "procedural_full_minus_one_summary.csv"
    with aggregate_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=aggregate_rows[0].keys())
        writer.writeheader()
        writer.writerows(aggregate_rows)

    markdown_path = root / "procedural_full_minus_one_summary.md"
    lines = [
        "| Configuration | Valid | w/o | w/ | Delta | Mech w/ | Mech Delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        def fmt(field: str) -> str:
            value = row[field]
            return "-" if value is None else f"{value:.4f}"

        lines.append(
            f"| {row['config']} | {row['valid_families']}/8 | {fmt('without')} | "
            f"{fmt('with')} | {fmt('delta')} | {fmt('mech_with')} | {fmt('mech_delta')} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n")

    print(markdown_path.read_text(), end="")
    print(f"wrote {detail_path}")
    print(f"wrote {aggregate_path}")


if __name__ == "__main__":
    main()
