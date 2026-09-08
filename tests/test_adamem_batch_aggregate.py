import json
from pathlib import Path

import pytest

from rsimem.adamem_batch_aggregate import aggregate_batches
from rsimem.adamem_experiment import AdaMemCondition


def _batch(root: Path, condition: AdaMemCondition, *, count: int = 3) -> None:
    root.mkdir(parents=True, exist_ok=True)
    ids = [f"run-{condition.value}-{i}" for i in range(1, count + 1)]
    (root / "batch_manifest.json").write_text(json.dumps({"condition": condition.value, "run_ids": ids}), encoding="utf-8")
    outcomes = {}
    for i, run_id in enumerate(ids, 1):
        run = root / run_id
        run.mkdir(parents=True)
        (run / "run_manifest.json").write_text(json.dumps({
            "run_id": run_id, "condition": condition.value, "replicate": i,
            "state_directory": f"state-{condition.value}-{i}", "trace_directory": f"trace-{condition.value}-{i}",
            "artifact_directory": f"artifact-{condition.value}-{i}", "mem0_collection": f"collection-{condition.value}-{i}",
            "port_offset": 20000 + i,
        }), encoding="utf-8")
        for phase in ("prefix", "suffix"):
            (run / phase).mkdir()
            (run / phase / "sequence_results.json").write_text(json.dumps({"episodes": [
                {"bucket": "evaluation", "task_id": "eval", "task_score": 1,
                 "token_usage": {"model_usage_complete": True, "input_tokens": 2}}
            ]}), encoding="utf-8")
        outcomes[run_id] = {"outcome": "updated" if condition is AdaMemCondition.ADAMEM_FULL_TRAJECTORY else ("no_update" if condition is AdaMemCondition.ADAMEM_TERMINAL else "static"), "candidate_policy_version": "policy"}
    (root / "batch_result.json").write_text(json.dumps({"accepted": True, "outcomes": outcomes}), encoding="utf-8")


def test_aggregate_requires_all_three_conditions(tmp_path: Path) -> None:
    roots = {}
    for condition in AdaMemCondition:
        roots[condition] = tmp_path / condition.value
        _batch(roots[condition], condition)
    report = aggregate_batches(roots)
    assert report["accepted"] is True
    assert report["conditions"][AdaMemCondition.ADAMEM_FULL_TRAJECTORY.value]["update_rate"] == 1.0


def test_aggregate_rejects_incomplete_condition(tmp_path: Path) -> None:
    roots = {}
    for condition in AdaMemCondition:
        roots[condition] = tmp_path / condition.value
        _batch(roots[condition], condition, count=2 if condition is AdaMemCondition.MEM0_STATIC else 3)
    with pytest.raises(ValueError, match="exactly three"):
        aggregate_batches(roots)
