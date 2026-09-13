from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.adamem.adamem_batch_audit import audit_batch
from experiments.adamem.adamem_experiment import AdaMemCondition


def _fixture(root: Path, *, count: int = 3) -> None:
    run_ids = [f"run-{i}" for i in range(count)]
    (root / "batch_manifest.json").write_text(json.dumps({"condition": AdaMemCondition.MEM0_STATIC.value, "run_ids": run_ids, "batch_id": "batch"}), encoding="utf-8")
    outcomes = {}
    for i, run_id in enumerate(run_ids, 1):
        path = root / run_id; path.mkdir(parents=True)
        manifest = {"run_id": run_id, "condition": AdaMemCondition.MEM0_STATIC.value, "replicate": i, "state_directory": f"state-{i}", "trace_directory": f"trace-{i}", "artifact_directory": f"artifacts-{i}", "mem0_collection": f"collection-{i}", "port_offset": 20000 + i}
        (path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        for phase in ("prefix", "suffix"):
            (path / phase).mkdir()
            (path / phase / "sequence_results.json").write_text(json.dumps({"episodes": [{"token_usage": {"model_usage_complete": True}}]}), encoding="utf-8")
        outcomes[run_id] = {"outcome": "static"}
    (root / "batch_result.json").write_text(json.dumps({"accepted": True, "outcomes": outcomes}), encoding="utf-8")


def test_accepts_three_complete_replicates(tmp_path: Path) -> None:
    _fixture(tmp_path)
    assert audit_batch(tmp_path)["accepted"] is True


def test_rejects_partial_batch(tmp_path: Path) -> None:
    _fixture(tmp_path, count=2)
    with pytest.raises(ValueError, match="exactly three"):
        audit_batch(tmp_path)
