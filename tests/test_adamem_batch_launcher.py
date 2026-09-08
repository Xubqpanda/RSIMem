from __future__ import annotations

import json
from pathlib import Path

import pytest

import rsimem.adamem_batch_launcher as launcher
from rsimem.adamem_experiment import AdaMemCondition


def _inputs(tmp_path: Path):
    sequence = tmp_path / "SM01.yaml"
    sequence.write_text("name: SM01\nepisodes: []\n", encoding="utf-8")
    config = tmp_path / "config"; config.write_text("config", encoding="utf-8")
    registry = tmp_path / "registry"; registry.write_text("registry", encoding="utf-8")
    return sequence, config, registry


def test_batch_manifest_has_three_isolated_replicates(tmp_path: Path) -> None:
    sequence, config, registry = _inputs(tmp_path)
    manifest = launcher.build_batch_manifest(
        source_sequence=sequence, family_id="SM01", config=config, registry=registry,
    )
    runs = [run for run in manifest.runs if run.condition is AdaMemCondition.MEM0_STATIC]
    assert len(runs) == 3
    assert len({run.mem0_collection for run in runs}) == 3
    assert manifest.base_model == "gpt-5.6-luna"


def test_formal_batch_rejects_other_concurrency(tmp_path: Path) -> None:
    sequence, config, registry = _inputs(tmp_path)
    with pytest.raises(ValueError, match="fixed at three"):
        launcher.run_replicate_batch(
            source_sequence=sequence, family_id="SM01", condition=AdaMemCondition.MEM0_STATIC,
            output_root=tmp_path, past_bin=tmp_path / "past", past_root=tmp_path,
            config=config, registry=registry, base_url="https://example.test/v1",
            api_key=None, batch_id="batch", cutover_label="learn-a", max_workers=2,
        )


def test_batch_writes_manifest_before_replicate_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sequence, config, registry = _inputs(tmp_path)
    seen: list[Path] = []
    def fake_run(**kwargs):
        path = kwargs["output_root"] / kwargs["run"].run_id / ".." / "batch_manifest.json"
        seen.append(path.resolve())
        from rsimem.adamem_runtime import AdaMemPolicyReceipt
        return type("Receipt", (), {"payload": lambda self: {"outcome": "static"}})()
    monkeypatch.setattr(launcher, "run_trajectory", fake_run)
    report = launcher.run_replicate_batch(
        source_sequence=sequence, family_id="SM01", condition=AdaMemCondition.MEM0_STATIC,
        output_root=tmp_path / "out", past_bin=tmp_path / "past", past_root=tmp_path,
        config=config, registry=registry, base_url="https://example.test/v1",
        api_key=None, batch_id="batch", cutover_label="learn-a", dry_run=True,
    )
    assert report["accepted"] is True
    assert (tmp_path / "out" / "batch" / "batch_manifest.json").is_file()
    assert len(seen) == 3
    manifest = json.loads((tmp_path / "out" / "batch" / "batch_manifest.json").read_text())
    assert manifest["condition_order"] == [AdaMemCondition.MEM0_STATIC.value]
    assert manifest["started_at"]
    assert report["provider_health"]["status"] == "healthy"
    assert report["finished_at"]
