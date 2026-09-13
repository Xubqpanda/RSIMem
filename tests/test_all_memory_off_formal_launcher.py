from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.base_memory.all_memory_off_formal_launcher import (
    CONDITION,
    SCHEMA,
    build_formal_manifest,
    dry_run_manifest,
    resume_formal_suite,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_manifest(tmp_path: Path) -> Path:
    config = tmp_path / "family.yaml"
    config.write_text("name: fixture\nepisodes: []\n", encoding="utf-8")
    source = {
        "schema": "rsimem-adamem-full-suite-formal-manifest-v1",
        "family_count": 26,
        "replicate_count": 3,
        "base_model": "gpt-5.6-luna",
        "formal_manifest_digest": "source-digest",
        "families": [
            {"family_id": f"F{index:02d}", "config": str(config), "config_digest": _digest(config), "category": "cross-workflow"}
            for index in range(26)
        ],
    }
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return path


def test_freeze_creates_only_the_new_all_memory_off_condition(tmp_path: Path) -> None:
    manifest = build_formal_manifest(source_formal_manifest=_source_manifest(tmp_path), output_root=tmp_path / "out")
    assert manifest["schema"] == SCHEMA
    assert manifest["condition"] == CONDITION.value
    assert manifest["family_count"] == 26
    assert manifest["replicate_count"] == 3
    assert manifest["backend_descriptor"]["memory_surface_policy"] == "all_memory_off"
    assert len(manifest["formal_manifest_digest"]) == 64


def test_dry_run_materializes_26_all_memory_off_receipts_without_provider(tmp_path: Path) -> None:
    frozen = build_formal_manifest(source_formal_manifest=_source_manifest(tmp_path), output_root=tmp_path / "formal")
    formal_path = tmp_path / "formal.json"
    formal_path.write_text(json.dumps(frozen), encoding="utf-8")
    config = tmp_path / "past.yaml"; config.write_text("runtime: local\n", encoding="utf-8")
    registry = tmp_path / "registry.yaml"; registry.write_text("models: {}\n", encoding="utf-8")
    report = dry_run_manifest(formal_manifest=formal_path, output_root=tmp_path / "dry", past_config=config, registry=registry)
    assert report["accepted"] is True
    assert report["family_count"] == 26
    first = Path(report["receipts"][0]["run_manifest"])
    run_manifest = json.loads(first.read_text())
    assert run_manifest["condition"] == CONDITION.value
    assert run_manifest["backend"]["episodic_memory_enabled"] is False
    rendered = first.parent / "sequence.yaml"
    assert "all_memory_off: true" in rendered.read_text(encoding="utf-8")


def test_resume_reuses_only_reauditable_batches(
    tmp_path: Path, monkeypatch
) -> None:
    frozen = build_formal_manifest(source_formal_manifest=_source_manifest(tmp_path), output_root=tmp_path / "formal")
    formal_path = tmp_path / "formal.json"
    formal_path.write_text(json.dumps(frozen), encoding="utf-8")
    batches = tmp_path / "batches"
    accepted = batches / "accepted"
    accepted.mkdir(parents=True)
    (accepted / "batch_manifest.json").write_text(json.dumps({
        "formal_manifest_digest": frozen["formal_manifest_digest"],
        "family_id": "F00",
    }), encoding="utf-8")
    incomplete = batches / "incomplete"
    incomplete.mkdir()
    (incomplete / "batch_manifest.json").write_text(json.dumps({
        "formal_manifest_digest": frozen["formal_manifest_digest"],
        "family_id": "F01",
    }), encoding="utf-8")

    import experiments.base_memory.all_memory_off_formal_launcher as launcher

    calls: list[str] = []

    def fake_audit(root: Path) -> dict[str, object]:
        if root == accepted:
            return {"accepted": True}
        if (root / "batch_result.json").is_file():
            return {"accepted": True}
        raise ValueError("incomplete")

    def fake_run(**kwargs):
        calls.append(kwargs["family_id"])
        root = kwargs["output_root"] / kwargs["batch_id"]
        root.mkdir(parents=True)
        (root / "batch_result.json").write_text("{}", encoding="utf-8")
        return {"accepted": True}

    monkeypatch.setattr(launcher, "audit_replicate_batch", fake_audit)
    monkeypatch.setattr(launcher, "run_replicate_batch", fake_run)
    progress = resume_formal_suite(
        formal_manifest=formal_path,
        output_root=batches,
        past_bin=tmp_path / "past", past_root=tmp_path, registry=tmp_path / "registry",
        past_config=tmp_path / "config", base_url="https://provider.example/v1",
        progress_output=tmp_path / "progress.json", family_ids=("F00", "F01"),
    )
    assert calls == ["F01"]
    assert progress["completed_families"] == ["F00", "F01"]
