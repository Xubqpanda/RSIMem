from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.base_memory.base_memory_formal_launcher import SCHEMA, _accepted_batch_roots, build_formal_manifest


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_freeze_binds_exactly_26_verified_family_configs(tmp_path: Path) -> None:
    config = tmp_path / "family.yaml"
    config.write_text("episodes: []\n", encoding="utf-8")
    source = {
        "schema": "rsimem-adamem-full-suite-formal-manifest-v1",
        "family_count": 26,
        "replicate_count": 3,
        "base_model": "gpt-5.6-luna",
        "formal_manifest_digest": "source-digest",
        "families": [
            {"family_id": f"F{index:02d}", "config": str(config),
             "config_digest": _digest(config), "category": "cross-workflow"}
            for index in range(26)
        ],
    }
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    manifest = build_formal_manifest(source_formal_manifest=source_path, output_root=tmp_path / "out")

    assert manifest["schema"] == SCHEMA
    assert manifest["family_count"] == 26
    assert manifest["replicate_count"] == 3
    assert manifest["backend_order"] == ["NoMemory", "HermesNative", "Mem0Static"]
    assert len(manifest["formal_manifest_digest"]) == 64


def test_continuous_runner_refuses_to_duplicate_an_incomplete_batch(tmp_path: Path) -> None:
    batch = tmp_path / "batches" / "in-flight"
    batch.mkdir(parents=True)
    (batch / "batch_manifest.json").write_text(json.dumps({
        "formal_manifest_digest": "frozen", "family_id": "F01", "condition": "NoMemory",
    }), encoding="utf-8")

    with pytest.raises(RuntimeError, match="matching formal batch is incomplete"):
        _accepted_batch_roots(formal_manifest_digest="frozen", family_id="F01", output_root=tmp_path / "batches")
