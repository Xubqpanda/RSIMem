from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.adamem.adamem_validation_manifest import (
    VALIDATION_FAMILIES,
    build_validation_manifest,
)
from experiments.adamem.adamem_validation_preflight import audit_manifest


def _write_source(tmp_path: Path, family_ids: tuple[str, ...]) -> Path:
    config = tmp_path / "sequence.yaml"
    config.write_text("name: frozen\n", encoding="utf-8")
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    source = {
        "schema": "rsimem-adamem-full-suite-formal-manifest-v1",
        "formal_manifest_digest": "protocol-digest",
        "base_model": "gpt-5.6-luna",
        "replicate_count": 3,
        "families": [
            {
                "family_id": family_id,
                "config": str(config),
                "config_digest": digest,
                "cutover_label": "learn",
                "category": "test",
            }
            for family_id in family_ids
        ],
    }
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return path


def test_validation_family_constant_is_the_complete_frozen_set() -> None:
    assert len(VALIDATION_FAMILIES) == 26
    assert len(set(VALIDATION_FAMILIES)) == 26
    assert "SM01_preference_adoption" in VALIDATION_FAMILIES
    assert "PG06_kappa_integration_review" in VALIDATION_FAMILIES


def test_manifest_builder_rejects_a_different_26_family_set(tmp_path: Path) -> None:
    family_ids = tuple(VALIDATION_FAMILIES[:-1]) + ("UNREGISTERED_family",)
    source = _write_source(tmp_path, family_ids)
    with pytest.raises(ValueError, match="frozen 26-family set"):
        build_validation_manifest(
            source_formal_manifest=source,
            output_root=tmp_path / "out",
            code_revision={"git_head": "head"},
            base_url="https://example.test/v1",
        )


def test_preflight_rejects_a_hand_edited_26_family_manifest(tmp_path: Path) -> None:
    family_ids = tuple(VALIDATION_FAMILIES[:-1]) + ("UNREGISTERED_family",)
    config = tmp_path / "sequence.yaml"
    config.write_text("name: frozen\n", encoding="utf-8")
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    manifest = {
        "schema": "rsimem-adamem-mem0-refactor-validation-manifest-v3",
        "family_count": 26,
        "families": [
            {
                "family_id": family_id,
                "config": str(config),
                "config_digest": digest,
                "cutover_label": "learn",
            }
            for family_id in family_ids
        ],
        "conditions": [
            "B0_mem0_static",
            "B2_mem0_adamem_full_trajectory",
        ],
        "replicate_count": 3,
        "surface_policy": {
            "schema": "rsimem-runtime-surface-policy-v1",
            "schema_version": 1,
            "policy_id": "mem0-semantic-v1",
            "semantic_memory_enabled": True,
            "episodic_memory_enabled": True,
            "procedural_memory_enabled": True,
            "profile_enabled": True,
        },
        "manifest_digest": "digest",
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen 26-family set"):
        audit_manifest(path)
