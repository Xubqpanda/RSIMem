from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.legacy.sensitivity.sensitivity_coverage import aggregate_sensitivity_coverage


def _write_pilot(root: Path, *, pilot_id: str = "pilot.one", forbidden: bool = False, legacy_audit_name: bool = False) -> None:
    root.mkdir(parents=True)
    conditions = [
        "no_persistence", "native_static", "type_matched_oracle",
        "shortcut_current_input", "wrong_mechanism",
    ]
    runs = [f"run.{index}" for index in range(5)]
    (root / "sensitivity_manifest.json").write_text(json.dumps({
        "runs": [
            {"run_id": run, "family_id": "SM01_preference_adoption", "panel": "semantic",
             "replicate": 1, "condition": condition}
            for run, condition in zip(runs, conditions, strict=True)
        ]
    }) + "\n", encoding="utf-8")
    (root / "sensitivity_pilot_plan.json").write_text(json.dumps({
        "schema": "rsimem-sensitivity-pilot-v1", "schema_version": 1,
        "pilot_id": pilot_id, "family_id": "SM01_preference_adoption",
        "panel": "semantic", "replicate": 1,
        "condition_order": conditions, "run_ids": runs,
    }) + "\n", encoding="utf-8")
    rows = [{
        "run_id": run, "condition": condition, "trace_count": 1,
        "memory_event_count": 1 if condition in {"native_static", "type_matched_oracle"} else 0,
        "issues": [], "ok": True,
    } for run, condition in zip(runs, conditions, strict=True)]
    audit = {
        "schema": "rsimem-sensitivity-pilot-audit-v1", "pilot_id": pilot_id,
        "provider_probe_ok": True, "run_count": 5, "runs": rows,
        "issues": [], "ok": True,
    }
    if forbidden:
        audit["score"] = 1.0
    name = "sensitivity_pilot_audit.json" if legacy_audit_name else "audit.json"
    (root / name).write_text(json.dumps(audit) + "\n", encoding="utf-8")


def test_coverage_aggregates_accepted_panel_and_condition_counts(tmp_path: Path) -> None:
    _write_pilot(tmp_path / "pilot")
    report = aggregate_sensitivity_coverage(tmp_path)
    semantic = report["panels"]["semantic"]
    assert semantic["accepted_pilot_count"] == 1
    assert semantic["accepted_family_ids"] == ["SM01_preference_adoption"]
    assert semantic["all_families_covered"] is False
    assert semantic["ready_for_replicate_analysis"] is False
    assert "SM02_constraint_retention" in semantic["missing_family_ids"]
    assert semantic["condition_coverage"]["native_static"] == 1
    assert report["records"][0]["pilot_ok"] is True


def test_coverage_rejects_benchmark_score_fields(tmp_path: Path) -> None:
    _write_pilot(tmp_path / "pilot", forbidden=True)
    with pytest.raises(ValueError, match="forbidden field"):
        aggregate_sensitivity_coverage(tmp_path)


def test_coverage_rejects_duplicate_pilot_identity(tmp_path: Path) -> None:
    _write_pilot(tmp_path / "one")
    _write_pilot(tmp_path / "two")
    with pytest.raises(ValueError, match="duplicate pilot identity"):
        aggregate_sensitivity_coverage(tmp_path)


def test_coverage_accepts_legacy_audit_filename(tmp_path: Path) -> None:
    _write_pilot(tmp_path / "pilot", legacy_audit_name=True)
    report = aggregate_sensitivity_coverage(tmp_path)
    assert report["panels"]["semantic"]["accepted_pilot_count"] == 1


def test_coverage_rejects_plan_manifest_identity_drift(tmp_path: Path) -> None:
    _write_pilot(tmp_path / "pilot")
    manifest_path = tmp_path / "pilot" / "sensitivity_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runs"][0]["condition"] = "wrong_mechanism"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        aggregate_sensitivity_coverage(tmp_path)
