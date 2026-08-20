"""EP02 should evaluate subset recall, not single-item recall."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
EP02_ROOT = REPO_ROOT / "self-evolve-tasks-v2" / "memory_ability" / "EP02_exception_list_recall"
FIXTURE_ROOT = REPO_ROOT / "self-evolve-tasks-v2" / "_shared" / "fixtures" / "f04_config"


def _load_json(path: Path):
    return json.loads(path.read_text())


def test_ep02_seed_builds_multi_candidate_exception_set() -> None:
    expectations = _load_json(EP02_ROOT / "EP02_I02_seed_exception" / "expectations.json")
    assert len(expectations["required_update_ids"]) == 3
    assert expectations["forbidden_update_ids"] == ["INTG-EU-013", "INTG-US-014", "INTG-AP-015"]


def test_ep02_eval_requires_subset_application() -> None:
    expectations = _load_json(EP02_ROOT / "EP02_I04_eval_near" / "expectations.json")
    assert expectations["required_update_ids"] == ["INTG-EU-002", "INTG-US-004", "INTG-AP-006"]
    assert expectations["forbidden_update_ids"] == ["INTG-EU-013", "INTG-US-014", "INTG-AP-015"]


def test_ep02_eval_fixture_contains_multiple_candidates() -> None:
    fixture = _load_json(FIXTURE_ROOT / "eval_near.json")
    integration_ids = {item["integration_id"] for item in fixture}
    assert len(integration_ids) == 6
    assert {
        "INTG-EU-002",
        "INTG-US-004",
        "INTG-AP-006",
        "INTG-EU-013",
        "INTG-US-014",
        "INTG-AP-015",
    } == integration_ids


def test_ep02_eval_far_fixture_increases_surface_distance_with_extra_distractors() -> None:
    near_fixture = _load_json(FIXTURE_ROOT / "eval_near.json")
    far_fixture = _load_json(FIXTURE_ROOT / "eval_far.json")
    near_ids = {item["integration_id"] for item in near_fixture}
    far_ids = {item["integration_id"] for item in far_fixture}

    assert len(far_ids) > len(near_ids)
    assert {"INTG-EU-018", "INTG-US-019"}.issubset(far_ids)


def test_ep02_config_recall_expectations_do_not_require_explicit_get_inspection() -> None:
    expectation_paths = sorted(EP02_ROOT.glob("EP02_I*/expectations.json"))
    assert expectation_paths
    for path in expectation_paths:
        expectations = _load_json(path)
        if expectations.get("mode") == "config_recall" and "inspected_ids" in expectations:
            assert expectations.get("inspection_optional") is True, path.name
