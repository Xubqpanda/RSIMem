"""EP02 no-persistence control should keep eval targets while capping success without persistence."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_YAML = (
    REPO_ROOT
    / "self-evolve-tasks-v2"
    / "memory_ability"
    / "EP02_exception_list_recall"
    / "EP02_I06_control_no_persistence"
    / "task.yaml"
)
EXPECTATIONS_JSON = (
    REPO_ROOT
    / "self-evolve-tasks-v2"
    / "memory_ability"
    / "EP02_exception_list_recall"
    / "EP02_I06_control_no_persistence"
    / "expectations.json"
)


def test_ep02_no_persistence_task_rubric_keeps_eval_target_but_not_abstention_as_success() -> None:
    task = yaml.safe_load(TASK_YAML.read_text())
    rubric = task["judge_rubric"]

    assert "do not guess" in task["prompt"]["text"]
    assert "same latent target as the eval episodes" in rubric
    assert "INTG-EU-002, INTG-US-004, INTG-AP-006" in rubric
    assert "abstention is not task success" in rubric
    assert "lucky guesses" in rubric


def test_ep02_no_persistence_expectations_keep_eval_targets_and_cap_missing_retrieval() -> None:
    expectations = json.loads(EXPECTATIONS_JSON.read_text())

    assert expectations["required_update_ids"] == [
        "INTG-EU-002",
        "INTG-US-004",
        "INTG-AP-006",
    ]
    assert expectations["forbidden_update_ids"] == [
        "INTG-EU-013",
        "INTG-US-014",
        "INTG-AP-015",
    ]
    assert expectations["update_requirements"] == {
        "INTG-EU-002": {"status": "active", "exception_approved": True},
        "INTG-US-004": {"status": "active", "exception_approved": True},
        "INTG-AP-006": {"status": "active", "exception_approved": True},
    }
    assert expectations["min_session_search_calls"] == 1
    assert expectations["session_search_weight"] == 0.1
    assert expectations["apply_mechanism_caps"] is True
    assert expectations["missing_retrieval_cap"] == 0.1
    assert expectations["retrieval_contract"]["evaluation_only"] is True
