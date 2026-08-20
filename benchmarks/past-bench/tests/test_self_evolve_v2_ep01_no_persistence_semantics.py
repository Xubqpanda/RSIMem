"""EP01 no-persistence control should keep eval scoring semantics while disabling persistence gating."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_YAML = (
    REPO_ROOT
    / "self-evolve-tasks-v2"
    / "memory_ability"
    / "EP01_prior_case_recall"
    / "EP01_I06_control_no_persistence"
    / "task.yaml"
)
EXPECTATIONS_JSON = (
    REPO_ROOT
    / "self-evolve-tasks-v2"
    / "memory_ability"
    / "EP01_prior_case_recall"
    / "EP01_I06_control_no_persistence"
    / "expectations.json"
)


def test_ep01_no_persistence_task_rubric_keeps_eval_success_standard() -> None:
    task = yaml.safe_load(TASK_YAML.read_text())
    rubric = task["judge_rubric"]

    assert "same success standard as the evaluation tasks" in rubric
    assert "score substantially lower" in rubric
    assert "Do not award high scores for merely stating that no record is available" in rubric


def test_ep01_no_persistence_expectations_keep_content_targets_but_drop_retrieval_gating() -> None:
    expectations = json.loads(EXPECTATIONS_JSON.read_text())

    assert expectations["required_prior_note_id"] == "MTG-EP01-SEED"
    assert expectations["required_output_keywords"] == ["Vendor X", "staging", "MTG-EP01-SEED"]
    assert expectations["min_session_search_calls"] == 0
    assert expectations["session_search_weight"] == 0.0
    assert expectations["apply_mechanism_caps"] is False
    assert expectations["retrieval_contract"]["evaluation_only"] is False
