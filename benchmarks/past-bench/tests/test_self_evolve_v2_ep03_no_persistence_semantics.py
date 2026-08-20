"""EP03 no-persistence control should keep eval scoring semantics while disabling persistence gating."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_YAML = (
    REPO_ROOT
    / "self-evolve-tasks-v2"
    / "update_ability"
    / "EP03_recall_then_modify"
    / "EP03_I06_control_no_persistence"
    / "task.yaml"
)
EXPECTATIONS_JSON = (
    REPO_ROOT
    / "self-evolve-tasks-v2"
    / "update_ability"
    / "EP03_recall_then_modify"
    / "EP03_I06_control_no_persistence"
    / "expectations.json"
)
FIXTURE_JSON = (
    REPO_ROOT
    / "self-evolve-tasks-v2"
    / "_shared"
    / "fixtures"
    / "ep03_notes"
    / "control_no_persistence.json"
)


def test_ep03_no_persistence_task_rubric_keeps_eval_success_standard() -> None:
    task = yaml.safe_load(TASK_YAML.read_text())
    rubric = task["judge_rubric"]

    assert "same success standard as the evaluation tasks" in rubric
    assert "mentor handoff line to 24 hours" in rubric
    assert "preserving the title, participants, and tags" in rubric
    assert "score substantially lower" in rubric


def test_ep03_no_persistence_expectations_keep_content_targets_but_cap_ungrounded_success() -> None:
    expectations = json.loads(EXPECTATIONS_JSON.read_text())

    assert expectations["required_prior_note_id"] == "DOC-431"
    assert expectations["required_updated_note_id"] == "DOC-741"
    assert expectations["target_modify_field"] == "content"
    assert expectations["preserved_fields"] == ["title", "participants", "tags"]
    assert expectations["required_shell_metadata"] == {
        "title": "Onboarding Playbook",
        "participants": ["alice@corp.com", "bob@corp.com", "carol@corp.com"],
        "tags": ["approved", "artifact-shell"],
    }
    assert expectations["shell_match_fields"] == ["title", "participants", "tags"]
    assert expectations["require_updated_note_differs_from_prior"] is True
    assert expectations["retrieval_snippet_source"] == "session_search_response_only"
    assert expectations["min_session_search_calls"] == 0
    assert expectations["session_search_weight"] == 0.0
    assert expectations["retrieval_contract"]["evaluation_only"] is False
    assert expectations["required_content_replacements"] == [
        {"old": "mentor assigned within 48h", "new": "mentor assigned within 24h"}
    ]
    assert expectations["require_retrieved_prior_snippets"] == [
        "DOC-431",
        "mentor assigned within 48h",
    ]
    assert expectations["max_without_grounding"] == 0.2


def test_ep03_no_persistence_fixture_uses_multiple_shell_candidates() -> None:
    fixture = json.loads(FIXTURE_JSON.read_text())
    shell_ids = [item["note_id"] for item in fixture if "artifact-shell" in item.get("tags", [])]

    assert shell_ids == ["DOC-741", "DOC-742", "DOC-743"]
