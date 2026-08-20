"""EP03 should not leak the seeded artifact through cold/eval fixtures or eval prompts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
EP03_ROOT = REPO_ROOT / "self-evolve-tasks-v2" / "update_ability" / "EP03_recall_then_modify"
FIXTURE_ROOT = REPO_ROOT / "self-evolve-tasks-v2" / "_shared" / "fixtures" / "ep03_notes"
HOME_FIXTURE_ROOT = (
    REPO_ROOT
    / "self-evolve-tasks-v2"
    / "_shared"
    / "home_fixtures"
    / "update_ability"
    / "EP03_recall_then_modify"
)
SEEDED_SNIPPETS = ("ONBOARDING PLAYBOOK v1", "mentor assigned within 48h")
SEEDED_IDS = ("DOC-431",)
SEED_METADATA = {
    "title": "Onboarding Playbook",
    "participants": ["alice@corp.com", "bob@corp.com", "carol@corp.com"],
    "tags": ["approved", "artifact-shell"],
}


def _prompt_text(task_path: Path) -> str:
    task = yaml.safe_load(task_path.read_text())
    prompt = task.get("prompt") or {}
    if isinstance(prompt, dict):
        return str(prompt.get("text") or "")
    return str(prompt)


def _load_fixture(name: str) -> list[dict]:
    return json.loads((FIXTURE_ROOT / name).read_text())


def test_ep03_cold_fixture_has_no_seeded_artifact() -> None:
    fixture = _load_fixture("cold.json")
    note_ids = {item["note_id"] for item in fixture}
    assert "DOC-431" not in note_ids
    fixture_text = json.dumps(fixture)
    for snippet in SEEDED_SNIPPETS:
        assert snippet not in fixture_text


def test_ep03_eval_fixtures_do_not_expose_seeded_content() -> None:
    for name in ("eval_near.json", "eval_far.json", "control_no_persistence.json", "control_shortcut.json", "control_wrong_mechanism.json"):
        fixture = _load_fixture(name)
        fixture_text = json.dumps(fixture)
        note_ids = {item["note_id"] for item in fixture}
        assert "DOC-431" not in note_ids
        for snippet in SEEDED_SNIPPETS:
            assert snippet not in fixture_text
        shell_count = sum(1 for item in fixture if "artifact-shell" in item.get("tags", []))
        assert shell_count >= 3, name


def test_ep03_eval_fixtures_have_exactly_one_metadata_matching_shell() -> None:
    for name in ("eval_near.json", "eval_far.json", "control_no_persistence.json", "control_shortcut.json", "control_wrong_mechanism.json"):
        fixture = _load_fixture(name)
        matching_shell_ids = [
            item["note_id"]
            for item in fixture
            if item.get("title") == SEED_METADATA["title"]
            and item.get("participants") == SEED_METADATA["participants"]
            and item.get("tags") == SEED_METADATA["tags"]
        ]
        assert len(matching_shell_ids) == 1, name


def test_ep03_eval_prompts_do_not_repeat_trigger_wording() -> None:
    for task_name in ("EP03_I04_eval_near", "EP03_I05_eval_far", "EP03_I06_control_no_persistence", "EP03_I07_control_shortcut", "EP03_I08_control_wrong_mechanism"):
        prompt_text = _prompt_text(EP03_ROOT / task_name / "task.yaml")
        for snippet in SEEDED_SNIPPETS:
            assert snippet not in prompt_text
        for identifier in SEEDED_IDS:
            assert identifier not in prompt_text
        assert "recover the prior approved" not in prompt_text.lower()
        assert "source of truth" not in prompt_text.lower()
        assert "do not trust" not in prompt_text.lower()


def test_ep03_home_fixture_seeds_searchable_session_db() -> None:
    db_path = HOME_FIXTURE_ROOT / "state.db"
    assert db_path.is_file()

    with sqlite3.connect(db_path) as conn:
        sessions = conn.execute(
            "select id, source from sessions order by id"
        ).fetchall()
        assert ("ep03_doc431_approved_playbook", "cli") in sessions
        assert all(source != "tool" for _, source in sessions)

        hits = conn.execute(
            "select content from messages_fts where messages_fts match ?",
            ('"DOC-431"',),
        ).fetchall()

    hit_text = "\n".join(row[0] for row in hits)
    assert "mentor assigned within 48h" in hit_text
    assert "DOC-437" not in hit_text
