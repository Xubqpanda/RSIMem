"""EP03 manifest order and single-home update wiring."""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
FAMILY_YAML = (
    REPO_ROOT
    / "self-evolve-tasks-v2"
    / "update_ability"
    / "EP03_recall_then_modify"
    / "family.yaml"
)
MANIFEST = (
    REPO_ROOT
    / "configs"
    / "self_evolve_v2"
    / "hermes_self_evolve_v2_ep03_recall_then_modify_only.yaml"
)


def test_ep03_family_declares_single_update_chain() -> None:
    family = yaml.safe_load(FAMILY_YAML.read_text())
    assert family["episode_order"] == [
        "EP03_I01_modify_cold",
        "EP03_I02_modify_seed",
        "EP03_I04_eval_near",
        "EP03_I05_eval_far",
        "EP03_I06_control_no_persistence",
        "EP03_I07_control_shortcut",
        "EP03_I08_control_wrong_mechanism",
    ]
    assert "EP03_I03_context_noise" not in family["episode_order"]
    assert family["total_episodes"] == 7


def test_ep03_family_uses_history_plan_and_cold_only_fixture() -> None:
    family = yaml.safe_load(FAMILY_YAML.read_text())
    history_plan = family["history_plan"]
    assert history_plan["anchors"] == [{"name": "ep03_post_update", "save_after": "EP03_I02_modify_seed"}]
    assert history_plan["branches"] == [
        {"mode": "fresh", "episodes": ["EP03_I01_modify_cold"]},
        {"mode": "continue", "episodes": ["EP03_I02_modify_seed"]},
        {
            "mode": "from_anchor",
            "anchor": "ep03_post_update",
            "episodes": [
                "EP03_I04_eval_near",
                "EP03_I05_eval_far",
                "EP03_I06_control_no_persistence",
                "EP03_I07_control_shortcut",
                "EP03_I08_control_wrong_mechanism",
            ],
        },
    ]
    assert family["episode_overrides"]["EP03_I01_modify_cold"]["initial_home_fixture_dir"].endswith(
        "self-evolve-tasks-v2/_shared/home_fixtures/update_ability/EP03_recall_then_modify"
    )
    assert family["episode_overrides"]["EP03_I02_modify_seed"]["stage"] == "update_or_stabilize"
    assert "initial_home_fixture_dir" not in family["episode_overrides"]["EP03_I02_modify_seed"]


def test_ep03_manifest_follows_single_home_chain() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text())
    episodes = {episode["label"]: episode for episode in manifest["episodes"]}

    cold = episodes["ep03_recall_then_modify_ep03_i01_modify_cold"]
    update = episodes["ep03_recall_then_modify_ep03_i02_modify_seed"]
    eval_near = episodes["ep03_recall_then_modify_ep03_i04_eval_near"]
    eval_far = episodes["ep03_recall_then_modify_ep03_i05_eval_far"]
    no_persistence = episodes["ep03_recall_then_modify_ep03_i06_control_no_persistence"]
    shortcut = episodes["ep03_recall_then_modify_ep03_i07_control_shortcut"]
    wrong_mechanism = episodes["ep03_recall_then_modify_ep03_i08_control_wrong_mechanism"]

    assert cold["history_mode"] == "fresh"
    assert cold["initial_home_fixture_dir"].endswith(
        "self-evolve-tasks-v2/_shared/home_fixtures/update_ability/EP03_recall_then_modify"
    )
    assert cold["shared_cold_run"] is False

    assert update["history_mode"] == "continue"
    assert update["history_save_anchor"] == "ep03_post_update"
    assert update["initial_home_fixture_dir"] == ""

    for episode in (eval_near, eval_far, no_persistence, shortcut, wrong_mechanism):
        assert episode["history_mode"] == "from_anchor"
        assert episode["history_load_anchor"] == "ep03_post_update"
        assert episode["initial_home_fixture_dir"] == ""

    assert no_persistence["persistence_allowed"] is False
    assert wrong_mechanism["mechanism"] == "mixed"
    assert wrong_mechanism["expected_persistence_signal"] == "session_search"
    assert wrong_mechanism["preseed_artifacts_dir"].endswith(
        "self-evolve-tasks-v2/_shared/preseed/ep03_wrong_mechanism_memory"
    )
