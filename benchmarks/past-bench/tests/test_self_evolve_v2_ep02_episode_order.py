"""EP02 manifest order and history wiring should mirror EP01."""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
FAMILY_YAML = (
    REPO_ROOT
    / "self-evolve-tasks-v2"
    / "memory_ability"
    / "EP02_exception_list_recall"
    / "family.yaml"
)
MANIFEST = (
    REPO_ROOT
    / "configs"
    / "self_evolve_v2"
    / "hermes_self_evolve_v2_ep02_exception_list_recall_only.yaml"
)


def test_ep02_family_declares_no_persistence_last() -> None:
    family = yaml.safe_load(FAMILY_YAML.read_text())
    assert family["episode_order"][-1] == "EP02_I06_control_no_persistence"


def test_ep02_manifest_keeps_no_persistence_last() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text())
    labels = [episode["label"] for episode in manifest["episodes"]]
    assert labels[-1] == "ep02_exception_list_recall_ep02_i06_control_no_persistence"


def test_ep02_family_declares_post_learn_anchor() -> None:
    family = yaml.safe_load(FAMILY_YAML.read_text())
    assert family["history_plan"]["anchors"] == [
        {"name": "ep02_post_learn", "save_after": "EP02_I03_context_noise"}
    ]


def test_ep02_manifest_branches_from_post_learn_anchor() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text())
    episodes = {episode["label"]: episode for episode in manifest["episodes"]}

    assert episodes["ep02_exception_list_recall_ep02_i03_context_noise"]["history_mode"] == "continue"
    assert (
        episodes["ep02_exception_list_recall_ep02_i03_context_noise"]["history_save_anchor"]
        == "ep02_post_learn"
    )

    for label in (
        "ep02_exception_list_recall_ep02_i04_eval_near",
        "ep02_exception_list_recall_ep02_i05_eval_far",
        "ep02_exception_list_recall_ep02_i07_control_shortcut",
        "ep02_exception_list_recall_ep02_i08_control_wrong_mechanism",
    ):
        assert episodes[label]["history_mode"] == "from_anchor"
        assert episodes[label]["history_load_anchor"] == "ep02_post_learn"

    assert episodes["ep02_exception_list_recall_ep02_i06_control_no_persistence"]["history_mode"] == "fresh"
    assert not episodes["ep02_exception_list_recall_ep02_i06_control_no_persistence"]["history_load_anchor"]


def test_ep02_manifest_overrides_wrong_mechanism_episode() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text())
    episodes = {episode["label"]: episode for episode in manifest["episodes"]}
    wrong_mechanism = episodes["ep02_exception_list_recall_ep02_i08_control_wrong_mechanism"]

    assert wrong_mechanism["mechanism"] == "mixed"
    assert wrong_mechanism["expected_persistence_signal"] == "session_search"
    assert wrong_mechanism["persistence_allowed"] is True
    assert wrong_mechanism["preseed_artifacts_dir"].endswith(
        "self-evolve-tasks-v2/_shared/preseed/ep02_wrong_mechanism_memory"
    )
