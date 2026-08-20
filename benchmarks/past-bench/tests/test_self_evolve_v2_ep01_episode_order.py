"""EP01 manifest order should keep the no-persistence control last."""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
FAMILY_YAML = (
    REPO_ROOT
    / "self-evolve-tasks-v2"
    / "memory_ability"
    / "EP01_prior_case_recall"
    / "family.yaml"
)
MANIFEST = (
    REPO_ROOT
    / "configs"
    / "self_evolve_v2"
    / "hermes_self_evolve_v2_ep01_prior_case_recall_only.yaml"
)


def test_ep01_family_declares_no_persistence_last() -> None:
    family = yaml.safe_load(FAMILY_YAML.read_text())
    assert family["episode_order"][-1] == "EP01_I06_control_no_persistence"


def test_ep01_manifest_keeps_no_persistence_last() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text())
    labels = [episode["label"] for episode in manifest["episodes"]]
    assert labels[-1] == "ep01_prior_case_recall_ep01_i06_control_no_persistence"


def test_ep01_family_declares_post_learn_anchor() -> None:
    family = yaml.safe_load(FAMILY_YAML.read_text())
    assert family["history_plan"]["anchors"] == [
        {"name": "ep01_post_learn", "save_after": "EP01_I03_recall_noise"}
    ]


def test_ep01_manifest_branches_from_post_learn_anchor() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text())
    episodes = {episode["label"]: episode for episode in manifest["episodes"]}

    assert episodes["ep01_prior_case_recall_ep01_i03_recall_noise"]["history_mode"] == "continue"
    assert (
        episodes["ep01_prior_case_recall_ep01_i03_recall_noise"]["history_save_anchor"]
        == "ep01_post_learn"
    )

    for label in (
        "ep01_prior_case_recall_ep01_i04_eval_near",
        "ep01_prior_case_recall_ep01_i05_eval_far",
        "ep01_prior_case_recall_ep01_i07_control_shortcut",
        "ep01_prior_case_recall_ep01_i08_control_wrong_mechanism",
    ):
        assert episodes[label]["history_mode"] == "from_anchor"
        assert episodes[label]["history_load_anchor"] == "ep01_post_learn"

    assert episodes["ep01_prior_case_recall_ep01_i06_control_no_persistence"]["history_mode"] == "fresh"
    assert not episodes["ep01_prior_case_recall_ep01_i06_control_no_persistence"]["history_load_anchor"]
