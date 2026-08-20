"""Static checks for mechanism-level EP01 control wiring."""

from __future__ import annotations

from pathlib import Path

import yaml

from past_bench.models.self_evolve import SelfEvolveSequenceDefinition


REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = (
    REPO_ROOT
    / "configs"
    / "self_evolve_v2"
    / "hermes_self_evolve_v2_ep01_prior_case_recall_only.yaml"
)


def _episodes_by_label() -> dict[str, dict]:
    document = yaml.safe_load(MANIFEST.read_text())
    return {episode["label"]: episode for episode in document["episodes"]}


def test_ep01_no_persistence_control_disables_persistence() -> None:
    episodes = _episodes_by_label()

    assert (
        episodes["ep01_prior_case_recall_ep01_i06_control_no_persistence"]["persistence_allowed"]
        is False
    )
    assert (
        episodes["ep01_prior_case_recall_ep01_i07_control_shortcut"]["persistence_allowed"]
        is True
    )
    assert (
        episodes["ep01_prior_case_recall_ep01_i08_control_wrong_mechanism"]["persistence_allowed"]
        is True
    )


def test_ep01_manifest_keeps_repo_relative_task_paths() -> None:
    episodes = _episodes_by_label()
    for episode in episodes.values():
        assert str(episode["task"]).startswith("../../self-evolve-tasks-v2/")


def test_ep01_manifest_history_fields_parse() -> None:
    seq = SelfEvolveSequenceDefinition.from_yaml(MANIFEST)
    episodes = {episode.label: episode for episode in seq.episodes}

    assert episodes["ep01_prior_case_recall_ep01_i04_eval_near"].history_mode == "from_anchor"
    assert (
        episodes["ep01_prior_case_recall_ep01_i04_eval_near"].history_load_anchor
        == "ep01_post_learn"
    )
    assert (
        episodes["ep01_prior_case_recall_ep01_i03_recall_noise"].history_save_anchor
        == "ep01_post_learn"
    )
