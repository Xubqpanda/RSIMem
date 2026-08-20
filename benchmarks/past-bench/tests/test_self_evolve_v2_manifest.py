from pathlib import Path

import pytest
import yaml

from past_bench.models.self_evolve import SelfEvolveSequenceDefinition
from past_bench.self_evolve_v2 import generate_manifest


def _write_task(task_dir: Path, task_id: str) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    task_dir.joinpath("task.yaml").write_text(
        yaml.safe_dump(
            {
                "task_id": task_id,
                "task_name": task_id,
                "prompt": {"text": "hi", "language": "zh"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_family_yaml(family_dir: Path) -> None:
    family_dir.mkdir(parents=True, exist_ok=True)
    family_dir.joinpath("family.yaml").write_text(
        yaml.safe_dump(
            {
                "family_id": "SM01_preference_adoption",
                "title": "Preference Adoption",
                "ability_dir": "memory_ability",
                "primary_ability": "memory_ability",
                "secondary_abilities": [],
                "primary_trigger": "explicit_instruction",
                "trigger_class": "explicit_instruction",
                "secondary_triggers": [],
                "optional_supporting_trigger": None,
                "supporting_stage": "learn_b",
                "supporting_signal": "surface-variation reinforcement",
                "supporting_signals": [],
                "distractor_signals": [],
                "trigger_injection_plan": {
                    "learn_a": "introduce the durable preference explicitly",
                    "learn_b": "reinforce the same preference under surface variation",
                },
                "primary_trigger_stages": ["learn_a"],
                "trigger_removal_plan": {
                    "eval_near": "remove the preference wording entirely",
                    "eval_far": "remove the preference wording entirely across domain shift",
                },
                "trigger_mixing_rule": "Do not add explicit_correction here; use a sibling family instead.",
                "sibling_comparison_plan": "Compare with an SM01 correction-based sibling if trigger sensitivity is needed.",
                "expected_substrate": "memory",
                "autonomy_required": True,
                "evaluation_requires_reuse": True,
                "transfer_distance": "near_far",
                "overwrite_mode": "not_required",
                "noise_profile": "medium",
                "family_length_tier": "tier1",
                "implementation_priority": "phase1",
                "instances_per_bucket": {
                    "cold": 1,
                    "learn": 2,
                    "eval_near": 1,
                    "eval_far": 1,
                    "control": 1,
                },
                "total_episodes": 6,
                "control_set": ["no_persistence", "shortcut", "wrong_mechanism"],
                "status": "skeleton",
                "what_is_learned": "A durable preference.",
                "what_persists": "The stored preference.",
                "what_counts_as_reuse": "Later tasks apply the preference.",
                "false_positive_pattern": "The eval prompt repeats the preference.",
                "must_demonstrate": [
                    "the preference is stored",
                    "the preference is reused later",
                ],
                "recommended_mock_services": ["notes"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_generate_manifest_from_v2_family(tmp_path: Path) -> None:
    repo_root = tmp_path
    framework_root = repo_root / "self-evolve-tasks-v2"
    family_dir = framework_root / "memory_ability" / "SM01_preference_adoption"
    config_dir = repo_root / "configs" / "self_evolve_v2"
    out_path = config_dir / "demo.yaml"

    _write_family_yaml(family_dir)
    _write_task(family_dir / "cold_baseline", "cold")
    _write_task(family_dir / "learn_a_seed", "learn-a")
    _write_task(family_dir / "learn_b_reinforce", "learn-b")
    _write_task(family_dir / "eval_near_transfer", "eval-near")
    _write_task(family_dir / "eval_far_shift", "eval-far")
    _write_task(family_dir / "control_no_persistence", "control")

    generated = generate_manifest(
        "memory_ability/SM01_preference_adoption",
        out_path=out_path,
        repo_root=repo_root,
        framework_root=framework_root,
    )

    assert generated == out_path
    document = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert [episode["stage"] for episode in document["episodes"]] == [
        "cold",
        "learn_a",
        "learn_b",
        "eval_near",
        "eval_far",
        "control",
    ]
    assert document["episodes"][0]["bucket"] == "baseline"
    assert document["episodes"][0]["shared_cold_run"] is True
    assert document["episodes"][3]["requires_fresh_session"] is True
    assert document["episodes"][0]["task"].startswith("../../self-evolve-tasks-v2/")
    assert document["episodes"][0]["trigger_phrases"] == []
    assert document["episodes"][0]["history_mode"] == "continue"
    assert document["episodes"][1]["history_mode"] == "continue"
    assert document["episodes"][3]["history_mode"] == "fresh"

    sequence = SelfEvolveSequenceDefinition.from_yaml(out_path)
    assert sequence.episodes[0].family_id == "SM01_preference_adoption"
    assert sequence.episodes[-1].bucket == "control"
    assert sequence.episodes[1].history_mode == "continue"


def test_generate_manifest_supports_nested_config_dirs(tmp_path: Path) -> None:
    repo_root = tmp_path
    framework_root = repo_root / "self-evolve-tasks-v2"
    family_dir = framework_root / "memory_ability" / "SM01_preference_adoption"
    out_path = repo_root / "configs" / "self_evolve_v2" / "latest" / "demo.yaml"

    _write_family_yaml(family_dir)
    _write_task(family_dir / "cold_baseline", "cold")
    _write_task(family_dir / "learn_a_seed", "learn-a")
    _write_task(family_dir / "learn_b_reinforce", "learn-b")
    _write_task(family_dir / "eval_near_transfer", "eval-near")
    _write_task(family_dir / "eval_far_shift", "eval-far")
    _write_task(family_dir / "control_no_persistence", "control")

    generate_manifest(
        "memory_ability/SM01_preference_adoption",
        out_path=out_path,
        repo_root=repo_root,
        framework_root=framework_root,
    )

    document = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert document["episodes"][0]["task"].startswith("../../../self-evolve-tasks-v2/")

    sequence = SelfEvolveSequenceDefinition.from_yaml(out_path)
    assert sequence.resolve_task_yaml(sequence.episodes[0].task) == family_dir / "cold_baseline" / "task.yaml"


def test_generate_manifest_disables_shared_cold_for_seeded_home_inputs(tmp_path: Path) -> None:
    repo_root = tmp_path
    framework_root = repo_root / "self-evolve-tasks-v2"
    family_dir = framework_root / "memory_ability" / "SM01_preference_adoption"
    out_path = repo_root / "configs" / "self_evolve_v2" / "demo.yaml"

    _write_family_yaml(family_dir)
    family_yaml_path = family_dir / "family.yaml"
    family_doc = yaml.safe_load(family_yaml_path.read_text(encoding="utf-8"))
    family_doc["episode_overrides"] = {
        "cold_baseline": {
            "initial_home_fixture_dir": "self-evolve-tasks-v2/_shared/home_fixtures/update_ability/demo_family",
        }
    }
    family_yaml_path.write_text(
        yaml.safe_dump(family_doc, sort_keys=False),
        encoding="utf-8",
    )
    _write_task(family_dir / "cold_baseline", "cold")
    _write_task(family_dir / "learn_a_seed", "learn-a")
    _write_task(family_dir / "learn_b_reinforce", "learn-b")
    _write_task(family_dir / "eval_near_transfer", "eval-near")
    _write_task(family_dir / "eval_far_shift", "eval-far")
    _write_task(family_dir / "control_no_persistence", "control")

    generate_manifest(
        "memory_ability/SM01_preference_adoption",
        out_path=out_path,
        repo_root=repo_root,
        framework_root=framework_root,
    )

    document = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    cold = document["episodes"][0]
    assert cold["initial_home_fixture_dir"].endswith(
        "self-evolve-tasks-v2/_shared/home_fixtures/update_ability/demo_family"
    )
    assert cold["shared_cold_run"] is False


def test_generate_manifest_rejects_bucket_mismatch(tmp_path: Path) -> None:
    repo_root = tmp_path
    framework_root = repo_root / "self-evolve-tasks-v2"
    family_dir = framework_root / "memory_ability" / "SM01_preference_adoption"

    _write_family_yaml(family_dir)
    _write_task(family_dir / "cold_baseline", "cold")
    _write_task(family_dir / "learn_a_seed", "learn-a")
    _write_task(family_dir / "eval_near_transfer", "eval-near")
    _write_task(family_dir / "eval_far_shift", "eval-far")
    _write_task(family_dir / "control_no_persistence", "control")

    with pytest.raises(ValueError, match="task bucket counts"):
        generate_manifest(
            "memory_ability/SM01_preference_adoption",
            repo_root=repo_root,
            framework_root=framework_root,
        )
