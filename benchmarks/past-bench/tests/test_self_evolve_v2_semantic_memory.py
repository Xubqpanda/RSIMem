import json
from pathlib import Path

from past_bench.models.self_evolve import SelfEvolveSequenceDefinition
from past_bench.models.task import TaskDefinition
from past_bench.self_evolve_v2 import (
    FRAMEWORK_ROOT_NAME,
    generate_manifest,
    load_families,
    validate_family_task_layout,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
FRAMEWORK_ROOT = REPO_ROOT / FRAMEWORK_ROOT_NAME
SEMANTIC_MEMORY_IDS = {
    "SM01_preference_adoption",
    "SM02_constraint_retention",
    "SM03_fact_correction",
    "SM04_rule_migration",
}
DIRECT_TRIGGER_SNIPPETS = {
    "memory_ability/SM01_preference_adoption/learn_a_tsv_preference": "Save this durable team default",
    "memory_ability/SM02_constraint_retention/learn_a_external_boundary": "Save this durable boundary",
    "update_ability/SM03_fact_correction/learn_a_seed_stale_fact": "Save this current project fact",
    "update_ability/SM03_fact_correction/update_corrected_freeze_date": "Authoritative update for future Phoenix work",
    "update_ability/SM04_rule_migration/learn_a_seed_old_rule": "Save this current long-term review rule",
    "update_ability/SM04_rule_migration/update_new_review_rule": "Authoritative migration for future reviews",
}


def _semantic_memory_families():
    families = load_families(FRAMEWORK_ROOT)
    return [family for family in families if family.family_id in SEMANTIC_MEMORY_IDS]


def test_semantic_memory_families_are_authored_and_have_tasks() -> None:
    families = _semantic_memory_families()
    assert {family.family_id for family in families} == SEMANTIC_MEMORY_IDS

    for family in families:
        assert family.status == "authored"
        assert family.benchmark_role == "canonical_anchor"
        assert family.compatibility_gate == "strict"
        assert family.trigger_strength == "T3"
        assert "direct_user_message" in family.trigger_visibility_paths
        assert family.phase1_canonical_sibling
        assert family.future_sibling_roadmap
        assert family.cold_design
        assert family.expected_cold_result
        assert family.control_design
        assert family.expected_control_result
        assert family.source_path is not None
        family_dir = family.source_path.parent
        task_dirs = validate_family_task_layout(family_dir, family)
        assert family.instances_per_bucket["control"] == 3
        assert len(task_dirs) == family.total_episodes
        for task_dir in task_dirs:
            assert (task_dir / "task.yaml").is_file()
            assert (task_dir / "expectations.json").is_file()
            assert (task_dir / "grader.py").is_file()
            assert list((task_dir / "fixtures").glob("*.json"))


def test_semantic_memory_manifest_and_expectations_align(tmp_path: Path) -> None:
    for family in _semantic_memory_families():
        manifest_path = tmp_path / f"{family.family_id.lower()}.yaml"
        family_rel = f"{family.primary_ability}/{family.family_id}"
        generated = generate_manifest(
            family_rel,
            out_path=manifest_path,
            repo_root=REPO_ROOT,
            framework_root=FRAMEWORK_ROOT,
        )
        assert generated == manifest_path

        sequence = SelfEvolveSequenceDefinition.from_yaml(manifest_path)
        assert len(sequence.episodes) == family.total_episodes
        control_episodes = [episode for episode in sequence.episodes if episode.bucket == "control"]
        assert len(control_episodes) == 3

        for episode in sequence.episodes:
            task_path = sequence.resolve_task_yaml(episode.task)
            task = TaskDefinition.from_yaml(task_path)
            expectations = json.loads((task_path.parent / "expectations.json").read_text(encoding="utf-8"))

            assert task.category == "self_evolve"
            assert task.judge_rubric
            assert expectations["latent_rule_id"] == episode.latent_rule_id
            assert expectations["expected_mechanism"] == "memory"
            assert expectations["expected_mechanism"] == episode.expected_persistence_signal
            assert expectations["bucket_role"] == episode.bucket
            assert expectations["mode"] == "notes_memory"
            assert expectations["artifact_contract"]["type"] == "memory"

            retrieval_contract = expectations["retrieval_contract"]
            control_kind = expectations.get("control_kind", "")
            if episode.bucket == "baseline" or control_kind == "no_persistence":
                assert expectations["artifact_contract"]["require_rule_keywords"] == []
                if episode.bucket == "baseline":
                    assert expectations["forbidden_artifact_keywords"]
            else:
                assert expectations["artifact_contract"]["require_rule_keywords"]
            if episode.bucket == "learn":
                if episode.stage == "learn_b" and retrieval_contract.get("min_memory_injections", 0) > 0:
                    assert expectations.get("min_memory_injections", 0) > 0
                else:
                    assert expectations.get("min_memory_calls", 0) > 0 or retrieval_contract.get("min_memory_calls", 0) > 0
            if episode.bucket == "evaluation":
                assert expectations.get("min_memory_injections", 0) > 0 or retrieval_contract.get("min_memory_injections", 0) > 0
            if episode.bucket == "baseline" and family.primary_ability == "update_ability":
                assert episode.initial_home_fixture_dir.endswith(
                    f"self-evolve-tasks-v2/_shared/home_fixtures/update_ability/{family.family_id}"
                )
            if episode.bucket == "learn" and family.primary_ability == "update_ability":
                if episode.stage == "update":
                    assert episode.history_mode == "continue"
                    assert episode.initial_home_fixture_dir == ""
            if episode.bucket == "control":
                assert expectations["control_kind"] in {
                    "no_persistence",
                    "shortcut",
                    "wrong_mechanism",
                }
                if expectations["control_kind"] == "shortcut":
                    assert expectations.get("shortcut_exposes_target_rule") is True
                    assert expectations.get("min_memory_injections", 0) == 0
                    assert retrieval_contract.get("min_memory_injections", 0) == 0
                elif expectations["control_kind"] == "wrong_mechanism":
                    assert expectations.get("apply_mechanism_caps") is True


def test_semantic_memory_cold_tasks_do_not_leak_target_rule() -> None:
    for family in _semantic_memory_families():
        family_dir = family.source_path.parent
        cold_dirs = [path for path in validate_family_task_layout(family_dir, family) if path.name.startswith("cold")]
        assert len(cold_dirs) == 1
        cold_dir = cold_dirs[0]
        task = TaskDefinition.from_yaml(cold_dir / "task.yaml")
        expectations = json.loads((cold_dir / "expectations.json").read_text(encoding="utf-8"))
        notes = json.loads((cold_dir / "fixtures" / "notes.json").read_text(encoding="utf-8"))

        combined = "\n".join(
            [
                task.prompt.text,
                *(note.get("content", "") for note in notes),
            ]
        )
        for forbidden in expectations["forbidden_artifact_keywords"]:
            assert forbidden not in combined, f"{cold_dir} leaked target rule keyword {forbidden!r}"


def test_sm04_cold_expectations_pin_the_migrated_rule_slot() -> None:
    expectations = json.loads(
        (
            FRAMEWORK_ROOT
            / "update_ability"
            / "SM04_rule_migration"
            / "cold_rule_baseline"
            / "expectations.json"
        ).read_text(encoding="utf-8")
    )

    assert "slot_checks" in expectations
    assert "r1_migrated_rule_token" in expectations["slot_checks"]


def test_semantic_memory_trigger_delivery_is_direct_and_shortcut_controls_are_explicit() -> None:
    for rel_path, snippet in DIRECT_TRIGGER_SNIPPETS.items():
        task = TaskDefinition.from_yaml(
            FRAMEWORK_ROOT / rel_path / "task.yaml"
        )
        assert snippet in task.prompt.text

    shortcut_controls = []
    for family in _semantic_memory_families():
        shortcut_controls.extend(family.source_path.parent.glob("control_shortcut_*/task.yaml"))
    assert len(shortcut_controls) == 4
    for task_path in shortcut_controls:
        task = TaskDefinition.from_yaml(task_path)
        expectations = json.loads((task_path.parent / "expectations.json").read_text(encoding="utf-8"))
        assert "For this task only" in task.prompt.text
        assert expectations["control_kind"] == "shortcut"
        assert expectations["shortcut_exposes_target_rule"] is True

    sm04_eval_prompt = TaskDefinition.from_yaml(
        FRAMEWORK_ROOT
        / "update_ability"
        / "SM04_rule_migration"
        / "eval_near_apply_migrated_rule"
        / "task.yaml"
    ).prompt.text
    assert "standards-migration" not in sm04_eval_prompt
