import json
from pathlib import Path

from past_bench.models.self_evolve import SelfEvolveSequenceDefinition
from past_bench.models.task import TaskDefinition
from past_bench.self_evolve_v2 import FRAMEWORK_ROOT_NAME, generate_manifest, validate_family_task_layout
from past_bench.self_evolve_v2.schema import FamilyMetadata


REPO_ROOT = Path(__file__).resolve().parent.parent
FRAMEWORK_ROOT = REPO_ROOT / FRAMEWORK_ROOT_NAME
SM_SIBLING_RELS = {
    "SM05_weak_trigger_preference_adoption": "memory_ability/SM05_weak_trigger_preference_adoption",
    "SM06_temporary_exception_pollution": "update_ability/SM06_temporary_exception_pollution",
    "SM07_scoped_rule_migration": "update_ability/SM07_scoped_rule_migration",
}
PROMPT_SNIPPETS = {
    "memory_ability/SM05_weak_trigger_preference_adoption/learn_a_explicit_save_preference": "Save this durable team default for future similar sessions",
    "memory_ability/SM05_weak_trigger_preference_adoption/learn_b_soft_preference": "standing priority-label style for later similar sessions",
    "memory_ability/SM05_weak_trigger_preference_adoption/update_or_stabilize_embedded_preference": "If the note contains a standing team summary preference",
    "update_ability/SM06_temporary_exception_pollution/learn_a_temporary_exception_seed": "Temporary exception for THIS WEEK ONLY",
    "update_ability/SM06_temporary_exception_pollution/update_or_stabilize_exception_expiry": "Last week's Maya Ortiz exception has expired.",
    "update_ability/SM07_scoped_rule_migration/update_or_stabilize_security_migration": "Ops routing is UNCHANGED.",
}


def _load_family(rel: str) -> FamilyMetadata:
    return FamilyMetadata.from_yaml(FRAMEWORK_ROOT / rel / "family.yaml")


def _expected_bucket_roles(stage: str) -> set[str]:
    if stage == "cold":
        return {"baseline"}
    if stage in {"learn_a", "learn_b", "learn_c", "learn"}:
        return {"learn"}
    if stage in {"update", "update_or_stabilize"}:
        return {"learn", "update_or_stabilize"}
    if stage in {"eval_near", "eval_far"}:
        return {"evaluation"}
    if stage == "control":
        return {"control"}
    raise AssertionError(f"unexpected stage {stage!r}")


def test_sm_sibling_families_are_authored_and_have_expected_layout() -> None:
    for family_id, rel in SM_SIBLING_RELS.items():
        family = _load_family(rel)
        assert family.family_id == family_id
        assert family.primary_ability in {"memory_ability", "update_ability"}
        assert family.status == "authored"
        assert family.benchmark_role == "diagnostic_sibling"
        assert family.compatibility_gate == "diagnostic"
        assert family.instances_per_bucket["control"] == 3
        assert family.source_path is not None

        task_dirs = validate_family_task_layout(family.source_path.parent, family)
        assert len(task_dirs) == family.total_episodes
        for task_dir in task_dirs:
            assert (task_dir / "task.yaml").is_file()
            assert (task_dir / "expectations.json").is_file()
            assert (task_dir / "grader.py").is_file()


def test_sm_sibling_manifests_align_with_expectations(tmp_path: Path) -> None:
    for family_id, rel in SM_SIBLING_RELS.items():
        family = _load_family(rel)
        manifest_path = tmp_path / f"{family_id.lower()}.yaml"
        generate_manifest(
            rel,
            out_path=manifest_path,
            repo_root=REPO_ROOT,
            framework_root=FRAMEWORK_ROOT,
        )

        sequence = SelfEvolveSequenceDefinition.from_yaml(manifest_path)
        assert len(sequence.episodes) == family.total_episodes
        control_kinds = set()

        for episode in sequence.episodes:
            task_path = sequence.resolve_task_yaml(episode.task)
            task = TaskDefinition.from_yaml(task_path)
            expectations = json.loads((task_path.parent / "expectations.json").read_text(encoding="utf-8"))

            assert task.category == "self_evolve"
            assert task.judge_rubric
            assert expectations["latent_rule_id"] == family.family_id.lower()
            assert expectations["expected_mechanism"] == "memory"
            assert expectations["mode"] == "notes_memory"
            assert expectations["bucket_role"] in _expected_bucket_roles(episode.stage)

            retrieval_contract = expectations.get("retrieval_contract", {})
            if episode.bucket == "baseline":
                assert expectations["artifact_contract"]["require_rule_keywords"] == []
                assert expectations["forbidden_artifact_keywords"]
            if episode.bucket == "learn":
                assert expectations.get("min_memory_calls", 0) > 0 or retrieval_contract.get("min_memory_calls", 0) > 0
            if episode.bucket == "evaluation":
                assert expectations.get("min_memory_injections", 0) > 0 or retrieval_contract.get("min_memory_injections", 0) > 0
            if episode.bucket == "control":
                control_kind = expectations["control_kind"]
                control_kinds.add(control_kind)
                assert control_kind in {"no_persistence", "shortcut", "wrong_mechanism"}
                if control_kind == "shortcut":
                    assert expectations["shortcut_exposes_target_rule"] is True
                    assert expectations.get("min_memory_injections", 0) == 0
                if control_kind == "wrong_mechanism":
                    assert retrieval_contract.get("evaluation_only") is False

            task_name = task_path.parent.name
            if family_id == "SM06_temporary_exception_pollution" and (
                expectations["bucket_role"] in {"learn", "update_or_stabilize", "evaluation"}
                or expectations.get("control_kind") == "no_persistence"
            ):
                assert expectations["one_off_rule_keywords"]
            if family_id == "SM07_scoped_rule_migration" and task_name != "cold_scope_baseline":
                assert expectations.get("stale_coexistence_penalty", 0.0) > 0 or episode.bucket == "learn"
            if family_id == "SM07_scoped_rule_migration" and task_name == "cold_scope_baseline":
                assert "slot_checks" in expectations
                assert "Priya Chen" in expectations["forbidden_recipients"]

        assert control_kinds == {"no_persistence", "shortcut", "wrong_mechanism"}


def test_sm_sibling_trigger_snippets_match_design() -> None:
    for rel, snippet in PROMPT_SNIPPETS.items():
        task = TaskDefinition.from_yaml(FRAMEWORK_ROOT / rel / "task.yaml")
        assert snippet in task.prompt.text


def test_sm05_all_tasks_require_structural_tsv_output() -> None:
    family_root = FRAMEWORK_ROOT / SM_SIBLING_RELS["SM05_weak_trigger_preference_adoption"]
    for expectations_path in family_root.glob("*/expectations.json"):
        expectations = json.loads(expectations_path.read_text(encoding="utf-8"))
        assert expectations.get("require_tsv_output") is True


def test_sm05_evaluation_checks_trigger_specific_rule_slots() -> None:
    family_root = FRAMEWORK_ROOT / SM_SIBLING_RELS["SM05_weak_trigger_preference_adoption"]
    for task_name in {
        "eval_near_apply_weak_trigger_style",
        "eval_far_transfer_weak_trigger_style",
    }:
        expectations = json.loads(
            (family_root / task_name / "expectations.json").read_text(encoding="utf-8")
        )
        assert set(expectations["slot_checks"]) == {
            "r1_t3_tsv_structure",
            "r2_t2_priority_normalization",
            "r3_t1_date_normalization",
        }
        assert expectations["slot_checks"]["r2_t2_priority_normalization"]["tsv_column_index"] == 1
        assert expectations["slot_checks"]["r3_t1_date_normalization"]["tsv_column_index"] == 3


def test_sm06_uses_action_item_owner_language_not_listed_owner_language() -> None:
    family_root = FRAMEWORK_ROOT / SM_SIBLING_RELS["SM06_temporary_exception_pollution"]
    for task_path in family_root.glob("*/task.yaml"):
        task = TaskDefinition.from_yaml(task_path)
        assert "listed owners" not in task.prompt.text.lower()

    for fixture_path in (FRAMEWORK_ROOT / "_shared" / "fixtures" / "sm06_notes").glob("*.json"):
        assert "listed owners" not in fixture_path.read_text(encoding="utf-8").lower()
