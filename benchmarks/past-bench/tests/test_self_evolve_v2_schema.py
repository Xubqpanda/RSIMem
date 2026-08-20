from pathlib import Path

import pytest

from past_bench.self_evolve_v2 import (
    ABILITY_DIMENSIONS,
    FRAMEWORK_ROOT_NAME,
    PHASE1_ABILITIES,
    FamilyMetadata,
    load_families,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FRAMEWORK_ROOT = REPO_ROOT / FRAMEWORK_ROOT_NAME
EXPECTED_FAMILY_COUNT = 26
AUTHORED_FAMILY_IDS = {
    "EP01_prior_case_recall",
    "EP02_exception_list_recall",
    "EP03_recall_then_modify",
    "PG01_release_decision_followup",
    "PG02_ops_exception_desk",
    "PG03_oncall_handoff_lookup",
    "PG04_temporary_waiver_audit",
    "PG05_change_freeze_followup",
    "PG06_kappa_integration_review",
    "SM01_preference_adoption",
    "SM02_constraint_retention",
    "SM03_fact_correction",
    "SM04_rule_migration",
    "SM05_weak_trigger_preference_adoption",
    "SM06_temporary_exception_pollution",
    "SM07_scoped_rule_migration",
}


def test_framework_root_exists() -> None:
    assert FRAMEWORK_ROOT.is_dir(), f"{FRAMEWORK_ROOT} missing"


def test_ability_dirs_complete() -> None:
    present = {
        path.name
        for path in FRAMEWORK_ROOT.iterdir()
        if path.is_dir() and not path.name.startswith("_") and path.name != "templates"
    }
    assert present == ABILITY_DIMENSIONS


def test_all_families_load() -> None:
    families = load_families(FRAMEWORK_ROOT)
    assert len(families) == EXPECTED_FAMILY_COUNT


def _all_families() -> list[FamilyMetadata]:
    return load_families(FRAMEWORK_ROOT)


@pytest.mark.parametrize("family", _all_families(), ids=lambda family: family.family_id)
def test_family_metadata_contract(family: FamilyMetadata) -> None:
    assert family.primary_ability in ABILITY_DIMENSIONS
    assert family.control_set
    assert family.control_set
    expected_priority = "phase1" if family.primary_ability in PHASE1_ABILITIES else "phase2"
    assert family.implementation_priority == expected_priority
    if family.family_id in AUTHORED_FAMILY_IDS:
        assert family.status == "authored"
    if family.trigger_injection_plan:
        if isinstance(family.trigger_injection_plan, dict):
            assert family.primary_trigger_stages
        assert family.trigger_mixing_rule
    if family.trigger_removal_plan and isinstance(family.trigger_removal_plan, dict):
        assert set(family.trigger_removal_plan) == {"eval_near", "eval_far"}
    if family.benchmark_role:
        assert family.sibling_comparison_plan


@pytest.mark.parametrize("family", _all_families(), ids=lambda family: family.family_id)
def test_family_directory_matches_metadata(family: FamilyMetadata) -> None:
    assert family.source_path is not None
    assert family.source_path.parent.name == family.family_id
    assert family.source_path.parent.parent.name == family.primary_ability
