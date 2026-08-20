"""Phase A contract test for Self-Evolve-Tasks-V2 family.yaml files.

Validates the §4 metadata tuple and §15 control-set requirements for every
family under `self-evolve-tasks-v2/<ability>/<family>/family.yaml`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from past_bench.models.self_evolve import (
    V2_ABILITIES,
    V2_BUCKETS,
    V2FamilyMetadata,
    load_v2_families,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
V2_ROOT = REPO_ROOT / "self-evolve-tasks-v2"

EXPECTED_FAMILY_COUNT = 26


def test_v2_root_exists() -> None:
    assert V2_ROOT.is_dir(), f"{V2_ROOT} missing"


def test_v2_ability_dirs_complete() -> None:
    present = {p.name for p in V2_ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")}
    missing = V2_ABILITIES - present
    assert not missing, f"missing ability dirs: {sorted(missing)}"


def test_v2_all_families_load() -> None:
    fams = load_v2_families(V2_ROOT)
    assert len(fams) == EXPECTED_FAMILY_COUNT, (
        f"expected {EXPECTED_FAMILY_COUNT} families, loaded {len(fams)}"
    )


def _all_families() -> list[V2FamilyMetadata]:
    return load_v2_families(V2_ROOT)


@pytest.mark.parametrize("fam", _all_families(), ids=lambda f: f.family_id)
def test_family_metadata_tuple_complete(fam: V2FamilyMetadata) -> None:
    """Every §4 field populated and consistent with parent dir."""
    assert fam.family_id
    assert fam.title
    assert fam.primary_ability in V2_ABILITIES
    assert fam.primary_trigger
    assert fam.expected_substrate
    assert fam.transfer_distance
    assert fam.family_length_tier
    assert fam.instances_per_bucket, "instances_per_bucket must be non-empty"
    # Validate every bucket key is recognized
    assert set(fam.instances_per_bucket).issubset(V2_BUCKETS)
    # total_episodes should match the sum if declared
    if fam.total_episodes:
        assert fam.total_episodes == sum(fam.instances_per_bucket.values()), (
            f"{fam.family_id}: total_episodes {fam.total_episodes} "
            f"!= sum {sum(fam.instances_per_bucket.values())}"
        )


@pytest.mark.parametrize("fam", _all_families(), ids=lambda f: f.family_id)
def test_family_has_required_controls(fam: V2FamilyMetadata) -> None:
    """Every family should declare at least one explicit control condition."""
    assert fam.control_set


@pytest.mark.parametrize("fam", _all_families(), ids=lambda f: f.family_id)
def test_family_dir_matches_metadata(fam: V2FamilyMetadata) -> None:
    """family.yaml must live under <primary_ability>/<family_id>/."""
    assert fam._source_path is not None
    parent_family = fam._source_path.parent.name
    parent_ability = fam._source_path.parent.parent.name
    assert parent_family == fam.family_id, (
        f"{fam._source_path}: dir name {parent_family} != family_id {fam.family_id}"
    )
    assert parent_ability == fam.primary_ability, (
        f"{fam._source_path}: ability dir {parent_ability} != primary_ability {fam.primary_ability}"
    )


def test_ep01_history_plan_loads() -> None:
    fam = V2FamilyMetadata.from_yaml(
        V2_ROOT / "memory_ability" / "EP01_prior_case_recall" / "family.yaml"
    )
    assert fam.history_plan is not None
    assert [anchor.name for anchor in fam.history_plan.anchors] == ["ep01_post_learn"]
    assert fam.history_plan.anchors[0].save_after == "EP01_I03_recall_noise"
