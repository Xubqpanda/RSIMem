"""Phase B reality-check: family.yaml `instances_per_bucket` vs actual task count.

Soft test — only validates families whose `status` is `authored` (hand-written
to match the bucket plan exactly). `skeleton` families have no tasks and
`legacy_migrated` families use the 5-episode legacy layout; both are
intentionally exempt until Phase C retrofits them.

This keeps `instances_per_bucket` load-bearing so any new family authored
under Phase D trips immediately if its task count drifts from the plan.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from past_bench.models.self_evolve import load_v2_families

REPO_ROOT = Path(__file__).resolve().parent.parent
V2_ROOT = REPO_ROOT / "self-evolve-tasks-v2"


def _authored_families():
    return [f for f in load_v2_families(V2_ROOT) if f.status == "authored"]


@pytest.mark.parametrize("fam", _authored_families(), ids=lambda f: f.family_id)
def test_authored_family_task_count_matches_plan(fam) -> None:
    fam_dir = V2_ROOT / fam.primary_ability / fam.family_id
    family_doc = yaml.safe_load(fam._source_path.read_text(encoding="utf-8")) if fam._source_path else {}
    episode_order = family_doc.get("episode_order") or []
    actual = len(episode_order) if episode_order else sum(
        1 for p in fam_dir.iterdir() if p.is_dir() and (p / "task.yaml").exists()
    )
    planned = sum(fam.instances_per_bucket.values())
    assert actual == planned, (
        f"{fam.family_id}: {actual} task dirs on disk but "
        f"instances_per_bucket sums to {planned}"
    )


def test_authored_families_are_at_least_zero() -> None:
    """Sentinel so pytest collects the module even when no family is authored yet."""
    assert len(_authored_families()) >= 0
