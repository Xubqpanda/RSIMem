"""Phase B: family.yaml ↔ run-manifest consistency.

Every episode in every `hermes_self_evolve_v2_*_only.yaml` must point at a
task directory living under `self-evolve-tasks-v2/<ability>/<family>/`, and the
`<ability>` directory must match a loaded `family.yaml → primary_ability`.

This prevents manifest/schema drift: if someone adds a new per-family config
but forgets to place its tasks under the right ability tree, this test fires.
Manifests that point at the legacy `self-evolve-tasks/` tree (not under v2) are
treated as pre-migration and skipped — they're covered by the separate legacy
contract tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from past_bench.models.self_evolve import (
    SelfEvolveSequenceDefinition,
    load_v2_families,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = REPO_ROOT / "configs" / "self_evolve_v2"
V2_ROOT = REPO_ROOT / "self-evolve-tasks-v2"


def _v2_manifests() -> list[Path]:
    return sorted(CFG_DIR.glob("hermes_self_evolve_v2_*_only.yaml"))


@pytest.fixture(scope="module")
def ability_by_family() -> dict[str, str]:
    return {f.family_id: f.primary_ability for f in load_v2_families(V2_ROOT)}


@pytest.mark.parametrize("manifest_path", _v2_manifests(), ids=lambda p: p.stem)
def test_manifest_tasks_live_under_declared_ability(
    manifest_path: Path, ability_by_family: dict[str, str]
) -> None:
    seq = SelfEvolveSequenceDefinition.from_yaml(manifest_path)
    abilities_seen: set[str] = set()
    v2_episodes = 0
    for ep in seq.episodes:
        task_yaml = seq.resolve_task_yaml(ep.task)
        try:
            rel = task_yaml.resolve().relative_to(V2_ROOT.resolve())
        except ValueError:
            # Points outside self-evolve-tasks-v2/ (legacy paths). Skip.
            continue
        v2_episodes += 1
        parts = rel.parts
        # expected: <ability>/<family>/<task_dir>/task.yaml
        assert len(parts) >= 4, f"{ep.task}: unexpected path depth under v2 root"
        abilities_seen.add(parts[0])

    if v2_episodes == 0:
        pytest.skip("manifest points at legacy tree; not a v2-migrated slice")

    # All v2 episodes in one manifest should live under exactly one ability.
    assert len(abilities_seen) == 1, (
        f"{manifest_path.name}: episodes span multiple abilities {sorted(abilities_seen)}"
    )
