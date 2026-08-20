"""Generated V2 manifests should only reference existing task paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from past_bench.models.self_evolve import SelfEvolveSequenceDefinition


REPO_ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = REPO_ROOT / "configs" / "self_evolve_v2"
V2_ROOT = REPO_ROOT / "self-evolve-tasks-v2"


def _v2_manifests() -> list[Path]:
    canonical_family_ids = {
        family_dir.name.lower()
        for family_dir in V2_ROOT.glob("*/*")
        if (family_dir / "family.yaml").exists()
    }
    manifests = []
    for path in sorted(CFG_DIR.glob("hermes_self_evolve_v2_*_only.yaml")):
        stem = path.stem.removeprefix("hermes_self_evolve_v2_").removesuffix("_only")
        if stem.lower() in canonical_family_ids:
            manifests.append(path)
    return manifests


@pytest.mark.parametrize("manifest_path", _v2_manifests(), ids=lambda p: p.stem)
def test_v2_manifest_task_paths_exist(manifest_path: Path) -> None:
    seq = SelfEvolveSequenceDefinition.from_yaml(manifest_path)
    missing: list[str] = []
    for episode in seq.episodes:
        task_yaml = seq.resolve_task_yaml(episode.task)
        if not task_yaml.exists():
            missing.append(f"{episode.label}: {task_yaml}")
    assert not missing, "\n".join(missing)
