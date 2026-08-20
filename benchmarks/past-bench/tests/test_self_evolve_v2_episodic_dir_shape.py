"""EP family directories must not mix episode prefixes."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
V2_ROOT = REPO_ROOT / "self-evolve-tasks-v2"


def _task_dirs(family_dir: Path) -> list[str]:
    return sorted(
        child.name
        for child in family_dir.iterdir()
        if child.is_dir() and (child / "task.yaml").exists()
    )


def test_episodic_family_dirs_keep_single_prefix_membership() -> None:
    expected_prefixes = {
        ("memory_ability", "EP01_prior_case_recall"): ("EP01_",),
        ("memory_ability", "EP02_exception_list_recall"): ("EP02_",),
        ("update_ability", "EP03_recall_then_modify"): ("EP03_",),
    }

    for (ability, family_name), allowed_prefixes in expected_prefixes.items():
        family_dir = V2_ROOT / ability / family_name
        members = _task_dirs(family_dir)
        assert members, f"{family_name} has no task dirs"
        bad = [name for name in members if not name.startswith(allowed_prefixes)]
        assert not bad, (
            f"{family_name} contains cross-family task dirs: {bad}; "
            f"allowed prefixes are {allowed_prefixes}"
        )
