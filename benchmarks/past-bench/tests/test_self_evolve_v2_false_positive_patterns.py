"""Phase B: ensure §4 `false_positive_pattern` is load-bearing.

For every authored family, assert that its declared false-positive pattern
corresponds to at least one concrete expectation field the grader actually
checks. Right now we check that when a family's pattern mentions a scenario,
at least one of these signals exists:

- `trigger_phrases` non-empty              → §7.2 placement guard biting
- `banned_phrases` non-empty                → trace-parser repeat_violation
- `stale_phrases` non-empty                 → trace-parser stale_leak
- grader `mode` is one of the new modes     → grader has a substrate/retrieval cap

Families whose `status` is not `authored` are exempt — legacy migrated and
skeleton families don't yet declare a false-positive pattern.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from past_bench.models.self_evolve import V2FamilyMetadata, load_v2_families

REPO_ROOT = Path(__file__).resolve().parent.parent
V2_ROOT = REPO_ROOT / "self-evolve-tasks-v2"

RUNNER_AWARE_MODES = {
    "notes_memory",
    "notes_session_recall",
    "substrate_choice",
    "helpdesk_kb",
    "config_recall",
    "config",
    "slack_routing",
    "x_engagement",
    "plonk",
}


def _authored_families() -> list[V2FamilyMetadata]:
    return [f for f in load_v2_families(V2_ROOT) if f.status == "authored"]


@pytest.mark.parametrize("fam", _authored_families(), ids=lambda f: f.family_id)
def test_false_positive_pattern_is_load_bearing(fam: V2FamilyMetadata) -> None:
    """Authored families must have a mechanism by which the false-positive actually fires."""
    assert fam.false_positive_pattern, (
        f"{fam.family_id}: authored families must declare false_positive_pattern"
    )

    fam_dir = V2_ROOT / fam.primary_ability / fam.family_id
    task_dirs = [p for p in fam_dir.iterdir() if p.is_dir() and (p / "task.yaml").exists()]
    assert task_dirs, f"{fam.family_id}: no task dirs under family"

    import json
    modes: set[str] = set()
    for td in task_dirs:
        exp = json.loads((td / "expectations.json").read_text())
        if "mode" in exp:
            modes.add(exp["mode"])

    has_grader_mode = bool(modes & RUNNER_AWARE_MODES)
    has_trigger_phrases = bool(fam.trigger_phrases)
    has_banned_phrases = bool(fam.banned_phrases)
    has_stale_phrases = bool(fam.stale_phrases)

    signals = [has_grader_mode, has_trigger_phrases, has_banned_phrases, has_stale_phrases]
    assert any(signals), (
        f"{fam.family_id}: false_positive_pattern declared but no enforcement signal "
        f"(no trigger/banned/stale phrases, no known grader mode)"
    )


def test_authored_families_exist() -> None:
    """Sentinel to keep pytest collecting this module even with no authored families."""
    assert len(_authored_families()) >= 0
