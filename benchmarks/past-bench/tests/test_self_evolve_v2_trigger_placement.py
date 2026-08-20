"""Phase B/E: §7.2 trigger-placement enforcement.

§7.2 says: "Trigger should mainly appear in learn_a / learn_b / reflection /
training. Trigger should generally not be repeated in eval_near / eval_far."

This test reads every task.yaml under the populated v2 families and asserts
that if the manifest entry declares `trigger_phrases`, those phrases do not
appear verbatim in the prompt of any episode whose bucket is
`eval_near`/`eval_far`/`evaluation`. Families that do not declare any
`trigger_phrases` are skipped (no false positives before Phase C authors the
phrases).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from past_bench.models.self_evolve import SelfEvolveSequenceDefinition

REPO_ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = REPO_ROOT / "configs" / "self_evolve_v2"


def _v2_manifests() -> list[Path]:
    return sorted(CFG_DIR.glob("hermes_self_evolve_v2_*_only.yaml"))


@pytest.mark.parametrize("manifest_path", _v2_manifests(), ids=lambda p: p.stem)
def test_trigger_phrases_absent_from_eval_prompts(manifest_path: Path) -> None:
    seq = SelfEvolveSequenceDefinition.from_yaml(manifest_path)
    violations: list[str] = []
    for ep in seq.episodes:
        stage = (ep.stage or ep.bucket).lower()
        if not any(k in stage for k in ("eval_near", "eval_far", "evaluation")):
            continue
        phrases = list(ep.trigger_phrases or [])
        if not phrases:
            continue
        task_yaml = seq.resolve_task_yaml(ep.task)
        if not task_yaml.exists():
            continue
        task_data = yaml.safe_load(task_yaml.read_text()) or {}
        prompt = task_data.get("prompt") or {}
        if isinstance(prompt, dict):
            prompt_text = str(prompt.get("text") or "").lower()
        else:
            prompt_text = str(prompt).lower()
        for phrase in phrases:
            if phrase.lower() in prompt_text:
                violations.append(f"{ep.task}: leaked trigger phrase {phrase!r}")
    assert not violations, "\n".join(violations)
