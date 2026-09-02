"""Prepare one fail-closed, case-specific PAST sensitivity execution.

The preparation boundary retains audit-side family/task selection in a local
PAST manifest.  The command handed to the runtime supplies only the opaque
RSIMem method case ID, never family or oracle identities.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from .memory.family_matrix import PastFamilyMatrix
from .research_protocol import SensitivityCondition
from .sensitivity import SensitivityCase
from .sensitivity_run import SensitivityDeployment, SensitivityRunSpec
from .oracle_seed_registry import OracleSeedRegistry


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedPastSensitivityLaunch:
    """Content-free identity and command for one already registered run."""

    run_id: str
    sequence_path: Path
    sequence_digest: str
    command: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.sequence_path.is_file():
            raise ValueError("prepared sensitivity sequence is missing")
        if len(self.sequence_digest) != 64:
            raise ValueError("prepared sensitivity sequence digest is invalid")
        if not self.command or "--rsimem-method-task-id" not in self.command:
            raise ValueError("prepared sensitivity command lacks opaque method task ID")


def _select_episodes(document: Mapping[str, object], deployment: SensitivityDeployment) -> list[dict[str, object]]:
    raw = document.get("episodes")
    if not isinstance(raw, list) or not raw or not all(isinstance(item, Mapping) for item in raw):
        raise ValueError("PAST family manifest has invalid episode data")
    episodes = [dict(item) for item in raw]
    selector = deployment.episode_selector
    if selector == "family.learn_eval":
        selected = [
            item for item in episodes
            if item.get("bucket") in {"learn", "evaluation"}
        ]
        if not selected:
            raise ValueError("native-static sensitivity slice has no learn/evaluation episodes")
        return selected
    if selector == "family.eval_only":
        selected = [item for item in episodes if item.get("bucket") == "evaluation"]
        if not selected:
            raise ValueError("oracle sensitivity slice has no evaluation episode")
        return selected
    matches = [
        (index, item)
        for index, item in enumerate(episodes)
        if str(item.get("label", "")).lower().endswith("_" + selector.lower())
    ]
    if len(matches) != 1:
        raise ValueError("sensitivity episode selector is not unique")
    index, target = matches[0]
    if target.get("history_mode") != "from_anchor":
        return [target]
    anchor = target.get("history_load_anchor")
    anchors = [
        index for index, item in enumerate(episodes)
        if item.get("history_save_anchor") == anchor
    ]
    if len(anchors) != 1 or anchors[0] >= index:
        raise ValueError("sensitivity slice anchor is missing or invalid")
    # The source sequence defines the anchor state by executing its complete
    # prefix. Retaining that prefix preserves native state transitions while
    # excluding unrelated future controls.
    return episodes[: anchors[0] + 1] + [target]


def prepare_past_sensitivity_launch(
    *,
    run: SensitivityRunSpec,
    deployment: SensitivityDeployment,
    past_bench_root: Path,
    output_directory: Path,
    past_bench_binary: str = "past-bench",
    agent: str = "hermes-luna",
    oracle_seed_registry: OracleSeedRegistry | None = None,
    oracle_trusted_root: Path | None = None,
    oracle_case: SensitivityCase | None = None,
) -> PreparedPastSensitivityLaunch:
    """Write a case slice and return, but do not execute, its PAST command."""

    if deployment.deployment_id != run.deployment_id:
        raise ValueError("sensitivity run/deployment mismatch")
    if (
        deployment.case_id != run.case_id
        or deployment.condition is not run.condition
        or deployment.panel.value != run.panel
    ):
        raise ValueError("sensitivity deployment identity mismatch")
    if not deployment.executable:
        raise ValueError("sensitivity deployment is not executable")
    oracle_registration = None
    if run.condition is SensitivityCondition.TYPE_MATCHED_ORACLE:
        if oracle_seed_registry is None or oracle_trusted_root is None or oracle_case is None:
            raise ValueError("oracle sensitivity launch requires a verified seed registry")
        if (
            oracle_case.case_id != run.case_id
            or oracle_case.condition is not SensitivityCondition.TYPE_MATCHED_ORACLE
            or oracle_case.panel.value != run.panel
            or oracle_case.target_kind.value != deployment.target_kind
        ):
            raise ValueError("oracle sensitivity case does not match run identity")
        oracle_registration = oracle_seed_registry.for_case(run.case_id)
        if oracle_registration.panel.value != run.panel or oracle_registration.target_kind.value != deployment.target_kind:
            raise ValueError("oracle seed registration does not match run kind")
    root = Path(past_bench_root).expanduser().resolve()
    if not (root / "pyproject.toml").is_file():
        raise ValueError("PAST-Bench root is invalid")
    spec = PastFamilyMatrix.create_default().spec_for(run.family_id)
    family_dir = root / spec.task_root
    family_file = family_dir / "family.yaml"
    try:
        family_bytes = family_file.read_bytes()
        source = yaml.safe_load(family_bytes) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("PAST family manifest is unreadable") from exc
    if not isinstance(source, Mapping) or source.get("family_id") != run.family_id:
        raise ValueError("PAST family identity mismatch")
    oracle_seed_source: Path | None = None
    if oracle_registration is not None:
        oracle_seed_source = oracle_registration.resolve(
            Path(oracle_trusted_root),
            oracle_case,
            hashlib.sha256(family_bytes).hexdigest(),
        )
    from past_bench.self_evolve_v2 import generate_manifest

    output = Path(output_directory).expanduser().resolve()
    run_root = output / run.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    base_path = run_root / ".family_base.yaml"
    family_relative = str(Path(spec.task_root).relative_to("self-evolve-tasks-v2"))
    generate_manifest(family_relative, out_path=base_path, repo_root=root)
    try:
        base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("generated PAST family manifest is unreadable") from exc
    if not isinstance(base, Mapping):
        raise ValueError("generated PAST family manifest is invalid")
    selected = _select_episodes(base, deployment)
    if oracle_seed_source is not None:
        oracle_target = run_root / "oracle-home"
        if oracle_target.exists():
            shutil.rmtree(oracle_target)
        shutil.copytree(oracle_seed_source, oracle_target, symlinks=False)
        for episode in selected:
            episode["oracle_home_seed_dir"] = "oracle-home"
    document = {
        "name": f"sensitivity.{run.run_id}",
        "description": "Registered RSIMem Stage 3 condition slice.",
        "hermes": dict(base.get("hermes") or {}),
        "episodes": selected,
    }
    target = run_root / "sequence.yaml"
    rendered = yaml.safe_dump(document, allow_unicode=False, sort_keys=True)
    target.write_text(rendered, encoding="utf-8")
    digest = _digest(document)
    persistence = (
        "without_persistence"
        if run.condition in {
            SensitivityCondition.NO_PERSISTENCE,
            SensitivityCondition.SHORTCUT_CURRENT_INPUT,
            SensitivityCondition.WRONG_MECHANISM,
        }
        else "with_persistence"
    )
    command = (
        past_bench_binary,
        "evolve",
        "--sequence", str(target),
        "--agent", agent,
        "--persistence-variant", persistence,
        "--rsimem-method-task-id", run.method_task_id,
        "--trace-dir", str(output / run.trace_directory),
        "--rsimem-sensitivity-state-dir", str(output / run.state_directory),
        "--rsimem-sensitivity-hermes-home-dir", str(output / run.hermes_home_directory),
    )
    return PreparedPastSensitivityLaunch(run.run_id, target, digest, command)


__all__ = ["PreparedPastSensitivityLaunch", "prepare_past_sensitivity_launch"]
