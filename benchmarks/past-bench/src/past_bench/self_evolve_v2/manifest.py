"""Sequence manifest generation for the self-evolve v2 framework."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from past_bench.models.self_evolve import SelfEvolveSequenceDefinition, V2FamilyMetadata

from .schema import CONFIG_ROOT_NAME, FRAMEWORK_ROOT_NAME

_STAGE_MAP = {
    "cold": ("baseline", "cold"),
    "learn_a": ("learn", "learn_a"),
    "learn_b": ("learn", "learn_b"),
    "learn_c": ("learn", "learn_c"),
    "learn": ("learn", "learn"),
    "update_or_stabilize": ("learn", "update"),
    "attempt": ("learn", "attempt"),
    "reflection": ("reflection", "reflection"),
    "confirmation": ("learn", "confirmation"),
    "training": ("learn", "training"),
    "eval_near": ("evaluation", "eval_near"),
    "eval_far": ("evaluation", "eval_far"),
    "control": ("control", "control"),
}

_STAGE_PREFIX_TO_CANONICAL = {
    "baseline": "cold",
    "cold": "cold",
    "learn_a": "learn_a",
    "learn_b": "learn_b",
    "learn_c": "learn_c",
    "learn": "learn",
    "update_or_stabilize": "update_or_stabilize",
    "update": "update_or_stabilize",
    "attempt": "attempt",
    "reflection": "reflection",
    "confirmation": "confirmation",
    "training": "training",
    "eval_near": "eval_near",
    "eval_far": "eval_far",
    "control": "control",
}

_CANONICAL_TO_BUCKET = {
    "cold": "cold",
    "learn_a": "learn",
    "learn_b": "learn",
    "learn_c": "learn",
    "learn": "learn",
    "update_or_stabilize": "update_or_stabilize",
    "attempt": "attempt",
    "reflection": "reflection",
    "confirmation": "confirmation",
    "training": "training",
    "eval_near": "eval_near",
    "eval_far": "eval_far",
    "control": "control",
}

_DEFAULT_STAGE_ORDER = {
    "cold": 0,
    "attempt": 1,
    "reflection": 2,
    "confirmation": 3,
    "learn_a": 4,
    "learn_b": 5,
    "learn_c": 6,
    "learn": 7,
    "update_or_stabilize": 8,
    "training": 9,
    "eval_near": 10,
    "eval_far": 11,
    "control": 12,
}


def _bucket_for_task(task_dir: Path) -> str:
    """Infer bucket key from task dir name."""

    name = task_dir.name.lower()
    if "cold" in name or "baseline" in name:
        return "cold"
    if "attempt" in name:
        return "attempt"
    if "eval_far" in name or "_far" in name:
        return "eval_far"
    if "eval_near" in name or "_near" in name:
        return "eval_near"
    if "control" in name:
        return "control"
    if "reflection" in name:
        return "reflection"
    if "confirmation" in name:
        return "confirmation"
    if "update" in name or "override" in name or "stabilize" in name:
        return "update_or_stabilize"
    if "learn_b" in name or "_b_" in name or name.endswith("_b"):
        return "learn_b"
    if "learn" in name:
        return "learn_a"
    return "learn_a"


def infer_stage_name(task_dir_name: str) -> str:
    """Infer the canonical stage name from a task directory name."""

    lowered = task_dir_name.lower()
    for prefix in sorted(_STAGE_PREFIX_TO_CANONICAL, key=len, reverse=True):
        if lowered == prefix or lowered.startswith(f"{prefix}_") or lowered.startswith(f"{prefix}-"):
            return _STAGE_PREFIX_TO_CANONICAL[prefix]
    raise ValueError(
        f"cannot infer stage from task directory name {task_dir_name!r}; "
        "expected prefixes like cold_, attempt_, eval_near_, control_"
    )


def _persistence_allowed_for_task(task_dir: Path) -> bool:
    name = task_dir.name.lower()
    return "no_persistence" not in name


def _ordered_task_dirs(family_dir: Path, family_doc: dict) -> list[Path]:
    task_dirs = [
        child
        for child in family_dir.iterdir()
        if child.is_dir() and not child.name.startswith("_") and (child / "task.yaml").exists()
    ]
    declared_order = family_doc.get("episode_order") or []
    if not declared_order:
        default_sorted = sorted(
            task_dirs,
            key=lambda task_dir: (_DEFAULT_STAGE_ORDER[_bucket_for_task(task_dir)], task_dir.name),
        )
        return default_sorted

    by_name = {task_dir.name: task_dir for task_dir in task_dirs}
    ordered: list[Path] = []
    seen: set[str] = set()
    for name in declared_order:
        task_dir = by_name.get(name)
        if task_dir is None:
            raise RuntimeError(f"episode_order entry {name!r} missing under {family_dir}")
        ordered.append(task_dir)
        seen.add(name)

    return ordered


def validate_family_task_layout(family_dir: Path, family: V2FamilyMetadata) -> list[Path]:
    """Validate that on-disk tasks match `instances_per_bucket`."""

    family_doc = yaml.safe_load((family_dir / "family.yaml").read_text(encoding="utf-8")) or {}
    task_dirs = _ordered_task_dirs(family_dir, family_doc)
    if not task_dirs:
        raise ValueError(f"{family.family_id}: no task directories found under {family_dir}")

    actual_counts: dict[str, int] = {}
    for task_dir in task_dirs:
        override = family.episode_overrides.get(task_dir.name)
        stage_name = override.stage if override is not None and override.stage else _bucket_for_task(task_dir)
        bucket_key = _CANONICAL_TO_BUCKET[stage_name]
        actual_counts[bucket_key] = actual_counts.get(bucket_key, 0) + 1

    expected_counts = {
        bucket: count
        for bucket, count in family.instances_per_bucket.items()
        if count > 0
    }
    if actual_counts != expected_counts:
        raise ValueError(
            f"{family.family_id}: task bucket counts {actual_counts} do not match "
            f"family.yaml {expected_counts}"
        )

    return task_dirs


def _build_history_specs(meta: V2FamilyMetadata) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    save_anchor_by_episode: dict[str, str] = {}
    history_by_episode: dict[str, tuple[str, str]] = {}

    if meta.history_plan is None:
        return save_anchor_by_episode, history_by_episode

    anchor_names = {anchor.name for anchor in meta.history_plan.anchors}
    for anchor in meta.history_plan.anchors:
        save_anchor_by_episode[anchor.save_after] = anchor.name

    for branch in meta.history_plan.branches:
        if branch.mode == "from_anchor" and branch.anchor not in anchor_names:
            raise RuntimeError(
                f"{meta.family_id}: history branch references unknown anchor {branch.anchor!r}"
            )
        for episode_name in branch.episodes:
            history_by_episode[episode_name] = (
                branch.mode,
                branch.anchor if branch.mode == "from_anchor" else "",
            )

    return save_anchor_by_episode, history_by_episode


def _manifest_ref_for_path(path_value: str, *, repo_root: Path, base_dir: Path, use_relative: bool) -> str:
    raw_path = Path(path_value)
    if use_relative:
        return os.path.relpath(repo_root / raw_path, start=base_dir)
    return str((repo_root / raw_path).resolve())


def generate_manifest(
    family_rel: str,
    out_path: Path | None = None,
    *,
    repo_root: str | Path | None = None,
    framework_root: str | Path | None = None,
) -> Path:
    """Generate a runnable sequence manifest from a family directory."""

    repo_root_path = Path(repo_root or Path(__file__).resolve().parents[3])
    framework_root_path = Path(framework_root or (repo_root_path / FRAMEWORK_ROOT_NAME))
    family_dir = framework_root_path / family_rel
    family_yaml_path = family_dir / "family.yaml"
    if not family_yaml_path.exists():
        raise FileNotFoundError(f"family.yaml missing: {family_yaml_path}")

    family_doc = yaml.safe_load(family_yaml_path.read_text(encoding="utf-8")) or {}
    family = V2FamilyMetadata.from_yaml(family_yaml_path)
    task_dirs = validate_family_task_layout(family_dir, family)

    save_anchor_by_episode, history_by_episode = _build_history_specs(family)

    if out_path is None:
        out_path = (
            repo_root_path
            / "configs"
            / CONFIG_ROOT_NAME
            / f"hermes_self_evolve_v2_{family.family_id.lower()}_only.yaml"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    default_configs = (repo_root_path / "configs" / CONFIG_ROOT_NAME).resolve()
    resolved_out = out_path.resolve()
    use_relative = resolved_out.parent == default_configs or default_configs in resolved_out.parents

    episodes: list[dict[str, object]] = []
    for task_dir in task_dirs:
        override = family.episode_overrides.get(task_dir.name)
        bucket_key = override.stage if override is not None and override.stage else _bucket_for_task(task_dir)
        bucket, stage = _STAGE_MAP.get(bucket_key, ("evaluation", bucket_key))
        history_mode, history_load_anchor = history_by_episode.get(
            task_dir.name,
            ("fresh" if bucket in {"evaluation", "control"} else "continue", ""),
        )
        mechanism = override.mechanism if override is not None else family.expected_substrate
        expected_signal = (
            override.expected_persistence_signal
            if override is not None and override.expected_persistence_signal
            else family.expected_substrate
        )
        persistence_allowed = (
            override.persistence_allowed
            if override is not None and override.persistence_allowed is not None
            else _persistence_allowed_for_task(task_dir)
        )
        if use_relative:
            task_ref = os.path.relpath(task_dir, start=resolved_out.parent)
        else:
            task_ref = task_dir.resolve()

        preseed_ref = ""
        if override is not None and override.preseed_artifacts_dir:
            preseed_ref = _manifest_ref_for_path(
                override.preseed_artifacts_dir,
                repo_root=repo_root_path,
                base_dir=resolved_out.parent,
                use_relative=use_relative,
            )
        initial_home_ref = ""
        if override is not None and override.initial_home_fixture_dir:
            initial_home_ref = _manifest_ref_for_path(
                override.initial_home_fixture_dir,
                repo_root=repo_root_path,
                base_dir=resolved_out.parent,
                use_relative=use_relative,
            )

        episodes.append(
            {
                "task": str(task_ref),
                "label": f"{family.family_id.lower()}_{task_dir.name.lower()}",
                "family_id": family.family_id,
                "mechanism": mechanism,
                "bucket": bucket,
                "stage": stage,
                "latent_rule_id": f"{family.family_id.lower()}_v1",
                "transfer_distance": (
                    "far"
                    if bucket_key == "eval_far"
                    else "near"
                    if bucket_key == "eval_near"
                    else "none"
                ),
                "noise_profile": "heavy" if bucket_key == "eval_far" else "none",
                "expected_persistence_signal": expected_signal,
                "reflection_required": bucket_key in {"reflection", "learn_b"},
                "evaluation_requires_retrieval": bucket in {"evaluation", "control"} and persistence_allowed,
                "requires_fresh_session": bucket in {"evaluation", "control"},
                "persistence_allowed": persistence_allowed,
                "initial_home_fixture_dir": initial_home_ref,
                "preseed_artifacts_dir": preseed_ref,
                "primary_trigger": family.primary_trigger,
                "trigger_phrases": list(family.trigger_phrases),
                "shared_cold_run": bucket_key == "cold" and not initial_home_ref,
                "history_mode": history_mode,
                "history_save_anchor": save_anchor_by_episode.get(task_dir.name, ""),
                "history_load_anchor": history_load_anchor,
            }
        )

    document = {
        "name": out_path.stem,
        "description": (
            f"V2 generated from {family_rel}/family.yaml ({family.title}). "
            f"Trigger={family.primary_trigger}, substrate={family.expected_substrate}, "
            f"tier={family.family_length_tier}."
        ),
        "hermes": {
            "enabled": True,
            "memory_enabled": True,
            "user_profile_enabled": True,
            "skills_enabled": True,
            "session_search_enabled": True,
            "reflection_enabled": True,
            "memory_nudge_interval": 1,
            "memory_flush_min_turns": 1,
            "skill_creation_nudge_interval": 1,
            "background_review_wait_s": 5.0,
        },
        "episodes": episodes,
    }

    SelfEvolveSequenceDefinition.model_validate(document)
    out_path.write_text(
        yaml.safe_dump(document, sort_keys=False, default_flow_style=False, width=200),
        encoding="utf-8",
    )
    return out_path
