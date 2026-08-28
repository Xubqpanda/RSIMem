"""Helpers for sequence-based self-evolve evaluation.

Modified by RSIMem to propagate request-level usage aggregates.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from ..models.content import TextBlock
from ..models.scoring import compute_task_score, is_pass
from ..models.self_evolve import (
    RSIMemAdaptiveWritebackConfig,
    RSIMemExtractionTrialProfile,
)
from ..models.task import Prompt, TaskDefinition
from ..models.trace import TraceMessage
from ..trace.reader import load_trace


def build_hermes_extra_body(
    *,
    home_dir: Path,
    artifacts_dir: Path,
    persistence_enabled: bool,
    memory_enabled: bool,
    user_profile_enabled: bool,
    skills_enabled: bool,
    session_search_enabled: bool,
    memory_nudge_interval: int,
    memory_flush_min_turns: int,
    skill_creation_nudge_interval: int,
    background_review_wait_s: float,
    initial_home_fixture_dir: Path | None = None,
    preseed_artifacts_dir: Path | None = None,
    rsimem_mode: str = "native",
    rsimem_adapter_failure_policy: str = "fail_closed",
    rsimem_verify_native_projection: bool = False,
    rsimem_lifecycle_evaluator_mode: str = "disabled",
    rsimem_lifecycle_policy_version: str = "phase1-dry-run-v1",
    rsimem_lifecycle_compiler_version: str = "uncompiled-v0",
    rsimem_lifecycle_timeout_seconds: float = 30.0,
    rsimem_lifecycle_max_output_tokens: int = 4096,
    rsimem_semantic_writeback_mode: str = "disabled",
    rsimem_semantic_writeback_timeout_seconds: float = 30.0,
    rsimem_semantic_writeback_max_output_tokens: int = 4096,
    rsimem_semantic_feedback_contract: str = "disabled",
    rsimem_adaptive_config: RSIMemAdaptiveWritebackConfig | dict | None = None,
    rsimem_adaptive_policy_source_path: str = "",
    rsimem_extraction_trial_profile: RSIMemExtractionTrialProfile | dict | None = None,
    rsimem_extraction_trial_source_path: str = "",
) -> dict[str, Any]:
    """Return a ``model.extra_body`` override for the Hermes adapter."""

    enabled_toolsets = ["memory"] if memory_enabled or user_profile_enabled else []
    if skills_enabled:
        enabled_toolsets.append("skills")
    if session_search_enabled:
        enabled_toolsets.append("session_search")

    semantic_writeback_mode = (
        rsimem_semantic_writeback_mode if persistence_enabled else "disabled"
    )
    semantic_writeback = {
        "mode": semantic_writeback_mode,
        "timeout_seconds": rsimem_semantic_writeback_timeout_seconds,
        "max_output_tokens": rsimem_semantic_writeback_max_output_tokens,
        "feedback_contract": (
            rsimem_semantic_feedback_contract
            if semantic_writeback_mode != "disabled"
            else "disabled"
        ),
    }
    if semantic_writeback_mode in {"static", "static_utility", "adaptive_utility"}:
        if rsimem_mode != "native+ledger":
            raise ValueError("static semantic writeback requires native+ledger mode")
        if semantic_writeback_mode == "adaptive_utility":
            if rsimem_adaptive_config is None:
                raise ValueError("adaptive semantic writeback requires adaptive config")
            adaptive = RSIMemAdaptiveWritebackConfig.model_validate(
                rsimem_adaptive_config
            )
            source = Path(rsimem_adaptive_policy_source_path).expanduser()
            if not rsimem_adaptive_policy_source_path or not source.is_file():
                raise ValueError(
                    "adaptive semantic writeback requires a prepared policy store"
                )
            source = source.resolve()
            target = (
                home_dir / adaptive.adaptive_policy_store_path
            ).expanduser().resolve()
            resolved_home = home_dir.expanduser().resolve()
            if not target.is_relative_to(resolved_home):
                raise ValueError("adaptive policy store escapes Hermes home")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.read_bytes() != source.read_bytes():
                    raise ValueError(
                        "attempt-local adaptive policy store conflicts with prepared store"
                    )
            else:
                shutil.copy2(source, target)
            semantic_writeback.update(adaptive.model_dump(
                mode="json",
                exclude={"schema_version", "prepared_policy_store_file"},
            ))
        elif rsimem_adaptive_config is not None:
            raise ValueError("adaptive config requires adaptive_utility mode")
        if rsimem_extraction_trial_profile is not None:
            if semantic_writeback_mode != "static":
                raise ValueError(
                    "extraction matched trial requires static semantic writeback"
                )
            profile = RSIMemExtractionTrialProfile.model_validate(
                rsimem_extraction_trial_profile
            )
            target_config = _materialize_extraction_trial_bundle(
                source_config_path=rsimem_extraction_trial_source_path,
                artifacts_dir=artifacts_dir,
                expected_profile=profile,
            )
            semantic_writeback.update({
                "extraction_runtime_scope": "matched_validation",
                "extraction_runtime_config_path": str(target_config),
            })
        elif rsimem_extraction_trial_source_path:
            raise ValueError("extraction trial source requires a trial profile")
        enabled_toolsets = [
            toolset for toolset in enabled_toolsets if toolset != "memory"
        ]
    elif rsimem_semantic_feedback_contract != "disabled" and persistence_enabled:
        raise ValueError("semantic feedback contract requires writeback mode")
    elif (
        rsimem_extraction_trial_profile is not None
        or rsimem_extraction_trial_source_path
    ) and persistence_enabled:
        raise ValueError("extraction trial requires static semantic writeback")

    return {
        "hermes": {
            "persistence_enabled": persistence_enabled,
            "session_search_enabled": session_search_enabled,
            "home_dir": str(home_dir),
            "capture_artifacts_dir": str(artifacts_dir),
            "background_review_wait_s": background_review_wait_s,
            "enabled_toolsets": enabled_toolsets,
            "config_overrides": {
                "memory": {
                    "memory_enabled": memory_enabled,
                    "user_profile_enabled": user_profile_enabled,
                    "nudge_interval": memory_nudge_interval,
                    "flush_min_turns": memory_flush_min_turns,
                },
                "skills": {
                    "creation_nudge_interval": skill_creation_nudge_interval,
                },
            },
            "initial_home_fixture_dir": str(initial_home_fixture_dir) if initial_home_fixture_dir else "",
            "preseed_artifacts_dir": str(preseed_artifacts_dir) if preseed_artifacts_dir else "",
            "rsimem": {
                "mode": rsimem_mode,
                "adapter_failure_policy": rsimem_adapter_failure_policy,
                "verify_native_projection": rsimem_verify_native_projection,
                "evidence_path": str(artifacts_dir / "rsimem_memory_events.jsonl"),
                "lifecycle": {
                    "evaluator_mode": rsimem_lifecycle_evaluator_mode,
                    "policy_version": rsimem_lifecycle_policy_version,
                    "compiler_version": rsimem_lifecycle_compiler_version,
                    "timeout_seconds": rsimem_lifecycle_timeout_seconds,
                    "max_output_tokens": rsimem_lifecycle_max_output_tokens,
                },
                "semantic_writeback": semantic_writeback,
            },
        }
    }


def _copy_immutable_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != source.read_bytes():
            raise ValueError(
                f"attempt-local extraction trial {target.name} conflicts with source"
            )
        return
    shutil.copy2(source, target)


def _materialize_extraction_trial_bundle(
    *,
    source_config_path: str,
    artifacts_dir: Path,
    expected_profile: RSIMemExtractionTrialProfile,
) -> Path:
    if not source_config_path:
        raise ValueError("extraction matched trial requires a source config")
    source = Path(source_config_path).expanduser().resolve()
    from rsimem.extraction_validation_runtime import (
        EXTRACTION_TRIAL_CONFIG_FILE,
        EXTRACTION_TRIAL_OFFLINE_DECISION_FILE,
        EXTRACTION_TRIAL_POLICY_STORE_FILE,
        load_extraction_matched_trial_profile,
    )

    resolved = load_extraction_matched_trial_profile(source)
    actual_profile = RSIMemExtractionTrialProfile.model_validate(
        resolved.profile()
    )
    if actual_profile != expected_profile:
        raise ValueError("extraction trial source profile differs from manifest")
    target_root = artifacts_dir.expanduser().resolve() / "rsimem_extraction_trial"
    targets = {
        resolved.config_path: target_root / EXTRACTION_TRIAL_CONFIG_FILE,
        resolved.policy_store_path: target_root / EXTRACTION_TRIAL_POLICY_STORE_FILE,
        resolved.offline_decision_path: (
            target_root / EXTRACTION_TRIAL_OFFLINE_DECISION_FILE
        ),
    }
    for source_path, target_path in targets.items():
        _copy_immutable_file(source_path, target_path)
    target_config = target_root / EXTRACTION_TRIAL_CONFIG_FILE
    copied = load_extraction_matched_trial_profile(target_config)
    copied_profile = RSIMemExtractionTrialProfile.model_validate(copied.profile())
    if copied_profile != expected_profile:
        raise ValueError("attempt-local extraction trial profile differs")
    return target_config


def materialize_task_hermes_seed(
    *,
    task: TaskDefinition,
    target_dir: Path,
    base_preseed_dir: Path | None = None,
) -> Path | None:
    """Materialize a task-level Hermes seed overlay into `target_dir`.

    Returns the directory to pass through `preseed_artifacts_dir`, or None
    when neither sequence-level nor task-level seed data exists.
    """

    seed = task.hermes_home_seed
    has_task_seed = bool(
        seed
        and any(
            str(value or "").strip()
            for value in (seed.memory_seed, seed.skill_seed, seed.session_seed)
        )
    )
    if base_preseed_dir is None and not has_task_seed:
        return None

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if base_preseed_dir is not None and base_preseed_dir.exists():
        _copy_seed_item(base_preseed_dir, target_dir)

    if not has_task_seed:
        return target_dir

    task_dir = Path(task.task_file).resolve().parent if task.task_file else Path.cwd()

    def _resolve_seed_path(value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (task_dir / path).resolve()
        return path

    if seed is not None and seed.memory_seed:
        _copy_memory_seed(_resolve_seed_path(seed.memory_seed), target_dir / "memories")
    if seed is not None and seed.skill_seed:
        _copy_skill_seed(_resolve_seed_path(seed.skill_seed), target_dir / "skills")
    if seed is not None and seed.session_seed:
        _copy_session_seed(_resolve_seed_path(seed.session_seed), target_dir)

    return target_dir


def _copy_seed_item(src: Path, dst: Path) -> None:
    if src.is_dir():
        for child in src.iterdir():
            child_dst = dst / child.name
            if child.is_dir():
                if child_dst.exists():
                    shutil.rmtree(child_dst)
                shutil.copytree(child, child_dst)
            else:
                child_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(child, child_dst)
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_memory_seed(src: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        for name in ("MEMORY.md", "USER.md"):
            candidate = src / name
            if candidate.exists():
                shutil.copy2(candidate, dst_dir / name)
        return
    shutil.copy2(src, dst_dir / src.name)


def _copy_skill_seed(src: Path, dst_dir: Path) -> None:
    if not src.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        for child in src.iterdir():
            child_dst = dst_dir / child.name
            if child.is_dir():
                if child_dst.exists():
                    shutil.rmtree(child_dst)
                shutil.copytree(child, child_dst)
            elif child.name == "SKILL.md":
                skill_dir = dst_dir / src.name
                skill_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(child, skill_dir / "SKILL.md")
        return

    # Allow pointing directly at a single SKILL.md file.
    if src.name == "SKILL.md":
        skill_dir = dst_dir / src.parent.name
        skill_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, skill_dir / "SKILL.md")


def _copy_session_seed(src: Path, dst_dir: Path) -> None:
    if not src.exists():
        return
    if src.is_dir():
        state_db = src / "state.db"
        sessions_dir = src / "sessions"
        session_seed_json = src / "session_seed.json"
        if state_db.exists():
            shutil.copy2(state_db, dst_dir / "state.db")
        if sessions_dir.exists():
            if (dst_dir / "sessions").exists():
                shutil.rmtree(dst_dir / "sessions")
            shutil.copytree(sessions_dir, dst_dir / "sessions")
        if session_seed_json.exists():
            shutil.copy2(session_seed_json, dst_dir / "session_seed.json")
        return

    if src.name == "state.db":
        shutil.copy2(src, dst_dir / "state.db")
    else:
        shutil.copy2(src, dst_dir / "session_seed.json")


def build_nanobot_extra_body(
    *,
    workspace_dir: Path,
    artifacts_dir: Path,
    persistence_enabled: bool,
    session_key: str,
    restrict_to_workspace: bool = True,
) -> dict[str, Any]:
    """Return a ``model.extra_body`` override for the Nanobot adapter."""

    return {
        "nanobot": {
            "workspace_dir": str(workspace_dir),
            "artifacts_dir": str(artifacts_dir),
            "persistence_enabled": persistence_enabled,
            "session_key": session_key,
            "restrict_to_workspace": restrict_to_workspace,
        }
    }


def build_zeroclaw_extra_body(
    *,
    workspace_dir: Path,
    home_dir: Path,
    session_history_dir: Path,
    artifacts_dir: Path,
    persistence_enabled: bool,
    recursion_limit: int = 100,
    system_prompt: str = "",
    memory_enabled: bool = True,
    session_search_enabled: bool = True,
) -> dict[str, Any]:
    """Return a ``model.extra_body`` override for the ZeroClaw adapter."""

    return {
        "zeroclaw": {
            "workspace_dir": str(workspace_dir),
            "home_dir": str(home_dir),
            "session_history_dir": str(session_history_dir),
            "artifacts_dir": str(artifacts_dir),
            "persistence_enabled": persistence_enabled,
            "recursion_limit": recursion_limit,
            "system_prompt": system_prompt,
            "memory_enabled": memory_enabled,
            "session_search_enabled": session_search_enabled,
        }
    }


def _copy_named_entries(src_dir: Path | None, dst_dir: Path, names: tuple[str, ...], *, overlay: bool) -> None:
    if src_dir is None or not src_dir.exists():
        return
    for name in names:
        src = src_dir / name
        dst = dst_dir / name
        if not src.exists():
            continue
        if src.is_dir():
            if dst.exists() and not overlay:
                continue
            if dst.exists() and overlay:
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            if dst.exists() and not overlay:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _append_memory_entries_to_store(src_dir: Path, dst_file: Path) -> None:
    entries: list[str] = []
    for name in ("MEMORY.md", "USER.md"):
        candidate = src_dir / name
        if not candidate.exists():
            continue
        text = candidate.read_text(encoding="utf-8")
        entries.extend(entry for entry in _split_memory_entries(text) if entry)
    if not entries:
        return
    current: dict[str, str] = {}
    if dst_file.exists():
        try:
            current = json.loads(dst_file.read_text(encoding="utf-8")) or {}
        except Exception:
            current = {}
    existing_normalized = {
        _normalize_memory_entry_text(str(value))
        for value in current.values()
        if value is not None and _normalize_memory_entry_text(str(value))
    }
    next_index = len(current) + 1
    for entry in entries:
        normalized = _normalize_memory_entry_text(entry)
        if not normalized or normalized in existing_normalized:
            continue
        current[f"entry_{next_index:03d}"] = entry
        existing_normalized.add(normalized)
        next_index += 1
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    dst_file.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")


def _materialize_zeroclaw_session_history(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    sessions_dir = src_dir / "sessions"
    if sessions_dir.exists():
        for child in sessions_dir.iterdir():
            target = dst_dir / child.name
            if child.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(child, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(child, target)

    session_seed_json = src_dir / "session_seed.json"
    if not session_seed_json.exists():
        return

    try:
        payload = json.loads(session_seed_json.read_text(encoding="utf-8")) or {}
    except Exception:
        shutil.copy2(session_seed_json, dst_dir / "session_seed.json")
        return

    sessions = payload.get("sessions")
    if not isinstance(sessions, list):
        shutil.copy2(session_seed_json, dst_dir / "session_seed.json")
        return

    for index, session in enumerate(sessions, start=1):
        if not isinstance(session, dict):
            continue
        session_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(session.get("id") or f"seed_{index:03d}")).strip("_")
        if not session_id:
            session_id = f"seed_{index:03d}"
        target = dst_dir / f"{index:03d}_{session_id}.json"
        target.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")


class PersistenceBackend:
    """Agent-specific runtime state backend for self-evolve episode sequences."""

    state_root_name = "state"

    def family_paths(self, variant_dir: Path, family_id: str) -> tuple[Path, Path]:
        family_root = variant_dir / "family_homes" / family_id
        return family_root / self.state_root_name, family_root / "history_anchors"

    def reset_state(self, state_root: Path) -> None:
        if state_root.exists():
            shutil.rmtree(state_root)
        state_root.mkdir(parents=True, exist_ok=True)

    def clone_state(self, src_root: Path, dst_root: Path) -> None:
        if dst_root.exists():
            shutil.rmtree(dst_root)
        if src_root.exists():
            shutil.copytree(src_root, dst_root)
        else:
            dst_root.mkdir(parents=True, exist_ok=True)

    def prepare_history(self, *, state_root: Path, episode: Any, history_anchors: dict[str, Path]) -> None:
        if episode.history_mode == "continue":
            state_root.mkdir(parents=True, exist_ok=True)
            return
        if episode.history_mode == "fresh":
            self.reset_state(state_root)
            return
        if episode.history_mode == "from_anchor":
            anchor_root = history_anchors.get(episode.history_load_anchor)
            if anchor_root is None:
                raise RuntimeError(
                    f"missing history anchor {episode.history_load_anchor!r} for episode {episode.label or episode.task}"
                )
            self.clone_state(anchor_root, state_root)
            return
        raise RuntimeError(f"unsupported history_mode {episode.history_mode!r}")

    def save_anchor(
        self,
        *,
        state_root: Path,
        episode: Any,
        anchors_dir: Path,
        history_anchors: dict[str, Path],
    ) -> None:
        if not episode.history_save_anchor:
            return
        anchor_root = anchors_dir / episode.history_save_anchor
        anchors_dir.mkdir(parents=True, exist_ok=True)
        self.clone_state(state_root, anchor_root)
        history_anchors[episode.history_save_anchor] = anchor_root

    def materialize_inputs(
        self,
        *,
        state_root: Path,
        initial_home_fixture_dir: Path | None,
        preseed_artifacts_dir: Path | None,
    ) -> None:
        raise NotImplementedError

    def build_extra_body(
        self,
        *,
        state_root: Path,
        artifacts_dir: Path,
        persistence_enabled: bool,
        sequence: Any,
        family_id: str,
        review_wait_s: float,
        tool_config: dict[str, bool],
    ) -> dict[str, Any]:
        raise NotImplementedError

    def snapshot_before(self, state_root: Path, *, include_contents: bool = False) -> dict[str, Any]:
        raise NotImplementedError

    def snapshot_after(self, artifacts_dir: Path) -> dict[str, Any]:
        raise NotImplementedError


class HermesPersistenceBackend(PersistenceBackend):
    state_root_name = "hermes_home"

    def materialize_inputs(
        self,
        *,
        state_root: Path,
        initial_home_fixture_dir: Path | None,
        preseed_artifacts_dir: Path | None,
    ) -> None:
        _copy_named_entries(
            initial_home_fixture_dir,
            state_root,
            ("memories", "skills", "sessions", "state.db", "state.db-wal", "state.db-shm"),
            overlay=False,
        )
        _copy_named_entries(
            preseed_artifacts_dir,
            state_root,
            ("memories", "skills"),
            overlay=True,
        )

    def build_extra_body(
        self,
        *,
        state_root: Path,
        artifacts_dir: Path,
        persistence_enabled: bool,
        sequence: Any,
        family_id: str,
        review_wait_s: float,
        tool_config: dict[str, bool],
    ) -> dict[str, Any]:
        return build_hermes_extra_body(
            home_dir=state_root,
            artifacts_dir=artifacts_dir,
            persistence_enabled=persistence_enabled,
            memory_enabled=tool_config["memory_enabled"],
            user_profile_enabled=tool_config["user_profile_enabled"],
            skills_enabled=tool_config["skills_enabled"],
            session_search_enabled=tool_config["session_search_enabled"],
            memory_nudge_interval=sequence.hermes.memory_nudge_interval,
            memory_flush_min_turns=sequence.hermes.memory_flush_min_turns,
            skill_creation_nudge_interval=sequence.hermes.skill_creation_nudge_interval,
            background_review_wait_s=review_wait_s,
            rsimem_mode=sequence.hermes.rsimem_mode,
            rsimem_adapter_failure_policy=(
                sequence.hermes.rsimem_adapter_failure_policy
            ),
            rsimem_verify_native_projection=(
                sequence.hermes.rsimem_verify_native_projection
            ),
            rsimem_lifecycle_evaluator_mode=(
                sequence.hermes.rsimem_lifecycle_evaluator_mode
            ),
            rsimem_lifecycle_policy_version=(
                sequence.hermes.rsimem_lifecycle_policy_version
            ),
            rsimem_lifecycle_compiler_version=(
                sequence.hermes.rsimem_lifecycle_compiler_version
            ),
            rsimem_lifecycle_timeout_seconds=(
                sequence.hermes.rsimem_lifecycle_timeout_seconds
            ),
            rsimem_lifecycle_max_output_tokens=(
                sequence.hermes.rsimem_lifecycle_max_output_tokens
            ),
            rsimem_semantic_writeback_mode=(
                sequence.hermes.rsimem_semantic_writeback_mode
            ),
            rsimem_semantic_writeback_timeout_seconds=(
                sequence.hermes.rsimem_semantic_writeback_timeout_seconds
            ),
            rsimem_semantic_writeback_max_output_tokens=(
                sequence.hermes.rsimem_semantic_writeback_max_output_tokens
            ),
            rsimem_semantic_feedback_contract=(
                sequence.hermes.rsimem_semantic_feedback_contract
            ),
            rsimem_adaptive_config=sequence.hermes.rsimem_adaptive_config,
            rsimem_adaptive_policy_source_path=(
                sequence.hermes.rsimem_adaptive_policy_source_path
            ),
            rsimem_extraction_trial_profile=(
                sequence.hermes.rsimem_extraction_trial_profile
            ),
            rsimem_extraction_trial_source_path=(
                sequence.hermes.rsimem_extraction_trial_source_path
            ),
        )

    def snapshot_before(self, state_root: Path, *, include_contents: bool = False) -> dict[str, Any]:
        return snapshot_hermes_home(state_root, include_contents=include_contents)

    def snapshot_after(self, artifacts_dir: Path) -> dict[str, Any]:
        return snapshot_hermes_artifacts(artifacts_dir)


class NanobotPersistenceBackend(PersistenceBackend):
    state_root_name = "nanobot_state"

    @staticmethod
    def workspace_dir(state_root: Path) -> Path:
        return state_root / "workspace"

    def reset_state(self, state_root: Path) -> None:
        super().reset_state(state_root)
        self.workspace_dir(state_root).mkdir(parents=True, exist_ok=True)

    def materialize_inputs(
        self,
        *,
        state_root: Path,
        initial_home_fixture_dir: Path | None,
        preseed_artifacts_dir: Path | None,
    ) -> None:
        workspace = self.workspace_dir(state_root)
        workspace.mkdir(parents=True, exist_ok=True)
        for src_dir in (initial_home_fixture_dir, preseed_artifacts_dir):
            if src_dir is None or not src_dir.exists():
                continue
            memories_dir = src_dir / "memories"
            if memories_dir.exists():
                target_memory_dir = workspace / "memory"
                target_memory_dir.mkdir(parents=True, exist_ok=True)
                for name in ("MEMORY.md", "USER.md"):
                    src = memories_dir / name
                    if src.exists():
                        shutil.copy2(src, target_memory_dir / name)
            skills_dir = src_dir / "skills"
            if skills_dir.exists():
                target_skills = workspace / "skills"
                target_skills.mkdir(parents=True, exist_ok=True)
                for child in skills_dir.iterdir():
                    dst = target_skills / child.name
                    if dst.exists():
                        shutil.rmtree(dst)
                    if child.is_dir():
                        shutil.copytree(child, dst)
            sessions_dir = src_dir / "sessions"
            if sessions_dir.exists():
                target_sessions = workspace / "sessions"
                if target_sessions.exists():
                    shutil.rmtree(target_sessions)
                shutil.copytree(sessions_dir, target_sessions)

    def build_extra_body(
        self,
        *,
        state_root: Path,
        artifacts_dir: Path,
        persistence_enabled: bool,
        sequence: Any,
        family_id: str,
        review_wait_s: float,
        tool_config: dict[str, bool],
    ) -> dict[str, Any]:
        del sequence, review_wait_s, tool_config
        return build_nanobot_extra_body(
            workspace_dir=self.workspace_dir(state_root),
            artifacts_dir=artifacts_dir,
            persistence_enabled=persistence_enabled,
            session_key=f"self-evolve:{family_id}",
            restrict_to_workspace=True,
        )

    def snapshot_before(self, state_root: Path, *, include_contents: bool = False) -> dict[str, Any]:
        return snapshot_nanobot_workspace(self.workspace_dir(state_root), include_contents=include_contents)

    def snapshot_after(self, artifacts_dir: Path) -> dict[str, Any]:
        return snapshot_nanobot_artifacts(artifacts_dir)


class ZeroClawPersistenceBackend(PersistenceBackend):
    state_root_name = "zeroclaw_state"

    @staticmethod
    def workspace_dir(state_root: Path) -> Path:
        return state_root / "workspace"

    @staticmethod
    def home_dir(state_root: Path) -> Path:
        return state_root / "home"

    @staticmethod
    def session_history_dir(state_root: Path) -> Path:
        return state_root / "session_history"

    def reset_state(self, state_root: Path) -> None:
        super().reset_state(state_root)
        self.workspace_dir(state_root).mkdir(parents=True, exist_ok=True)
        self.home_dir(state_root).mkdir(parents=True, exist_ok=True)
        self.session_history_dir(state_root).mkdir(parents=True, exist_ok=True)

    def materialize_inputs(
        self,
        *,
        state_root: Path,
        initial_home_fixture_dir: Path | None,
        preseed_artifacts_dir: Path | None,
    ) -> None:
        workspace = self.workspace_dir(state_root)
        home = self.home_dir(state_root)
        session_history = self.session_history_dir(state_root)
        workspace.mkdir(parents=True, exist_ok=True)
        home.mkdir(parents=True, exist_ok=True)
        session_history.mkdir(parents=True, exist_ok=True)
        for src_dir in (initial_home_fixture_dir, preseed_artifacts_dir):
            if src_dir is None or not src_dir.exists():
                continue
            skills_dir = src_dir / "skills"
            if skills_dir.exists():
                target_skills = workspace / "skills"
                target_skills.mkdir(parents=True, exist_ok=True)
                for child in skills_dir.iterdir():
                    dst = target_skills / child.name
                    if dst.exists():
                        shutil.rmtree(dst)
                    if child.is_dir():
                        shutil.copytree(child, dst)
            memories_dir = src_dir / "memories"
            if memories_dir.exists():
                _append_memory_entries_to_store(memories_dir, home / ".zeroclaw" / "memory_store.json")
            _materialize_zeroclaw_session_history(src_dir, session_history)

    def build_extra_body(
        self,
        *,
        state_root: Path,
        artifacts_dir: Path,
        persistence_enabled: bool,
        sequence: Any,
        family_id: str,
        review_wait_s: float,
        tool_config: dict[str, bool],
    ) -> dict[str, Any]:
        return build_zeroclaw_extra_body(
            workspace_dir=self.workspace_dir(state_root),
            home_dir=self.home_dir(state_root),
            session_history_dir=self.session_history_dir(state_root),
            artifacts_dir=artifacts_dir,
            persistence_enabled=persistence_enabled,
            recursion_limit=100,
            system_prompt=getattr(sequence.hermes, "system_prompt", ""),
            memory_enabled=tool_config["memory_enabled"] or tool_config["user_profile_enabled"],
            session_search_enabled=tool_config["session_search_enabled"],
        )

    def snapshot_before(self, state_root: Path, *, include_contents: bool = False) -> dict[str, Any]:
        return snapshot_zeroclaw_state(
            home_dir=self.home_dir(state_root),
            workspace_dir=self.workspace_dir(state_root),
            include_contents=include_contents,
        )

    def snapshot_after(self, artifacts_dir: Path) -> dict[str, Any]:
        return snapshot_zeroclaw_artifacts(artifacts_dir)


def make_persistence_backend(agent_name: str) -> PersistenceBackend:
    if agent_name.startswith("hermes"):
        return HermesPersistenceBackend()
    if agent_name == "nanobot":
        return NanobotPersistenceBackend()
    if agent_name == "zeroclaw":
        return ZeroClawPersistenceBackend()
    raise RuntimeError(f"Agent {agent_name} does not support self-evolve persistence backends")


def resolve_episode_tool_config(
    *,
    persistence_enabled: bool,
    expected_signal: str,
    memory_enabled: bool,
    user_profile_enabled: bool,
    skills_enabled: bool,
    session_search_enabled: bool,
) -> dict[str, bool]:
    """Resolve Hermes persistence toolsets for one episode.

    `without_persistence` disables all persistence-facing toolsets.
    `with_persistence` constrains single-mechanism families to the expected toolset
    so score gains are easier to attribute.
    """

    if not persistence_enabled:
        return {
            "memory_enabled": False,
            "user_profile_enabled": False,
            "skills_enabled": False,
            "session_search_enabled": False,
        }

    if expected_signal == "skill":
        return {
            "memory_enabled": False,
            "user_profile_enabled": False,
            "skills_enabled": skills_enabled,
            "session_search_enabled": False,
        }
    if expected_signal == "memory":
        return {
            "memory_enabled": memory_enabled,
            "user_profile_enabled": user_profile_enabled,
            "skills_enabled": False,
            "session_search_enabled": False,
        }
    if expected_signal == "session_search":
        return {
            "memory_enabled": False,
            "user_profile_enabled": False,
            "skills_enabled": False,
            "session_search_enabled": session_search_enabled,
        }
    return {
        "memory_enabled": memory_enabled,
        "user_profile_enabled": user_profile_enabled,
        "skills_enabled": skills_enabled,
        "session_search_enabled": session_search_enabled,
    }


_MEMORY_ENTRY_DELIMITER = "\n§\n"
_WRITE_ENDPOINT_PATTERNS = (
    "/helpdesk/tickets/update",
    "/helpdesk/tickets/close",
    "/notes/share",
    "/config/notify",
    "/config/integrations/update",
)
_MEMORY_WRITE_ACTIONS = {"add", "replace", "remove"}
_MEMORY_READ_ACTIONS = {"get", "read", "list", "search", "view"}
_INFRA_BLOCK_PATTERNS = (
    "Session database unavailable",
    "session database unavailable",
    "Connection refused",
    "All cross-session memory services are unavailable",
    "All data sources are unavailable",
)
_CONTROL_REPLY_SENTINELS = {"NO_REPLY", "HEARTBEAT_OK"}


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _split_memory_entries(text: str) -> list[str]:
    if not text.strip():
        return []
    return [entry.strip() for entry in text.split(_MEMORY_ENTRY_DELIMITER) if entry.strip()]


def _normalize_memory_entry_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_\u4e00-\u9fff-]+", text.lower()))


def _similarity(left: str, right: str) -> float:
    left_words = _word_set(left)
    right_words = _word_set(right)
    if not left_words or not right_words:
        return 0.0
    overlap = len(left_words & right_words)
    total = len(left_words | right_words)
    return overlap / total if total else 0.0


def _pair_updated_entries(before_entries: list[str], after_entries: list[str]) -> tuple[list[dict[str, str]], list[str], list[str]]:
    remaining_before = list(before_entries)
    remaining_after = list(after_entries)
    updated: list[dict[str, str]] = []

    while remaining_before and remaining_after:
        best_score = 0.0
        best_pair: tuple[int, int] | None = None
        for left_idx, left_entry in enumerate(remaining_before):
            for right_idx, right_entry in enumerate(remaining_after):
                score = _similarity(left_entry, right_entry)
                if score > best_score:
                    best_score = score
                    best_pair = (left_idx, right_idx)
        if best_pair is None or best_score < 0.20:
            break
        left_idx, right_idx = best_pair
        updated.append({
            "before": remaining_before.pop(left_idx),
            "after": remaining_after.pop(right_idx),
        })

    return updated, remaining_before, remaining_after


def _safe_json_loads(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _strip_control_reply_text(text: str) -> str:
    if not text:
        return ""

    stripped = text.strip()
    if stripped in _CONTROL_REPLY_SENTINELS:
        return ""

    lines = [
        line
        for line in stripped.splitlines()
        if line.strip() != "[[reply_to_current]]"
    ]
    cleaned = "\n".join(lines).strip()
    if cleaned in _CONTROL_REPLY_SENTINELS:
        return ""
    return cleaned


def _extract_trace_final_text(messages: list[TraceMessage]) -> str:
    for msg in reversed(messages):
        if msg.message.role == "assistant":
            return msg.message.text
    return ""


def _recover_session_assistant_text(session_path: Path) -> str:
    candidates = _session_assistant_texts(session_path)
    while candidates:
        candidate = _strip_control_reply_text(candidates[-1])
        if candidate:
            return candidate
        candidates.pop()
    return ""


def _session_assistant_texts(session_path: Path) -> list[str]:
    if not session_path.exists():
        return []
    try:
        session = json.loads(session_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    messages = session.get("messages", [])
    assistant_texts: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        assistant_texts.append(content)
    return assistant_texts


def _resolve_effective_final_text(
    messages: list[TraceMessage],
    artifact_summary: dict[str, Any],
) -> str:
    trace_final_text = _extract_trace_final_text(messages)
    cleaned_trace_final = _strip_control_reply_text(trace_final_text)
    if cleaned_trace_final:
        return cleaned_trace_final

    session_file = artifact_summary.get("internal_tools", {}).get("session_file")
    if session_file:
        recovered = _recover_session_assistant_text(Path(session_file))
        if recovered:
            return recovered
    return trace_final_text


def _messages_with_effective_final_text(
    messages: list[TraceMessage],
    final_text: str,
) -> list[TraceMessage]:
    if not final_text:
        return messages

    updated = [message.model_copy(deep=True) for message in messages]
    for idx in range(len(updated) - 1, -1, -1):
        if updated[idx].message.role != "assistant":
            continue
        updated[idx].message.content = [TextBlock(text=final_text)]
        return updated
    return updated


def _snapshot_skill_docs(skills_dir: Path, *, include_contents: bool) -> dict[str, dict[str, Any]]:
    skill_docs: dict[str, dict[str, Any]] = {}
    if not skills_dir.exists():
        return skill_docs

    for skill_md in sorted(skills_dir.rglob("SKILL.md")):
        content = _read_text(skill_md)
        skill_name = skill_md.parent.name
        doc = {
            "path": str(skill_md),
            "sha1": _sha1_text(content),
            "preview": content[:200],
        }
        if include_contents:
            doc["content"] = content
        skill_docs[skill_name] = doc
    return skill_docs


def _snapshot_hermes_tree(
    root_dir: Path,
    *,
    session_path: Path | None = None,
    include_contents: bool = False,
) -> dict[str, Any]:
    memories_dir = root_dir / "memories"
    skills_dir = root_dir / "skills"
    memory_path = memories_dir / "MEMORY.md"
    user_path = memories_dir / "USER.md"
    memory_text = _read_text(memory_path)
    user_text = _read_text(user_path)
    skill_docs = _snapshot_skill_docs(skills_dir, include_contents=include_contents)

    snapshot = {
        "artifacts_dir": str(root_dir),
        "memory_file_exists": memory_path.exists(),
        "user_file_exists": user_path.exists(),
        "memory_chars": len(memory_text),
        "user_chars": len(user_text),
        "memory_entries": _split_memory_entries(memory_text),
        "user_entries": _split_memory_entries(user_text),
        "skill_count": len(skill_docs),
        "skill_names": sorted(skill_docs),
        "skill_docs": skill_docs,
        "internal_tools": _extract_internal_tool_usage(session_path),
    }
    if include_contents:
        snapshot["memory_text"] = memory_text
        snapshot["user_text"] = user_text
    return snapshot


def snapshot_hermes_home(home_dir: Path, *, include_contents: bool = False) -> dict[str, Any]:
    """Snapshot Hermes persistence state directly from the runtime home directory."""

    if not home_dir.exists():
        return _empty_artifact_summary(home_dir)
    return _snapshot_hermes_tree(home_dir, include_contents=include_contents)


def _extract_internal_tool_usage(session_path: Path | None) -> dict[str, Any]:
    if session_path is None or not session_path.exists():
        return {
            "session_file": None,
            "tool_call_counts": {},
            "memory_calls": 0,
            "memory_action_counts": {},
            "memory_write_count": 0,
            "memory_read_count": 0,
            "skill_manage_calls": 0,
            "skill_manage_action_counts": {},
            "skill_create_count": 0,
            "skill_update_count": 0,
            "session_search_calls": 0,
            "skill_view_calls": 0,
            "skills_list_calls": 0,
            "skill_read_count": 0,
            "calls": [],
        }

    data = json.loads(session_path.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    memory_actions: Counter[str] = Counter()
    skill_manage_actions: Counter[str] = Counter()
    calls: list[dict[str, Any]] = []

    for message_index, msg in enumerate(data.get("messages", [])):
        message_timestamp = msg.get("timestamp") or msg.get("created_at")
        for tool_call in msg.get("tool_calls", []) or []:
            fn = tool_call.get("function", {})
            name = fn.get("name") or tool_call.get("name")
            if name:
                counts[name] += 1
            args = _safe_json_loads(fn.get("arguments") or tool_call.get("arguments") or tool_call.get("args"))
            if name == "memory":
                action = str(args.get("action", "")).strip()
                if action:
                    memory_actions[action] += 1
            elif name == "skill_manage":
                action = str(args.get("action", "")).strip()
                if action:
                    skill_manage_actions[action] += 1
            calls.append({
                "name": name or "",
                "args": args,
                "timestamp": message_timestamp,
                "message_index": message_index,
            })

    return {
        "session_file": str(session_path),
        "tool_call_counts": dict(counts),
        "memory_calls": counts.get("memory", 0),
        "memory_action_counts": dict(memory_actions),
        "memory_write_count": sum(count for action, count in memory_actions.items() if action in _MEMORY_WRITE_ACTIONS),
        "memory_read_count": sum(count for action, count in memory_actions.items() if action in _MEMORY_READ_ACTIONS),
        "skill_manage_calls": counts.get("skill_manage", 0),
        "skill_manage_action_counts": dict(skill_manage_actions),
        "skill_create_count": skill_manage_actions.get("create", 0),
        "skill_update_count": (
            skill_manage_actions.get("edit", 0)
            + skill_manage_actions.get("patch", 0)
            + skill_manage_actions.get("write_file", 0)
            + skill_manage_actions.get("remove_file", 0)
        ),
        "session_search_calls": counts.get("session_search", 0),
        "skill_view_calls": counts.get("skill_view", 0),
        "skills_list_calls": counts.get("skills_list", 0),
        "skill_read_count": counts.get("skill_view", 0),
        "calls": calls,
    }


def snapshot_hermes_artifacts(artifacts_dir: Path) -> dict[str, Any]:
    """Summarize copied Hermes persistence artifacts for one episode."""

    session_path = artifacts_dir / "session_current.json"
    if not session_path.exists():
        session_path = artifacts_dir / "session_latest.json"
    return _snapshot_hermes_tree(artifacts_dir, session_path=session_path, include_contents=True)


def _blank_internal_tools(session_file: str | None = None) -> dict[str, Any]:
    return {
        "session_file": session_file,
        "tool_call_counts": {},
        "memory_calls": 0,
        "memory_action_counts": {},
        "memory_write_count": 0,
        "memory_read_count": 0,
        "skill_manage_calls": 0,
        "skill_manage_action_counts": {},
        "skill_create_count": 0,
        "skill_update_count": 0,
        "session_search_calls": 0,
        "skill_view_calls": 0,
        "skills_list_calls": 0,
        "skill_read_count": 0,
        "calls": [],
    }


def _normalize_nanobot_tool_call(name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    path_value = str(args.get("path") or "")
    normalized_path = path_value.replace("\\", "/")
    if name == "read_file":
        if "/skills/" in normalized_path or normalized_path.endswith("/skills"):
            return "skill_view", args
        if "/memory/" in normalized_path:
            return "memory", {"action": "read", **args}
        if "/sessions/" in normalized_path:
            return "session_search", args
    if name == "list_dir" and ("/skills/" in normalized_path or normalized_path.endswith("/skills")):
        return "skills_list", args
    return name, args


def _extract_nanobot_internal_tool_usage(
    sessions_dir: Path,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    session_files = sorted(sessions_dir.glob("*.jsonl")) if sessions_dir.exists() else []
    if not session_files:
        return _blank_internal_tools(None)

    session_file = session_files[-1]
    counts: Counter[str] = Counter()
    memory_actions: Counter[str] = Counter()
    calls: list[dict[str, Any]] = []
    prior_history = False
    if metadata_path is not None and metadata_path.exists():
        try:
            prior_history = bool(json.loads(metadata_path.read_text(encoding="utf-8")).get("prior_history"))
        except Exception:
            prior_history = False

    with open(session_file, encoding="utf-8") as fh:
        for raw_line in fh:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            payload = json.loads(raw_line)
            if payload.get("_type") == "metadata":
                continue
            timestamp = payload.get("timestamp")
            for tool_call in payload.get("tool_calls") or []:
                raw_name = tool_call.get("name") or tool_call.get("function", {}).get("name") or ""
                raw_args = tool_call.get("args")
                if raw_args is None:
                    raw_args = tool_call.get("arguments") or tool_call.get("function", {}).get("arguments")
                args = _safe_json_loads(raw_args)
                name, args = _normalize_nanobot_tool_call(raw_name, args)
                counts[name] += 1
                if name == "memory":
                    memory_actions["read"] += 1
                calls.append({
                    "name": name,
                    "args": args,
                    "timestamp": timestamp,
                })

    if prior_history:
        counts["session_search"] += 1
        calls.insert(0, {"name": "session_search", "args": {"source": "session_history"}, "timestamp": None})

    return {
        "session_file": str(session_file),
        "tool_call_counts": dict(counts),
        "memory_calls": counts.get("memory", 0),
        "memory_action_counts": dict(memory_actions),
        "memory_write_count": 0,
        "memory_read_count": memory_actions.get("read", 0),
        "skill_manage_calls": 0,
        "skill_manage_action_counts": {},
        "skill_create_count": 0,
        "skill_update_count": 0,
        "session_search_calls": counts.get("session_search", 0),
        "skill_view_calls": counts.get("skill_view", 0),
        "skills_list_calls": counts.get("skills_list", 0),
        "skill_read_count": counts.get("skill_view", 0) + counts.get("skills_list", 0),
        "calls": calls,
    }


def snapshot_nanobot_workspace(workspace_dir: Path, *, include_contents: bool = False) -> dict[str, Any]:
    if not workspace_dir.exists():
        return _empty_artifact_summary(workspace_dir)
    memories_dir = workspace_dir / "memory"
    skills_dir = workspace_dir / "skills"
    memory_path = memories_dir / "MEMORY.md"
    user_path = memories_dir / "USER.md"
    memory_text = _read_text(memory_path)
    user_text = _read_text(user_path)
    skill_docs = _snapshot_skill_docs(skills_dir, include_contents=include_contents)
    internal_tools = _extract_nanobot_internal_tool_usage(
        workspace_dir / "sessions",
        workspace_dir / "nanobot_metadata.json",
    )
    snapshot = {
        "artifacts_dir": str(workspace_dir),
        "memory_file_exists": memory_path.exists(),
        "user_file_exists": user_path.exists(),
        "memory_chars": len(memory_text),
        "user_chars": len(user_text),
        "memory_entries": _split_memory_entries(memory_text),
        "user_entries": _split_memory_entries(user_text),
        "skill_count": len(skill_docs),
        "skill_names": sorted(skill_docs),
        "skill_docs": skill_docs,
        "internal_tools": internal_tools,
    }
    if include_contents:
        snapshot["memory_text"] = memory_text
        snapshot["user_text"] = user_text
    return snapshot


def snapshot_nanobot_artifacts(artifacts_dir: Path) -> dict[str, Any]:
    if not artifacts_dir.exists():
        return _empty_artifact_summary(artifacts_dir)
    session_metadata = artifacts_dir / "nanobot_metadata.json"
    snapshot = snapshot_nanobot_workspace(artifacts_dir, include_contents=True)
    snapshot["internal_tools"] = _extract_nanobot_internal_tool_usage(
        artifacts_dir / "sessions",
        session_metadata,
    )
    return snapshot


def _load_zeroclaw_memory_entries(memory_store_path: Path) -> tuple[list[str], str]:
    if not memory_store_path.exists():
        return [], ""
    try:
        payload = json.loads(memory_store_path.read_text(encoding="utf-8")) or {}
    except Exception:
        payload = {}
    entries = [str(value) for _, value in sorted(payload.items()) if value is not None]
    return entries, "\n§\n".join(entries)


def _normalize_zeroclaw_tool_call(name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if name == "memory_recall":
        return "memory", {"action": "search", **args}
    if name == "memory_store":
        return "memory", {"action": "add", **args}
    path_value = str(args.get("path") or "")
    normalized_path = path_value.replace("\\", "/")
    if name == "file_read":
        if "/skills/" in normalized_path or normalized_path.endswith("/skills"):
            return "skill_view", args
        if normalized_path.endswith("memory_store.json"):
            return "memory", {"action": "read", **args}
    return name, args


def _extract_zeroclaw_internal_tool_usage(transcript_path: Path | None) -> dict[str, Any]:
    if transcript_path is None or not transcript_path.exists():
        return _blank_internal_tools(None)
    payload = json.loads(transcript_path.read_text(encoding="utf-8")) or {}
    counts: Counter[str] = Counter()
    memory_actions: Counter[str] = Counter()
    calls: list[dict[str, Any]] = []
    for message_index, message in enumerate(payload.get("messages", [])):
        for tool_call in message.get("tool_calls") or []:
            raw_name = tool_call.get("name") or ""
            args = tool_call.get("args") or {}
            name, args = _normalize_zeroclaw_tool_call(raw_name, args)
            counts[name] += 1
            if name == "memory":
                action = str(args.get("action") or "").strip() or "read"
                memory_actions[action] += 1
            calls.append({
                "name": name,
                "args": args,
                "timestamp": f"m{message_index}",
            })
    return {
        "session_file": str(transcript_path),
        "tool_call_counts": dict(counts),
        "memory_calls": counts.get("memory", 0),
        "memory_action_counts": dict(memory_actions),
        "memory_write_count": sum(count for action, count in memory_actions.items() if action in _MEMORY_WRITE_ACTIONS),
        "memory_read_count": sum(count for action, count in memory_actions.items() if action in _MEMORY_READ_ACTIONS),
        "skill_manage_calls": 0,
        "skill_manage_action_counts": {},
        "skill_create_count": 0,
        "skill_update_count": 0,
        "session_search_calls": counts.get("session_search", 0),
        "skill_view_calls": counts.get("skill_view", 0),
        "skills_list_calls": counts.get("skills_list", 0),
        "skill_read_count": counts.get("skill_view", 0) + counts.get("skills_list", 0),
        "calls": calls,
    }


def snapshot_zeroclaw_state(
    *,
    home_dir: Path,
    workspace_dir: Path,
    include_contents: bool = False,
) -> dict[str, Any]:
    memory_store_path = home_dir / ".zeroclaw" / "memory_store.json"
    memory_entries, memory_text = _load_zeroclaw_memory_entries(memory_store_path)
    skill_docs = _snapshot_skill_docs(workspace_dir / "skills", include_contents=include_contents)
    snapshot = {
        "artifacts_dir": str(workspace_dir),
        "memory_file_exists": memory_store_path.exists(),
        "user_file_exists": False,
        "memory_chars": len(memory_text),
        "user_chars": 0,
        "memory_entries": memory_entries,
        "user_entries": [],
        "skill_count": len(skill_docs),
        "skill_names": sorted(skill_docs),
        "skill_docs": skill_docs,
        "internal_tools": _blank_internal_tools(None),
    }
    if include_contents:
        snapshot["memory_text"] = memory_text
        snapshot["user_text"] = ""
    return snapshot


def snapshot_zeroclaw_artifacts(artifacts_dir: Path) -> dict[str, Any]:
    memory_store_path = artifacts_dir / ".zeroclaw" / "memory_store.json"
    memory_entries, memory_text = _load_zeroclaw_memory_entries(memory_store_path)
    skill_docs = _snapshot_skill_docs(artifacts_dir / "skills", include_contents=True)
    return {
        "artifacts_dir": str(artifacts_dir),
        "memory_file_exists": memory_store_path.exists(),
        "user_file_exists": False,
        "memory_chars": len(memory_text),
        "user_chars": 0,
        "memory_entries": memory_entries,
        "user_entries": [],
        "skill_count": len(skill_docs),
        "skill_names": sorted(skill_docs),
        "skill_docs": skill_docs,
        "internal_tools": _extract_zeroclaw_internal_tool_usage(
            artifacts_dir / "zeroclaw_transcript.json"
        ),
        "memory_text": memory_text,
        "user_text": "",
    }


def _rule_keyword_hits(snapshot: dict[str, Any], keywords: list[str]) -> dict[str, Any]:
    if not keywords:
        return {
            "required_keywords": [],
            "hit_keywords": [],
            "hit_count": 0,
            "hit_rate": 0.0,
        }

    combined: list[str] = []
    combined.extend(snapshot.get("memory_entries", []))
    combined.extend(snapshot.get("user_entries", []))
    for doc in snapshot.get("skill_docs", {}).values():
        content = doc.get("content") or doc.get("preview", "")
        if content:
            combined.append(content)

    combined_text = "\n".join(combined).lower()
    hit_keywords = [keyword for keyword in keywords if keyword and keyword.lower() in combined_text]
    return {
        "required_keywords": keywords,
        "hit_keywords": hit_keywords,
        "hit_count": len(hit_keywords),
        "hit_rate": round(len(hit_keywords) / len(keywords), 6),
    }


def diff_artifact_snapshots(
    *,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    rule_keywords: list[str] | None = None,
    forbidden_rule_keywords: list[str] | None = None,
) -> dict[str, Any]:
    before = before or _empty_artifact_summary(Path("."))
    after = after or _empty_artifact_summary(Path("."))

    before_rules = before.get("memory_entries", []) + before.get("user_entries", [])
    after_rules = after.get("memory_entries", []) + after.get("user_entries", [])
    updated_rules, remaining_before, remaining_after = _pair_updated_entries(before_rules, after_rules)

    before_docs = before.get("skill_docs", {})
    after_docs = after.get("skill_docs", {})
    before_skill_names = set(before_docs)
    after_skill_names = set(after_docs)
    changed_skill_names = sorted(
        name
        for name in before_skill_names & after_skill_names
        if before_docs[name].get("sha1") != after_docs[name].get("sha1")
    )

    result = {
        "added_rules": remaining_after,
        "updated_rules": updated_rules,
        "deleted_rules": remaining_before,
        "changed_skill_names": changed_skill_names,
        "added_skill_names": sorted(after_skill_names - before_skill_names),
        "deleted_skill_names": sorted(before_skill_names - after_skill_names),
        "memory_entry_delta": len(after_rules) - len(before_rules),
        "skill_count_delta": after.get("skill_count", 0) - before.get("skill_count", 0),
        "rule_keyword_hits": _rule_keyword_hits(after, rule_keywords or []),
    }
    if forbidden_rule_keywords:
        result["forbidden_rule_keyword_hits"] = _rule_keyword_hits(after, forbidden_rule_keywords)
    return result


def _first_write_dispatch(dispatches) -> dict[str, Any] | None:
    for dispatch in dispatches:
        if any(pattern in dispatch.endpoint_url for pattern in _WRITE_ENDPOINT_PATTERNS):
            return {
                "tool_name": dispatch.tool_name,
                "endpoint_url": dispatch.endpoint_url,
                "timestamp": dispatch.timestamp,
            }
    return None


def _first_relevant_internal_call(
    internal_tools: dict[str, Any],
    *,
    expected_signal: str,
) -> dict[str, Any] | None:
    for call in internal_tools.get("calls", []):
        name = call.get("name")
        args = call.get("args", {})
        if expected_signal == "session_search" and name == "session_search":
            return call
        if expected_signal == "skill" and name in {"skills_list", "skill_view"}:
            return call
        if expected_signal == "memory" and name == "memory" and args.get("action") in _MEMORY_READ_ACTIONS:
            return call
        if expected_signal == "mixed":
            if name == "session_search" or name in {"skills_list", "skill_view"}:
                return call
            if name == "memory" and args.get("action") in _MEMORY_READ_ACTIONS:
                return call
    return None


def compute_retrieval_signals(
    *,
    dispatches,
    artifact_before: dict[str, Any] | None,
    internal_tools: dict[str, Any],
    expected_signal: str,
) -> dict[str, Any]:
    artifact_before = artifact_before or {}
    has_memory_before = bool(
        artifact_before.get("memory_file_exists") or artifact_before.get("user_file_exists")
    )
    first_write = _first_write_dispatch(dispatches)
    first_internal = _first_relevant_internal_call(internal_tools, expected_signal=expected_signal)

    first_write_at = _parse_timestamp(first_write["timestamp"]) if first_write else None
    first_internal_at = _parse_timestamp(first_internal.get("timestamp")) if first_internal else None

    memory_read_count = internal_tools.get("memory_read_count", 0)
    memory_injection_count = 1 if has_memory_before and expected_signal in {"memory", "mixed"} else 0
    skill_read_count = internal_tools.get("skill_read_count", 0)
    session_search_count = internal_tools.get("session_search_calls", 0)
    memory_injection_used = memory_injection_count > 0
    explicit_signal_used = memory_read_count > 0 or skill_read_count > 0 or session_search_count > 0
    explicit_retrieval_count = memory_read_count + skill_read_count + session_search_count
    explicit_retrieval_before_first_update = (
        bool(first_internal_at and first_write_at and first_internal_at <= first_write_at)
        if first_write_at and first_internal_at
        else bool(first_internal)
    )

    if expected_signal == "memory":
        used_expected_signal = memory_read_count > 0 or memory_injection_used
        retrieval_before_first_update = explicit_retrieval_before_first_update or memory_injection_used
    elif expected_signal == "skill":
        used_expected_signal = skill_read_count > 0
        retrieval_before_first_update = explicit_retrieval_before_first_update
    elif expected_signal == "session_search":
        used_expected_signal = session_search_count > 0
        retrieval_before_first_update = explicit_retrieval_before_first_update
    else:
        used_expected_signal = explicit_signal_used or memory_injection_used
        retrieval_before_first_update = explicit_retrieval_before_first_update or memory_injection_used

    retrieval_signal_count = memory_injection_count + memory_read_count + skill_read_count + session_search_count

    return {
        "expected_signal": expected_signal,
        "memory_read_count": memory_read_count,
        "memory_injection_count": memory_injection_count,
        "skill_read_count": skill_read_count,
        "session_search_count": session_search_count,
        "retrieval_signal_count": retrieval_signal_count,
        "explicit_retrieval_count": explicit_retrieval_count,
        "explicit_signal_used": explicit_signal_used,
        "used_expected_signal": used_expected_signal,
        "retrieval_before_first_update": retrieval_before_first_update,
        "explicit_retrieval_before_first_update": explicit_retrieval_before_first_update,
        "retrieval_before_final_response": used_expected_signal,
        "first_retrieval_at": first_internal.get("timestamp") if first_internal else None,
        "first_write_at": first_write["timestamp"] if first_write else None,
        "first_write_endpoint": first_write["endpoint_url"] if first_write else None,
    }


def _trace_totals(end) -> dict[str, int | float | bool | None]:
    if end is None:
        return {
            "model_input_tokens": 0,
            "model_output_tokens": 0,
            "total_tokens": 0,
            "cache_read_tokens": None,
            "cache_write_tokens": None,
            "reasoning_tokens": None,
            "model_request_count": None,
            "model_retry_count": None,
            "model_usage_complete": False,
            "model_time_s": 0.0,
            "tool_time_s": 0.0,
            "other_time_s": 0.0,
            "wall_time_s": 0.0,
        }

    model_input_tokens = getattr(end, "model_input_tokens", getattr(end, "input_tokens", 0))
    model_output_tokens = getattr(end, "model_output_tokens", getattr(end, "output_tokens", 0))
    total_tokens = getattr(end, "total_tokens", model_input_tokens + model_output_tokens)
    model_time_s = getattr(end, "model_time_s", 0.0)
    tool_time_s = getattr(end, "tool_time_s", 0.0)
    other_time_s = getattr(end, "other_time_s", 0.0)
    wall_time_s = getattr(end, "wall_time_s", 0.0)
    cache_read_tokens = getattr(end, "cache_read_tokens", None)
    cache_write_tokens = getattr(end, "cache_write_tokens", None)
    reasoning_tokens = getattr(end, "reasoning_tokens", None)
    model_request_count = getattr(end, "model_request_count", None)
    model_retry_count = getattr(end, "model_retry_count", None)
    model_usage_complete = getattr(end, "model_usage_complete", False)

    if not total_tokens:
        total_tokens = model_input_tokens + model_output_tokens
    if not other_time_s and wall_time_s:
        other_time_s = max(0.0, wall_time_s - model_time_s - tool_time_s)

    return {
        "model_input_tokens": model_input_tokens,
        "model_output_tokens": model_output_tokens,
        "total_tokens": total_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "reasoning_tokens": reasoning_tokens,
        "model_request_count": model_request_count,
        "model_retry_count": model_retry_count,
        "model_usage_complete": model_usage_complete,
        "model_time_s": wall_time_s if not model_time_s and not tool_time_s else model_time_s,
        "tool_time_s": tool_time_s,
        "other_time_s": other_time_s,
        "wall_time_s": wall_time_s,
    }


def _detect_infra_block(
    *,
    final_text: str,
    expected_signal: str,
    internal_tools: dict[str, Any],
) -> tuple[bool, str]:
    if expected_signal != "session_search":
        return False, ""
    if internal_tools.get("session_search_calls", 0) > 0:
        return False, ""

    for pattern in _INFRA_BLOCK_PATTERNS:
        if pattern in final_text:
            return True, pattern
    return False, ""


def grade_episode(
    *,
    trace_path: Path,
    task,
    tasks_dir: Path,
    task_dir: Path,
    grader,
    judge,
    env_snapshot: dict | None,
    artifact_before: dict[str, Any] | None,
    artifact_summary: dict[str, Any],
    expected_persistence_signal: str = "",
) -> dict[str, Any]:
    """Compute episode-level grading and efficiency metrics."""

    from ..cli import _grade_with_optional_params
    from ..graders.self_evolve_helpers import (
        compute_notes_memory_hard_pass,
        compute_notes_memory_task_components,
        compute_self_evolve_mechanism_scores,
    )

    start, messages, dispatches, media_events, end, audit_data = load_trace(trace_path)
    expectations = _load_expectations(task)
    artifact_contract = expectations.get("artifact_contract", {})
    artifact_diff = diff_artifact_snapshots(
        before=artifact_before,
        after=artifact_summary,
        rule_keywords=artifact_contract.get("require_rule_keywords", []),
        forbidden_rule_keywords=artifact_contract.get("forbidden_rule_keywords", []),
    )
    retrieval_signals = compute_retrieval_signals(
        dispatches=dispatches,
        artifact_before=artifact_before,
        internal_tools=artifact_summary.get("internal_tools", {}),
        expected_signal=expected_persistence_signal or expectations.get("expected_mechanism", ""),
    )
    mechanism_scores = compute_self_evolve_mechanism_scores(
        expectations=expectations,
        artifact_before=artifact_before or {},
        artifact_after=artifact_summary,
        artifact_diff=artifact_diff,
        retrieval_signals=retrieval_signals,
    )

    grader_env_snapshot = dict(env_snapshot or {})
    grader_env_snapshot["_self_evolve_internal_tools"] = artifact_summary.get("internal_tools", {})
    grader_env_snapshot["_self_evolve_skill_count"] = artifact_summary.get("skill_count", 0)
    grader_env_snapshot["_self_evolve_has_memory_artifact"] = bool(
        artifact_summary.get("memory_file_exists") or artifact_summary.get("user_file_exists")
    )
    grader_env_snapshot["_self_evolve_mechanism_metrics"] = mechanism_scores
    grader_env_snapshot["_self_evolve_memory_entries"] = artifact_summary.get("memory_entries", [])
    grader_env_snapshot["_self_evolve_skill_docs"] = {
        name: doc.get("preview", "") for name, doc in artifact_summary.get("skill_docs", {}).items()
    }

    final_text = _resolve_effective_final_text(messages, artifact_summary)
    if expectations.get("mode") == "notes_memory":
        session_file = artifact_summary.get("internal_tools", {}).get("session_file")
        if session_file:
            notes_audit = audit_data.get("notes", {}) if audit_data else {}
            current_components = compute_notes_memory_task_components(
                expectations=expectations,
                final_text=final_text,
                notes=notes_audit,
            )
            best_text = final_text
            best_score = current_components.get("content_correctness", 0.0)
            for candidate in _session_assistant_texts(Path(session_file)):
                candidate_text = _strip_control_reply_text(candidate)
                if not candidate_text:
                    continue
                candidate_components = compute_notes_memory_task_components(
                    expectations=expectations,
                    final_text=candidate_text,
                    notes=notes_audit,
                )
                candidate_score = candidate_components.get("content_correctness", 0.0)
                if candidate_score > best_score or (
                    candidate_score == best_score and len(candidate_text) > len(best_text)
                ):
                    best_text = candidate_text
                    best_score = candidate_score
            final_text = best_text
    messages_for_grading = _messages_with_effective_final_text(messages, final_text)

    scores = _grade_with_optional_params(
        grader,
        messages_for_grading,
        dispatches,
        task,
        audit_data=audit_data,
        judge=judge,
        media_events=media_events,
        env_snapshot=grader_env_snapshot,
    )
    task_score = compute_task_score(scores)
    totals = _trace_totals(end)
    internal = artifact_summary.get("internal_tools", {})
    task_components: dict[str, float] = {}
    hard_pass = is_pass(task_score)
    if expectations.get("mode") == "notes_memory":
        task_components = compute_notes_memory_task_components(
            expectations=expectations,
            final_text=final_text,
            notes=audit_data.get("notes", {}) if audit_data else {},
        )
        hard_pass = compute_notes_memory_hard_pass(
            task_components,
            expectations=expectations,
            artifact_summary=artifact_summary,
            artifact_diff=artifact_diff,
            retrieval_signals=retrieval_signals,
        )
    infra_blocked, infra_block_reason = _detect_infra_block(
        final_text=final_text,
        expected_signal=expected_persistence_signal or expectations.get("expected_mechanism", ""),
        internal_tools=internal,
    )

    return {
        "trace": str(trace_path),
        "trace_id": start.trace_id,
        "task_id": task.task_id,
        "task_name": task.task_name,
        "final_response_text": final_text,
        "scores": scores.model_dump(mode="json"),
        "judge_score": getattr(scores, "communication", 0.0),
        "task_score": task_score,
        "task_components": task_components,
        "hard_pass": hard_pass,
        "passed": hard_pass,
        "total_turns": end.total_turns if end else 0,
        "tool_dispatch_count": len(dispatches),
        "token_usage": {
            "input_tokens": totals["model_input_tokens"],
            "output_tokens": totals["model_output_tokens"],
            "total_tokens": totals["total_tokens"],
            "cache_read_tokens": totals["cache_read_tokens"],
            "cache_write_tokens": totals["cache_write_tokens"],
            "reasoning_tokens": totals["reasoning_tokens"],
            "model_request_count": totals["model_request_count"],
            "model_retry_count": totals["model_retry_count"],
            "model_usage_complete": totals["model_usage_complete"],
        },
        "timing": {
            "wall_time_s": totals["wall_time_s"],
            "model_time_s": totals["model_time_s"],
            "tool_time_s": totals["tool_time_s"],
            "other_time_s": totals["other_time_s"],
        },
        "artifacts": artifact_summary,
        "artifact_diff": artifact_diff,
        "retrieval_signals": retrieval_signals,
        "mechanism_scores": mechanism_scores,
        "internal_tools": internal,
        "env_snapshot_present": env_snapshot is not None,
        "infra_blocked": infra_blocked,
        "infra_block_reason": infra_block_reason,
    }


def make_reflection_task(
    *,
    task_id: str,
    task_name: str,
    prompt_text: str,
) -> TaskDefinition:
    """Build an in-memory reflection task for Hermes post-attempt review."""

    return TaskDefinition(
        task_id=task_id,
        task_name=task_name,
        category="self_evolve_reflection",
        difficulty="medium",
        prompt=Prompt(text=prompt_text, language="en"),
        tools=[],
        tool_endpoints=[],
        services=[],
        judge_rubric="",
        reference_solution="",
    )


def _load_expectations(task: TaskDefinition) -> dict[str, Any]:
    task_path = Path(task.task_file).resolve() if task.task_file else None
    if task_path is None:
        return {}
    expectation_path = task_path.parent / "expectations.json"
    if not expectation_path.exists():
        return {}
    return json.loads(expectation_path.read_text(encoding="utf-8"))


def build_reflection_prompt(
    *,
    trace_path: Path,
    task: TaskDefinition,
    attempt_result: dict[str, Any],
) -> str:
    """Create a standardized feedback prompt Hermes can persist from."""

    _, messages, _, _, _, audit_data = load_trace(trace_path)
    expectations = _load_expectations(task)
    mode = expectations.get("mode", "")
    expected_mechanism = expectations.get("expected_mechanism", "")
    final_text = attempt_result.get("final_response_text", "")
    score = attempt_result["task_score"]
    result_text = "PASS" if attempt_result["passed"] else "FAIL"

    mistakes: list[str] = []
    correct_rule = expectations.get("reflection_rules", [])

    if mode == "helpdesk_kb":
        helpdesk = audit_data.get("helpdesk", {}) if audit_data else {}
        updates = helpdesk.get("updated_tickets", [])
        update_map = {item["ticket_id"]: item for item in updates}
        required_ids = expectations.get("required_ticket_ids", [])
        excluded_ids = set(expectations.get("excluded_ticket_ids", []))
        expected_priority = expectations.get("expected_priority")
        expected_category = expectations.get("expected_category")
        expected_reason_code = expectations.get("expected_reason_code")
        exact_tags = expectations.get("expected_exact_tags")

        for ticket_id in required_ids:
            item = update_map.get(ticket_id)
            if item is None:
                mistakes.append(f"Required ticket {ticket_id} was not updated.")
                continue
            if expected_priority and item.get("priority") != expected_priority:
                mistakes.append(
                    f"{ticket_id} has the wrong priority: got {item.get('priority')!r}, expected {expected_priority!r}."
                )
            if expected_category and item.get("category") != expected_category:
                mistakes.append(
                    f"{ticket_id} has the wrong category: got {item.get('category')!r}, expected {expected_category!r}."
                )
            if expected_reason_code and item.get("reason_code") != expected_reason_code:
                mistakes.append(
                    f"{ticket_id} has the wrong reason_code: got {item.get('reason_code')!r}, expected {expected_reason_code!r}."
                )
            if exact_tags and item.get("tags", []) != exact_tags:
                mistakes.append(
                    f"{ticket_id} has the wrong tags: got {item.get('tags', [])!r}, expected {exact_tags!r}."
                )

        for ticket_id in sorted(update_map):
            if ticket_id in excluded_ids:
                mistakes.append(f"Ticket {ticket_id} was updated even though it should have been left untouched.")

        required_keywords = expectations.get("required_summary_keywords", [])
        missing_keywords = [keyword for keyword in required_keywords if keyword not in final_text]
        if missing_keywords:
            mistakes.append(f"The final report is missing required information: {', '.join(missing_keywords)}.")

    if not mistakes:
        mistakes.append("This attempt was broadly correct. No execution mistake requires correction.")

    if not correct_rule:
        correct_rule = [
            "Prefer a reusable rule instead of memorizing the answer to this specific task.",
            "If the lesson is useful across sessions, update memory or create or update a skill.",
        ]

    attempt_preview = final_text.strip() or "(no final text output)"
    if len(attempt_preview) > 1000:
        attempt_preview = attempt_preview[:1000] + "\n...[truncated]..."

    mistake_block = "\n".join(f"- {line}" for line in mistakes)
    rule_block = "\n".join(f"- {line}" for line in correct_rule)

    persistence_instruction = (
        "- Only write an invariant procedure into a skill if it holds across cases. If it still depends on a specific product name, system name, ticket id, department name, training-case wording, or fixed output example, explicitly say Nothing to save.\n"
        "- Prefer putting reusable rules into a skill; do not store a procedure rule only in memory.\n"
        "- The skill name, description, and body must all stay generic. They must not include entities from the training case, task ids, ticket ids, or hints like from this session.\n"
        "- If a similar skill already exists, patch the existing skill instead of creating a new variant that fits only one case.\n"
    )
    if expected_mechanism == "memory":
        persistence_instruction = (
            "- Prefer writing durable preferences or stable rules into memory; do not create a skill.\n"
            "- If this was only a one-off answer, explicitly say Nothing to save.\n"
        )
    elif expected_mechanism == "session_search":
        persistence_instruction = (
            "- These tasks mainly rely on later recall through session_search. Do not copy the exception list into memory or a skill.\n"
            "- Unless you discovered a general safety rule that applies across tasks, say Nothing to save.\n"
        )
    elif expected_mechanism == "mixed":
        persistence_instruction = (
            "- Persist a correlation or procedure only if it is invariant across cases. If it still depends on a specific system name, entity name, training-task why text, or fixed tag example, explicitly say Nothing to save.\n"
            "- If you save a procedural rule, prefer a skill. Use memory only for stable preferences or facts.\n"
            "- The skill name, description, and body must remain generic and must not directly repeat training-case entities such as LedgerLane, ParcelMesh, or RolloutHub.\n"
            "- If a similar skill already exists, patch that skill instead of piling on more case-specific variants.\n"
        )

    return (
        f"You just finished a self-evolve training round. Do not redo the task. Only reflect and decide what, if anything, should be persisted.\n\n"
        f"Task:\n{task.prompt.text.strip()}\n\n"
        f"Your attempt summary:\n{attempt_preview}\n\n"
        f"Result: {result_text}\n"
        f"Score: {score:.3f}\n\n"
        f"Mistakes:\n{mistake_block}\n\n"
        f"Correct rule:\n{rule_block}\n\n"
        "Instruction:\n"
        "- First, briefly summarize the part of this experience that is genuinely reusable.\n"
        f"{persistence_instruction}"
        "- Do not save one-off task outcomes or just record the final answer to this task.\n"
        "- If nothing is worth saving, explicitly say Nothing to save.\n"
    )


def summarize_reflection_episode(*, trace_path: Path, artifact_summary: dict[str, Any], task_id: str, label: str) -> dict[str, Any]:
    """Collect non-graded telemetry for a reflection episode."""

    start, messages, dispatches, _, end, _ = load_trace(trace_path)
    totals = _trace_totals(end)
    final_text = ""
    if messages and messages[-1].message.content:
        final_text = messages[-1].message.content[0].text

    return {
        "trace": str(trace_path),
        "trace_id": start.trace_id,
        "task_id": task_id,
        "task_name": label,
        "final_response_text": final_text,
        "episode_kind": "reflection",
        "bucket": "reflection",
        "task_score": 0.0,
        "passed": False,
        "total_turns": end.total_turns if end else 0,
        "tool_dispatch_count": len(dispatches),
        "token_usage": {
            "input_tokens": totals["model_input_tokens"],
            "output_tokens": totals["model_output_tokens"],
            "total_tokens": totals["total_tokens"],
            "cache_read_tokens": totals["cache_read_tokens"],
            "cache_write_tokens": totals["cache_write_tokens"],
            "reasoning_tokens": totals["reasoning_tokens"],
            "model_request_count": totals["model_request_count"],
            "model_retry_count": totals["model_retry_count"],
            "model_usage_complete": totals["model_usage_complete"],
        },
        "timing": {
            "wall_time_s": totals["wall_time_s"],
            "model_time_s": totals["model_time_s"],
            "tool_time_s": totals["tool_time_s"],
            "other_time_s": totals["other_time_s"],
        },
        "artifacts": artifact_summary,
        "artifact_diff": {},
        "retrieval_signals": {},
        "mechanism_scores": {},
        "internal_tools": artifact_summary.get("internal_tools", {}),
        "env_snapshot_present": False,
    }


def _empty_artifact_summary(artifacts_dir: Path) -> dict[str, Any]:
    return {
        "artifacts_dir": str(artifacts_dir),
        "memory_file_exists": False,
        "user_file_exists": False,
        "memory_chars": 0,
        "user_chars": 0,
        "memory_entries": [],
        "user_entries": [],
        "skill_count": 0,
        "skill_names": [],
        "skill_docs": {},
        "internal_tools": {
            "session_file": None,
            "tool_call_counts": {},
            "memory_calls": 0,
            "memory_action_counts": {},
            "memory_write_count": 0,
            "memory_read_count": 0,
            "skill_manage_calls": 0,
            "skill_manage_action_counts": {},
            "skill_create_count": 0,
            "skill_update_count": 0,
            "session_search_calls": 0,
            "skill_view_calls": 0,
            "skills_list_calls": 0,
            "skill_read_count": 0,
            "calls": [],
        },
    }


def choose_calibration_candidate(
    *,
    target_pass_count: int,
    score_min: float,
    score_max: float,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pick the candidate whose blank-memory baseline is closest to the target."""

    mid_score = (score_min + score_max) / 2.0

    def _sort_key(item: dict[str, Any]) -> tuple[float, float, float]:
        pass_count = item["baseline_summary"]["pass_count"]
        avg_score = item["baseline_summary"]["avg_task_score"]
        band_penalty = 0.0 if score_min <= avg_score <= score_max else min(
            abs(avg_score - score_min),
            abs(avg_score - score_max),
        )
        return (
            abs(pass_count - target_pass_count),
            band_penalty,
            abs(avg_score - mid_score),
        )

    exact = [item for item in candidates if item["baseline_summary"]["pass_count"] == target_pass_count]
    pool = exact or candidates
    return sorted(pool, key=_sort_key)[0]


def _task_component(episode: dict[str, Any], name: str) -> float:
    task_components = episode.get("task_components", {})
    if name in task_components:
        return float(task_components.get(name, 0.0))
    return float(episode.get("task_score", 0.0))


def _memory_write_episode(episode: dict[str, Any]) -> bool:
    internal_tools = episode.get("internal_tools", {})
    artifact_diff = episode.get("artifact_diff", {})
    return bool(
        internal_tools.get("memory_write_count", 0) > 0
        or internal_tools.get("memory_calls", 0) > 0
        or artifact_diff.get("memory_entry_delta", 0) > 0
    )


def _retention_horizon(near_score: float, far_score: float) -> float:
    if near_score <= 0:
        return 0.0
    return round(max(0.0, min(1.0, far_score / near_score)), 6)


def _episode_memory_pollution_rate(episode: dict[str, Any]) -> float:
    artifact_diff = episode.get("artifact_diff", {})
    keyword_hits = artifact_diff.get("rule_keyword_hits", {})
    rule_keywords = [keyword for keyword in keyword_hits.get("required_keywords", []) if keyword]
    candidate_entries = list(artifact_diff.get("added_rules", []))
    candidate_entries.extend(
        item.get("after", "")
        for item in artifact_diff.get("updated_rules", [])
        if isinstance(item, dict)
    )
    candidate_entries = [entry for entry in candidate_entries if entry]

    if candidate_entries and rule_keywords:
        relevant_entries = sum(
            1
            for entry in candidate_entries
            if any(keyword in entry for keyword in rule_keywords)
        )
        return round(max(0.0, 1.0 - (relevant_entries / len(candidate_entries))), 6)

    return round(max(0.0, 1.0 - episode.get("mechanism_scores", {}).get("artifact_quality_score", 0.0)), 6)


def _aggregate_bucket(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    blocked_episodes = [ep for ep in episodes if ep.get("infra_blocked")]
    scored_episodes = [ep for ep in episodes if not ep.get("infra_blocked")]

    if not scored_episodes:
        return {
            "episode_count": 0,
            "blocked_episode_count": len(blocked_episodes),
            "avg_task_score": 0.0,
            "avg_grounding_score": 0.0,
            "avg_content_correctness": 0.0,
            "avg_share_correctness": 0.0,
            "pass_rate": 0.0,
            "avg_tool_dispatch_count": 0.0,
            "avg_total_tokens": 0.0,
            "avg_wall_time_s": 0.0,
            "episodes_with_memory": 0,
            "episodes_with_skills": 0,
            "memory_calls": 0,
            "skill_manage_calls": 0,
            "session_search_calls": 0,
            "memory_write_count": 0,
            "memory_read_count": 0,
            "memory_injection_count": 0,
            "skill_create_count": 0,
            "skill_update_count": 0,
            "artifact_reuse_rate": 0.0,
            "retrieval_before_success_rate": 0.0,
            "avg_artifact_quality_score": 0.0,
        }

    successful = [ep for ep in scored_episodes if ep["passed"]]
    return {
        "episode_count": len(scored_episodes),
        "blocked_episode_count": len(blocked_episodes),
        "avg_task_score": mean(ep["task_score"] for ep in scored_episodes),
        "avg_grounding_score": round(mean(_task_component(ep, "grounding") for ep in scored_episodes), 6),
        "avg_content_correctness": round(
            mean(_task_component(ep, "content_correctness") for ep in scored_episodes),
            6,
        ),
        "avg_share_correctness": round(mean(_task_component(ep, "share_correctness") for ep in scored_episodes), 6),
        "pass_rate": mean(1.0 if ep["passed"] else 0.0 for ep in scored_episodes),
        "avg_tool_dispatch_count": mean(ep["tool_dispatch_count"] for ep in scored_episodes),
        "avg_total_tokens": mean(ep["token_usage"]["total_tokens"] for ep in scored_episodes),
        "avg_wall_time_s": mean(ep["timing"]["wall_time_s"] for ep in scored_episodes),
        "episodes_with_memory": sum(
            1 for ep in scored_episodes if ep["artifacts"]["memory_file_exists"] or ep["artifacts"]["user_file_exists"]
        ),
        "episodes_with_skills": sum(1 for ep in scored_episodes if ep["artifacts"]["skill_count"] > 0),
        "memory_calls": sum(ep["internal_tools"].get("memory_calls", 0) for ep in scored_episodes),
        "skill_manage_calls": sum(ep["internal_tools"].get("skill_manage_calls", 0) for ep in scored_episodes),
        "session_search_calls": sum(ep["internal_tools"].get("session_search_calls", 0) for ep in scored_episodes),
        "memory_write_count": sum(ep["internal_tools"].get("memory_write_count", 0) for ep in scored_episodes),
        "memory_read_count": sum(ep.get("retrieval_signals", {}).get("memory_read_count", 0) for ep in scored_episodes),
        "memory_injection_count": sum(ep.get("retrieval_signals", {}).get("memory_injection_count", 0) for ep in scored_episodes),
        "skill_create_count": sum(ep["internal_tools"].get("skill_create_count", 0) for ep in scored_episodes),
        "skill_update_count": sum(ep["internal_tools"].get("skill_update_count", 0) for ep in scored_episodes),
        "artifact_reuse_rate": round(
            mean(
                1.0 if ep.get("retrieval_signals", {}).get("used_expected_signal") else 0.0
                for ep in scored_episodes
            ),
            6,
        ),
        "retrieval_before_success_rate": round(
            mean(
                1.0
                if ep.get("retrieval_signals", {}).get("retrieval_before_first_update")
                else 0.0
                for ep in successful
            ),
            6,
        ) if successful else 0.0,
        "avg_artifact_quality_score": round(
            mean(ep.get("mechanism_scores", {}).get("artifact_quality_score", 0.0) for ep in scored_episodes),
            6,
        ),
    }


def _summarize_family(family_id: str, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    def _stage_episodes(stage: str) -> list[dict[str, Any]]:
        return [ep for ep in episodes if ep.get("stage") == stage and not ep.get("infra_blocked")]

    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        by_bucket[ep["bucket"]].append(ep)

    bucket_summary = {bucket: _aggregate_bucket(items) for bucket, items in sorted(by_bucket.items())}
    baseline = bucket_summary.get("baseline", _aggregate_bucket([]))
    evaluation = bucket_summary.get("evaluation", _aggregate_bucket([]))
    learn = bucket_summary.get("learn", _aggregate_bucket([]))
    evaluation_episodes = [ep for ep in episodes if ep.get("bucket") == "evaluation" and not ep.get("infra_blocked")]
    update_episodes = _stage_episodes("learn_b") + _stage_episodes("eval_near") + _stage_episodes("eval_far")
    if not update_episodes:
        update_episodes = [ep for ep in episodes if ep.get("bucket") in {"learn", "evaluation"} and not ep.get("infra_blocked")]
    learn_write_episodes = [
        ep
        for ep in episodes
        if ep.get("bucket") == "learn" and not ep.get("infra_blocked") and _memory_write_episode(ep)
    ]
    attribution_episodes = [
        ep for ep in episodes if ep.get("bucket") != "baseline" and not ep.get("infra_blocked")
    ] or [ep for ep in episodes if not ep.get("infra_blocked")]
    expected_signal = episodes[0].get("expected_persistence_signal") or episodes[0].get("mechanism", "mixed")
    mechanism_confidence = round(
        mean(ep.get("mechanism_scores", {}).get("mechanism_confidence", 0.0) for ep in attribution_episodes),
        6,
    ) if attribution_episodes else 0.0
    shortcut_resistance = round(
        mean(ep.get("mechanism_scores", {}).get("shortcut_resistance", 0.0) for ep in attribution_episodes),
        6,
    ) if attribution_episodes else 0.0
    transfer_quality = round(
        mean(ep.get("mechanism_scores", {}).get("transfer_quality", 0.0) for ep in attribution_episodes),
        6,
    ) if attribution_episodes else 0.0
    eval_near = _aggregate_bucket(_stage_episodes("eval_near"))
    eval_far = _aggregate_bucket(_stage_episodes("eval_far"))
    delta_score = evaluation["avg_task_score"] - baseline["avg_task_score"]
    write_precision = round(learn["avg_artifact_quality_score"], 6)
    recall_accuracy = round(
        mean(
            _task_component(ep, "content_correctness")
            if ep.get("retrieval_signals", {}).get("used_expected_signal")
            else 0.0
            for ep in evaluation_episodes
        ),
        6,
    ) if evaluation_episodes else 0.0
    update_correctness = round(
        mean(
            (
                ep.get("mechanism_scores", {}).get(
                    "stale_memory_resistance_score",
                    ep.get("mechanism_scores", {}).get("artifact_quality_score", 0.0),
                )
                + _task_component(ep, "content_correctness")
            )
            / 2.0
            for ep in update_episodes
        ),
        6,
    ) if update_episodes else 0.0
    retention_horizon = _retention_horizon(eval_near["avg_task_score"], eval_far["avg_task_score"])
    memory_pollution_rate = round(
        mean(_episode_memory_pollution_rate(ep) for ep in learn_write_episodes),
        6,
    ) if learn_write_episodes else 0.0
    mechanism_score = round(
        mean(
            [
                write_precision,
                recall_accuracy,
                update_correctness,
                retention_horizon,
                1.0 - memory_pollution_rate,
            ]
        ),
        6,
    )
    outcome_delta = round(delta_score, 6)
    evolve_score = round(max(0.0, outcome_delta) * mechanism_score, 6)
    evolve_index = evolve_score

    if delta_score <= 0:
        attribution_label = "none"
    elif expected_signal == "session_search" and (
        mechanism_confidence >= 0.75
        and evaluation["artifact_reuse_rate"] > 0
        and evaluation["retrieval_before_success_rate"] > 0
    ):
        attribution_label = "strong"
    elif expected_signal == "session_search" and (
        mechanism_confidence >= 0.5
        and evaluation["artifact_reuse_rate"] > 0
    ):
        attribution_label = "medium"
    elif (
        mechanism_confidence >= 0.75
        and evaluation["artifact_reuse_rate"] > 0
        and evaluation["avg_artifact_quality_score"] >= 0.5
    ):
        attribution_label = "strong"
    elif (
        mechanism_confidence >= 0.5
        and (evaluation["artifact_reuse_rate"] > 0 or evaluation["avg_artifact_quality_score"] >= 0.5)
    ):
        attribution_label = "medium"
    else:
        attribution_label = "weak"

    return {
        "family_id": family_id,
        "mechanism": episodes[0].get("mechanism", "mixed") if episodes else "mixed",
        "expected_persistence_signal": expected_signal,
        "episodes": episodes,
        "bucket_summary": bucket_summary,
        "metrics": {
            "write_precision": write_precision,
            "recall_accuracy": recall_accuracy,
            "update_correctness": update_correctness,
            "retention_horizon": retention_horizon,
            "retention_horizon_score_delta": round(eval_far["avg_task_score"] - eval_near["avg_task_score"], 6),
            "retention_horizon_pass_delta": round(eval_far["pass_rate"] - eval_near["pass_rate"], 6),
            "memory_pollution_rate": memory_pollution_rate,
        },
        "evolve": {
            "outcome_delta": outcome_delta,
            "mechanism_score": mechanism_score,
            "evolve_score": evolve_score,
        },
        "improvement": {
            "task_score_delta_eval_minus_baseline": outcome_delta,
            "pass_rate_delta_eval_minus_baseline": round(
                evaluation["pass_rate"] - baseline["pass_rate"], 6
            ),
            "memory_artifacts_present": any(
                ep["artifacts"]["memory_file_exists"] or ep["artifacts"]["user_file_exists"]
                for ep in episodes
            ),
            "skill_artifacts_present": any(ep["artifacts"]["skill_count"] > 0 for ep in episodes),
            "session_search_used": any(ep["internal_tools"].get("session_search_calls", 0) > 0 for ep in episodes),
            "infra_blocked_episodes": sum(1 for ep in episodes if ep.get("infra_blocked")),
        },
        "attribution": {
            "label": attribution_label,
            "mechanism_confidence": mechanism_confidence,
            "transfer_quality": transfer_quality,
            "shortcut_resistance": shortcut_resistance,
            "stability": 1.0,
            "evolve_index": evolve_index,
        },
    }


def summarize_sequence(*, sequence_name: str, variant: str, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        families[ep["family_id"]].append(ep)
        buckets[ep["bucket"]].append(ep)

    family_summaries = {
        family_id: _summarize_family(family_id, items)
        for family_id, items in sorted(families.items())
    }
    bucket_summary = {
        bucket: _aggregate_bucket(items)
        for bucket, items in sorted(buckets.items())
    }
    baseline_summary = bucket_summary.get("baseline", _aggregate_bucket([]))
    evaluation_summary = bucket_summary.get("evaluation", _aggregate_bucket([]))

    family_deltas = [
        summary["improvement"]["task_score_delta_eval_minus_baseline"]
        for summary in family_summaries.values()
        if summary["bucket_summary"].get("baseline", {}).get("episode_count", 0) > 0
        and summary["bucket_summary"].get("evaluation", {}).get("episode_count", 0) > 0
    ]
    family_pass_deltas = [
        summary["improvement"]["pass_rate_delta_eval_minus_baseline"]
        for summary in family_summaries.values()
        if summary["bucket_summary"].get("baseline", {}).get("episode_count", 0) > 0
        and summary["bucket_summary"].get("evaluation", {}).get("episode_count", 0) > 0
    ]
    evolve_indices = [summary["attribution"]["evolve_index"] for summary in family_summaries.values()]
    outcome_deltas = [summary["evolve"]["outcome_delta"] for summary in family_summaries.values()]
    mechanism_scores = [summary["evolve"]["mechanism_score"] for summary in family_summaries.values()]
    evolve_scores = [summary["evolve"]["evolve_score"] for summary in family_summaries.values()]
    write_precisions = [summary["metrics"]["write_precision"] for summary in family_summaries.values()]
    recall_accuracies = [summary["metrics"]["recall_accuracy"] for summary in family_summaries.values()]
    update_correctness_scores = [summary["metrics"]["update_correctness"] for summary in family_summaries.values()]
    retention_horizons = [summary["metrics"]["retention_horizon"] for summary in family_summaries.values()]
    memory_pollution_rates = [summary["metrics"]["memory_pollution_rate"] for summary in family_summaries.values()]
    strong_attribution_families = sum(
        1 for summary in family_summaries.values() if summary["attribution"]["label"] == "strong"
    )

    return {
        "sequence": sequence_name,
        "variant": variant,
        "episodes": episodes,
        "bucket_summary": bucket_summary,
        "family_summary": family_summaries,
        "benchmark_signal": {
            "avg_family_task_score_delta": round(mean(family_deltas), 6) if family_deltas else 0.0,
            "avg_family_pass_rate_delta": round(mean(family_pass_deltas), 6) if family_pass_deltas else 0.0,
            "baseline_avg_score": baseline_summary["avg_task_score"],
            "baseline_pass_rate": baseline_summary["pass_rate"],
            "baseline_headroom": round(1.0 - baseline_summary["avg_task_score"], 6),
            "evaluation_avg_score": evaluation_summary["avg_task_score"],
            "evaluation_pass_rate": evaluation_summary["pass_rate"],
            "families_with_memory_artifacts": sum(
                1 for summary in family_summaries.values() if summary["improvement"]["memory_artifacts_present"]
            ),
            "families_with_skill_artifacts": sum(
                1 for summary in family_summaries.values() if summary["improvement"]["skill_artifacts_present"]
            ),
            "families_using_session_search": sum(
                1 for summary in family_summaries.values() if summary["improvement"]["session_search_used"]
            ),
            "families_with_infra_blocks": sum(
                1 for summary in family_summaries.values() if summary["improvement"]["infra_blocked_episodes"] > 0
            ),
            "avg_write_precision": round(mean(write_precisions), 6) if write_precisions else 0.0,
            "avg_recall_accuracy": round(mean(recall_accuracies), 6) if recall_accuracies else 0.0,
            "avg_update_correctness": round(mean(update_correctness_scores), 6) if update_correctness_scores else 0.0,
            "avg_retention_horizon": round(mean(retention_horizons), 6) if retention_horizons else 0.0,
            "avg_memory_pollution_rate": round(mean(memory_pollution_rates), 6) if memory_pollution_rates else 0.0,
            "avg_outcome_delta": round(mean(outcome_deltas), 6) if outcome_deltas else 0.0,
            "avg_mechanism_score": round(mean(mechanism_scores), 6) if mechanism_scores else 0.0,
            "avg_evolve_score": round(mean(evolve_scores), 6) if evolve_scores else 0.0,
            "avg_evolve_index": round(mean(evolve_indices), 6) if evolve_indices else 0.0,
            "strong_attribution_families": strong_attribution_families,
        },
    }


def summarize_single_task_sequence(
    *,
    sequence_name: str,
    variant: str,
    selected_candidate: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    measured = [ep for ep in episodes if ep.get("episode_kind", "attempt") == "attempt"]
    reflections = [ep for ep in episodes if ep.get("episode_kind") == "reflection"]

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in measured:
        buckets[ep["bucket"]].append(ep)
    bucket_summary = {bucket: _aggregate_bucket(items) for bucket, items in sorted(buckets.items())}
    reflection_summary = {
        "episode_count": len(reflections),
        "episodes_with_memory": sum(
            1 for ep in reflections if ep["artifacts"]["memory_file_exists"] or ep["artifacts"]["user_file_exists"]
        ),
        "episodes_with_skills": sum(1 for ep in reflections if ep["artifacts"]["skill_count"] > 0),
        "memory_calls": sum(ep["internal_tools"].get("memory_calls", 0) for ep in reflections),
        "skill_manage_calls": sum(ep["internal_tools"].get("skill_manage_calls", 0) for ep in reflections),
        "session_search_calls": sum(ep["internal_tools"].get("session_search_calls", 0) for ep in reflections),
    }

    baseline = bucket_summary.get("baseline", _aggregate_bucket([]))
    retention = bucket_summary.get("retention", _aggregate_bucket([]))
    transfer = bucket_summary.get("transfer", _aggregate_bucket([]))

    return {
        "sequence": sequence_name,
        "variant": variant,
        "selected_candidate": selected_candidate,
        "episodes": episodes,
        "bucket_summary": bucket_summary,
        "reflection_summary": reflection_summary,
        "benchmark_signal": {
            "baseline_pass_rate": baseline["pass_rate"],
            "baseline_avg_score": baseline["avg_task_score"],
            "retention_pass_rate": retention["pass_rate"],
            "retention_avg_score": retention["avg_task_score"],
            "transfer_pass_rate": transfer["pass_rate"],
            "transfer_avg_score": transfer["avg_task_score"],
            "retention_score_delta": round(retention["avg_task_score"] - baseline["avg_task_score"], 6),
            "retention_pass_delta": round(retention["pass_rate"] - baseline["pass_rate"], 6),
            "transfer_score_delta_vs_baseline": round(transfer["avg_task_score"] - baseline["avg_task_score"], 6),
            "transfer_pass_delta_vs_baseline": round(transfer["pass_rate"] - baseline["pass_rate"], 6),
            "reflection_memory_artifacts": reflection_summary["episodes_with_memory"],
            "reflection_skill_artifacts": reflection_summary["episodes_with_skills"],
            "reflection_memory_calls": reflection_summary["memory_calls"],
            "reflection_skill_calls": reflection_summary["skill_manage_calls"],
            "reflection_search_calls": reflection_summary["session_search_calls"],
        },
    }


def summarize_comparison(*, with_persistence: dict[str, Any], without_persistence: dict[str, Any]) -> dict[str, Any]:
    def _overall_eval(summary: dict[str, Any]) -> dict[str, Any]:
        return summary["bucket_summary"].get("evaluation", _aggregate_bucket([]))

    on_eval = _overall_eval(with_persistence)
    off_eval = _overall_eval(without_persistence)
    on_signal = with_persistence["benchmark_signal"]
    off_signal = without_persistence["benchmark_signal"]
    ablation_outcome_delta = round(on_eval["avg_task_score"] - off_eval["avg_task_score"], 6)
    return {
        "with_persistence": with_persistence,
        "without_persistence": without_persistence,
        "delta": {
            "evaluation_avg_task_score": ablation_outcome_delta,
            "evaluation_pass_rate": round(on_eval["pass_rate"] - off_eval["pass_rate"], 6),
            "evaluation_avg_tool_dispatch_count": round(
                on_eval["avg_tool_dispatch_count"] - off_eval["avg_tool_dispatch_count"], 6
            ),
            "ablation_outcome_delta": ablation_outcome_delta,
            "ablation_evolve_score": round(
                max(0.0, ablation_outcome_delta) * on_signal["avg_mechanism_score"],
                6,
            ),
            "avg_family_task_score_delta": round(
                on_signal["avg_family_task_score_delta"] - off_signal["avg_family_task_score_delta"], 6
            ),
            "avg_family_pass_rate_delta": round(
                on_signal["avg_family_pass_rate_delta"] - off_signal["avg_family_pass_rate_delta"], 6
            ),
            "avg_outcome_delta": round(on_signal["avg_outcome_delta"] - off_signal["avg_outcome_delta"], 6),
            "avg_mechanism_score": round(on_signal["avg_mechanism_score"] - off_signal["avg_mechanism_score"], 6),
            "avg_evolve_score": round(on_signal["avg_evolve_score"] - off_signal["avg_evolve_score"], 6),
        },
    }


def summarize_single_task_comparison(*, with_persistence: dict[str, Any], without_persistence: dict[str, Any]) -> dict[str, Any]:
    on_signal = with_persistence["benchmark_signal"]
    off_signal = without_persistence["benchmark_signal"]
    return {
        "with_persistence": with_persistence,
        "without_persistence": without_persistence,
        "delta": {
            "retention_avg_score": round(on_signal["retention_avg_score"] - off_signal["retention_avg_score"], 6),
            "retention_pass_rate": round(on_signal["retention_pass_rate"] - off_signal["retention_pass_rate"], 6),
            "transfer_avg_score": round(on_signal["transfer_avg_score"] - off_signal["transfer_avg_score"], 6),
            "transfer_pass_rate": round(on_signal["transfer_pass_rate"] - off_signal["transfer_pass_rate"], 6),
            "retention_score_delta": round(on_signal["retention_score_delta"] - off_signal["retention_score_delta"], 6),
            "retention_pass_delta": round(on_signal["retention_pass_delta"] - off_signal["retention_pass_delta"], 6),
            "transfer_score_delta_vs_baseline": round(
                on_signal["transfer_score_delta_vs_baseline"] - off_signal["transfer_score_delta_vs_baseline"], 6
            ),
            "transfer_pass_delta_vs_baseline": round(
                on_signal["transfer_pass_delta_vs_baseline"] - off_signal["transfer_pass_delta_vs_baseline"], 6
            ),
            "reflection_memory_calls": on_signal["reflection_memory_calls"] - off_signal["reflection_memory_calls"],
            "reflection_skill_calls": on_signal["reflection_skill_calls"] - off_signal["reflection_skill_calls"],
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
