"""Materialize and execute the Stage 0 base-memory smoke protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Mapping

import yaml

from rsimem.evaluation.acceptance import require_complete_phase
from rsimem.hosts.hermes.runtime import build_past_environment
from rsimem.memory.surface_policy import RuntimeSurfacePolicy

_past_environment = build_past_environment
_require_accepted_phase = require_complete_phase
from .base_memory_experiment import (
    HISTORICAL_BASE_MEMORY_CONDITIONS,
    BaseMemoryComparisonManifest,
    BaseMemoryCondition,
    BaseMemoryRunSpec,
    FROZEN_MODEL_ID,
)


def _digest_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _read_sequence(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("episodes"), list):
        raise ValueError("base-memory source sequence must contain episodes")
    return value


def _backend_descriptor(condition: BaseMemoryCondition) -> dict[str, object]:
    if condition is BaseMemoryCondition.ALL_MEMORY_OFF:
        surface_policy = RuntimeSurfacePolicy.all_memory_off()
    elif condition is BaseMemoryCondition.NO_MEMORY:
        surface_policy = RuntimeSurfacePolicy.from_hermes_flags(
            memory_enabled=False, user_profile_enabled=False,
            skills_enabled=True, session_search_enabled=True,
            policy_id="semantic-disabled-v1",
        )
    else:
        surface_policy = RuntimeSurfacePolicy.from_hermes_flags(
            memory_enabled=True, user_profile_enabled=True,
            skills_enabled=True, session_search_enabled=True,
            policy_id=("hermes-native-v1" if condition is BaseMemoryCondition.HERMES_NATIVE
                       else "mem0-semantic-v1"),
        )
    if condition is BaseMemoryCondition.ALL_MEMORY_OFF:
        return {
            "backend_id": "all-memory-off-v1",
            "semantic_memory_enabled": False,
            "episodic_memory_enabled": False,
            "procedural_memory_enabled": False,
            "hermes_memory_writer": False,
            "rsimem_mode": "native+ledger",
            "semantic_writeback_mode": "disabled",
            "persistence_variant": "with_persistence",
            "memory_surface_policy": "all_memory_off",
            "surface_policy": surface_policy.payload(),
        }
    if condition is BaseMemoryCondition.NO_MEMORY:
        return {
            "backend_id": "none-v1", "semantic_memory_enabled": False,
            "hermes_memory_writer": False, "rsimem_mode": "native+ledger",
            "semantic_writeback_mode": "disabled", "persistence_variant": "with_persistence",
            "surface_policy": surface_policy.payload(),
        }
    if condition is BaseMemoryCondition.HERMES_NATIVE:
        return {
            "backend_id": "hermes-native-semantic-v1", "semantic_memory_enabled": True,
            "hermes_memory_writer": True, "rsimem_mode": "native+ledger",
            "semantic_writeback_mode": "disabled", "persistence_variant": "with_persistence",
            "surface_policy": surface_policy.payload(),
        }
    return {
        "backend_id": "mem0-flat-hermes-v1", "semantic_memory_enabled": True,
        "hermes_memory_writer": False, "rsimem_mode": "native+ledger",
        "semantic_writeback_mode": "static", "persistence_variant": "with_persistence",
        "surface_policy": surface_policy.payload(),
    }


def _materialize_sequence(
    source: Mapping[str, object], condition: BaseMemoryCondition, *, source_directory: Path | None = None,
) -> dict[str, object]:
    result = dict(source)
    episodes = result.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("base-memory source sequence must contain episodes")
    materialized_episodes: list[dict[str, object]] = []
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise ValueError("base-memory source episode must be a mapping")
        materialized = dict(episode)
        # Explicit per-run state/home roots require ordinary execution.  The
        # PAST shared-cold optimization is a cross-variant state reuse path and
        # is incompatible with the Stage 0 isolation contract.
        materialized["shared_cold_run"] = False
        task = materialized.get("task")
        if not isinstance(task, str) or not task:
            raise ValueError("base-memory source episode task is invalid")
        if source_directory is not None and not Path(task).is_absolute():
            materialized["task"] = str((source_directory / task).resolve())
        materialized_episodes.append(materialized)
    result["episodes"] = materialized_episodes
    hermes = dict(result.get("hermes") or {})
    descriptor = _backend_descriptor(condition)
    # Preserve the ordinary with_persistence execution topology. The historical
    # NoMemory condition disables only semantic memory; AllMemoryOff explicitly
    # disables every persistence-facing agent surface.
    if condition is BaseMemoryCondition.ALL_MEMORY_OFF:
        hermes["memory_enabled"] = False
        hermes["user_profile_enabled"] = False
        hermes["skills_enabled"] = False
        hermes["session_search_enabled"] = False
        hermes["all_memory_off"] = True
    elif condition is BaseMemoryCondition.NO_MEMORY:
        hermes["memory_enabled"] = False
        hermes["user_profile_enabled"] = False
    else:
        hermes["memory_enabled"] = True
        hermes["user_profile_enabled"] = True
    hermes["rsimem_mode"] = descriptor["rsimem_mode"]
    hermes["rsimem_semantic_writeback_mode"] = descriptor["semantic_writeback_mode"]
    hermes["reasoning_effort"] = "none"
    result["hermes"] = hermes
    return result


def _past_command(*, past_bin: Path, sequence: Path, trace_dir: Path, config: Path, registry: Path, run: BaseMemoryRunSpec, base_url: str) -> list[str]:
    descriptor = _backend_descriptor(run.condition)
    command = [
        str(past_bin), "evolve", "--sequence", str(sequence), "--agent", "hermes",
        "--runtime", "local", "--sandbox", "--sandbox-tools", "--no-judge",
        "--persistence-variant", "with_persistence", "--config", str(config),
        "--registry", str(registry), "--trace-dir", str(trace_dir), "--model", FROZEN_MODEL_ID,
        "--base-url", base_url, "--rsimem-mode", str(descriptor["rsimem_mode"]),
        "--rsimem-semantic-writeback-mode", str(descriptor["semantic_writeback_mode"]),
        "--port-offset", str(run.port_offset),
        "--rsimem-state-dir", str(trace_dir.parent / run.state_directory),
        "--rsimem-hermes-home-dir", str(trace_dir.parent / run.hermes_home_directory),
        "--rsimem-artifact-dir", str(trace_dir.parent / run.artifact_directory),
    ]
    return command


def prepare_comparison(*, source_sequence: Path, output_root: Path, family_id: str, config: Path, registry: Path, token_budget: int, run_prefix: str = "base-memory", temperature: float = 0.0, port_base: int = 12000, conditions: tuple[BaseMemoryCondition, ...] = HISTORICAL_BASE_MEMORY_CONDITIONS) -> BaseMemoryComparisonManifest:
    source = _read_sequence(source_sequence)
    manifest = BaseMemoryComparisonManifest.create(
        family_id=family_id, source_sequence_digest=_digest_path(source_sequence),
        fixture_digest=hashlib.sha256(yaml.safe_dump(source, sort_keys=True).encode("utf-8")).hexdigest(),
        config_digest=_digest_path(config), registry_digest=_digest_path(registry),
        token_budget=token_budget, run_prefix=run_prefix, temperature=temperature,
        port_base=port_base, conditions=conditions,
    )
    _write_json(output_root / "base_memory_manifest.json", manifest.payload())
    for run in manifest.runs:
        root = output_root / run.run_id
        sequence = _materialize_sequence(
            source, run.condition, source_directory=source_sequence.parent,
        )
        sequence_file = root / "sequence.yaml"
        sequence_file.parent.mkdir(parents=True, exist_ok=True)
        sequence_file.write_text(yaml.safe_dump(sequence, allow_unicode=False, sort_keys=False), encoding="utf-8")
        _write_json(root / "run_manifest.json", {
            **manifest.payload(), "run_id": run.run_id, "condition": run.condition.value,
            "backend": _backend_descriptor(run.condition), **run.payload(),
            "sequence_digest": _digest_path(sequence_file),
        })
    return manifest


def run_comparison(
    *, source_sequence: Path, output_root: Path, family_id: str, past_bin: Path,
    past_root: Path, config: Path, registry: Path, base_url: str, token_budget: int,
    run_prefix: str = "base-memory", condition: BaseMemoryCondition | None = None,
    dry_run: bool = False, port_base: int = 12000,
) -> BaseMemoryComparisonManifest:
    normalized_condition = BaseMemoryCondition(condition) if condition is not None else None
    selected_conditions = (
        (BaseMemoryCondition.ALL_MEMORY_OFF,)
        if normalized_condition is BaseMemoryCondition.ALL_MEMORY_OFF
        else HISTORICAL_BASE_MEMORY_CONDITIONS
    )
    manifest = prepare_comparison(
        source_sequence=source_sequence, output_root=output_root, family_id=family_id,
        config=config, registry=registry, token_budget=token_budget, run_prefix=run_prefix,
        port_base=port_base, conditions=selected_conditions,
    )
    selected_runs = tuple(
        run for run in manifest.runs
        if normalized_condition is None or run.condition is normalized_condition
    )
    if not selected_runs:
        raise ValueError("requested base-memory condition is absent from manifest")
    for run in selected_runs:
        root = output_root / run.run_id
        command = _past_command(
            past_bin=past_bin, sequence=root / "sequence.yaml", trace_dir=root / run.trace_directory,
            config=config, registry=registry, run=run, base_url=base_url,
        )
        _write_json(root / "launch.json", {"command": command, "dry_run": dry_run})
        if not dry_run:
            environment = _past_environment(
                base_url=base_url, api_key=os.environ.get("GPT_LUNA_API_KEY"),
            )
            subprocess.run(command, cwd=past_root, check=True, env=environment)
            _require_accepted_phase(root / run.trace_directory)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--past-bin", type=Path, required=True)
    parser.add_argument("--past-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--run-prefix", default="base-memory")
    parser.add_argument("--condition", choices=[item.value for item in BaseMemoryCondition])
    parser.add_argument("--port-base", type=int, default=12000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    run_comparison(
        source_sequence=args.sequence.resolve(), output_root=args.output_root.resolve(), family_id=args.family_id,
        past_bin=args.past_bin.resolve(), past_root=args.past_root.resolve(), config=args.config.resolve(),
        registry=args.registry.resolve(), base_url=args.base_url, token_budget=args.token_budget,
        run_prefix=args.run_prefix,
        condition=(BaseMemoryCondition(args.condition) if args.condition else None),
        dry_run=args.dry_run, port_base=args.port_base,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
