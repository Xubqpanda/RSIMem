"""Prepare a manifest-bound native-static PAST attribution execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from .memory.family_matrix import PastFamilyMatrix
from .native_attribution_run import NativeAttributionRunSpec, _family_runtime_identity


@dataclass(frozen=True, slots=True)
class PreparedNativeAttributionLaunch:
    run_id: str
    sequence_path: Path
    sequence_digest: str
    command: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.sequence_path.is_file() or len(self.sequence_digest) != 64:
            raise ValueError("prepared native sequence identity is invalid")
        required = {
            "--rsimem-method-task-id", "--rsimem-state-dir",
            "--rsimem-hermes-home-dir", "--rsimem-artifact-dir",
            "--port-offset", "--model", "--base-url",
        }
        if not required.issubset(self.command):
            raise ValueError("prepared native command lacks isolation arguments")


def prepare_native_attribution_launch(
    *,
    run: NativeAttributionRunSpec,
    past_bench_root: Path,
    output_root: Path,
    past_bench_binary: str = "past-bench",
    agent: str = "hermes-luna",
) -> PreparedNativeAttributionLaunch:
    """Verify a frozen run and materialize its native-only sequence."""

    if run.condition != "native_static":
        raise ValueError("native attribution launcher only accepts native_static")
    root = Path(past_bench_root).expanduser().resolve()
    spec = PastFamilyMatrix.create_default().spec_for(run.family_id)
    family_digest, fixture_digest, base_ports, episodes, task_ids, base_services = _family_runtime_identity(
        family_id=run.family_id, task_root=spec.task_root, past_bench_root=root
    )
    if family_digest != run.family_source_digest or fixture_digest != run.fixture_digest:
        raise ValueError("native attribution family or fixture identity drift")
    if episodes != run.native_episode_ids:
        raise ValueError("native attribution episode identity drift")
    if task_ids != run.native_task_ids:
        raise ValueError("native attribution task identity drift")
    if tuple(port + run.port_offset for port in base_ports) != run.service_ports:
        raise ValueError("native attribution service port identity drift")
    expected_services = tuple({
        **value.payload(), "port": value.port + run.port_offset,
    } for value in base_services)
    if expected_services != tuple(value.payload() for value in run.service_identities):
        raise ValueError("native attribution service fixture identity drift")

    from past_bench.self_evolve_v2 import generate_manifest

    output = Path(output_root).expanduser().resolve()
    run_root = output / run.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    generated = run_root / ".family_base.yaml"
    family_relative = str(Path(spec.task_root).relative_to("self-evolve-tasks-v2"))
    generate_manifest(family_relative, out_path=generated, repo_root=root)
    try:
        base = yaml.safe_load(generated.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("generated family manifest is unreadable") from exc
    if not isinstance(base, Mapping) or not isinstance(base.get("episodes"), list):
        raise ValueError("generated family manifest is malformed")
    selected: list[dict[str, object]] = []
    for episode in base["episodes"]:
        if not isinstance(episode, Mapping):
            raise ValueError("generated family episode is malformed")
        task_path = Path(str(episode.get("task", "")))
        if not task_path.is_absolute():
            task_path = (generated.parent / task_path).resolve()
        if task_path.name in run.native_episode_ids:
            selected.append(dict(episode))
    if tuple(Path(str(episode["task"])).name for episode in selected) != run.native_episode_ids:
        raise ValueError("generated native episode selection does not match run manifest")
    for episode in selected:
        if episode.get("bucket") == "control":
            raise ValueError("control episode entered native attribution sequence")
        episode["shared_cold_run"] = False
    document = {
        "name": f"native-attribution.{run.run_id}",
        "description": "Manifest-bound native-static failure-attribution execution.",
        "hermes": dict(base.get("hermes") or {}),
        "episodes": selected,
    }
    target = run_root / "sequence.yaml"
    rendered = yaml.safe_dump(document, allow_unicode=False, sort_keys=True)
    target.write_text(rendered, encoding="utf-8")
    sequence_digest = hashlib.sha256(
        json.dumps(document, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    command = (
        past_bench_binary, "evolve", "--sequence", str(target), "--agent", agent,
        "--runtime", "local", "--model", run.model_id, "--base-url", f"https://{run.provider_id}",
        "--persistence-variant", "with_persistence", "--rsimem-mode", "native+ledger",
        "--rsimem-method-task-id", run.method_case_id, "--port-offset", str(run.port_offset),
        "--trace-dir", str(output / run.trace_directory),
        "--rsimem-state-dir", str(output / run.state_directory),
        "--rsimem-hermes-home-dir", str(output / run.hermes_home_directory),
        "--rsimem-artifact-dir", str(output / run.artifact_directory),
    )
    return PreparedNativeAttributionLaunch(run.run_id, target, sequence_digest, command)


__all__ = ["PreparedNativeAttributionLaunch", "prepare_native_attribution_launch"]
