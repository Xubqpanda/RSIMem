"""Fail-closed audit for one Stage 0 base-memory smoke trio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from .base_memory_experiment import (
    HISTORICAL_BASE_MEMORY_CONDITIONS,
    BaseMemoryCondition,
    compare_run_manifests,
)
from rsimem.memory.surface_policy import MemorySurface, RuntimeSurfacePolicy, SurfaceOperation, capability_check


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"base-memory audit cannot read {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"base-memory audit requires object {path}")
    return value


def _manifest(root: Path, condition: BaseMemoryCondition) -> dict[str, object]:
    payload = _read_json(root / "run_manifest.json")
    if payload.get("condition") != condition.value:
        raise ValueError(f"base-memory run condition mismatch for {condition.value}")
    return payload


def _complete_usage(root: Path, manifest: Mapping[str, object]) -> dict[str, int]:
    trace = manifest.get("trace_directory")
    if not isinstance(trace, str):
        raise ValueError("base-memory run manifest misses trace directory")
    payload = _read_json(root / trace / "sequence_results.json")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("base-memory sequence has no episode results")
    if any(
        not isinstance(item, Mapping)
        or not isinstance(item.get("token_usage"), Mapping)
        or item["token_usage"].get("model_usage_complete") is not True
        for item in episodes
    ):
        raise ValueError("base-memory smoke has incomplete model usage")
    return {
        "episodes": len(episodes),
        "input_tokens": sum(int(item["token_usage"].get("input_tokens") or 0) for item in episodes),
        "output_tokens": sum(int(item["token_usage"].get("output_tokens") or 0) for item in episodes),
    }


def _mem0_evidence(root: Path) -> dict[str, object]:
    kinds: set[str] = set()
    event_count = 0
    for path in root.rglob("rsimem_semantic_operations.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            payload = event.get("payload") if isinstance(event, Mapping) else None
            if isinstance(payload, Mapping) and isinstance(payload.get("kind"), str):
                kinds.add(payload["kind"])
                event_count += 1
    required = {"fact_extraction", "mutation", "retrieval", "injection"}
    missing = required - kinds
    if missing:
        raise ValueError("Mem0Static smoke lacks operation evidence: " + ", ".join(sorted(missing)))
    return {"operation_event_count": event_count, "operation_kinds": sorted(kinds)}


def _all_memory_off_evidence(root: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    """Fail closed on any observed semantic, episodic, or procedural surface."""
    descriptor = manifest.get("backend")
    expected = {
        "semantic_memory_enabled": False,
        "episodic_memory_enabled": False,
        "procedural_memory_enabled": False,
        "semantic_writeback_mode": "disabled",
        "memory_surface_policy": "all_memory_off",
    }
    if not isinstance(descriptor, Mapping) or any(descriptor.get(key) != value for key, value in expected.items()):
        raise ValueError("AllMemoryOff backend descriptor is incomplete")
    policy_payload = descriptor.get("surface_policy")
    policy = (
        RuntimeSurfacePolicy.from_payload(policy_payload)
        if isinstance(policy_payload, Mapping)
        else RuntimeSurfacePolicy.all_memory_off()
    )
    if any(capability_check(policy, surface, SurfaceOperation.RETRIEVE) for surface in MemorySurface):
        raise ValueError("AllMemoryOff surface policy exposes a Memory surface")
    trace = manifest.get("trace_directory")
    if not isinstance(trace, str):
        raise ValueError("AllMemoryOff run manifest misses trace directory")
    sequence = _read_json(root / trace / "sequence_results.json")
    episodes = sequence.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("AllMemoryOff sequence has no episode results")
    forbidden_counts = {
        "memory_calls", "memory_write_count", "memory_read_count",
        "skill_manage_calls", "skill_create_count", "skill_update_count",
        "session_search_calls", "skill_view_calls", "skills_list_calls",
        "skill_read_count",
    }
    observed: dict[str, int] = {key: 0 for key in forbidden_counts}
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise ValueError("AllMemoryOff episode result is invalid")
        tools = episode.get("internal_tools")
        signals = episode.get("retrieval_signals")
        if not isinstance(tools, Mapping) or not isinstance(signals, Mapping):
            raise ValueError("AllMemoryOff episode lacks tool/retrieval evidence")
        for key in forbidden_counts:
            value = tools.get(key, 0)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"AllMemoryOff tool evidence is invalid: {key}")
            observed[key] += value
        for key in ("memory_read_count", "memory_injection_count", "skill_read_count", "session_search_count", "retrieval_signal_count"):
            # Reflection episodes have no retrieval summary. Absent fields
            # represent no observed retrieval; explicit nonzero evidence still
            # rejects the run.
            value = signals.get(key, 0)
            if value != 0:
                raise ValueError(f"AllMemoryOff observed retrieval evidence: {key}")
    nonempty_evidence = []
    for path in root.rglob("rsimem_memory_events.jsonl"):
        if path.read_text(encoding="utf-8").strip():
            nonempty_evidence.append(str(path.relative_to(root)))
    if any(observed.values()) or nonempty_evidence:
        details = sorted(key for key, value in observed.items() if value)
        details.extend(nonempty_evidence)
        raise ValueError("AllMemoryOff observed Memory surface: " + ", ".join(details))
    forbidden_artifacts = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if (
            "memories" in relative.parts
            or "skills" in relative.parts
            or path.name in {"MEMORY.md", "USER.md", "SKILL.md", "state.db", "session_seed.json"}
        ):
            forbidden_artifacts.append(str(relative))
    if forbidden_artifacts:
        raise ValueError(
            "AllMemoryOff contains Memory seed or storage artifacts: "
            + ", ".join(sorted(forbidden_artifacts))
        )
    return {
        "tool_counts": observed,
        "memory_event_files": 0,
        "memory_seed_artifacts": 0,
        "surface_policy_id": policy.policy_id,
    }


def audit_all_memory_off_smoke(root: Path) -> dict[str, object]:
    """Audit a single representative AllMemoryOff run."""
    root = Path(root)
    manifest = _manifest(root, BaseMemoryCondition.ALL_MEMORY_OFF)
    usage = _complete_usage(root, manifest)
    evidence = _all_memory_off_evidence(root, manifest)
    return {
        "schema": "rsimem-all-memory-off-smoke-audit-v1",
        "accepted": True,
        "condition": BaseMemoryCondition.ALL_MEMORY_OFF.value,
        "usage": usage,
        "all_memory_off_evidence": evidence,
    }


def audit_smoke_trio(roots: Mapping[BaseMemoryCondition, Path]) -> dict[str, object]:
    required = set(HISTORICAL_BASE_MEMORY_CONDITIONS)
    normalized = {BaseMemoryCondition(key): Path(value) for key, value in roots.items()}
    if set(normalized) != required:
        raise ValueError("base-memory smoke requires exactly NoMemory/HermesNative/Mem0Static")
    manifests = {condition: _manifest(root, condition) for condition, root in normalized.items()}
    base = manifests[BaseMemoryCondition.NO_MEMORY]
    differences = {
        condition.value: sorted(compare_run_manifests(base, manifest))
        for condition, manifest in manifests.items()
        if condition is not BaseMemoryCondition.NO_MEMORY
    }
    descriptors = {condition: manifest.get("backend") for condition, manifest in manifests.items()}
    if not isinstance(descriptors[BaseMemoryCondition.NO_MEMORY], Mapping) or descriptors[BaseMemoryCondition.NO_MEMORY].get("semantic_memory_enabled") is not False:
        raise ValueError("NoMemory must disable semantic memory")
    if not isinstance(descriptors[BaseMemoryCondition.HERMES_NATIVE], Mapping) or descriptors[BaseMemoryCondition.HERMES_NATIVE].get("backend_id") != "hermes-native-semantic-v1":
        raise ValueError("HermesNative backend identity is invalid")
    if not isinstance(descriptors[BaseMemoryCondition.MEM0_STATIC], Mapping) or descriptors[BaseMemoryCondition.MEM0_STATIC].get("backend_id") != "mem0-flat-hermes-v1":
        raise ValueError("Mem0Static backend identity is invalid")
    usage = {condition.value: _complete_usage(normalized[condition], manifests[condition]) for condition in HISTORICAL_BASE_MEMORY_CONDITIONS}
    return {
        "schema": "rsimem-base-memory-smoke-audit-v1", "accepted": True,
        "usage": usage, "manifest_differences": differences,
        "backends": {condition.value: descriptors[condition] for condition in HISTORICAL_BASE_MEMORY_CONDITIONS},
        "mem0_operation_evidence": _mem0_evidence(normalized[BaseMemoryCondition.MEM0_STATIC]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-memory", type=Path)
    parser.add_argument("--hermes-native", type=Path)
    parser.add_argument("--mem0-static", type=Path)
    parser.add_argument("--all-memory-off", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.all_memory_off is not None:
        if any(value is not None for value in (args.no_memory, args.hermes_native, args.mem0_static)):
            parser.error("--all-memory-off cannot be combined with the historical smoke trio")
        report = audit_all_memory_off_smoke(args.all_memory_off)
    else:
        if any(value is None for value in (args.no_memory, args.hermes_native, args.mem0_static)):
            parser.error("historical smoke requires --no-memory, --hermes-native, and --mem0-static")
        report = audit_smoke_trio({
            BaseMemoryCondition.NO_MEMORY: args.no_memory,
            BaseMemoryCondition.HERMES_NATIVE: args.hermes_native,
            BaseMemoryCondition.MEM0_STATIC: args.mem0_static,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
