"""Fail-closed audit for one Stage 0 base-memory smoke trio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from .base_memory_experiment import BaseMemoryCondition, compare_run_manifests


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


def audit_smoke_trio(roots: Mapping[BaseMemoryCondition, Path]) -> dict[str, object]:
    required = set(BaseMemoryCondition)
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
    usage = {condition.value: _complete_usage(normalized[condition], manifests[condition]) for condition in BaseMemoryCondition}
    return {
        "schema": "rsimem-base-memory-smoke-audit-v1", "accepted": True,
        "usage": usage, "manifest_differences": differences,
        "backends": {condition.value: descriptors[condition] for condition in BaseMemoryCondition},
        "mem0_operation_evidence": _mem0_evidence(normalized[BaseMemoryCondition.MEM0_STATIC]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-memory", type=Path, required=True)
    parser.add_argument("--hermes-native", type=Path, required=True)
    parser.add_argument("--mem0-static", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
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
