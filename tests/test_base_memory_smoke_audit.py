from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.base_memory.base_memory_experiment import HISTORICAL_BASE_MEMORY_CONDITIONS, BaseMemoryCondition
from experiments.base_memory.base_memory_smoke_audit import audit_smoke_trio


def _run(root: Path, condition: BaseMemoryCondition) -> None:
    root.mkdir()
    trace = root / "trace"; trace.mkdir()
    (trace / "sequence_results.json").write_text(json.dumps({"episodes": [{"token_usage": {"model_usage_complete": True, "input_tokens": 1, "output_tokens": 2}}]}), encoding="utf-8")
    backend = {
        BaseMemoryCondition.NO_MEMORY: {"backend_id": "none-v1", "semantic_memory_enabled": False},
        BaseMemoryCondition.HERMES_NATIVE: {"backend_id": "hermes-native-semantic-v1", "semantic_memory_enabled": True},
        BaseMemoryCondition.MEM0_STATIC: {"backend_id": "mem0-flat-hermes-v1", "semantic_memory_enabled": True},
    }[condition]
    manifest = {"manifest_id": condition.value, "runs": [], "sequence_digest": condition.value, "run_id": condition.value, "condition": condition.value, "backend": backend, "port_offset": 1, "state_directory": condition.value + "-state", "trace_directory": "trace", "artifact_directory": condition.value + "-artifacts", "hermes_home_directory": condition.value + "-home", "base_model": "gpt-5.6-luna", "source_sequence_digest": "a" * 64, "fixture_digest": "b" * 64, "temperature": 0.0, "token_budget": 1, "config_digest": "c" * 64, "registry_digest": "d" * 64}
    (root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if condition is BaseMemoryCondition.MEM0_STATIC:
        (root / "rsimem_semantic_operations.jsonl").write_text("\n".join(json.dumps({"payload": {"kind": kind}}) for kind in ("fact_extraction", "mutation", "retrieval", "injection")), encoding="utf-8")


def test_accepts_complete_three_backend_smoke(tmp_path: Path) -> None:
    roots = {condition: tmp_path / condition.value for condition in HISTORICAL_BASE_MEMORY_CONDITIONS}
    for condition, root in roots.items(): _run(root, condition)
    report = audit_smoke_trio(roots)
    assert report["accepted"] is True
    assert report["mem0_operation_evidence"]["operation_event_count"] == 4


def test_rejects_missing_mem0_injection_evidence(tmp_path: Path) -> None:
    roots = {condition: tmp_path / condition.value for condition in HISTORICAL_BASE_MEMORY_CONDITIONS}
    for condition, root in roots.items(): _run(root, condition)
    path = roots[BaseMemoryCondition.MEM0_STATIC] / "rsimem_semantic_operations.jsonl"
    path.write_text(json.dumps({"payload": {"kind": "fact_extraction"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="operation evidence"):
        audit_smoke_trio(roots)
