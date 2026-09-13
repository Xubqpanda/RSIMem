from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.base_memory.base_memory_smoke_audit import audit_all_memory_off_smoke


def _write_run(root: Path, *, session_search_calls: int = 0, memory_event: bool = False) -> None:
    trace = root / "trace"
    trace.mkdir(parents=True)
    (root / "run_manifest.json").write_text(json.dumps({
        "condition": "AllMemoryOff",
        "trace_directory": "trace",
        "backend": {
            "semantic_memory_enabled": False,
            "episodic_memory_enabled": False,
            "procedural_memory_enabled": False,
            "semantic_writeback_mode": "disabled",
            "memory_surface_policy": "all_memory_off",
        },
    }), encoding="utf-8")
    (trace / "sequence_results.json").write_text(json.dumps({"episodes": [{
        "token_usage": {"model_usage_complete": True, "input_tokens": 3, "output_tokens": 2},
        "internal_tools": {
            "memory_calls": 0, "memory_write_count": 0, "memory_read_count": 0,
            "skill_manage_calls": 0, "skill_create_count": 0, "skill_update_count": 0,
            "session_search_calls": session_search_calls, "skill_view_calls": 0,
            "skills_list_calls": 0, "skill_read_count": 0,
        },
        "retrieval_signals": {
            "memory_read_count": 0, "memory_injection_count": 0, "skill_read_count": 0,
            "session_search_count": 0, "retrieval_signal_count": 0,
        },
    }]}), encoding="utf-8")
    (root / "artifacts").mkdir()
    (root / "artifacts" / "rsimem_memory_events.jsonl").write_text("event\n" if memory_event else "", encoding="utf-8")


def test_all_memory_off_smoke_accepts_complete_zero_surface_evidence(tmp_path: Path) -> None:
    _write_run(tmp_path)
    report = audit_all_memory_off_smoke(tmp_path)
    assert report["accepted"] is True
    assert report["all_memory_off_evidence"]["tool_counts"]["session_search_calls"] == 0


def test_all_memory_off_smoke_accepts_reflection_without_retrieval_summary(tmp_path: Path) -> None:
    _write_run(tmp_path)
    sequence_path = tmp_path / "trace" / "sequence_results.json"
    sequence = json.loads(sequence_path.read_text())
    sequence["episodes"][0]["retrieval_signals"] = {}
    sequence_path.write_text(json.dumps(sequence), encoding="utf-8")
    assert audit_all_memory_off_smoke(tmp_path)["accepted"] is True


def test_all_memory_off_smoke_rejects_memory_surface(tmp_path: Path) -> None:
    _write_run(tmp_path, session_search_calls=1)
    with pytest.raises(ValueError, match="Memory surface"):
        audit_all_memory_off_smoke(tmp_path)


def test_all_memory_off_smoke_rejects_rsimem_memory_event(tmp_path: Path) -> None:
    _write_run(tmp_path, memory_event=True)
    with pytest.raises(ValueError, match="Memory surface"):
        audit_all_memory_off_smoke(tmp_path)


def test_all_memory_off_smoke_rejects_seeded_memory_artifact(tmp_path: Path) -> None:
    _write_run(tmp_path)
    seed = tmp_path / "trace" / "artifacts" / "memories"
    seed.mkdir(parents=True)
    (seed / "MEMORY.md").write_text("forbidden", encoding="utf-8")
    with pytest.raises(ValueError, match="seed or storage artifacts"):
        audit_all_memory_off_smoke(tmp_path)
