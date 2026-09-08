import json
from pathlib import Path

import pytest

from rsimem.adamem_experiment import AdaMemCondition
from rsimem.adamem_smoke_audit import audit_smoke_trio


def _write_run(root: Path, condition: AdaMemCondition, *, outcome: str, applied: bool) -> None:
    root.mkdir()
    common = {
        "schema": "rsimem-adamem-run-manifest-v1", "protocol_id": "adamem-trajectory-baseline-v1",
        "run_id": condition.value, "condition": condition.value,
        "feedback_view": {AdaMemCondition.MEM0_STATIC: None, AdaMemCondition.ADAMEM_TERMINAL: "terminal", AdaMemCondition.ADAMEM_FULL_TRAJECTORY: "full_trajectory"}[condition],
        "source_sequence_digest": "a" * 64, "split_id": "split", "base_model": "gpt-5.6-luna", "meta_agent_model": "gpt-5.6-luna",
        "temperature": 0.0, "update_budget": 1, "policy_update_space": "versioned_semantic_extraction_policy_only",
        "mem0_backend": "mem0-flat-hermes-v1", "past_bin_digest": "b" * 64, "config_digest": "c" * 64, "registry_digest": "d" * 64,
        "port_offset": list(AdaMemCondition).index(condition) + 1000,
    }
    (root / "run_manifest.json").write_text(json.dumps(common), encoding="utf-8")
    (root / "policy_receipt.json").write_text(json.dumps({"outcome": outcome, "feedback_view": common["feedback_view"]}), encoding="utf-8")
    (root / "policy_binding.json").write_text(json.dumps({"adamem_policy_applied": applied}), encoding="utf-8")
    for phase in ("prefix", "suffix"):
        (root / phase).mkdir()
        (root / phase / "sequence_results.json").write_text(json.dumps({"episodes": [{"token_usage": {"model_usage_complete": True}}]}), encoding="utf-8")


def test_smoke_audit_accepts_valid_no_update_and_update(tmp_path: Path) -> None:
    roots = {condition: tmp_path / condition.value for condition in AdaMemCondition}
    _write_run(roots[AdaMemCondition.MEM0_STATIC], AdaMemCondition.MEM0_STATIC, outcome="static", applied=False)
    _write_run(roots[AdaMemCondition.ADAMEM_TERMINAL], AdaMemCondition.ADAMEM_TERMINAL, outcome="no_update", applied=False)
    _write_run(roots[AdaMemCondition.ADAMEM_FULL_TRAJECTORY], AdaMemCondition.ADAMEM_FULL_TRAJECTORY, outcome="updated", applied=True)
    report = audit_smoke_trio(roots)
    assert report["accepted"] is True
    assert report["outcomes"]["B2_mem0_adamem_full_trajectory"] == "updated"


def test_smoke_audit_rejects_incomplete_usage(tmp_path: Path) -> None:
    roots = {condition: tmp_path / condition.value for condition in AdaMemCondition}
    _write_run(roots[AdaMemCondition.MEM0_STATIC], AdaMemCondition.MEM0_STATIC, outcome="static", applied=False)
    _write_run(roots[AdaMemCondition.ADAMEM_TERMINAL], AdaMemCondition.ADAMEM_TERMINAL, outcome="no_update", applied=False)
    _write_run(roots[AdaMemCondition.ADAMEM_FULL_TRAJECTORY], AdaMemCondition.ADAMEM_FULL_TRAJECTORY, outcome="updated", applied=True)
    bad = roots[AdaMemCondition.ADAMEM_TERMINAL] / "prefix" / "sequence_results.json"
    bad.write_text(json.dumps({"episodes": [{"token_usage": {"model_usage_complete": False}}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        audit_smoke_trio(roots)
