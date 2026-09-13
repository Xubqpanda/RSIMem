"""Fail-closed matched audit for the post-refactor AdaMem/Mem0 validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from rsimem.memory.surface_policy import RuntimeSurfacePolicy

from .adamem_batch_audit import audit_batch
from .adamem_experiment import AdaMemCondition
from .adamem_launcher import compare_run_manifests


_FORBIDDEN_FEEDBACK_KEYS = {
    "answer", "gold_answer", "golden_feedback", "grader", "judge",
    "official_score", "score", "hidden_answer", "reference_answer",
    "future_evaluation", "task_score",
}


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"validation evidence must be an object: {path}")
    return value


def _assert_feedback_safe(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).strip().lower().replace("-", "_") in _FORBIDDEN_FEEDBACK_KEYS:
                raise ValueError(f"AdaMem updater evidence contains forbidden field: {key}")
            _assert_feedback_safe(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_feedback_safe(child)


def _run_manifests(batch_root: Path, audit: Mapping[str, object]) -> dict[int, dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    for item in audit.get("replicates", ()):
        if not isinstance(item, Mapping):
            raise ValueError("validation audit replicate is malformed")
        run_root = batch_root / str(item["run_id"])
        rows[int(item["replicate"])] = _load(run_root / "run_manifest.json")
    if set(rows) != {1, 2, 3}:
        raise ValueError("validation batch must contain replicate 1, 2, and 3")
    return rows


def audit_matched_family(*, family_id: str, batch_roots: Mapping[AdaMemCondition, Path]) -> dict[str, object]:
    expected_conditions = {
        AdaMemCondition.MEM0_STATIC,
        AdaMemCondition.ADAMEM_FULL_TRAJECTORY,
    }
    if set(batch_roots) != expected_conditions:
        raise ValueError("matched validation requires B0 and B2")
    reports = {condition: audit_batch(root) for condition, root in batch_roots.items()}
    manifests = {condition: _run_manifests(batch_roots[condition], reports[condition]) for condition in expected_conditions}
    base = manifests[AdaMemCondition.MEM0_STATIC][1]
    expected_views = {
        AdaMemCondition.MEM0_STATIC: None,
        AdaMemCondition.ADAMEM_FULL_TRAJECTORY: "full_trajectory",
    }
    policy = RuntimeSurfacePolicy.from_payload(base["surface_policy"])
    if policy.policy_id != "mem0-semantic-v1":
        raise ValueError("validation run does not use frozen Mem0 surface policy")
    for condition in expected_conditions:
        for replicate, manifest in manifests[condition].items():
            if manifest.get("family_id") not in {None, family_id}:
                raise ValueError("validation family identity drift")
            if manifest.get("surface_policy") != base.get("surface_policy"):
                raise ValueError("validation surface policy drift")
            if manifest.get("feedback_view") != expected_views[condition]:
                raise ValueError(f"{condition.value} feedback view drift")
            compare_run_manifests(
                batch_roots[AdaMemCondition.MEM0_STATIC] / str(reports[AdaMemCondition.MEM0_STATIC]["replicates"][replicate - 1]["run_id"]) / "run_manifest.json",
                batch_roots[condition] / str(reports[condition]["replicates"][replicate - 1]["run_id"]) / "run_manifest.json",
            )
            receipt = _load(batch_roots[condition] / str(reports[condition]["replicates"][replicate - 1]["run_id"]) / "policy_receipt.json")
            if receipt.get("feedback_view") != expected_views[condition]:
                raise ValueError("policy receipt feedback view drift")
            if condition is not AdaMemCondition.MEM0_STATIC:
                request = _load(batch_roots[condition] / str(reports[condition]["replicates"][replicate - 1]["run_id"]) / "feedback_request.json")
                _assert_feedback_safe(request)
    return {
        "schema": "rsimem-adamem-mem0-refactor-matched-audit-v1",
        "accepted": True,
        "family_id": family_id,
        "conditions": {condition.value: reports[condition] for condition in expected_conditions},
        "surface_policy": policy.payload(),
    }


__all__ = ["audit_matched_family"]
