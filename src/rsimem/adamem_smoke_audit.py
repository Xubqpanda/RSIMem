"""Fail-closed audit for one B0/B1/B2 AdaMem smoke trio."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .adamem_experiment import AdaMemCondition
from .adamem_launcher import compare_run_manifests


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"unreadable AdaMem audit file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"AdaMem audit file must be an object: {path}")
    return value


def _complete_usage(run_root: Path) -> bool:
    paths = sorted(run_root.glob("*/sequence_results.json"))
    if len(paths) != 2:
        return False
    for path in paths:
        episodes = _load(path).get("episodes")
        if not isinstance(episodes, list) or not episodes:
            return False
        for episode in episodes:
            usage = episode.get("token_usage") if isinstance(episode, Mapping) else None
            if not isinstance(usage, Mapping) or usage.get("model_usage_complete") is not True:
                return False
    return True


def _assert_updated_policy_is_used(run_root: Path, receipt: Mapping[str, object]) -> None:
    """Prove B2's activated policy reached a later Mem0 extraction boundary."""
    expected = receipt.get("candidate_policy_version")
    if not isinstance(expected, str) or not expected:
        raise ValueError("AdaMem updated receipt has no candidate policy version")
    suffix = run_root / "suffix"
    found = False
    for path in suffix.rglob("rsimem_semantic_operations.jsonl"):
        for raw in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(raw)
            except ValueError as exc:
                raise ValueError("AdaMem semantic operation evidence is malformed") from exc
            payload = event.get("payload") if isinstance(event, Mapping) else None
            if not isinstance(payload, Mapping) or payload.get("kind") != "policy_parameter":
                continue
            if payload.get("revision") != expected:
                raise ValueError("AdaMem suffix policy revision differs from receipt")
            provenance = payload.get("provenance_ref")
            if not isinstance(provenance, str) or not provenance.startswith("prompt-binding."):
                raise ValueError("AdaMem suffix policy evidence lacks binding provenance")
            found = True
    if not found:
        raise ValueError("AdaMem updated run lacks suffix policy evidence")


def audit_smoke_trio(run_roots: Mapping[AdaMemCondition, Path]) -> dict[str, object]:
    """Audit the minimum runnable comparison; never calculate a quality claim."""
    expected = set(AdaMemCondition)
    resolved = {AdaMemCondition(key): Path(value) for key, value in run_roots.items()}
    if set(resolved) != expected:
        raise ValueError("AdaMem smoke audit requires exactly B0, B1, and B2")
    manifests = {condition: root / "run_manifest.json" for condition, root in resolved.items()}
    receipts = {condition: _load(root / "policy_receipt.json") for condition, root in resolved.items()}
    bindings = {condition: _load(root / "policy_binding.json") for condition, root in resolved.items()}
    if any(not _complete_usage(root) for root in resolved.values()):
        raise ValueError("AdaMem smoke trio has incomplete model usage")
    differences = {
        condition.value: compare_run_manifests(manifests[AdaMemCondition.MEM0_STATIC], path)
        for condition, path in manifests.items()
        if condition is not AdaMemCondition.MEM0_STATIC
    }
    b0, b1, b2 = (receipts[condition] for condition in AdaMemCondition)
    if (
        b0.get("outcome") != "static"
        or b1.get("feedback_view") != "terminal"
        or b2.get("feedback_view") != "full_trajectory"
    ):
        raise ValueError("AdaMem feedback-view receipt differs from condition")
    if bindings[AdaMemCondition.MEM0_STATIC].get("adamem_policy_applied") is not False:
        raise ValueError("B0 must retain Mem0Static binding")
    if b1.get("outcome") == "no_update" and bindings[AdaMemCondition.ADAMEM_TERMINAL].get("adamem_policy_applied") is not False:
        raise ValueError("B1 no-update must retain Mem0Static binding")
    if b2.get("outcome") == "updated" and bindings[AdaMemCondition.ADAMEM_FULL_TRAJECTORY].get("adamem_policy_applied") is not True:
        raise ValueError("B2 update lacks AdaMem extraction binding")
    if b2.get("outcome") == "updated":
        _assert_updated_policy_is_used(
            resolved[AdaMemCondition.ADAMEM_FULL_TRAJECTORY], b2
        )
    return {
        "schema": "rsimem-adamem-smoke-audit-v1",
        "accepted": True,
        "run_ids": {condition.value: _load(path).get("run_id") for condition, path in manifests.items()},
        "manifest_differences": differences,
        "outcomes": {condition.value: receipts[condition].get("outcome") for condition in AdaMemCondition},
    }


__all__ = ["audit_smoke_trio"]
