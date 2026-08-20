"""Self-Evolve-Tasks-V2 Layer 3 trigger-specific diagnostics (§14.3).

Most Layer 3 metrics require signals the runner does not yet collect
(first-use latency per rule, repeat-violation detection, reward sensitivity
curves). This module establishes the schema and computes what can be derived
from existing bucket summaries; the rest carries a `needs_runner_collection`
basis so Phase C/D work knows what to plumb.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class V2Layer3Metrics(BaseModel):
    """Trigger-specific diagnostics per family (§14.3)."""

    family_id: str
    primary_trigger: str

    # explicit_instruction
    adoption_latency: float | None = None
    first_use_correctness: float | None = None

    # explicit_correction
    correction_retention: float | None = None
    repeat_violation_rate: float | None = None

    # conflict_update
    overwrite_speed: float | None = None
    overwrite_completeness: float | None = None
    stale_leak_penalty: float | None = None

    # repeated_cases_induction
    sample_efficiency: float | None = None
    induction_quality: float | None = None
    transfer_without_keyword_overlap: float | None = None

    # failure_reflection
    reflection_to_behavior_conversion: float | None = None
    reflection_reuse_rate: float | None = None

    # environment_feedback
    reward_sensitivity: float | None = None
    convergence_quality: float | None = None
    policy_stability: float | None = None

    basis: dict[str, str] = Field(default_factory=dict)


def _bs(bucket_summary: dict[str, Any], bucket: str, key: str) -> float | None:
    block = bucket_summary.get(bucket) or {}
    v = block.get(key)
    return float(v) if v is not None else None


def compute_layer3(
    family_id: str,
    primary_trigger: str,
    with_family: dict[str, Any],
    with_block: dict[str, Any],
) -> V2Layer3Metrics:
    bs = with_family.get("bucket_summary") or {}
    sig = with_block.get("benchmark_signal") or {}
    basis: dict[str, str] = {}

    m = V2Layer3Metrics(family_id=family_id, primary_trigger=primary_trigger)

    if primary_trigger == "conflict_update":
        upd = sig.get("avg_update_correctness")
        if upd is not None:
            m.overwrite_completeness = float(upd)
            basis["overwrite_completeness"] = "runner_emitted_avg_update_correctness"
        else:
            basis["overwrite_completeness"] = "needs_runner_collection"
        basis["overwrite_speed"] = "needs_runner_collection"
        basis["stale_leak_penalty"] = "needs_runner_collection"

    elif primary_trigger == "explicit_instruction":
        eval_score = _bs(bs, "eval_near", "avg_task_score") or _bs(bs, "evaluation", "avg_task_score")
        if eval_score is not None:
            m.first_use_correctness = eval_score
            basis["first_use_correctness"] = "proxy_eval_near_avg_task_score"
        else:
            basis["first_use_correctness"] = "needs_runner_collection"
        basis["adoption_latency"] = "needs_runner_collection"

    elif primary_trigger == "explicit_correction":
        retention = sig.get("avg_retention_horizon")
        if retention is not None:
            m.correction_retention = float(retention)
            basis["correction_retention"] = "runner_emitted_avg_retention_horizon"
        else:
            basis["correction_retention"] = "needs_runner_collection"
        basis["repeat_violation_rate"] = "needs_runner_collection"

    elif primary_trigger == "repeated_cases_induction":
        near = _bs(bs, "eval_near", "avg_task_score")
        far = _bs(bs, "eval_far", "avg_task_score")
        if near is not None and far is not None:
            # Weak proxy: if far ≈ near, the induction transferred without keyword overlap.
            m.transfer_without_keyword_overlap = max(0.0, min(1.0, far / near)) if near > 0 else 0.0
            basis["transfer_without_keyword_overlap"] = "proxy_far_over_near"
        else:
            basis["transfer_without_keyword_overlap"] = "needs_runner_collection"

        # sample_efficiency: eval_near score / number of learn episodes used.
        # Higher = learned with fewer examples.
        learn_eps = _bs(bs, "learn", "episode_count")
        eval_near_s = _bs(bs, "eval_near", "avg_task_score") or _bs(bs, "evaluation", "avg_task_score")
        if learn_eps and learn_eps > 0 and eval_near_s is not None:
            m.sample_efficiency = min(1.0, eval_near_s / learn_eps)
            basis["sample_efficiency"] = "proxy_eval_score_over_learn_count"
        else:
            basis["sample_efficiency"] = "needs_runner_collection"

        # induction_quality: far transfer / baseline delta (how much of the
        # lift survives surface changes). Higher = the induced rule is
        # truly abstract.
        baseline = _bs(bs, "baseline", "avg_task_score") or _bs(bs, "cold", "avg_task_score")
        if far is not None and baseline is not None:
            m.induction_quality = max(0.0, far - baseline)
            basis["induction_quality"] = "proxy_far_minus_baseline"
        else:
            basis["induction_quality"] = "needs_runner_collection"

    elif primary_trigger == "failure_reflection":
        refl = _bs(bs, "reflection", "avg_task_score")
        eval_near = _bs(bs, "eval_near", "avg_task_score") or _bs(bs, "evaluation", "avg_task_score")
        if refl is not None and eval_near is not None:
            m.reflection_to_behavior_conversion = max(0.0, eval_near - refl)
            basis["reflection_to_behavior_conversion"] = "proxy_eval_minus_reflection"
        else:
            basis["reflection_to_behavior_conversion"] = "needs_runner_collection"

        # reflection_reuse_rate: retrieval_before_success_rate in eval buckets
        # after reflection fired — measures whether the reflected rule gets
        # actually retrieved later.
        near_r = _bs(bs, "eval_near", "retrieval_before_success_rate")
        far_r = _bs(bs, "eval_far", "retrieval_before_success_rate")
        flat_r = _bs(bs, "evaluation", "retrieval_before_success_rate")
        parts = [x for x in (near_r, far_r, flat_r) if x is not None]
        if parts:
            m.reflection_reuse_rate = sum(parts) / len(parts)
            basis["reflection_reuse_rate"] = "proxy_avg_eval_retrieval_before_success"
        else:
            basis["reflection_reuse_rate"] = "needs_runner_collection"

    elif primary_trigger == "environment_feedback":
        for k in ("reward_sensitivity", "convergence_quality", "policy_stability"):
            basis[k] = "needs_runner_collection"

    m.basis = basis
    return m
