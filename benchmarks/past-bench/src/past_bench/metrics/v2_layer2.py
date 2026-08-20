"""Self-Evolve-Tasks-V2 Layer 2 ability-specific metrics (§14.2).

Each ability has 3–5 metrics that diagnose whether the intended capability
actually evolved. Where the runner already emits the signal, this module maps
it through; where it does not, the field is `None` with a `basis` tag so Phase
C/D work knows what still needs collection.

Runner-emitted fields we harvest from `sequence_comparison.json`:

| Runner key                       | Mapped metric                                    |
| -------------------------------- | ------------------------------------------------ |
| `avg_write_precision`            | `memory_ability.write_precision`                 |
| `avg_recall_accuracy`            | `memory_ability.recall_accuracy`                 |
| `avg_update_correctness`         | `update_ability.overwrite_correctness`           |
| `avg_retention_horizon`          | memory/update retention proxy                    |
| `avg_memory_pollution_rate`      | memory/update pollution proxy                    |
| `bucket_summary.eval_near/far.skill_view_calls` | `procedural_ability.skill_reuse_rate` (proxy) |
| `bucket_summary.learn.skill_create_count`       | `procedural_ability.skill_creation_rate`      |
| `bucket_summary.eval_*.retrieval_before_success_rate` | memory/IG proactive retrieval proxy |

Everything else is `None` with a `basis` like `"needs_runner_collection"`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class V2Layer2Metrics(BaseModel):
    """Ability-specific diagnostics per family (§14.2)."""

    family_id: str
    primary_ability: str

    # memory_ability
    write_precision: float | None = None
    recall_accuracy: float | None = None
    overwrite_correctness: float | None = None
    retention_horizon: float | None = None
    memory_pollution_rate: float | None = None

    # memory / information gathering retrieval diagnostics
    proactive_recall_rate: float | None = None
    recall_grounding_accuracy: float | None = None
    repeat_avoidance: float | None = None
    over_storage_penalty: float | None = None

    # procedural_ability
    skill_creation_rate: float | None = None
    skill_reuse_rate: float | None = None
    patch_correctness: float | None = None
    transfer_gain: float | None = None
    overfit_penalty: float | None = None

    # proactive_information_gathering
    proactive_retrieval_rate: float | None = None
    retrieval_precision: float | None = None
    substrate_selection_accuracy: float | None = None

    # update_ability
    preserve_correctness: float | None = None
    stale_leak_penalty: float | None = None

    basis: dict[str, str] = Field(default_factory=dict)


def _bs(bucket_summary: dict[str, Any], bucket: str, key: str) -> float | None:
    block = bucket_summary.get(bucket) or {}
    v = block.get(key)
    return float(v) if v is not None else None


def _benchmark_signal(with_block: dict[str, Any]) -> dict[str, Any]:
    return with_block.get("benchmark_signal") or {}


def compute_layer2(
    family_id: str,
    primary_ability: str,
    with_family: dict[str, Any],
    with_block: dict[str, Any],
) -> V2Layer2Metrics:
    """Compute Layer 2 metrics for one family.

    Parameters
    ----------
    with_family:
        `sequence_comparison.json['with_persistence']['family_summary'][fid]`
    with_block:
        `sequence_comparison.json['with_persistence']` (for benchmark_signal)
    """
    bs = with_family.get("bucket_summary") or {}
    sig = _benchmark_signal(with_block)
    basis: dict[str, str] = {}

    m = V2Layer2Metrics(family_id=family_id, primary_ability=primary_ability)

    if primary_ability == "memory_ability":
        m.write_precision = _val(sig.get("avg_write_precision"), basis, "write_precision")
        m.recall_accuracy = _val(sig.get("avg_recall_accuracy"), basis, "recall_accuracy")
        m.retention_horizon = _val(
            sig.get("avg_retention_horizon"), basis, "retention_horizon"
        )
        m.memory_pollution_rate = _val(
            sig.get("avg_memory_pollution_rate"), basis, "memory_pollution_rate"
        )
        near = _bs(bs, "eval_near", "retrieval_before_success_rate")
        far = _bs(bs, "eval_far", "retrieval_before_success_rate")
        flat = _bs(bs, "evaluation", "retrieval_before_success_rate")
        if near is not None or far is not None:
            parts = [x for x in (near, far) if x is not None]
            m.proactive_recall_rate = sum(parts) / len(parts)
            basis["proactive_recall_rate"] = "measured_eval_near_far"
        elif flat is not None:
            m.proactive_recall_rate = flat
            basis["proactive_recall_rate"] = "measured_flat_eval"
        else:
            basis["proactive_recall_rate"] = "no_retrieval_signal"

        # recall_grounding_accuracy: proxy from artifact_quality_score in eval
        aq = _bs(bs, "eval_near", "avg_artifact_quality_score") or _bs(bs, "evaluation", "avg_artifact_quality_score")
        if aq is not None:
            m.recall_grounding_accuracy = aq
            basis["recall_grounding_accuracy"] = "proxy_avg_artifact_quality_score"
        else:
            basis["recall_grounding_accuracy"] = "needs_runner_collection"

        # over_storage_penalty: ratio of memory writes in learn/reflection to
        # total (high = agent over-stored episodic content as durable memory)
        learn_mw = _bs(bs, "learn", "memory_write_count") or 0
        eval_mw = _bs(bs, "eval_near", "memory_write_count") or _bs(bs, "evaluation", "memory_write_count") or 0
        total_mw = learn_mw + eval_mw
        if total_mw > 0:
            m.over_storage_penalty = min(1.0, learn_mw / total_mw)
            basis["over_storage_penalty"] = "ratio_learn_to_total_memory_writes"
        else:
            basis["over_storage_penalty"] = "no_memory_writes"

        # repeat_avoidance: 1 - rate of duplicate retrievals across episodes.
        # Requires trace-level counting; best effort via artifact_reuse_rate.
        reuse = _bs(bs, "evaluation", "artifact_reuse_rate") or _bs(bs, "eval_near", "artifact_reuse_rate")
        if reuse is not None:
            m.repeat_avoidance = reuse
            basis["repeat_avoidance"] = "proxy_artifact_reuse_rate"
        else:
            basis["repeat_avoidance"] = "needs_runner_collection"

    elif primary_ability == "procedural_ability":
        learn_skills = _bs(bs, "learn", "skill_create_count")
        learn_episodes = _bs(bs, "learn", "episode_count")
        if learn_skills is not None and learn_episodes and learn_episodes > 0:
            m.skill_creation_rate = learn_skills / learn_episodes
            basis["skill_creation_rate"] = "measured"
        else:
            basis["skill_creation_rate"] = "no_learn_skill_data"

        near = _bs(bs, "eval_near", "avg_task_score")
        far = _bs(bs, "eval_far", "avg_task_score")
        if near is not None and far is not None:
            m.transfer_gain = far - near
            basis["transfer_gain"] = "measured"
        else:
            basis["transfer_gain"] = "no_near_far_split"

        eval_retrieval = _bs(bs, "evaluation", "retrieval_before_success_rate")
        if eval_retrieval is not None:
            m.skill_reuse_rate = eval_retrieval
            basis["skill_reuse_rate"] = "proxy_retrieval_before_success"

        # patch_correctness: for PC02 (sop_patch), use update_correctness from
        # the runner's benchmark_signal if present
        upd = sig.get("avg_update_correctness")
        if upd is not None:
            m.patch_correctness = float(upd)
            basis["patch_correctness"] = "runner_emitted_avg_update_correctness"
        else:
            basis["patch_correctness"] = "needs_runner_collection"

        # overfit_penalty: eval_near >> eval_far means agent overfit the learn cases
        near = _bs(bs, "eval_near", "avg_task_score")
        far = _bs(bs, "eval_far", "avg_task_score")
        if near is not None and far is not None and near > 0:
            m.overfit_penalty = max(0.0, (near - far) / near)
            basis["overfit_penalty"] = "proxy_near_far_gap"
        else:
            basis["overfit_penalty"] = "no_near_far_split"

    elif primary_ability == "proactive_information_gathering":
        retrieval = _bs(bs, "evaluation", "retrieval_before_success_rate")
        if retrieval is not None:
            m.proactive_retrieval_rate = retrieval
            basis["proactive_retrieval_rate"] = "proxy_retrieval_before_success"
        else:
            basis["proactive_retrieval_rate"] = "needs_runner_collection"
        for k in ("retrieval_precision", "substrate_selection_accuracy"):
            basis[k] = "needs_runner_collection"

    elif primary_ability == "update_ability":
        upd = sig.get("avg_update_correctness")
        if upd is not None:
            m.overwrite_correctness = float(upd)
            m.patch_correctness = float(upd)
            basis["overwrite_correctness"] = "runner_emitted_avg_update_correctness"
            basis["patch_correctness"] = "runner_emitted_avg_update_correctness"
        else:
            basis["overwrite_correctness"] = "needs_runner_collection"
            basis["patch_correctness"] = "needs_runner_collection"

        poll = sig.get("avg_memory_pollution_rate")
        if poll is not None:
            m.memory_pollution_rate = float(poll)
            basis["memory_pollution_rate"] = "runner_emitted"
        else:
            basis["memory_pollution_rate"] = "needs_runner_collection"
        for k in ("preserve_correctness", "stale_leak_penalty"):
            basis[k] = "needs_runner_collection"

    m.basis = basis
    return m


def _val(v: Any, basis: dict[str, str], key: str) -> float | None:
    if v is None:
        basis[key] = "needs_runner_collection"
        return None
    basis[key] = "runner_emitted"
    return float(v)
