from __future__ import annotations

from typing import Any

from past_bench.runner.self_evolve import summarize_sequence


FAMILY_META = {
    "F01_skill_bootstrap": ("skill", "skill"),
    "F02_skill_patch": ("skill", "skill"),
    "F03_memory_preference": ("memory", "memory"),
    "F04_session_recall": ("session_search", "session_search"),
    "F05_hidden_correlation": ("skill", "skill"),
    "F06_stale_conflict_update": ("memory", "memory"),
    "F07_failure_to_rule": ("skill", "skill"),
}


def _artifacts(signal: str, *, present: bool) -> dict[str, Any]:
    return {
        "memory_file_exists": present and signal in {"memory", "mixed", "session_search"},
        "user_file_exists": False,
        "skill_count": 1 if present and signal in {"skill", "mixed"} else 0,
    }


def _internal_tools(signal: str, *, writes: bool, reads: bool) -> dict[str, int]:
    return {
        "memory_calls": 1 if writes and signal in {"memory", "mixed"} else 0,
        "skill_manage_calls": 1 if writes and signal in {"skill", "mixed"} else 0,
        "session_search_calls": 1 if reads and signal in {"session_search", "mixed"} else 0,
        "memory_write_count": 1 if writes and signal in {"memory", "mixed"} else 0,
        "skill_create_count": 1 if writes and signal in {"skill", "mixed"} else 0,
        "skill_update_count": 1 if writes and signal in {"skill", "mixed"} else 0,
    }


def _retrieval(signal: str, *, reads: bool) -> dict[str, Any]:
    return {
        "used_expected_signal": reads,
        "retrieval_before_first_update": reads,
        "memory_read_count": 1 if reads and signal in {"memory", "mixed"} else 0,
        "session_search_count": 1 if reads and signal in {"session_search", "mixed"} else 0,
        "skill_read_count": 1 if reads and signal in {"skill", "mixed"} else 0,
    }


def _episode(
    family_id: str,
    *,
    bucket: str,
    task_score: float,
    mechanism_confidence: float,
    used_signal: bool,
    writes: bool,
    label: str,
    transfer_quality: float = 1.0,
    shortcut_resistance: float = 1.0,
) -> dict[str, Any]:
    mechanism, signal = FAMILY_META[family_id]
    return {
        "family_id": family_id,
        "label": label,
        "bucket": bucket,
        "mechanism": mechanism,
        "expected_persistence_signal": signal,
        "task_score": task_score,
        "passed": task_score >= 0.7,
        "tool_dispatch_count": 4,
        "token_usage": {"total_tokens": 200},
        "timing": {"wall_time_s": 1.0},
        "artifacts": _artifacts(signal, present=writes or used_signal),
        "internal_tools": _internal_tools(signal, writes=writes, reads=used_signal),
        "retrieval_signals": _retrieval(signal, reads=used_signal),
        "mechanism_scores": {
            "mechanism_confidence": mechanism_confidence,
            "artifact_quality_score": mechanism_confidence,
            "transfer_quality": transfer_quality,
            "shortcut_resistance": shortcut_resistance,
        },
    }


def synthesize_agent_summary(agent_name: str) -> dict[str, Any]:
    base_scores = {
        "F01_skill_bootstrap": 0.46,
        "F02_skill_patch": 0.44,
        "F03_memory_preference": 0.49,
        "F04_session_recall": 0.42,
        "F05_hidden_correlation": 0.40,
        "F06_stale_conflict_update": 0.47,
        "F07_failure_to_rule": 0.38,
    }

    agent_profiles = {
        "NoPersistenceAgent": {
            "eval": {family: base + 0.01 for family, base in base_scores.items()},
            "learn": 0.48,
            "mechanism_confidence": 0.0,
            "used_signal": False,
            "writes": False,
        },
        "WriteOnlyAgent": {
            "eval": {family: base + 0.03 for family, base in base_scores.items()},
            "learn": 0.55,
            "mechanism_confidence": 0.28,
            "used_signal": False,
            "writes": True,
        },
        "ReadOnlyCorrectPreseedAgent": {
            "eval": {family: base + 0.12 for family, base in base_scores.items()},
            "learn": 0.52,
            "mechanism_confidence": 0.58,
            "used_signal": True,
            "writes": False,
        },
        "RuleLearnerAgent": {
            "eval": {family: base + 0.26 for family, base in base_scores.items()},
            "learn": 0.66,
            "mechanism_confidence": 0.92,
            "used_signal": True,
            "writes": True,
        },
        "StaleFollowerAgent": {
            "eval": {
                **{family: base + 0.21 for family, base in base_scores.items()},
                "F02_skill_patch": 0.31,
                "F06_stale_conflict_update": 0.34,
            },
            "learn": 0.61,
            "mechanism_confidence": 0.72,
            "used_signal": True,
            "writes": True,
        },
        "ReflectionLearnerAgent": {
            "eval": {family: base + 0.24 for family, base in base_scores.items()},
            "learn": 0.64,
            "mechanism_confidence": 0.89,
            "used_signal": True,
            "writes": True,
            "family_confidence": {"F07_failure_to_rule": 0.97},
            "family_eval": {"F07_failure_to_rule": 0.79},
        },
    }

    profile = agent_profiles[agent_name]
    episodes: list[dict[str, Any]] = []
    for family_id, base_score in base_scores.items():
        confidence = profile.get("family_confidence", {}).get(family_id, profile["mechanism_confidence"])
        eval_score = profile.get("family_eval", {}).get(family_id, profile["eval"][family_id])
        episodes.extend(
            [
                _episode(
                    family_id,
                    bucket="baseline",
                    task_score=base_score,
                    mechanism_confidence=0.1 if profile["writes"] or profile["used_signal"] else 0.0,
                    used_signal=False,
                    writes=False,
                    label=f"{family_id}_baseline",
                    transfer_quality=0.0,
                    shortcut_resistance=0.5,
                ),
                _episode(
                    family_id,
                    bucket="learn",
                    task_score=profile["learn"],
                    mechanism_confidence=confidence,
                    used_signal=profile["used_signal"],
                    writes=profile["writes"],
                    label=f"{family_id}_learn_a",
                ),
                _episode(
                    family_id,
                    bucket="learn",
                    task_score=profile["learn"] - 0.02,
                    mechanism_confidence=confidence,
                    used_signal=profile["used_signal"],
                    writes=profile["writes"],
                    label=f"{family_id}_learn_b",
                ),
                _episode(
                    family_id,
                    bucket="evaluation",
                    task_score=eval_score,
                    mechanism_confidence=confidence,
                    used_signal=profile["used_signal"],
                    writes=profile["writes"],
                    label=f"{family_id}_eval_near",
                ),
                _episode(
                    family_id,
                    bucket="evaluation",
                    task_score=max(0.0, eval_score - 0.02),
                    mechanism_confidence=confidence,
                    used_signal=profile["used_signal"],
                    writes=profile["writes"],
                    label=f"{family_id}_eval_far",
                    transfer_quality=0.95,
                ),
            ]
        )

    return summarize_sequence(
        sequence_name="synthetic_self_evolve_v2",
        variant=agent_name,
        episodes=episodes,
    )
