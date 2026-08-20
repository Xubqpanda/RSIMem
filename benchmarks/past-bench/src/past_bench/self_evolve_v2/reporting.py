"""Evaluation contract for the self-evolve v2 benchmark."""

from __future__ import annotations

UNIVERSAL_METRIC_NAMES = (
    "BaselineHeadroom",
    "OutcomeDelta",
    "AblationLift",
    "AutonomousReuseRate",
    "TransferRobustness",
    "ShortcutResistance",
    "MechanismConfidence",
    "FamilyEvolveScore",
)

ABILITY_METRIC_NAMES = {
    "memory_ability": (
        "write_precision",
        "recall_accuracy",
        "retention_horizon",
        "proactive_recall_rate",
        "recall_grounding_accuracy",
        "over_storage_penalty",
    ),
    "procedural_ability": (
        "skill_creation_rate",
        "skill_reuse_rate",
        "transfer_gain",
        "overfit_penalty",
    ),
    "proactive_information_gathering": (
        "proactive_retrieval_rate",
        "retrieval_precision",
        "substrate_selection_accuracy",
    ),
    "update_ability": (
        "overwrite_correctness",
        "preserve_correctness",
        "stale_leak_penalty",
        "memory_pollution_rate",
        "patch_correctness",
    ),
}

TRIGGER_DIAGNOSTIC_NAMES = {
    "explicit_instruction": ("adoption_latency", "first_use_correctness"),
    "explicit_correction": ("correction_retention", "repeat_violation_rate"),
    "conflict_update": (
        "overwrite_speed",
        "overwrite_completeness",
        "stale_leak_penalty",
    ),
    "repeated_cases_induction": (
        "sample_efficiency",
        "induction_quality",
        "transfer_without_keyword_overlap",
    ),
    "failure_reflection": (
        "reflection_to_behavior_conversion",
        "reflection_reuse_rate",
    ),
    "environment_feedback": (
        "reward_sensitivity",
        "convergence_quality",
        "policy_stability",
    ),
}
