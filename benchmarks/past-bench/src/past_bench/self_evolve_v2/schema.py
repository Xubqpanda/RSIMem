"""Schema for the self-evolve v2 benchmark framework."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FRAMEWORK_ROOT_NAME = "self-evolve-tasks-v2"
CONFIG_ROOT_NAME = "self_evolve_v2"

ABILITY_DIMENSIONS = {
    "memory_ability",
    "proactive_information_gathering",
    "procedural_ability",
    "update_ability",
}

PHASE1_ABILITIES = set(ABILITY_DIMENSIONS)
PHASE2_ABILITIES = ABILITY_DIMENSIONS - PHASE1_ABILITIES

CANONICAL_TRIGGER_CLASSES = {
    "explicit_instruction",
    "explicit_correction",
    "conflict_update",
    "repeated_cases_induction",
    "failure_reflection",
    "environment_feedback",
    "durable_rule_hint",
    "embedded_policy_exposure",
    "passive_policy_exposure",
    "weak_conflict_update",
}

TRIGGER_CATEGORIES = {
    "direct_explicit_instruction",
    "direct_explicit_save_instruction",
    "explicit_correction",
    "conflict_update",
    "embedded_durable_instruction",
    "embedded_update_notice",
    "repeated_cases_induction",
    "multi_turn_reinforcement",
    "failure_reflection",
    "post_hoc_review_extraction",
    "environment_feedback",
    "embedded_policy",
    "passive_document_exposure",
    "weak_conflict_update",
    "weak_explicit_future_use",
    "delayed_outcome_feedback",
    "mixed_signal_arbitration",
}
TRIGGER_STRENGTHS = {"T0", "T1", "T2", "T3"}
TRIGGER_VISIBILITY_PATHS = {
    "direct_user_message",
    "direct_system_or_policy_message",
    "explicit_follow_up_correction",
    "required_document_or_note_reading",
    "multi_turn_reinforcement_single_session",
    "repeated_evidence_across_learn_episodes",
    "reflection_or_review_time_extraction",
    "structured_scope_metadata",
    "delayed_outcome_signal",
}
BENCHMARK_ROLES = {"canonical_anchor", "diagnostic_sibling"}
COMPATIBILITY_GATES = {"strict", "diagnostic"}

EXPECTED_SUBSTRATES = {
    "memory",
    "skill",
    "session_search",
    "planner_policy",
    "prompt_rule",
    "tool_policy",
    "collaboration_playbook",
    "mixed",
}

FAMILY_LENGTH_TIERS = {"tier1", "tier2", "tier3"}
TRANSFER_DISTANCES = {"none", "near", "far", "near_far"}
OVERWRITE_MODES = {"not_required", "required"}
NOISE_PROFILES = {"low", "medium", "high"}
STAGE_KEYS = {
    "cold",
    "learn",
    "learn_a",
    "learn_b",
    "learn_c",
    "update_or_stabilize",
    "attempt",
    "reflection",
    "confirmation",
    "training",
    "eval_near",
    "eval_far",
    "control",
}
BUCKET_KEYS = {
    "cold",
    "learn",
    "update_or_stabilize",
    "attempt",
    "reflection",
    "confirmation",
    "training",
    "eval_near",
    "eval_far",
    "control",
}
TEACHING_BUCKETS = {
    "learn",
    "update_or_stabilize",
    "attempt",
    "reflection",
    "training",
}
CONTROL_KINDS = {
    "no_persistence",
    "shortcut",
    "wrong_mechanism",
    "shared_cold_run",
    "reflection_off",
    "carrier_ablation",
    "noise_gap",
    "domain_shift",
    "stale_procedure",
    "surface_memorization",
    "too_few_examples",
}
FAMILY_STATUSES = {"skeleton", "task_in_progress", "authored", "out_of_scope", "v2_architecture_aligned"}
IMPLEMENTATION_PRIORITIES = {"phase1", "phase2"}

REQUIRED_CONTROL_SET = {"no_persistence", "shortcut", "wrong_mechanism"}
REQUIRED_BUCKETS = {"cold", "eval_near", "eval_far", "control"}
TRIGGER_INJECTION_STAGES = {
    "learn",
    "learn_a",
    "learn_b",
    "learn_c",
    "update_or_stabilize",
    "attempt",
    "reflection",
    "training",
}
TRIGGER_REMOVAL_STAGES = {"eval_near", "eval_far"}


class FamilyMetadata(BaseModel):
    """family.yaml contract for `self-evolve-tasks-v2/<ability>/<family>/`."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    family_id: str
    title: str
    ability_dir: str
    primary_ability: str
    secondary_abilities: list[str] = Field(default_factory=list)
    primary_trigger: str
    trigger_category: str | None = None
    trigger_strength: str | None = None
    trigger_visibility_paths: list[str] = Field(default_factory=list)
    benchmark_role: str | None = None
    compatibility_gate: str | None = None
    trigger_class: str | None = None
    secondary_triggers: list[str] = Field(default_factory=list)
    optional_supporting_trigger: str | None = None
    supporting_stage: str | None = None
    supporting_signal: str | None = None
    supporting_signals: list[str] = Field(default_factory=list)
    distractor_signals: list[str] = Field(default_factory=list)
    trigger_injection_plan: dict[str, str] | str = Field(default_factory=dict)
    primary_trigger_stages: list[str] = Field(default_factory=list)
    trigger_removal_plan: dict[str, str] | str = Field(default_factory=dict)
    trigger_mixing_rule: str = ""
    sibling_comparison_plan: str = ""
    expected_substrate: str
    autonomy_required: bool = True
    evaluation_requires_reuse: bool = True
    transfer_distance: str
    overwrite_mode: str = "not_required"
    noise_profile: str = "medium"
    family_length_tier: str
    implementation_priority: str = "phase1"
    instances_per_bucket: dict[str, int] = Field(default_factory=dict)
    total_episodes: int = 0
    control_set: list[str] = Field(default_factory=list)
    status: str = "skeleton"
    what_is_learned: str = ""
    what_persists: str = ""
    what_counts_as_reuse: str = ""
    false_positive_pattern: str = ""
    must_demonstrate: list[str] = Field(default_factory=list)
    recommended_mock_services: list[str] = Field(default_factory=list)
    phase1_canonical_sibling: str | None = None
    future_sibling_roadmap: list[str] | str = Field(default_factory=list)
    cold_design: str = ""
    expected_cold_result: str = ""
    control_design: str = ""
    expected_control_result: str = ""
    authoring_notes: str | None = None

    _source_path: Path | None = None

    @field_validator("primary_ability")
    @classmethod
    def _check_primary_ability(cls, value: str) -> str:
        if value not in ABILITY_DIMENSIONS:
            raise ValueError(f"unknown primary_ability: {value}")
        return value

    @field_validator("expected_substrate")
    @classmethod
    def _check_expected_substrate(cls, value: str) -> str:
        if value not in EXPECTED_SUBSTRATES:
            raise ValueError(f"unknown expected_substrate: {value}")
        return value

    @field_validator("trigger_category")
    @classmethod
    def _check_trigger_category(cls, value: str | None) -> str | None:
        if value is not None and value not in TRIGGER_CATEGORIES:
            raise ValueError(f"unknown trigger_category: {value}")
        return value

    @field_validator("trigger_strength")
    @classmethod
    def _check_trigger_strength(cls, value: str | None) -> str | None:
        if value is not None and value not in TRIGGER_STRENGTHS:
            raise ValueError(f"unknown trigger_strength: {value}")
        return value

    @field_validator("benchmark_role")
    @classmethod
    def _check_benchmark_role(cls, value: str | None) -> str | None:
        if value is not None and value not in BENCHMARK_ROLES:
            raise ValueError(f"unknown benchmark_role: {value}")
        return value

    @field_validator("compatibility_gate")
    @classmethod
    def _check_compatibility_gate(cls, value: str | None) -> str | None:
        if value is not None and value not in COMPATIBILITY_GATES:
            raise ValueError(f"unknown compatibility_gate: {value}")
        return value

    @field_validator("trigger_class")
    @classmethod
    def _check_trigger_class(cls, value: str | None) -> str | None:
        if value is not None and value not in CANONICAL_TRIGGER_CLASSES:
            raise ValueError(f"unknown trigger_class: {value}")
        return value

    @field_validator("supporting_stage")
    @classmethod
    def _check_supporting_stage(cls, value: str | None) -> str | None:
        if value is not None and value not in TRIGGER_INJECTION_STAGES:
            raise ValueError(f"unknown supporting_stage: {value}")
        return value

    @field_validator("family_length_tier")
    @classmethod
    def _check_family_length_tier(cls, value: str) -> str:
        if value not in FAMILY_LENGTH_TIERS:
            raise ValueError(f"unknown family_length_tier: {value}")
        return value

    @field_validator("implementation_priority")
    @classmethod
    def _check_implementation_priority(cls, value: str) -> str:
        if value not in IMPLEMENTATION_PRIORITIES:
            raise ValueError(f"unknown implementation_priority: {value}")
        return value

    @field_validator("transfer_distance")
    @classmethod
    def _check_transfer_distance(cls, value: str) -> str:
        if value not in TRANSFER_DISTANCES:
            raise ValueError(f"unknown transfer_distance: {value}")
        return value

    @field_validator("overwrite_mode")
    @classmethod
    def _check_overwrite_mode(cls, value: str) -> str:
        if value not in OVERWRITE_MODES:
            raise ValueError(f"unknown overwrite_mode: {value}")
        return value

    @field_validator("noise_profile")
    @classmethod
    def _check_noise_profile(cls, value: str) -> str:
        if value not in NOISE_PROFILES:
            raise ValueError(f"unknown noise_profile: {value}")
        return value

    @field_validator("status")
    @classmethod
    def _check_status(cls, value: str) -> str:
        if value not in FAMILY_STATUSES:
            raise ValueError(f"unknown status: {value}")
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> "FamilyMetadata":
        if self.ability_dir != self.primary_ability:
            raise ValueError(
                f"ability_dir={self.ability_dir} does not match primary_ability={self.primary_ability}"
            )
        if len(self.secondary_abilities) > 2:
            raise ValueError("secondary_abilities may contain at most two items")
        if self.trigger_class and self.primary_trigger != self.trigger_class:
            raise ValueError(
                f"{self.family_id}: primary_trigger must match trigger_class when trigger_class is set"
            )
        bad_visibility_paths = set(self.trigger_visibility_paths) - TRIGGER_VISIBILITY_PATHS
        if bad_visibility_paths:
            raise ValueError(
                f"{self.family_id}: unknown trigger_visibility_paths {sorted(bad_visibility_paths)}"
            )

        expected_priority = "phase1" if self.primary_ability in PHASE1_ABILITIES else "phase2"
        if self.implementation_priority != expected_priority:
            raise ValueError(
                f"{self.family_id}: implementation_priority must be {expected_priority}"
            )

        bucket_keys = set(self.instances_per_bucket)
        unknown_buckets = bucket_keys - BUCKET_KEYS
        if unknown_buckets:
            raise ValueError(f"{self.family_id}: unknown bucket keys {sorted(unknown_buckets)}")
        if self.total_episodes and self.total_episodes != sum(self.instances_per_bucket.values()):
            raise ValueError(
                f"{self.family_id}: total_episodes does not match instances_per_bucket"
            )

        control_set = set(self.control_set)
        unknown_controls = control_set - CONTROL_KINDS
        if unknown_controls:
            raise ValueError(
                f"{self.family_id}: unknown control kinds {sorted(unknown_controls)}"
            )
        return self
        if not REQUIRED_CONTROL_SET.issubset(control_set):
            raise ValueError(
                f"{self.family_id}: control_set must include {sorted(REQUIRED_CONTROL_SET)}"
            )

        injection_keys = set(self.trigger_injection_plan)
        if not injection_keys:
            raise ValueError(f"{self.family_id}: trigger_injection_plan must not be empty")
        unknown_injection_keys = injection_keys - TRIGGER_INJECTION_STAGES
        if unknown_injection_keys:
            raise ValueError(
                f"{self.family_id}: unknown trigger_injection_plan stages {sorted(unknown_injection_keys)}"
            )
        if any(not text.strip() for text in self.trigger_injection_plan.values()):
            raise ValueError(f"{self.family_id}: trigger_injection_plan values must be non-empty")

        primary_trigger_stages = set(self.primary_trigger_stages)
        if not primary_trigger_stages:
            raise ValueError(f"{self.family_id}: primary_trigger_stages must not be empty")
        unknown_primary_trigger_stages = primary_trigger_stages - injection_keys
        if unknown_primary_trigger_stages:
            raise ValueError(
                f"{self.family_id}: primary_trigger_stages must be declared in trigger_injection_plan"
            )

        if self.optional_supporting_trigger and not self.supporting_stage:
            raise ValueError(
                f"{self.family_id}: supporting_stage is required when optional_supporting_trigger is set"
            )
        if (self.supporting_stage or self.supporting_signal) and not (
            self.optional_supporting_trigger or self.supporting_signal
        ):
            raise ValueError(
                f"{self.family_id}: supporting_stage/supporting_signal must describe a supporting trigger-like signal"
            )
        if self.supporting_stage and self.supporting_stage not in injection_keys:
            raise ValueError(
                f"{self.family_id}: supporting_stage must also appear in trigger_injection_plan"
            )

        removal_keys = set(self.trigger_removal_plan)
        if removal_keys != TRIGGER_REMOVAL_STAGES:
            raise ValueError(
                f"{self.family_id}: trigger_removal_plan must define exactly {sorted(TRIGGER_REMOVAL_STAGES)}"
            )
        if any(not text.strip() for text in self.trigger_removal_plan.values()):
            raise ValueError(f"{self.family_id}: trigger_removal_plan values must be non-empty")

        if not self.trigger_mixing_rule.strip():
            raise ValueError(f"{self.family_id}: trigger_mixing_rule must not be empty")
        if not self.sibling_comparison_plan.strip():
            raise ValueError(f"{self.family_id}: sibling_comparison_plan must not be empty")

        if self.primary_ability != "trigger_and_arbitration":
            if self.distractor_signals:
                raise ValueError(
                    f"{self.family_id}: distractor_signals are reserved for trigger_and_arbitration families"
                )
            if len(self.supporting_signals) > 1:
                raise ValueError(
                    f"{self.family_id}: use optional_supporting_trigger/supporting_signal instead of multiple supporting_signals"
                )
        else:
            if not self.supporting_signals and self.optional_supporting_trigger is None:
                raise ValueError(
                    f"{self.family_id}: trigger_and_arbitration families should mark supporting_signals or optional_supporting_trigger"
                )

        if self.trigger_class == "failure_reflection":
            if self.instances_per_bucket.get("attempt", 0) <= 0:
                raise ValueError(f"{self.family_id}: failure_reflection families need attempt")
            if self.instances_per_bucket.get("reflection", 0) <= 0:
                raise ValueError(
                    f"{self.family_id}: failure_reflection families need reflection"
                )

        if self.trigger_class == "environment_feedback":
            if self.instances_per_bucket.get("training", 0) <= 0:
                raise ValueError(
                    f"{self.family_id}: environment_feedback families need training"
                )
            if "training" not in primary_trigger_stages:
                raise ValueError(
                    f"{self.family_id}: environment_feedback families need training in primary_trigger_stages"
                )

        if self.overwrite_mode == "required":
            if self.instances_per_bucket.get("update_or_stabilize", 0) <= 0:
                raise ValueError(
                    f"{self.family_id}: overwrite families need update_or_stabilize"
                )

        if self.trigger_class == "explicit_correction":
            if self.instances_per_bucket.get("learn", 0) < 2 and self.instances_per_bucket.get(
                "update_or_stabilize", 0
            ) <= 0:
                raise ValueError(
                    f"{self.family_id}: explicit_correction families need learn_a + learn_b or update_or_stabilize"
                )
            if primary_trigger_stages.isdisjoint({"learn_b", "update_or_stabilize"}):
                raise ValueError(
                    f"{self.family_id}: explicit_correction should trigger in learn_b or update_or_stabilize"
                )

        if self.trigger_class == "conflict_update":
            if self.instances_per_bucket.get("learn", 0) <= 0:
                raise ValueError(f"{self.family_id}: conflict_update families need learn")
            if primary_trigger_stages.isdisjoint({"learn_b", "update_or_stabilize"}):
                raise ValueError(
                    f"{self.family_id}: conflict_update should trigger in learn_b or update_or_stabilize"
                )

        if self.trigger_class == "repeated_cases_induction":
            if self.instances_per_bucket.get("learn", 0) < 2:
                raise ValueError(
                    f"{self.family_id}: repeated_cases_induction families need at least two learn cases"
                )
            if len(primary_trigger_stages) < 2:
                raise ValueError(
                    f"{self.family_id}: repeated_cases_induction should span at least two primary_trigger_stages"
                )

        if self.trigger_class == "failure_reflection" and "reflection" not in primary_trigger_stages:
            raise ValueError(
                f"{self.family_id}: failure_reflection families need reflection in primary_trigger_stages"
            )

        if self.benchmark_role == "canonical_anchor":
            if self.compatibility_gate != "strict":
                raise ValueError(
                    f"{self.family_id}: canonical anchors must use the strict compatibility gate"
                )
            if self.trigger_strength not in {"T2", "T3"}:
                raise ValueError(
                    f"{self.family_id}: canonical anchors must use T2/T3 trigger strength"
                )
        if self.benchmark_role == "diagnostic_sibling" and self.compatibility_gate != "diagnostic":
            raise ValueError(
                f"{self.family_id}: diagnostic siblings must use the diagnostic compatibility gate"
            )

        if self.status == "authored":
            authored_required_text = {
                "cold_design": self.cold_design,
                "expected_cold_result": self.expected_cold_result,
                "control_design": self.control_design,
                "expected_control_result": self.expected_control_result,
            }
            missing_authored_text = [
                name for name, value in authored_required_text.items() if not value.strip()
            ]
            if missing_authored_text:
                raise ValueError(
                    f"{self.family_id}: authored families must define {sorted(missing_authored_text)}"
                )
            if self.benchmark_role is None:
                raise ValueError(f"{self.family_id}: authored families must define benchmark_role")
            if self.compatibility_gate is None:
                raise ValueError(
                    f"{self.family_id}: authored families must define compatibility_gate"
                )
            if self.trigger_category is None:
                raise ValueError(
                    f"{self.family_id}: authored families must define trigger_category"
                )
            if self.trigger_strength is None:
                raise ValueError(
                    f"{self.family_id}: authored families must define trigger_strength"
                )
            if not self.trigger_visibility_paths:
                raise ValueError(
                    f"{self.family_id}: authored families must define trigger_visibility_paths"
                )
            if not (self.phase1_canonical_sibling or "").strip():
                raise ValueError(
                    f"{self.family_id}: authored families must define phase1_canonical_sibling"
                )
            if not self.future_sibling_roadmap:
                raise ValueError(
                    f"{self.family_id}: authored families must define future_sibling_roadmap"
                )

        return self

    @property
    def source_path(self) -> Path | None:
        return self._source_path

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FamilyMetadata":
        yaml_path = Path(path)
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        family = cls.model_validate(data)
        family._source_path = yaml_path.resolve()
        return family


def load_families(root: str | Path) -> list[FamilyMetadata]:
    """Load every family.yaml under the self-evolve v2 framework root."""

    root_path = Path(root)
    families: list[FamilyMetadata] = []
    for family_yaml in sorted(root_path.glob("*/*/family.yaml")):
        families.append(FamilyMetadata.from_yaml(family_yaml))
    return families
