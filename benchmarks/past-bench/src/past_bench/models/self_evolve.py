"""Sequence manifests for cross-task self-evolve evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── V2 ability-grouped schema (Self-Evolve-Tasks-V2 Design Architecture §4) ──

V2_ABILITIES = {
    "proactive_information_gathering",
    "memory_ability",
    "procedural_ability",
    "update_ability",
}

V2_TRIGGERS = {
    "durable_rule_hint",
    "embedded_policy_exposure",
    "explicit_instruction",
    "explicit_correction",
    "conflict_update",
    "repeated_cases_induction",
    "failure_reflection",
    "environment_feedback",
    "passive_policy_exposure",
    "weak_conflict_update",
}

V2_SUBSTRATES = {
    "memory",
    "skill",
    "session_search",
    "planner_policy",
    "prompt_rule",
    "tool_policy",
    "collaboration_playbook",
    "mixed",
}

V2_TIERS = {"tier1", "tier2", "tier3"}

V2_TRANSFER_DISTANCES = {"none", "near", "far", "near_far"}

V2_OVERWRITE_MODES = {"not_required", "required"}

V2_BUCKETS = {
    "cold",
    "learn",
    "update_or_stabilize",
    "attempt",
    "confirmation",
    "reflection",
    "training",
    "eval_near",
    "eval_far",
    "control",
}

V2_CONTROL_KINDS = {
    "no_persistence",
    "shortcut",
    "wrong_mechanism",
    "shared_cold_run",
    "reflection_off",
    "surface_memorization",
    "stale_procedure",
    "too_few_examples",
    "carrier_ablation",
    "noise_gap",
    "domain_shift",
}

V2_STATUSES = {"skeleton", "legacy_migrated", "authored", "v2_architecture_aligned", "out_of_scope"}
V2_HISTORY_MODES = {"continue", "fresh", "from_anchor"}


class V2LegacyEpisodeRef(BaseModel):
    """Cross-reference to a legacy self-evolve-tasks/ family."""

    legacy_family_id: str
    legacy_group: str
    legacy_path: str
    trigger: str
    rationale: str = ""
    episodes: list[str] = Field(default_factory=list)


class V2HistoryAnchor(BaseModel):
    """Named runtime snapshot saved after a specific episode."""

    name: str
    save_after: str


class V2HistoryBranch(BaseModel):
    """Group of episodes that share the same runtime-history behavior."""

    mode: str
    episodes: list[str] = Field(default_factory=list)
    anchor: str = ""

    @field_validator("mode")
    @classmethod
    def _check_history_mode(cls, v: str) -> str:
        if v not in V2_HISTORY_MODES:
            raise ValueError(f"history mode {v!r} not in {sorted(V2_HISTORY_MODES)}")
        return v

    @model_validator(mode="after")
    def _validate_branch(self) -> "V2HistoryBranch":
        if not self.episodes:
            raise ValueError("history branch episodes must not be empty")
        if self.mode == "from_anchor" and not self.anchor:
            raise ValueError("from_anchor branches must declare anchor")
        if self.mode != "from_anchor" and self.anchor:
            raise ValueError("only from_anchor branches may declare anchor")
        return self


class V2HistoryPlan(BaseModel):
    """Family-level runtime-history orchestration."""

    anchors: list[V2HistoryAnchor] = Field(default_factory=list)
    branches: list[V2HistoryBranch] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_uniqueness(self) -> "V2HistoryPlan":
        anchor_names = [anchor.name for anchor in self.anchors]
        if len(anchor_names) != len(set(anchor_names)):
            raise ValueError("history anchor names must be unique")

        save_after_ids = [anchor.save_after for anchor in self.anchors]
        if len(save_after_ids) != len(set(save_after_ids)):
            raise ValueError("history anchors must not share the same save_after episode")

        assigned: set[str] = set()
        for branch in self.branches:
            overlap = assigned.intersection(branch.episodes)
            if overlap:
                raise ValueError(
                    f"history branches must not overlap; duplicate episodes: {sorted(overlap)}"
                )
            assigned.update(branch.episodes)
        return self


class V2EpisodeOverride(BaseModel):
    """Optional family.yaml override for a single episode."""

    stage: str = ""
    mechanism: Literal["skill", "memory", "session_search", "mixed"] | str = "mixed"
    expected_persistence_signal: Literal["skill", "memory", "session_search", "mixed"] | str = ""
    persistence_allowed: bool | None = None
    preseed_artifacts_dir: str = ""
    initial_home_fixture_dir: str = ""


class V2FamilyMetadata(BaseModel):
    """family.yaml schema for self-evolve-tasks-v2/<ability>/<family>/.

    Implements the tuple from Self-Evolve-Tasks-V2 Design Architecture §4:
    <primary_ability, secondary_abilities, primary_trigger,
     expected_substrate, transfer_distance, control_set>.
    """

    family_id: str
    title: str
    ability_dir: str
    primary_ability: str
    secondary_abilities: list[str] = Field(default_factory=list)
    primary_trigger: str
    supporting_stage: str = ""
    supporting_signal: str | None = ""
    secondary_trigger: list[str] = Field(default_factory=list)
    expected_substrate: str
    autonomy_required: bool = True
    evaluation_requires_reuse: bool = True
    transfer_distance: str
    overwrite_mode: str = "not_required"
    noise_profile: str = "medium"
    family_length_tier: str
    instances_per_bucket: dict[str, int] = Field(default_factory=dict)
    total_episodes: int = 0
    control_set: list[str] = Field(default_factory=list)
    status: str = "skeleton"
    scope_note: str | None = None
    legacy_source: str | None = None
    legacy_episodes: list[str] = Field(default_factory=list)
    cross_ref_legacy_families: list[V2LegacyEpisodeRef] = Field(default_factory=list)

    # §4 explicit family declarations
    latent_rule_id: str = ""
    what_is_learned: str = ""
    what_persists: str = ""
    what_counts_as_reuse: str = ""
    false_positive_pattern: str = ""
    must_demonstrate: list[str] = Field(default_factory=list)

    # §7.2 trigger-placement enforcement inputs
    trigger_phrases: list[str] = Field(default_factory=list)
    banned_phrases: list[str] = Field(default_factory=list)
    stale_phrases: list[str] = Field(default_factory=list)

    # §10 cold design — optional calibration anchor; saturation validator uses it
    calibration_baseline_score: float | None = None
    saturation_ceiling: float = 0.85
    history_plan: V2HistoryPlan | None = None
    episode_overrides: dict[str, V2EpisodeOverride] = Field(default_factory=dict)

    _source_path: Path | None = None

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("primary_ability")
    @classmethod
    def _check_ability(cls, v: str) -> str:
        if v not in V2_ABILITIES:
            raise ValueError(f"primary_ability {v!r} not in {sorted(V2_ABILITIES)}")
        return v

    @field_validator("primary_trigger")
    @classmethod
    def _check_trigger(cls, v: str) -> str:
        if v not in V2_TRIGGERS:
            raise ValueError(f"primary_trigger {v!r} not in {sorted(V2_TRIGGERS)}")
        return v

    @field_validator("expected_substrate")
    @classmethod
    def _check_substrate(cls, v: str) -> str:
        if v not in V2_SUBSTRATES:
            raise ValueError(f"expected_substrate {v!r} not in {sorted(V2_SUBSTRATES)}")
        return v

    @field_validator("family_length_tier")
    @classmethod
    def _check_tier(cls, v: str) -> str:
        if v not in V2_TIERS:
            raise ValueError(f"family_length_tier {v!r} not in {sorted(V2_TIERS)}")
        return v

    @field_validator("transfer_distance")
    @classmethod
    def _check_transfer(cls, v: str) -> str:
        if v not in V2_TRANSFER_DISTANCES:
            raise ValueError(
                f"transfer_distance {v!r} not in {sorted(V2_TRANSFER_DISTANCES)}"
            )
        return v

    @field_validator("overwrite_mode")
    @classmethod
    def _check_overwrite(cls, v: str) -> str:
        if v not in V2_OVERWRITE_MODES:
            raise ValueError(f"overwrite_mode {v!r} not in {sorted(V2_OVERWRITE_MODES)}")
        return v

    @field_validator("status")
    @classmethod
    def _check_status(cls, v: str) -> str:
        if v not in V2_STATUSES:
            raise ValueError(f"status {v!r} not in {sorted(V2_STATUSES)}")
        return v

    @model_validator(mode="after")
    def _check_tuple(self) -> "V2FamilyMetadata":
        if self.primary_ability != self.ability_dir:
            raise ValueError(
                f"primary_ability={self.primary_ability} "
                f"does not match ability_dir={self.ability_dir}"
            )
        bad = set(self.instances_per_bucket) - V2_BUCKETS
        if bad:
            raise ValueError(f"unknown buckets in instances_per_bucket: {sorted(bad)}")
        if not self.control_set:
            raise ValueError(
                f"{self.family_id}: control_set must be non-empty "
                f"(§15 requires at least one control)"
            )
        bad_ctrl = set(self.control_set) - V2_CONTROL_KINDS
        if bad_ctrl:
            raise ValueError(f"unknown control kinds: {sorted(bad_ctrl)}")
        if self.status == "authored" and "no_persistence" not in self.control_set:
            raise ValueError(
                f"{self.family_id}: control_set must include 'no_persistence' (§15)"
            )
        # §10 Cold saturation guard. Only checked when a calibration anchor is
        # declared; legacy_migrated families without calibration anchors are
        # exempt because their 5-episode cold may legitimately score high.
        if (
            self.calibration_baseline_score is not None
            and self.status == "authored"
            and self.calibration_baseline_score > self.saturation_ceiling
        ):
            raise ValueError(
                f"{self.family_id}: calibration_baseline_score="
                f"{self.calibration_baseline_score} exceeds saturation ceiling "
                f"{self.saturation_ceiling} (§10 requires headroom)"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "V2FamilyMetadata":
        p = Path(path)
        data = yaml.safe_load(p.read_text()) or {}
        obj = cls.model_validate(data)
        obj._source_path = p
        return obj


def load_v2_families(root: str | Path) -> list[V2FamilyMetadata]:
    """Load every family.yaml under self-evolve-tasks-v2/<ability>/<family>/."""
    root_path = Path(root)
    families: list[V2FamilyMetadata] = []
    for yaml_path in sorted(root_path.glob("*/*/family.yaml")):
        families.append(V2FamilyMetadata.from_yaml(yaml_path))
    return families


class SelfEvolveEpisode(BaseModel):
    """One task instance within a self-evolve evaluation sequence."""

    task: str
    label: str = ""
    family_id: str = ""
    mechanism: Literal["skill", "memory", "session_search", "mixed"] | str = "mixed"
    bucket: Literal["baseline", "learn", "evaluation", "context"] | str = "evaluation"
    stage: str = ""
    phase: str = ""
    cluster_id: str = ""
    latent_rule_id: str = ""
    transfer_distance: Literal["none", "near", "far"] | str = "none"
    noise_profile: Literal["none", "light", "heavy", "stale_conflict"] | str = "none"
    expected_persistence_signal: Literal["skill", "memory", "session_search", "mixed"] | str = ""
    reflection_required: bool = False
    evaluation_requires_retrieval: bool = False
    conflict_mode: str = ""
    learn_signal_type: str = ""
    noise_level: Literal["none", "light", "heavy"] | str = "none"
    requires_fresh_session: bool = True
    persistence_allowed: bool = True
    preseed_artifacts_dir: str = ""
    initial_home_fixture_dir: str = ""
    shared_cold_run: bool = False
    history_mode: Literal["continue", "fresh", "from_anchor"] | str = ""
    history_save_anchor: str = ""
    history_load_anchor: str = ""
    # V2 Design Architecture §7.2: per-episode trigger metadata so the runner
    # can enforce that triggers appear only in learn/attempt/reflection buckets
    # and not in eval_near/eval_far.
    primary_trigger: str = ""
    active_triggers: list[str] = Field(default_factory=list)
    trigger_active: bool = False
    trigger_phrases: list[str] = Field(default_factory=list)
    trigger_injection_note: str = ""
    trigger_removed: bool = False
    trigger_removal_note: str = ""

    @model_validator(mode="after")
    def _backfill_compat_fields(self) -> "SelfEvolveEpisode":
        if not self.family_id and self.cluster_id:
            self.family_id = self.cluster_id
        if not self.stage and self.phase:
            self.stage = self.phase

        bucket_explicit = "bucket" in self.model_fields_set
        if not bucket_explicit:
            phase = self.phase.lower()
            if phase == "teach":
                self.bucket = "baseline"
            elif phase == "transfer":
                self.bucket = "evaluation"
            else:
                self.bucket = "evaluation"

        if not self.stage:
            self.stage = str(self.bucket)
        if not self.family_id:
            self.family_id = "default-family"
        if not self.expected_persistence_signal:
            self.expected_persistence_signal = str(self.mechanism)
        if not self.trigger_active:
            self.trigger_active = bool(self.active_triggers)

        requires_fresh_explicit = "requires_fresh_session" in self.model_fields_set
        if not requires_fresh_explicit:
            self.requires_fresh_session = self.bucket in {"evaluation", "control"}

        history_mode_explicit = "history_mode" in self.model_fields_set
        if not history_mode_explicit:
            if self.history_load_anchor:
                self.history_mode = "from_anchor"
            elif self.requires_fresh_session:
                self.history_mode = "fresh"
            else:
                self.history_mode = "continue"

        if self.history_mode == "from_anchor" and not self.history_load_anchor:
            raise ValueError("history_mode='from_anchor' requires history_load_anchor")
        if self.history_mode in {"continue", "fresh"} and self.history_load_anchor:
            raise ValueError(
                f"history_mode={self.history_mode!r} cannot declare history_load_anchor"
            )
        if self.history_mode not in V2_HISTORY_MODES:
            raise ValueError(f"unknown history_mode {self.history_mode!r}")

        transfer_explicit = "transfer_distance" in self.model_fields_set
        if not transfer_explicit and self.bucket == "evaluation":
            self.transfer_distance = "near"

        reflection_explicit = "reflection_required" in self.model_fields_set
        if not reflection_explicit and self.bucket == "learn":
            self.reflection_required = True

        noise_level_explicit = "noise_level" in self.model_fields_set
        if not noise_level_explicit:
            if self.noise_profile in {"light", "heavy"}:
                self.noise_level = self.noise_profile
            elif self.noise_profile == "stale_conflict":
                self.noise_level = "heavy"
            else:
                self.noise_level = "none"

        if self.noise_profile == "stale_conflict" and not self.conflict_mode:
            self.conflict_mode = "stale_conflict"
        return self


class RSIMemAdaptiveParameterConfig(BaseModel):
    """Runtime-owned adaptive parameter transported without learned content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parameter_id: str
    name: Literal[
        "extraction_accept_threshold",
        "internal_operation_accept_threshold",
        "consolidation_update_threshold",
        "retrieval_accept_threshold",
    ]
    prompt_ref: str
    baseline_value: float = Field(ge=0.0, le=1.0)

    @field_validator("parameter_id", "prompt_ref")
    @classmethod
    def _nonempty_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("adaptive parameter identity must not be empty")
        return value


class RSIMemAdaptiveWritebackConfig(BaseModel):
    """Strict transport contract for an attempt-local adaptive policy store."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    prepared_policy_store_file: str
    adaptive_policy_store_path: str
    adaptive_trusted_roots: list[str]
    adaptive_parameters: list[RSIMemAdaptiveParameterConfig]

    @field_validator("adaptive_policy_store_path")
    @classmethod
    def _attempt_local_store(cls, value: str) -> str:
        path = Path(value.strip())
        if not value.strip() or path.is_absolute() or ".." in path.parts:
            raise ValueError("adaptive policy store must be relative to Hermes home")
        return path.as_posix()

    @field_validator("prepared_policy_store_file")
    @classmethod
    def _prepared_store_file(cls, value: str) -> str:
        path = Path(value.strip())
        if (
            not value.strip()
            or path.is_absolute()
            or ".." in path.parts
            or path.name != path.as_posix()
        ):
            raise ValueError("prepared adaptive policy store must be a sibling file")
        return path.as_posix()

    @field_validator("adaptive_trusted_roots")
    @classmethod
    def _trusted_roots(cls, value: list[str]) -> list[str]:
        if not value or any(not item.strip() for item in value):
            raise ValueError("adaptive trusted roots must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("adaptive trusted roots must be unique")
        return value

    @field_validator("adaptive_parameters")
    @classmethod
    def _parameters(
        cls,
        value: list[RSIMemAdaptiveParameterConfig],
    ) -> list[RSIMemAdaptiveParameterConfig]:
        if not value:
            raise ValueError("adaptive parameters must not be empty")
        identities = [(item.parameter_id, item.name) for item in value]
        if len(identities) != len(set(identities)):
            raise ValueError("adaptive parameters must be unique")
        return value


class RSIMemExtractionTrialProfile(BaseModel):
    """Content-free identity for one validation-only extraction trial bundle."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )

    schema_version: Literal[1] = Field(alias="schemaVersion")
    preparation: Literal["extraction_matched_trial_store"]
    deployment_scope: Literal["matched_validation_only"] = Field(
        alias="deploymentScope"
    )
    official_evaluation: Literal[False] = Field(alias="officialEvaluation")
    validation_only: Literal[True] = Field(alias="validationOnly")
    production_activation_allowed: Literal[False] = Field(
        alias="productionActivationAllowed"
    )
    trial_id: str = Field(alias="trialId")
    slot_id: Literal["mem0-flat.semantic.extraction"] = Field(alias="slotId")
    parent_artifact_id: str = Field(alias="parentArtifactId")
    parent_artifact_digest: str = Field(alias="parentArtifactDigest")
    candidate_artifact_id: str = Field(alias="candidateArtifactId")
    candidate_artifact_digest: str = Field(alias="candidateArtifactDigest")
    offline_decision_id: str = Field(alias="offlineDecisionId")
    config_digest: str = Field(alias="configDigest")
    policy_store_digest: str = Field(alias="policyStoreDigest")
    offline_decision_digest: str = Field(alias="offlineDecisionDigest")

    @field_validator(
        "trial_id",
        "parent_artifact_id",
        "candidate_artifact_id",
        "offline_decision_id",
    )
    @classmethod
    def _identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("extraction trial identity must not be empty")
        return value

    @field_validator(
        "parent_artifact_digest",
        "candidate_artifact_digest",
        "config_digest",
        "policy_store_digest",
        "offline_decision_digest",
    )
    @classmethod
    def _digest(cls, value: str) -> str:
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("extraction trial digest must be sha256")
        return value


class RSIMemExtractionOfflineValidationProfile(BaseModel):
    """Content-free identity for an offline candidate observation bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_version: Literal[1] = Field(alias="schemaVersion")
    preparation: Literal["extraction_offline_validation_store"]
    deployment_scope: Literal["offline_validation_only"] = Field(alias="deploymentScope")
    official_evaluation: Literal[False] = Field(alias="officialEvaluation")
    validation_only: Literal[True] = Field(alias="validationOnly")
    production_activation_allowed: Literal[False] = Field(alias="productionActivationAllowed")
    validation_id: str = Field(alias="validationId")
    slot_id: Literal["mem0-flat.semantic.extraction"] = Field(alias="slotId")
    parent_artifact_id: str = Field(alias="parentArtifactId")
    parent_artifact_digest: str = Field(alias="parentArtifactDigest")
    candidate_artifact_id: str = Field(alias="candidateArtifactId")
    candidate_artifact_digest: str = Field(alias="candidateArtifactDigest")
    static_safety_report_id: str = Field(alias="staticSafetyReportId")
    deterministic_suite_report_id: str = Field(alias="deterministicSuiteReportId")
    config_digest: str = Field(alias="configDigest")
    candidate_artifact_file_digest: str = Field(alias="candidateArtifactFileDigest")

    @field_validator(
        "validation_id", "parent_artifact_id", "candidate_artifact_id",
        "static_safety_report_id", "deterministic_suite_report_id",
    )
    @classmethod
    def _identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("extraction offline identity must not be empty")
        return value

    @field_validator(
        "parent_artifact_digest", "candidate_artifact_digest", "config_digest",
        "candidate_artifact_file_digest",
    )
    @classmethod
    def _offline_digest(cls, value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("extraction offline digest must be sha256")
        return value


class HermesPersistenceConfig(BaseModel):
    """Hermes-specific persistence knobs for the evaluation harness."""

    enabled: bool = True
    memory_enabled: bool = True
    user_profile_enabled: bool = True
    skills_enabled: bool = True
    session_search_enabled: bool = True
    reflection_enabled: bool = True
    memory_nudge_interval: int = 1
    memory_flush_min_turns: int = 1
    skill_creation_nudge_interval: int = 1
    background_review_wait_s: float = 1.0
    preseed_artifacts_dir: str = ""
    initial_home_fixture_dir: str = ""
    rsimem_mode: Literal[
        "native",
        "native+ledger",
        "native+adapter+ledger",
    ] = "native"
    rsimem_adapter_failure_policy: Literal[
        "fail_closed",
        "bypass_native",
    ] = "fail_closed"
    rsimem_verify_native_projection: bool = False
    rsimem_lifecycle_evaluator_mode: Literal[
        "disabled",
        "deterministic",
        "injected_json",
    ] = "disabled"
    rsimem_lifecycle_policy_version: str = "phase1-dry-run-v1"
    rsimem_lifecycle_compiler_version: str = "uncompiled-v0"
    rsimem_lifecycle_timeout_seconds: float = 30.0
    rsimem_lifecycle_max_output_tokens: int = 4096
    rsimem_semantic_writeback_mode: Literal[
        "disabled",
        "static",
        "static_utility",
        "adaptive_utility",
    ] = "disabled"
    rsimem_semantic_writeback_timeout_seconds: float = 30.0
    rsimem_semantic_writeback_max_output_tokens: int = 4096
    rsimem_semantic_feedback_contract: Literal[
        "disabled",
        "sm01_tsv_v1",
        "sm02_boundary_v1",
        "sm03_fact_correction_v1",
        "sm05_normalized_tsv_v1",
    ] = "disabled"
    rsimem_adaptive_config: RSIMemAdaptiveWritebackConfig | None = None
    rsimem_adaptive_policy_source_path: str = Field(default="", exclude=True)
    rsimem_extraction_trial_profile: RSIMemExtractionTrialProfile | None = None
    rsimem_extraction_trial_source_path: str = Field(default="", exclude=True)
    rsimem_extraction_offline_profile: RSIMemExtractionOfflineValidationProfile | None = None
    rsimem_extraction_offline_source_path: str = Field(default="", exclude=True)
    rsimem_revocation_registry_path: str = ""

    @model_validator(mode="after")
    def _validate_adaptive_writeback_pair(self):
        selected = self.rsimem_semantic_writeback_mode == "adaptive_utility"
        if selected and self.rsimem_adaptive_config is None:
            raise ValueError("adaptive semantic writeback requires adaptive config")
        if not selected and self.rsimem_adaptive_config is not None:
            raise ValueError("adaptive config requires adaptive_utility mode")
        extraction_selected = self.rsimem_extraction_trial_profile is not None
        offline_selected = self.rsimem_extraction_offline_profile is not None
        if extraction_selected and offline_selected:
            raise ValueError("extraction trial and offline profile are mutually exclusive")
        if extraction_selected and self.rsimem_semantic_writeback_mode != "static":
            raise ValueError(
                "extraction matched trial requires static semantic writeback"
            )
        if extraction_selected != bool(self.rsimem_extraction_trial_source_path):
            raise ValueError(
                "extraction trial profile and source path must be configured together"
            )
        if offline_selected != bool(self.rsimem_extraction_offline_source_path):
            raise ValueError(
                "extraction offline profile and source path must be configured together"
            )
        if (extraction_selected or offline_selected) and self.rsimem_semantic_writeback_mode != "static":
            raise ValueError("extraction validation requires static semantic writeback")
        if extraction_selected and self.rsimem_adaptive_config is not None:
            raise ValueError(
                "extraction trial cannot use legacy adaptive utility config"
            )
        return self


class SelfEvolveTaskRef(BaseModel):
    """Reference to a task directory or task.yaml used by the benchmark."""

    task: str
    label: str = ""


class SingleTaskBenchmarkConfig(BaseModel):
    """Protocol config for repeated single-task evolve evaluation."""

    mechanism: Literal["skill", "memory", "session_search", "mixed"] | str = "mixed"
    baseline_repeats: int = 3
    target_baseline_pass_count: int = 1
    calibration_score_min: float = 0.45
    calibration_score_max: float = 0.82
    teach_rounds: int = 3
    retention_repeats: int = 5
    candidates: list[SelfEvolveTaskRef] = Field(default_factory=list)
    transfer_tasks: list[SelfEvolveTaskRef] = Field(default_factory=list)


class SelfEvolveSequenceDefinition(BaseModel):
    """A benchmark-owned ordered sequence of tasks."""

    name: str
    description: str = ""
    hermes: HermesPersistenceConfig = Field(default_factory=HermesPersistenceConfig)
    mode: Literal["episodes", "single_task"] = "episodes"
    single_task: SingleTaskBenchmarkConfig | None = None
    episodes: list[SelfEvolveEpisode] = Field(default_factory=list)
    manifest_file: str | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _validate_mode(self) -> "SelfEvolveSequenceDefinition":
        if self.mode == "single_task":
            if self.single_task is None:
                raise ValueError("single_task config is required when mode='single_task'")
            if not self.single_task.candidates:
                raise ValueError("single_task.candidates must not be empty")
            if not self.single_task.transfer_tasks:
                raise ValueError("single_task.transfer_tasks must not be empty")
        else:
            if not self.episodes:
                raise ValueError("episodes must not be empty when mode='episodes'")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SelfEvolveSequenceDefinition":
        manifest_path = Path(path).resolve()
        with open(manifest_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        data["manifest_file"] = str(manifest_path)
        return cls.model_validate(data)

    @property
    def manifest_path(self) -> Path:
        if not self.manifest_file:
            raise ValueError("manifest_file is not set")
        return Path(self.manifest_file)

    def resolve_task_yaml(self, task_ref: str) -> Path:
        path = Path(task_ref)
        if not path.is_absolute():
            path = (self.manifest_path.parent / path).resolve()
        if path.is_dir():
            path = path / "task.yaml"
        return path
