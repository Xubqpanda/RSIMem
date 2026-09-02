"""Frozen PAST-Bench family applicability matrix for Stage 1.

Family identity is an audit/split/report concern.  The method-facing view
contains only the pre-registered task and memory contract and deliberately
omits family IDs, filesystem roots, and role labels.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Sequence

from .contracts import MemoryKind


PAST_FAMILY_MATRIX_SCHEMA_VERSION = 1
PAST_FAMILY_MATRIX_SCHEMA = "rsimem-past-family-matrix-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,255}$")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _id(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")
    return value


def _strings(values: Sequence[str], name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    result = tuple(values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    for value in result:
        _id(value, name)
    return result


class FamilyPanel(StrEnum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    AUXILIARY = "auxiliary"


class FamilyRole(StrEnum):
    TARGET = "target"
    AUXILIARY = "auxiliary"
    EXCLUDED = "excluded"


@dataclass(frozen=True, slots=True)
class PastFamilySpec:
    """Applicability and confounder contract for one PAST family."""

    family_id: str
    panel: FamilyPanel
    role: FamilyRole
    task_root: str
    task_sequence: tuple[str, ...]
    stages: tuple[str, ...]
    metric: str
    memory_opportunity: str
    target_kind: MemoryKind | None
    confounders: tuple[str, ...]
    role_reason: str
    schema: str = PAST_FAMILY_MATRIX_SCHEMA
    schema_version: int = PAST_FAMILY_MATRIX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != PAST_FAMILY_MATRIX_SCHEMA or self.schema_version != PAST_FAMILY_MATRIX_SCHEMA_VERSION:
            raise ValueError("unsupported PAST family spec schema")
        _id(self.family_id, "family ID")
        object.__setattr__(self, "panel", FamilyPanel(self.panel))
        object.__setattr__(self, "role", FamilyRole(self.role))
        _id(self.task_root, "family task root")
        object.__setattr__(self, "task_sequence", _strings(self.task_sequence, "task sequence"))
        object.__setattr__(self, "stages", _strings(self.stages, "family stages"))
        _id(self.metric, "family metric")
        _id(self.memory_opportunity, "memory opportunity")
        if self.target_kind is not None:
            object.__setattr__(self, "target_kind", MemoryKind(self.target_kind))
        object.__setattr__(self, "confounders", _strings(self.confounders, "family confounders"))
        if not isinstance(self.role_reason, str) or not self.role_reason.strip():
            raise ValueError("family role reason must not be empty")
        if self.role is FamilyRole.TARGET:
            if self.target_kind is None or self.panel is FamilyPanel.AUXILIARY:
                raise ValueError("target family must declare a non-auxiliary target kind")
        elif self.role is FamilyRole.AUXILIARY:
            if self.panel is not FamilyPanel.AUXILIARY or self.target_kind is not None:
                raise ValueError("auxiliary family must be target-kind neutral")
        elif self.target_kind is not None:
            raise ValueError("excluded family must not declare a target kind")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "family_id": self.family_id,
            "panel": self.panel.value,
            "role": self.role.value,
            "task_root": self.task_root,
            "task_sequence": list(self.task_sequence),
            "stages": list(self.stages),
            "metric": self.metric,
            "memory_opportunity": self.memory_opportunity,
            "target_kind": self.target_kind.value if self.target_kind else None,
            "confounders": list(self.confounders),
            "role_reason": self.role_reason,
        }

    def payload(self) -> dict[str, object]:
        return self.identity_payload()

    def method_visible_payload(self) -> dict[str, object]:
        """Return the allowlisted view supplied to a method/updater."""

        return {
            "task_sequence": list(self.task_sequence),
            "stages": list(self.stages),
            "metric": self.metric,
            "memory_opportunity": self.memory_opportunity,
            "target_kind": self.target_kind.value if self.target_kind else None,
            "confounders": list(self.confounders),
        }


EXPECTED_PAST_FAMILY_IDS = (
    "SM01_preference_adoption",
    "SM02_constraint_retention",
    "SM03_fact_correction",
    "SM04_rule_migration",
    "SM05_weak_trigger_preference_adoption",
    "SM06_temporary_exception_pollution",
    "SM07_scoped_rule_migration",
    "EP01_prior_case_recall",
    "EP02_exception_list_recall",
    "EP03_recall_then_modify",
    "PC01_sop_bootstrap_01",
    "PC01_sop_bootstrap_02",
    "PC01_sop_bootstrap_03",
    "PC01_sop_bootstrap_04",
    "PC01_sop_bootstrap_05",
    "PC01_sop_bootstrap_06",
    "PC02_sop_patch_01",
    "PC02_sop_patch_02",
    "PC03_latent_rule_induction_01",
    "PC04_failure_to_rule_01",
    "PG01_release_decision_followup",
    "PG02_ops_exception_desk",
    "PG03_oncall_handoff_lookup",
    "PG04_temporary_waiver_audit",
    "PG05_change_freeze_followup",
    "PG06_kappa_integration_review",
)


def _spec(
    family_id: str,
    panel: FamilyPanel,
    task_root: str,
    opportunity: str,
    *,
    target_kind: MemoryKind | None,
    role: FamilyRole = FamilyRole.TARGET,
    sequence: tuple[str, ...],
    stages: tuple[str, ...],
    confounders: tuple[str, ...],
    role_reason: str,
) -> PastFamilySpec:
    return PastFamilySpec(
        family_id=family_id,
        panel=panel,
        role=role,
        task_root=task_root,
        task_sequence=sequence,
        stages=stages,
        metric="past_bench.official_task_metric.v1",
        memory_opportunity=opportunity,
        target_kind=target_kind,
        confounders=confounders,
        role_reason=role_reason,
    )


def default_past_family_specs() -> tuple[PastFamilySpec, ...]:
    """Return the 26 pre-registered PAST families in stable ID order."""

    semantic = (
        ("SM01_preference_adoption", "preference_adoption", "user_preference_visible_at_future_task"),
        ("SM02_constraint_retention", "constraint_retention", "user_constraint_visible_at_future_task"),
        ("SM03_fact_correction", "fact_correction", "corrected_fact_retrievable_at_future_task"),
        ("SM04_rule_migration", "rule_migration", "migrated_rule_retrievable_at_future_task"),
        ("SM05_weak_trigger_preference_adoption", "weak_trigger_preference_adoption", "implicit_preference_opportunity"),
        ("SM06_temporary_exception_pollution", "temporary_exception_pollution", "exception_scope_and_expiry"),
        ("SM07_scoped_rule_migration", "scoped_rule_migration", "scoped_rule_applicability"),
    )
    specs: list[PastFamilySpec] = []
    for family_id, root, opportunity in semantic:
        specs.append(_spec(
            family_id,
            FamilyPanel.SEMANTIC,
            f"self-evolve-tasks-v2/{'memory_ability' if family_id in {'SM01_preference_adoption', 'SM02_constraint_retention', 'SM05_weak_trigger_preference_adoption'} else 'update_ability'}/{family_id}",
            opportunity,
            target_kind=MemoryKind.SEMANTIC,
            sequence=("learn", "persist", "future_opportunity", "evaluate"),
            stages=("formation", "admission", "commit", "retrieval", "outcome"),
            confounders=("current_input_restatement", "wrong_mechanism_extra_text", "cross_task_state"),
            role_reason="pre-registered semantic sensitivity target family",
        ))
    episodic = (
        ("EP01_prior_case_recall", "prior_case_recall", "prior_episode_context_and_outcome"),
        ("EP02_exception_list_recall", "exception_list_recall", "prior_exception_episode_context"),
        ("EP03_recall_then_modify", "recall_then_modify", "prior_episode_retrieval_before_update"),
    )
    for family_id, root, opportunity in episodic:
        root_group = "update_ability" if family_id == "EP03_recall_then_modify" else "memory_ability"
        specs.append(_spec(
            family_id,
            FamilyPanel.EPISODIC,
            f"self-evolve-tasks-v2/{root_group}/{family_id}",
            opportunity,
            target_kind=MemoryKind.EPISODIC,
            sequence=("episode_context", "episode_close", "future_recall", "evaluate"),
            stages=("formation", "commit", "retrieval", "exposure", "outcome"),
            confounders=("current_input_restatement", "context_noise", "wrong_mechanism_extra_text"),
            role_reason="pre-registered episodic sensitivity target family",
        ))
    procedural_ids = (
        *(f"PC01_sop_bootstrap_{index:02d}" for index in range(1, 7)),
        "PC02_sop_patch_01",
        "PC02_sop_patch_02",
        "PC03_latent_rule_induction_01",
        "PC04_failure_to_rule_01",
    )
    for family_id in procedural_ids:
        root_group = "update_ability" if family_id.startswith("PC02") else "procedural_ability"
        opportunity = (
            "validated_skill_invocation"
            if family_id.startswith("PC01")
            else "skill_patch_or_failure_recovery"
        )
        specs.append(_spec(
            family_id,
            FamilyPanel.PROCEDURAL,
            f"self-evolve-tasks-v2/{root_group}/{family_id}",
            opportunity,
            target_kind=MemoryKind.PROCEDURAL,
            sequence=("demonstrate", "validate", "future_invocation", "evaluate"),
            stages=("formation", "validation", "commit", "exposure", "outcome"),
            confounders=("current_input_restatement", "tool_availability", "wrong_mechanism_extra_text"),
            role_reason="pre-registered procedural sensitivity target family",
        ))
    for index, name in enumerate((
        "release_decision_followup",
        "ops_exception_desk",
        "oncall_handoff_lookup",
        "temporary_waiver_audit",
        "change_freeze_followup",
        "kappa_integration_review",
    ), start=1):
        family_id = f"PG{index:02d}_{name}"
        specs.append(_spec(
            family_id,
            FamilyPanel.AUXILIARY,
            f"self-evolve-tasks-v2/proactive_information_gathering/{family_id}",
            "proactive_information_gap_followup",
            target_kind=None,
            role=FamilyRole.AUXILIARY,
            sequence=("information_gap", "followup", "evaluate"),
            stages=("trigger", "source_selection", "outcome"),
            confounders=("current_input_restatement", "benchmark_only_annotation"),
            role_reason="auxiliary process-feedback family; not a memory sensitivity target",
        ))
    return tuple(specs)


@dataclass(frozen=True, slots=True)
class PastFamilyMatrix:
    matrix_id: str
    families: tuple[PastFamilySpec, ...]
    conditions: tuple[str, ...]
    replicate_count: int
    practical_improvement_threshold: float
    paired_procedure: str
    schema: str = PAST_FAMILY_MATRIX_SCHEMA
    schema_version: int = PAST_FAMILY_MATRIX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != PAST_FAMILY_MATRIX_SCHEMA or self.schema_version != PAST_FAMILY_MATRIX_SCHEMA_VERSION:
            raise ValueError("unsupported PAST family matrix schema")
        _id(self.matrix_id, "family matrix ID")
        if len(self.families) != len(EXPECTED_PAST_FAMILY_IDS):
            raise ValueError("PAST family matrix must contain 26 families")
        ids = tuple(spec.family_id for spec in self.families)
        if set(ids) != set(EXPECTED_PAST_FAMILY_IDS) or len(ids) != len(set(ids)):
            raise ValueError("PAST family matrix family identity is incomplete")
        object.__setattr__(self, "conditions", _strings(self.conditions, "sensitivity conditions"))
        required = {
            "no_persistence", "native_static", "type_matched_oracle",
            "shortcut_current_input", "wrong_mechanism",
        }
        if set(self.conditions) != required:
            raise ValueError("sensitivity conditions do not match the frozen matrix")
        if type(self.replicate_count) is not int or self.replicate_count < 1:
            raise ValueError("replicate count must be positive")
        if not isinstance(self.practical_improvement_threshold, (int, float)) or not 0.0 < self.practical_improvement_threshold <= 1.0:
            raise ValueError("practical improvement threshold must be in (0, 1]")
        _id(self.paired_procedure, "paired procedure")
        expected = f"past-family-matrix.{_digest(self.identity_payload())[:40]}"
        if self.matrix_id != expected:
            raise ValueError("PAST family matrix ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "families": [spec.payload() for spec in self.families],
            "conditions": list(self.conditions),
            "replicate_count": self.replicate_count,
            "practical_improvement_threshold": self.practical_improvement_threshold,
            "paired_procedure": self.paired_procedure,
        }

    @property
    def matrix_digest(self) -> str:
        return _digest(self.identity_payload())

    def payload(self) -> dict[str, object]:
        return {"matrix_id": self.matrix_id, **self.identity_payload()}

    def spec_for(self, family_id: str) -> PastFamilySpec:
        _id(family_id, "family ID")
        try:
            return next(spec for spec in self.families if spec.family_id == family_id)
        except StopIteration as exc:
            raise ValueError("family is not in the frozen PAST matrix") from exc

    def method_view_for(self, family_id: str) -> dict[str, object]:
        return self.spec_for(family_id).method_visible_payload()

    @classmethod
    def create_default(
        cls,
        *,
        replicate_count: int = 3,
        practical_improvement_threshold: float = 0.05,
    ) -> "PastFamilyMatrix":
        values = {
            "families": default_past_family_specs(),
            "conditions": (
                "no_persistence", "native_static", "type_matched_oracle",
                "shortcut_current_input", "wrong_mechanism",
            ),
            "replicate_count": replicate_count,
            "practical_improvement_threshold": practical_improvement_threshold,
            "paired_procedure": "paired_delta.v1",
            "schema": PAST_FAMILY_MATRIX_SCHEMA,
            "schema_version": PAST_FAMILY_MATRIX_SCHEMA_VERSION,
        }
        identity = {
            "schema": values["schema"],
            "schema_version": values["schema_version"],
            "families": [spec.payload() for spec in values["families"]],
            "conditions": list(values["conditions"]),
            "replicate_count": replicate_count,
            "practical_improvement_threshold": practical_improvement_threshold,
            "paired_procedure": values["paired_procedure"],
        }
        return cls(matrix_id=f"past-family-matrix.{_digest(identity)[:40]}", **values)

    @classmethod
    def from_payload(cls, value: object) -> "PastFamilyMatrix":
        fields = {
            "matrix_id", "schema", "schema_version", "families", "conditions",
            "replicate_count", "practical_improvement_threshold", "paired_procedure",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("malformed PAST family matrix")
        if not isinstance(value["families"], list) or not isinstance(value["conditions"], list):
            raise ValueError("PAST family matrix collections must be lists")
        try:
            families = tuple(PastFamilySpec(**item) for item in value["families"])
            matrix = cls(
                matrix_id=value["matrix_id"],
                schema=value["schema"],
                schema_version=value["schema_version"],
                families=families,
                conditions=tuple(value["conditions"]),
                replicate_count=value["replicate_count"],
                practical_improvement_threshold=value["practical_improvement_threshold"],
                paired_procedure=value["paired_procedure"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed PAST family matrix") from exc
        if matrix.payload() != dict(value):
            raise ValueError("non-canonical PAST family matrix")
        return matrix


__all__ = [
    "EXPECTED_PAST_FAMILY_IDS",
    "FamilyPanel",
    "FamilyRole",
    "PAST_FAMILY_MATRIX_SCHEMA",
    "PAST_FAMILY_MATRIX_SCHEMA_VERSION",
    "PastFamilyMatrix",
    "PastFamilySpec",
    "default_past_family_specs",
]
