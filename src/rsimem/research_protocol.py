"""Versioned Stage 1 research protocol and immutable manifest store."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping, Sequence

from .memory.contracts import MemoryKind
from .memory.evidence_planes import EvidencePlane
from .memory.family_matrix import PastFamilyMatrix
from .memory.lifecycle_surfaces import LIFECYCLE_SURFACE_SCHEMA
from .memory.taxonomy import MemoryControlKind, MemoryUnitDescriptor


RESEARCH_PROTOCOL_SCHEMA_VERSION = 1
RESEARCH_PROTOCOL_SCHEMA = "rsimem-research-protocol-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,255}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _id(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a sha256 digest")
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


class ComparisonLevel(StrEnum):
    VANILLA_NO_PERSISTENCE = "vanilla_no_persistence"
    HERMES_NATIVE_STATIC = "hermes_native_static"
    EXISTING_METHOD = "existing_method"
    EXISTING_METHOD_RSIMEM = "existing_method_rsimem"
    SENSITIVITY = "sensitivity"


class SensitivityCondition(StrEnum):
    NO_PERSISTENCE = "no_persistence"
    NATIVE_STATIC = "native_static"
    TYPE_MATCHED_ORACLE = "type_matched_oracle"
    SHORTCUT_CURRENT_INPUT = "shortcut_current_input"
    WRONG_MECHANISM = "wrong_mechanism"


@dataclass(frozen=True, slots=True)
class ExperimentSplit:
    split_id: str
    train_family_ids: tuple[str, ...]
    validation_family_ids: tuple[str, ...]
    final_family_ids: tuple[str, ...]
    task_template_group_ids: tuple[str, ...]
    leakage_rules: tuple[str, ...]

    def __post_init__(self) -> None:
        _id(self.split_id, "split ID")
        train = _strings(self.train_family_ids, "train family IDs")
        validation = _strings(self.validation_family_ids, "validation family IDs")
        final = _strings(self.final_family_ids, "final family IDs")
        if set(train) & (set(validation) | set(final)) or set(validation) & set(final):
            raise ValueError("family IDs must not cross split roles")
        object.__setattr__(self, "train_family_ids", train)
        object.__setattr__(self, "validation_family_ids", validation)
        object.__setattr__(self, "final_family_ids", final)
        object.__setattr__(self, "task_template_group_ids", _strings(self.task_template_group_ids, "task-template group IDs"))
        object.__setattr__(self, "leakage_rules", _strings(self.leakage_rules, "split leakage rules"))

    def payload(self) -> dict[str, object]:
        return {
            "split_id": self.split_id,
            "train_family_ids": list(self.train_family_ids),
            "validation_family_ids": list(self.validation_family_ids),
            "final_family_ids": list(self.final_family_ids),
            "task_template_group_ids": list(self.task_template_group_ids),
            "leakage_rules": list(self.leakage_rules),
        }


@dataclass(frozen=True, slots=True)
class ConditionContract:
    condition_id: SensitivityCondition
    level: ComparisonLevel
    target_kind: MemoryKind | None
    oracle_only: bool
    independent_state_directory: bool
    persistence_mode: str
    mechanism: str
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition_id", SensitivityCondition(self.condition_id))
        object.__setattr__(self, "level", ComparisonLevel(self.level))
        if self.target_kind is not None:
            object.__setattr__(self, "target_kind", MemoryKind(self.target_kind))
        if type(self.oracle_only) is not bool or type(self.independent_state_directory) is not bool:
            raise ValueError("condition boolean fields must be bool")
        _id(self.persistence_mode, "condition persistence mode")
        _id(self.mechanism, "condition mechanism")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("condition description must not be empty")
        if self.condition_id is SensitivityCondition.TYPE_MATCHED_ORACLE and not self.oracle_only:
            raise ValueError("type-matched oracle must be audit-only")
        if self.oracle_only and self.condition_id is not SensitivityCondition.TYPE_MATCHED_ORACLE:
            raise ValueError("only type-matched oracle may be oracle-only")

    def payload(self) -> dict[str, object]:
        return {
            "condition_id": self.condition_id.value,
            "level": self.level.value,
            "target_kind": self.target_kind.value if self.target_kind else None,
            "oracle_only": self.oracle_only,
            "independent_state_directory": self.independent_state_directory,
            "persistence_mode": self.persistence_mode,
            "mechanism": self.mechanism,
            "description": self.description,
        }

    def method_visible_payload(self) -> dict[str, object]:
        """Return only condition facts safe for method/updater input."""

        return {
            "condition": self.condition_id.value,
            "target_kind": self.target_kind.value if self.target_kind else None,
            "persistence_mode": self.persistence_mode,
            "mechanism": self.mechanism,
        }


@dataclass(frozen=True, slots=True)
class MetricContract:
    primary_metric: str
    practical_improvement_threshold: float
    replicate_count: int
    paired_procedure: str
    mechanism_fields: tuple[str, ...]
    raw_resource_fields: tuple[str, ...]
    updater_evidence_plane: EvidencePlane = EvidencePlane.PURE_PROCESS

    def __post_init__(self) -> None:
        _id(self.primary_metric, "primary metric")
        if not isinstance(self.practical_improvement_threshold, (int, float)) or not 0.0 < self.practical_improvement_threshold <= 1.0:
            raise ValueError("practical improvement threshold must be in (0, 1]")
        if type(self.replicate_count) is not int or self.replicate_count < 1:
            raise ValueError("metric replicate count must be positive")
        _id(self.paired_procedure, "paired procedure")
        object.__setattr__(self, "mechanism_fields", _strings(self.mechanism_fields, "mechanism fields"))
        object.__setattr__(self, "raw_resource_fields", _strings(self.raw_resource_fields, "raw resource fields"))
        object.__setattr__(self, "updater_evidence_plane", EvidencePlane(self.updater_evidence_plane))
        if self.updater_evidence_plane is not EvidencePlane.PURE_PROCESS:
            raise ValueError("updater evidence must stay in pure_process plane")

    def payload(self) -> dict[str, object]:
        return {
            "primary_metric": self.primary_metric,
            "practical_improvement_threshold": self.practical_improvement_threshold,
            "replicate_count": self.replicate_count,
            "paired_procedure": self.paired_procedure,
            "mechanism_fields": list(self.mechanism_fields),
            "raw_resource_fields": list(self.raw_resource_fields),
            "updater_evidence_plane": self.updater_evidence_plane.value,
        }


@dataclass(frozen=True, slots=True)
class ResearchProtocol:
    protocol_id: str
    memory_units: tuple[MemoryUnitDescriptor, ...]
    memory_control_kinds: tuple[MemoryControlKind, ...]
    family_matrix_id: str
    family_matrix_digest: str
    lifecycle_surface_schema: str
    comparison_levels: tuple[ComparisonLevel, ...]
    conditions: tuple[ConditionContract, ...]
    split: ExperimentSplit
    metric: MetricContract
    model_id: str
    provider_id: str
    wrapper_id: str
    tool_budget: int
    max_turns: int
    sampling_temperature: float
    retry_policy: str
    schema: str = RESEARCH_PROTOCOL_SCHEMA
    schema_version: int = RESEARCH_PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != RESEARCH_PROTOCOL_SCHEMA or self.schema_version != RESEARCH_PROTOCOL_SCHEMA_VERSION:
            raise ValueError("unsupported research protocol schema")
        _id(self.protocol_id, "research protocol ID")
        if not self.memory_units or len({unit.unit_id for unit in self.memory_units}) != len(self.memory_units):
            raise ValueError("research protocol requires unique memory units")
        controls = tuple(MemoryControlKind(value) for value in self.memory_control_kinds)
        if set(controls) != set(MemoryControlKind):
            raise ValueError("research protocol must separate all memory control kinds")
        object.__setattr__(self, "memory_control_kinds", controls)
        _id(self.family_matrix_id, "family matrix ID")
        _sha(self.family_matrix_digest, "family matrix digest")
        if self.lifecycle_surface_schema != LIFECYCLE_SURFACE_SCHEMA:
            raise ValueError("research protocol lifecycle schema mismatch")
        levels = tuple(ComparisonLevel(value) for value in self.comparison_levels)
        if len(levels) != len(set(levels)) or ComparisonLevel.VANILLA_NO_PERSISTENCE not in levels or ComparisonLevel.HERMES_NATIVE_STATIC not in levels or ComparisonLevel.SENSITIVITY not in levels:
            raise ValueError("comparison levels are incomplete")
        object.__setattr__(self, "comparison_levels", levels)
        conditions = tuple(self.conditions)
        if {condition.condition_id for condition in conditions} != set(SensitivityCondition):
            raise ValueError("all five sensitivity conditions are required")
        if len(conditions) != len(set(condition.condition_id for condition in conditions)):
            raise ValueError("sensitivity conditions must be unique")
        object.__setattr__(self, "conditions", conditions)
        if not isinstance(self.split, ExperimentSplit) or not isinstance(self.metric, MetricContract):
            raise ValueError("research protocol split and metric contracts are required")
        _id(self.model_id, "model ID")
        _id(self.provider_id, "provider ID")
        _id(self.wrapper_id, "wrapper ID")
        if type(self.tool_budget) is not int or self.tool_budget < 1 or type(self.max_turns) is not int or self.max_turns < 1:
            raise ValueError("tool budget and max turns must be positive integers")
        if not isinstance(self.sampling_temperature, (int, float)) or not 0.0 <= self.sampling_temperature <= 2.0:
            raise ValueError("sampling temperature must be in [0, 2]")
        _id(self.retry_policy, "retry policy")
        expected = f"research-protocol.{_digest(self.identity_payload())[:40]}"
        if self.protocol_id != expected:
            raise ValueError("research protocol ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "memory_units": [unit.payload() for unit in self.memory_units],
            "memory_control_kinds": [value.value for value in self.memory_control_kinds],
            "family_matrix_id": self.family_matrix_id,
            "family_matrix_digest": self.family_matrix_digest,
            "lifecycle_surface_schema": self.lifecycle_surface_schema,
            "comparison_levels": [value.value for value in self.comparison_levels],
            "conditions": [condition.payload() for condition in self.conditions],
            "split": self.split.payload(),
            "metric": self.metric.payload(),
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "wrapper_id": self.wrapper_id,
            "tool_budget": self.tool_budget,
            "max_turns": self.max_turns,
            "sampling_temperature": self.sampling_temperature,
            "retry_policy": self.retry_policy,
        }

    @property
    def protocol_digest(self) -> str:
        return _digest(self.identity_payload())

    def payload(self) -> dict[str, object]:
        return {"protocol_id": self.protocol_id, **self.identity_payload()}

    @classmethod
    def from_payload(cls, value: object) -> "ResearchProtocol":
        if not isinstance(value, Mapping):
            raise ValueError("malformed research protocol manifest")
        fields = {
            "protocol_id", "schema", "schema_version", "memory_units", "memory_control_kinds",
            "family_matrix_id", "family_matrix_digest", "lifecycle_surface_schema",
            "comparison_levels", "conditions", "split", "metric", "model_id", "provider_id",
            "wrapper_id", "tool_budget", "max_turns", "sampling_temperature", "retry_policy",
        }
        if set(value) != fields or not isinstance(value["memory_units"], list) or not isinstance(value["conditions"], list):
            raise ValueError("malformed research protocol manifest")
        try:
            units = tuple(MemoryUnitDescriptor.from_payload(item) for item in value["memory_units"])
            conditions = tuple(
                ConditionContract(
                    condition_id=item["condition_id"], level=item["level"], target_kind=item["target_kind"],
                    oracle_only=item["oracle_only"], independent_state_directory=item["independent_state_directory"],
                    persistence_mode=item["persistence_mode"], mechanism=item["mechanism"], description=item["description"],
                ) for item in value["conditions"]
            )
            split_value = value["split"]
            metric_value = value["metric"]
            if not isinstance(split_value, Mapping) or not isinstance(metric_value, Mapping):
                raise ValueError("nested protocol contract is malformed")
            split = ExperimentSplit(
                split_id=split_value["split_id"], train_family_ids=tuple(split_value["train_family_ids"]),
                validation_family_ids=tuple(split_value["validation_family_ids"]), final_family_ids=tuple(split_value["final_family_ids"]),
                task_template_group_ids=tuple(split_value["task_template_group_ids"]), leakage_rules=tuple(split_value["leakage_rules"]),
            )
            metric = MetricContract(
                primary_metric=metric_value["primary_metric"], practical_improvement_threshold=metric_value["practical_improvement_threshold"],
                replicate_count=metric_value["replicate_count"], paired_procedure=metric_value["paired_procedure"],
                mechanism_fields=tuple(metric_value["mechanism_fields"]), raw_resource_fields=tuple(metric_value["raw_resource_fields"]),
                updater_evidence_plane=metric_value["updater_evidence_plane"],
            )
            protocol = cls(
                protocol_id=value["protocol_id"], schema=value["schema"], schema_version=value["schema_version"],
                memory_units=units, memory_control_kinds=tuple(value["memory_control_kinds"]),
                family_matrix_id=value["family_matrix_id"], family_matrix_digest=value["family_matrix_digest"],
                lifecycle_surface_schema=value["lifecycle_surface_schema"], comparison_levels=tuple(value["comparison_levels"]),
                conditions=conditions, split=split, metric=metric, model_id=value["model_id"], provider_id=value["provider_id"],
                wrapper_id=value["wrapper_id"], tool_budget=value["tool_budget"], max_turns=value["max_turns"],
                sampling_temperature=value["sampling_temperature"], retry_policy=value["retry_policy"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed research protocol manifest") from exc
        if protocol.payload() != dict(value):
            raise ValueError("non-canonical research protocol manifest")
        return protocol

    def method_visible_condition(self, condition_id: SensitivityCondition | str) -> dict[str, object]:
        try:
            condition = next(value for value in self.conditions if value.condition_id == SensitivityCondition(condition_id))
        except StopIteration as exc:
            raise ValueError("condition is not in the research protocol") from exc
        return condition.method_visible_payload()

    @classmethod
    def create(
        cls,
        *,
        memory_units: tuple[MemoryUnitDescriptor, ...],
        family_matrix: PastFamilyMatrix,
        split: ExperimentSplit,
        sensitivity_target_kind: MemoryKind = MemoryKind.SEMANTIC,
        model_id: str = "gpt-5.6-luna",
        provider_id: str = "coding.tu-zi.com/v1",
        wrapper_id: str = "past-bench.hermes.v1",
        tool_budget: int = 32,
        max_turns: int = 20,
        sampling_temperature: float = 0.0,
        retry_policy: str = "bounded_retry.v1",
    ) -> "ResearchProtocol":
        if not isinstance(family_matrix, PastFamilyMatrix):
            raise TypeError("research protocol requires a PastFamilyMatrix")
        sensitivity_target_kind = MemoryKind(sensitivity_target_kind)
        conditions = (
            ConditionContract(SensitivityCondition.NO_PERSISTENCE, ComparisonLevel.VANILLA_NO_PERSISTENCE, None, False, True, "disabled", "none", "no cross-task memory"),
            ConditionContract(SensitivityCondition.NATIVE_STATIC, ComparisonLevel.HERMES_NATIVE_STATIC, None, False, True, "native_static", "native_memory", "Hermes native static memory"),
            ConditionContract(SensitivityCondition.TYPE_MATCHED_ORACLE, ComparisonLevel.SENSITIVITY, sensitivity_target_kind, True, True, "oracle_static", "type_matched", "audit-only target mechanism upper bound"),
            ConditionContract(SensitivityCondition.SHORTCUT_CURRENT_INPUT, ComparisonLevel.SENSITIVITY, sensitivity_target_kind, False, True, "shortcut", "current_input", "current-input shortcut control"),
            ConditionContract(SensitivityCondition.WRONG_MECHANISM, ComparisonLevel.SENSITIVITY, sensitivity_target_kind, False, True, "wrong_mechanism", "extra_text", "wrong-mechanism control"),
        )
        metric = MetricContract(
            primary_metric="past_bench.official_task_metric.v1",
            practical_improvement_threshold=family_matrix.practical_improvement_threshold,
            replicate_count=family_matrix.replicate_count,
            paired_procedure=family_matrix.paired_procedure,
            mechanism_fields=(
                "formation_coverage", "retrieval", "exposure", "attributable_use",
                "unknown_use", "correct_update", "harmful_update", "abstention",
                "negative_transfer", "surface_failure",
            ),
            raw_resource_fields=(
                "input_tokens", "output_tokens", "cache_read_tokens",
                "cache_write_tokens", "reasoning_tokens", "latency_ms",
                "storage_bytes", "api_calls", "retry_count",
            ),
        )
        values = {
            "memory_units": tuple(memory_units),
            "memory_control_kinds": tuple(MemoryControlKind),
            "family_matrix_id": family_matrix.matrix_id,
            "family_matrix_digest": family_matrix.matrix_digest,
            "lifecycle_surface_schema": LIFECYCLE_SURFACE_SCHEMA,
            "comparison_levels": (
                ComparisonLevel.VANILLA_NO_PERSISTENCE,
                ComparisonLevel.HERMES_NATIVE_STATIC,
                ComparisonLevel.SENSITIVITY,
            ),
            "conditions": conditions,
            "split": split,
            "metric": metric,
            "model_id": model_id,
            "provider_id": provider_id,
            "wrapper_id": wrapper_id,
            "tool_budget": tool_budget,
            "max_turns": max_turns,
            "sampling_temperature": sampling_temperature,
            "retry_policy": retry_policy,
            "schema": RESEARCH_PROTOCOL_SCHEMA,
            "schema_version": RESEARCH_PROTOCOL_SCHEMA_VERSION,
        }
        identity = {
            "schema": values["schema"],
            "schema_version": values["schema_version"],
            **{key: (value.payload() if hasattr(value, "payload") else [item.payload() for item in value] if key in {"memory_units", "conditions"} else [item.value for item in value] if key in {"memory_control_kinds", "comparison_levels"} else value) for key, value in values.items() if key not in {"schema", "schema_version"}},
        }
        # Keep ID construction in one place and let the dataclass verify it.
        return cls(protocol_id=f"research-protocol.{_digest(identity)[:40]}", **values)


class JsonResearchProtocolStore:
    """Crash-safe immutable store for one research protocol manifest."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()

    def freeze(self, protocol: ResearchProtocol) -> bool:
        if not isinstance(protocol, ResearchProtocol):
            raise TypeError("protocol must be a ResearchProtocol")
        target = self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        serialized = _canonical(protocol.payload()) + "\n"
        lock_path = target.with_name(target.name + ".lock")
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if target.is_symlink() or lock_path.is_symlink():
                raise ValueError("research protocol store cannot be symlinked")
            if target.exists():
                if target.read_text(encoding="utf-8") != serialized:
                    raise ValueError("research protocol conflicts with existing manifest")
                return False
            descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
                directory = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return True

    def get(self) -> ResearchProtocol:
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError("research protocol manifest is missing or symlinked")
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return ResearchProtocol.from_payload(value)
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("malformed research protocol"):
                raise
            raise ValueError("malformed research protocol manifest") from exc


def _default_memory_units() -> tuple[MemoryUnitDescriptor, ...]:
    return (
        MemoryUnitDescriptor(
            unit_id="semantic.fact.v1", kind=MemoryKind.SEMANTIC,
            content_schema="semantic.fact.v1", scope="user",
            source_provenance=("runtime.observation.v1",),
            temporal_identity="validity.current.v1", applicability=("task.answering",),
            version="v1", owner_method="native.hermes.v1",
        ),
        MemoryUnitDescriptor(
            unit_id="episodic.experience.v1", kind=MemoryKind.EPISODIC,
            content_schema="episodic.experience.v1", scope="session",
            source_provenance=("runtime.observation.v1",),
            temporal_identity="episode.closed.v1", applicability=("similar_task.recall",),
            version="v1", owner_method="native.hermes.v1",
        ),
        MemoryUnitDescriptor(
            unit_id="procedural.skill.v1", kind=MemoryKind.PROCEDURAL,
            content_schema="procedural.skill.v1", scope="user",
            source_provenance=("runtime.observation.v1",),
            temporal_identity="skill.versioned.v1", applicability=("tool_execution",),
            version="v1", owner_method="native.hermes.v1",
        ),
    )


def default_research_protocol() -> ResearchProtocol:
    """Build the repository's result-independent Stage 1 protocol."""

    matrix = PastFamilyMatrix.create_default()
    split = ExperimentSplit(
        split_id="past-family-split-v1",
        train_family_ids=("SM01_preference_adoption", "SM02_constraint_retention", "EP01_prior_case_recall", "PC01_sop_bootstrap_01"),
        validation_family_ids=("SM03_fact_correction", "EP02_exception_list_recall", "PC02_sop_patch_01"),
        final_family_ids=("SM04_rule_migration", "EP03_recall_then_modify", "PC04_failure_to_rule_01"),
        task_template_group_ids=("stage1.train.v1", "stage1.validation.v1", "stage1.final.v1"),
        leakage_rules=("no_answer_or_value_reuse", "no_family_id_in_method_view", "no_cross_condition_state"),
    )
    return ResearchProtocol.create(
        memory_units=_default_memory_units(),
        family_matrix=matrix,
        split=split,
    )


def _parser():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol = default_research_protocol()
    JsonResearchProtocolStore(args.output).freeze(protocol)
    print(_canonical({"protocolId": protocol.protocol_id, "protocolDigest": protocol.protocol_digest}))
    return 0


__all__ = [
    "ComparisonLevel",
    "ConditionContract",
    "ExperimentSplit",
    "JsonResearchProtocolStore",
    "MetricContract",
    "RESEARCH_PROTOCOL_SCHEMA",
    "RESEARCH_PROTOCOL_SCHEMA_VERSION",
    "ResearchProtocol",
    "SensitivityCondition",
    "default_research_protocol",
]


if __name__ == "__main__":
    raise SystemExit(main())
