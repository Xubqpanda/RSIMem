"""Result-independent PAST memory-sensitivity matrix contracts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Sequence

from .adapter_contracts import FeedbackCondition
from .memory.contracts import MemoryKind
from .memory.evidence_planes import EvidencePlane, EvidenceSourceKind, validate_plane_source
from .memory.family_matrix import FamilyPanel, FamilyRole, PastFamilyMatrix
from .research_protocol import (
    ConditionContract,
    ResearchProtocol,
    SensitivityCondition,
)


SENSITIVITY_SCHEMA_VERSION = 1
SENSITIVITY_SCHEMA = "rsimem-past-sensitivity-v1"
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


class SensitivityPanel(StrEnum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"

    @property
    def memory_kind(self) -> MemoryKind:
        return MemoryKind(self.value)

    @property
    def family_panel(self) -> FamilyPanel:
        return FamilyPanel(self.value)


@dataclass(frozen=True, slots=True)
class OracleArtifact:
    """Audit-only minimal target mechanism, never an updater input."""

    artifact_id: str
    panel: SensitivityPanel
    target_kind: MemoryKind
    mechanism: str
    minimal_field_ids: tuple[str, ...]
    content_digest: str
    evidence_plane: EvidencePlane = EvidencePlane.BENCHMARK_AUDIT
    evidence_source: EvidenceSourceKind = EvidenceSourceKind.BENCHMARK_CONTRACT
    oracle_only: bool = True
    schema: str = SENSITIVITY_SCHEMA
    schema_version: int = SENSITIVITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != SENSITIVITY_SCHEMA or self.schema_version != SENSITIVITY_SCHEMA_VERSION:
            raise ValueError("unsupported sensitivity oracle schema")
        _id(self.artifact_id, "oracle artifact ID")
        object.__setattr__(self, "panel", SensitivityPanel(self.panel))
        object.__setattr__(self, "target_kind", MemoryKind(self.target_kind))
        if self.target_kind is not self.panel.memory_kind:
            raise ValueError("oracle target kind must match sensitivity panel")
        _id(self.mechanism, "oracle mechanism")
        fields = tuple(self.minimal_field_ids)
        if not fields or len(fields) != len(set(fields)):
            raise ValueError("oracle minimal fields must be nonempty and unique")
        for value in fields:
            _id(value, "oracle minimal field")
        object.__setattr__(self, "minimal_field_ids", fields)
        _sha(self.content_digest, "oracle content digest")
        plane, source = validate_plane_source(self.evidence_plane, self.evidence_source)
        object.__setattr__(self, "evidence_plane", plane)
        object.__setattr__(self, "evidence_source", source)
        if self.oracle_only is not True:
            raise ValueError("sensitivity oracle must be oracle_only")

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "panel": self.panel.value,
            "target_kind": self.target_kind.value,
            "mechanism": self.mechanism,
            "minimal_field_ids": list(self.minimal_field_ids),
            "content_digest": self.content_digest,
            "evidence_plane": self.evidence_plane.value,
            "evidence_source": self.evidence_source.value,
            "oracle_only": self.oracle_only,
        }


@dataclass(frozen=True, slots=True)
class SensitivityCase:
    """One family/condition pair in the matched matrix."""

    case_id: str
    panel: SensitivityPanel
    family_id: str
    condition: SensitivityCondition
    target_kind: MemoryKind
    state_identity: str
    oracle_artifact_id: str | None
    mechanism: str
    observation_plane: EvidencePlane = EvidencePlane.PURE_PROCESS
    schema: str = SENSITIVITY_SCHEMA
    schema_version: int = SENSITIVITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != SENSITIVITY_SCHEMA or self.schema_version != SENSITIVITY_SCHEMA_VERSION:
            raise ValueError("unsupported sensitivity case schema")
        _id(self.case_id, "sensitivity case ID")
        object.__setattr__(self, "panel", SensitivityPanel(self.panel))
        _id(self.family_id, "sensitivity family ID")
        object.__setattr__(self, "condition", SensitivityCondition(self.condition))
        object.__setattr__(self, "target_kind", MemoryKind(self.target_kind))
        if self.target_kind is not self.panel.memory_kind:
            raise ValueError("sensitivity case target kind must match panel")
        _id(self.state_identity, "sensitivity state identity")
        if self.oracle_artifact_id is not None:
            _id(self.oracle_artifact_id, "sensitivity oracle artifact ID")
        _id(self.mechanism, "sensitivity mechanism")
        object.__setattr__(self, "observation_plane", EvidencePlane(self.observation_plane))
        if self.observation_plane is not EvidencePlane.PURE_PROCESS:
            raise ValueError("sensitivity observations must be pure_process")

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "panel": self.panel.value,
            "family_id": self.family_id,
            "condition": self.condition.value,
            "target_kind": self.target_kind.value,
            "state_identity": self.state_identity,
            "oracle_artifact_id": self.oracle_artifact_id,
            "mechanism": self.mechanism,
            "observation_plane": self.observation_plane.value,
        }

    def method_visible_payload(self) -> dict[str, object]:
        return {
            "condition": self.condition.value,
            "target_kind": self.target_kind.value,
            "mechanism": self.mechanism,
            "oracle_available": self.oracle_artifact_id is not None,
        }


@dataclass(frozen=True, slots=True)
class SensitivityMatrix:
    matrix_id: str
    panel: SensitivityPanel
    target_kind: MemoryKind
    family_ids: tuple[str, ...]
    cases: tuple[SensitivityCase, ...]
    oracle_artifacts: tuple[OracleArtifact, ...]
    protocol_id: str
    family_matrix_digest: str
    threshold: float
    replicate_count: int
    schema: str = SENSITIVITY_SCHEMA
    schema_version: int = SENSITIVITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != SENSITIVITY_SCHEMA or self.schema_version != SENSITIVITY_SCHEMA_VERSION:
            raise ValueError("unsupported sensitivity matrix schema")
        _id(self.matrix_id, "sensitivity matrix ID")
        object.__setattr__(self, "panel", SensitivityPanel(self.panel))
        object.__setattr__(self, "target_kind", MemoryKind(self.target_kind))
        if self.target_kind is not self.panel.memory_kind:
            raise ValueError("sensitivity matrix target kind must match panel")
        families = tuple(self.family_ids)
        if not families or len(families) != len(set(families)):
            raise ValueError("sensitivity matrix families must be nonempty and unique")
        for family_id in families:
            _id(family_id, "sensitivity family ID")
        object.__setattr__(self, "family_ids", families)
        expected_conditions = set(SensitivityCondition)
        if len(self.cases) != len(families) * len(expected_conditions):
            raise ValueError("sensitivity matrix must have five conditions per family")
        case_keys = {(case.family_id, case.condition) for case in self.cases}
        if case_keys != {(family_id, condition) for family_id in families for condition in expected_conditions}:
            raise ValueError("sensitivity matrix cases are incomplete or duplicated")
        if any(case.panel is not self.panel or case.target_kind is not self.target_kind for case in self.cases):
            raise ValueError("sensitivity case panel/kind mismatch")
        artifact_ids = {artifact.artifact_id for artifact in self.oracle_artifacts}
        if any(case.oracle_artifact_id not in artifact_ids for case in self.cases if case.oracle_artifact_id is not None):
            raise ValueError("sensitivity case references missing oracle artifact")
        _id(self.protocol_id, "sensitivity protocol ID")
        _sha(self.family_matrix_digest, "sensitivity family matrix digest")
        if not isinstance(self.threshold, (int, float)) or not 0.0 < self.threshold <= 1.0:
            raise ValueError("sensitivity threshold must be in (0, 1]")
        if type(self.replicate_count) is not int or self.replicate_count < 1:
            raise ValueError("sensitivity replicate count must be positive")
        expected = f"sensitivity-matrix.{_digest(self.identity_payload())[:40]}"
        if self.matrix_id != expected:
            raise ValueError("sensitivity matrix ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "panel": self.panel.value,
            "target_kind": self.target_kind.value,
            "family_ids": list(self.family_ids),
            "cases": [case.payload() for case in self.cases],
            "oracle_artifacts": [artifact.payload() for artifact in self.oracle_artifacts],
            "protocol_id": self.protocol_id,
            "family_matrix_digest": self.family_matrix_digest,
            "threshold": self.threshold,
            "replicate_count": self.replicate_count,
        }

    @property
    def matrix_digest(self) -> str:
        return _digest(self.identity_payload())

    def payload(self) -> dict[str, object]:
        return {"matrix_id": self.matrix_id, **self.identity_payload()}

    def method_visible_case(self, family_id: str, condition: SensitivityCondition | str) -> dict[str, object]:
        matches = [case for case in self.cases if case.family_id == family_id and case.condition == SensitivityCondition(condition)]
        if len(matches) != 1:
            raise ValueError("sensitivity case is not registered")
        return matches[0].method_visible_payload()

    @classmethod
    def create_for_panel(
        cls,
        *,
        panel: SensitivityPanel,
        protocol: ResearchProtocol,
        family_matrix: PastFamilyMatrix,
    ) -> "SensitivityMatrix":
        panel = SensitivityPanel(panel)
        if family_matrix.matrix_id != protocol.family_matrix_id or family_matrix.matrix_digest != protocol.family_matrix_digest:
            raise ValueError("sensitivity matrix family identity does not match protocol")
        target_specs = tuple(
            spec for spec in family_matrix.families
            if spec.panel is panel.family_panel and spec.role is FamilyRole.TARGET
        )
        if not target_specs:
            raise ValueError("sensitivity panel has no target families")
        target_kind = panel.memory_kind
        oracle_fields = {
            SensitivityPanel.SEMANTIC: ("fact", "scope", "validity"),
            SensitivityPanel.EPISODIC: ("episode_id", "context", "outcome", "provenance"),
            SensitivityPanel.PROCEDURAL: ("applicability", "steps", "version", "validation"),
        }[panel]
        oracle_identity = {
            "panel": panel.value,
            "target_kind": target_kind.value,
            "mechanism": "type_matched",
            "minimal_field_ids": list(oracle_fields),
            "protocol_id": protocol.protocol_id,
        }
        oracle_digest = _digest(oracle_identity)
        oracle = OracleArtifact(
            artifact_id=f"oracle.{panel.value}.{oracle_digest[:32]}",
            panel=panel,
            target_kind=target_kind,
            mechanism="type_matched",
            minimal_field_ids=oracle_fields,
            content_digest=oracle_digest,
        )
        cases: list[SensitivityCase] = []
        for spec in target_specs:
            for condition in SensitivityCondition:
                condition_contract = next(value for value in protocol.conditions if value.condition_id is condition)
                if condition is SensitivityCondition.TYPE_MATCHED_ORACLE and condition_contract.target_kind is not target_kind:
                    raise ValueError("protocol oracle target kind does not match panel")
                identity = {"panel": panel.value, "family_id": spec.family_id, "condition": condition.value, "protocol": protocol.protocol_id}
                case_digest = _digest(identity)
                cases.append(SensitivityCase(
                    case_id=f"sensitivity-case.{case_digest[:40]}",
                    panel=panel,
                    family_id=spec.family_id,
                    condition=condition,
                    target_kind=target_kind,
                    state_identity=f"state.{panel.value}.{case_digest[:32]}",
                    oracle_artifact_id=oracle.artifact_id if condition is SensitivityCondition.TYPE_MATCHED_ORACLE else None,
                    mechanism=condition_contract.mechanism,
                ))
        values = {
            "panel": panel.value,
            "target_kind": target_kind.value,
            "family_ids": [spec.family_id for spec in target_specs],
            "cases": [case.payload() for case in cases],
            "oracle_artifacts": [oracle.payload()],
            "protocol_id": protocol.protocol_id,
            "family_matrix_digest": family_matrix.matrix_digest,
            "threshold": protocol.metric.practical_improvement_threshold,
            "replicate_count": protocol.metric.replicate_count,
            "schema": SENSITIVITY_SCHEMA,
            "schema_version": SENSITIVITY_SCHEMA_VERSION,
        }
        return cls(
            matrix_id=f"sensitivity-matrix.{_digest(values)[:40]}",
            panel=panel,
            target_kind=target_kind,
            family_ids=tuple(spec.family_id for spec in target_specs),
            cases=tuple(cases),
            oracle_artifacts=(oracle,),
            protocol_id=protocol.protocol_id,
            family_matrix_digest=family_matrix.matrix_digest,
            threshold=protocol.metric.practical_improvement_threshold,
            replicate_count=protocol.metric.replicate_count,
        )


__all__ = [
    "OracleArtifact",
    "SENSITIVITY_SCHEMA",
    "SENSITIVITY_SCHEMA_VERSION",
    "SensitivityCase",
    "SensitivityMatrix",
    "SensitivityPanel",
]
