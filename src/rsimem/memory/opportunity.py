"""Host-neutral, deployment-visible opportunity evidence.

An opportunity is not inferred from a benchmark family or lifecycle stage.  It
must be emitted by a visible surface (input, environment, tool schema, user
request, or a frozen application-owned schema) and is kept separate from the
later memory-use/outcome attribution decision.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Mapping

from .evidence_planes import (
    EvidencePlane,
    EvidenceSourceKind,
    validate_plane_source,
    validate_pure_process_payload,
)


OPPORTUNITY_SCHEMA_VERSION = 1
OPPORTUNITY_SCHEMA = "rsimem-opportunity-evidence-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,255}$")
_REQUIREMENT = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


def _id(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")
    return value


def _require_requirement(value: object, name: str = "semantic requirement") -> str:
    if not isinstance(value, str) or _REQUIREMENT.fullmatch(value) is None:
        raise ValueError(f"{name} must be normalized and stable")
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be sha256")
    return value


def _timestamp(value: object, name: str) -> str:
    if not isinstance(value, str) or _ISO_UTC.fullmatch(value) is None:
        raise ValueError(f"{name} must be an ISO UTC timestamp")
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO UTC timestamp") from exc
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _has_observable_payload(value: object) -> bool:
    """Return whether a source carries a concrete visible observation."""

    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return bool(value)
    return True


class OpportunitySurface(StrEnum):
    CURRENT_INPUT = "current_input"
    ENVIRONMENT_STATE = "environment_state"
    TOOL_SCHEMA = "tool_schema"
    USER_REQUEST = "user_request"
    APPLICATION_SCHEMA = "application_schema"


class OpportunityResolutionStatus(StrEnum):
    OBSERVED = "observed"
    CURRENT_INPUT_CONFOUNDED = "current_input_confounded"
    UNRESOLVED = "unresolved"
    CENSORED = "censored"


@dataclass(frozen=True, slots=True)
class ApplicationOpportunitySchema:
    """A public, versioned schema owned by the application, not a benchmark."""

    schema_id: str
    version: str
    requirement_ids: tuple[str, ...]
    schema_digest: str
    schema_version: int = OPPORTUNITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OPPORTUNITY_SCHEMA_VERSION:
            raise ValueError("unsupported application opportunity schema")
        _id(self.schema_id, "application opportunity schema ID")
        _id(self.version, "application opportunity schema version")
        if not self.requirement_ids:
            raise ValueError("application opportunity schema requires requirements")
        if len(self.requirement_ids) != len(set(self.requirement_ids)):
            raise ValueError("application opportunity requirements must be unique")
        for value in self.requirement_ids:
            _require_requirement(value)
        _digest(self.schema_digest, "application opportunity schema digest")
        expected = self._digest_for(
            self.schema_id, self.version, self.requirement_ids, self.schema_version
        )
        if self.schema_digest != expected:
            raise ValueError("application opportunity schema digest mismatch")

    @staticmethod
    def _digest_for(
        schema_id: str,
        version: str,
        requirement_ids: tuple[str, ...],
        schema_version: int,
    ) -> str:
        payload = {
            "schema_version": schema_version,
            "schema_id": schema_id,
            "version": version,
            "requirement_ids": list(requirement_ids),
        }
        return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()

    @classmethod
    def create(
        cls,
        *,
        schema_id: str,
        version: str,
        requirement_ids: tuple[str, ...],
    ) -> "ApplicationOpportunitySchema":
        values = tuple(requirement_ids)
        digest = cls._digest_for(schema_id, version, values, OPPORTUNITY_SCHEMA_VERSION)
        return cls(schema_id, version, values, digest)

    def permits(self, requirement_id: str) -> bool:
        return requirement_id in self.requirement_ids

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "schema_id": self.schema_id,
            "version": self.version,
            "requirement_ids": list(self.requirement_ids),
            "schema_digest": self.schema_digest,
        }


@dataclass(frozen=True, slots=True)
class OpportunityEvidence:
    evidence_id: str
    source_surface: OpportunitySurface
    semantic_requirement: str
    observation_time: str
    operation_id: str
    provenance_id: str
    source_digest: str
    application_schema_id: str | None = None
    application_schema_digest: str | None = None
    schema_version: int = OPPORTUNITY_SCHEMA_VERSION
    evidence_plane: EvidencePlane = EvidencePlane.PURE_PROCESS
    evidence_source: EvidenceSourceKind = EvidenceSourceKind.RUNTIME_OBSERVATION

    def __post_init__(self) -> None:
        if self.schema_version != OPPORTUNITY_SCHEMA_VERSION:
            raise ValueError("unsupported opportunity evidence schema")
        plane, source = validate_plane_source(self.evidence_plane, self.evidence_source)
        if plane != EvidencePlane.PURE_PROCESS:
            raise ValueError("opportunity evidence must be pure_process")
        object.__setattr__(self, "evidence_plane", plane)
        object.__setattr__(self, "evidence_source", source)
        object.__setattr__(self, "source_surface", OpportunitySurface(self.source_surface))
        if self.source_surface is OpportunitySurface.APPLICATION_SCHEMA:
            if source is not EvidenceSourceKind.APPLICATION_CONTRACT:
                raise ValueError(
                    "application-schema opportunity requires application contract source"
                )
        elif source is EvidenceSourceKind.APPLICATION_CONTRACT:
            raise ValueError(
                "application contract source requires application-schema surface"
            )
        _id(self.operation_id, "opportunity operation ID")
        _id(self.provenance_id, "opportunity provenance ID")
        _require_requirement(self.semantic_requirement)
        _timestamp(self.observation_time, "opportunity observation time")
        _digest(self.source_digest, "opportunity source digest")
        if self.source_surface == OpportunitySurface.APPLICATION_SCHEMA:
            if self.application_schema_id is None or self.application_schema_digest is None:
                raise ValueError("application opportunity requires schema identity")
        if self.application_schema_id is not None:
            _id(self.application_schema_id, "application opportunity schema ID")
        if self.application_schema_digest is not None:
            _digest(self.application_schema_digest, "application opportunity schema digest")
        expected = f"opportunity-evidence.{self._identity_digest()[:40]}"
        if self.evidence_id != expected:
            raise ValueError("opportunity evidence ID mismatch")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_surface": self.source_surface.value,
            "semantic_requirement": self.semantic_requirement,
            "observation_time": self.observation_time,
            "operation_id": self.operation_id,
            "provenance_id": self.provenance_id,
            "source_digest": self.source_digest,
            "application_schema_id": self.application_schema_id,
            "application_schema_digest": self.application_schema_digest,
            "evidence_plane": self.evidence_plane.value,
            "evidence_source": self.evidence_source.value,
        }

    def _identity_digest(self) -> str:
        return hashlib.sha256(_canonical(self._identity_payload()).encode("utf-8")).hexdigest()

    @classmethod
    def create(
        cls,
        *,
        source_surface: OpportunitySurface | str,
        semantic_requirement: str,
        observation_time: str,
        operation_id: str,
        provenance_id: str,
        source_payload: object,
        application_schema: ApplicationOpportunitySchema | None = None,
    ) -> "OpportunityEvidence":
        surface = OpportunitySurface(source_surface)
        # Even deployment-visible payloads are untrusted at this boundary:
        # reject benchmark/grader metadata before deriving the source digest.
        validate_pure_process_payload(source_payload)
        if surface == OpportunitySurface.APPLICATION_SCHEMA:
            if application_schema is None or not application_schema.permits(semantic_requirement):
                raise ValueError("application opportunity is not in the frozen schema")
        elif application_schema is not None:
            raise ValueError("application schema identity requires application surface")
        if not _has_observable_payload(source_payload):
            raise ValueError("opportunity source payload must contain an observation")
        source_digest = hashlib.sha256(_canonical(source_payload).encode("utf-8")).hexdigest()
        values = {
            "source_surface": surface,
            "semantic_requirement": _require_requirement(semantic_requirement),
            "observation_time": observation_time,
            "operation_id": operation_id,
            "provenance_id": provenance_id,
            "source_digest": source_digest,
            "application_schema_id": application_schema.schema_id if application_schema else None,
            "application_schema_digest": application_schema.schema_digest if application_schema else None,
            "schema_version": OPPORTUNITY_SCHEMA_VERSION,
            "evidence_plane": EvidencePlane.PURE_PROCESS,
            "evidence_source": (
                EvidenceSourceKind.APPLICATION_CONTRACT
                if application_schema is not None
                else EvidenceSourceKind.RUNTIME_OBSERVATION
            ),
        }
        identity = {
            "schema_version": values["schema_version"],
            "source_surface": surface.value,
            "semantic_requirement": values["semantic_requirement"],
            "observation_time": values["observation_time"],
            "operation_id": values["operation_id"],
            "provenance_id": values["provenance_id"],
            "source_digest": source_digest,
            "application_schema_id": values["application_schema_id"],
            "application_schema_digest": values["application_schema_digest"],
            "evidence_plane": EvidencePlane.PURE_PROCESS.value,
            "evidence_source": values["evidence_source"].value,
        }
        evidence_id = "opportunity-evidence." + hashlib.sha256(
            _canonical(identity).encode("utf-8")
        ).hexdigest()[:40]
        return cls(evidence_id=evidence_id, **values)

    def payload(self) -> dict[str, object]:
        return {
            "schema": OPPORTUNITY_SCHEMA,
            **self._identity_payload(),
            "evidence_id": self.evidence_id,
        }

    @classmethod
    def from_payload(cls, value: object) -> "OpportunityEvidence":
        fields = {
            "schema", "schema_version", "evidence_id", "source_surface",
            "semantic_requirement", "observation_time", "operation_id",
            "provenance_id", "source_digest", "application_schema_id",
            "application_schema_digest", "evidence_plane", "evidence_source",
        }
        if not isinstance(value, Mapping) or set(value) != fields or value["schema"] != OPPORTUNITY_SCHEMA:
            raise ValueError("malformed opportunity evidence")
        try:
            return cls(
                evidence_id=value["evidence_id"],
                source_surface=OpportunitySurface(value["source_surface"]),
                semantic_requirement=value["semantic_requirement"],
                observation_time=value["observation_time"],
                operation_id=value["operation_id"],
                provenance_id=value["provenance_id"],
                source_digest=value["source_digest"],
                application_schema_id=value["application_schema_id"],
                application_schema_digest=value["application_schema_digest"],
                schema_version=value["schema_version"],
                evidence_plane=EvidencePlane(value["evidence_plane"]),
                evidence_source=EvidenceSourceKind(value["evidence_source"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed opportunity evidence") from exc


@dataclass(frozen=True, slots=True)
class OpportunityResolution:
    evidence_id: str
    status: OpportunityResolutionStatus
    reason_code: str
    current_input_confounded: bool
    observation_complete: bool

    def __post_init__(self) -> None:
        _id(self.evidence_id, "opportunity evidence ID")
        object.__setattr__(self, "status", OpportunityResolutionStatus(self.status))
        if type(self.current_input_confounded) is not bool or type(self.observation_complete) is not bool:
            raise TypeError("opportunity resolution flags must be bool")
        if not self.reason_code or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.reason_code):
            raise ValueError("opportunity resolution reason is invalid")

    def payload(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "current_input_confounded": self.current_input_confounded,
            "observation_complete": self.observation_complete,
        }


def resolve_opportunity(
    evidence: OpportunityEvidence | None,
    *,
    current_input_requirements: tuple[str, ...] = (),
    observation_complete: bool = True,
) -> OpportunityResolution:
    """Resolve only visible evidence; absence never becomes an opportunity."""

    if type(observation_complete) is not bool:
        raise TypeError("opportunity observation completeness must be bool")
    if evidence is None:
        return OpportunityResolution(
            "opportunity.none",
            OpportunityResolutionStatus.UNRESOLVED,
            "opportunity_not_observed",
            False,
            observation_complete,
        )
    for value in current_input_requirements:
        _require_requirement(value, "current input requirement")
    if not observation_complete:
        return OpportunityResolution(
            evidence.evidence_id,
            OpportunityResolutionStatus.CENSORED,
            "observation_censored",
            False,
            False,
        )
    if evidence.semantic_requirement in set(current_input_requirements):
        return OpportunityResolution(
            evidence.evidence_id,
            OpportunityResolutionStatus.CURRENT_INPUT_CONFOUNDED,
            "current_input_confounded",
            True,
            True,
        )
    return OpportunityResolution(
        evidence.evidence_id,
        OpportunityResolutionStatus.OBSERVED,
        "opportunity_observed",
        False,
        True,
    )


__all__ = [
    "ApplicationOpportunitySchema",
    "OPPORTUNITY_SCHEMA",
    "OpportunityEvidence",
    "OpportunityResolution",
    "OpportunityResolutionStatus",
    "OpportunitySurface",
    "resolve_opportunity",
]
