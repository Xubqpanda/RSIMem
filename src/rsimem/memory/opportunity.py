"""Host-neutral, deployment-visible opportunity evidence.

An opportunity is not inferred from a benchmark family or lifecycle stage.  It
must be emitted by a visible surface (input, environment, tool schema, user
request, or a frozen application-owned schema) and is kept separate from the
later memory-use/outcome attribution decision.
"""

from __future__ import annotations

import hashlib
import json
import fcntl
import os
import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Mapping
from pathlib import Path

from .evidence_planes import (
    EvidencePlane,
    EvidenceSourceKind,
    validate_plane_source,
    validate_pure_process_payload,
)
from ._atomic_jsonl import replace_jsonl


OPPORTUNITY_SCHEMA_VERSION = 2
OPPORTUNITY_SCHEMA = "rsimem-opportunity-evidence-v2"
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

    @classmethod
    def from_payload(cls, value: object) -> "ApplicationOpportunitySchema":
        """Rebuild a frozen application contract without trusting its digest.

        The constructor recomputes the canonical digest and therefore rejects
        tampered requirement lists or schema identities during replay.
        """

        fields = {
            "schema_version",
            "schema_id",
            "version",
            "requirement_ids",
            "schema_digest",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("malformed application opportunity schema")
        try:
            requirement_ids = tuple(value["requirement_ids"])
        except (KeyError, TypeError) as exc:
            raise ValueError("malformed application opportunity schema") from exc
        try:
            return cls(
                schema_id=value["schema_id"],
                version=value["version"],
                requirement_ids=requirement_ids,
                schema_digest=value["schema_digest"],
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed application opportunity schema") from exc


class JsonApplicationOpportunitySchemaRegistry:
    """Append-only registry for trusted application opportunity schemas.

    An opportunity evidence payload intentionally carries only the schema ID,
    version and digest.  The registry is the ownership boundary that binds
    those strings to a previously published, immutable application contract during
    replay.  A schema ID/version pair may be registered repeatedly only when
    the canonical payload is identical; a conflicting replacement fails
    closed.
    """

    def __init__(self, path: Path) -> None:
        # Preserve the final component so a schema registry symlink fails
        # closed instead of redirecting a trusted application contract.
        self.path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self._records: dict[tuple[str, str], str] = {}
        self._load()

    @staticmethod
    def _canonical(value: Mapping[str, object]) -> str:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _key(schema: ApplicationOpportunitySchema) -> tuple[str, str]:
        return (schema.schema_id, schema.version)

    def _load(self) -> None:
        if self.path.is_symlink():
            raise ValueError("application opportunity schema registry cannot be a symlink")
        if not self.path.exists():
            return
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                schema = ApplicationOpportunitySchema.from_payload(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"malformed application opportunity schema at line {line_number}"
                ) from exc
            canonical = self._canonical(schema.payload())
            key = self._key(schema)
            previous = self._records.get(key)
            if previous is not None and previous != canonical:
                raise ValueError("conflicting application opportunity schema")
            self._records[key] = canonical

    def records(self) -> tuple[ApplicationOpportunitySchema, ...]:
        """Return the canonical registry contents in stable key order."""

        if self.path.is_symlink():
            raise ValueError("application opportunity schema registry cannot be a symlink")
        if self.lock_path.is_symlink():
            raise ValueError("application opportunity schema registry lock cannot be a symlink")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            try:
                self._records.clear()
                self._load()
                return tuple(
                    ApplicationOpportunitySchema.from_payload(json.loads(value))
                    for _, value in sorted(self._records.items())
                )
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def register(self, schema: ApplicationOpportunitySchema) -> bool:
        """Persist ``schema`` once; return ``False`` for an exact replay."""

        if not isinstance(schema, ApplicationOpportunitySchema):
            raise TypeError("schema registry accepts ApplicationOpportunitySchema only")
        serialized = self._canonical(schema.payload())
        key = self._key(schema)
        if self.path.is_symlink():
            raise ValueError("application opportunity schema registry cannot be a symlink")
        if self.lock_path.is_symlink():
            raise ValueError("application opportunity schema registry lock cannot be a symlink")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                self._records.clear()
                self._load()
                previous = self._records.get(key)
                if previous is not None:
                    if previous != serialized:
                        raise ValueError("conflicting application opportunity schema")
                    return False
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(serialized + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                self._records[key] = serialized
                return True
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def resolve(
        self,
        schema_id: str,
        version: str,
        schema_digest: str,
    ) -> ApplicationOpportunitySchema:
        """Resolve an identity and reject unknown or conflicting contracts."""

        _id(schema_id, "application opportunity schema ID")
        _id(version, "application opportunity schema version")
        _digest(schema_digest, "application opportunity schema digest")
        self.records()
        serialized = self._records.get((schema_id, version))
        if serialized is None:
            raise ValueError("unknown application opportunity schema")
        schema = ApplicationOpportunitySchema.from_payload(json.loads(serialized))
        if schema.schema_digest != schema_digest:
            raise ValueError("application opportunity schema digest conflict")
        return schema


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
    application_schema_version: str | None = None
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
            if (
                self.application_schema_id is None
                or self.application_schema_version is None
                or self.application_schema_digest is None
            ):
                raise ValueError("application opportunity requires schema identity")
        elif any(
            value is not None
            for value in (
                self.application_schema_id,
                self.application_schema_version,
                self.application_schema_digest,
            )
        ):
            raise ValueError(
                "runtime opportunity cannot carry application schema identity"
            )
        if self.application_schema_id is not None:
            _id(self.application_schema_id, "application opportunity schema ID")
        if self.application_schema_version is not None:
            _id(self.application_schema_version, "application opportunity schema version")
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
            "application_schema_version": self.application_schema_version,
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
        if application_schema is not None and not isinstance(
            application_schema, ApplicationOpportunitySchema
        ):
            raise TypeError("application opportunity schema has the wrong type")
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
            "application_schema_version": application_schema.version if application_schema else None,
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
            "application_schema_version": values["application_schema_version"],
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
    def from_payload(
        cls,
        value: object,
        *,
        application_schema: ApplicationOpportunitySchema | None = None,
    ) -> "OpportunityEvidence":
        fields = {
            "schema", "schema_version", "evidence_id", "source_surface",
            "semantic_requirement", "observation_time", "operation_id",
            "provenance_id", "source_digest", "application_schema_id",
            "application_schema_version", "application_schema_digest",
            "evidence_plane", "evidence_source",
        }
        if not isinstance(value, Mapping) or set(value) != fields or value["schema"] != OPPORTUNITY_SCHEMA:
            raise ValueError("malformed opportunity evidence")
        try:
            surface = OpportunitySurface(value["source_surface"])
            if surface is OpportunitySurface.APPLICATION_SCHEMA:
                if application_schema is None:
                    raise ValueError(
                        "application opportunity schema registry is required"
                    )
                if (
                    application_schema.schema_id != value["application_schema_id"]
                    or application_schema.version
                    != value["application_schema_version"]
                    or application_schema.schema_digest
                    != value["application_schema_digest"]
                    or not application_schema.permits(value["semantic_requirement"])
                ):
                    raise ValueError(
                        "application opportunity schema does not match evidence"
                    )
            return cls(
                evidence_id=value["evidence_id"],
                source_surface=surface,
                semantic_requirement=value["semantic_requirement"],
                observation_time=value["observation_time"],
                operation_id=value["operation_id"],
                provenance_id=value["provenance_id"],
                source_digest=value["source_digest"],
                application_schema_id=value["application_schema_id"],
                application_schema_version=value["application_schema_version"],
                application_schema_digest=value["application_schema_digest"],
                schema_version=value["schema_version"],
                evidence_plane=EvidencePlane(value["evidence_plane"]),
                evidence_source=EvidenceSourceKind(value["evidence_source"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed opportunity evidence") from exc


class JsonOpportunityEvidenceLog:
    """Crash-safe, idempotent storage for runtime opportunity evidence.

    The log stores only the content-free :class:`OpportunityEvidence` payload
    (source digest, operation identity and provenance).  It deliberately does
    not accept benchmark-family labels or final-evaluation fields.
    """

    def __init__(
        self,
        path: Path,
        *,
        schema_registry: JsonApplicationOpportunitySchemaRegistry | None = None,
    ) -> None:
        # Preserve the final component so runtime opportunity evidence cannot
        # be redirected through a symlinked log path.
        self.path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        if schema_registry is not None and not isinstance(
            schema_registry, JsonApplicationOpportunitySchemaRegistry
        ):
            raise TypeError("opportunity log schema registry has the wrong type")
        self.schema_registry = schema_registry
        self._records: dict[str, str] = {}
        self._load()

    @staticmethod
    def _canonical(value: Mapping[str, object]) -> str:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    def _load(self) -> None:
        if self.path.is_symlink():
            raise ValueError("opportunity evidence log cannot be a symlink")
        if not self.path.exists():
            return
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                application_schema = None
                if (
                    isinstance(payload, Mapping)
                    and payload.get("source_surface")
                    == OpportunitySurface.APPLICATION_SCHEMA.value
                ):
                    if self.schema_registry is None:
                        raise ValueError(
                            "application opportunity schema registry is required"
                        )
                    application_schema = self.schema_registry.resolve(
                        payload["application_schema_id"],
                        payload["application_schema_version"],
                        payload["application_schema_digest"],
                    )
                evidence = OpportunityEvidence.from_payload(
                    payload,
                    application_schema=application_schema,
                )
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"malformed opportunity evidence at line {line_number}"
                ) from exc
            canonical = self._canonical(evidence.payload())
            previous = self._records.get(evidence.evidence_id)
            if previous is not None and previous != canonical:
                raise ValueError("conflicting opportunity evidence")
            self._records[evidence.evidence_id] = canonical

    def records(self) -> tuple[OpportunityEvidence, ...]:
        if self.path.is_symlink():
            raise ValueError("opportunity evidence log cannot be a symlink")
        if self.lock_path.is_symlink():
            raise ValueError("opportunity evidence lock cannot be a symlink")
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            try:
                # Reload under the lock so an independent bridge instance can
                # observe records written since construction.
                self._records.clear()
                self._load()
                result = []
                for _, value in sorted(self._records.items()):
                    payload = json.loads(value)
                    application_schema = None
                    if (
                        payload.get("source_surface")
                        == OpportunitySurface.APPLICATION_SCHEMA.value
                    ):
                        if self.schema_registry is None:
                            raise ValueError(
                                "application opportunity schema registry is required"
                            )
                        application_schema = self.schema_registry.resolve(
                            payload["application_schema_id"],
                            payload["application_schema_version"],
                            payload["application_schema_digest"],
                        )
                    result.append(
                        OpportunityEvidence.from_payload(
                            payload,
                            application_schema=application_schema,
                        )
                    )
                return tuple(result)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def append(self, evidence: OpportunityEvidence) -> bool:
        if not isinstance(evidence, OpportunityEvidence):
            raise TypeError("opportunity log accepts OpportunityEvidence only")
        serialized = self._canonical(evidence.payload())
        if self.path.is_symlink():
            raise ValueError("opportunity evidence log cannot be a symlink")
        if self.lock_path.is_symlink():
            raise ValueError("opportunity evidence lock cannot be a symlink")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                self._records.clear()
                self._load()
                previous = self._records.get(evidence.evidence_id)
                if previous is not None:
                    if previous != serialized:
                        raise ValueError("conflicting opportunity evidence")
                    return False
                replace_jsonl(
                    self.path,
                    tuple(value for _, value in sorted(self._records.items()))
                    + (serialized,),
                    error_name="opportunity evidence log",
                )
                self._records[evidence.evidence_id] = serialized
                return True
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


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
    "JsonApplicationOpportunitySchemaRegistry",
    "OPPORTUNITY_SCHEMA",
    "OpportunityEvidence",
    "JsonOpportunityEvidenceLog",
    "OpportunityResolution",
    "OpportunityResolutionStatus",
    "OpportunitySurface",
    "resolve_opportunity",
]
