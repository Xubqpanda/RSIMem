"""Versioned set-level semantic provenance.

Durable rules are sometimes split into several extracted facts.  This module
keeps their semantic unit explicit: a complete set can receive one primary
attribution, while partial retrieval or exposure remains unresolved.  No
benchmark family or grader metadata is needed to construct a binding.
"""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from .evidence_planes import EvidencePlane, EvidenceSourceKind, validate_plane_source
from ._atomic_jsonl import replace_jsonl


ARTIFACT_SET_SCHEMA_VERSION = 1
ARTIFACT_SET_SCHEMA = "rsimem-artifact-set-semantic-binding-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,255}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SEMANTIC_KEY = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _id(value: object, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")


def _ids(values: tuple[str, ...], name: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if not values:
        raise ValueError(f"{name} must not be empty")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")
    for value in values:
        _id(value, name)


def _sha(value: object, name: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be sha256")


class ArtifactSetResolutionStatus(StrEnum):
    COMPLETE = "complete"
    UNRESOLVED = "unresolved"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class ArtifactSetSemanticBinding:
    binding_id: str
    semantic_unit_id: str
    semantic_key: str | None
    member_artifact_ids: tuple[str, ...]
    member_fact_ids: tuple[str, ...]
    complete: bool
    source_digest: str
    provenance_id: str
    matcher_version: str | None = None
    equivalence_digest: str | None = None
    schema_version: int = ARTIFACT_SET_SCHEMA_VERSION
    evidence_plane: EvidencePlane = EvidencePlane.PURE_PROCESS
    evidence_source: EvidenceSourceKind = EvidenceSourceKind.RUNTIME_OBSERVATION

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_SET_SCHEMA_VERSION:
            raise ValueError("unsupported artifact-set binding schema")
        plane, source = validate_plane_source(self.evidence_plane, self.evidence_source)
        if plane != EvidencePlane.PURE_PROCESS or source != EvidenceSourceKind.RUNTIME_OBSERVATION:
            raise ValueError("artifact-set binding must be pure_process runtime evidence")
        object.__setattr__(self, "evidence_plane", plane)
        object.__setattr__(self, "evidence_source", source)
        _id(self.binding_id, "artifact-set binding ID")
        _id(self.semantic_unit_id, "semantic unit ID")
        _ids(self.member_artifact_ids, "member artifact IDs")
        _ids(self.member_fact_ids, "member fact IDs")
        if type(self.complete) is not bool:
            raise TypeError("artifact-set completeness must be bool")
        if self.semantic_key is not None and (
            not isinstance(self.semantic_key, str)
            or _SEMANTIC_KEY.fullmatch(self.semantic_key) is None
        ):
            raise ValueError("semantic key must be normalized and stable")
        _sha(self.source_digest, "artifact-set source digest")
        _id(self.provenance_id, "artifact-set provenance ID")
        if self.matcher_version is not None:
            _id(self.matcher_version, "artifact-set matcher version")
        if self.equivalence_digest is not None:
            _sha(self.equivalence_digest, "artifact-set equivalence digest")
        if self.matcher_version is not None and self.equivalence_digest is None:
            raise ValueError("matcher evidence requires equivalence digest")
        digest = _digest(self._identity_payload())
        if self.binding_id != f"artifact-set-binding.{digest[:40]}":
            raise ValueError("artifact-set binding ID mismatch")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "semantic_unit_id": self.semantic_unit_id,
            "semantic_key": self.semantic_key,
            "member_artifact_ids": list(self.member_artifact_ids),
            "member_fact_ids": list(self.member_fact_ids),
            "complete": self.complete,
            "source_digest": self.source_digest,
            "provenance_id": self.provenance_id,
            "matcher_version": self.matcher_version,
            "equivalence_digest": self.equivalence_digest,
            "evidence_plane": self.evidence_plane.value,
            "evidence_source": self.evidence_source.value,
        }

    @classmethod
    def create(
        cls,
        *,
        semantic_unit_id: str,
        member_artifact_ids: tuple[str, ...],
        member_fact_ids: tuple[str, ...],
        complete: bool,
        source_digest: str,
        provenance_id: str,
        semantic_key: str | None = None,
        matcher_version: str | None = None,
        equivalence_digest: str | None = None,
    ) -> "ArtifactSetSemanticBinding":
        values: dict[str, object] = {
            "semantic_unit_id": semantic_unit_id,
            "semantic_key": semantic_key,
            "member_artifact_ids": tuple(member_artifact_ids),
            "member_fact_ids": tuple(member_fact_ids),
            "complete": complete,
            "source_digest": source_digest,
            "provenance_id": provenance_id,
            "matcher_version": matcher_version,
            "equivalence_digest": equivalence_digest,
            "schema_version": ARTIFACT_SET_SCHEMA_VERSION,
            "evidence_plane": EvidencePlane.PURE_PROCESS,
            "evidence_source": EvidenceSourceKind.RUNTIME_OBSERVATION,
        }
        digest = _digest(values)
        return cls(binding_id=f"artifact-set-binding.{digest[:40]}", **values)

    @property
    def primary_unit_id(self) -> str:
        """Stable single primary identity for the whole semantic unit."""

        return f"semantic-unit.{_digest({'schema_version': self.schema_version, 'semantic_unit_id': self.semantic_unit_id})[:40]}"

    def payload(self) -> dict[str, object]:
        return {"schema": ARTIFACT_SET_SCHEMA, "binding_id": self.binding_id, **self._identity_payload()}

    @classmethod
    def from_payload(cls, value: object) -> "ArtifactSetSemanticBinding":
        fields = {
            "schema", "binding_id", "schema_version", "semantic_unit_id", "semantic_key",
            "member_artifact_ids", "member_fact_ids", "complete", "source_digest",
            "provenance_id", "matcher_version", "equivalence_digest", "evidence_plane",
            "evidence_source",
        }
        if not isinstance(value, Mapping) or set(value) != fields or value.get("schema") != ARTIFACT_SET_SCHEMA:
            raise ValueError("malformed artifact-set semantic binding")
        if not isinstance(value["member_artifact_ids"], list) or not isinstance(value["member_fact_ids"], list):
            raise ValueError("malformed artifact-set member collections")
        try:
            return cls(
                binding_id=value["binding_id"],
                semantic_unit_id=value["semantic_unit_id"],
                semantic_key=value["semantic_key"],
                member_artifact_ids=tuple(value["member_artifact_ids"]),
                member_fact_ids=tuple(value["member_fact_ids"]),
                complete=value["complete"],
                source_digest=value["source_digest"],
                provenance_id=value["provenance_id"],
                matcher_version=value["matcher_version"],
                equivalence_digest=value["equivalence_digest"],
                schema_version=value["schema_version"],
                evidence_plane=value["evidence_plane"],
                evidence_source=value["evidence_source"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed artifact-set semantic binding") from exc


class JsonArtifactSetBindingLog:
    """Crash-safe append-only storage for runtime set-level bindings."""

    def __init__(self, path: Path) -> None:
        # Preserve the final component so an owner-controlled binding log
        # cannot be redirected by a symlink.
        self.path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self._records: dict[str, str] = {}
        self._load()

    @staticmethod
    def _canonical(value: Mapping[str, object]) -> str:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    def _load(self) -> None:
        if self.path.is_symlink():
            raise ValueError("artifact-set binding log cannot be a symlink")
        if not self.path.exists():
            return
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                binding = ArtifactSetSemanticBinding.from_payload(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"malformed artifact-set binding at line {line_number}"
                ) from exc
            canonical = self._canonical(binding.payload())
            previous = self._records.get(binding.binding_id)
            if previous is not None and previous != canonical:
                raise ValueError("conflicting artifact-set binding")
            self._records[binding.binding_id] = canonical

    def records(self) -> tuple[ArtifactSetSemanticBinding, ...]:
        if self.path.is_symlink():
            raise ValueError("artifact-set binding log cannot be a symlink")
        if self.lock_path.is_symlink():
            raise ValueError("artifact-set binding lock cannot be a symlink")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            try:
                self._records.clear()
                self._load()
                return tuple(
                    ArtifactSetSemanticBinding.from_payload(json.loads(value))
                    for _, value in sorted(self._records.items())
                )
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def append(self, binding: ArtifactSetSemanticBinding) -> bool:
        if not isinstance(binding, ArtifactSetSemanticBinding):
            raise TypeError("artifact-set log accepts ArtifactSetSemanticBinding only")
        serialized = self._canonical(binding.payload())
        if self.path.is_symlink():
            raise ValueError("artifact-set binding log cannot be a symlink")
        if self.lock_path.is_symlink():
            raise ValueError("artifact-set binding lock cannot be a symlink")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                self._records.clear()
                self._load()
                previous = self._records.get(binding.binding_id)
                if previous is not None:
                    if previous != serialized:
                        raise ValueError("conflicting artifact-set binding")
                    return False
                replace_jsonl(
                    self.path,
                    tuple(value for _, value in sorted(self._records.items()))
                    + (serialized,),
                    error_name="artifact-set binding log",
                )
                self._records[binding.binding_id] = serialized
                return True
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True, slots=True)
class ArtifactSetResolution:
    binding_id: str
    status: ArtifactSetResolutionStatus
    reason_code: str
    retrieved_complete: bool
    exposed_complete: bool
    primary: bool

    def __post_init__(self) -> None:
        _id(self.binding_id, "artifact-set binding ID")
        object.__setattr__(self, "status", ArtifactSetResolutionStatus(self.status))
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.reason_code):
            raise ValueError("artifact-set resolution reason is invalid")
        for value in (self.retrieved_complete, self.exposed_complete, self.primary):
            if type(value) is not bool:
                raise TypeError("artifact-set resolution flags must be bool")

    def payload(self) -> dict[str, object]:
        return {
            "binding_id": self.binding_id,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "retrieved_complete": self.retrieved_complete,
            "exposed_complete": self.exposed_complete,
            "primary": self.primary,
        }


def resolve_artifact_set(
    binding: ArtifactSetSemanticBinding,
    *,
    retrieved_member_artifact_ids: tuple[str, ...],
    exposed_member_artifact_ids: tuple[str, ...],
    observed_source_digest: str | None = None,
) -> ArtifactSetResolution:
    """Require complete, exact member retrieval and exposure for attribution."""

    if not isinstance(binding, ArtifactSetSemanticBinding):
        raise TypeError("artifact-set resolver requires ArtifactSetSemanticBinding")
    retrieved = tuple(retrieved_member_artifact_ids)
    exposed = tuple(exposed_member_artifact_ids)
    _ids(retrieved, "retrieved member artifact IDs") if retrieved else None
    _ids(exposed, "exposed member artifact IDs") if exposed else None
    members = set(binding.member_artifact_ids)
    if observed_source_digest is not None:
        _sha(observed_source_digest, "observed source digest")
        if observed_source_digest != binding.source_digest:
            return ArtifactSetResolution(
                binding.binding_id,
                ArtifactSetResolutionStatus.AMBIGUOUS,
                "member_source_mismatch",
                False,
                False,
                False,
            )
    if not binding.complete:
        return ArtifactSetResolution(binding.binding_id, ArtifactSetResolutionStatus.UNRESOLVED, "binding_incomplete", False, False, False)
    if not set(retrieved).issubset(members):
        return ArtifactSetResolution(binding.binding_id, ArtifactSetResolutionStatus.AMBIGUOUS, "retrieval_member_mismatch", False, False, False)
    if set(retrieved) != members:
        return ArtifactSetResolution(binding.binding_id, ArtifactSetResolutionStatus.UNRESOLVED, "partial_retrieval", False, False, False)
    if not set(exposed).issubset(members):
        return ArtifactSetResolution(binding.binding_id, ArtifactSetResolutionStatus.AMBIGUOUS, "exposure_member_mismatch", True, False, False)
    if set(exposed) != members:
        return ArtifactSetResolution(binding.binding_id, ArtifactSetResolutionStatus.UNRESOLVED, "partial_exposure", True, False, False)
    return ArtifactSetResolution(binding.binding_id, ArtifactSetResolutionStatus.COMPLETE, "complete_member_set", True, True, True)


__all__ = [
    "ARTIFACT_SET_SCHEMA",
    "ARTIFACT_SET_SCHEMA_VERSION",
    "ArtifactSetResolution",
    "ArtifactSetResolutionStatus",
    "ArtifactSetSemanticBinding",
    "JsonArtifactSetBindingLog",
    "resolve_artifact_set",
]
