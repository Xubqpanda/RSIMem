"""Fail-closed stale artifact and revocation registry."""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

from .evidence_planes import EvidencePlane, EvidenceSourceKind, validate_plane_source


REVOCATION_SCHEMA_VERSION = 1
REVOCATION_SCHEMA = "rsimem-revocation-registry-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,255}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    import hashlib
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _id(value: object, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")


def _sha(value: object, name: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be sha256")


def _timestamp(value: object, name: str) -> None:
    if not isinstance(value, str) or _ISO_UTC.fullmatch(value) is None:
        raise ValueError(f"{name} must be an ISO UTC timestamp")
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO UTC timestamp") from exc


@dataclass(frozen=True, slots=True)
class RevocationEntry:
    revocation_id: str
    artifact_id: str
    artifact_schema_version: int
    artifact_digest: str
    evidence_plane: EvidencePlane
    evidence_source: EvidenceSourceKind
    revoked_at: str
    reason_code: str
    schema_version: int = REVOCATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REVOCATION_SCHEMA_VERSION:
            raise ValueError("unsupported revocation entry schema")
        _id(self.revocation_id, "revocation ID")
        _id(self.artifact_id, "revoked artifact ID")
        if type(self.artifact_schema_version) is not int or self.artifact_schema_version < 1:
            raise ValueError("artifact schema version must be a positive integer")
        _sha(self.artifact_digest, "revoked artifact digest")
        validate_plane_source(self.evidence_plane, self.evidence_source)
        _timestamp(self.revoked_at, "revocation timestamp")
        if not isinstance(self.reason_code, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.reason_code):
            raise ValueError("revocation reason is invalid")
        expected = f"revocation.{_digest(self._identity_payload())[:40]}"
        if self.revocation_id != expected:
            raise ValueError("revocation ID mismatch")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "artifact_schema_version": self.artifact_schema_version,
            "artifact_digest": self.artifact_digest,
            "evidence_plane": EvidencePlane(self.evidence_plane).value,
            "evidence_source": EvidenceSourceKind(self.evidence_source).value,
            "revoked_at": self.revoked_at,
            "reason_code": self.reason_code,
        }

    @classmethod
    def create(
        cls,
        *,
        artifact_id: str,
        artifact_schema_version: int,
        artifact_digest: str,
        evidence_plane: EvidencePlane | str,
        evidence_source: EvidenceSourceKind | str,
        revoked_at: str,
        reason_code: str,
    ) -> "RevocationEntry":
        values = {
            "artifact_id": artifact_id,
            "artifact_schema_version": artifact_schema_version,
            "artifact_digest": artifact_digest,
            "evidence_plane": EvidencePlane(evidence_plane),
            "evidence_source": EvidenceSourceKind(evidence_source),
            "revoked_at": revoked_at,
            "reason_code": reason_code,
            "schema_version": REVOCATION_SCHEMA_VERSION,
        }
        return cls(revocation_id=f"revocation.{_digest(values)[:40]}", **values)

    def payload(self) -> dict[str, object]:
        return {"schema": REVOCATION_SCHEMA, "revocation_id": self.revocation_id, **self._identity_payload()}

    @classmethod
    def from_payload(cls, value: object) -> "RevocationEntry":
        fields = {
            "schema", "revocation_id", "schema_version", "artifact_id",
            "artifact_schema_version", "artifact_digest", "evidence_plane",
            "evidence_source", "revoked_at", "reason_code",
        }
        if not isinstance(value, Mapping) or set(value) != fields or value.get("schema") != REVOCATION_SCHEMA:
            raise ValueError("malformed revocation entry")
        try:
            return cls(
                revocation_id=value["revocation_id"], artifact_id=value["artifact_id"],
                artifact_schema_version=value["artifact_schema_version"], artifact_digest=value["artifact_digest"],
                evidence_plane=value["evidence_plane"], evidence_source=value["evidence_source"],
                revoked_at=value["revoked_at"], reason_code=value["reason_code"],
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed revocation entry") from exc


class JsonRevocationRegistry:
    """Append-only registry; missing/corrupt state is never treated as empty."""

    def __init__(self, path: Path) -> None:
        # Keep the final component unresolved so a symlink cannot redirect
        # revocation state into an unrelated trusted-looking file.
        self.path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    def _reject_symlink(self) -> None:
        if self.path.is_symlink():
            raise ValueError("revocation registry cannot be a symlink")

    @contextmanager
    def _lock(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read(self) -> dict[str, str]:
        self._reject_symlink()
        if not self.path.exists():
            raise ValueError("revocation registry is missing")
        records: dict[str, str] = {}
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                entry = RevocationEntry.from_payload(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"malformed revocation registry at line {line_number}") from exc
            canonical = _canonical(entry.payload())
            previous = records.get(entry.revocation_id)
            if previous is not None and previous != canonical:
                raise ValueError("conflicting revocation entry")
            records[entry.revocation_id] = canonical
        return records

    def initialize(self) -> None:
        """Create an explicitly empty registry before it is required."""

        with self._lock():
            self._reject_symlink()
            if self.path.exists():
                self._read()
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

    def append(self, entry: RevocationEntry) -> bool:
        if not isinstance(entry, RevocationEntry):
            raise TypeError("revocation entry has the wrong type")
        serialized = _canonical(entry.payload())
        with self._lock():
            records = self._read()
            previous = records.get(entry.revocation_id)
            if previous is not None:
                if previous != serialized:
                    raise ValueError("conflicting revocation entry")
                return False
            # Artifact IDs are content-addressed identities.  Two entries for
            # the same artifact/schema must agree on its digest and evidence
            # plane; otherwise a malformed registry could make one revoked
            # incarnation look active.
            for existing_serialized in records.values():
                existing = RevocationEntry.from_payload(
                    json.loads(existing_serialized)
                )
                if (
                    existing.artifact_id == entry.artifact_id
                    and existing.artifact_schema_version
                    == entry.artifact_schema_version
                    and (
                        existing.artifact_digest != entry.artifact_digest
                        or existing.evidence_plane != entry.evidence_plane
                        or existing.evidence_source != entry.evidence_source
                    )
                ):
                    raise ValueError("conflicting revocation identity")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return True

    def assert_active(
        self,
        *,
        artifact_id: str,
        artifact_schema_version: int,
        artifact_digest: str,
        evidence_plane: EvidencePlane | str,
        evidence_source: EvidenceSourceKind | str,
    ) -> None:
        _id(artifact_id, "artifact ID")
        _sha(artifact_digest, "artifact digest")
        plane, source = validate_plane_source(evidence_plane, evidence_source)
        if type(artifact_schema_version) is not int or artifact_schema_version < 1:
            raise ValueError("artifact schema version must be a positive integer")
        with self._lock():
            records = self._read()
        for serialized in records.values():
            entry = RevocationEntry.from_payload(json.loads(serialized))
            if (
                entry.artifact_id == artifact_id
                and entry.artifact_schema_version == artifact_schema_version
                and (
                    entry.artifact_digest != artifact_digest
                    or entry.evidence_plane != plane
                    or entry.evidence_source != source
                )
            ):
                raise ValueError("conflicting revocation identity")
            if (
                entry.artifact_id == artifact_id
                and entry.artifact_schema_version == artifact_schema_version
                and entry.artifact_digest == artifact_digest
                and entry.evidence_plane == plane
                and entry.evidence_source == source
            ):
                raise ValueError("artifact is revoked")


__all__ = [
    "REVOCATION_SCHEMA",
    "REVOCATION_SCHEMA_VERSION",
    "JsonRevocationRegistry",
    "RevocationEntry",
]
