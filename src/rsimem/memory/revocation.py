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
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from .evidence_planes import EvidencePlane, EvidenceSourceKind, validate_plane_source


REVOCATION_SCHEMA_VERSION = 2
REVOCATION_SCHEMA = "rsimem-revocation-registry-v2"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,255}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


class RevocationScope(StrEnum):
    """Whether a revocation entry carries typed provenance.

    Legacy artifacts were produced before evidence planes existed.  Their
    identity must still be revocable, but assigning a current plane/source to
    them would invent provenance.  ``LEGACY_UNTYPED`` therefore matches any
    typed lookup for the same artifact identity.
    """

    TYPED = "typed"
    LEGACY_UNTYPED = "legacy_untyped"


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
    evidence_plane: EvidencePlane | None
    evidence_source: EvidenceSourceKind | None
    revoked_at: str
    reason_code: str
    scope: RevocationScope = RevocationScope.TYPED
    schema_version: int = REVOCATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REVOCATION_SCHEMA_VERSION:
            raise ValueError("unsupported revocation entry schema")
        _id(self.revocation_id, "revocation ID")
        _id(self.artifact_id, "revoked artifact ID")
        if type(self.artifact_schema_version) is not int or self.artifact_schema_version < 1:
            raise ValueError("artifact schema version must be a positive integer")
        _sha(self.artifact_digest, "revoked artifact digest")
        object.__setattr__(self, "scope", RevocationScope(self.scope))
        if self.scope is RevocationScope.LEGACY_UNTYPED:
            if self.evidence_plane is not None or self.evidence_source is not None:
                raise ValueError("legacy revocation entries cannot claim evidence provenance")
        else:
            if self.evidence_plane is None or self.evidence_source is None:
                raise ValueError("typed revocation entries require evidence provenance")
            plane, source = validate_plane_source(self.evidence_plane, self.evidence_source)
            object.__setattr__(self, "evidence_plane", plane)
            object.__setattr__(self, "evidence_source", source)
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
            "evidence_plane": (
                self.evidence_plane.value if self.evidence_plane is not None else None
            ),
            "evidence_source": (
                self.evidence_source.value if self.evidence_source is not None else None
            ),
            "revoked_at": self.revoked_at,
            "reason_code": self.reason_code,
            "scope": self.scope.value,
        }

    @classmethod
    def create(
        cls,
        *,
        artifact_id: str,
        artifact_schema_version: int,
        artifact_digest: str,
        evidence_plane: EvidencePlane | str | None,
        evidence_source: EvidenceSourceKind | str | None,
        revoked_at: str,
        reason_code: str,
        scope: RevocationScope | str = RevocationScope.TYPED,
    ) -> "RevocationEntry":
        resolved_scope = RevocationScope(scope)
        if resolved_scope is RevocationScope.LEGACY_UNTYPED:
            if evidence_plane is not None or evidence_source is not None:
                raise ValueError("legacy revocation entries cannot claim evidence provenance")
            resolved_plane = None
            resolved_source = None
        else:
            if evidence_plane is None or evidence_source is None:
                raise ValueError("typed revocation entries require evidence provenance")
            resolved_plane = EvidencePlane(evidence_plane)
            resolved_source = EvidenceSourceKind(evidence_source)
        values = {
            "artifact_id": artifact_id,
            "artifact_schema_version": artifact_schema_version,
            "artifact_digest": artifact_digest,
            "evidence_plane": resolved_plane,
            "evidence_source": resolved_source,
            "revoked_at": revoked_at,
            "reason_code": reason_code,
            "scope": resolved_scope,
            "schema_version": REVOCATION_SCHEMA_VERSION,
        }
        return cls(revocation_id=f"revocation.{_digest({**values})[:40]}", **values)

    def payload(self) -> dict[str, object]:
        return {"schema": REVOCATION_SCHEMA, "revocation_id": self.revocation_id, **self._identity_payload()}

    @classmethod
    def from_payload(cls, value: object) -> "RevocationEntry":
        fields = {
            "schema", "revocation_id", "schema_version", "artifact_id",
            "artifact_schema_version", "artifact_digest", "evidence_plane",
            "evidence_source", "revoked_at", "reason_code", "scope",
        }
        if not isinstance(value, Mapping) or set(value) != fields or value.get("schema") != REVOCATION_SCHEMA:
            raise ValueError("malformed revocation entry")
        try:
            return cls(
                revocation_id=value["revocation_id"], artifact_id=value["artifact_id"],
                artifact_schema_version=value["artifact_schema_version"], artifact_digest=value["artifact_digest"],
                evidence_plane=value["evidence_plane"], evidence_source=value["evidence_source"],
                revoked_at=value["revoked_at"], reason_code=value["reason_code"],
                scope=value["scope"],
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
        if self.lock_path.is_symlink():
            raise ValueError("revocation registry lock cannot be a symlink")
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
                raise ValueError(
                    f"malformed revocation registry at line {line_number}"
                )
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
            # Artifact IDs are content-addressed identities. Two entries for
            # one artifact/schema must agree on its digest. Typed entries must
            # also agree on plane/source; a legacy-untyped entry deliberately
            # applies across all typed lookup surfaces without inventing
            # provenance for the old artifact.
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
                        or (
                            existing.scope is RevocationScope.TYPED
                            and entry.scope is RevocationScope.TYPED
                            and (
                                existing.evidence_plane != entry.evidence_plane
                                or existing.evidence_source != entry.evidence_source
                            )
                        )
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
                and entry.artifact_digest != artifact_digest
            ):
                raise ValueError("conflicting revocation identity")
            if (
                entry.artifact_id == artifact_id
                and entry.artifact_schema_version == artifact_schema_version
                and entry.artifact_digest == artifact_digest
                and (
                    entry.scope is RevocationScope.LEGACY_UNTYPED
                    or (
                        entry.evidence_plane == plane
                        and entry.evidence_source == source
                    )
                )
            ):
                raise ValueError("artifact is revoked")
            if (
                entry.artifact_id == artifact_id
                and entry.artifact_schema_version == artifact_schema_version
                and entry.artifact_digest == artifact_digest
                and entry.scope is RevocationScope.TYPED
                and (
                    entry.evidence_plane != plane
                    or entry.evidence_source != source
                )
            ):
                raise ValueError("conflicting revocation identity")


__all__ = [
    "REVOCATION_SCHEMA",
    "REVOCATION_SCHEMA_VERSION",
    "JsonRevocationRegistry",
    "RevocationScope",
    "RevocationEntry",
]
