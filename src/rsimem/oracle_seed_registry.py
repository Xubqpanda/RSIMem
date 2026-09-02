"""Audit-only, verified storage for type-matched PAST oracle seed homes.

Seed contents never cross this contract.  It stores a case-bound relative
directory and a complete tree digest, while enforcing that an oracle changes
only the target memory-kind surface.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .memory.contracts import MemoryKind
from .sensitivity import SensitivityCase, SensitivityPanel


ORACLE_SEED_REGISTRY_SCHEMA = "rsimem-oracle-seed-registry-v1"
ORACLE_SEED_REGISTRY_SCHEMA_VERSION = 1
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
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value


def _relative(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{name} must be a relative path")
    path = Path(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must not escape the trusted seed root")
    return path.as_posix()


def oracle_seed_tree_digest(seed_home: Path) -> str:
    """Digest file names and bytes while rejecting symlinks and unknown types."""

    root = Path(seed_home)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("oracle seed home must be a regular directory")
    records: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("oracle seed home must not contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("oracle seed home contains an unsupported entry")
        records.append((path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()))
    if not records:
        raise ValueError("oracle seed home must not be empty")
    return _digest(records)


def _validate_seed_layout(seed_home: Path, kind: MemoryKind) -> None:
    files = {
        path.relative_to(seed_home).as_posix()
        for path in seed_home.rglob("*")
        if path.is_file()
    }
    top_level = {Path(path).parts[0] for path in files}
    if kind is MemoryKind.SEMANTIC:
        if files != {"memories/MEMORY.md"}:
            raise ValueError("semantic oracle seed must contain only memories/MEMORY.md")
    elif kind is MemoryKind.EPISODIC:
        required = {"state.db"}
        if not required <= files or not top_level <= {"state.db", "state.db-wal", "state.db-shm", "sessions"}:
            raise ValueError("episodic oracle seed must contain only native session state")
        if not any(path.startswith("sessions/") for path in files):
            raise ValueError("episodic oracle seed must include session artifacts")
    elif kind is MemoryKind.PROCEDURAL:
        if not files or not top_level == {"skills"} or not all(path.endswith("/SKILL.md") for path in files):
            raise ValueError("procedural oracle seed must contain only skill documents")
    else:  # defensive: MemoryKind is closed, but do not silently accept drift.
        raise ValueError("oracle seed kind is unsupported")


@dataclass(frozen=True, slots=True)
class OracleSeedRegistration:
    """One case-bound audited seed reference; no seed content is serialized."""

    registration_id: str
    case_id: str
    oracle_artifact_id: str
    panel: SensitivityPanel
    target_kind: MemoryKind
    family_source_digest: str
    seed_home: str
    seed_tree_digest: str
    authoring_basis: str
    schema: str = ORACLE_SEED_REGISTRY_SCHEMA
    schema_version: int = ORACLE_SEED_REGISTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != ORACLE_SEED_REGISTRY_SCHEMA or self.schema_version != ORACLE_SEED_REGISTRY_SCHEMA_VERSION:
            raise ValueError("unsupported oracle seed registration schema")
        for value, name in (
            (self.registration_id, "oracle seed registration ID"),
            (self.case_id, "oracle seed case ID"),
            (self.oracle_artifact_id, "oracle artifact ID"),
            (self.authoring_basis, "oracle authoring basis"),
        ):
            _id(value, name)
        object.__setattr__(self, "panel", SensitivityPanel(self.panel))
        object.__setattr__(self, "target_kind", MemoryKind(self.target_kind))
        if self.target_kind is not self.panel.memory_kind:
            raise ValueError("oracle seed target kind must match panel")
        object.__setattr__(self, "seed_home", _relative(self.seed_home, "oracle seed home"))
        _sha(self.family_source_digest, "oracle seed family source digest")
        _sha(self.seed_tree_digest, "oracle seed tree digest")
        expected = "oracle-seed-registration." + _digest(self.identity_payload())[:40]
        if self.registration_id != expected:
            raise ValueError("oracle seed registration ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "oracle_artifact_id": self.oracle_artifact_id,
            "panel": self.panel.value,
            "target_kind": self.target_kind.value,
            "family_source_digest": self.family_source_digest,
            "seed_home": self.seed_home,
            "seed_tree_digest": self.seed_tree_digest,
            "authoring_basis": self.authoring_basis,
        }

    def payload(self) -> dict[str, object]:
        return {"registration_id": self.registration_id, **self.identity_payload()}

    def resolve(self, trusted_root: Path, case: SensitivityCase, family_source_digest: str) -> Path:
        """Verify case/panel/provenance/tree binding before exposing local path."""

        if case.case_id != self.case_id or case.oracle_artifact_id != self.oracle_artifact_id:
            raise ValueError("oracle seed registration case/artifact mismatch")
        if case.panel is not self.panel or case.target_kind is not self.target_kind:
            raise ValueError("oracle seed registration panel/kind mismatch")
        if self.family_source_digest != family_source_digest:
            raise ValueError("oracle seed registration family source drift")
        root = Path(trusted_root).expanduser().resolve()
        if root.is_symlink() or not root.is_dir():
            raise ValueError("oracle seed trusted root is invalid")
        target = root / self.seed_home
        if not target.is_relative_to(root):
            raise ValueError("oracle seed home escapes trusted root")
        if oracle_seed_tree_digest(target) != self.seed_tree_digest:
            raise ValueError("oracle seed tree digest mismatch")
        _validate_seed_layout(target, self.target_kind)
        return target


@dataclass(frozen=True, slots=True)
class OracleSeedRegistry:
    """Canonical registry of all expected case-bound oracle seed homes."""

    registry_id: str
    registrations: tuple[OracleSeedRegistration, ...]
    schema: str = ORACLE_SEED_REGISTRY_SCHEMA
    schema_version: int = ORACLE_SEED_REGISTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != ORACLE_SEED_REGISTRY_SCHEMA or self.schema_version != ORACLE_SEED_REGISTRY_SCHEMA_VERSION:
            raise ValueError("unsupported oracle seed registry schema")
        registrations = tuple(self.registrations)
        if len({item.case_id for item in registrations}) != len(registrations):
            raise ValueError("oracle seed registry case IDs must be unique")
        object.__setattr__(self, "registrations", registrations)
        expected = "oracle-seed-registry." + _digest(self.identity_payload())[:40]
        if self.registry_id != expected:
            raise ValueError("oracle seed registry ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "registrations": [item.payload() for item in self.registrations],
        }

    def payload(self) -> dict[str, object]:
        return {"registry_id": self.registry_id, **self.identity_payload()}

    def for_case(self, case_id: str) -> OracleSeedRegistration:
        matches = [item for item in self.registrations if item.case_id == case_id]
        if len(matches) != 1:
            raise ValueError("oracle seed registration is missing")
        return matches[0]

    @classmethod
    def from_payload(cls, value: object) -> "OracleSeedRegistry":
        if not isinstance(value, Mapping) or set(value) != {"registry_id", "schema", "schema_version", "registrations"}:
            raise ValueError("malformed oracle seed registry")
        registrations = value.get("registrations")
        if not isinstance(registrations, list):
            raise ValueError("malformed oracle seed registry")
        try:
            result = cls(
                registry_id=value["registry_id"], schema=value["schema"], schema_version=value["schema_version"],
                registrations=tuple(OracleSeedRegistration(**item) for item in registrations),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed oracle seed registry") from exc
        if result.payload() != dict(value):
            raise ValueError("non-canonical oracle seed registry")
        return result

    @classmethod
    def load(cls, path: Path) -> "OracleSeedRegistry":
        try:
            return cls.from_payload(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("oracle seed registry is unreadable") from exc


def create_oracle_seed_registration(
    *,
    case: SensitivityCase,
    family_source_digest: str,
    seed_home: str,
    seed_tree_digest: str,
    authoring_basis: str = "benchmark_oracle_authoring.v1",
) -> OracleSeedRegistration:
    if case.oracle_artifact_id is None:
        raise ValueError("oracle seed registration requires an oracle sensitivity case")
    values = {
        "case_id": case.case_id,
        "oracle_artifact_id": case.oracle_artifact_id,
        "panel": case.panel,
        "target_kind": case.target_kind,
        "family_source_digest": family_source_digest,
        "seed_home": seed_home,
        "seed_tree_digest": seed_tree_digest,
        "authoring_basis": authoring_basis,
    }
    return OracleSeedRegistration(
        registration_id="oracle-seed-registration." + _digest({
            **values,
            "panel": case.panel.value,
            "target_kind": case.target_kind.value,
        })[:40],
        **values,
    )


def create_oracle_seed_registry(registrations: Sequence[OracleSeedRegistration]) -> OracleSeedRegistry:
    values = {
        "schema": ORACLE_SEED_REGISTRY_SCHEMA,
        "schema_version": ORACLE_SEED_REGISTRY_SCHEMA_VERSION,
        "registrations": [item.payload() for item in registrations],
    }
    return OracleSeedRegistry(
        registry_id="oracle-seed-registry." + _digest(values)[:40],
        registrations=tuple(registrations),
    )


__all__ = [
    "ORACLE_SEED_REGISTRY_SCHEMA",
    "ORACLE_SEED_REGISTRY_SCHEMA_VERSION",
    "OracleSeedRegistration",
    "OracleSeedRegistry",
    "create_oracle_seed_registration",
    "create_oracle_seed_registry",
    "oracle_seed_tree_digest",
]
