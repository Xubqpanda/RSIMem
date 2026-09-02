"""Versioned memory taxonomy and memory/control identity contracts.

The taxonomy classifies persisted memory by the content of a memory unit.  It
does not classify a unit by the algorithm that produced it or by the task that
later consumes it.  Feedback, quality estimates, Q-values, and policy state
are represented separately so they cannot be mistaken for memory content.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Sequence

from .contracts import MemoryKind


MEMORY_TAXONOMY_SCHEMA_VERSION = 1
MEMORY_TAXONOMY_SCHEMA = "rsimem-memory-taxonomy-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,255}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")
    return value


def _nonempty_strings(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _digest_value(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a sha256 digest")
    return value


class MemoryTransform(StrEnum):
    """Declared transform when a unit combines two memory kinds."""

    CONSOLIDATION = "consolidation"
    DISTILLATION = "distillation"
    DERIVATION = "derivation"
    PROJECTION = "projection"


class MemoryControlKind(StrEnum):
    """State that controls a method but is not persisted memory content."""

    FEEDBACK = "feedback"
    Q_VALUE = "q_value"
    QUALITY = "quality"
    POLICY_STATE = "policy_state"


@dataclass(frozen=True, slots=True)
class MemoryUnitDescriptor:
    """Identity and applicability of one persisted memory unit type."""

    unit_id: str
    kind: MemoryKind
    content_schema: str
    scope: str
    source_provenance: tuple[str, ...]
    temporal_identity: str
    applicability: tuple[str, ...]
    version: str
    owner_method: str
    secondary_kind: MemoryKind | None = None
    transform: MemoryTransform | None = None
    schema: str = MEMORY_TAXONOMY_SCHEMA
    schema_version: int = MEMORY_TAXONOMY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != MEMORY_TAXONOMY_SCHEMA or self.schema_version != MEMORY_TAXONOMY_SCHEMA_VERSION:
            raise ValueError("unsupported memory unit descriptor schema")
        _identifier(self.unit_id, "memory unit ID")
        object.__setattr__(self, "kind", MemoryKind(self.kind))
        _identifier(self.content_schema, "memory content schema")
        _identifier(self.scope, "memory scope")
        object.__setattr__(
            self,
            "source_provenance",
            _nonempty_strings(self.source_provenance, "source provenance"),
        )
        _identifier(self.temporal_identity, "temporal identity")
        object.__setattr__(
            self,
            "applicability",
            _nonempty_strings(self.applicability, "applicability"),
        )
        _identifier(self.version, "memory unit version")
        _identifier(self.owner_method, "owner method")
        if self.secondary_kind is not None:
            object.__setattr__(self, "secondary_kind", MemoryKind(self.secondary_kind))
            if self.secondary_kind == self.kind:
                raise ValueError("secondary memory kind must differ from primary kind")
        if self.transform is not None:
            object.__setattr__(self, "transform", MemoryTransform(self.transform))
        if (self.secondary_kind is None) != (self.transform is None):
            raise ValueError("secondary kind and transform must be declared together")

    @property
    def primary_kind(self) -> MemoryKind:
        """Explicit name for the kind that owns this unit's experiment row."""

        return self.kind

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "unit_id": self.unit_id,
            "kind": self.kind.value,
            "content_schema": self.content_schema,
            "scope": self.scope,
            "source_provenance": list(self.source_provenance),
            "temporal_identity": self.temporal_identity,
            "applicability": list(self.applicability),
            "version": self.version,
            "owner_method": self.owner_method,
            "secondary_kind": self.secondary_kind.value if self.secondary_kind else None,
            "transform": self.transform.value if self.transform else None,
        }

    @property
    def descriptor_digest(self) -> str:
        return _digest(self.identity_payload())

    def payload(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "descriptor_digest": self.descriptor_digest,
        }

    @classmethod
    def from_payload(cls, value: object) -> "MemoryUnitDescriptor":
        fields = {
            "schema", "schema_version", "unit_id", "kind", "content_schema",
            "scope", "source_provenance", "temporal_identity", "applicability",
            "version", "owner_method", "secondary_kind", "transform",
            "descriptor_digest",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("malformed memory unit descriptor")
        if not isinstance(value["source_provenance"], list) or not isinstance(value["applicability"], list):
            raise ValueError("memory unit descriptor collections must be lists")
        try:
            descriptor = cls(
                schema=value["schema"],
                schema_version=value["schema_version"],
                unit_id=value["unit_id"],
                kind=value["kind"],
                content_schema=value["content_schema"],
                scope=value["scope"],
                source_provenance=tuple(value["source_provenance"]),
                temporal_identity=value["temporal_identity"],
                applicability=tuple(value["applicability"]),
                version=value["version"],
                owner_method=value["owner_method"],
                secondary_kind=value["secondary_kind"],
                transform=value["transform"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed memory unit descriptor") from exc
        if _digest_value(value["descriptor_digest"], "descriptor digest") != descriptor.descriptor_digest:
            raise ValueError("memory unit descriptor digest mismatch")
        if descriptor.payload() != dict(value):
            raise ValueError("non-canonical memory unit descriptor")
        return descriptor


@dataclass(frozen=True, slots=True)
class MemoryControlDescriptor:
    """Content-free identity for method feedback or policy state."""

    control_id: str
    control_kind: MemoryControlKind
    content_schema: str
    version: str
    owner_method: str
    source_provenance: tuple[str, ...]
    value_digest: str
    schema: str = MEMORY_TAXONOMY_SCHEMA
    schema_version: int = MEMORY_TAXONOMY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != MEMORY_TAXONOMY_SCHEMA or self.schema_version != MEMORY_TAXONOMY_SCHEMA_VERSION:
            raise ValueError("unsupported memory control descriptor schema")
        _identifier(self.control_id, "memory control ID")
        object.__setattr__(self, "control_kind", MemoryControlKind(self.control_kind))
        _identifier(self.content_schema, "control schema")
        _identifier(self.version, "control version")
        _identifier(self.owner_method, "control owner method")
        object.__setattr__(
            self,
            "source_provenance",
            _nonempty_strings(self.source_provenance, "control source provenance"),
        )
        _digest_value(self.value_digest, "control value digest")

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "control_id": self.control_id,
            "control_kind": self.control_kind.value,
            "content_schema": self.content_schema,
            "version": self.version,
            "owner_method": self.owner_method,
            "source_provenance": list(self.source_provenance),
            "value_digest": self.value_digest,
        }


__all__ = [
    "MEMORY_TAXONOMY_SCHEMA",
    "MEMORY_TAXONOMY_SCHEMA_VERSION",
    "MemoryControlDescriptor",
    "MemoryControlKind",
    "MemoryKind",
    "MemoryTransform",
    "MemoryUnitDescriptor",
]
