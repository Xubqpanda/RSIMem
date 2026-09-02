"""Versioned lifecycle-surface and ownership contracts.

These contracts provide one host-neutral vocabulary for semantic, episodic,
and procedural memory.  A method receives credit only for surfaces it owns;
events from another owner or from an observed-only surface remain diagnostic.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Sequence

from .contracts import MemoryKind
from .evidence_planes import EvidencePlane, EvidenceSourceKind, validate_plane_source
from .policy_contracts import PolicyLayer


LIFECYCLE_SURFACE_SCHEMA_VERSION = 1
LIFECYCLE_SURFACE_SCHEMA = "rsimem-memory-lifecycle-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,255}$")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _id(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")
    return value


def _ids(values: Sequence[str], name: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    result = tuple(values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    for value in result:
        _id(value, name)
    return result


class MemoryLifecycleSurface(StrEnum):
    TRIGGER = "trigger"
    SOURCE_SELECTION = "source_selection"
    CONSTRUCTION = "construction"
    ADMISSION_MAINTENANCE = "admission_maintenance"
    COMMIT_VERSIONING = "commit_versioning"
    RETRIEVAL_EXPOSURE = "retrieval_exposure"


def surface_for_policy_layer(layer: PolicyLayer | str) -> MemoryLifecycleSurface:
    """Map legacy six-layer policy names to the new memory vocabulary.

    The old ``extraction`` operation describes construction only.  It must
    not be treated as evidence that a method owns trigger, admission, commit,
    or exposure behavior.
    """

    return {
        PolicyLayer.TRIGGER: MemoryLifecycleSurface.TRIGGER,
        PolicyLayer.SOURCE_SELECTION: MemoryLifecycleSurface.SOURCE_SELECTION,
        PolicyLayer.EXTRACTION: MemoryLifecycleSurface.CONSTRUCTION,
        PolicyLayer.ADMISSION: MemoryLifecycleSurface.ADMISSION_MAINTENANCE,
        PolicyLayer.COMMIT: MemoryLifecycleSurface.COMMIT_VERSIONING,
        PolicyLayer.EXPOSURE: MemoryLifecycleSurface.RETRIEVAL_EXPOSURE,
    }[PolicyLayer(layer)]


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    """Content-free evidence for one lifecycle surface transition."""

    event_id: str
    event_type: str
    producer: str
    owner_method: str
    memory_kind: MemoryKind
    surface: MemoryLifecycleSurface
    input_ids: tuple[str, ...]
    output_ids: tuple[str, ...]
    revision: str
    observation_cutoff: str
    evidence_plane: EvidencePlane
    evidence_source: EvidenceSourceKind
    parent_event_ids: tuple[str, ...] = ()
    schema: str = LIFECYCLE_SURFACE_SCHEMA
    schema_version: int = LIFECYCLE_SURFACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != LIFECYCLE_SURFACE_SCHEMA or self.schema_version != LIFECYCLE_SURFACE_SCHEMA_VERSION:
            raise ValueError("unsupported lifecycle event schema")
        _id(self.event_id, "lifecycle event ID")
        _id(self.event_type, "lifecycle event type")
        _id(self.producer, "lifecycle event producer")
        _id(self.owner_method, "lifecycle event owner method")
        object.__setattr__(self, "memory_kind", MemoryKind(self.memory_kind))
        object.__setattr__(self, "surface", MemoryLifecycleSurface(self.surface))
        object.__setattr__(self, "input_ids", _ids(self.input_ids, "lifecycle input IDs"))
        object.__setattr__(self, "output_ids", _ids(self.output_ids, "lifecycle output IDs"))
        _id(self.revision, "lifecycle revision")
        _id(self.observation_cutoff, "observation cutoff")
        plane, source = validate_plane_source(self.evidence_plane, self.evidence_source)
        object.__setattr__(self, "evidence_plane", plane)
        object.__setattr__(self, "evidence_source", source)
        object.__setattr__(self, "parent_event_ids", _ids(self.parent_event_ids, "parent event IDs"))

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "event_type": self.event_type,
            "producer": self.producer,
            "owner_method": self.owner_method,
            "memory_kind": self.memory_kind.value,
            "surface": self.surface.value,
            "input_ids": list(self.input_ids),
            "output_ids": list(self.output_ids),
            "revision": self.revision,
            "observation_cutoff": self.observation_cutoff,
            "evidence_plane": self.evidence_plane.value,
            "evidence_source": self.evidence_source.value,
            "parent_event_ids": list(self.parent_event_ids),
        }

    def payload(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            **self.identity_payload(),
        }

    @classmethod
    def create(
        cls,
        *,
        event_type: str,
        producer: str,
        owner_method: str,
        memory_kind: MemoryKind,
        surface: MemoryLifecycleSurface,
        input_ids: tuple[str, ...] = (),
        output_ids: tuple[str, ...] = (),
        revision: str,
        observation_cutoff: str,
        evidence_plane: EvidencePlane,
        evidence_source: EvidenceSourceKind,
        parent_event_ids: tuple[str, ...] = (),
    ) -> "LifecycleEvent":
        values = {
            "schema": LIFECYCLE_SURFACE_SCHEMA,
            "schema_version": LIFECYCLE_SURFACE_SCHEMA_VERSION,
            "event_type": event_type,
            "producer": producer,
            "owner_method": owner_method,
            "memory_kind": MemoryKind(memory_kind).value,
            "surface": MemoryLifecycleSurface(surface).value,
            "input_ids": list(input_ids),
            "output_ids": list(output_ids),
            "revision": revision,
            "observation_cutoff": observation_cutoff,
            "evidence_plane": EvidencePlane(evidence_plane).value,
            "evidence_source": EvidenceSourceKind(evidence_source).value,
            "parent_event_ids": list(parent_event_ids),
        }
        return cls(event_id=f"lifecycle-event.{_digest(values)[:40]}", **values)

    @classmethod
    def from_payload(cls, value: object) -> "LifecycleEvent":
        fields = {
            "event_id", "schema", "schema_version", "event_type", "producer",
            "owner_method", "memory_kind", "surface", "input_ids", "output_ids",
            "revision", "observation_cutoff", "evidence_plane", "evidence_source",
            "parent_event_ids",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("malformed lifecycle event")
        collections = ("input_ids", "output_ids", "parent_event_ids")
        if any(not isinstance(value[name], list) for name in collections):
            raise ValueError("lifecycle event collections must be lists")
        try:
            event = cls(
                event_id=value["event_id"],
                schema=value["schema"],
                schema_version=value["schema_version"],
                event_type=value["event_type"],
                producer=value["producer"],
                owner_method=value["owner_method"],
                memory_kind=value["memory_kind"],
                surface=value["surface"],
                input_ids=tuple(value["input_ids"]),
                output_ids=tuple(value["output_ids"]),
                revision=value["revision"],
                observation_cutoff=value["observation_cutoff"],
                evidence_plane=value["evidence_plane"],
                evidence_source=value["evidence_source"],
                parent_event_ids=tuple(value["parent_event_ids"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed lifecycle event") from exc
        expected = LifecycleEvent.create(
            event_type=event.event_type,
            producer=event.producer,
            owner_method=event.owner_method,
            memory_kind=event.memory_kind,
            surface=event.surface,
            input_ids=event.input_ids,
            output_ids=event.output_ids,
            revision=event.revision,
            observation_cutoff=event.observation_cutoff,
            evidence_plane=event.evidence_plane,
            evidence_source=event.evidence_source,
            parent_event_ids=event.parent_event_ids,
        )
        if event.event_id != expected.event_id or event.payload() != dict(value):
            raise ValueError("non-canonical lifecycle event")
        return event


@dataclass(frozen=True, slots=True)
class MethodLifecycleDescriptor:
    """Declared lifecycle ownership for one memory method."""

    method_id: str
    primary_kind: MemoryKind
    owned_surfaces: tuple[MemoryLifecycleSurface, ...]
    read_surfaces: tuple[MemoryLifecycleSurface, ...] = ()
    observe_surfaces: tuple[MemoryLifecycleSurface, ...] = ()
    schema: str = LIFECYCLE_SURFACE_SCHEMA
    schema_version: int = LIFECYCLE_SURFACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != LIFECYCLE_SURFACE_SCHEMA or self.schema_version != LIFECYCLE_SURFACE_SCHEMA_VERSION:
            raise ValueError("unsupported method lifecycle descriptor schema")
        _id(self.method_id, "method ID")
        object.__setattr__(self, "primary_kind", MemoryKind(self.primary_kind))
        for name in ("owned_surfaces", "read_surfaces", "observe_surfaces"):
            values = tuple(MemoryLifecycleSurface(value) for value in getattr(self, name))
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
            object.__setattr__(self, name, values)

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "method_id": self.method_id,
            "primary_kind": self.primary_kind.value,
            "owned_surfaces": [value.value for value in self.owned_surfaces],
            "read_surfaces": [value.value for value in self.read_surfaces],
            "observe_surfaces": [value.value for value in self.observe_surfaces],
        }


class LifecycleOwnership(StrEnum):
    OWNED = "owned"
    FOREIGN_OWNER = "foreign_owner"
    SURFACE_NOT_OWNED = "surface_not_owned"
    KIND_MISMATCH = "kind_mismatch"


@dataclass(frozen=True, slots=True)
class LifecycleAttribution:
    event_id: str
    method_id: str
    ownership: LifecycleOwnership

    @property
    def eligible_for_method_credit(self) -> bool:
        return self.ownership is LifecycleOwnership.OWNED


def classify_event_ownership(
    event: LifecycleEvent,
    method: MethodLifecycleDescriptor,
) -> LifecycleAttribution:
    if event.memory_kind != method.primary_kind:
        ownership = LifecycleOwnership.KIND_MISMATCH
    elif event.owner_method != method.method_id:
        ownership = LifecycleOwnership.FOREIGN_OWNER
    elif event.surface not in method.owned_surfaces:
        ownership = LifecycleOwnership.SURFACE_NOT_OWNED
    else:
        ownership = LifecycleOwnership.OWNED
    return LifecycleAttribution(event.event_id, method.method_id, ownership)


__all__ = [
    "LIFECYCLE_SURFACE_SCHEMA",
    "LIFECYCLE_SURFACE_SCHEMA_VERSION",
    "LifecycleAttribution",
    "LifecycleEvent",
    "LifecycleOwnership",
    "MemoryLifecycleSurface",
    "MethodLifecycleDescriptor",
    "classify_event_ownership",
    "surface_for_policy_layer",
]
