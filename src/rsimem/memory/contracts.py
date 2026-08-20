"""Host-neutral contracts for typed agent memory backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, TypeVar, runtime_checkable


class MemoryKind(StrEnum):
    """Standard memory taxonomy used by RSIMem."""

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


class MemoryAccessMode(StrEnum):
    """How a host normally exposes a memory kind to an agent."""

    EAGER = "eager"
    SEARCH = "search"
    PROGRESSIVE = "progressive"


class MemoryMutationAction(StrEnum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"


class MemoryEventKind(StrEnum):
    QUERY = "query"
    RETRIEVED = "retrieved"
    MUTATION_REQUESTED = "mutation_requested"
    MUTATION_COMMITTED = "mutation_committed"
    MUTATION_REJECTED = "mutation_rejected"
    INJECTED = "injected"


def _frozen_metadata(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


_EnumT = TypeVar("_EnumT", bound=StrEnum)


def _coerce_enum(enum_type: type[_EnumT], value: _EnumT | str) -> _EnumT:
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"invalid {enum_type.__name__}: {value}") from exc


@dataclass(frozen=True, slots=True)
class MemoryResource:
    """A file bundled with a procedural memory artifact."""

    path: str
    content: bytes
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        path = PurePosixPath(self.path)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("resource path must stay inside the memory artifact")
        if str(path) in {".", "SKILL.md"}:
            raise ValueError("resource path must identify a supporting file")
        object.__setattr__(self, "path", str(path))


@dataclass(frozen=True, slots=True)
class MemoryArtifact:
    """Canonical memory value independent of a storage framework."""

    artifact_id: str
    kind: MemoryKind
    content: str
    namespace: str = "default"
    title: str | None = None
    revision: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    resources: tuple[MemoryResource, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _coerce_enum(MemoryKind, self.kind))
        if not self.artifact_id.strip():
            raise ValueError("artifact_id must not be empty")
        if not self.namespace.strip():
            raise ValueError("namespace must not be empty")
        if not self.content.strip():
            raise ValueError("content must not be empty")
        resource_paths = [resource.path for resource in self.resources]
        if len(resource_paths) != len(set(resource_paths)):
            raise ValueError("resource paths must be unique")
        if self.resources and self.kind != MemoryKind.PROCEDURAL:
            raise ValueError("supporting resources require procedural memory")
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    """A typed retrieval request routed to exactly one memory kind."""

    kind: MemoryKind
    text: str
    namespace: str = "default"
    limit: int = 5
    filters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _coerce_enum(MemoryKind, self.kind))
        if not self.namespace.strip():
            raise ValueError("namespace must not be empty")
        if self.limit < 1:
            raise ValueError("limit must be positive")
        object.__setattr__(self, "filters", _frozen_metadata(self.filters))


@dataclass(frozen=True, slots=True)
class MemoryHit:
    artifact: MemoryArtifact
    rank: int
    score: float | None = None
    backend: str = ""

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be positive")
        if not self.backend.strip():
            raise ValueError("hit backend must not be empty")


@dataclass(frozen=True, slots=True)
class MemoryMutation:
    """Add, update, or delete one canonical memory artifact."""

    action: MemoryMutationAction
    kind: MemoryKind
    artifact: MemoryArtifact | None = None
    artifact_id: str | None = None
    expected_revision: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action",
            _coerce_enum(MemoryMutationAction, self.action),
        )
        object.__setattr__(self, "kind", _coerce_enum(MemoryKind, self.kind))
        if self.action in {MemoryMutationAction.ADD, MemoryMutationAction.UPDATE}:
            if self.artifact is None:
                raise ValueError(f"{self.action.value} requires artifact")
            if self.artifact.kind != self.kind:
                raise ValueError("artifact kind must match mutation kind")
        if self.action == MemoryMutationAction.ADD and self.artifact_id is not None:
            raise ValueError("add does not accept a target artifact_id")
        if self.action == MemoryMutationAction.DELETE:
            if self.artifact is not None:
                raise ValueError("delete accepts artifact_id instead of artifact")
            if not self.resolved_artifact_id:
                raise ValueError("delete requires artifact_id")
        if self.action == MemoryMutationAction.UPDATE and not self.resolved_artifact_id:
            raise ValueError("update requires artifact_id")

    @property
    def resolved_artifact_id(self) -> str | None:
        return self.artifact_id or (self.artifact.artifact_id if self.artifact else None)


@dataclass(frozen=True, slots=True)
class MemoryMutationResult:
    accepted: bool
    backend: str
    action: MemoryMutationAction
    artifact_id: str | None = None
    revision: str | None = None
    reason_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action",
            _coerce_enum(MemoryMutationAction, self.action),
        )
        if not self.backend.strip():
            raise ValueError("result backend must not be empty")


@dataclass(frozen=True, slots=True)
class MemoryKindCapability:
    kind: MemoryKind
    access_mode: MemoryAccessMode
    readable: bool = True
    writable: bool = True
    updatable: bool = True
    deletable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _coerce_enum(MemoryKind, self.kind))
        object.__setattr__(
            self,
            "access_mode",
            _coerce_enum(MemoryAccessMode, self.access_mode),
        )


@dataclass(frozen=True, slots=True)
class MemoryBackendDescriptor:
    name: str
    capabilities: tuple[MemoryKindCapability, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("backend name must not be empty")
        kinds = [capability.kind for capability in self.capabilities]
        if not kinds:
            raise ValueError("backend must declare at least one memory kind")
        if len(kinds) != len(set(kinds)):
            raise ValueError("backend capability kinds must be unique")

    def capability_for(self, kind: MemoryKind) -> MemoryKindCapability | None:
        return next((item for item in self.capabilities if item.kind == kind), None)


@dataclass(frozen=True, slots=True)
class MemoryEvent:
    """Content-free lifecycle evidence emitted by the memory runtime."""

    kind: MemoryEventKind
    memory_kind: MemoryKind
    backend: str
    artifact_ids: tuple[str, ...] = ()
    query_chars: int | None = None
    content_chars: int | None = None
    reason_code: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _coerce_enum(MemoryEventKind, self.kind))
        object.__setattr__(
            self,
            "memory_kind",
            _coerce_enum(MemoryKind, self.memory_kind),
        )
        if not self.backend.strip():
            raise ValueError("event backend must not be empty")
        object.__setattr__(self, "attributes", _frozen_metadata(self.attributes))


@dataclass(frozen=True, slots=True)
class MemoryMessage:
    """A host-neutral message supplied to a memory compiler."""

    role: str
    content: str
    name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise ValueError("message role must not be empty")
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class MemoryExperience:
    """Completed host episode that may be compiled into durable memory."""

    experience_id: str
    session_id: str
    messages: tuple[MemoryMessage, ...]
    task_id: str | None = None
    outcome: str | None = None
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.experience_id.strip() or not self.session_id.strip():
            raise ValueError("experience_id and session_id must not be empty")
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))


@runtime_checkable
class MemoryBackend(Protocol):
    @property
    def descriptor(self) -> MemoryBackendDescriptor: ...

    def get(self, artifact_id: str) -> MemoryArtifact | None: ...

    def query(self, query: MemoryQuery) -> Sequence[MemoryHit]: ...

    def mutate(self, mutation: MemoryMutation) -> MemoryMutationResult: ...

    def close(self) -> None: ...


@runtime_checkable
class MemoryObserver(Protocol):
    def record(self, event: MemoryEvent) -> None: ...


@runtime_checkable
class MemoryCompiler(Protocol):
    """Turn a completed experience into typed memory mutations."""

    @property
    def name(self) -> str: ...

    @property
    def output_kinds(self) -> frozenset[MemoryKind]: ...

    def compile(self, experience: MemoryExperience) -> Sequence[MemoryMutation]: ...
