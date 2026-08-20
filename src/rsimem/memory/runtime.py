"""Backend routing and lifecycle evidence for typed agent memory."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .contracts import (
    MemoryBackend,
    MemoryEvent,
    MemoryEventKind,
    MemoryHit,
    MemoryKind,
    MemoryMutation,
    MemoryMutationAction,
    MemoryMutationResult,
    MemoryObserver,
    MemoryQuery,
)


class MemoryBackendRegistry:
    """Route each memory kind to one explicitly selected backend."""

    def __init__(self) -> None:
        self._backends: dict[str, MemoryBackend] = {}
        self._routes: dict[MemoryKind, str] = {}

    def register(self, backend: MemoryBackend, *, select: bool = True) -> None:
        name = backend.descriptor.name
        if name in self._backends:
            raise ValueError(f"memory backend already registered: {name}")
        if select:
            for capability in backend.descriptor.capabilities:
                if capability.kind in self._routes:
                    selected = self._routes[capability.kind]
                    raise ValueError(
                        f"memory kind {capability.kind.value} already routed to {selected}"
                    )
        self._backends[name] = backend
        if select:
            for capability in backend.descriptor.capabilities:
                self._routes[capability.kind] = name

    def select(self, kind: MemoryKind, backend_name: str) -> None:
        backend = self._backends.get(backend_name)
        if backend is None:
            raise KeyError(f"unknown memory backend: {backend_name}")
        if backend.descriptor.capability_for(kind) is None:
            raise ValueError(f"backend {backend_name} does not support {kind.value}")
        self._routes[kind] = backend_name

    def resolve(self, kind: MemoryKind) -> MemoryBackend:
        backend_name = self._routes.get(kind)
        if backend_name is None:
            raise KeyError(f"no backend selected for memory kind: {kind.value}")
        return self._backends[backend_name]

    def close(self) -> None:
        for backend in self._backends.values():
            backend.close()


class MemoryRuntime:
    """Apply memory operations while emitting privacy-safe lifecycle evidence."""

    def __init__(
        self,
        registry: MemoryBackendRegistry,
        *,
        observers: Iterable[MemoryObserver] = (),
    ) -> None:
        self.registry = registry
        self.observers = tuple(observers)

    def _record(self, event: MemoryEvent) -> None:
        for observer in self.observers:
            observer.record(event)

    def query(self, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        backend = self.registry.resolve(query.kind)
        capability = backend.descriptor.capability_for(query.kind)
        if capability is None or not capability.readable:
            raise PermissionError(f"backend {backend.descriptor.name} cannot read {query.kind.value}")
        self._record(MemoryEvent(
            kind=MemoryEventKind.QUERY,
            memory_kind=query.kind,
            backend=backend.descriptor.name,
            query_chars=len(query.text),
            attributes={"limit": query.limit, "namespace": query.namespace},
        ))
        hits = tuple(backend.query(query))[:query.limit]
        for hit in hits:
            if hit.artifact.kind != query.kind:
                raise ValueError(
                    f"backend {backend.descriptor.name} returned {hit.artifact.kind.value} "
                    f"for a {query.kind.value} query"
                )
            if hit.backend != backend.descriptor.name:
                raise ValueError(
                    f"backend {backend.descriptor.name} returned a hit owned by {hit.backend}"
                )
        self._record(MemoryEvent(
            kind=MemoryEventKind.RETRIEVED,
            memory_kind=query.kind,
            backend=backend.descriptor.name,
            artifact_ids=tuple(hit.artifact.artifact_id for hit in hits),
            content_chars=sum(len(hit.artifact.content) for hit in hits),
            attributes={"count": len(hits)},
        ))
        return hits

    def mutate(self, mutation: MemoryMutation) -> MemoryMutationResult:
        backend = self.registry.resolve(mutation.kind)
        capability = backend.descriptor.capability_for(mutation.kind)
        if capability is None:
            raise ValueError(f"backend {backend.descriptor.name} does not support {mutation.kind.value}")
        supported = {
            MemoryMutationAction.ADD: capability.writable,
            MemoryMutationAction.UPDATE: capability.updatable,
            MemoryMutationAction.DELETE: capability.deletable,
        }[mutation.action]
        artifact_id = mutation.resolved_artifact_id
        self._record(MemoryEvent(
            kind=MemoryEventKind.MUTATION_REQUESTED,
            memory_kind=mutation.kind,
            backend=backend.descriptor.name,
            artifact_ids=(artifact_id,) if artifact_id else (),
            content_chars=len(mutation.artifact.content) if mutation.artifact else None,
            attributes={"action": mutation.action.value},
        ))
        if not supported:
            result = MemoryMutationResult(
                accepted=False,
                backend=backend.descriptor.name,
                action=mutation.action,
                artifact_id=artifact_id,
                reason_code="operation_not_supported",
            )
        else:
            result = backend.mutate(mutation)
            if result.backend != backend.descriptor.name:
                raise ValueError(
                    f"backend {backend.descriptor.name} returned a result owned by "
                    f"{result.backend}"
                )
            if result.action != mutation.action:
                raise ValueError(
                    f"backend {backend.descriptor.name} returned the wrong mutation action"
                )
        self._record(MemoryEvent(
            kind=(
                MemoryEventKind.MUTATION_COMMITTED
                if result.accepted
                else MemoryEventKind.MUTATION_REJECTED
            ),
            memory_kind=mutation.kind,
            backend=backend.descriptor.name,
            artifact_ids=(result.artifact_id,) if result.artifact_id else (),
            reason_code=result.reason_code,
            attributes={"action": mutation.action.value},
        ))
        return result

    def mark_injected(
        self,
        hits: Sequence[MemoryHit],
        *,
        surface: str,
    ) -> None:
        if not surface.strip():
            raise ValueError("injection surface must not be empty")
        grouped: dict[tuple[MemoryKind, str], list[MemoryHit]] = {}
        for hit in hits:
            grouped.setdefault((hit.artifact.kind, hit.backend), []).append(hit)
        for (kind, backend), group in grouped.items():
            self._record(MemoryEvent(
                kind=MemoryEventKind.INJECTED,
                memory_kind=kind,
                backend=backend,
                artifact_ids=tuple(hit.artifact.artifact_id for hit in group),
                content_chars=sum(len(hit.artifact.content) for hit in group),
                attributes={"count": len(group), "surface": surface},
            ))

    def close(self) -> None:
        self.registry.close()
