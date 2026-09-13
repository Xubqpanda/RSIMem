"""Capability registry for concrete memory systems."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rsimem.memory.contracts import MemoryKind
from rsimem.memory.runtime import MemoryBackendRegistry
from .episodic.hermes_native import HermesEpisodicBackend
from .procedural.hermes_native import HermesProceduralBackend
from .semantic.hermes_native import HermesSemanticBackend


@dataclass(frozen=True, slots=True)
class MemorySystemCapability:
    system_id: str
    memory_kinds: tuple[MemoryKind, ...]

    def supports(self, kind: MemoryKind | str) -> bool:
        return MemoryKind(kind) in self.memory_kinds


MEM0_FLAT_CAPABILITY = MemorySystemCapability(
    "mem0_flat", (MemoryKind.SEMANTIC,)
)
HERMES_NATIVE_CAPABILITY = MemorySystemCapability(
    "hermes_native", (
        MemoryKind.SEMANTIC, MemoryKind.EPISODIC, MemoryKind.PROCEDURAL,
    )
)


def capability_for(system_id: str) -> MemorySystemCapability:
    values = {
        MEM0_FLAT_CAPABILITY.system_id: MEM0_FLAT_CAPABILITY,
        HERMES_NATIVE_CAPABILITY.system_id: HERMES_NATIVE_CAPABILITY,
    }
    try:
        return values[system_id]
    except KeyError as exc:
        raise ValueError(f"unknown memory system: {system_id}") from exc


def build_hermes_native_registry(hermes_home: Path) -> MemoryBackendRegistry:
    """Build the native three-surface registry for one isolated Hermes home."""
    home = hermes_home.expanduser().resolve()
    registry = MemoryBackendRegistry()
    registry.register(HermesSemanticBackend(home / "memories"))
    registry.register(HermesEpisodicBackend(home / "state.db"))
    registry.register(HermesProceduralBackend(home / "skills"))
    return registry


__all__ = [
    "HERMES_NATIVE_CAPABILITY", "MEM0_FLAT_CAPABILITY", "MemorySystemCapability",
    "build_hermes_native_registry", "capability_for",
]
