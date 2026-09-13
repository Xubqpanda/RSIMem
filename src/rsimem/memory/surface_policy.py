"""Canonical runtime availability policy for every persistence-facing surface."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping


RUNTIME_SURFACE_POLICY_SCHEMA = "rsimem-runtime-surface-policy-v1"
RUNTIME_SURFACE_POLICY_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class MemorySurface(StrEnum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    PROFILE = "profile"


class SurfaceOperation(StrEnum):
    FORM = "form"
    PERSIST = "persist"
    RETRIEVE = "retrieve"
    EXPOSE = "expose"
    USE = "use"
    OUTCOME = "outcome"


_SIGNAL_SURFACE = {
    "memory": MemorySurface.SEMANTIC,
    "session_search": MemorySurface.EPISODIC,
    "skill": MemorySurface.PROCEDURAL,
}


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class RuntimeSurfacePolicy:
    """The only authority for whether a persistence-facing surface is usable."""

    policy_id: str
    semantic_enabled: bool
    episodic_enabled: bool
    procedural_enabled: bool
    profile_enabled: bool
    schema_version: int = RUNTIME_SURFACE_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_SURFACE_POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported runtime surface policy schema")
        if not isinstance(self.policy_id, str) or _IDENTIFIER.fullmatch(self.policy_id) is None:
            raise ValueError("runtime surface policy ID is invalid")
        for name in ("semantic_enabled", "episodic_enabled", "procedural_enabled", "profile_enabled"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")

    @classmethod
    def all_memory_off(cls) -> "RuntimeSurfacePolicy":
        return cls("all-memory-off-v1", False, False, False, False)

    @classmethod
    def from_hermes_flags(
        cls,
        *,
        memory_enabled: bool,
        user_profile_enabled: bool,
        skills_enabled: bool,
        session_search_enabled: bool,
        all_memory_off: bool = False,
        policy_id: str | None = None,
    ) -> "RuntimeSurfacePolicy":
        values = {
            "semantic_enabled": memory_enabled,
            "episodic_enabled": session_search_enabled,
            "procedural_enabled": skills_enabled,
            "profile_enabled": user_profile_enabled,
        }
        if any(type(value) is not bool for value in values.values()):
            raise TypeError("runtime surface switches must be bool")
        if all_memory_off and any(values.values()):
            raise ValueError("AllMemoryOff cannot expose a Memory surface")
        selected_id = policy_id or (
            "all-memory-off-v1" if all_memory_off else "hermes-persistence-v1"
        )
        return cls(selected_id, **values)

    def for_expected_signal(self, expected_signal: str, *, persistence_enabled: bool = True) -> "RuntimeSurfacePolicy":
        """Constrain availability for one episode without changing its family label."""
        if not persistence_enabled or expected_signal in {"", "mixed"}:
            return self if persistence_enabled else self.all_memory_off()
        surface = _SIGNAL_SURFACE.get(expected_signal)
        if surface is None:
            raise ValueError(f"unknown expected persistence signal: {expected_signal}")
        return RuntimeSurfacePolicy(
            policy_id=f"{self.policy_id}-episode-{expected_signal}",
            semantic_enabled=self.semantic_enabled and surface is MemorySurface.SEMANTIC,
            episodic_enabled=self.episodic_enabled and surface is MemorySurface.EPISODIC,
            procedural_enabled=self.procedural_enabled and surface is MemorySurface.PROCEDURAL,
            profile_enabled=self.profile_enabled and surface is MemorySurface.SEMANTIC,
        )

    def enabled(self, surface: MemorySurface | str) -> bool:
        surface = MemorySurface(surface)
        return {
            MemorySurface.SEMANTIC: self.semantic_enabled,
            MemorySurface.EPISODIC: self.episodic_enabled,
            MemorySurface.PROCEDURAL: self.procedural_enabled,
            MemorySurface.PROFILE: self.profile_enabled,
        }[surface]

    def hermes_flags(self) -> dict[str, bool]:
        return {
            "memory_enabled": self.semantic_enabled,
            "user_profile_enabled": self.profile_enabled,
            "skills_enabled": self.procedural_enabled,
            "session_search_enabled": self.episodic_enabled,
        }

    def allows(
        self,
        surface: MemorySurface | str,
        operation: SurfaceOperation | str,
        *,
        expected_signal: str | None = None,
    ) -> bool:
        """Canonical capability check shared by host, benchmark, and audit code."""
        operation = SurfaceOperation(operation)
        if expected_signal not in (None, "", "mixed"):
            constrained = self.for_expected_signal(expected_signal)
            return constrained.enabled(surface)
        if not self.enabled(surface):
            return False
        if operation is SurfaceOperation.OUTCOME:
            return True
        return True

    def payload(self) -> dict[str, object]:
        return {
            "schema": RUNTIME_SURFACE_POLICY_SCHEMA,
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "semantic_memory_enabled": self.semantic_enabled,
            "episodic_memory_enabled": self.episodic_enabled,
            "procedural_memory_enabled": self.procedural_enabled,
            "profile_enabled": self.profile_enabled,
            "digest": _digest({
                "policy_id": self.policy_id,
                "semantic_memory_enabled": self.semantic_enabled,
                "episodic_memory_enabled": self.episodic_enabled,
                "procedural_memory_enabled": self.procedural_enabled,
                "profile_enabled": self.profile_enabled,
            }),
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, object]) -> "RuntimeSurfacePolicy":
        if value.get("schema") != RUNTIME_SURFACE_POLICY_SCHEMA:
            raise ValueError("unsupported runtime surface policy")
        result = cls(
            policy_id=str(value["policy_id"]),
            semantic_enabled=value["semantic_memory_enabled"],
            episodic_enabled=value["episodic_memory_enabled"],
            procedural_enabled=value["procedural_memory_enabled"],
            profile_enabled=value["profile_enabled"],
            schema_version=int(value["schema_version"]),
        )
        if result.payload() != dict(value):
            raise ValueError("runtime surface policy digest or payload is not canonical")
        return result


def capability_check(
    policy: RuntimeSurfacePolicy,
    surface: MemorySurface | str,
    operation: SurfaceOperation | str,
    *,
    expected_signal: str | None = None,
) -> bool:
    """Named function for callers that should not inspect policy fields directly."""
    return policy.allows(surface, operation, expected_signal=expected_signal)


__all__ = [
    "MemorySurface",
    "RuntimeSurfacePolicy",
    "SurfaceOperation",
    "RUNTIME_SURFACE_POLICY_SCHEMA",
    "RUNTIME_SURFACE_POLICY_SCHEMA_VERSION",
    "capability_check",
]
