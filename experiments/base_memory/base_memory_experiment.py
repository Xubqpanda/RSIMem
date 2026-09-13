"""Base-memory experiment contracts.

This contract deliberately does not reuse PAST's ``without_persistence``
variant.  That variant changes the whole persistence/tool surface, whereas
the historical Stage 0 NoMemory control keeps the same episode protocol and
only removes the semantic-memory substrate. ``AllMemoryOff`` is a separate,
explicit condition for the next formal baseline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum


SCHEMA = "rsimem-base-memory-comparison-manifest-v1"
PROTOCOL_ID = "base-memory-comparison-v1"
FROZEN_MODEL_ID = "gpt-5.6-luna"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError(f"{name} must be a nonempty stable identifier")
    return value


class BaseMemoryCondition(StrEnum):
    NO_MEMORY = "NoMemory"
    ALL_MEMORY_OFF = "AllMemoryOff"
    HERMES_NATIVE = "HermesNative"
    MEM0_STATIC = "Mem0Static"


HISTORICAL_BASE_MEMORY_CONDITIONS = (
    BaseMemoryCondition.NO_MEMORY,
    BaseMemoryCondition.HERMES_NATIVE,
    BaseMemoryCondition.MEM0_STATIC,
)


@dataclass(frozen=True, slots=True)
class BaseMemoryRunSpec:
    run_id: str
    condition: BaseMemoryCondition
    state_directory: str
    trace_directory: str
    artifact_directory: str
    hermes_home_directory: str
    port_offset: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition", BaseMemoryCondition(self.condition))
        _identifier(self.run_id, "base-memory run ID")
        for value in (
            self.state_directory, self.trace_directory, self.artifact_directory,
            self.hermes_home_directory,
        ):
            _identifier(value, "base-memory isolated identity")
        if type(self.port_offset) is not int or self.port_offset < 0:
            raise ValueError("base-memory port offset must be nonnegative")

    def payload(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "condition": self.condition.value,
            "state_directory": self.state_directory,
            "trace_directory": self.trace_directory,
            "artifact_directory": self.artifact_directory,
            "hermes_home_directory": self.hermes_home_directory,
            "port_offset": self.port_offset,
        }


@dataclass(frozen=True, slots=True)
class BaseMemoryComparisonManifest:
    manifest_id: str
    family_id: str
    source_sequence_digest: str
    fixture_digest: str
    base_model: str
    temperature: float
    token_budget: int
    config_digest: str
    registry_digest: str
    runs: tuple[BaseMemoryRunSpec, ...]
    schema: str = SCHEMA
    protocol_id: str = PROTOCOL_ID

    def __post_init__(self) -> None:
        if self.schema != SCHEMA or self.protocol_id != PROTOCOL_ID:
            raise ValueError("unsupported base-memory comparison protocol")
        for value, name in (
            (self.manifest_id, "manifest ID"), (self.family_id, "family ID"),
            (self.source_sequence_digest, "source sequence digest"),
            (self.fixture_digest, "fixture digest"), (self.config_digest, "config digest"),
            (self.registry_digest, "registry digest"),
        ):
            _identifier(value, name)
        if self.base_model != FROZEN_MODEL_ID:
            raise ValueError("base-memory comparison requires frozen model " + FROZEN_MODEL_ID)
        if type(self.temperature) is not float or self.temperature < 0:
            raise ValueError("base-memory temperature is invalid")
        if type(self.token_budget) is not int or self.token_budget < 1:
            raise ValueError("base-memory token budget is invalid")
        for value, name in (
            (self.source_sequence_digest, "source sequence"), (self.fixture_digest, "fixture"),
            (self.config_digest, "config"), (self.registry_digest, "registry"),
        ):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"base-memory {name} digest must be sha256")
        runs = tuple(self.runs)
        if not runs:
            raise ValueError("base-memory manifest requires at least one condition")
        if len({run.condition for run in runs}) != len(runs):
            raise ValueError("base-memory manifest conditions must be unique")
        if len({run.run_id for run in runs}) != len(runs):
            raise ValueError("base-memory run IDs must be unique")
        for field in ("state_directory", "trace_directory", "artifact_directory", "hermes_home_directory"):
            if len({getattr(run, field) for run in runs}) != len(runs):
                raise ValueError(f"base-memory {field} must be isolated")
        if len({run.port_offset for run in runs}) != len(runs):
            raise ValueError("base-memory port offsets must be isolated")

    @classmethod
    def create(
        cls, *, family_id: str, source_sequence_digest: str, fixture_digest: str,
        config_digest: str, registry_digest: str, token_budget: int,
        run_prefix: str = "base-memory", temperature: float = 0.0,
        port_base: int = 12000,
        conditions: tuple[BaseMemoryCondition, ...] = HISTORICAL_BASE_MEMORY_CONDITIONS,
    ) -> "BaseMemoryComparisonManifest":
        _identifier(run_prefix, "base-memory run prefix")
        if type(port_base) is not int or port_base < 0:
            raise ValueError("base-memory port base must be nonnegative")
        selected_conditions = tuple(BaseMemoryCondition(condition) for condition in conditions)
        if not selected_conditions or len(selected_conditions) != len(set(selected_conditions)):
            raise ValueError("base-memory conditions must be nonempty and unique")
        runs = tuple(
            BaseMemoryRunSpec(
                run_id=f"{run_prefix}-{condition.value.lower()}", condition=condition,
                state_directory=f"state-{condition.value.lower()}",
                trace_directory=f"trace-{condition.value.lower()}",
                artifact_directory=f"artifacts-{condition.value.lower()}",
                hermes_home_directory=f"home-{condition.value.lower()}",
                port_offset=port_base + index * 100,
            )
            for index, condition in enumerate(selected_conditions)
        )
        identity = {
            "family_id": family_id, "source_sequence_digest": source_sequence_digest,
            "fixture_digest": fixture_digest, "base_model": FROZEN_MODEL_ID,
            "temperature": temperature, "token_budget": token_budget,
            "config_digest": config_digest, "registry_digest": registry_digest,
            "runs": [run.payload() for run in runs],
        }
        return cls(
            manifest_id="base-memory." + _digest(identity)[:24], family_id=family_id,
            source_sequence_digest=source_sequence_digest, fixture_digest=fixture_digest,
            base_model=FROZEN_MODEL_ID, temperature=temperature, token_budget=token_budget,
            config_digest=config_digest, registry_digest=registry_digest, runs=runs,
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema, "protocol_id": self.protocol_id,
            "manifest_id": self.manifest_id, "family_id": self.family_id,
            "source_sequence_digest": self.source_sequence_digest,
            "fixture_digest": self.fixture_digest, "base_model": self.base_model,
            "temperature": self.temperature, "token_budget": self.token_budget,
            "config_digest": self.config_digest, "registry_digest": self.registry_digest,
            "runs": [run.payload() for run in self.runs],
        }


def compare_run_manifests(left: dict[str, object], right: dict[str, object]) -> dict[str, tuple[object, object]]:
    """Permit backend identity and isolated runtime paths, nothing else."""

    if set(left) != set(right):
        raise ValueError("base-memory run manifest shape differs")
    # ``manifest_id``/``runs`` identify the complete comparison assembly and
    # naturally differ when independently launched conditions use fresh run
    # roots. ``sequence_digest`` differs only because the selected backend is
    # rendered into the otherwise source-bound sequence. The source digest and
    # all protocol/model/config identities remain mandatory equalities.
    allowed = {
        "manifest_id", "runs", "run_id", "condition", "backend", "sequence_digest",
        "port_offset", "state_directory", "trace_directory", "artifact_directory",
        "hermes_home_directory",
    }
    differences = {key: (left[key], right[key]) for key in left if left[key] != right[key]}
    unexpected = set(differences) - allowed
    if unexpected:
        raise ValueError("base-memory identity drift: " + ", ".join(sorted(unexpected)))
    return differences


__all__ = [
    "BaseMemoryComparisonManifest", "BaseMemoryCondition", "BaseMemoryRunSpec",
    "FROZEN_MODEL_ID", "PROTOCOL_ID", "SCHEMA", "compare_run_manifests",
    "HISTORICAL_BASE_MEMORY_CONDITIONS",
]
