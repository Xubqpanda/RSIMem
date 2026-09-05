"""Frozen protocol for native failure attribution and one-axis repair."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


PROTOCOL_SCHEMA = "rsimem-native-attribution-repair-protocol-v1"
PROTOCOL_SCHEMA_VERSION = 1
PROTOCOL_NAME = "native-attribution-repair-v1"
HISTORICAL_STATUS = "historical_exploratory_only"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class NativeAttributionRepairProtocol:
    protocol_id: str
    schema: str
    schema_version: int
    baseline_condition: str
    background_lower_bound: str
    execution_flow: tuple[str, ...]
    memory_kinds: tuple[str, ...]
    lifecycle_surfaces: tuple[str, ...]
    failure_labels: tuple[str, ...]
    repair_axes: tuple[str, ...]
    evidence_planes: tuple[str, ...]
    updater_allowed_planes: tuple[str, ...]
    audit_only_fields: tuple[str, ...]
    raw_resource_fields: tuple[str, ...]
    panels: Mapping[str, tuple[str, ...]]
    isolation_dimensions: tuple[str, ...]
    infrastructure_exclusions: tuple[str, ...]
    historical_experiments: Mapping[str, str]
    provider_id: str
    model_id: str
    sampling_temperature: float
    retry_policy: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "baseline_condition": self.baseline_condition,
            "background_lower_bound": self.background_lower_bound,
            "execution_flow": list(self.execution_flow),
            "memory_kinds": list(self.memory_kinds),
            "lifecycle_surfaces": list(self.lifecycle_surfaces),
            "failure_labels": list(self.failure_labels),
            "repair_axes": list(self.repair_axes),
            "evidence_planes": list(self.evidence_planes),
            "updater_allowed_planes": list(self.updater_allowed_planes),
            "audit_only_fields": list(self.audit_only_fields),
            "raw_resource_fields": list(self.raw_resource_fields),
            "panels": {key: list(value) for key, value in sorted(self.panels.items())},
            "isolation_dimensions": list(self.isolation_dimensions),
            "infrastructure_exclusions": list(self.infrastructure_exclusions),
            "historical_experiments": dict(sorted(self.historical_experiments.items())),
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "sampling_temperature": self.sampling_temperature,
            "retry_policy": self.retry_policy,
        }

    @property
    def protocol_digest(self) -> str:
        return _digest(self.identity_payload())

    def payload(self) -> dict[str, object]:
        return {"protocol_id": self.protocol_id, **self.identity_payload()}

    def __post_init__(self) -> None:
        if self.schema != PROTOCOL_SCHEMA or self.schema_version != PROTOCOL_SCHEMA_VERSION:
            raise ValueError("unsupported native attribution protocol schema")
        expected = f"{PROTOCOL_NAME}.{self.protocol_digest[:40]}"
        if self.protocol_id != expected:
            raise ValueError("native attribution protocol ID does not match its content")
        if self.baseline_condition != "native_static":
            raise ValueError("native_static must be the attribution baseline")
        if self.background_lower_bound != "no_persistence":
            raise ValueError("no_persistence may only be the background lower bound")
        forbidden = {"shortcut_current_input", "wrong_mechanism", "type_matched_oracle"}
        if forbidden.intersection(self.execution_flow):
            raise ValueError("historical sensitivity conditions cannot enter the execution flow")
        if tuple(self.memory_kinds) != ("semantic", "episodic", "procedural"):
            raise ValueError("protocol must cover the three frozen memory kinds")
        if self.updater_allowed_planes != ("pure_process",):
            raise ValueError("the updater may consume only pure_process evidence")
        if {"official_score", "grader", "hidden_answer"}.intersection(
            self.updater_allowed_planes
        ):
            raise ValueError("audit fields cannot enter the updater plane")
        if not all(value == HISTORICAL_STATUS for value in self.historical_experiments.values()):
            raise ValueError("legacy experiment groups must remain historical exploratory evidence")
        if set(self.repair_axes) != {
            "formation", "persistence", "maintenance", "retrieval", "application"
        }:
            raise ValueError("one-axis repair vocabulary is incomplete")
        if len({family for values in self.panels.values() for family in values}) != 26:
            raise ValueError("protocol must freeze exactly 26 distinct PAST families")

    @classmethod
    def create(cls) -> "NativeAttributionRepairProtocol":
        values: dict[str, object] = {
            "schema": PROTOCOL_SCHEMA,
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "baseline_condition": "native_static",
            "background_lower_bound": "no_persistence",
            "execution_flow": (
                "native_static_execution",
                "lifecycle_evidence",
                "failure_attribution",
                "one_axis_repair",
                "same_downstream_task",
                "paired_headroom",
                "feedback_to_update",
            ),
            "memory_kinds": ("semantic", "episodic", "procedural"),
            "lifecycle_surfaces": (
                "trigger_source", "formation", "persistence", "maintenance",
                "retrieval", "exposure_application", "downstream",
            ),
            "failure_labels": (
                "formation_missing", "formation_incorrect", "persistence_failed",
                "maintenance_stale", "maintenance_conflict", "maintenance_pollution",
                "retrieval_missed", "retrieval_wrong", "application_ignored",
                "non_memory_failure", "unresolved",
            ),
            "repair_axes": ("formation", "persistence", "maintenance", "retrieval", "application"),
            "evidence_planes": ("pure_process", "benchmark_audit", "final_evaluation"),
            "updater_allowed_planes": ("pure_process",),
            "audit_only_fields": ("task_component_score", "official_score", "grader", "hidden_answer"),
            "raw_resource_fields": (
                "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
                "reasoning_tokens", "latency_ms", "storage_bytes", "api_calls", "retry_count",
            ),
            "panels": {
                "semantic": tuple(f"SM0{index}" for index in range(1, 8)),
                "episodic": tuple(f"EP0{index}" for index in range(1, 4)),
                "procedural": tuple(
                    [f"PC01_{index:02d}" for index in range(1, 7)]
                    + ["PC02_01", "PC02_02", "PC03_01", "PC04_01"]
                ),
                "retrieval_auxiliary": tuple(f"PG0{index}" for index in range(1, 7)),
            },
            "isolation_dimensions": (
                "service_process", "service_port", "service_fixture", "hermes_home",
                "state", "session", "artifact", "trace", "native_anchor",
            ),
            "infrastructure_exclusions": (
                "provider_failure", "service_failure", "fixture_identity_mismatch",
                "usage_incomplete", "state_identity_mismatch",
            ),
            "historical_experiments": {
                "five_condition_sensitivity": HISTORICAL_STATUS,
                "shortcut_current_input": HISTORICAL_STATUS,
                "wrong_mechanism": HISTORICAL_STATUS,
                "type_matched_oracle": HISTORICAL_STATUS,
            },
            "provider_id": "coding.tu-zi.com/v1",
            "model_id": "gpt-5.6-luna",
            "sampling_temperature": 0.0,
            "retry_policy": "bounded_retry.v1",
        }
        identity = {
            key: ({name: list(items) for name, items in sorted(value.items())}
                  if key == "panels"
                  else dict(sorted(value.items())) if isinstance(value, dict)
                  else list(value) if isinstance(value, tuple)
                  else value)
            for key, value in values.items()
        }
        return cls(
            protocol_id=f"{PROTOCOL_NAME}.{_digest(identity)[:40]}",
            **values,  # type: ignore[arg-type]
        )

    @classmethod
    def from_payload(cls, value: object) -> "NativeAttributionRepairProtocol":
        if not isinstance(value, Mapping):
            raise ValueError("malformed native attribution protocol")
        expected = set(cls.create().payload())
        if set(value) != expected or not isinstance(value.get("panels"), Mapping):
            raise ValueError("malformed native attribution protocol")
        try:
            protocol = cls(
                protocol_id=value["protocol_id"],
                schema=value["schema"],
                schema_version=value["schema_version"],
                baseline_condition=value["baseline_condition"],
                background_lower_bound=value["background_lower_bound"],
                execution_flow=tuple(value["execution_flow"]),
                memory_kinds=tuple(value["memory_kinds"]),
                lifecycle_surfaces=tuple(value["lifecycle_surfaces"]),
                failure_labels=tuple(value["failure_labels"]),
                repair_axes=tuple(value["repair_axes"]),
                evidence_planes=tuple(value["evidence_planes"]),
                updater_allowed_planes=tuple(value["updater_allowed_planes"]),
                audit_only_fields=tuple(value["audit_only_fields"]),
                raw_resource_fields=tuple(value["raw_resource_fields"]),
                panels={key: tuple(items) for key, items in value["panels"].items()},
                isolation_dimensions=tuple(value["isolation_dimensions"]),
                infrastructure_exclusions=tuple(value["infrastructure_exclusions"]),
                historical_experiments=dict(value["historical_experiments"]),
                provider_id=value["provider_id"],
                model_id=value["model_id"],
                sampling_temperature=value["sampling_temperature"],
                retry_policy=value["retry_policy"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed native attribution protocol") from exc
        if protocol.payload() != dict(value):
            raise ValueError("non-canonical native attribution protocol")
        return protocol


class NativeAttributionProtocolStore:
    """Crash-safe append-once store for the current research protocol."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()

    def freeze(self, protocol: NativeAttributionRepairProtocol) -> bool:
        serialized = _canonical(protocol.payload()) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self.path.is_symlink() or lock_path.is_symlink():
                raise ValueError("protocol store cannot be symlinked")
            if self.path.exists():
                if self.path.read_text(encoding="utf-8") != serialized:
                    raise ValueError("protocol conflicts with existing manifest")
                return False
            descriptor, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return True

    def get(self) -> NativeAttributionRepairProtocol:
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError("protocol manifest is missing or symlinked")
        try:
            return NativeAttributionRepairProtocol.from_payload(
                json.loads(self.path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError("malformed native attribution protocol") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol = NativeAttributionRepairProtocol.create()
    NativeAttributionProtocolStore(args.output).freeze(protocol)
    print(_canonical({"protocolId": protocol.protocol_id, "protocolDigest": protocol.protocol_digest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
