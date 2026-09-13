"""Immutable native-static run manifests for failure-attribution studies."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from rsimem.memory.family_matrix import PastFamilyMatrix
from .native_attribution_protocol import NativeAttributionRepairProtocol


RUN_SCHEMA = "rsimem-native-attribution-run-v1"
RUN_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,255}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_PORT_STRIDE = 257
_MAX_SERVICE_PORT = 32767


@dataclass(frozen=True, slots=True)
class NativeServiceIdentitySpec:
    episode_id: str
    task_id: str
    service: str
    port: int
    fixture_digest: str

    def __post_init__(self) -> None:
        _identifier(self.episode_id, "service episode ID")
        _identifier(self.task_id, "service task ID")
        _identifier(self.service, "service name")
        if type(self.port) is not int or self.port < 1024 or self.port > _MAX_SERVICE_PORT:
            raise ValueError("service identity port is invalid")
        _sha(self.fixture_digest, "service fixture digest")

    def payload(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "service": self.service,
            "port": self.port,
            "fixture_digest": self.fixture_digest,
        }

    @classmethod
    def from_payload(cls, value: object) -> "NativeServiceIdentitySpec":
        if not isinstance(value, Mapping) or set(value) != {
            "episode_id", "task_id", "service", "port", "fixture_digest",
        }:
            raise ValueError("malformed native service identity")
        return cls(**value)


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a sha256 digest")
    return value


def _relative(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{name} must be a nonempty relative path")
    path = Path(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must not escape the batch root")
    return path.as_posix()


def _tree_digest(root: Path) -> str:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("family source root is missing or symlinked")
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("family source tree cannot contain symlinks")
        if path.is_file():
            entries.append({
                "path": path.relative_to(root).as_posix(),
                "digest": _file_digest(path),
                "size": path.stat().st_size,
            })
    return _digest(entries)


def _family_runtime_identity(
    *, family_id: str, task_root: str, past_bench_root: Path
) -> tuple[
    str, str, tuple[int, ...], tuple[str, ...], tuple[str, ...],
    tuple[NativeServiceIdentitySpec, ...]
]:
    family_root = (past_bench_root / task_root).resolve()
    family_file = family_root / "family.yaml"
    try:
        family_doc = yaml.safe_load(family_file.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("family source is unreadable") from exc
    if not isinstance(family_doc, Mapping) or family_doc.get("family_id") != family_id:
        raise ValueError("family source identity mismatch")
    order = family_doc.get("episode_order")
    if not isinstance(order, list) or not order or not all(isinstance(v, str) for v in order):
        raise ValueError("family episode order is malformed")

    fixture_entries: list[dict[str, object]] = []
    ports: set[int] = set()
    native_episodes: list[str] = []
    native_tasks: list[str] = []
    service_identities: list[NativeServiceIdentitySpec] = []
    for episode_name in order:
        if "control" in episode_name.lower():
            continue
        task_path = family_root / episode_name / "task.yaml"
        try:
            task = yaml.safe_load(task_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError("family task source is unreadable") from exc
        if not isinstance(task, Mapping):
            raise ValueError("family task source is malformed")
        task_id = task.get("task_id")
        if not isinstance(task_id, str):
            raise ValueError("family task ID is malformed")
        native_episodes.append(episode_name)
        native_tasks.append(task_id)
        services = task.get("services") or []
        if not isinstance(services, list):
            raise ValueError("family task services are malformed")
        for service in services:
            if not isinstance(service, Mapping) or not isinstance(service.get("port"), int):
                raise ValueError("family task service is malformed")
            ports.add(service["port"])
            env = service.get("env") or {}
            if not isinstance(env, Mapping):
                raise ValueError("family task service environment is malformed")
            service_fixtures: dict[str, dict[str, object]] = {}
            for key, raw in sorted(env.items()):
                if not isinstance(key, str) or not key.endswith("_FIXTURES"):
                    continue
                if not isinstance(raw, str):
                    raise ValueError("fixture path must be text")
                fixture_path = Path(raw)
                if not fixture_path.is_absolute():
                    fixture_path = past_bench_root / fixture_path
                fixture_path = fixture_path.resolve()
                if fixture_path.is_symlink() or not fixture_path.is_file():
                    raise ValueError("declared family fixture is missing or symlinked")
                fixture_entries.append({
                    "episode": episode_name,
                    "service": service.get("name"),
                    "key": key,
                    "digest": _file_digest(fixture_path),
                    "size": fixture_path.stat().st_size,
                })
                service_fixtures[key] = {
                    "digest": _file_digest(fixture_path),
                    "size": fixture_path.stat().st_size,
                }
            service_name = service.get("name")
            if not isinstance(service_name, str):
                raise ValueError("family service name is malformed")
            service_identities.append(NativeServiceIdentitySpec(
                episode_id=episode_name,
                task_id=task_id,
                service=service_name,
                port=service["port"],
                fixture_digest=_digest(service_fixtures),
            ))
    if not native_episodes or not ports:
        raise ValueError("native family must declare episodes and service ports")
    return (
        _tree_digest(family_root),
        _digest(fixture_entries),
        tuple(sorted(ports)),
        tuple(native_episodes),
        tuple(native_tasks),
        tuple(service_identities),
    )


@dataclass(frozen=True, slots=True)
class NativeAttributionRunSpec:
    run_id: str
    method_case_id: str
    family_id: str
    panel: str
    memory_kind: str | None
    replicate: int
    condition: str
    seed: int
    port_offset: int
    service_ports: tuple[int, ...]
    family_source_digest: str
    fixture_digest: str
    native_episode_ids: tuple[str, ...]
    native_task_ids: tuple[str, ...]
    service_identities: tuple[NativeServiceIdentitySpec, ...]
    state_directory: str
    hermes_home_directory: str
    session_directory: str
    artifact_directory: str
    trace_directory: str
    initial_home_digest: str
    protocol_id: str
    protocol_digest: str
    provider_id: str
    model_id: str
    rsimem_commit: str
    past_bench_commit: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.run_id, "run ID"), (self.family_id, "family ID"),
            (self.method_case_id, "method case ID"),
            (self.panel, "panel"), (self.condition, "condition"),
            (self.protocol_id, "protocol ID"), (self.provider_id, "provider ID"),
            (self.model_id, "model ID"), (self.rsimem_commit, "RSIMem commit"),
            (self.past_bench_commit, "PAST-Bench commit"),
        ):
            _identifier(value, name)
        if self.condition != "native_static":
            raise ValueError("attribution runs must use native_static")
        if self.memory_kind is not None:
            _identifier(self.memory_kind, "memory kind")
        if type(self.replicate) is not int or self.replicate < 1:
            raise ValueError("replicate must be positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.port_offset) is not int or self.port_offset < 0:
            raise ValueError("port offset must be nonnegative")
        ports = tuple(self.service_ports)
        if not ports or len(ports) != len(set(ports)) or any(
            type(port) is not int or port < 1024 or port > _MAX_SERVICE_PORT for port in ports
        ):
            raise ValueError("service ports must be unique non-privileged ports")
        object.__setattr__(self, "service_ports", ports)
        episodes = tuple(self.native_episode_ids)
        if not episodes or len(episodes) != len(set(episodes)):
            raise ValueError("native episode IDs must be nonempty and unique")
        for value in episodes:
            _identifier(value, "native episode ID")
            if "control" in value.lower():
                raise ValueError("control episode cannot enter a native attribution run")
        object.__setattr__(self, "native_episode_ids", episodes)
        tasks = tuple(self.native_task_ids)
        if len(tasks) != len(episodes) or len(set(tasks)) != len(tasks):
            raise ValueError("native task IDs must align with episodes and be unique")
        for value in tasks:
            _identifier(value, "native task ID")
        object.__setattr__(self, "native_task_ids", tasks)
        try:
            identities = tuple(
                value
                if isinstance(value, NativeServiceIdentitySpec)
                else NativeServiceIdentitySpec.from_payload(value)
                for value in self.service_identities
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("native service identities are malformed") from exc
        if not identities or len({
            (value.episode_id, value.task_id, value.service, value.port)
            for value in identities
        }) != len(identities):
            raise ValueError("native service identities must be nonempty and unique")
        if {value.port for value in identities} != set(ports):
            raise ValueError("native service identities do not cover run ports")
        if not {value.episode_id for value in identities}.issubset(set(episodes)):
            raise ValueError("native service identity references an unknown episode")
        object.__setattr__(self, "service_identities", identities)
        for value, name in (
            (self.family_source_digest, "family source digest"),
            (self.fixture_digest, "fixture digest"),
            (self.initial_home_digest, "initial home digest"),
            (self.protocol_digest, "protocol digest"),
        ):
            _sha(value, name)
        directories = (
            self.state_directory, self.hermes_home_directory, self.session_directory,
            self.artifact_directory, self.trace_directory,
        )
        for value in directories:
            _relative(value, "run directory")
        if len(set(directories)) != len(directories):
            raise ValueError("run directories must be distinct")
        if self.run_id != "native-run." + _digest(self.identity_payload())[:40]:
            raise ValueError("native attribution run ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "method_case_id": self.method_case_id,
            "family_id": self.family_id, "panel": self.panel,
            "memory_kind": self.memory_kind, "replicate": self.replicate,
            "condition": self.condition, "seed": self.seed,
            "port_offset": self.port_offset, "service_ports": list(self.service_ports),
            "family_source_digest": self.family_source_digest,
            "fixture_digest": self.fixture_digest,
            "native_episode_ids": list(self.native_episode_ids),
            "native_task_ids": list(self.native_task_ids),
            "service_identities": [value.payload() for value in self.service_identities],
            "state_directory": self.state_directory,
            "hermes_home_directory": self.hermes_home_directory,
            "session_directory": self.session_directory,
            "artifact_directory": self.artifact_directory,
            "trace_directory": self.trace_directory,
            "initial_home_digest": self.initial_home_digest,
            "protocol_id": self.protocol_id, "protocol_digest": self.protocol_digest,
            "provider_id": self.provider_id, "model_id": self.model_id,
            "rsimem_commit": self.rsimem_commit, "past_bench_commit": self.past_bench_commit,
        }

    def payload(self) -> dict[str, object]:
        return {"run_id": self.run_id, **self.identity_payload()}

    @classmethod
    def from_payload(cls, value: object) -> "NativeAttributionRunSpec":
        if not isinstance(value, Mapping):
            raise ValueError("malformed native attribution run")
        fields = {
            "run_id", "method_case_id", "family_id", "panel", "memory_kind",
            "replicate", "condition", "seed", "port_offset", "service_ports",
            "family_source_digest", "fixture_digest", "native_episode_ids",
            "native_task_ids", "service_identities",
            "state_directory", "hermes_home_directory", "session_directory",
            "artifact_directory", "trace_directory", "initial_home_digest",
            "protocol_id", "protocol_digest", "provider_id", "model_id",
            "rsimem_commit", "past_bench_commit",
        }
        if set(value) != fields:
            raise ValueError("malformed native attribution run")
        try:
            run = cls(
                run_id=value["run_id"], method_case_id=value["method_case_id"],
                family_id=value["family_id"], panel=value["panel"],
                memory_kind=value["memory_kind"], replicate=value["replicate"],
                condition=value["condition"], seed=value["seed"],
                port_offset=value["port_offset"], service_ports=tuple(value["service_ports"]),
                family_source_digest=value["family_source_digest"],
                fixture_digest=value["fixture_digest"],
                native_episode_ids=tuple(value["native_episode_ids"]),
                native_task_ids=tuple(value["native_task_ids"]),
                service_identities=tuple(
                    NativeServiceIdentitySpec.from_payload(item)
                    for item in value["service_identities"]
                ),
                state_directory=value["state_directory"],
                hermes_home_directory=value["hermes_home_directory"],
                session_directory=value["session_directory"],
                artifact_directory=value["artifact_directory"],
                trace_directory=value["trace_directory"],
                initial_home_digest=value["initial_home_digest"],
                protocol_id=value["protocol_id"], protocol_digest=value["protocol_digest"],
                provider_id=value["provider_id"], model_id=value["model_id"],
                rsimem_commit=value["rsimem_commit"],
                past_bench_commit=value["past_bench_commit"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed native attribution run") from exc
        if run.payload() != dict(value):
            raise ValueError("non-canonical native attribution run")
        return run


@dataclass(frozen=True, slots=True)
class NativeAttributionRunManifest:
    manifest_id: str
    batch_id: str
    runs: tuple[NativeAttributionRunSpec, ...]
    protocol_id: str
    protocol_digest: str
    schema: str = RUN_SCHEMA
    schema_version: int = RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != RUN_SCHEMA or self.schema_version != RUN_SCHEMA_VERSION:
            raise ValueError("unsupported native attribution run schema")
        _identifier(self.batch_id, "batch ID")
        _identifier(self.protocol_id, "protocol ID")
        _sha(self.protocol_digest, "protocol digest")
        runs = tuple(self.runs)
        if not runs or len({run.run_id for run in runs}) != len(runs):
            raise ValueError("manifest requires unique runs")
        if any(
            run.protocol_id != self.protocol_id or run.protocol_digest != self.protocol_digest
            for run in runs
        ):
            raise ValueError("run protocol identity mismatch")
        slots = {(run.family_id, run.replicate) for run in runs}
        if len(slots) != len(runs):
            raise ValueError("family/replicate slots must be unique")
        ports = [port for run in runs for port in run.service_ports]
        if len(ports) != len(set(ports)):
            raise ValueError("service ports collide across runs")
        directories = [
            value for run in runs for value in (
                run.state_directory, run.hermes_home_directory, run.session_directory,
                run.artifact_directory, run.trace_directory,
            )
        ]
        if len(directories) != len(set(directories)):
            raise ValueError("run directories collide across runs")
        object.__setattr__(self, "runs", runs)
        if self.manifest_id != "native-manifest." + _digest(self.identity_payload())[:40]:
            raise ValueError("native attribution manifest ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema, "schema_version": self.schema_version,
            "batch_id": self.batch_id, "protocol_id": self.protocol_id,
            "protocol_digest": self.protocol_digest,
            "runs": [run.payload() for run in self.runs],
        }

    def payload(self) -> dict[str, object]:
        return {"manifest_id": self.manifest_id, **self.identity_payload()}

    @classmethod
    def from_payload(cls, value: object) -> "NativeAttributionRunManifest":
        if not isinstance(value, Mapping) or set(value) != {
            "manifest_id", "schema", "schema_version", "batch_id", "protocol_id",
            "protocol_digest", "runs",
        } or not isinstance(value.get("runs"), list):
            raise ValueError("malformed native attribution run manifest")
        try:
            manifest = cls(
                manifest_id=value["manifest_id"], schema=value["schema"],
                schema_version=value["schema_version"], batch_id=value["batch_id"],
                protocol_id=value["protocol_id"], protocol_digest=value["protocol_digest"],
                runs=tuple(NativeAttributionRunSpec.from_payload(item) for item in value["runs"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed native attribution run manifest") from exc
        if manifest.payload() != dict(value):
            raise ValueError("non-canonical native attribution run manifest")
        return manifest


def build_native_attribution_manifest(
    *,
    batch_id: str,
    protocol: NativeAttributionRepairProtocol,
    past_bench_root: Path,
    rsimem_commit: str,
    past_bench_commit: str,
    replicate_count: int = 3,
    port_base_offset: int = 0,
) -> NativeAttributionRunManifest:
    _identifier(batch_id, "batch ID")
    if type(replicate_count) is not int or replicate_count < 1:
        raise ValueError("replicate count must be positive")
    root = Path(past_bench_root).expanduser().resolve()
    if not (root / "pyproject.toml").is_file():
        raise ValueError("PAST-Bench root is invalid")
    matrix = PastFamilyMatrix.create_default(replicate_count=replicate_count)
    runs: list[NativeAttributionRunSpec] = []
    slot = 0
    empty_home_digest = _digest([])
    for spec in matrix.families:
        family_digest, fixture_digest, base_ports, episodes, task_ids, base_services = _family_runtime_identity(
            family_id=spec.family_id, task_root=spec.task_root, past_bench_root=root
        )
        method_case_id = "native-case." + _digest({
            "protocol": protocol.protocol_id,
            "family_source_digest": family_digest,
            "fixture_digest": fixture_digest,
        })[:40]
        for replicate in range(1, replicate_count + 1):
            offset = port_base_offset + slot * _PORT_STRIDE
            actual_ports = tuple(port + offset for port in base_ports)
            directory_root = f"runs/{spec.family_id}/replicate-{replicate:02d}/native-static"
            values = {
                "method_case_id": method_case_id,
                "family_id": spec.family_id,
                "panel": spec.panel.value,
                "memory_kind": spec.target_kind.value if spec.target_kind else None,
                "replicate": replicate,
                "condition": "native_static",
                "seed": int(_digest({"batch": batch_id, "family": spec.family_id, "replicate": replicate})[:8], 16),
                "port_offset": offset,
                "service_ports": actual_ports,
                "family_source_digest": family_digest,
                "fixture_digest": fixture_digest,
                "native_episode_ids": episodes,
                "native_task_ids": task_ids,
                "service_identities": tuple(NativeServiceIdentitySpec(
                    episode_id=value.episode_id,
                    task_id=value.task_id,
                    service=value.service,
                    port=value.port + offset,
                    fixture_digest=value.fixture_digest,
                ) for value in base_services),
                "state_directory": directory_root + "/state",
                "hermes_home_directory": directory_root + "/hermes-home",
                "session_directory": directory_root + "/hermes-home/sessions",
                "artifact_directory": directory_root + "/artifacts",
                "trace_directory": directory_root + "/trace",
                "initial_home_digest": empty_home_digest,
                "protocol_id": protocol.protocol_id,
                "protocol_digest": protocol.protocol_digest,
                "provider_id": protocol.provider_id,
                "model_id": protocol.model_id,
                "rsimem_commit": rsimem_commit,
                "past_bench_commit": past_bench_commit,
            }
            run_identity = {
                **values,
                "service_identities": [
                    value.payload() for value in values["service_identities"]
                ],
            }
            runs.append(NativeAttributionRunSpec(
                run_id="native-run." + _digest(run_identity)[:40], **values
            ))
            slot += 1
    identity = {
        "schema": RUN_SCHEMA, "schema_version": RUN_SCHEMA_VERSION,
        "batch_id": batch_id, "protocol_id": protocol.protocol_id,
        "protocol_digest": protocol.protocol_digest,
        "runs": [run.payload() for run in runs],
    }
    return NativeAttributionRunManifest(
        manifest_id="native-manifest." + _digest(identity)[:40],
        batch_id=batch_id, runs=tuple(runs), protocol_id=protocol.protocol_id,
        protocol_digest=protocol.protocol_digest,
    )


class NativeAttributionRunManifestStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()

    def initialize(self, manifest: NativeAttributionRunManifest) -> bool:
        serialized = _canonical(manifest.payload()) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self.path.is_symlink() or lock_path.is_symlink():
                raise ValueError("run manifest store cannot be symlinked")
            if self.path.exists():
                if self.path.read_text(encoding="utf-8") != serialized:
                    raise ValueError("run manifest conflicts with existing manifest")
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

    def get(self) -> NativeAttributionRunManifest:
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError("run manifest is missing or symlinked")
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return NativeAttributionRunManifest.from_payload(value)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError("malformed native attribution run manifest") from exc


__all__ = [
    "NativeAttributionRunManifest", "NativeAttributionRunManifestStore",
    "NativeAttributionRunSpec", "NativeServiceIdentitySpec",
    "build_native_attribution_manifest",
]
