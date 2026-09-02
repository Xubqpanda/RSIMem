"""Immutable, fail-closed run specifications for PAST sensitivity studies.

The matrix is an audit-plane object.  A runner needs the PAST family to select
work, but the method boundary receives only the opaque sensitivity case ID.
This module deliberately stores deployment identities and digests, never
oracle or task content.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .research_protocol import SensitivityCondition
from .sensitivity import SensitivityMatrix


SENSITIVITY_RUN_SCHEMA = "rsimem-past-sensitivity-run-v1"
SENSITIVITY_RUN_SCHEMA_VERSION = 1
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
        raise ValueError(f"{name} must be a sha256 digest")
    return value


def _relative_path(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{name} must be a nonempty relative path")
    path = Path(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must not escape the batch root")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class SensitivityDeployment:
    """One host-owned condition deployment, represented without content."""

    deployment_id: str
    condition: SensitivityCondition
    mechanism: str
    host_state_digest: str
    launcher_config_digest: str
    executable: bool
    schema: str = SENSITIVITY_RUN_SCHEMA
    schema_version: int = SENSITIVITY_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != SENSITIVITY_RUN_SCHEMA or self.schema_version != SENSITIVITY_RUN_SCHEMA_VERSION:
            raise ValueError("unsupported sensitivity deployment schema")
        _id(self.deployment_id, "sensitivity deployment ID")
        object.__setattr__(self, "condition", SensitivityCondition(self.condition))
        _id(self.mechanism, "sensitivity deployment mechanism")
        _sha(self.host_state_digest, "sensitivity deployment host-state digest")
        _sha(self.launcher_config_digest, "sensitivity deployment launcher-config digest")
        if type(self.executable) is not bool:
            raise ValueError("sensitivity deployment executable must be bool")

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "deployment_id": self.deployment_id,
            "condition": self.condition.value,
            "mechanism": self.mechanism,
            "host_state_digest": self.host_state_digest,
            "launcher_config_digest": self.launcher_config_digest,
            "executable": self.executable,
        }


@dataclass(frozen=True, slots=True)
class SensitivityRunSpec:
    """One isolated case/condition/replicate execution specification."""

    run_id: str
    case_id: str
    family_id: str
    condition: SensitivityCondition
    panel: str
    replicate: int
    method_task_id: str
    deployment_id: str
    state_directory: str
    hermes_home_directory: str
    trace_directory: str
    matrix_digest: str
    protocol_digest: str
    provider_id: str
    model_id: str
    tool_budget: int
    max_turns: int
    retry_policy: str
    schema: str = SENSITIVITY_RUN_SCHEMA
    schema_version: int = SENSITIVITY_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != SENSITIVITY_RUN_SCHEMA or self.schema_version != SENSITIVITY_RUN_SCHEMA_VERSION:
            raise ValueError("unsupported sensitivity run schema")
        for value, name in (
            (self.run_id, "sensitivity run ID"),
            (self.case_id, "sensitivity case ID"),
            (self.family_id, "sensitivity family ID"),
            (self.panel, "sensitivity panel"),
            (self.method_task_id, "method task ID"),
            (self.deployment_id, "sensitivity deployment ID"),
            (self.provider_id, "provider ID"),
            (self.model_id, "model ID"),
            (self.retry_policy, "retry policy"),
        ):
            _id(value, name)
        object.__setattr__(self, "condition", SensitivityCondition(self.condition))
        if self.method_task_id != self.case_id:
            raise ValueError("method task ID must equal the opaque sensitivity case ID")
        if type(self.replicate) is not int or self.replicate < 1:
            raise ValueError("sensitivity replicate must be positive")
        if type(self.tool_budget) is not int or self.tool_budget < 1:
            raise ValueError("sensitivity tool budget must be positive")
        if type(self.max_turns) is not int or self.max_turns < 1:
            raise ValueError("sensitivity max turns must be positive")
        for value, name in (
            (self.state_directory, "state directory"),
            (self.hermes_home_directory, "Hermes home directory"),
            (self.trace_directory, "trace directory"),
        ):
            _relative_path(value, name)
        _sha(self.matrix_digest, "sensitivity matrix digest")
        _sha(self.protocol_digest, "sensitivity protocol digest")
        expected = "sensitivity-run." + _digest(self.identity_payload())[:40]
        if self.run_id != expected:
            raise ValueError("sensitivity run ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "family_id": self.family_id,
            "condition": self.condition.value,
            "panel": self.panel,
            "replicate": self.replicate,
            "method_task_id": self.method_task_id,
            "deployment_id": self.deployment_id,
            "state_directory": self.state_directory,
            "hermes_home_directory": self.hermes_home_directory,
            "trace_directory": self.trace_directory,
            "matrix_digest": self.matrix_digest,
            "protocol_digest": self.protocol_digest,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "tool_budget": self.tool_budget,
            "max_turns": self.max_turns,
            "retry_policy": self.retry_policy,
        }

    def payload(self) -> dict[str, object]:
        return {"run_id": self.run_id, **self.identity_payload()}

    def method_runtime_metadata(self) -> dict[str, str]:
        """The complete launcher-provided method metadata allowlist."""

        return {"rsimem_method_task_id": self.method_task_id}


@dataclass(frozen=True, slots=True)
class SensitivityRunManifest:
    """A registered batch plan before any provider or benchmark execution."""

    manifest_id: str
    batch_id: str
    matrix_id: str
    matrix_digest: str
    protocol_id: str
    protocol_digest: str
    deployments: tuple[SensitivityDeployment, ...]
    runs: tuple[SensitivityRunSpec, ...]
    rsimem_commit: str
    past_bench_commit: str
    schema: str = SENSITIVITY_RUN_SCHEMA
    schema_version: int = SENSITIVITY_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != SENSITIVITY_RUN_SCHEMA or self.schema_version != SENSITIVITY_RUN_SCHEMA_VERSION:
            raise ValueError("unsupported sensitivity manifest schema")
        for value, name in (
            (self.batch_id, "sensitivity batch ID"),
            (self.matrix_id, "sensitivity matrix ID"),
            (self.protocol_id, "sensitivity protocol ID"),
            (self.rsimem_commit, "RSIMem commit"),
            (self.past_bench_commit, "PAST-Bench commit"),
        ):
            _id(value, name)
        _sha(self.matrix_digest, "sensitivity matrix digest")
        _sha(self.protocol_digest, "sensitivity protocol digest")
        deployments = tuple(self.deployments)
        if len(deployments) != len(SensitivityCondition):
            raise ValueError("sensitivity manifest requires one deployment per condition")
        if {item.condition for item in deployments} != set(SensitivityCondition):
            raise ValueError("sensitivity deployments are incomplete or duplicated")
        if len({item.deployment_id for item in deployments}) != len(deployments):
            raise ValueError("sensitivity deployment IDs must be unique")
        object.__setattr__(self, "deployments", deployments)
        runs = tuple(self.runs)
        if not runs or len({run.run_id for run in runs}) != len(runs):
            raise ValueError("sensitivity manifest requires unique runs")
        if any(run.matrix_digest != self.matrix_digest or run.protocol_digest != self.protocol_digest for run in runs):
            raise ValueError("sensitivity run matrix/protocol identity mismatch")
        if {run.deployment_id for run in runs} - {item.deployment_id for item in deployments}:
            raise ValueError("sensitivity run references an unknown deployment")
        directories = [
            value
            for run in runs
            for value in (run.state_directory, run.hermes_home_directory, run.trace_directory)
        ]
        if len(directories) != len(set(directories)):
            raise ValueError("sensitivity run directories must be isolated")
        object.__setattr__(self, "runs", runs)
        expected = "sensitivity-manifest." + _digest(self.identity_payload())[:40]
        if self.manifest_id != expected:
            raise ValueError("sensitivity manifest ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "batch_id": self.batch_id,
            "matrix_id": self.matrix_id,
            "matrix_digest": self.matrix_digest,
            "protocol_id": self.protocol_id,
            "protocol_digest": self.protocol_digest,
            "deployments": [item.payload() for item in self.deployments],
            "runs": [run.payload() for run in self.runs],
            "rsimem_commit": self.rsimem_commit,
            "past_bench_commit": self.past_bench_commit,
        }

    def payload(self) -> dict[str, object]:
        return {"manifest_id": self.manifest_id, **self.identity_payload()}

    @property
    def execution_ready(self) -> bool:
        return all(item.executable for item in self.deployments)

    def require_execution_ready(self) -> None:
        missing = sorted(item.condition.value for item in self.deployments if not item.executable)
        if missing:
            raise ValueError("sensitivity condition deployments are not executable: " + ", ".join(missing))


def build_sensitivity_run_manifest(
    *,
    batch_id: str,
    matrix: SensitivityMatrix,
    protocol_digest: str,
    provider_id: str,
    model_id: str,
    tool_budget: int,
    max_turns: int,
    retry_policy: str,
    deployments: Sequence[SensitivityDeployment],
    rsimem_commit: str,
    past_bench_commit: str,
) -> SensitivityRunManifest:
    """Expand every matrix case and replicate into independently isolated runs."""

    _id(batch_id, "sensitivity batch ID")
    _sha(protocol_digest, "sensitivity protocol digest")
    deployment_by_condition = {item.condition: item for item in deployments}
    if len(deployment_by_condition) != len(tuple(deployments)):
        raise ValueError("sensitivity deployment conditions must be unique")
    if set(deployment_by_condition) != set(SensitivityCondition):
        raise ValueError("sensitivity deployment conditions are incomplete")
    runs: list[SensitivityRunSpec] = []
    for case in matrix.cases:
        deployment = deployment_by_condition[case.condition]
        for replicate in range(1, matrix.replicate_count + 1):
            directory_root = f"runs/{case.case_id}/replicate-{replicate:02d}"
            values = {
                "case_id": case.case_id,
                "family_id": case.family_id,
                "condition": case.condition.value,
                "panel": case.panel.value,
                "replicate": replicate,
                "method_task_id": case.case_id,
                "deployment_id": deployment.deployment_id,
                "state_directory": directory_root + "/state",
                "hermes_home_directory": directory_root + "/hermes-home",
                "trace_directory": directory_root + "/trace",
                "matrix_digest": matrix.matrix_digest,
                "protocol_digest": protocol_digest,
                "provider_id": provider_id,
                "model_id": model_id,
                "tool_budget": tool_budget,
                "max_turns": max_turns,
                "retry_policy": retry_policy,
            }
            runs.append(SensitivityRunSpec(
                run_id="sensitivity-run." + _digest(values)[:40],
                **values,
            ))
    values = {
        "schema": SENSITIVITY_RUN_SCHEMA,
        "schema_version": SENSITIVITY_RUN_SCHEMA_VERSION,
        "batch_id": batch_id,
        "matrix_id": matrix.matrix_id,
        "matrix_digest": matrix.matrix_digest,
        "protocol_id": matrix.protocol_id,
        "protocol_digest": protocol_digest,
        "deployments": [item.payload() for item in deployments],
        "runs": [run.payload() for run in runs],
        "rsimem_commit": rsimem_commit,
        "past_bench_commit": past_bench_commit,
    }
    return SensitivityRunManifest(
        manifest_id="sensitivity-manifest." + _digest(values)[:40],
        batch_id=batch_id,
        matrix_id=matrix.matrix_id,
        matrix_digest=matrix.matrix_digest,
        protocol_id=matrix.protocol_id,
        protocol_digest=protocol_digest,
        deployments=tuple(deployments),
        runs=tuple(runs),
        rsimem_commit=rsimem_commit,
        past_bench_commit=past_bench_commit,
    )


class SensitivityRunManifestStore:
    """Append-once registration for an immutable Stage 3 run manifest."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def initialize(self, manifest: SensitivityRunManifest) -> None:
        expected = manifest.payload()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".sensitivity-manifest-", dir=self.path.parent
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                temporary.write_text(json.dumps(expected, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                try:
                    os.link(temporary, self.path)
                except FileExistsError:
                    pass
                else:
                    return
            finally:
                temporary.unlink(missing_ok=True)
            existing = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("sensitivity manifest store is unreadable") from exc
        if existing != expected:
            raise ValueError("sensitivity manifest store already contains a different manifest")


__all__ = [
    "SENSITIVITY_RUN_SCHEMA",
    "SENSITIVITY_RUN_SCHEMA_VERSION",
    "SensitivityDeployment",
    "SensitivityRunManifest",
    "SensitivityRunManifestStore",
    "SensitivityRunSpec",
    "build_sensitivity_run_manifest",
]
