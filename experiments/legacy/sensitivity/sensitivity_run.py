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
from .sensitivity import SensitivityMatrix, SensitivityPanel


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
    """One case-bound, host-owned condition deployment without content.

    The deployment is deliberately more specific than a condition label.  A
    semantic preseed or a shortcut task for one PAST family is not evidence
    that an episodic or procedural family has the same condition available.
    """

    deployment_id: str
    case_id: str
    panel: SensitivityPanel
    target_kind: str
    condition: SensitivityCondition
    mechanism: str
    episode_selector: str
    state_mode: str
    host_state_digest: str
    launcher_config_digest: str
    executable: bool
    schema: str = SENSITIVITY_RUN_SCHEMA
    schema_version: int = SENSITIVITY_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != SENSITIVITY_RUN_SCHEMA or self.schema_version != SENSITIVITY_RUN_SCHEMA_VERSION:
            raise ValueError("unsupported sensitivity deployment schema")
        _id(self.deployment_id, "sensitivity deployment ID")
        _id(self.case_id, "sensitivity deployment case ID")
        object.__setattr__(self, "panel", SensitivityPanel(self.panel))
        _id(self.target_kind, "sensitivity deployment target kind")
        if self.target_kind != self.panel.memory_kind.value:
            raise ValueError("sensitivity deployment target kind must match panel")
        object.__setattr__(self, "condition", SensitivityCondition(self.condition))
        _id(self.mechanism, "sensitivity deployment mechanism")
        _id(self.episode_selector, "sensitivity deployment episode selector")
        if self.state_mode not in {
            "no_persistence", "native_static", "oracle_seed",
            "shortcut_task", "wrong_mechanism_task",
        }:
            raise ValueError("sensitivity deployment state mode is invalid")
        _sha(self.host_state_digest, "sensitivity deployment host-state digest")
        _sha(self.launcher_config_digest, "sensitivity deployment launcher-config digest")
        if type(self.executable) is not bool:
            raise ValueError("sensitivity deployment executable must be bool")

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "deployment_id": self.deployment_id,
            "case_id": self.case_id,
            "panel": self.panel.value,
            "target_kind": self.target_kind,
            "condition": self.condition.value,
            "mechanism": self.mechanism,
            "episode_selector": self.episode_selector,
            "state_mode": self.state_mode,
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
        expected_case_conditions = {
            (run.case_id, run.condition)
            for run in self.runs
        }
        if {(item.case_id, item.condition) for item in deployments} != expected_case_conditions:
            raise ValueError("sensitivity deployments are incomplete or duplicated")
        if len({item.deployment_id for item in deployments}) != len(deployments):
            raise ValueError("sensitivity deployment IDs must be unique")
        object.__setattr__(self, "deployments", deployments)
        runs = tuple(self.runs)
        if not runs or len({run.run_id for run in runs}) != len(runs):
            raise ValueError("sensitivity manifest requires unique runs")
        if any(run.matrix_digest != self.matrix_digest or run.protocol_digest != self.protocol_digest for run in runs):
            raise ValueError("sensitivity run matrix/protocol identity mismatch")
        deployment_by_id = {item.deployment_id: item for item in deployments}
        if {run.deployment_id for run in runs} - set(deployment_by_id):
            raise ValueError("sensitivity run references an unknown deployment")
        for run in runs:
            deployment = deployment_by_id[run.deployment_id]
            if (
                deployment.case_id != run.case_id
                or deployment.condition is not run.condition
                or deployment.panel.value != run.panel
            ):
                raise ValueError("sensitivity run deployment does not match case identity")
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
    deployment_by_case_condition = {(item.case_id, item.condition): item for item in deployments}
    if len(deployment_by_case_condition) != len(tuple(deployments)):
        raise ValueError("sensitivity deployment case conditions must be unique")
    expected_case_conditions = {
        (case.case_id, case.condition)
        for case in matrix.cases
    }
    if set(deployment_by_case_condition) != expected_case_conditions:
        raise ValueError("sensitivity deployment case conditions are incomplete")
    runs: list[SensitivityRunSpec] = []
    for case in matrix.cases:
        deployment = deployment_by_case_condition[(case.case_id, case.condition)]
        if (
            deployment.panel is not case.panel
            or deployment.target_kind != case.target_kind.value
        ):
            raise ValueError("sensitivity deployment panel/kind does not match case")
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


def planned_deployments_from_catalog(
    *,
    matrix: SensitivityMatrix,
    catalog: object,
) -> tuple[SensitivityDeployment, ...]:
    """Translate verified family-level availability into deployments.

    Availability is sufficient for native and named control task slices: the
    launcher can resolve their immutable task selector directly from the
    checked-in family manifest. Correct-type oracle artifacts remain
    non-executable until a case-bound seed registry verifies their host state.
    """

    from .past_sensitivity_catalog import PastSensitivityCatalog

    if not isinstance(catalog, PastSensitivityCatalog):
        raise TypeError("sensitivity deployment planning requires a PAST catalog")
    if catalog.matrix_id != matrix.matrix_id or catalog.matrix_digest != matrix.matrix_digest:
        raise ValueError("sensitivity catalog does not match matrix identity")
    entries = {(entry.case_id, entry.condition): entry for entry in catalog.entries}
    expected = {(case.case_id, case.condition) for case in matrix.cases}
    if set(entries) != expected:
        raise ValueError("sensitivity catalog entries are incomplete")
    state_modes = {
        SensitivityCondition.NO_PERSISTENCE: "no_persistence",
        SensitivityCondition.NATIVE_STATIC: "native_static",
        SensitivityCondition.TYPE_MATCHED_ORACLE: "oracle_seed",
        SensitivityCondition.SHORTCUT_CURRENT_INPUT: "shortcut_task",
        SensitivityCondition.WRONG_MECHANISM: "wrong_mechanism_task",
    }
    deployments: list[SensitivityDeployment] = []
    for case in matrix.cases:
        entry = entries[(case.case_id, case.condition)]
        selector = entry.episode_selector or f"unavailable.{case.condition.value}"
        values = {
            "case_id": case.case_id,
            "panel": case.panel,
            "target_kind": case.target_kind.value,
            "condition": case.condition,
            "mechanism": case.mechanism,
            "episode_selector": selector,
            "state_mode": state_modes[case.condition],
            "host_state_digest": _digest({"availability": entry.payload()}),
            "launcher_config_digest": _digest({"availability": entry.payload(), "mode": state_modes[case.condition]}),
            "executable": (
                entry.available
                and case.condition is not SensitivityCondition.TYPE_MATCHED_ORACLE
            ),
        }
        deployments.append(SensitivityDeployment(
            deployment_id="sensitivity-deployment." + _digest(values)[:40],
            **values,
        ))
    return tuple(deployments)


def apply_verified_oracle_seed_registry(
    *,
    matrix: SensitivityMatrix,
    catalog: object,
    deployments: Sequence[SensitivityDeployment],
    registry: object,
    trusted_root: Path,
) -> tuple[SensitivityDeployment, ...]:
    """Upgrade only oracle deployments backed by verified, case-bound seeds."""

    from .oracle_seed_registry import OracleSeedRegistry
    from .past_sensitivity_catalog import PastSensitivityCatalog

    if not isinstance(catalog, PastSensitivityCatalog) or not isinstance(registry, OracleSeedRegistry):
        raise TypeError("oracle deployment upgrade requires catalog and seed registry")
    if catalog.matrix_id != matrix.matrix_id or catalog.matrix_digest != matrix.matrix_digest:
        raise ValueError("oracle deployment catalog does not match matrix")
    by_case = {(item.case_id, item.condition): item for item in catalog.entries}
    result: list[SensitivityDeployment] = []
    for deployment in deployments:
        if deployment.condition is not SensitivityCondition.TYPE_MATCHED_ORACLE:
            result.append(deployment)
            continue
        case_matches = [case for case in matrix.cases if case.case_id == deployment.case_id]
        if len(case_matches) != 1:
            raise ValueError("oracle deployment case is not in matrix")
        case = case_matches[0]
        availability = by_case.get((case.case_id, case.condition))
        if availability is None:
            raise ValueError("oracle deployment catalog entry is missing")
        try:
            registration = registry.for_case(case.case_id)
            seed = registration.resolve(trusted_root, case, availability.family_source_digest)
        except ValueError:
            # Missing or invalid registrations cannot silently make a case runnable.
            result.append(deployment)
            continue
        values = {
            "case_id": deployment.case_id,
            "panel": deployment.panel,
            "target_kind": deployment.target_kind,
            "condition": deployment.condition,
            "mechanism": deployment.mechanism,
            "episode_selector": "family.eval_only",
            "state_mode": deployment.state_mode,
            "host_state_digest": registration.seed_tree_digest,
            "launcher_config_digest": _digest({
                "previous": deployment.launcher_config_digest,
                "registration": registration.payload(),
            }),
            "executable": True,
        }
        result.append(SensitivityDeployment(
            deployment_id="sensitivity-deployment." + _digest({
                **values,
                "panel": deployment.panel.value,
                "condition": deployment.condition.value,
            })[:40],
            **values,
        ))
        if not seed.is_dir():  # resolve already verifies this; preserve explicit contract.
            raise ValueError("verified oracle seed is not a directory")
    return tuple(result)


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
    "apply_verified_oracle_seed_registry",
    "planned_deployments_from_catalog",
]
