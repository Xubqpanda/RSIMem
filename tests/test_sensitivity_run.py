from __future__ import annotations

import hashlib

import pytest

from rsimem.memory import MemoryKind
from rsimem.memory.family_matrix import PastFamilyMatrix
from rsimem.research_protocol import (
    ResearchProtocol,
    SensitivityCondition,
    default_research_protocol,
)
from rsimem.sensitivity import SensitivityMatrix, SensitivityPanel
from rsimem.sensitivity_run import (
    SensitivityDeployment,
    SensitivityRunManifestStore,
    build_sensitivity_run_manifest,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _matrix() -> SensitivityMatrix:
    base = default_research_protocol()
    protocol = ResearchProtocol.create(
        memory_units=base.memory_units,
        family_matrix=PastFamilyMatrix.create_default(),
        split=base.split,
        sensitivity_target_kind=MemoryKind.SEMANTIC,
    )
    return SensitivityMatrix.create_for_panel(
        panel=SensitivityPanel.SEMANTIC,
        protocol=protocol,
        family_matrix=PastFamilyMatrix.create_default(),
    )


def _deployments() -> tuple[SensitivityDeployment, ...]:
    return tuple(
        SensitivityDeployment(
            deployment_id=f"deployment.{condition.value}",
            condition=condition,
            mechanism=f"deployment.{condition.value}",
            host_state_digest=_digest("host-state:" + condition.value),
            launcher_config_digest=_digest("launcher-config:" + condition.value),
            executable=condition.value in {"no_persistence", "native_static"},
        )
        for condition in SensitivityCondition
    )


def _manifest():
    matrix = _matrix()
    return build_sensitivity_run_manifest(
        batch_id="stage3-semantic-fixture",
        matrix=matrix,
        protocol_digest=_digest("protocol"),
        provider_id="coding.tu-zi.com/v1",
        model_id="gpt-5.6-luna",
        tool_budget=32,
        max_turns=20,
        retry_policy="bounded_retry.v1",
        deployments=_deployments(),
        rsimem_commit="commit.rsimem.fixture",
        past_bench_commit="commit.past.fixture",
    )


def test_expands_every_case_condition_and_replicate_with_isolated_directories() -> None:
    manifest = _manifest()
    assert len(manifest.runs) == 7 * 5 * 3
    assert len({run.run_id for run in manifest.runs}) == len(manifest.runs)
    locations = [
        location
        for run in manifest.runs
        for location in (run.state_directory, run.hermes_home_directory, run.trace_directory)
    ]
    assert len(locations) == len(set(locations))
    assert manifest.execution_ready is False
    with pytest.raises(ValueError, match="not executable"):
        manifest.require_execution_ready()


def test_method_metadata_is_opaque_and_replay_stable() -> None:
    first = _manifest()
    second = _manifest()
    assert first.payload() == second.payload()
    run = first.runs[0]
    assert run.method_runtime_metadata() == {"rsimem_method_task_id": run.case_id}
    visible = str(run.method_runtime_metadata())
    assert run.family_id not in visible
    assert "oracle" not in visible
    assert "grader" not in visible
    assert "answer" not in visible


def test_manifest_store_is_append_once_and_rejects_conflicts(tmp_path) -> None:
    manifest = _manifest()
    store = SensitivityRunManifestStore(tmp_path / "sensitivity_manifest.json")
    store.initialize(manifest)
    store.initialize(manifest)
    changed = build_sensitivity_run_manifest(
        batch_id="stage3-semantic-other",
        matrix=_matrix(),
        protocol_digest=_digest("protocol"),
        provider_id="coding.tu-zi.com/v1",
        model_id="gpt-5.6-luna",
        tool_budget=32,
        max_turns=20,
        retry_policy="bounded_retry.v1",
        deployments=_deployments(),
        rsimem_commit="commit.rsimem.fixture",
        past_bench_commit="commit.past.fixture",
    )
    with pytest.raises(ValueError, match="different manifest"):
        store.initialize(changed)
