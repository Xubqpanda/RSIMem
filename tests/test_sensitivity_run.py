from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rsimem.memory import MemoryKind
from rsimem.memory.family_matrix import PastFamilyMatrix
from rsimem.past_sensitivity_catalog import build_past_sensitivity_catalog
from rsimem.oracle_seed_registry import create_oracle_seed_registration, create_oracle_seed_registry, oracle_seed_tree_digest
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
    apply_verified_oracle_seed_registry,
    planned_deployments_from_catalog,
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


def _deployments(matrix: SensitivityMatrix) -> tuple[SensitivityDeployment, ...]:
    return tuple(
        SensitivityDeployment(
            deployment_id=f"deployment.{case.case_id}",
            case_id=case.case_id,
            panel=case.panel,
            target_kind=case.target_kind.value,
            condition=case.condition,
            mechanism=f"deployment.{case.condition.value}",
            episode_selector=f"episode.{case.condition.value}",
            state_mode={
                SensitivityCondition.NO_PERSISTENCE: "no_persistence",
                SensitivityCondition.NATIVE_STATIC: "native_static",
                SensitivityCondition.TYPE_MATCHED_ORACLE: "oracle_seed",
                SensitivityCondition.SHORTCUT_CURRENT_INPUT: "shortcut_task",
                SensitivityCondition.WRONG_MECHANISM: "wrong_mechanism_task",
            }[case.condition],
            host_state_digest=_digest("host-state:" + case.case_id),
            launcher_config_digest=_digest("launcher-config:" + case.case_id),
            executable=case.condition in {
                SensitivityCondition.NO_PERSISTENCE,
                SensitivityCondition.NATIVE_STATIC,
            },
        )
        for case in matrix.cases
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
        deployments=_deployments(matrix),
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
        deployments=_deployments(_matrix()),
        rsimem_commit="commit.rsimem.fixture",
        past_bench_commit="commit.past.fixture",
    )
    with pytest.raises(ValueError, match="different manifest"):
        store.initialize(changed)


def test_manifest_rejects_cross_case_deployment_reuse() -> None:
    matrix = _matrix()
    deployments = list(_deployments(matrix))
    first = deployments[0]
    second = deployments[1]
    deployments[1] = SensitivityDeployment(
        deployment_id=second.deployment_id,
        case_id=first.case_id,
        panel=first.panel,
        target_kind=first.target_kind,
        condition=second.condition,
        mechanism=second.mechanism,
        episode_selector=second.episode_selector,
        state_mode=second.state_mode,
        host_state_digest=second.host_state_digest,
        launcher_config_digest=second.launcher_config_digest,
        executable=second.executable,
    )
    with pytest.raises(ValueError, match="case conditions"):
        build_sensitivity_run_manifest(
            batch_id="stage3-semantic-invalid",
            matrix=matrix,
            protocol_digest=_digest("protocol"),
            provider_id="coding.tu-zi.com/v1",
            model_id="gpt-5.6-luna",
            tool_budget=32,
            max_turns=20,
            retry_policy="bounded_retry.v1",
            deployments=deployments,
            rsimem_commit="commit.rsimem.fixture",
            past_bench_commit="commit.past.fixture",
        )


def test_catalog_planning_preserves_per_case_unavailability() -> None:
    matrix = _matrix()
    catalog = build_past_sensitivity_catalog(
        matrix=matrix,
        past_bench_root=Path(__file__).resolve().parents[1] / "benchmarks" / "past-bench",
    )
    deployments = planned_deployments_from_catalog(matrix=matrix, catalog=catalog)
    assert len(deployments) == len(matrix.cases)
    assert all(not item.executable for item in deployments)
    oracle = next(item for item in deployments if item.condition is SensitivityCondition.TYPE_MATCHED_ORACLE)
    assert oracle.episode_selector == "unavailable.type_matched_oracle"


def test_verified_oracle_registry_upgrades_only_its_case(tmp_path) -> None:
    matrix = _matrix()
    root = Path(__file__).resolve().parents[1] / "benchmarks" / "past-bench"
    catalog = build_past_sensitivity_catalog(matrix=matrix, past_bench_root=root)
    planned = planned_deployments_from_catalog(matrix=matrix, catalog=catalog)
    oracle_case = next(case for case in matrix.cases if case.condition is SensitivityCondition.TYPE_MATCHED_ORACLE)
    seed_root = tmp_path / "seeds"
    seed = seed_root / "semantic"
    (seed / "memories").mkdir(parents=True)
    (seed / "memories" / "MEMORY.md").write_text("registered fact", encoding="utf-8")
    family_digest = next(
        entry.family_source_digest
        for entry in catalog.entries
        if entry.case_id == oracle_case.case_id
    )
    registry = create_oracle_seed_registry((create_oracle_seed_registration(
        case=oracle_case,
        family_source_digest=family_digest,
        seed_home="semantic",
        seed_tree_digest=oracle_seed_tree_digest(seed),
    ),))
    resolved = apply_verified_oracle_seed_registry(
        matrix=matrix,
        catalog=catalog,
        deployments=planned,
        registry=registry,
        trusted_root=seed_root,
    )
    upgraded = next(item for item in resolved if item.case_id == oracle_case.case_id)
    assert upgraded.executable is True
    assert upgraded.episode_selector == "family.eval_only"
    assert upgraded.host_state_digest == oracle_seed_tree_digest(seed)
    assert sum(item.executable for item in resolved) == 1


def test_checked_in_sm01_registry_upgrades_only_sm01_oracle_case() -> None:
    matrix = _matrix()
    root = Path(__file__).resolve().parents[1]
    past_root = root / "benchmarks" / "past-bench"
    catalog = build_past_sensitivity_catalog(matrix=matrix, past_bench_root=past_root)
    planned = planned_deployments_from_catalog(matrix=matrix, catalog=catalog)
    from rsimem.oracle_seed_registry import OracleSeedRegistry

    resolved = apply_verified_oracle_seed_registry(
        matrix=matrix,
        catalog=catalog,
        deployments=planned,
        registry=OracleSeedRegistry.load(root / "configs/sensitivity/oracle_seed_registry_sm01.json"),
        trusted_root=past_root / "self-evolve-tasks-v2" / "_rsimem_oracles",
    )
    executable = tuple(item for item in resolved if item.executable)
    assert len(executable) == 1
    assert executable[0].condition is SensitivityCondition.TYPE_MATCHED_ORACLE
    assert executable[0].case_id == next(
        case.case_id for case in matrix.cases
        if case.family_id == "SM01_preference_adoption"
        and case.condition is SensitivityCondition.TYPE_MATCHED_ORACLE
    )
