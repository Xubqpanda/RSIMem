from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from rsimem.memory import MemoryKind
from rsimem.memory.family_matrix import PastFamilyMatrix
from rsimem.past_sensitivity_catalog import build_past_sensitivity_catalog
from rsimem.past_sensitivity_launcher import prepare_past_sensitivity_launch
from rsimem.oracle_seed_registry import create_oracle_seed_registration, create_oracle_seed_registry, oracle_seed_tree_digest
from rsimem.research_protocol import ResearchProtocol, SensitivityCondition, default_research_protocol
from rsimem.sensitivity import SensitivityMatrix, SensitivityPanel
from rsimem.sensitivity_run import build_sensitivity_run_manifest, planned_deployments_from_catalog


ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "past-bench"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _matrix():
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


def _manifest():
    matrix = _matrix()
    catalog = build_past_sensitivity_catalog(matrix=matrix, past_bench_root=ROOT)
    deployments = planned_deployments_from_catalog(matrix=matrix, catalog=catalog)
    manifest = build_sensitivity_run_manifest(
        batch_id="launcher-fixture",
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
    return manifest


def test_prepared_native_slice_keeps_learn_eval_and_opaque_method_id(tmp_path: Path) -> None:
    manifest = _manifest()
    run = next(item for item in manifest.runs if item.condition is SensitivityCondition.NATIVE_STATIC)
    deployment = next(item for item in manifest.deployments if item.deployment_id == run.deployment_id)
    prepared = prepare_past_sensitivity_launch(
        run=run,
        deployment=replace(deployment, executable=True),
        past_bench_root=ROOT,
        output_directory=tmp_path,
    )
    document = yaml.safe_load(prepared.sequence_path.read_text(encoding="utf-8"))
    assert all(item["bucket"] in {"learn", "evaluation"} for item in document["episodes"])
    assert "--rsimem-method-task-id" in prepared.command
    assert prepared.command[prepared.command.index("--rsimem-method-task-id") + 1] == run.case_id
    assert run.family_id not in prepared.command
    assert prepared.command[prepared.command.index("--trace-dir") + 1].endswith(run.trace_directory)
    assert prepared.command[
        prepared.command.index("--rsimem-sensitivity-state-dir") + 1
    ].endswith(run.state_directory)
    assert prepared.command[
        prepared.command.index("--rsimem-sensitivity-hermes-home-dir") + 1
    ].endswith(run.hermes_home_directory)


def test_preparation_rejects_unregistered_or_nonexecutable_deployment(tmp_path: Path) -> None:
    manifest = _manifest()
    run = manifest.runs[0]
    deployment = next(item for item in manifest.deployments if item.deployment_id == run.deployment_id)
    with pytest.raises(ValueError, match="not executable"):
        prepare_past_sensitivity_launch(
            run=run,
            deployment=deployment,
            past_bench_root=ROOT,
            output_directory=tmp_path,
        )


def test_oracle_preparation_requires_verified_registry_and_copies_seed(tmp_path: Path) -> None:
    manifest = _manifest()
    run = next(item for item in manifest.runs if item.condition is SensitivityCondition.TYPE_MATCHED_ORACLE)
    deployment = next(item for item in manifest.deployments if item.deployment_id == run.deployment_id)
    with pytest.raises(ValueError, match="verified seed registry"):
        prepare_past_sensitivity_launch(
            run=run,
            deployment=replace(deployment, executable=True, episode_selector="family.eval_only"),
            past_bench_root=ROOT,
            output_directory=tmp_path,
        )
    trusted = tmp_path / "trusted"
    seed = trusted / "oracle"
    (seed / "memories").mkdir(parents=True)
    (seed / "memories" / "MEMORY.md").write_text("oracle fact", encoding="utf-8")
    oracle_case = next(item for item in _matrix().cases if item.case_id == run.case_id)
    family_file = ROOT / PastFamilyMatrix.create_default().spec_for(run.family_id).task_root / "family.yaml"
    registration = create_oracle_seed_registration(
        case=oracle_case,
        family_source_digest=hashlib.sha256(family_file.read_bytes()).hexdigest(),
        seed_home="oracle",
        seed_tree_digest=oracle_seed_tree_digest(seed),
    )
    registry = create_oracle_seed_registry((registration,))
    prepared = prepare_past_sensitivity_launch(
        run=run,
        deployment=replace(deployment, executable=True, episode_selector="family.eval_only"),
        past_bench_root=ROOT,
        output_directory=tmp_path / "prepared",
        oracle_seed_registry=registry,
        oracle_trusted_root=trusted,
        oracle_case=oracle_case,
    )
    document = yaml.safe_load(prepared.sequence_path.read_text(encoding="utf-8"))
    assert document["episodes"]
    assert all(item["bucket"] == "evaluation" for item in document["episodes"])
    assert all(item["oracle_home_seed_dir"] == "oracle-home" for item in document["episodes"])
    assert (prepared.sequence_path.parent / "oracle-home" / "memories" / "MEMORY.md").exists()
    with pytest.raises(ValueError, match="run/deployment mismatch"):
        prepare_past_sensitivity_launch(
            run=run,
            deployment=replace(deployment, deployment_id="deployment.other", executable=True),
            past_bench_root=ROOT,
            output_directory=tmp_path,
        )
