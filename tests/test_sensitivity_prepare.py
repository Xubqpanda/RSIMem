from __future__ import annotations

from pathlib import Path

import yaml

from rsimem.sensitivity import SensitivityPanel
from rsimem.sensitivity_prepare import prepare_registered_sensitivity_batch


def test_prepares_checked_in_sm01_oracle_without_provider_execution(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    past_root = root / "benchmarks" / "past-bench"
    case_id = "sensitivity-case.6afbbc1f125a1107a7f37550e78cdb5300235fb1"
    prepared = prepare_registered_sensitivity_batch(
        panel=SensitivityPanel.SEMANTIC,
        case_id=case_id,
        replicate=1,
        batch_id="stage3-sm01-oracle-preparation",
        rsimem_root=root,
        past_bench_root=past_root,
        registry_path=root / "configs/sensitivity/oracle_seed_registry_sm01.json",
        trusted_seed_root=past_root / "self-evolve-tasks-v2" / "_rsimem_oracles",
        output_root=tmp_path,
        past_bench_binary="past-bench",
    )
    document = yaml.safe_load(prepared.launch.sequence_path.read_text(encoding="utf-8"))
    assert prepared.manifest_path.is_file()
    assert document["episodes"]
    assert all(item["bucket"] == "evaluation" for item in document["episodes"])
    assert all(item["oracle_home_seed_dir"] == "oracle-home" for item in document["episodes"])
    assert (prepared.launch.sequence_path.parent / "oracle-home" / "memories" / "MEMORY.md").is_file()
    assert prepared.launch.command[prepared.launch.command.index("--rsimem-method-task-id") + 1] == case_id
    assert "SM01_preference_adoption" not in prepared.launch.command
    deployments = tuple(item for item in prepared.manifest.deployments if item.executable)
    assert len(deployments) == 1
    assert deployments[0].case_id == case_id
