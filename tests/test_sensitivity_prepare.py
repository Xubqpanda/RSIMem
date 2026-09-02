from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rsimem.research_protocol import SensitivityCondition
from rsimem.sensitivity import SensitivityPanel
from rsimem.sensitivity_prepare import prepare_registered_sensitivity_batch


@pytest.mark.parametrize("condition", tuple(SensitivityCondition))
def test_prepares_all_sm01_conditions_without_provider_execution(
    condition: SensitivityCondition, tmp_path: Path
) -> None:
    root = Path(__file__).resolve().parents[1]
    past_root = root / "benchmarks" / "past-bench"
    from rsimem.sensitivity_prepare import _matrix

    _, matrix = _matrix(SensitivityPanel.SEMANTIC)
    case_id = next(
        case.case_id for case in matrix.cases
        if case.family_id == "SM01_preference_adoption" and case.condition is condition
    )
    prepared = prepare_registered_sensitivity_batch(
        panel=SensitivityPanel.SEMANTIC,
        case_id=case_id,
        replicate=1,
        batch_id="stage3-sm01-" + condition.value,
        rsimem_root=root,
        past_bench_root=past_root,
        registry_path=root / "configs/sensitivity/oracle_seed_registry_sm01.json",
        trusted_seed_root=past_root / "self-evolve-tasks-v2" / "_rsimem_oracles",
        output_root=tmp_path / condition.value,
        past_bench_binary="past-bench",
    )
    document = yaml.safe_load(prepared.launch.sequence_path.read_text(encoding="utf-8"))
    assert prepared.manifest_path.is_file()
    assert document["episodes"]
    assert prepared.launch.command[prepared.launch.command.index("--rsimem-method-task-id") + 1] == case_id
    assert "SM01_preference_adoption" not in prepared.launch.command
    persistence = prepared.launch.command[prepared.launch.command.index("--persistence-variant") + 1]
    if condition is SensitivityCondition.NATIVE_STATIC:
        assert all(item["bucket"] in {"learn", "evaluation"} for item in document["episodes"])
        assert persistence == "with_persistence"
    elif condition is SensitivityCondition.TYPE_MATCHED_ORACLE:
        assert all(item["bucket"] == "evaluation" for item in document["episodes"])
        assert all(item["oracle_home_seed_dir"] == "oracle-home" for item in document["episodes"])
        assert (prepared.launch.sequence_path.parent / "oracle-home" / "memories" / "MEMORY.md").is_file()
        assert persistence == "with_persistence"
    else:
        assert len(document["episodes"]) == 1
        assert document["episodes"][0]["bucket"] == "control"
        assert persistence == "without_persistence"
    executable_oracles = tuple(
        item for item in prepared.manifest.deployments
        if item.executable and item.condition.value == "type_matched_oracle"
    )
    assert len(executable_oracles) == 1
    assert executable_oracles[0].case_id == next(
        case.case_id for case in matrix.cases
        if case.family_id == "SM01_preference_adoption"
        and case.condition is SensitivityCondition.TYPE_MATCHED_ORACLE
    )
