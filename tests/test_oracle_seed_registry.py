from __future__ import annotations

import hashlib

import pytest
from pathlib import Path

from rsimem.memory import MemoryKind
from rsimem.memory.family_matrix import PastFamilyMatrix
from rsimem.oracle_seed_registry import (
    OracleSeedRegistry,
    create_oracle_seed_registration,
    create_oracle_seed_registry,
    oracle_seed_tree_digest,
)
from rsimem.research_protocol import ResearchProtocol, SensitivityCondition, default_research_protocol
from rsimem.sensitivity import SensitivityMatrix, SensitivityPanel


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _case(panel: SensitivityPanel):
    base = default_research_protocol()
    protocol = ResearchProtocol.create(
        memory_units=base.memory_units,
        family_matrix=PastFamilyMatrix.create_default(),
        split=base.split,
        sensitivity_target_kind=MemoryKind(panel.value),
    )
    matrix = SensitivityMatrix.create_for_panel(
        panel=panel, protocol=protocol, family_matrix=PastFamilyMatrix.create_default()
    )
    return next(item for item in matrix.cases if item.condition is SensitivityCondition.TYPE_MATCHED_ORACLE)


def _seed(root, panel: SensitivityPanel):
    home = root / panel.value
    if panel is SensitivityPanel.SEMANTIC:
        (home / "memories").mkdir(parents=True)
        (home / "memories" / "MEMORY.md").write_text("fact", encoding="utf-8")
    elif panel is SensitivityPanel.EPISODIC:
        (home / "sessions").mkdir(parents=True)
        (home / "sessions" / "case.json").write_text("{}", encoding="utf-8")
        (home / "state.db").write_bytes(b"sqlite")
    else:
        (home / "skills" / "procedure").mkdir(parents=True)
        (home / "skills" / "procedure" / "SKILL.md").write_text("steps", encoding="utf-8")
    return home


@pytest.mark.parametrize("panel", tuple(SensitivityPanel))
def test_registry_resolves_only_matching_type_isolated_seed(tmp_path, panel: SensitivityPanel) -> None:
    case = _case(panel)
    root = tmp_path / "trusted"
    home = _seed(root, panel)
    registration = create_oracle_seed_registration(
        case=case,
        family_source_digest=_digest("family"),
        seed_home=home.relative_to(root).as_posix(),
        seed_tree_digest=oracle_seed_tree_digest(home),
    )
    registry = create_oracle_seed_registry((registration,))
    assert registry.for_case(case.case_id).resolve(root, case, _digest("family")) == home
    assert OracleSeedRegistry.from_payload(registry.payload()) == registry


def test_registry_rejects_layout_drift_and_digest_drift(tmp_path) -> None:
    case = _case(SensitivityPanel.SEMANTIC)
    root = tmp_path / "trusted"
    home = _seed(root, SensitivityPanel.SEMANTIC)
    registration = create_oracle_seed_registration(
        case=case,
        family_source_digest=_digest("family"),
        seed_home="semantic",
        seed_tree_digest=oracle_seed_tree_digest(home),
    )
    (home / "skills").mkdir()
    (home / "skills" / "wrong.txt").write_text("wrong-kind", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        registration.resolve(root, case, _digest("family"))
    (home / "skills" / "wrong.txt").unlink()
    (home / "skills").rmdir()
    (home / "memories" / "MEMORY.md").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        registration.resolve(root, case, _digest("family"))


def test_checked_in_sm01_registration_resolves_against_real_family_source() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = OracleSeedRegistry.load(root / "configs/sensitivity/oracle_seed_registry_sm01.json")
    base = default_research_protocol()
    protocol = ResearchProtocol.create(
        memory_units=base.memory_units,
        family_matrix=PastFamilyMatrix.create_default(),
        split=base.split,
        sensitivity_target_kind=MemoryKind.SEMANTIC,
    )
    matrix = SensitivityMatrix.create_for_panel(
        panel=SensitivityPanel.SEMANTIC,
        protocol=protocol,
        family_matrix=PastFamilyMatrix.create_default(),
    )
    case = next(
        item for item in matrix.cases
        if item.family_id == "SM01_preference_adoption"
        and item.condition is SensitivityCondition.TYPE_MATCHED_ORACLE
    )
    family_file = root / "benchmarks/past-bench" / PastFamilyMatrix.create_default().spec_for(case.family_id).task_root / "family.yaml"
    family_digest = hashlib.sha256(family_file.read_bytes()).hexdigest()
    seed_root = root / "benchmarks/past-bench/self-evolve-tasks-v2/_rsimem_oracles"
    assert registry.for_case(case.case_id).resolve(seed_root, case, family_digest).is_dir()
