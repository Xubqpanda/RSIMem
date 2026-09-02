from __future__ import annotations

import hashlib

import pytest

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
