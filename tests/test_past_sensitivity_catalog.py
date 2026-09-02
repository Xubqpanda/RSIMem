from __future__ import annotations

from pathlib import Path

from rsimem.memory import MemoryKind
from rsimem.memory.family_matrix import PastFamilyMatrix
from rsimem.past_sensitivity_catalog import build_past_sensitivity_catalog
from rsimem.research_protocol import ResearchProtocol, SensitivityCondition, default_research_protocol
from rsimem.sensitivity import SensitivityMatrix, SensitivityPanel


ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "past-bench"


def _matrix(panel: SensitivityPanel) -> SensitivityMatrix:
    base = default_research_protocol()
    protocol = ResearchProtocol.create(
        memory_units=base.memory_units,
        family_matrix=PastFamilyMatrix.create_default(),
        split=base.split,
        sensitivity_target_kind=MemoryKind(panel.value),
    )
    return SensitivityMatrix.create_for_panel(
        panel=panel,
        protocol=protocol,
        family_matrix=PastFamilyMatrix.create_default(),
    )


def test_catalog_uses_only_family_control_identity_and_marks_missing_oracle() -> None:
    matrix = _matrix(SensitivityPanel.SEMANTIC)
    catalog = build_past_sensitivity_catalog(matrix=matrix, past_bench_root=ROOT)
    assert len(catalog.entries) == 7 * 5
    assert not catalog.execution_ready
    oracle = [entry for entry in catalog.entries if entry.condition is SensitivityCondition.TYPE_MATCHED_ORACLE]
    assert len(oracle) == 7
    assert all(entry.reason == "type_matched_oracle_seed_missing" for entry in oracle)
    semantic_wrong = [
        entry for entry in catalog.entries
        if entry.condition is SensitivityCondition.WRONG_MECHANISM
    ]
    assert len(semantic_wrong) == 7
    assert all(entry.available for entry in semantic_wrong)
    assert all(entry.episode_selector is not None for entry in semantic_wrong)
    visible = str(catalog.payload())
    assert "judge_rubric" not in visible
    assert "expectations" not in visible
    assert "task.yaml" not in visible


def test_catalog_detects_procedural_control_gaps_without_relabeling_them() -> None:
    catalog = build_past_sensitivity_catalog(
        matrix=_matrix(SensitivityPanel.PROCEDURAL),
        past_bench_root=ROOT,
    )
    missing_wrong = [
        entry for entry in catalog.entries
        if entry.condition is SensitivityCondition.WRONG_MECHANISM
    ]
    assert len(missing_wrong) == 10
    assert all(entry.reason == "wrong_mechanism_control_missing" for entry in missing_wrong)
    assert all(entry.episode_selector is None for entry in missing_wrong)
