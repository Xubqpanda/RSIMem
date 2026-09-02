from __future__ import annotations

from pathlib import Path

import yaml

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


def test_catalog_exposes_real_procedural_controls_without_relabeling_legacy_ones() -> None:
    catalog = build_past_sensitivity_catalog(
        matrix=_matrix(SensitivityPanel.PROCEDURAL),
        past_bench_root=ROOT,
    )
    wrong_mechanism = [
        entry for entry in catalog.entries
        if entry.condition is SensitivityCondition.WRONG_MECHANISM
    ]
    no_persistence = [
        entry for entry in catalog.entries
        if entry.condition is SensitivityCondition.NO_PERSISTENCE
    ]
    assert len(wrong_mechanism) == len(no_persistence) == 10
    assert all(entry.available and entry.reason == "available" for entry in wrong_mechanism)
    assert all(entry.available and entry.reason == "available" for entry in no_persistence)
    assert all(entry.episode_selector and "wrong_mechanism" in entry.episode_selector for entry in wrong_mechanism)
    assert all(entry.episode_selector and "no_persistence" in entry.episode_selector for entry in no_persistence)

    selectors = {entry.episode_selector for entry in wrong_mechanism}
    assert not any("surface_memorization" in selector for selector in selectors)
    assert not any("stale_procedure" in selector for selector in selectors)
    assert not any("too_few_examples" in selector for selector in selectors)


def test_procedural_control_assets_are_task_bound_and_disable_persistence() -> None:
    family_matrix = PastFamilyMatrix.create_default()
    procedural_families = tuple(
        spec.family_id for spec in family_matrix.families if spec.target_kind is MemoryKind.PROCEDURAL
    )
    for family_id in procedural_families:
        family_dir = ROOT / family_matrix.spec_for(family_id).task_root
        document = yaml.safe_load((family_dir / "family.yaml").read_text(encoding="utf-8"))
        controls = tuple(document["episode_order"])
        wrong = [name for name in controls if "control_wrong_mechanism" in name]
        no_persistence = [name for name in controls if "control_no_persistence" in name]
        assert len(wrong) == len(no_persistence) == 1
        for name in (*wrong, *no_persistence):
            task_path = family_dir / name / "task.yaml"
            task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
            assert task["control_type"] in {"wrong_mechanism", "no_persistence"}
            assert task.get("persistence_disabled") is True
            assert (family_dir / name / "expectations.json").is_file()
        wrong_task = yaml.safe_load((family_dir / wrong[0] / "task.yaml").read_text(encoding="utf-8"))
        prompt = wrong_task["prompt"]["text"]
        assert "plain current reference text" in prompt
        assert "skill artifact" in prompt
