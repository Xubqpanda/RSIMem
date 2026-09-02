from __future__ import annotations

import pytest

from rsimem.memory import MemoryKind
from rsimem.memory.family_matrix import (
    EXPECTED_PAST_FAMILY_IDS,
    FamilyPanel,
    FamilyRole,
    PastFamilyMatrix,
    default_past_family_specs,
)


def test_default_matrix_contains_all_26_families_and_five_conditions() -> None:
    matrix = PastFamilyMatrix.create_default()
    assert len(matrix.families) == 26
    assert tuple(spec.family_id for spec in matrix.families) == EXPECTED_PAST_FAMILY_IDS
    assert {spec.panel for spec in matrix.families} == {
        FamilyPanel.SEMANTIC,
        FamilyPanel.EPISODIC,
        FamilyPanel.PROCEDURAL,
        FamilyPanel.AUXILIARY,
    }
    assert {spec.target_kind for spec in matrix.families if spec.role is FamilyRole.TARGET} == {
        MemoryKind.SEMANTIC,
        MemoryKind.EPISODIC,
        MemoryKind.PROCEDURAL,
    }
    assert len([spec for spec in matrix.families if spec.role is FamilyRole.AUXILIARY]) == 6
    assert set(matrix.conditions) == {
        "no_persistence",
        "native_static",
        "type_matched_oracle",
        "shortcut_current_input",
        "wrong_mechanism",
    }
    assert matrix.replicate_count == 3
    assert matrix.practical_improvement_threshold == 0.05


def test_method_view_excludes_family_and_filesystem_audit_identity() -> None:
    matrix = PastFamilyMatrix.create_default()
    view = matrix.method_view_for("SM01_preference_adoption")
    assert "family_id" not in view
    assert "task_root" not in view
    assert "role" not in view
    assert view["target_kind"] == "semantic"
    assert view["memory_opportunity"] == "user_preference_visible_at_future_task"

    serialized = str(view)
    assert "SM01_preference_adoption" not in serialized
    assert "self-evolve-tasks-v2" not in serialized


def test_matrix_round_trips_and_rejects_missing_family() -> None:
    matrix = PastFamilyMatrix.create_default()
    assert PastFamilyMatrix.from_payload(matrix.payload()) == matrix
    with pytest.raises(ValueError, match="not in the frozen"):
        matrix.spec_for("SM99_unknown")

    incomplete = list(default_past_family_specs())[:-1]
    with pytest.raises(ValueError, match="26 families"):
        PastFamilyMatrix(
            matrix_id="past-family-matrix.invalid",
            families=tuple(incomplete),
            conditions=matrix.conditions,
            replicate_count=3,
            practical_improvement_threshold=0.05,
            paired_procedure="paired_delta.v1",
        )


def test_family_role_contract_rejects_wrong_panel_or_target_kind() -> None:
    base = default_past_family_specs()[0]
    with pytest.raises(ValueError, match="auxiliary"):
        type(base)(
            family_id="PG99_invalid",
            panel=FamilyPanel.AUXILIARY,
            role=FamilyRole.TARGET,
            task_root="root.invalid",
            task_sequence=("observe",),
            stages=("outcome",),
            metric="metric.v1",
            memory_opportunity="opportunity.v1",
            target_kind=MemoryKind.SEMANTIC,
            confounders=("confounder.v1",),
            role_reason="invalid fixture",
        )
