from __future__ import annotations

from pathlib import Path

import pytest

from rsimem.extraction_split_plan import (
    ExtractionSplitAssignment,
    ExtractionSplitPlan,
    ExtractionSplitRole,
    load_extraction_split_plan,
    write_extraction_split_plan,
)


def _assignment(role: ExtractionSplitRole, index: int) -> ExtractionSplitAssignment:
    return ExtractionSplitAssignment(
        role,
        f"family-{index}",
        f"group-{index}",
        f"{index:064x}",
    )


def test_split_plan_round_trips_and_matches_current_batch(tmp_path) -> None:
    plan = ExtractionSplitPlan.create(tuple(
        _assignment(role, index)
        for index, role in enumerate(ExtractionSplitRole, start=1)
    ))
    path = tmp_path / "split-plan.json"
    assert write_extraction_split_plan(path, plan) is True
    assert write_extraction_split_plan(path, plan) is False
    assert load_extraction_split_plan(path) == plan
    assert plan.assignment_for(
        role=ExtractionSplitRole.VALIDATION,
        family_id="family-2",
        task_template_group_id="group-2",
        task_manifest_digest=f"{2:064x}",
    ) == next(
        value for value in plan.assignments
        if value.role is ExtractionSplitRole.VALIDATION
    )
    with pytest.raises(ValueError, match="does not match"):
        plan.assignment_for(
            role=ExtractionSplitRole.VALIDATION,
            family_id="family-wrong",
            task_template_group_id="group-2",
            task_manifest_digest=f"{2:064x}",
        )


def test_split_plan_rejects_cross_role_manifest_and_missing_roles() -> None:
    digest = "a" * 64
    with pytest.raises(ValueError, match="cross split roles"):
        ExtractionSplitPlan.create((
            ExtractionSplitAssignment(ExtractionSplitRole.TRAIN, "f1", "g1", digest),
            ExtractionSplitAssignment(ExtractionSplitRole.VALIDATION, "f2", "g2", digest),
            _assignment(ExtractionSplitRole.FINAL_TEST, 3),
        ))
    with pytest.raises(ValueError, match="train, validation"):
        ExtractionSplitPlan.create((_assignment(ExtractionSplitRole.TRAIN, 1),))
    with pytest.raises(ValueError, match="train, validation"):
        ExtractionSplitPlan.create((
            _assignment(ExtractionSplitRole.TRAIN, 1),
            _assignment(ExtractionSplitRole.TRAIN, 2),
            _assignment(ExtractionSplitRole.FINAL_TEST, 3),
        ))


def test_split_plan_rejects_unstable_identifiers() -> None:
    with pytest.raises(ValueError, match="stable identifier"):
        ExtractionSplitAssignment(
            ExtractionSplitRole.TRAIN,
            "family/with-slash",
            "group-1",
            "a" * 64,
        )


def test_checked_in_heldout_plan_matches_current_vendored_families() -> None:
    root = Path(__file__).resolve().parents[1]
    plan = load_extraction_split_plan(
        root / "configs/extraction_split_plan_sm02_sm03_sm04.json"
    )
    assert plan.assignment_for(
        role=ExtractionSplitRole.TRAIN,
        family_id="SM02_constraint_retention",
        task_template_group_id="sm02-process-pilot-train-v1",
        task_manifest_digest="698882eefe9817b1827517ea921c686f08cfd76c4a90ab70a29cabe9dda4884d",
    )
    assert plan.assignment_for(
        role=ExtractionSplitRole.VALIDATION,
        family_id="SM03_fact_correction",
        task_template_group_id="sm03-correction-heldout-validation-v1",
        task_manifest_digest="cece4019357f08d7bde746e012683542a699e61756ede76c91d4f6641dced54c",
    )
    assert plan.assignment_for(
        role=ExtractionSplitRole.FINAL_TEST,
        family_id="SM04_rule_migration",
        task_template_group_id="sm04-migration-heldout-final-v1",
        task_manifest_digest="0022269fd7e8e033c0b9c50fd65930ebacc1bb3483734c4e2251a685a44470e0",
    )
    with pytest.raises(ValueError, match="does not match split plan"):
        plan.assignment_for(
            role=ExtractionSplitRole.TRAIN,
            family_id="SM03_fact_correction",
            task_template_group_id="sm03-correction-heldout-validation-v1",
            task_manifest_digest="cece4019357f08d7bde746e012683542a699e61756ede76c91d4f6641dced54c",
        )
