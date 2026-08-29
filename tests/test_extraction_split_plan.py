from __future__ import annotations

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
