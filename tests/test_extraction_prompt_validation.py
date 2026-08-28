from __future__ import annotations

import hashlib
from dataclasses import fields, replace

import pytest

from rsimem.memory.extraction_feedback import (
    ExtractionFeedbackLabel,
    ExtractionSetStatus,
)
from rsimem.memory.extraction_prompt_validation import (
    ExtractionAcceptanceCriteria,
    ExtractionPromptMatchedValidator,
    ExtractionPromptValidationSplit,
    ExtractionQualityMetrics,
    ExtractionValidationReplay,
    JsonExtractionValidationDecisionStore,
    ExtractionSplitAssignment,
    ExtractionValidationObservation,
    ExtractionValidationSplitRole,
    ExtractionValidationVariant,
)


PARENT = "extraction.parent-v1"
PROPOSAL = "extraction.proposal-v2"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _split() -> ExtractionPromptValidationSplit:
    return ExtractionPromptValidationSplit(
        "split.extraction-v1",
        (ExtractionSplitAssignment(
            ExtractionValidationSplitRole.VALIDATION,
            "SM01_preference_adoption",
            "SM01.report-template",
            _sha("SM01 report task manifest"),
        ),),
    )


def _observation(
    variant: ExtractionValidationVariant,
    replicate: int,
    label: ExtractionFeedbackLabel,
    *,
    status: ExtractionSetStatus = ExtractionSetStatus.NONEMPTY,
    changed_output: bool | None = None,
    failure_counts: tuple[int, int, int, int] = (0, 0, 0, 0),
    missed_assessable: bool = False,
) -> ExtractionValidationObservation:
    proposal = variant == ExtractionValidationVariant.PROPOSAL
    return ExtractionValidationObservation.create(
        variant=variant,
        replicate=replicate,
        family_id="SM01_preference_adoption",
        task_template_group_id="SM01.report-template",
        task_id=f"SM01_EVAL_{replicate:03d}",
        run_id=f"run.{variant.value}.{replicate}",
        episode_id=f"episode.{variant.value}.{replicate}",
        extraction_set_id=f"extraction-set.{variant.value}.{replicate}",
        task_manifest_digest=_sha("SM01 report task manifest"),
        model_profile_digest=_sha("model-profile"),
        budget_id="budget.validation-v1",
        persistence_state_digest=_sha(f"pre-state-{replicate}"),
        extraction_artifact_id=PROPOSAL if proposal else PARENT,
        extraction_artifact_digest=_sha(PROPOSAL if proposal else PARENT),
        extraction_output_digest=_sha(
            f"output-{replicate}-"
            f"{'proposal' if proposal and changed_output is not False else 'parent'}"
        ),
        label=label,
        extraction_status=status,
        missed_assessable=missed_assessable,
        failure_counts=failure_counts,
    )


def _criteria(**overrides) -> ExtractionAcceptanceCriteria:
    values = {
        "minimum_matched_pairs": 3,
        "minimum_resolved_examples": 2,
        "minimum_useful_rate_delta": 0.1,
        "maximum_harmful_rate_delta": 0.0,
        "minimum_coverage_ratio": 1.0,
        "maximum_empty_rate": 0.2,
        "maximum_missed_rate_delta": 0.0,
        "required_metrics": ("harmful_rate",),
        "proposal_budget_id": "proposal-budget.validation-v1",
    }
    values.update(overrides)
    return ExtractionAcceptanceCriteria(**values)


def _pairs(
    parent_labels,
    proposal_labels,
    **proposal_kwargs,
) -> tuple[ExtractionValidationObservation, ...]:
    values = []
    for replicate, (parent, proposal) in enumerate(
        zip(parent_labels, proposal_labels),
        start=1,
    ):
        values.extend((
            _observation(ExtractionValidationVariant.PARENT, replicate, parent),
            _observation(
                ExtractionValidationVariant.PROPOSAL,
                replicate,
                proposal,
                **proposal_kwargs,
            ),
        ))
    return tuple(values)


def test_prompt_validation_accepts_strict_useful_rate_improvement() -> None:
    observations = _pairs(
        (
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.HARMFUL,
            ExtractionFeedbackLabel.UNRESOLVED,
        ),
        (
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.UNRESOLVED,
        ),
    )
    decision = ExtractionPromptMatchedValidator().evaluate(
        split=_split(),
        observations=observations,
        parent_artifact_id=PARENT,
        proposal_artifact_id=PROPOSAL,
        criteria=_criteria(),
    )
    assert decision.accepted is True
    assert decision.reason_codes == ("extraction_validation_passed",)
    assert decision.parent_metrics.resolved_useful_rate == 0.5
    assert decision.proposal_metrics.resolved_useful_rate == 1.0
    assert decision.useful_rate_delta == 0.5
    assert decision.parent_metrics.unresolved_count == 1
    assert decision.proposal_metrics.unresolved_count == 1
    with pytest.raises(ValueError, match="decision ID mismatch"):
        replace(decision, changed_extraction_count=0)
    with pytest.raises(ValueError, match="ratios do not match"):
        replace(decision.parent_metrics, resolved_useful_rate=0.75)


def test_equal_quality_and_no_intervention_cannot_activate() -> None:
    observations = _pairs(
        (
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.HARMFUL,
            ExtractionFeedbackLabel.UNRESOLVED,
        ),
        (
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.HARMFUL,
            ExtractionFeedbackLabel.UNRESOLVED,
        ),
        changed_output=False,
    )
    decision = ExtractionPromptMatchedValidator().evaluate(
        split=_split(),
        observations=observations,
        parent_artifact_id=PARENT,
        proposal_artifact_id=PROPOSAL,
        criteria=_criteria(),
    )
    assert decision.accepted is False
    assert "useful_rate_not_improved" in decision.reason_codes
    assert "no_extraction_intervention" in decision.reason_codes


def test_empty_collapse_and_safety_failure_override_quality() -> None:
    observations = list(_pairs(
        (
            ExtractionFeedbackLabel.HARMFUL,
            ExtractionFeedbackLabel.HARMFUL,
            ExtractionFeedbackLabel.USEFUL,
        ),
        (
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.UNRESOLVED,
        ),
    ))
    for index, observation in enumerate(observations):
        if observation.variant == ExtractionValidationVariant.PROPOSAL:
            observations[index] = _observation(
                ExtractionValidationVariant.PROPOSAL,
                observation.replicate,
                observation.label,
                status=ExtractionSetStatus.EMPTY,
                failure_counts=(1, 0, 0, 0),
            )
    decision = ExtractionPromptMatchedValidator().evaluate(
        split=_split(),
        observations=tuple(observations),
        parent_artifact_id=PARENT,
        proposal_artifact_id=PROPOSAL,
        criteria=_criteria(),
    )
    assert decision.useful_rate_delta is not None
    assert decision.useful_rate_delta > 0
    assert decision.accepted is False
    assert "coverage_collapse" in decision.reason_codes
    assert "empty_rate_exceeded" in decision.reason_codes
    assert "safety_failure" in decision.reason_codes


def test_required_unknown_missed_metric_rejects() -> None:
    observations = _pairs(
        (
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.HARMFUL,
            ExtractionFeedbackLabel.UNRESOLVED,
        ),
        (
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.UNRESOLVED,
        ),
    )
    decision = ExtractionPromptMatchedValidator().evaluate(
        split=_split(),
        observations=observations,
        parent_artifact_id=PARENT,
        proposal_artifact_id=PROPOSAL,
        criteria=_criteria(required_metrics=("harmful_rate", "missed_rate")),
    )
    assert decision.accepted is False
    assert "missed_rate_unknown" in decision.reason_codes


def test_harmful_rate_regression_rejects_despite_useful_rate_gain() -> None:
    parent_labels = (
        ExtractionFeedbackLabel.USEFUL,
        ExtractionFeedbackLabel.HARMFUL,
        ExtractionFeedbackLabel.HARMFUL,
        *([ExtractionFeedbackLabel.UNRESOLVED] * 7),
    )
    proposal_labels = (
        ExtractionFeedbackLabel.USEFUL,
        ExtractionFeedbackLabel.USEFUL,
        ExtractionFeedbackLabel.HARMFUL,
        ExtractionFeedbackLabel.HARMFUL,
        *([ExtractionFeedbackLabel.UNRESOLVED] * 6),
    )
    observations = list(_pairs(parent_labels, proposal_labels))
    for index, observation in enumerate(observations):
        if (
            observation.variant == ExtractionValidationVariant.PROPOSAL
            and observation.replicate >= 5
        ):
            observations[index] = _observation(
                ExtractionValidationVariant.PROPOSAL,
                observation.replicate,
                observation.label,
                status=ExtractionSetStatus.NONE,
            )
    decision = ExtractionPromptMatchedValidator().evaluate(
        split=_split(),
        observations=tuple(observations),
        parent_artifact_id=PARENT,
        proposal_artifact_id=PROPOSAL,
        criteria=_criteria(
            minimum_matched_pairs=10,
            minimum_resolved_examples=4,
            minimum_coverage_ratio=0.3,
        ),
    )
    assert decision.useful_rate_delta is not None
    assert decision.useful_rate_delta > 0
    assert decision.harmful_rate_delta is not None
    assert decision.harmful_rate_delta > 0
    assert decision.accepted is False
    assert "harmful_rate_regression" in decision.reason_codes


def test_missed_rate_regression_rejects_despite_useful_rate_gain() -> None:
    observations = list(_pairs(
        (
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.HARMFUL,
            ExtractionFeedbackLabel.MISSED,
            ExtractionFeedbackLabel.UNRESOLVED,
        ),
        (
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.MISSED,
            ExtractionFeedbackLabel.MISSED,
        ),
        missed_assessable=True,
    ))
    for index, observation in enumerate(observations):
        if observation.variant == ExtractionValidationVariant.PARENT:
            observations[index] = _observation(
                ExtractionValidationVariant.PARENT,
                observation.replicate,
                observation.label,
                missed_assessable=True,
            )
    decision = ExtractionPromptMatchedValidator().evaluate(
        split=_split(),
        observations=tuple(observations),
        parent_artifact_id=PARENT,
        proposal_artifact_id=PROPOSAL,
        criteria=_criteria(
            minimum_matched_pairs=4,
            required_metrics=("harmful_rate", "missed_rate"),
        ),
    )
    assert decision.useful_rate_delta == 0.5
    assert decision.missed_rate_delta == 0.25
    assert decision.accepted is False
    assert "missed_rate_regression" in decision.reason_codes


def test_pair_identity_and_split_manifest_drift_fail_closed() -> None:
    observations = list(_pairs(
        (
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.HARMFUL,
            ExtractionFeedbackLabel.UNRESOLVED,
        ),
        (
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.UNRESOLVED,
        ),
    ))
    proposal = observations[1]
    observations[1] = _observation(
        ExtractionValidationVariant.PROPOSAL,
        proposal.replicate + 10,
        proposal.label,
    )
    with pytest.raises(ValueError, match="complete parent/proposal pairs"):
        ExtractionPromptMatchedValidator().evaluate(
            split=_split(),
            observations=tuple(observations),
            parent_artifact_id=PARENT,
            proposal_artifact_id=PROPOSAL,
            criteria=_criteria(),
        )

    with pytest.raises(ValueError, match="crosses validation split roles"):
        ExtractionPromptValidationSplit(
            "split.leakage",
            (
                ExtractionSplitAssignment(
                    ExtractionValidationSplitRole.TRAIN,
                    "SM01_preference_adoption",
                    "group.train",
                    _sha("same manifest"),
                ),
                ExtractionSplitAssignment(
                    ExtractionValidationSplitRole.FINAL,
                    "SM05_weak_trigger_preference_adoption",
                    "group.final",
                    _sha("same manifest"),
                ),
            ),
        )


def test_validation_contract_has_no_score_cost_or_fake_uncertainty_surface() -> None:
    names = {
        field.name
        for contract in (
            ExtractionValidationObservation,
            ExtractionAcceptanceCriteria,
            ExtractionQualityMetrics,
        )
        for field in fields(contract)
    }
    assert not names & {
        "score",
        "task_score",
        "cost",
        "cost_weight",
        "maximum_cost_ratio",
        "stability",
        "uncertainty",
        "grader",
        "answer",
    }
    with pytest.raises(ValueError, match="strictly positive"):
        replace(_criteria(), minimum_useful_rate_delta=0.0)


def test_validation_decision_store_and_raw_observation_replay(tmp_path) -> None:
    observations = _pairs(
        (
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.HARMFUL,
            ExtractionFeedbackLabel.UNRESOLVED,
        ),
        (
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.UNRESOLVED,
        ),
    )
    split = _split()
    criteria = _criteria()
    decision = ExtractionPromptMatchedValidator().evaluate(
        split=split,
        observations=observations,
        parent_artifact_id=PARENT,
        proposal_artifact_id=PROPOSAL,
        criteria=criteria,
    )
    store = JsonExtractionValidationDecisionStore(tmp_path / "decisions")
    path, created = store.put(decision)
    assert created is True
    assert JsonExtractionValidationDecisionStore(
        tmp_path / "decisions"
    ).put(decision) == (path, False)
    restored = JsonExtractionValidationDecisionStore(
        tmp_path / "decisions"
    ).get(decision.decision_id)
    assert restored == decision
    ExtractionValidationReplay().verify(
        restored,
        split=split,
        observations=observations,
        parent_artifact_id=PARENT,
        proposal_artifact_id=PROPOSAL,
        criteria=criteria,
    )
    serialized = path.read_text(encoding="utf-8")
    assert not any(value in serialized for value in (
        "task_score",
        "cost_weight",
        "maximum_cost_ratio",
        "uncertainty",
        "grader",
        "answer_key",
    ))

    changed = list(observations)
    changed[1] = _observation(
        ExtractionValidationVariant.PROPOSAL,
        1,
        ExtractionFeedbackLabel.HARMFUL,
    )
    with pytest.raises(ValueError, match="replay mismatch"):
        ExtractionValidationReplay().verify(
            restored,
            split=split,
            observations=tuple(changed),
            parent_artifact_id=PARENT,
            proposal_artifact_id=PROPOSAL,
            criteria=criteria,
        )

    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed extraction validation"):
        store.get(decision.decision_id)
    with pytest.raises(ValueError, match="conflicts with its ID"):
        store.put(decision)
