from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rsimem.lifecycle import RawResourceUsage
from rsimem.memory.extraction_feedback import (
    ExtractionFeedbackLabel,
    ExtractionSetStatus,
)
from rsimem.memory.extraction_offline_validation import (
    CapturedDeterministicExtractionExecutor,
    DeterministicExtractionCase,
    DeterministicExtractionCategory,
    DeterministicExtractionExpectation,
    DeterministicExtractionSuiteRunner,
    DeterministicSourceMessage,
    ExtractionCandidateStaticValidator,
    ExtractionOfflineDecisionStatus,
    ExtractionPromptOfflineValidator,
    OfflineMetricName,
)
from rsimem.memory.extraction_policy_artifact import (
    ExtractionGenerationProvenance,
    ExtractionPolicyRule,
    ExtractionPolicySpec,
    ExtractionPromptPolicyArtifact,
    ExtractionRuleEdit,
    ExtractionRuleEditAction,
)
from rsimem.memory.extraction_prompt_validation import (
    ExtractionAcceptanceCriteria,
    ExtractionPromptMatchedValidator,
    ExtractionPromptValidationSplit,
    ExtractionSplitAssignment,
    ExtractionValidationObservation,
    ExtractionValidationSplitRole,
    ExtractionValidationVariant,
)
from rsimem.memory.prompt_components import text_digest
from rsimem.memory_systems.mem0_flat import (
    MEM0_FLAT_EXTRACTION_MAX_BODY_CHARS,
    MEM0_FLAT_EXTRACTION_SLOT,
    MEM0_FLAT_EXTRACTION_SLOT_ID,
    Mem0FlatPromptAdapter,
)


def _provenance() -> ExtractionGenerationProvenance:
    return ExtractionGenerationProvenance(
        "gpt-5.6-luna",
        "1" * 64,
        "optimizer-corpus.train-v1",
        "cutoff-v1",
        "2" * 64,
        "3" * 64,
        RawResourceUsage(input_tokens=100, output_tokens=20, model_requests=1),
    )


def _parent() -> ExtractionPromptPolicyArtifact:
    return Mem0FlatPromptAdapter().export_root_policy_artifact(
        MEM0_FLAT_EXTRACTION_SLOT_ID
    )


def _candidate(
    *,
    parent: ExtractionPromptPolicyArtifact | None = None,
    text: str = (
        "Keep durable user preferences, constraints, and rules that are likely "
        "to remain useful in later tasks."
    ),
) -> ExtractionPromptPolicyArtifact:
    parent = parent or _parent()
    return ExtractionPromptPolicyArtifact.create_child(
        parent=parent,
        policy_version=f"candidate.{text_digest(text)[:16]}",
        edits=(ExtractionRuleEdit(
            "edit.refine-future-scope",
            ExtractionRuleEditAction.REPLACE,
            "future-useful-scope",
            ExtractionPolicyRule("future-useful-scope", text),
        ),),
        generation_provenance=_provenance(),
    )


def _cases() -> tuple[DeterministicExtractionCase, ...]:
    values = (
        (
            "case.durable-preference",
            DeterministicExtractionCategory.DURABLE_PREFERENCE,
            (DeterministicSourceMessage(
                "user",
                "I prefer concise status summaries in future tasks.",
            ),),
        ),
        (
            "case.durable-constraint",
            DeterministicExtractionCategory.DURABLE_CONSTRAINT,
            (DeterministicSourceMessage(
                "user",
                "Never share internal incident notes outside the response team.",
            ),),
        ),
        (
            "case.temporary-request",
            DeterministicExtractionCategory.TEMPORARY_REQUEST,
            (DeterministicSourceMessage(
                "user",
                "For this reply only, write the heading in uppercase.",
            ),),
        ),
        (
            "case.unresolved-claim",
            DeterministicExtractionCategory.UNRESOLVED_CLAIM,
            (DeterministicSourceMessage(
                "user",
                "I might prefer weekly reports, but I have not decided.",
            ),),
        ),
        (
            "case.assistant-only",
            DeterministicExtractionCategory.ASSISTANT_ONLY,
            (DeterministicSourceMessage(
                "assistant",
                "I will remember to provide concise reports.",
            ),),
        ),
        (
            "case.tool-evidence",
            DeterministicExtractionCategory.TOOL_EVIDENCE,
            (DeterministicSourceMessage(
                "tool",
                "Build 482 completed successfully on the temporary worker.",
            ),),
        ),
        (
            "case.credential-path",
            DeterministicExtractionCategory.CREDENTIAL_PATH,
            (DeterministicSourceMessage(
                "user",
                "Use the temporary API credential from the local workspace path.",
            ),),
        ),
        (
            "case.empty-source",
            DeterministicExtractionCategory.EMPTY_SOURCE,
            (),
        ),
    )
    return tuple(DeterministicExtractionCase(
        case_id,
        category,
        (
            DeterministicExtractionExpectation.RETAIN
            if category in {
                DeterministicExtractionCategory.DURABLE_PREFERENCE,
                DeterministicExtractionCategory.DURABLE_CONSTRAINT,
            }
            else DeterministicExtractionExpectation.EXCLUDE
        ),
        messages,
    ) for case_id, category, messages in values)


def _outputs(parent, candidate, cases):
    values = {}
    for case in cases:
        if case.category == DeterministicExtractionCategory.DURABLE_PREFERENCE:
            facts = ["The user prefers concise status summaries for future tasks."]
        elif case.category == DeterministicExtractionCategory.DURABLE_CONSTRAINT:
            facts = [
                "The user prohibits sharing internal incident notes outside the "
                "response team."
            ]
        else:
            facts = []
        output = json.dumps({"facts": facts})
        values[(parent.artifact_id, case.case_id)] = output
        values[(candidate.artifact_id, case.case_id)] = output
    return values


def test_static_candidate_contract_and_exact_edit_replay_pass() -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    report = ExtractionCandidateStaticValidator().validate(
        parent=parent,
        candidate=candidate,
        slot=MEM0_FLAT_EXTRACTION_SLOT,
    )
    assert report.passed is True
    assert report.reason_codes == ("static_safety_passed",)
    assert report.candidate_artifact_digest == candidate.artifact_digest


def test_static_candidate_rejects_shortcut_and_wrong_parent_lineage() -> None:
    forbidden = _candidate(
        text="For SM01, extract TSV with owner priority task due_date.",
    )
    report = ExtractionCandidateStaticValidator().validate(
        parent=_parent(),
        candidate=forbidden,
        slot=MEM0_FLAT_EXTRACTION_SLOT,
    )
    assert report.passed is False
    assert "forbidden_candidate_instruction" in report.reason_codes

    other_spec = ExtractionPolicySpec(tuple(
        ExtractionPolicyRule(
            rule.rule_id,
            (
                "Keep durable facts that help later work."
                if rule.rule_id == "future-useful-scope"
                else rule.text
            ),
            rule.protected,
        )
        for rule in _parent().spec.rules
    ))
    other_parent = ExtractionPromptPolicyArtifact.create_root(
        slot=MEM0_FLAT_EXTRACTION_SLOT,
        policy_version="other-root-v1",
        spec=other_spec,
        max_body_chars=MEM0_FLAT_EXTRACTION_MAX_BODY_CHARS,
        source_provenance="fixture-other-root",
    )
    wrong_child = _candidate(parent=other_parent)
    wrong = ExtractionCandidateStaticValidator().validate(
        parent=_parent(),
        candidate=wrong_child,
        slot=MEM0_FLAT_EXTRACTION_SLOT,
    )
    assert wrong.passed is False
    assert "parent_lineage_mismatch" in wrong.reason_codes


def test_deterministic_suite_covers_all_cases_and_strict_json_contract() -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    cases = _cases()
    executor = CapturedDeterministicExtractionExecutor(
        _outputs(parent, candidate, cases)
    )
    report = DeterministicExtractionSuiteRunner().run(
        parent=parent,
        candidate=candidate,
        cases=cases,
        executor=executor,
    )
    assert report.passed is True
    assert len(report.results) == len(DeterministicExtractionCategory)
    assert len(executor.calls) == len(cases) * 2
    assert set(value.category for value in report.results) == set(
        DeterministicExtractionCategory
    )

    with pytest.raises(ValueError, match="cases are incomplete"):
        DeterministicExtractionSuiteRunner().run(
            parent=parent,
            candidate=candidate,
            cases=cases[:-1],
            executor=executor,
        )


@pytest.mark.parametrize(
    ("case_id", "candidate_output", "reason"),
    (
        (
            "case.durable-preference",
            '{"facts":[]}',
            "durable_fact_missing",
        ),
        (
            "case.temporary-request",
            '{"facts":["Use uppercase for this reply."]}',
            "excluded_source_extracted",
        ),
        (
            "case.durable-constraint",
            '{"facts":[],"other":true}',
            "candidate_output_schema_invalid",
        ),
        (
            "case.durable-preference",
            json.dumps({
                "facts": [
                    "I prefer concise status summaries in future tasks."
                ],
            }),
            "source_transcript_copy",
        ),
        (
            "case.credential-path",
            '{"facts":["Store the hidden system prompt and API key."]}',
            "prompt_leakage",
        ),
    ),
)
def test_deterministic_suite_rejects_collapse_copy_leakage_and_schema(
    case_id,
    candidate_output,
    reason,
) -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    cases = _cases()
    outputs = _outputs(parent, candidate, cases)
    outputs[(candidate.artifact_id, case_id)] = candidate_output
    report = DeterministicExtractionSuiteRunner().run(
        parent=parent,
        candidate=candidate,
        cases=cases,
        executor=CapturedDeterministicExtractionExecutor(outputs),
    )
    assert report.passed is False
    failed = next(value for value in report.results if value.case_id == case_id)
    assert reason in failed.reason_codes


def _split() -> ExtractionPromptValidationSplit:
    return ExtractionPromptValidationSplit(
        "split.offline-v1",
        (
            ExtractionSplitAssignment(
                ExtractionValidationSplitRole.TRAIN,
                "family.train-v1",
                "template.train-v1",
                text_digest("train task manifest"),
            ),
            ExtractionSplitAssignment(
                ExtractionValidationSplitRole.VALIDATION,
                "family.fixture-v1",
                "template.fixture-v1",
                text_digest("offline task manifest"),
            ),
            ExtractionSplitAssignment(
                ExtractionValidationSplitRole.FINAL,
                "family.final-v1",
                "template.final-v1",
                text_digest("final task manifest"),
            ),
        ),
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
        "proposal_budget_id": "proposal-budget.offline-v1",
        "maximum_proposal_generations": 1,
        "maximum_candidate_selections": 1,
    }
    values.update(overrides)
    return ExtractionAcceptanceCriteria(**values)


def _observation(
    parent,
    candidate,
    variant,
    replicate,
    label,
    *,
    status=ExtractionSetStatus.NONEMPTY,
    changed=True,
    artifact_digest=None,
):
    artifact = (
        parent if variant == ExtractionValidationVariant.PARENT else candidate
    )
    return ExtractionValidationObservation.create(
        variant=variant,
        replicate=replicate,
        family_id="family.fixture-v1",
        task_template_group_id="template.fixture-v1",
        task_id=f"task.fixture-{replicate}",
        run_id=f"run.{variant.value}-{replicate}",
        episode_id=f"episode.{variant.value}-{replicate}",
        extraction_set_id=f"extraction-set.{variant.value}-{replicate}",
        task_manifest_digest=text_digest("offline task manifest"),
        model_profile_digest=text_digest("offline model profile"),
        budget_id="budget.offline-v1",
        persistence_state_digest=text_digest(f"state-{replicate}"),
        extraction_artifact_id=artifact.artifact_id,
        extraction_artifact_digest=artifact_digest or artifact.body_digest,
        extraction_output_digest=text_digest(
            f"output-{replicate}-"
            f"{'candidate' if variant == ExtractionValidationVariant.PROPOSAL and changed else 'parent'}"
        ),
        label=label,
        extraction_status=status,
        missed_assessable=None,
    )


def _pairs(parent, candidate, parent_labels, candidate_labels, **candidate_kwargs):
    values = []
    for replicate, (parent_label, candidate_label) in enumerate(
        zip(parent_labels, candidate_labels),
        start=1,
    ):
        values.extend((
            _observation(
                parent,
                candidate,
                ExtractionValidationVariant.PARENT,
                replicate,
                parent_label,
            ),
            _observation(
                parent,
                candidate,
                ExtractionValidationVariant.PROPOSAL,
                replicate,
                candidate_label,
                **candidate_kwargs,
            ),
        ))
    return tuple(values)


def _safety_and_suite(parent, candidate):
    safety = ExtractionCandidateStaticValidator().validate(
        parent=parent,
        candidate=candidate,
        slot=MEM0_FLAT_EXTRACTION_SLOT,
    )
    cases = _cases()
    suite = DeterministicExtractionSuiteRunner().run(
        parent=parent,
        candidate=candidate,
        cases=cases,
        executor=CapturedDeterministicExtractionExecutor(
            _outputs(parent, candidate, cases)
        ),
    )
    return safety, suite


def test_offline_strict_improvement_is_only_accepted_for_matched_trial() -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    safety, suite = _safety_and_suite(parent, candidate)
    decision = ExtractionPromptOfflineValidator().evaluate(
        parent=parent,
        candidate=candidate,
        split=_split(),
        observations=_pairs(
            parent,
            candidate,
            (
                ExtractionFeedbackLabel.USEFUL,
                ExtractionFeedbackLabel.HARMFUL,
                ExtractionFeedbackLabel.USEFUL,
            ),
            (
                ExtractionFeedbackLabel.USEFUL,
                ExtractionFeedbackLabel.USEFUL,
                ExtractionFeedbackLabel.USEFUL,
            ),
        ),
        criteria=_criteria(),
        static_safety=safety,
        deterministic_suite=suite,
    )

    assert decision.status == (
        ExtractionOfflineDecisionStatus.ACCEPTED_FOR_MATCHED_TRIAL
    )
    assert decision.eligible_next_stage == "matched_trial"
    assert decision.reason_codes == ("offline_validation_passed",)
    assert not hasattr(decision, "active")
    ratios = {value.metric: value for value in decision.candidate_ratios}
    assert ratios[OfflineMetricName.RESOLVED_USEFUL_RATE].payload() == {
        "metric": "resolved_useful_rate",
        "numerator": 3,
        "denominator": 3,
        "unknown_count": 0,
        "value": 1.0,
    }
    assert ratios[OfflineMetricName.OBSERVED_HARMFUL_RATE].numerator == 0
    assert ratios[OfflineMetricName.NONEMPTY_COVERAGE].denominator == 3
    assert ratios[OfflineMetricName.EMPTY_EXTRACTION_RATE].numerator == 0
    assert ratios[OfflineMetricName.HIGH_CONFIDENCE_MISSED_RATE].value is None
    assert ratios[OfflineMetricName.HIGH_CONFIDENCE_MISSED_RATE].unknown_count == 3


def test_offline_equal_quality_and_empty_collapse_are_rejected() -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    safety, suite = _safety_and_suite(parent, candidate)
    labels = (
        ExtractionFeedbackLabel.USEFUL,
        ExtractionFeedbackLabel.HARMFUL,
        ExtractionFeedbackLabel.UNRESOLVED,
    )
    equal = ExtractionPromptOfflineValidator().evaluate(
        parent=parent,
        candidate=candidate,
        split=_split(),
        observations=_pairs(parent, candidate, labels, labels, changed=False),
        criteria=_criteria(),
        static_safety=safety,
        deterministic_suite=suite,
    )
    assert equal.status == ExtractionOfflineDecisionStatus.REJECTED
    assert "useful_rate_not_improved" in equal.reason_codes
    assert "no_extraction_intervention" in equal.reason_codes

    collapse = ExtractionPromptOfflineValidator().evaluate(
        parent=parent,
        candidate=candidate,
        split=_split(),
        observations=_pairs(
            parent,
            candidate,
            (
                ExtractionFeedbackLabel.HARMFUL,
                ExtractionFeedbackLabel.HARMFUL,
                ExtractionFeedbackLabel.USEFUL,
            ),
            (
                ExtractionFeedbackLabel.USEFUL,
                ExtractionFeedbackLabel.USEFUL,
                ExtractionFeedbackLabel.USEFUL,
            ),
            status=ExtractionSetStatus.EMPTY,
        ),
        criteria=_criteria(),
        static_safety=safety,
        deterministic_suite=suite,
    )
    assert collapse.status == ExtractionOfflineDecisionStatus.REJECTED
    assert "coverage_collapse" in collapse.reason_codes
    assert "empty_rate_exceeded" in collapse.reason_codes


def test_single_useful_fact_cannot_hide_coverage_collapse() -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    safety, suite = _safety_and_suite(parent, candidate)
    parent_labels = (
        ExtractionFeedbackLabel.HARMFUL,
        ExtractionFeedbackLabel.HARMFUL,
        ExtractionFeedbackLabel.HARMFUL,
        ExtractionFeedbackLabel.USEFUL,
    )
    candidate_labels = (
        ExtractionFeedbackLabel.USEFUL,
        ExtractionFeedbackLabel.UNRESOLVED,
        ExtractionFeedbackLabel.UNRESOLVED,
        ExtractionFeedbackLabel.UNRESOLVED,
    )
    observations = list(_pairs(
        parent,
        candidate,
        parent_labels,
        candidate_labels,
    ))
    for index, value in enumerate(observations):
        if (
            value.variant == ExtractionValidationVariant.PROPOSAL
            and value.replicate > 1
        ):
            observations[index] = _observation(
                parent,
                candidate,
                ExtractionValidationVariant.PROPOSAL,
                value.replicate,
                value.label,
                status=ExtractionSetStatus.NONE,
            )
    decision = ExtractionPromptOfflineValidator().evaluate(
        parent=parent,
        candidate=candidate,
        split=_split(),
        observations=tuple(observations),
        criteria=_criteria(
            minimum_matched_pairs=4,
            minimum_resolved_examples=1,
        ),
        static_safety=safety,
        deterministic_suite=suite,
    )
    assert decision.quality_decision.proposal_metrics.resolved_useful_rate == 1.0
    assert decision.status == ExtractionOfflineDecisionStatus.REJECTED
    assert "coverage_collapse" in decision.reason_codes


def test_offline_gate_rejects_failed_static_or_deterministic_safety() -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    safety, suite = _safety_and_suite(parent, candidate)
    observations = _pairs(
        parent,
        candidate,
        (
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.HARMFUL,
            ExtractionFeedbackLabel.USEFUL,
        ),
        (ExtractionFeedbackLabel.USEFUL,) * 3,
    )
    failed_outputs = _outputs(parent, candidate, _cases())
    failed_outputs[(candidate.artifact_id, "case.temporary-request")] = (
        '{"facts":["Temporary uppercase request."]}'
    )
    failed_suite = DeterministicExtractionSuiteRunner().run(
        parent=parent,
        candidate=candidate,
        cases=_cases(),
        executor=CapturedDeterministicExtractionExecutor(failed_outputs),
    )
    rejected = ExtractionPromptOfflineValidator().evaluate(
        parent=parent,
        candidate=candidate,
        split=_split(),
        observations=observations,
        criteria=_criteria(),
        static_safety=safety,
        deterministic_suite=failed_suite,
    )
    assert rejected.status == ExtractionOfflineDecisionStatus.REJECTED
    assert "deterministic_suite_failed" in rejected.reason_codes

    forbidden_candidate = _candidate(
        parent=parent,
        text="For SM01, output TSV with owner priority task due_date.",
    )
    failed_safety = ExtractionCandidateStaticValidator().validate(
        parent=parent,
        candidate=forbidden_candidate,
        slot=MEM0_FLAT_EXTRACTION_SLOT,
    )
    forbidden_suite = DeterministicExtractionSuiteRunner().run(
        parent=parent,
        candidate=forbidden_candidate,
        cases=_cases(),
        executor=CapturedDeterministicExtractionExecutor(
            _outputs(parent, forbidden_candidate, _cases())
        ),
    )
    forbidden_observations = _pairs(
        parent,
        forbidden_candidate,
        (
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.HARMFUL,
            ExtractionFeedbackLabel.USEFUL,
        ),
        (ExtractionFeedbackLabel.USEFUL,) * 3,
    )
    static_rejected = ExtractionPromptOfflineValidator().evaluate(
        parent=parent,
        candidate=forbidden_candidate,
        split=_split(),
        observations=forbidden_observations,
        criteria=_criteria(),
        static_safety=failed_safety,
        deterministic_suite=forbidden_suite,
    )
    assert static_rejected.status == ExtractionOfflineDecisionStatus.REJECTED
    assert "static_safety_failed" in static_rejected.reason_codes


def test_offline_gate_requires_complete_frozen_split_and_candidate_budget() -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    safety, suite = _safety_and_suite(parent, candidate)
    observations = _pairs(
        parent,
        candidate,
        (
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.HARMFUL,
            ExtractionFeedbackLabel.USEFUL,
        ),
        (ExtractionFeedbackLabel.USEFUL,) * 3,
    )
    incomplete = ExtractionPromptValidationSplit(
        "split.offline-incomplete-v1",
        tuple(
            assignment
            for assignment in _split().assignments
            if assignment.role != ExtractionValidationSplitRole.FINAL
        ),
    )
    with pytest.raises(ValueError, match="split roles are incomplete"):
        ExtractionPromptOfflineValidator().evaluate(
            parent=parent,
            candidate=candidate,
            split=incomplete,
            observations=observations,
            criteria=_criteria(),
            static_safety=safety,
            deterministic_suite=suite,
        )

    for budget_field in (
        "maximum_proposal_generations",
        "maximum_candidate_selections",
    ):
        with pytest.raises(ValueError, match="one frozen candidate"):
            ExtractionPromptOfflineValidator().evaluate(
                parent=parent,
                candidate=candidate,
                split=_split(),
                observations=observations,
                criteria=_criteria(**{budget_field: 2}),
                static_safety=safety,
                deterministic_suite=suite,
            )


def test_offline_gate_rejects_observation_body_digest_drift() -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    safety, suite = _safety_and_suite(parent, candidate)
    observations = list(_pairs(
        parent,
        candidate,
        (
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.HARMFUL,
            ExtractionFeedbackLabel.USEFUL,
        ),
        (ExtractionFeedbackLabel.USEFUL,) * 3,
    ))
    observations[1] = _observation(
        parent,
        candidate,
        ExtractionValidationVariant.PROPOSAL,
        1,
        ExtractionFeedbackLabel.USEFUL,
        artifact_digest=text_digest("different candidate body"),
    )

    with pytest.raises(ValueError, match="observation body digest mismatch"):
        ExtractionPromptOfflineValidator().evaluate(
            parent=parent,
            candidate=candidate,
            split=_split(),
            observations=tuple(observations),
            criteria=_criteria(),
            static_safety=safety,
            deterministic_suite=suite,
        )


def test_offline_gate_rejects_static_and_suite_report_identity_mismatch() -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    other_candidate = _candidate(
        parent=parent,
        text=(
            "Keep stable user preferences and constraints when they are likely "
            "to help with future work."
        ),
    )
    safety, suite = _safety_and_suite(parent, candidate)
    other_safety, other_suite = _safety_and_suite(parent, other_candidate)
    observations = _pairs(
        parent,
        candidate,
        (
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.HARMFUL,
            ExtractionFeedbackLabel.USEFUL,
        ),
        (ExtractionFeedbackLabel.USEFUL,) * 3,
    )
    arguments = {
        "parent": parent,
        "candidate": candidate,
        "split": _split(),
        "observations": observations,
        "criteria": _criteria(),
    }

    with pytest.raises(ValueError, match="static safety report join mismatch"):
        ExtractionPromptOfflineValidator().evaluate(
            **arguments,
            static_safety=other_safety,
            deterministic_suite=suite,
        )
    with pytest.raises(ValueError, match="deterministic suite report join mismatch"):
        ExtractionPromptOfflineValidator().evaluate(
            **arguments,
            static_safety=safety,
            deterministic_suite=other_suite,
        )


def test_offline_decision_rejects_detached_quality_and_ratio_evidence() -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    safety, suite = _safety_and_suite(parent, candidate)
    observations = _pairs(
        parent,
        candidate,
        (
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.HARMFUL,
            ExtractionFeedbackLabel.USEFUL,
        ),
        (ExtractionFeedbackLabel.USEFUL,) * 3,
    )
    criteria = _criteria()
    decision = ExtractionPromptOfflineValidator().evaluate(
        parent=parent,
        candidate=candidate,
        split=_split(),
        observations=observations,
        criteria=criteria,
        static_safety=safety,
        deterministic_suite=suite,
    )

    with pytest.raises(ValueError, match="ratio evidence mismatch"):
        replace(decision, candidate_ratios=decision.parent_ratios)

    other_split = ExtractionPromptValidationSplit(
        "split.offline-other-v1",
        _split().assignments,
    )
    other_quality = ExtractionPromptMatchedValidator().evaluate(
        split=other_split,
        observations=observations,
        parent_artifact_id=parent.artifact_id,
        proposal_artifact_id=candidate.artifact_id,
        criteria=criteria,
    )
    with pytest.raises(ValueError, match="quality decision join mismatch"):
        replace(decision, quality_decision=other_quality)
