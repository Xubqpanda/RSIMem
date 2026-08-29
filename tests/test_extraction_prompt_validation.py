from __future__ import annotations

import hashlib
from dataclasses import fields, replace
from pathlib import Path

import pytest

from rsimem.memory import extraction_prompt_validation as validation_module
from rsimem.memory.extraction_feedback import (
    ArtifactSemanticBinding,
    DeploymentObservation,
    ExposureMode,
    ExtractedFactEvidence,
    ExtractionFeedbackBuilder,
    ExtractionFeedbackLabel,
    ExtractionSourceEvidence,
    ExtractionSetStatus,
    FactDisposition,
    FeedbackOperationJoin,
    FutureMemoryEvidence,
    ObservableToolEvent,
    default_feedback_contract_registry,
)
from rsimem.memory.extraction_projection import (
    ExtractionSourceRecord,
    JsonLiveExtractionFeedbackRecordLog,
    LiveExtractionFeedbackRecord,
)
from extraction_fingerprint_support import extraction_activation_fixture
from rsimem.memory.extraction_prompt_validation import (
    ExtractionAcceptanceCriteria,
    ExtractionPromptMatchedValidator,
    JsonExtractionValidationObservationStore,
    ExtractionPromptValidationSplit,
    ExtractionQualityMetrics,
    ExtractionValidationReplay,
    JsonExtractionValidationDecisionStore,
    ExtractionSplitAssignment,
    ExtractionValidationObservation,
    ExtractionValidationSafetyEvidence,
    ExtractionValidationSplitRole,
    ExtractionValidationVariant,
)
from rsimem.memory.extraction_validation_adapter import (
    ExtractionValidationObservationAssembler,
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
    missed_assessable: bool | None = None,
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
        missed_assessable=(
            True
            if label == ExtractionFeedbackLabel.MISSED
            and missed_assessable is None
            else missed_assessable
        ),
        failure_counts=failure_counts,
    )


def test_validation_observation_store_is_restart_safe_and_split_bound(tmp_path: Path) -> None:
    split = _split()
    observation = _observation(
        ExtractionValidationVariant.PARENT,
        1,
        ExtractionFeedbackLabel.USEFUL,
    )
    store = JsonExtractionValidationObservationStore(tmp_path / "observations", split=split)
    path, created = store.put(observation)
    assert created is True
    assert store.put(observation) == (path, False)
    restarted = JsonExtractionValidationObservationStore(
        tmp_path / "observations", split=split
    )
    assert restarted.get(observation.observation_id) == observation
    assert restarted.records() == (observation,)

    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed extraction validation observation"):
        restarted.get(observation.observation_id)

    outside = ExtractionValidationObservation.create(
        variant=ExtractionValidationVariant.PARENT,
        replicate=1,
        family_id="other-family",
        task_template_group_id="other-template",
        task_id="other-task",
        run_id="run.other",
        episode_id="episode.other",
        extraction_set_id="extraction-set.other",
        task_manifest_digest=_sha("other manifest"),
        model_profile_digest=_sha("model profile"),
        budget_id="budget.other",
        persistence_state_digest=_sha("state other"),
        extraction_artifact_id=PARENT,
        extraction_artifact_digest=_sha("parent body"),
        extraction_output_digest=_sha("other output"),
        label=ExtractionFeedbackLabel.UNRESOLVED,
        extraction_status=ExtractionSetStatus.EMPTY,
        missed_assessable=None,
    )
    with pytest.raises(ValueError, match="outside the frozen split"):
        restarted.put(outside)


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
        "maximum_proposal_generations": 1,
        "maximum_candidate_selections": 1,
    }
    values.update(overrides)
    return ExtractionAcceptanceCriteria(**values)


def _live_evidence():
    source = ExtractionSourceEvidence(
        "source.validation",
        _sha("source projection"),
        "extraction-set.validation",
        ExtractionSetStatus.NONEMPTY,
        ("preference.summary.tsv",),
        (ExtractedFactEvidence(
            "fact.validation",
            ("preference.summary.tsv",),
            FactDisposition.PERSISTED,
            artifact_id="memory.validation",
        ),),
    )
    source_record = ExtractionSourceRecord.create(
        family_id="SM01_preference_adoption",
        stage="learn_a",
        run_id="run.source",
        episode_id="episode.source",
        session_id="session.source",
        task_id="SM01_LEARN_A_001",
        compilation_id="compilation.validation",
        extraction_artifact_id=PARENT,
        extraction_artifact_digest=_sha(PARENT),
        extraction_output_digest=_sha("extracted TSV preference"),
        source=source,
        activation=extraction_activation_fixture(
            compilation_id="compilation.validation",
            extraction_operation_id=source.extraction_set_id,
            component_artifact_id=PARENT,
            component_artifact_digest=_sha(PARENT),
            parsed_output_digest=_sha("extracted TSV preference"),
            persisted_artifact_ids=("memory.validation",),
            mutation_ids=("mutation.validation",),
        ),
    )
    deployment = DeploymentObservation(
        "observation.validation",
        "SM01_preference_adoption",
        "eval_near",
        "SM01_EVAL_001",
        _sha("current input"),
        (),
        ("preference.summary.tsv",),
        "owner\tpriority\ttask\tdue_date\nIris\tHigh\tShip\t2026/09/01",
        (ObservableToolEvent(
            "tool.share",
            "notes_share",
            True,
            recipient_ids=("iris",),
        ),),
        True,
    )
    future = FutureMemoryEvidence(
        "opportunity.validation",
        ExposureMode.EAGER_SYSTEM_PROMPT,
        (ArtifactSemanticBinding(
            "memory.validation",
            ("preference.summary.tsv",),
        ),),
        "operation.opportunity",
        "operation.injection",
    )
    dataset = ExtractionFeedbackBuilder(
        default_feedback_contract_registry()
    ).build(
        source,
        deployment,
        future,
        operation_join=FeedbackOperationJoin(
            "operation.opportunity",
            "operation.use",
            "operation.outcome",
        ),
    )
    live = LiveExtractionFeedbackRecord.create(
        family_id="SM01_preference_adoption",
        stage="eval_near",
        run_id="run.validation.parent",
        trace_id="trace.validation.parent",
        episode_id="episode.validation.parent",
        session_id="session.validation.parent",
        task_id="SM01_EVAL_001",
        deployment_observation_id=deployment.observation_id,
        source_record_id=source_record.record_id,
        opportunity_operation_id="operation.opportunity",
        use_operation_id="operation.use",
        outcome_operation_id="operation.outcome",
        dataset=dataset,
    )
    return source_record, live


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


def test_zero_resolved_evidence_cannot_activate() -> None:
    observations = _pairs(
        (ExtractionFeedbackLabel.UNRESOLVED,) * 3,
        (ExtractionFeedbackLabel.CENSORED,) * 3,
    )
    decision = ExtractionPromptMatchedValidator().evaluate(
        split=_split(),
        observations=observations,
        parent_artifact_id=PARENT,
        proposal_artifact_id=PROPOSAL,
        criteria=_criteria(),
    )
    assert decision.accepted is False
    assert decision.parent_metrics.resolved_useful_rate is None
    assert decision.proposal_metrics.resolved_useful_rate is None
    assert "insufficient_resolved_examples" in decision.reason_codes
    assert "useful_rate_not_improved" in decision.reason_codes


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
    with pytest.raises(ValueError, match="budgets must be positive"):
        replace(_criteria(), maximum_proposal_generations=0)
    validation_source = Path(validation_module.__file__).read_text(encoding="utf-8")
    assert "extraction_projection" not in validation_source
    assert "hermes" not in validation_source.casefold()
    assert "mem0" not in validation_source.casefold()


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


def test_live_feedback_assembler_uses_persisted_identity_and_fingerprints(
    tmp_path,
) -> None:
    source, live = _live_evidence()
    safety = ExtractionValidationSafetyEvidence.create(
        live_feedback_record_id=live.record_id,
        source_record_id=source.record_id,
        audit_id="audit.validation.parent",
        audit_digest=_sha("audit payload"),
        evidence_cutoff_operation_id="operation.outcome",
        complete=True,
        schema_failure_count=0,
        safety_failure_count=0,
        prompt_leakage_failure_count=0,
        native_writer_failure_count=0,
    )
    observation = ExtractionValidationObservationAssembler().assemble(
        live_feedback=live,
        source=source,
        safety=safety,
        variant=ExtractionValidationVariant.PARENT,
        replicate=1,
        task_template_group_id="SM01.report-template",
        task_manifest_digest=_sha("SM01 report task manifest"),
        model_profile_digest=_sha("model-profile"),
        budget_id="budget.validation-v1",
        persistence_state_digest=_sha("pre-state-1"),
    )
    assert observation.run_id == live.run_id
    assert observation.episode_id == live.episode_id
    assert observation.task_id == live.task_id
    assert observation.extraction_set_id == source.source.extraction_set_id
    assert observation.extraction_artifact_id == source.extraction_artifact_id
    assert observation.extraction_output_digest == source.extraction_output_digest
    assert observation.label == ExtractionFeedbackLabel.USEFUL
    assert observation.missed_assessable is True

    restored = LiveExtractionFeedbackRecord.from_payload(live.payload())
    assert restored == live
    log = JsonLiveExtractionFeedbackRecordLog(tmp_path / "live-feedback.jsonl")
    assert log.append(live) is True
    assert JsonLiveExtractionFeedbackRecordLog(
        tmp_path / "live-feedback.jsonl"
    ).append(live) is False
    assert log.records() == (live,)
    serialized = log.path.read_text(encoding="utf-8")
    assert not any(value in serialized for value in (
        "task_score",
        "grader",
        "answer_key",
        "cost_weight",
    ))

    wrong_safety = ExtractionValidationSafetyEvidence.create(
        live_feedback_record_id="live-extraction-feedback.other",
        source_record_id=source.record_id,
        audit_id="audit.validation.wrong",
        audit_digest=_sha("wrong audit payload"),
        evidence_cutoff_operation_id="operation.outcome",
        complete=True,
        schema_failure_count=0,
        safety_failure_count=0,
        prompt_leakage_failure_count=0,
        native_writer_failure_count=0,
    )
    with pytest.raises(ValueError, match="safety evidence join mismatch"):
        ExtractionValidationObservationAssembler().assemble(
            live_feedback=live,
            source=source,
            safety=wrong_safety,
            variant=ExtractionValidationVariant.PARENT,
            replicate=1,
            task_template_group_id="SM01.report-template",
            task_manifest_digest=_sha("SM01 report task manifest"),
            model_profile_digest=_sha("model-profile"),
            budget_id="budget.validation-v1",
            persistence_state_digest=_sha("pre-state-1"),
        )

    incomplete_safety = ExtractionValidationSafetyEvidence.create(
        live_feedback_record_id=live.record_id,
        source_record_id=source.record_id,
        audit_id="audit.validation.incomplete",
        audit_digest=_sha("incomplete audit payload"),
        evidence_cutoff_operation_id="operation.outcome",
        complete=False,
        schema_failure_count=0,
        safety_failure_count=0,
        prompt_leakage_failure_count=0,
        native_writer_failure_count=0,
    )
    with pytest.raises(ValueError, match="safety audit is incomplete"):
        ExtractionValidationObservationAssembler().assemble(
            live_feedback=live,
            source=source,
            safety=incomplete_safety,
            variant=ExtractionValidationVariant.PARENT,
            replicate=1,
            task_template_group_id="SM01.report-template",
            task_manifest_digest=_sha("SM01 report task manifest"),
            model_profile_digest=_sha("model-profile"),
            budget_id="budget.validation-v1",
            persistence_state_digest=_sha("pre-state-1"),
        )

    log.path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed live extraction feedback"):
        log.records()
