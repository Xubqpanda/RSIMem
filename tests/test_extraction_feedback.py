from __future__ import annotations

import hashlib
from dataclasses import fields, replace

import pytest

from rsimem.memory.extraction_feedback import (
    ArtifactSemanticBinding,
    AttributionConfidence,
    DeploymentObservation,
    ExposureMode,
    ExtractedFactEvidence,
    ExtractionFeedbackBuilder,
    ExtractionFeedbackLabel,
    ExtractionFeedbackLevel,
    ExtractionQualityIssue,
    ExtractionSetStatus,
    ExtractionSourceEvidence,
    FactDisposition,
    FutureMemoryEvidence,
    MissedExtractionEvidence,
    ObservableToolEvent,
    default_feedback_contract_registry,
)
from rsimem.memory.extraction_projection import JsonExtractionFeedbackDatasetLog


SM01 = "SM01_preference_adoption"
SM02 = "SM02_constraint_retention"
SM05 = "SM05_weak_trigger_preference_adoption"
TSV_KEY = "preference.summary.tsv"
BOUNDARY_KEY = "constraint.share.exclude_ava_chen"
SM05_KEYS = (
    TSV_KEY,
    "preference.priority.normalized",
    "preference.date.yyyy_mm_dd",
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source(
    keys: tuple[str, ...],
    *,
    status: ExtractionSetStatus = ExtractionSetStatus.NONEMPTY,
    issue: ExtractionQualityIssue | None = None,
    dispositions: tuple[FactDisposition, ...] | None = None,
) -> ExtractionSourceEvidence:
    dispositions = dispositions or tuple(FactDisposition.PERSISTED for _ in keys)
    facts = tuple(
        ExtractedFactEvidence(
            f"fact.{index}",
            key,
            disposition,
            artifact_id=(f"artifact.{index}" if disposition == FactDisposition.PERSISTED else None),
            quality_issue=issue if index == 1 else None,
        )
        for index, (key, disposition) in enumerate(zip(keys, dispositions), start=1)
    )
    return ExtractionSourceEvidence(
        "source.fixture",
        _sha("source projection"),
        "extraction-set.fixture",
        status,
        keys,
        facts,
    )


def _future(
    keys: tuple[str, ...],
    *,
    exposure: ExposureMode = ExposureMode.EAGER_SYSTEM_PROMPT,
) -> FutureMemoryEvidence:
    bindings = tuple(
        ArtifactSemanticBinding(f"artifact.{index}", key)
        for index, key in enumerate(keys, start=1)
    )
    return FutureMemoryEvidence(
        "opportunity.fixture",
        exposure,
        bindings,
        "operation.opportunity",
        None if exposure == ExposureMode.NOT_EXPOSED else "operation.injection",
    )


def _share(*recipients: str, success: bool = True) -> ObservableToolEvent:
    return ObservableToolEvent(
        "tool.share",
        "notes_share",
        success,
        subject_ids=("note.1",),
        recipient_ids=tuple(recipients),
    )


def _observation(
    family: str,
    task_keys: tuple[str, ...],
    *,
    current_keys: tuple[str, ...] = (),
    final_response: str = "owner\tpriority\ttask\tdue_date\nA\thigh\tShip\t2026/09/01",
    recipients: tuple[str, ...] = ("owner_a",),
    completed: bool = True,
    complete: bool = True,
) -> DeploymentObservation:
    return DeploymentObservation(
        "observation.fixture",
        family,
        "eval_near",
        "task.fixture",
        _sha("current input"),
        current_keys,
        task_keys,
        final_response,
        (_share(*recipients),),
        completed,
        observation_complete=complete,
        censor_reason=None if complete else "window_incomplete",
    )


def _primary(dataset):
    return next(example for example in dataset.examples if example.primary)


def test_sm01_useful_requires_opportunity_use_and_successful_outcome() -> None:
    builder = ExtractionFeedbackBuilder(default_feedback_contract_registry())
    source = _source((TSV_KEY,))
    future = _future((TSV_KEY,))
    observation = _observation(SM01, (TSV_KEY,))

    dataset = builder.build(source, observation, future)
    assert [example.level for example in dataset.examples] == [
        ExtractionFeedbackLevel.SOURCE,
        ExtractionFeedbackLevel.EXTRACTION_SET,
        ExtractionFeedbackLevel.FACT,
    ]
    assert sum(example.primary for example in dataset.examples) == 1
    assert _primary(dataset).label == ExtractionFeedbackLabel.USEFUL
    assert dataset.examples[-1].label == ExtractionFeedbackLabel.USEFUL
    assert dataset.examples[-1].attribution_confidence == AttributionConfidence.HIGH

    no_opportunity = builder.build(
        source,
        replace(observation, task_semantic_keys=()),
        future,
    )
    assert _primary(no_opportunity).label == ExtractionFeedbackLabel.UNRESOLVED
    no_use = builder.build(
        source,
        replace(observation, final_response="Ordinary prose."),
        future,
    )
    assert _primary(no_use).label == ExtractionFeedbackLabel.UNRESOLVED
    no_outcome = builder.build(
        source,
        replace(observation, completed=False),
        future,
    )
    assert _primary(no_outcome).label == ExtractionFeedbackLabel.UNRESOLVED


def test_current_input_confounding_and_not_exposed_cannot_claim_memory_use() -> None:
    builder = ExtractionFeedbackBuilder(default_feedback_contract_registry())
    source = _source((TSV_KEY,))
    observation = _observation(SM01, (TSV_KEY,), current_keys=(TSV_KEY,))
    confounded = builder.build(source, observation, _future((TSV_KEY,)))
    assert _primary(confounded).label == ExtractionFeedbackLabel.UNRESOLVED
    assert _primary(confounded).reason_codes == ("current_input_confounded",)

    not_exposed = builder.build(
        source,
        replace(observation, current_input_semantic_keys=()),
        _future((TSV_KEY,), exposure=ExposureMode.NOT_EXPOSED),
    )
    assert _primary(not_exposed).label == ExtractionFeedbackLabel.UNRESOLVED
    assert _primary(not_exposed).reason_codes == ("use_not_bound_to_memory",)


def test_eager_injection_without_explicit_use_is_unresolved_not_harmful() -> None:
    dataset = ExtractionFeedbackBuilder(default_feedback_contract_registry()).build(
        _source((TSV_KEY,)),
        _observation(SM01, (TSV_KEY,), final_response="No table was produced."),
        _future((TSV_KEY,)),
    )
    assert _primary(dataset).label == ExtractionFeedbackLabel.UNRESOLVED
    assert _primary(dataset).reason_codes == ("injected_not_used",)
    assert dataset.examples[-1].label == ExtractionFeedbackLabel.UNRESOLVED


def test_sm05_multi_fact_success_counts_one_primary_and_no_fact_reward_copy() -> None:
    dataset = ExtractionFeedbackBuilder(default_feedback_contract_registry()).build(
        _source(SM05_KEYS),
        _observation(SM05, SM05_KEYS),
        _future(SM05_KEYS),
    )
    assert _primary(dataset).label == ExtractionFeedbackLabel.USEFUL
    assert len([example for example in dataset.examples if example.primary]) == 1
    fact_examples = [
        example for example in dataset.examples
        if example.level == ExtractionFeedbackLevel.FACT
    ]
    assert len(fact_examples) == 3
    assert {example.label for example in fact_examples} == {
        ExtractionFeedbackLabel.UNRESOLVED
    }


def test_multi_key_fact_and_shared_artifact_remain_set_level_only() -> None:
    source = ExtractionSourceEvidence(
        "source.multi-key",
        _sha("multi-key source"),
        "extraction-set.multi-key",
        ExtractionSetStatus.NONEMPTY,
        SM05_KEYS,
        (
            ExtractedFactEvidence(
                "fact.multi-key",
                SM05_KEYS,
                FactDisposition.PERSISTED,
                artifact_id="artifact.shared",
            ),
            ExtractedFactEvidence(
                "fact.second",
                (TSV_KEY,),
                FactDisposition.PERSISTED,
                artifact_id="artifact.shared",
            ),
        ),
    )
    future = FutureMemoryEvidence(
        "opportunity.fixture",
        ExposureMode.EAGER_SYSTEM_PROMPT,
        (ArtifactSemanticBinding("artifact.shared", SM05_KEYS),),
        "operation.opportunity",
        "operation.injection",
    )
    dataset = ExtractionFeedbackBuilder(default_feedback_contract_registry()).build(
        source,
        _observation(SM05, SM05_KEYS),
        future,
    )
    assert _primary(dataset).label == ExtractionFeedbackLabel.USEFUL
    fact_examples = tuple(
        example for example in dataset.examples
        if example.level == ExtractionFeedbackLevel.FACT
    )
    assert fact_examples[0].semantic_key is None
    assert {example.label for example in fact_examples} == {
        ExtractionFeedbackLabel.UNRESOLVED
    }


def test_sm02_uses_boundary_parser_and_can_attribute_harmful_share() -> None:
    builder = ExtractionFeedbackBuilder(default_feedback_contract_registry())
    source = _source((BOUNDARY_KEY,))
    future = _future((BOUNDARY_KEY,))
    positive = builder.build(
        source,
        _observation(SM02, (BOUNDARY_KEY,), final_response="No TSV required."),
        future,
    )
    assert _primary(positive).label == ExtractionFeedbackLabel.USEFUL

    harmful = builder.build(
        source,
        _observation(
            SM02,
            (BOUNDARY_KEY,),
            final_response="Still no TSV.",
            recipients=("owner_a", "ava_chen"),
        ),
        future,
    )
    assert _primary(harmful).label == ExtractionFeedbackLabel.HARMFUL
    assert _primary(harmful).reason_codes == ("memory_use_harmfully_attributed",)


def test_extraction_owned_quality_issue_is_harmful_without_blame_broadcast() -> None:
    dataset = ExtractionFeedbackBuilder(default_feedback_contract_registry()).build(
        _source((TSV_KEY,), issue=ExtractionQualityIssue.UNSUPPORTED),
        _observation(SM01, (TSV_KEY,), final_response="No explicit use."),
        _future((TSV_KEY,)),
    )
    assert _primary(dataset).label == ExtractionFeedbackLabel.HARMFUL
    assert dataset.examples[-1].label == ExtractionFeedbackLabel.HARMFUL
    assert dataset.examples[-1].reason_codes == ("extraction_unsupported",)


@pytest.mark.parametrize(
    ("status", "keys", "dispositions"),
    [
        (ExtractionSetStatus.EMPTY, (), ()),
        (ExtractionSetStatus.NONE, (), ()),
        (ExtractionSetStatus.FILTERED, (TSV_KEY,), (FactDisposition.FILTERED,)),
        (
            ExtractionSetStatus.MUTATION_FAILED,
            (TSV_KEY,),
            (FactDisposition.MUTATION_FAILED,),
        ),
    ],
)
def test_empty_filtered_none_and_failed_mutation_sources_are_retained(
    status,
    keys,
    dispositions,
) -> None:
    source = _source(keys, status=status, dispositions=dispositions)
    dataset = ExtractionFeedbackBuilder(default_feedback_contract_registry()).build(
        source,
        _observation(SM01, (TSV_KEY,), final_response="No table."),
        _future(()),
    )
    assert {example.level for example in dataset.examples} >= {
        ExtractionFeedbackLevel.SOURCE,
        ExtractionFeedbackLevel.EXTRACTION_SET,
    }
    assert _primary(dataset).label == ExtractionFeedbackLabel.UNRESOLVED


def test_missed_requires_exact_source_opportunity_and_absence_outcome_chain() -> None:
    registry = default_feedback_contract_registry()
    builder = ExtractionFeedbackBuilder(registry)
    source = ExtractionSourceEvidence(
        "source.missed",
        _sha("missed source"),
        "extraction-set.missed",
        ExtractionSetStatus.EMPTY,
        (TSV_KEY,),
        (),
    )
    observation = _observation(
        SM01,
        (TSV_KEY,),
        final_response="No table.",
        completed=False,
    )
    future = _future(())
    _, resolution = registry.resolve(observation, future)
    assert resolution.successful_outcome is False
    assert resolution.outcome_operation_id is not None
    missed = MissedExtractionEvidence(
        "missed.fixture",
        TSV_KEY,
        _sha("source span"),
        future.future_opportunity_id,
        resolution.outcome_operation_id,
        True,
    )
    dataset = builder.build(source, observation, future, missed=(missed,))
    assert _primary(dataset).label == ExtractionFeedbackLabel.MISSED

    wrong_outcome = replace(missed, absence_outcome_operation_id="outcome.wrong")
    unresolved = builder.build(source, observation, future, missed=(wrong_outcome,))
    assert _primary(unresolved).label == ExtractionFeedbackLabel.UNRESOLVED


def test_incomplete_observation_is_censored_and_unknown_family_fails_closed() -> None:
    builder = ExtractionFeedbackBuilder(default_feedback_contract_registry())
    censored = builder.build(
        _source((TSV_KEY,)),
        _observation(SM01, (TSV_KEY,), complete=False),
        _future((TSV_KEY,)),
    )
    assert _primary(censored).label == ExtractionFeedbackLabel.CENSORED

    with pytest.raises(KeyError, match="unregistered feedback family"):
        builder.build(
            _source((TSV_KEY,)),
            replace(
                _observation(SM01, (TSV_KEY,)),
                family_id="SM99_unknown_family",
            ),
            _future((TSV_KEY,)),
        )


def test_feedback_observation_api_has_no_official_evaluation_surface() -> None:
    names = {field.name for field in fields(DeploymentObservation)}
    assert not names & {"score", "grader", "answer", "expectation", "judge"}
    contracts = default_feedback_contract_registry()
    digests = {
        contracts.resolver(family).contract.contract_digest
        for family in (SM01, SM02, SM05)
    }
    parsers = {
        contracts.resolver(family).contract.use.parser_id
        for family in (SM01, SM02, SM05)
    }
    assert len(digests) == 3
    assert len(parsers) == 3


def test_feedback_dataset_log_is_idempotent_and_fails_closed(tmp_path) -> None:
    dataset = ExtractionFeedbackBuilder(default_feedback_contract_registry()).build(
        _source((TSV_KEY,)),
        _observation(SM01, (TSV_KEY,)),
        _future((TSV_KEY,)),
    )
    path = tmp_path / "feedback.jsonl"
    log = JsonExtractionFeedbackDatasetLog(path)
    assert log.append(dataset) is True
    assert JsonExtractionFeedbackDatasetLog(path).append(dataset) is False
    with pytest.raises(ValueError, match="conflicting extraction feedback"):
        log.append(replace(
            dataset,
            source_projection_digest=_sha("different source"),
        ))

    path.write_text('{"dataset_id":"partial"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="malformed extraction feedback"):
        JsonExtractionFeedbackDatasetLog(path).append(dataset)
