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
    FeedbackOperationJoin,
    ExtractionQualityIssue,
    ExtractionSetStatus,
    ExtractionSourceEvidence,
    FactDisposition,
    FutureMemoryEvidence,
    MissedExtractionEvidence,
    ObservableToolEvent,
    default_feedback_contract_registry,
    detect_user_source_semantic_keys,
)
from rsimem.memory.extraction_projection import JsonExtractionFeedbackDatasetLog
from rsimem.memory.evidence_planes import EvidencePlane, EvidenceSourceKind


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


def test_family_bound_feedback_cannot_be_relabelled_as_pure_process() -> None:
    dataset = ExtractionFeedbackBuilder(
        default_feedback_contract_registry()
    ).build(
        _source((TSV_KEY,)),
        _observation(SM01, (TSV_KEY,)),
        _future((TSV_KEY,)),
    )
    assert dataset.evidence_plane is EvidencePlane.BENCHMARK_AUDIT
    assert dataset.evidence_source is EvidenceSourceKind.BENCHMARK_CONTRACT
    with pytest.raises(ValueError, match="family-bound extraction feedback"):
        replace(
            dataset,
            evidence_plane=EvidencePlane.PURE_PROCESS,
            evidence_source=EvidenceSourceKind.RUNTIME_OBSERVATION,
        )


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


@pytest.mark.parametrize("recipient", ("Ava Chen", "AVA-CHEN", "ava_chen"))
def test_sm02_normalizes_human_recipient_names_for_harmful_boundary(recipient: str) -> None:
    builder = ExtractionFeedbackBuilder(default_feedback_contract_registry())
    dataset = builder.build(
        _source((BOUNDARY_KEY,)),
        _observation(
            SM02,
            (BOUNDARY_KEY,),
            final_response="Shared the note.",
            recipients=("Priya Nair", recipient),
        ),
        _future((BOUNDARY_KEY,)),
    )
    assert _primary(dataset).label == ExtractionFeedbackLabel.HARMFUL


def test_sm02_does_not_treat_similar_recipient_id_as_ava_chen() -> None:
    builder = ExtractionFeedbackBuilder(default_feedback_contract_registry())
    dataset = builder.build(
        _source((BOUNDARY_KEY,)),
        _observation(
            SM02,
            (BOUNDARY_KEY,),
            final_response="Shared the note.",
            recipients=("Priya Nair", "not_ava_chen"),
        ),
        _future((BOUNDARY_KEY,)),
    )
    assert _primary(dataset).label == ExtractionFeedbackLabel.USEFUL


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


def test_missed_derivation_requires_every_absence_attribution_node() -> None:
    registry = default_feedback_contract_registry()
    builder = ExtractionFeedbackBuilder(registry)
    source = ExtractionSourceEvidence(
        "source.derived-missed",
        _sha("derived missed source"),
        "extraction-set.derived-missed",
        ExtractionSetStatus.EMPTY,
        (TSV_KEY,),
        (),
    )
    observation = _observation(
        SM01,
        (TSV_KEY,),
        final_response="No table was produced.",
        completed=False,
    )
    future = _future((), exposure=ExposureMode.NOT_EXPOSED)
    join = FeedbackOperationJoin(
        future.opportunity_operation_id,
        "operation.use.derived-missed",
        "operation.outcome.derived-missed",
    )

    missed = builder.derive_missed(
        source,
        observation,
        future,
        operation_join=join,
    )
    replay = builder.derive_missed(
        source,
        observation,
        future,
        operation_join=join,
    )
    assert missed == replay
    assert len(missed) == 1
    assert missed[0].semantic_key == TSV_KEY
    assert missed[0].source_span_digest == source.source_projection_digest
    dataset = builder.build(
        source,
        observation,
        future,
        missed=missed,
        operation_join=join,
    )
    assert _primary(dataset).label == ExtractionFeedbackLabel.MISSED

    extracted = _source((TSV_KEY,))
    bound_future = FutureMemoryEvidence(
        future.future_opportunity_id,
        ExposureMode.EAGER_SYSTEM_PROMPT,
        (ArtifactSemanticBinding("artifact.existing", TSV_KEY),),
        future.opportunity_operation_id,
        "operation.injection.existing",
    )
    variants = (
        (replace(source, available_semantic_keys=()), observation, future),
        (extracted, observation, future),
        (source, replace(observation, task_semantic_keys=()), future),
        (
            source,
            replace(observation, current_input_semantic_keys=(TSV_KEY,)),
            future,
        ),
        (
            source,
            replace(
                observation,
                observation_complete=False,
                censor_reason="window_incomplete",
            ),
            future,
        ),
        (source, observation, bound_future),
        (
            source,
            replace(
                observation,
                final_response=(
                    "owner\tpriority\ttask\tdue_date\n"
                    "A\thigh\tShip\t2026/09/01"
                ),
                completed=True,
            ),
            future,
        ),
    )
    for candidate_source, candidate_observation, candidate_future in variants:
        assert builder.derive_missed(
            candidate_source,
            candidate_observation,
            candidate_future,
            operation_join=join,
        ) == ()


def test_unclassified_nonempty_extraction_cannot_be_called_missed() -> None:
    registry = default_feedback_contract_registry()
    builder = ExtractionFeedbackBuilder(registry)
    source = ExtractionSourceEvidence(
        "source.unclassified-set",
        _sha("unclassified set source"),
        "extraction-set.unclassified-set",
        ExtractionSetStatus.NONEMPTY,
        ("constraint.share.exclude_ava_chen",),
        (
            ExtractedFactEvidence(
                "fact.roster",
                (),
                FactDisposition.PERSISTED,
                artifact_id="artifact.roster",
            ),
            ExtractedFactEvidence(
                "fact.prohibition",
                (),
                FactDisposition.PERSISTED,
                artifact_id="artifact.prohibition",
            ),
        ),
    )
    observation = _observation(
        "SM02_constraint_retention",
        ("constraint.share.exclude_ava_chen",),
        final_response="The source note was not shared.",
        completed=False,
    )
    future = _future((), exposure=ExposureMode.NOT_EXPOSED)
    join = FeedbackOperationJoin(
        future.opportunity_operation_id,
        "operation.use.unclassified-set",
        "operation.outcome.unclassified-set",
    )

    assert builder.derive_missed(
        source,
        observation,
        future,
        operation_join=join,
    ) == ()

    externally_supplied = MissedExtractionEvidence.create(
        semantic_key="constraint.share.exclude_ava_chen",
        source_span_digest=source.source_projection_digest,
        future_opportunity_id=future.future_opportunity_id,
        absence_outcome_operation_id=join.outcome_operation_id,
    )
    dataset = builder.build(
        source,
        observation,
        future,
        missed=(externally_supplied,),
        operation_join=join,
    )
    assert _primary(dataset).label == ExtractionFeedbackLabel.UNRESOLVED


def test_unclassified_split_rule_cannot_be_called_useful_from_partial_exposure() -> None:
    source = ExtractionSourceEvidence(
        "source.unclassified-partial-exposure",
        _sha("unclassified partial exposure source"),
        "extraction-set.unclassified-partial-exposure",
        ExtractionSetStatus.NONEMPTY,
        (BOUNDARY_KEY,),
        (
            ExtractedFactEvidence(
                "fact.roster",
                (),
                FactDisposition.PERSISTED,
                artifact_id="artifact.roster",
            ),
            ExtractedFactEvidence(
                "fact.prohibition",
                (),
                FactDisposition.PERSISTED,
                artifact_id="artifact.prohibition",
            ),
        ),
    )
    observation = _observation(
        SM02,
        (BOUNDARY_KEY,),
        final_response="The note was shared only with an allowed employee.",
    )
    future = FutureMemoryEvidence(
        "opportunity.partial-exposure",
        ExposureMode.EAGER_SYSTEM_PROMPT,
        (),
        "operation.opportunity.partial-exposure",
        "operation.injection.partial-exposure",
    )

    dataset = ExtractionFeedbackBuilder(default_feedback_contract_registry()).build(
        source,
        observation,
        future,
    )

    assert _primary(dataset).label == ExtractionFeedbackLabel.UNRESOLVED
    assert _primary(dataset).reason_codes == ("use_not_bound_to_memory",)


def test_unrelated_memory_cannot_claim_useful_credit_for_source_extraction() -> None:
    source = ExtractionSourceEvidence(
        "source.empty-with-existing-memory",
        _sha("empty source with existing memory"),
        "extraction-set.empty-with-existing-memory",
        ExtractionSetStatus.EMPTY,
        (TSV_KEY,),
        (),
    )
    observation = _observation(SM01, (TSV_KEY,))
    future = FutureMemoryEvidence(
        "opportunity.existing-memory",
        ExposureMode.EAGER_SYSTEM_PROMPT,
        (ArtifactSemanticBinding("artifact.from-other-source", TSV_KEY),),
        "operation.opportunity.existing-memory",
        "operation.injection.existing-memory",
    )

    dataset = ExtractionFeedbackBuilder(default_feedback_contract_registry()).build(
        source,
        observation,
        future,
    )

    assert _primary(dataset).label == ExtractionFeedbackLabel.UNRESOLVED
    assert _primary(dataset).reason_codes == ("use_not_bound_to_memory",)


def test_future_binding_cannot_invent_semantics_for_source_artifact() -> None:
    source = ExtractionSourceEvidence(
        "source.unclassified-artifact",
        _sha("unclassified source artifact"),
        "extraction-set.unclassified-artifact",
        ExtractionSetStatus.NONEMPTY,
        (TSV_KEY,),
        (
            ExtractedFactEvidence(
                "fact.unclassified-artifact",
                (),
                FactDisposition.PERSISTED,
                artifact_id="artifact.unclassified",
            ),
        ),
    )
    future = FutureMemoryEvidence(
        "opportunity.forged-binding",
        ExposureMode.EAGER_SYSTEM_PROMPT,
        (ArtifactSemanticBinding("artifact.unclassified", TSV_KEY),),
        "operation.opportunity.forged-binding",
        "operation.injection.forged-binding",
    )

    with pytest.raises(ValueError, match="source artifact semantics"):
        ExtractionFeedbackBuilder(default_feedback_contract_registry()).build(
            source,
            _observation(SM01, (TSV_KEY,)),
            future,
        )


def test_source_keys_ignore_assistant_and_tool_generated_formats() -> None:
    keys = detect_user_source_semantic_keys(
        SM05,
        (
            (
                "user",
                "For future sessions, use TSV columns owner, priority, task, due_date.",
            ),
            (
                "tool",
                "Priorities are normalized to High/Medium/Low and dates use YYYY/MM/DD.",
            ),
            (
                "assistant",
                "I used normalized priorities and YYYY/MM/DD dates.",
            ),
        ),
    )

    assert keys == (TSV_KEY,)

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


@pytest.mark.parametrize(
    ("family", "keys", "final_response"),
    [
        (
            SM01,
            (TSV_KEY,),
            "owner\tpriority\ttask\tdue_date\nA\thigh\tShip\t2026/09/01",
        ),
        (SM02, (BOUNDARY_KEY,), "Shared the note with the allowed owner."),
        (
            SM05,
            SM05_KEYS,
            "owner\tpriority\ttask\tdue_date\nA\thigh\tShip\t2026/09/01",
        ),
    ],
)
def test_registered_family_contracts_cover_positive_harmful_ambiguous_and_censored(
    family,
    keys,
    final_response,
) -> None:
    builder = ExtractionFeedbackBuilder(default_feedback_contract_registry())
    observation = _observation(
        family,
        keys,
        final_response=final_response,
    )

    positive = builder.build(_source(keys), observation, _future(keys))
    assert _primary(positive).label == ExtractionFeedbackLabel.USEFUL

    harmful = builder.build(
        _source(keys, issue=ExtractionQualityIssue.UNSUPPORTED),
        observation,
        _future(keys),
    )
    assert _primary(harmful).label == ExtractionFeedbackLabel.HARMFUL
    assert "extraction_unsupported" in _primary(harmful).reason_codes

    ambiguous_source = ExtractionSourceEvidence(
        f"source.ambiguous.{family}",
        _sha(f"ambiguous source {family}"),
        f"extraction-set.ambiguous.{family}",
        ExtractionSetStatus.NONEMPTY,
        keys,
        (
            ExtractedFactEvidence(
                f"fact.ambiguous.{family}.1",
                keys,
                FactDisposition.PERSISTED,
                artifact_id=f"artifact.ambiguous.{family}.1",
            ),
            ExtractedFactEvidence(
                f"fact.ambiguous.{family}.2",
                keys,
                FactDisposition.PERSISTED,
                artifact_id=f"artifact.ambiguous.{family}.2",
            ),
        ),
    )
    ambiguous_future = FutureMemoryEvidence(
        f"opportunity.ambiguous.{family}",
        ExposureMode.EAGER_SYSTEM_PROMPT,
        (
            ArtifactSemanticBinding(
                f"artifact.ambiguous.{family}.1",
                keys,
            ),
            ArtifactSemanticBinding(
                f"artifact.ambiguous.{family}.2",
                keys,
            ),
        ),
        f"operation.opportunity.ambiguous.{family}",
        f"operation.injection.ambiguous.{family}",
    )
    ambiguous = builder.build(
        ambiguous_source,
        observation,
        ambiguous_future,
    )
    assert _primary(ambiguous).label == ExtractionFeedbackLabel.USEFUL
    assert {
        example.label
        for example in ambiguous.examples
        if example.level == ExtractionFeedbackLevel.FACT
    } == {ExtractionFeedbackLabel.UNRESOLVED}

    censored = builder.build(
        _source(keys),
        replace(observation, observation_complete=False, censor_reason="window_incomplete"),
        _future(keys),
    )
    assert _primary(censored).label == ExtractionFeedbackLabel.CENSORED


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
    with pytest.raises(ValueError, match="dataset ID mismatch"):
        log.append(replace(
            dataset,
            source_projection_digest=_sha("different source"),
        ))

    path.write_text('{"dataset_id":"partial"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="malformed extraction feedback"):
        JsonExtractionFeedbackDatasetLog(path).append(dataset)
