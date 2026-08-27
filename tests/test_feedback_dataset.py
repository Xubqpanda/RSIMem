from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from rsimem.lifecycle import RawResourceUsage
from rsimem.memory.attribution import DeterministicFirstAttributor
from rsimem.memory.feedback_dataset import (
    CandidateDisposition,
    DelayedFeedbackConfig,
    DelayedFeedbackDatasetBuilder,
    ExposureState,
    FeedbackEstimator,
    FeedbackLabel,
    FeedbackObservationWindow,
    JsonDelayedFeedbackDatasetStore,
    PropensitySource,
    audit_feedback_dataset,
    build_feedback_dataset_report,
    validate_feedback_estimator,
)
from rsimem.memory.ingestion import InternalMemoryAction
from rsimem.memory.operation_graph import (
    AppendOnlyOperationEvidenceLog,
    ArtifactKind,
    ArtifactNode,
    AtomicOperationRecorder,
    MutationEdge,
    OperationContext,
    OperationGraph,
    OperationKind,
    OperationRecord,
    OperationStatus,
    materialize_operation_graph,
)


PRIVATE_MEMORY = "The private user prefers a four-column TSV."
POLICY_VERSION = "mem0-flat.utility.fixture"
FEATURE_SCHEMA = "semantic-static-utility-features-v1"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _context(episode: str, task: str) -> OperationContext:
    return OperationContext(
        "run-feedback",
        episode,
        f"session-{episode}",
        task,
        POLICY_VERSION,
        "prompt-v1",
        "framework-v1",
    )


def _artifact(
    recorder: AtomicOperationRecorder,
    artifact_id: str,
    kind: ArtifactKind,
    *,
    revision: str | None = None,
) -> str:
    recorder.record_artifact(ArtifactNode(
        artifact_id,
        kind,
        "feedback-fixture-v1",
        _sha(artifact_id),
        len(artifact_id),
        None,
        revision,
        "fixture-provenance",
    ))
    return artifact_id


def _operation(
    recorder: AtomicOperationRecorder,
    operation_id: str,
    kind: OperationKind,
    context: OperationContext,
    *,
    parents: tuple[str, ...] = (),
    inputs: tuple[str, ...] = (),
    outputs: tuple[str, ...] = (),
    status: OperationStatus = OperationStatus.SUCCESS,
    reason_code: str | None = None,
    model_requests: int = 0,
    retry_count: int = 0,
    storage_bytes: int = 0,
) -> str:
    recorder.record_operation(OperationRecord(
        operation_id,
        kind,
        context,
        parents,
        inputs,
        outputs,
        "attempt-0",
        status,
        reason_code,
        1,
        RawResourceUsage(
            input_tokens=1,
            output_tokens=1,
            cache_read_tokens=0,
            cache_write_tokens=0,
            reasoning_tokens=0,
            model_requests=model_requests,
            retry_count=retry_count,
            duration_ms=1,
            storage_bytes=storage_bytes,
        ),
    ))
    return operation_id


def _graph(
    exposure: str,
    *,
    superseded: bool = False,
    irrelevant: bool = False,
    second_retrieval: bool = False,
    used_failure: bool = False,
) -> OperationGraph:
    log = AppendOnlyOperationEvidenceLog()
    recorder = AtomicOperationRecorder(log)
    learn = _context("learn", "task-learn")
    future = _context("future", "task-future")
    parameter = _artifact(recorder, "parameter.fact", ArtifactKind.POLICY_PARAMETER)
    fact = _artifact(recorder, "artifact.fact", ArtifactKind.EXTRACTED_FACT)
    memory = _artifact(
        recorder,
        "memory.feedback",
        ArtifactKind.MEMORY_ARTIFACT,
        revision="revision-1",
    )
    retrieval_result = _artifact(
        recorder,
        "artifact.retrieval",
        ArtifactKind.RETRIEVAL_RESULT,
    )
    injection_artifact = _artifact(
        recorder,
        "artifact.injection",
        ArtifactKind.INJECTION,
    )
    use_artifact = _artifact(
        recorder,
        "artifact.use",
        ArtifactKind.USE_EVIDENCE,
    )
    outcome_artifact = _artifact(
        recorder,
        "artifact.outcome",
        ArtifactKind.OUTCOME,
    )

    source = _operation(recorder, "op.source", OperationKind.SOURCE_OBSERVATION, learn)
    extraction = _operation(
        recorder,
        "op.extraction",
        OperationKind.FACT_EXTRACTION,
        learn,
        parents=(source,),
        inputs=(parameter,),
        outputs=(fact,),
        model_requests=1,
    )
    related = _operation(
        recorder,
        "op.related",
        OperationKind.RELATED_MEMORY_RETRIEVAL,
        learn,
        parents=(extraction,),
    )
    decision = _operation(
        recorder,
        "op.decision",
        OperationKind.INTERNAL_OPERATION_DECISION,
        learn,
        parents=(related,),
        inputs=(parameter, fact),
        model_requests=1,
    )
    resolution = _operation(
        recorder,
        "op.resolution",
        OperationKind.TARGET_RESOLUTION,
        learn,
        parents=(decision,),
    )
    validation = _operation(
        recorder,
        "op.validation",
        OperationKind.VALIDATION,
        learn,
        parents=(resolution,),
        inputs=(fact,),
    )
    mutation = _operation(
        recorder,
        "op.mutation",
        OperationKind.MUTATION,
        learn,
        parents=(validation,),
        inputs=(fact,),
        outputs=(memory,),
        storage_bytes=64,
    )
    recorder.record_mutation(MutationEdge(
        "mutation.feedback",
        mutation,
        (decision,),
        InternalMemoryAction.ADD,
        memory,
        None,
        None,
        _sha(memory),
        "receipt.feedback",
    ))
    verification = _operation(
        recorder,
        "op.verification",
        OperationKind.REREAD_VERIFICATION,
        learn,
        parents=(mutation,),
        inputs=(memory,),
    )
    query = _operation(
        recorder,
        "op.query",
        OperationKind.FUTURE_QUERY,
        future,
        parents=(verification,),
    )
    retrieval_status = (
        OperationStatus.NONE
        if exposure in {"not_retrieved", "policy_filtered"}
        else OperationStatus.SUCCESS
    )
    retrieval_inputs = (
        (memory,)
        if exposure == "policy_filtered"
        else ()
        if retrieval_status == OperationStatus.NONE
        else (memory,)
    )
    retrieval = _operation(
        recorder,
        "op.retrieval",
        OperationKind.RETRIEVAL,
        future,
        parents=(query,),
        inputs=retrieval_inputs,
        outputs=(
            () if retrieval_status == OperationStatus.NONE else (retrieval_result,)
        ),
        status=retrieval_status,
        reason_code=(
            "policy_filtered"
            if exposure == "policy_filtered"
            else "retrieval_miss"
            if retrieval_status == OperationStatus.NONE
            else None
        ),
    )
    retrieval_parent = retrieval
    if second_retrieval:
        second_query = _operation(
            recorder,
            "op.query-2",
            OperationKind.FUTURE_QUERY,
            future,
            parents=(retrieval,),
        )
        retrieval_parent = _operation(
            recorder,
            "op.retrieval-2",
            OperationKind.RETRIEVAL,
            future,
            parents=(second_query,),
            inputs=(memory,),
            outputs=(retrieval_result,),
        )
    injection_status = (
        OperationStatus.SUCCESS
        if exposure in {"injected_not_used", "used", "censored"}
        else OperationStatus.NONE
    )
    injection = _operation(
        recorder,
        "op.injection",
        OperationKind.INJECTION,
        future,
        parents=(retrieval_parent,),
        inputs=(memory, retrieval_result) if retrieval_status == OperationStatus.SUCCESS else (),
        outputs=((injection_artifact,) if injection_status == OperationStatus.SUCCESS else ()),
        status=injection_status,
        reason_code=(
            None
            if injection_status == OperationStatus.SUCCESS
            else "retrieved_not_injected"
            if retrieval_status == OperationStatus.SUCCESS
            else "retrieval_miss"
        ),
    )
    use_status = (
        OperationStatus.SUCCESS if exposure in {"used", "censored"} else OperationStatus.NONE
    )
    use = _operation(
        recorder,
        "op.use",
        OperationKind.USE,
        future,
        parents=(injection,),
        inputs=((memory, injection_artifact) if injection_status == OperationStatus.SUCCESS else ()),
        outputs=((use_artifact,) if use_status == OperationStatus.SUCCESS else ()),
        status=use_status,
        reason_code=(
            None
            if use_status == OperationStatus.SUCCESS
            else "retrieved_but_unused"
            if injection_status == OperationStatus.SUCCESS
            else "not_exposed"
        ),
    )
    tool = _operation(
        recorder,
        "op.tool",
        OperationKind.TOOL_BEHAVIOR,
        future,
        parents=(use,),
        inputs=((memory,) if use_status == OperationStatus.SUCCESS else ()),
        status=(OperationStatus.SUCCESS if use_status == OperationStatus.SUCCESS else OperationStatus.NONE),
        reason_code=(None if use_status == OperationStatus.SUCCESS else "tool_not_called"),
        retry_count=1,
    )
    if exposure == "censored":
        outcome_status = OperationStatus.NONE
        outcome_reason = "observation_censored"
    elif exposure == "used" and not used_failure:
        outcome_status = OperationStatus.SUCCESS
        outcome_reason = None
    else:
        outcome_status = OperationStatus.FAILED
        outcome_reason = "task_failed"
    outcome = _operation(
        recorder,
        "op.outcome",
        OperationKind.DOWNSTREAM_OUTCOME,
        future,
        parents=(tool,),
        inputs=((use_artifact,) if use_status == OperationStatus.SUCCESS else ()),
        outputs=(outcome_artifact,),
        status=outcome_status,
        reason_code=outcome_reason,
    )
    if superseded:
        outcome = _operation(
            recorder,
            "op.supersession",
            OperationKind.SUPERSESSION,
            future,
            parents=(outcome,),
            inputs=(memory,),
        )
    _operation(
        recorder,
        "op.recovery",
        OperationKind.RECOVERY,
        future,
        parents=(outcome,),
        inputs=(memory,),
    )
    if irrelevant:
        _operation(
            recorder,
            "op.irrelevant",
            OperationKind.TOOL_BEHAVIOR,
            future,
        )
    return materialize_operation_graph(log.events)


def _dataset(
    graph: OperationGraph,
    *,
    complete: bool = True,
    reports=(),
    window_version: str = "semantic-observation-window-v1",
    cutoff_operation_id: str | None = None,
):
    window = FeedbackObservationWindow.create(
        graph,
        cutoff_operation_id=cutoff_operation_id,
        complete=complete,
        censor_reason=None if complete else "observation_window_closed",
        version=window_version,
    )
    config = DelayedFeedbackConfig(
        POLICY_VERSION,
        FEATURE_SCHEMA,
        window_version=window_version,
    )
    return DelayedFeedbackDatasetBuilder(config).build(
        graph,
        window,
        attribution_reports=reports,
    )


def _shared_retrieval_graph() -> OperationGraph:
    log = AppendOnlyOperationEvidenceLog()
    recorder = AtomicOperationRecorder(log)
    learn = _context("learn", "task-learn")
    future = _context("future", "task-future")
    parameter = _artifact(recorder, "parameter.shared", ArtifactKind.POLICY_PARAMETER)
    retrieval_result = _artifact(
        recorder,
        "artifact.shared-retrieval",
        ArtifactKind.RETRIEVAL_RESULT,
    )
    injection_artifact = _artifact(
        recorder,
        "artifact.shared-injection",
        ArtifactKind.INJECTION,
    )
    use_artifact = _artifact(
        recorder,
        "artifact.shared-use",
        ArtifactKind.USE_EVIDENCE,
    )
    outcome_artifact = _artifact(
        recorder,
        "artifact.shared-outcome",
        ArtifactKind.OUTCOME,
    )
    verifications = []
    memories = []
    for ordinal in (1, 2):
        fact = _artifact(
            recorder,
            f"artifact.fact-{ordinal}",
            ArtifactKind.EXTRACTED_FACT,
        )
        memory = _artifact(
            recorder,
            f"memory.shared-{ordinal}",
            ArtifactKind.MEMORY_ARTIFACT,
            revision=f"revision-{ordinal}",
        )
        memories.append(memory)
        source = _operation(
            recorder,
            f"op.source-{ordinal}",
            OperationKind.SOURCE_OBSERVATION,
            learn,
        )
        extraction = _operation(
            recorder,
            f"op.extraction-{ordinal}",
            OperationKind.FACT_EXTRACTION,
            learn,
            parents=(source,),
            inputs=(parameter,),
            outputs=(fact,),
        )
        decision = _operation(
            recorder,
            f"op.decision-{ordinal}",
            OperationKind.INTERNAL_OPERATION_DECISION,
            learn,
            parents=(extraction,),
            inputs=(parameter, fact),
        )
        mutation = _operation(
            recorder,
            f"op.mutation-{ordinal}",
            OperationKind.MUTATION,
            learn,
            parents=(decision,),
            inputs=(fact,),
            outputs=(memory,),
        )
        recorder.record_mutation(MutationEdge(
            f"mutation.shared-{ordinal}",
            mutation,
            (decision,),
            InternalMemoryAction.ADD,
            memory,
            None,
            None,
            _sha(memory),
            f"receipt.shared-{ordinal}",
        ))
        verifications.append(_operation(
            recorder,
            f"op.verification-{ordinal}",
            OperationKind.REREAD_VERIFICATION,
            learn,
            parents=(mutation,),
            inputs=(memory,),
        ))
    query = _operation(
        recorder,
        "op.shared-query",
        OperationKind.FUTURE_QUERY,
        future,
        parents=tuple(verifications),
    )
    retrieval = _operation(
        recorder,
        "op.shared-retrieval",
        OperationKind.RETRIEVAL,
        future,
        parents=(query,),
        inputs=tuple(memories),
        outputs=(retrieval_result,),
    )
    injection = _operation(
        recorder,
        "op.shared-injection",
        OperationKind.INJECTION,
        future,
        parents=(retrieval,),
        inputs=(*memories, retrieval_result),
        outputs=(injection_artifact,),
    )
    use = _operation(
        recorder,
        "op.shared-use",
        OperationKind.USE,
        future,
        parents=(injection,),
        inputs=(memories[0], injection_artifact),
        outputs=(use_artifact,),
    )
    _operation(
        recorder,
        "op.shared-outcome",
        OperationKind.DOWNSTREAM_OUTCOME,
        future,
        parents=(use,),
        inputs=(use_artifact,),
        outputs=(outcome_artifact,),
    )
    return materialize_operation_graph(log.events)


def test_positive_negative_unresolved_and_censored_labels_are_distinct() -> None:
    positive_graph = _graph("used")
    positive = _dataset(positive_graph).examples[0]
    assert positive.label == FeedbackLabel.POSITIVE
    assert positive.exposure_state == ExposureState.USED
    assert positive.tool_operation_ids == ("op.tool",)
    assert positive.recovery_operation_ids == ("op.recovery",)
    assert positive.source_operation_ids == ("op.source",)
    assert positive.extraction_operation_ids == ("op.extraction",)
    assert positive.related_retrieval_operation_ids == ("op.related",)
    assert positive.decision_operation_ids == ("op.decision",)
    assert positive.target_resolution_operation_ids == ("op.resolution",)
    assert positive.validation_operation_ids == ("op.validation",)
    assert positive.verification_operation_ids == ("op.verification",)
    assert positive.query_operation_ids == ("op.query",)
    assert positive.retrieval_operation_ids == ("op.retrieval",)
    assert positive.injection_operation_ids == ("op.injection",)
    assert positive.use_operation_ids == ("op.use",)
    assert positive.outcome_operation_ids == ("op.outcome",)
    assert positive.mutation_operation_id == "op.mutation"
    assert positive.memory_revision == "revision-1"
    assert positive.observation_cutoff_operation_id == "op.recovery"
    assert positive.resources.retry_count == 1
    assert positive.exposure_opportunity is True
    assert positive.entered_candidate_set is True
    assert positive.candidate_disposition == CandidateDisposition.INCLUDED
    assert positive.selection_propensity == 1.0
    assert positive.propensity_source == PropensitySource.DETERMINISTIC

    negative_graph = _graph("injected_not_used")
    report = DeterministicFirstAttributor().attribute(negative_graph)
    negative = _dataset(negative_graph, reports=(report,)).examples[0]
    assert negative.label == FeedbackLabel.NEGATIVE
    assert negative.exposure_state == ExposureState.INJECTED_NOT_USED
    assert negative.failure_subgraph_operation_ids == (
        "op.retrieval",
        "op.use",
    )
    assert negative.attributed_operation_ids == negative.failure_subgraph_operation_ids
    assert len(negative.attribution_record_ids) == 1
    assert negative.attribution_methods[0].value == "deterministic_exposure"
    assert negative.failure_categories[0].value == "retrieved_but_unused"

    unresolved_graph = _graph("retrieved_not_injected")
    unresolved = _dataset(unresolved_graph).examples[0]
    assert unresolved.label == FeedbackLabel.UNRESOLVED
    assert unresolved.exposure_state == ExposureState.RETRIEVED_NOT_INJECTED

    censored_graph = _graph("censored")
    censored = _dataset(censored_graph, complete=False).examples[0]
    assert censored.label == FeedbackLabel.CENSORED
    assert censored.exposure_state == ExposureState.CENSORED


@pytest.mark.parametrize(
    ("exposure", "state"),
    (
        ("not_retrieved", ExposureState.NOT_RETRIEVED),
        ("retrieved_not_injected", ExposureState.RETRIEVED_NOT_INJECTED),
    ),
)
def test_unexposed_memory_is_not_mislabeled_negative(exposure, state) -> None:
    example = _dataset(_graph(exposure)).examples[0]
    assert example.label == FeedbackLabel.UNRESOLVED
    assert example.exposure_state == state


def test_exposure_bias_distinguishes_miss_filter_and_no_opportunity() -> None:
    miss = _dataset(_graph("not_retrieved")).examples[0]
    assert miss.exposure_opportunity is True
    assert miss.entered_candidate_set is False
    assert miss.candidate_disposition == CandidateDisposition.UNKNOWN
    assert miss.selection_propensity is None
    assert miss.propensity_source == PropensitySource.MISSING

    filtered = _dataset(_graph("policy_filtered")).examples[0]
    assert filtered.exposure_opportunity is True
    assert filtered.entered_candidate_set is False
    assert filtered.candidate_disposition == CandidateDisposition.FILTERED
    assert filtered.selection_propensity == 0.0
    assert filtered.propensity_source == PropensitySource.DETERMINISTIC

    no_opportunity = _dataset(
        _graph("not_retrieved"),
        cutoff_operation_id="op.verification",
    ).examples[0]
    assert no_opportunity.exposure_opportunity is False
    assert no_opportunity.entered_candidate_set is False
    assert no_opportunity.candidate_disposition == CandidateDisposition.NOT_ELIGIBLE
    assert no_opportunity.selection_propensity is None
    assert no_opportunity.propensity_source == PropensitySource.MISSING


def test_propensity_estimators_fail_closed_and_direct_estimator_accepts_missing() -> None:
    missing = _dataset(_graph("not_retrieved"))
    validate_feedback_estimator(missing, FeedbackEstimator.DIRECT)
    for estimator in (
        FeedbackEstimator.INVERSE_PROPENSITY_WEIGHTED,
        FeedbackEstimator.DOUBLY_ROBUST,
    ):
        with pytest.raises(ValueError, match="requires propensity"):
            validate_feedback_estimator(missing, estimator)

    zero = _dataset(_graph("policy_filtered"))
    for estimator in (
        FeedbackEstimator.INVERSE_PROPENSITY_WEIGHTED,
        FeedbackEstimator.DOUBLY_ROBUST,
    ):
        with pytest.raises(ValueError, match="strictly positive"):
            validate_feedback_estimator(zero, estimator)

    validate_feedback_estimator(
        _dataset(_graph("used")),
        FeedbackEstimator.INVERSE_PROPENSITY_WEIGHTED,
    )


def test_feedback_report_surfaces_exposure_and_censoring_counts() -> None:
    datasets = (
        _dataset(_graph("used")),
        _dataset(_graph("policy_filtered")),
        _dataset(_graph("censored"), complete=False),
    )
    combined = replace(
        datasets[0],
        examples=tuple(dataset.examples[0] for dataset in datasets),
    )
    report = build_feedback_dataset_report(combined)

    assert report.observation_count == 3
    assert report.opportunity_count == 3
    assert report.candidate_count == 2
    assert report.filtered_count == 1
    assert report.missing_propensity_count == 0
    assert report.censored_count == 1
    assert report.censoring_rate == pytest.approx(1 / 3)
    assert dict(report.label_counts) == {
        FeedbackLabel.POSITIVE: 1,
        FeedbackLabel.NEGATIVE: 0,
        FeedbackLabel.UNRESOLVED: 1,
        FeedbackLabel.CENSORED: 1,
    }
    assert dict(report.exposure_counts)[ExposureState.CENSORED] == 1


@pytest.mark.parametrize(
    ("changes", "error"),
    (
        ({"exposure_opportunity": False}, "exposure opportunity"),
        ({"entered_candidate_set": False}, "candidate inclusion"),
        (
            {
                "candidate_disposition": CandidateDisposition.FILTERED,
                "entered_candidate_set": False,
                "selection_propensity": 0.0,
            },
            "exposure_bias_evidence_mismatch",
        ),
        (
            {
                "selection_propensity": 0.5,
                "propensity_source": PropensitySource.LOGGED,
            },
            "exposure_bias_evidence_mismatch",
        ),
    ),
)
def test_exposure_bias_contract_and_audit_reject_tampering(changes, error) -> None:
    graph = _graph("used")
    dataset = _dataset(graph)
    try:
        tampered = replace(dataset.examples[0], **changes)
    except (TypeError, ValueError) as exc:
        assert error in str(exc)
        return
    audit = audit_feedback_dataset(replace(dataset, examples=(tampered,)), graph)
    assert error in audit.issues


def test_used_memory_with_unattributed_failure_remains_unresolved() -> None:
    example = _dataset(_graph("used", used_failure=True)).examples[0]
    assert example.label == FeedbackLabel.UNRESOLVED
    assert example.exposure_state == ExposureState.USED
    assert example.label_reason_codes == ("used_without_attributed_success",)


def test_supersession_and_multiple_retrievals_are_joined() -> None:
    graph = _graph(
        "retrieved_not_injected",
        superseded=True,
        second_retrieval=True,
    )
    example = _dataset(graph).examples[0]
    assert example.label == FeedbackLabel.NEGATIVE
    assert example.exposure_state == ExposureState.SUPERSEDED
    assert example.retrieval_operation_ids == ("op.retrieval", "op.retrieval-2")
    assert example.supersession_operation_ids == ("op.supersession",)


def test_shared_query_multiple_hits_preserves_artifact_specific_use() -> None:
    dataset = _dataset(_shared_retrieval_graph())
    by_artifact = {
        example.memory_artifact_id: example for example in dataset.examples
    }
    first = by_artifact["memory.shared-1"]
    second = by_artifact["memory.shared-2"]

    assert first.retrieval_operation_ids == second.retrieval_operation_ids == (
        "op.shared-retrieval",
    )
    assert first.query_operation_ids == second.query_operation_ids == (
        "op.shared-query",
    )
    assert first.label == FeedbackLabel.POSITIVE
    assert first.exposure_state == ExposureState.USED
    assert second.label == FeedbackLabel.NEGATIVE
    assert second.exposure_state == ExposureState.INJECTED_NOT_USED


def test_replay_window_versions_and_irrelevant_operations_are_deterministic(
    tmp_path,
) -> None:
    graph = _graph("used")
    first = _dataset(graph)
    replay = _dataset(graph)
    assert first == replay
    assert first.payload() == replay.payload()

    versioned = _dataset(
        graph,
        window_version="semantic-observation-window-v2",
    )
    assert versioned.dataset_id != first.dataset_id
    assert versioned.examples[0].label == first.examples[0].label
    store = JsonDelayedFeedbackDatasetStore(tmp_path / "datasets")
    first_path, first_created = store.put(first)
    replay_path, replay_created = store.put(replay)
    versioned_path, versioned_created = store.put(versioned)
    assert first_created is True
    assert replay_created is False
    assert versioned_created is True
    assert first_path == replay_path
    assert versioned_path != first_path
    assert first_path.exists() and versioned_path.exists()

    with_irrelevant = _dataset(_graph("used", irrelevant=True))
    assert with_irrelevant.examples[0].label == first.examples[0].label
    assert "op.irrelevant" not in with_irrelevant.examples[0].operation_ids

    first_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicts"):
        store.put(first)


def test_integrity_audit_detects_orphan_revision_duplicate_and_future_leakage() -> None:
    graph = _graph("injected_not_used")
    report = DeterministicFirstAttributor().attribute(graph)
    dataset = _dataset(graph, reports=(report,))
    example = dataset.examples[0]

    without_attributed = OperationGraph(
        graph.artifacts,
        tuple(
            operation
            for operation in graph.operations
            if operation.operation_id != "op.use"
        ),
        graph.mutations,
    )
    orphan = audit_feedback_dataset(dataset, without_attributed)
    assert orphan.ok is False
    assert "orphan_operation" in orphan.issues

    wrong_revision = replace(example, memory_revision="revision-wrong")
    revision_dataset = replace(dataset, examples=(wrong_revision,))
    revision = audit_feedback_dataset(revision_dataset, graph)
    assert "revision_mismatch" in revision.issues

    wrong_label = replace(
        example,
        label=FeedbackLabel.POSITIVE,
        exposure_state=ExposureState.USED,
    )
    wrong_label_dataset = replace(dataset, examples=(wrong_label,))
    label_audit = audit_feedback_dataset(wrong_label_dataset, graph)
    assert "label_evidence_mismatch" in label_audit.issues

    wrong_kind = replace(example, source_operation_ids=("op.query",))
    wrong_kind_dataset = replace(dataset, examples=(wrong_kind,))
    kind_audit = audit_feedback_dataset(wrong_kind_dataset, graph)
    assert "operation_kind_mismatch" in kind_audit.issues

    duplicate_dataset = replace(dataset, examples=(example, example))
    duplicate = audit_feedback_dataset(duplicate_dataset, graph)
    assert "duplicate_example" in duplicate.issues

    duplicate_graph = OperationGraph(
        graph.artifacts,
        (*graph.operations, graph.operations[0]),
        graph.mutations,
    )
    duplicate_operation = audit_feedback_dataset(dataset, duplicate_graph)
    assert "duplicate_operation" in duplicate_operation.issues

    missing_artifact_graph = OperationGraph(
        tuple(
            artifact
            for artifact in graph.artifacts
            if artifact.artifact_id != example.memory_artifact_id
        ),
        graph.operations,
        graph.mutations,
    )
    missing_artifact = audit_feedback_dataset(dataset, missing_artifact_graph)
    assert "orphan_artifact" in missing_artifact.issues

    earlier_window = FeedbackObservationWindow.create(
        graph,
        cutoff_operation_id="op.injection",
        complete=True,
    )
    future_dataset = replace(dataset, window=earlier_window)
    future = audit_feedback_dataset(future_dataset, graph)
    assert "future_leakage" in future.issues

    duplicate_graph = OperationGraph(
        (*graph.artifacts, graph.artifacts[0]),
        (*graph.operations, graph.operations[0]),
        (*graph.mutations, graph.mutations[0]),
    )
    duplicated = audit_feedback_dataset(dataset, duplicate_graph)
    assert {
        "duplicate_artifact",
        "duplicate_operation",
        "duplicate_mutation",
    } <= set(duplicated.issues)

    without_memory = OperationGraph(
        tuple(
            artifact
            for artifact in graph.artifacts
            if artifact.artifact_id != example.memory_artifact_id
        ),
        graph.operations,
        graph.mutations,
    )
    missing_artifact = audit_feedback_dataset(dataset, without_memory)
    assert "orphan_artifact" in missing_artifact.issues

    cycle_operations = tuple(
        replace(
            operation,
            parent_operation_ids=("op.recovery",),
        )
        if operation.operation_id == "op.source"
        else operation
        for operation in graph.operations
    )
    cyclic = audit_feedback_dataset(
        dataset,
        OperationGraph(graph.artifacts, cycle_operations, graph.mutations),
    )
    assert "operation_cycle" in cyclic.issues


def test_dataset_payload_is_content_free_and_has_no_hidden_score_surface() -> None:
    graph = _graph("used")
    dataset = _dataset(graph)
    audit = audit_feedback_dataset(dataset, graph)
    serialized = json.dumps(dataset.payload(), sort_keys=True)

    assert audit.ok is True
    assert PRIVATE_MEMORY not in serialized
    for forbidden in (
        '"content"',
        '"prompt"',
        '"response"',
        '"memory"',
        '"score"',
        '"grader"',
        '"expectation"',
    ):
        assert forbidden not in serialized
    assert dataset.examples[0].resources.model_requests == 2
    assert dataset.examples[0].resources.storage_bytes == 64


def test_immediate_failure_attribution_cannot_determine_delayed_label() -> None:
    graph = _graph("used")
    operations = tuple(
        replace(
            operation,
            status=OperationStatus.NONE,
            reason_code="extraction_miss",
        )
        if operation.operation_id == "op.extraction"
        else operation
        for operation in graph.operations
    )
    inconsistent_fixture = OperationGraph(
        graph.artifacts,
        operations,
        graph.mutations,
    )
    report = DeterministicFirstAttributor().attribute(inconsistent_fixture)
    example = _dataset(
        inconsistent_fixture,
        reports=(report,),
    ).examples[0]

    assert example.attributed_operation_ids == ("op.extraction",)
    assert example.label == FeedbackLabel.POSITIVE
    assert example.label_reason_codes == ("used_with_successful_outcome",)
    dataset = _dataset(inconsistent_fixture, reports=(report,))
    assert audit_feedback_dataset(dataset, inconsistent_fixture).ok
