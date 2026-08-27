from __future__ import annotations

import json

import pytest

from rsimem.lifecycle import RawResourceUsage
from rsimem.memory.attribution import (
    AttributionBudget,
    AttributionModelError,
    AttributionMethod,
    DeterministicFirstAttributor,
    FailureCategory,
    ModelAttributionResponse,
)
from rsimem.memory.operation_graph import (
    AppendOnlyOperationEvidenceLog,
    ArtifactKind,
    ArtifactNode,
    AtomicOperationRecorder,
    OperationContext,
    OperationKind,
    OperationRecord,
    OperationStatus,
    materialize_operation_graph,
)


def _sha(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _context() -> OperationContext:
    return OperationContext(
        "run-attribution",
        "episode-attribution",
        "session-attribution",
        "task-attribution",
        "mem0-flat-v2",
        "prompt-v2",
        "framework-v1",
    )


def _artifact(
    recorder: AtomicOperationRecorder,
    artifact_id: str,
    kind: ArtifactKind,
) -> str:
    recorder.record_artifact(ArtifactNode(
        artifact_id,
        kind,
        "attribution-fixture-v1",
        _sha(artifact_id),
        len(artifact_id),
        None,
        "revision-1" if kind == ArtifactKind.MEMORY_ARTIFACT else None,
        "fixture-provenance",
    ))
    return artifact_id


def _operation(
    recorder: AtomicOperationRecorder,
    operation_id: str,
    kind: OperationKind,
    *,
    parents: tuple[str, ...] = (),
    inputs: tuple[str, ...] = (),
    outputs: tuple[str, ...] = (),
    status: OperationStatus = OperationStatus.SUCCESS,
    reason_code: str | None = None,
) -> str:
    recorder.record_operation(OperationRecord(
        operation_id,
        kind,
        _context(),
        parents,
        inputs,
        outputs,
        "attempt-0",
        status,
        reason_code,
        0,
        RawResourceUsage(),
    ))
    return operation_id


def _five_failure_graph():
    log = AppendOnlyOperationEvidenceLog()
    recorder = AtomicOperationRecorder(log)
    fact_prompt = _artifact(recorder, "parameter.fact-prompt", ArtifactKind.POLICY_PARAMETER)
    update_prompt = _artifact(recorder, "parameter.update-prompt", ArtifactKind.POLICY_PARAMETER)
    retrieval_parameters = _artifact(
        recorder,
        "parameter.retrieval",
        ArtifactKind.POLICY_PARAMETER,
    )
    memory = _artifact(recorder, "memory.fixture", ArtifactKind.MEMORY_ARTIFACT)
    retrieval_result = _artifact(
        recorder,
        "artifact.retrieval-result",
        ArtifactKind.RETRIEVAL_RESULT,
    )
    injection = _artifact(recorder, "artifact.injection", ArtifactKind.INJECTION)

    source = _operation(recorder, "op.source", OperationKind.SOURCE_OBSERVATION)
    extraction = _operation(
        recorder,
        "op.extraction-miss",
        OperationKind.FACT_EXTRACTION,
        parents=(source,),
        inputs=(fact_prompt,),
        status=OperationStatus.NONE,
        reason_code="extraction_miss",
    )
    decision = _operation(
        recorder,
        "op.update-decision",
        OperationKind.INTERNAL_OPERATION_DECISION,
        parents=(source,),
        inputs=(update_prompt,),
    )
    _operation(
        recorder,
        "op.wrong-target",
        OperationKind.TARGET_RESOLUTION,
        parents=(decision,),
        inputs=(memory,),
        status=OperationStatus.REJECTED,
        reason_code="wrong_update_target",
    )
    add_decision = _operation(
        recorder,
        "op.add-decision",
        OperationKind.INTERNAL_OPERATION_DECISION,
        parents=(source,),
        inputs=(update_prompt,),
    )
    _operation(
        recorder,
        "op.duplicate-add",
        OperationKind.MUTATION,
        parents=(add_decision,),
        inputs=(memory,),
        status=OperationStatus.REJECTED,
        reason_code="duplicate_add",
    )
    query = _operation(
        recorder,
        "op.query",
        OperationKind.FUTURE_QUERY,
        parents=(source,),
    )
    _operation(
        recorder,
        "op.retrieval-miss",
        OperationKind.RETRIEVAL,
        parents=(query,),
        inputs=(retrieval_parameters,),
        status=OperationStatus.NONE,
        reason_code="retrieval_miss",
    )
    retrieval = _operation(
        recorder,
        "op.retrieval-hit",
        OperationKind.RETRIEVAL,
        parents=(query,),
        inputs=(retrieval_parameters, memory),
        outputs=(retrieval_result,),
    )
    injected = _operation(
        recorder,
        "op.injection",
        OperationKind.INJECTION,
        parents=(retrieval,),
        inputs=(memory, retrieval_result),
        outputs=(injection,),
    )
    _operation(
        recorder,
        "op.retrieved-unused",
        OperationKind.USE,
        parents=(injected,),
        inputs=(memory, injection),
        status=OperationStatus.NONE,
        reason_code="retrieved_but_unused",
    )
    del extraction
    return materialize_operation_graph(log.events)


def test_five_failures_map_to_distinct_minimal_operation_categories() -> None:
    graph = _five_failure_graph()
    report = DeterministicFirstAttributor().attribute(graph)

    assert [record.category for record in report.records] == [
        FailureCategory.EXTRACTION_MISS,
        FailureCategory.WRONG_UPDATE_TARGET,
        FailureCategory.DUPLICATE_ADD,
        FailureCategory.RETRIEVAL_MISS,
        FailureCategory.RETRIEVED_BUT_UNUSED,
    ]
    assert all(
        record.method != AttributionMethod.MODEL for record in report.records
    )
    by_category = {record.category: record for record in report.records}
    assert by_category[FailureCategory.EXTRACTION_MISS].policy_parameter_ids == (
        "parameter.fact-prompt",
    )
    assert by_category[FailureCategory.WRONG_UPDATE_TARGET].policy_parameter_ids == (
        "parameter.update-prompt",
    )
    assert by_category[FailureCategory.RETRIEVAL_MISS].policy_parameter_ids == (
        "parameter.retrieval",
    )
    assert by_category[FailureCategory.RETRIEVED_BUT_UNUSED].candidate_operation_ids == (
        "op.retrieval-hit",
        "op.retrieved-unused",
    )
    assert report.model_call_count == 0
    serialized = json.dumps(report.observer_evidence(), sort_keys=True)
    assert "private source text" not in serialized
    assert "model response text" not in serialized


def _task_failure_graph(*, exposed: bool = False):
    log = AppendOnlyOperationEvidenceLog()
    recorder = AtomicOperationRecorder(log)
    prompt = _artifact(recorder, "parameter.fact-prompt", ArtifactKind.POLICY_PARAMETER)
    source = _operation(recorder, "op.source", OperationKind.SOURCE_OBSERVATION)
    extraction = _operation(
        recorder,
        "op.extraction",
        OperationKind.FACT_EXTRACTION,
        parents=(source,),
        inputs=(prompt,),
    )
    outcome_parents = (extraction,)
    if exposed:
        memory = _artifact(recorder, "memory.exposed", ArtifactKind.MEMORY_ARTIFACT)
        retrieval = _operation(
            recorder,
            "op.exposed-retrieval",
            OperationKind.RETRIEVAL,
            parents=(extraction,),
            inputs=(memory,),
        )
        injection = _operation(
            recorder,
            "op.exposed-injection",
            OperationKind.INJECTION,
            parents=(retrieval,),
            inputs=(memory,),
        )
        outcome_parents = (extraction, injection)
    outcome = _operation(
        recorder,
        "op.outcome-failed",
        OperationKind.DOWNSTREAM_OUTCOME,
        parents=outcome_parents,
        status=OperationStatus.FAILED,
        reason_code="task_failed",
    )
    future = _operation(
        recorder,
        "op.future-hidden",
        OperationKind.FACT_EXTRACTION,
        parents=(outcome,),
        inputs=(prompt,),
        status=OperationStatus.NONE,
        reason_code="extraction_miss",
    )
    return materialize_operation_graph(log.events), extraction, outcome, future


def test_task_failure_is_unresolved_and_cutoff_excludes_future_evidence() -> None:
    graph, extraction, outcome, future = _task_failure_graph()
    attributor = DeterministicFirstAttributor()
    report = attributor.attribute(graph, cutoff_operation_id=outcome)

    assert len(report.records) == 1
    record = report.records[0]
    assert record.category == FailureCategory.UNRESOLVED_TASK_FAILURE
    assert record.method == AttributionMethod.UNRESOLVED
    assert record.candidate_operation_ids == (outcome,)
    assert extraction not in record.candidate_operation_ids
    assert future not in report.window.visible_operation_ids
    assert report.model_call_count == 0


class _ModelClient:
    def __init__(self) -> None:
        self.calls = []

    def attribute(self, request):
        self.calls.append(request)
        return ModelAttributionResponse(
            FailureCategory.EXTRACTION_MISS,
            ("op.extraction",),
            (),
            0.6,
            RawResourceUsage(
                input_tokens=5,
                output_tokens=2,
                cache_read_tokens=1,
                cache_write_tokens=1,
                reasoning_tokens=1,
                model_requests=1,
                retry_count=1,
                duration_ms=3,
            ),
        )


def test_model_fallback_is_explicit_deduplicated_budgeted_and_accounted() -> None:
    graph, _, outcome, _ = _task_failure_graph(exposed=True)
    client = _ModelClient()
    attributor = DeterministicFirstAttributor(
        model_client=client,
        model_enabled=True,
        budget=AttributionBudget(
            max_calls=1,
            max_input_tokens=10,
            max_output_tokens=10,
            max_wall_time_ms=10_000,
        ),
    )
    first = attributor.attribute(graph, cutoff_operation_id=outcome)
    second = attributor.attribute(graph, cutoff_operation_id=outcome)

    assert len(client.calls) == 1
    assert first.records[0].method == AttributionMethod.MODEL
    assert first.model_usage.model_requests == 1
    assert first.model_usage.input_tokens == 5
    assert second.records[0].method == AttributionMethod.UNRESOLVED
    assert second.records[0].reason_code == "model_budget_exhausted"
    assert attributor.policy_update_usage.model_requests == 1
    assert attributor.policy_update_usage.input_tokens == 5
    assert attributor.policy_update_usage.cache_read_tokens == 1
    assert attributor.policy_update_usage.cache_write_tokens == 1
    assert attributor.policy_update_usage.reasoning_tokens == 1
    assert attributor.policy_update_usage.retry_count == 1
    serialized_request = json.dumps(
        [dict(value) for value in client.calls[0].operation_records],
        sort_keys=True,
    )
    assert "prompt text" not in serialized_request
    assert "response text" not in serialized_request


def test_disabled_and_censored_attribution_never_change_or_call_control_path() -> None:
    graph, _, outcome, _ = _task_failure_graph()
    client = _ModelClient()
    before = graph
    disabled = DeterministicFirstAttributor(
        enabled=False,
        model_client=client,
        model_enabled=True,
        budget=AttributionBudget(1, 10, 10, 10_000),
    ).attribute(graph, cutoff_operation_id=outcome)
    assert disabled.disabled is True
    assert disabled.records == ()
    assert client.calls == []
    assert graph == before

    unexposed = DeterministicFirstAttributor(
        model_client=client,
        model_enabled=True,
        budget=AttributionBudget(1, 10, 10, 10_000),
    ).attribute(graph, cutoff_operation_id=outcome)
    assert unexposed.records[0].reason_code == "model_sample_ineligible"
    assert client.calls == []

    log = AppendOnlyOperationEvidenceLog()
    recorder = AtomicOperationRecorder(log)
    source = _operation(recorder, "op.censored-source", OperationKind.SOURCE_OBSERVATION)
    censored = _operation(
        recorder,
        "op.censored",
        OperationKind.DOWNSTREAM_OUTCOME,
        parents=(source,),
        status=OperationStatus.NONE,
        reason_code="observation_censored",
    )
    report = DeterministicFirstAttributor(
        model_client=client,
        model_enabled=True,
        budget=AttributionBudget(1, 10, 10, 10_000),
    ).attribute(materialize_operation_graph(log.events), cutoff_operation_id=censored)
    assert report.records == ()
    assert client.calls == []


def test_model_response_unknown_identity_fails_closed_but_usage_is_not_free() -> None:
    graph, _, outcome, _ = _task_failure_graph(exposed=True)

    class InvalidClient(_ModelClient):
        def attribute(self, request):
            self.calls.append(request)
            return ModelAttributionResponse(
                FailureCategory.EXTRACTION_MISS,
                ("op.unknown",),
                (),
                0.5,
                RawResourceUsage(input_tokens=1, output_tokens=1, model_requests=1),
            )

    client = InvalidClient()
    attributor = DeterministicFirstAttributor(
        model_client=client,
        model_enabled=True,
        budget=AttributionBudget(1, 10, 10, 10_000),
    )
    report = attributor.attribute(graph, cutoff_operation_id=outcome)
    assert len(client.calls) == 1
    assert report.records[0].method == AttributionMethod.UNRESOLVED
    assert report.records[0].reason_code == "invalid_model_attribution"
    assert report.model_call_count == 1
    assert attributor.policy_update_usage.model_requests == 1


def test_structured_model_failure_retains_usage_and_stops_on_unknown_tokens() -> None:
    graph, _, outcome, _ = _task_failure_graph(exposed=True)

    class FailingClient:
        def __init__(self):
            self.calls = 0

        def attribute(self, request):
            del request
            self.calls += 1
            raise AttributionModelError(
                "model_attribution_timeout",
                RawResourceUsage(
                    input_tokens=None,
                    output_tokens=None,
                    model_requests=1,
                    retry_count=1,
                    duration_ms=10,
                ),
            )

    client = FailingClient()
    attributor = DeterministicFirstAttributor(
        model_client=client,
        model_enabled=True,
        budget=AttributionBudget(2, 100, 100, 10_000),
    )
    first = attributor.attribute(graph, cutoff_operation_id=outcome)
    second = attributor.attribute(graph, cutoff_operation_id=outcome)
    assert first.records[0].reason_code == "model_attribution_timeout"
    assert first.model_call_count == 1
    assert first.model_usage.retry_count == 1
    assert second.records[0].reason_code == "model_budget_exhausted"
    assert second.model_call_count == 0
    assert client.calls == 1
    assert attributor.policy_update_usage.input_tokens is None


def test_batch_sampling_is_deterministic_and_deduplicates_windows() -> None:
    graph, _, outcome, _ = _task_failure_graph()
    attributor = DeterministicFirstAttributor()
    selected = attributor.attribute_batch(
        ((graph, outcome), (graph, outcome)),
        sample_rate=1.0,
        sample_key="fixed-batch",
    )
    sampled_out = attributor.attribute_batch(
        ((graph, outcome),),
        sample_rate=0.0,
        sample_key="fixed-batch",
    )
    assert len(selected) == 1
    assert sampled_out == ()
    with pytest.raises(ValueError, match="sample_rate"):
        attributor.attribute_batch(((graph, outcome),), sample_rate=1.1)
