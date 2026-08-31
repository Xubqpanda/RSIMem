from __future__ import annotations

from dataclasses import replace

import pytest

from rsimem.lifecycle import RawResourceUsage
from rsimem.memory.attribution import DeterministicFirstAttributor, FailureCategory
from rsimem.memory.backends import build_hermes_native_registry
from rsimem.memory.contracts import MemoryKind, MemoryQuery
from rsimem.memory.future_trace import (
    SemanticFeedbackContract,
    SemanticFeedbackResolver,
    SemanticFutureEvidence,
    SemanticFutureTraceRecorder,
)
from rsimem.memory.extraction_feedback import (
    DeploymentObservation,
    ObservableToolEvent,
    detect_current_input_semantic_keys,
    detect_extracted_fact_semantic_keys,
    detect_source_semantic_keys,
)
from rsimem.memory.operation_graph import (
    AppendOnlyOperationEvidenceLog,
    AtomicOperationRecorder,
    OperationContext,
    OperationKind,
    OperationRecord,
    OperationStatus,
    materialize_operation_graph,
)


def _environment(tmp_path, *, memory: str | None):
    home = tmp_path / "home"
    memories = home / "memories"
    memories.mkdir(parents=True)
    if memory is not None:
        (memories / "USER.md").write_text(memory, encoding="utf-8")
    registry = build_hermes_native_registry(home)
    log = AppendOnlyOperationEvidenceLog()
    recorder = AtomicOperationRecorder(log)
    context = OperationContext(
        "run-future",
        "episode-future",
        "session-future",
        "task-future",
        "policy-v1",
        "prompt-v1",
        "framework-v1",
    )
    recorder.record_operation(OperationRecord(
        "op.learn-verification",
        OperationKind.REREAD_VERIFICATION,
        context,
        (),
        (),
        (),
        "attempt-0",
        OperationStatus.SUCCESS,
        None,
        0,
        RawResourceUsage(),
    ))
    return registry, log, SemanticFutureTraceRecorder(recorder, context)


def _observation(
    *,
    family_id: str = "SM01_preference_adoption",
    stage: str = "eval_near",
    response: str,
    completed: bool = True,
    current_keys: tuple[str, ...] = (),
    task_keys: tuple[str, ...] = ("preference.summary.tsv",),
    recipients: tuple[str, ...] = ("owner",),
) -> DeploymentObservation:
    return DeploymentObservation(
        "observation.fixture",
        family_id,
        stage,
        "task.fixture",
        "a" * 64,
        current_keys,
        task_keys,
        response,
        (ObservableToolEvent(
            "tool.share",
            "notes_share",
            True,
            recipient_ids=recipients,
        ),),
        completed,
    )


def test_future_retrieval_miss_is_distinct_from_unexposed_use(tmp_path) -> None:
    registry, log, recorder = _environment(tmp_path, memory=None)
    future = recorder.record_prompt_injection(
        registry,
        "",
        namespace="user",
        parent_operation_ids=("op.learn-verification",),
    )
    recorder.record_use_and_outcome(
        future,
        used_artifact_ids=(),
        outcome_status=OperationStatus.FAILED,
        outcome_reason_code="task_failed",
    )
    graph = materialize_operation_graph(log.events)
    reasons = {item.kind: item.reason_code for item in graph.operations}
    assert reasons[OperationKind.RETRIEVAL] == "retrieval_miss"
    assert reasons[OperationKind.USE] == "not_exposed"
    report = DeterministicFirstAttributor().attribute(graph)
    assert [record.category for record in report.records] == [
        FailureCategory.RETRIEVAL_MISS,
    ]
    registry.close()


def test_future_trace_reuses_authoritative_adapter_hits(tmp_path, monkeypatch) -> None:
    """An adapter retrieval is not repeated while recording future evidence."""

    registry, log, recorder = _environment(tmp_path, memory="Use TSV for durable reports.")
    backend = registry.resolve(MemoryKind.SEMANTIC)
    authoritative_hits = tuple(backend.query(MemoryQuery(
        MemoryKind.SEMANTIC,
        "",
        namespace="user",
        limit=10,
    )))
    assert authoritative_hits

    def fail_query(_query):
        raise AssertionError("future trace performed a second semantic query")

    monkeypatch.setattr(backend, "query", fail_query)
    future = recorder.record_prompt_injection(
        registry,
        "Base prompt.\nUse TSV for durable reports.",
        namespace="user",
        parent_operation_ids=("op.learn-verification",),
        retrieved_hits=authoritative_hits,
    )
    assert future.memory_artifact_ids == tuple(
        hit.artifact.artifact_id for hit in authoritative_hits
    )
    assert future.injection_artifact_id is not None
    registry.close()


def test_sm01_feedback_contract_uses_only_predeclared_deployment_signal() -> None:
    future = SemanticFutureEvidence(
        "op.query",
        "op.retrieval",
        "op.injection",
        ("memory.one",),
        ("revision.one",),
        "artifact.injection",
    )
    resolver = SemanticFeedbackResolver(
        SemanticFeedbackContract.SM01_TSV_V1,
        family_id="SM01_preference_adoption",
        stage="eval_near",
    )
    positive = resolver.resolve(future, _observation(
        response=(
            "owner\tpriority\ttask\tdue_date\n"
            "Iris Chen\tHigh\tFix drift\t2026/04/28"
        ),
    ))
    assert positive.used_artifact_ids == ("memory.one",)
    assert positive.outcome_status == OperationStatus.SUCCESS
    assert positive.outcome_reason_code is None
    assert positive.reuse_signal_observed is True

    negative = resolver.resolve(
        future,
        _observation(response="- Iris Chen: Fix drift"),
    )
    assert negative.used_artifact_ids == ()
    assert negative.outcome_status == OperationStatus.NONE
    assert negative.outcome_reason_code == "injected_not_used"
    assert negative.reuse_signal_observed is False


def test_family_match_without_visible_requirement_does_not_create_opportunity() -> None:
    future = SemanticFutureEvidence(
        "op.query.no-scope",
        "op.retrieval.no-scope",
        "op.injection.no-scope",
        ("memory.one",),
        ("revision.one",),
        "artifact.injection",
    )
    resolver = SemanticFeedbackResolver(
        SemanticFeedbackContract.SM01_TSV_V1,
        family_id="SM01_preference_adoption",
        stage="eval_near",
    )
    result = resolver.resolve(
        future,
        _observation(
            response="owner\tpriority\ttask\tdue_date\na\tb\tc\td",
            task_keys=(),
        ),
    )
    assert result.eligible is True
    assert result.used_artifact_ids == ()
    assert result.reuse_signal_observed is False
    assert result.outcome_reason_code == "opportunity_not_observed"


def test_sm03_fact_correction_contract_resolves_corrected_value() -> None:
    future = SemanticFutureEvidence(
        "op.query",
        "op.retrieval",
        "op.injection",
        ("memory.one",),
        ("revision.one",),
        "artifact.injection",
    )
    resolver = SemanticFeedbackResolver(
        SemanticFeedbackContract.SM03_FACT_CORRECTION_V1,
        family_id="SM03_fact_correction",
        stage="eval_near",
    )
    positive = resolver.resolve(future, _observation(
        family_id="SM03_fact_correction",
        response=(
            "The current Phoenix freeze date is 2026-06-16. "
            "The note was shared with the rollout owners."
        ),
        task_keys=("fact.phoenix.release_freeze_date",),
    ))
    assert positive.used_artifact_ids == ("memory.one",)
    assert positive.outcome_status == OperationStatus.SUCCESS
    assert positive.reuse_signal_observed is True

    harmful = resolver.resolve(future, _observation(
        family_id="SM03_fact_correction",
        response=(
            "The current Phoenix freeze date is 2026-06-14. "
            "The note was shared with the rollout owners."
        ),
        task_keys=("fact.phoenix.release_freeze_date",),
    ))
    assert harmful.outcome_status == OperationStatus.FAILED
    assert harmful.outcome_reason_code == "memory_use_harmfully_attributed"

    stale_reference = resolver.resolve(future, _observation(
        family_id="SM03_fact_correction",
        response=(
            "The previous Phoenix freeze date was 2026-06-14. "
            "The note was shared with the rollout owners."
        ),
        task_keys=("fact.phoenix.release_freeze_date",),
    ))
    assert stale_reference.outcome_status == OperationStatus.FAILED


def test_sm03_fact_correction_keys_are_deterministic_and_confounded() -> None:
    key = "fact.phoenix.release_freeze_date"
    source = (
        "Authoritative update: the Phoenix release freeze date is 2026-06-16 "
        "going forward; the old date is obsolete.",
    )
    assert detect_source_semantic_keys("SM03_fact_correction", source) == (key,)
    assert detect_extracted_fact_semantic_keys(
        "SM03_fact_correction",
        "Phoenix release freeze date: 2026-06-16",
    ) == (key,)
    assert detect_extracted_fact_semantic_keys(
        "SM03_fact_correction",
        "Phoenix release freeze date: 2026-06-14",
    ) == ()
    assert detect_current_input_semantic_keys(
        "SM03_fact_correction",
        "For this task only, the Phoenix freeze date is 2026-06-16.",
    ) == (key,)
    assert detect_current_input_semantic_keys(
        "SM03_fact_correction",
        "Include the current Phoenix freeze date in the summary.",
    ) == ()


def test_sm03_fact_correction_contract_preserves_unresolved_and_censored() -> None:
    future = SemanticFutureEvidence(
        "op.query",
        "op.retrieval",
        "op.injection",
        ("memory.one",),
        ("revision.one",),
        "artifact.injection",
    )
    resolver = SemanticFeedbackResolver(
        SemanticFeedbackContract.SM03_FACT_CORRECTION_V1,
        family_id="SM03_fact_correction",
        stage="eval_far",
    )
    no_share_observation = _observation(
        family_id="SM03_fact_correction",
        stage="eval_far",
        response="The current Phoenix freeze date is 2026-06-16.",
        task_keys=("fact.phoenix.release_freeze_date",),
        recipients=(),
    )
    no_share_observation = replace(
        no_share_observation,
        tool_events=(ObservableToolEvent("tool.no-share", "notes_share", False),),
    )
    no_share = resolver.resolve(future, no_share_observation)
    assert no_share.outcome_status == OperationStatus.NONE
    assert no_share.outcome_reason_code == "outcome_not_attributable"

    confounded = resolver.resolve(future, _observation(
        family_id="SM03_fact_correction",
        stage="eval_far",
        response=(
            "The current Phoenix freeze date is 2026-06-16. "
            "The note was shared with the rollout owners."
        ),
        current_keys=("fact.phoenix.release_freeze_date",),
        task_keys=("fact.phoenix.release_freeze_date",),
    ))
    assert confounded.outcome_status == OperationStatus.NONE
    assert confounded.outcome_reason_code == "current_input_confounded"

    # The helper's completed flag is separate from observation completeness;
    # use an interrupted observation to exercise the censoring contract.
    censored = resolver.resolve(future, DeploymentObservation(
        "observation.sm03-censored",
        "SM03_fact_correction",
        "eval_far",
        "task.fixture",
        "a" * 64,
        (),
        ("fact.phoenix.release_freeze_date",),
        "The current Phoenix freeze date is 2026-06-16.",
        (ObservableToolEvent(
            "tool.share.censored",
            "notes_share",
            True,
            recipient_ids=("owner",),
        ),),
        False,
        observation_complete=False,
        censor_reason="execution_incomplete",
    ))
    assert censored.outcome_status == OperationStatus.NONE
    assert censored.outcome_reason_code == "execution_incomplete"


def test_sm01_feedback_contract_censors_ineligible_or_ambiguous_evidence() -> None:
    ineligible = SemanticFeedbackResolver(
        SemanticFeedbackContract.SM01_TSV_V1,
        family_id="SM01_preference_adoption",
        stage="learn_a",
    )
    future = SemanticFutureEvidence(
        "op.query",
        "op.retrieval",
        "op.injection",
        ("memory.one",),
        ("revision.one",),
        "artifact.injection",
    )
    censored = ineligible.resolve(
        future,
        _observation(
            stage="learn_a",
            response="owner\tpriority\ttask\tdue_date\na\tb\tc\td",
        ),
    )
    assert censored.eligible is False
    assert censored.outcome_status == OperationStatus.NONE
    assert censored.outcome_reason_code == "observation_censored"

    set_level = SemanticFeedbackResolver(
        SemanticFeedbackContract.SM01_TSV_V1,
        family_id="SM01_preference_adoption",
        stage="eval_far",
    ).resolve(SemanticFutureEvidence(
        "op.query",
        "op.retrieval",
        "op.injection",
        ("memory.one", "memory.two"),
        ("revision.one", "revision.two"),
        "artifact.injection",
    ), _observation(
        stage="eval_far",
        response="owner\tpriority\ttask\tdue_date\na\tb\tc\td",
    ))
    assert set_level.eligible is True
    assert set_level.used_artifact_ids == ("memory.one", "memory.two")
    assert set_level.outcome_status == OperationStatus.SUCCESS

    with pytest.raises(ValueError, match="requires family SM01_preference_adoption"):
        SemanticFeedbackResolver(
            SemanticFeedbackContract.SM01_TSV_V1,
            family_id="SM02_constraint_retention",
            stage="eval_near",
        )


@pytest.mark.parametrize(
    ("contract", "family_id", "response", "task_keys", "recipients"),
    (
        (
            SemanticFeedbackContract.SM02_BOUNDARY_V1,
            "SM02_constraint_retention",
            "The note was shared with the permitted recipients.",
            ("constraint.share.exclude_ava_chen",),
            ("iris_chen",),
        ),
        (
            SemanticFeedbackContract.SM05_NORMALIZED_TSV_V1,
            "SM05_weak_trigger_preference_adoption",
            "owner\tpriority\ttask\tdue_date\nIris\tHigh\tFix drift\t2026/04/28",
            (
                "preference.summary.tsv",
                "preference.priority.normalized",
                "preference.date.yyyy_mm_dd",
            ),
            ("iris_chen",),
        ),
    ),
)
def test_family_feedback_contracts_resolve_only_their_registered_family(
    contract: SemanticFeedbackContract,
    family_id: str,
    response: str,
    task_keys: tuple[str, ...],
    recipients: tuple[str, ...],
) -> None:
    future = SemanticFutureEvidence(
        "op.query",
        "op.retrieval",
        "op.injection",
        tuple(f"memory.{index}" for index, _ in enumerate(task_keys, start=1)),
        tuple(f"revision.{index}" for index, _ in enumerate(task_keys, start=1)),
        "artifact.injection",
    )
    resolver = SemanticFeedbackResolver(
        contract,
        family_id=family_id,
        stage="eval_near",
    )
    resolution = resolver.resolve(future, _observation(
        family_id=family_id,
        response=response,
        task_keys=task_keys,
        recipients=recipients,
    ))
    assert resolution.used_artifact_ids == future.memory_artifact_ids
    assert resolution.outcome_status == OperationStatus.SUCCESS
    assert resolution.reuse_signal_observed is True

    wrong_family = "SM01_preference_adoption"
    with pytest.raises(ValueError, match="requires family"):
        SemanticFeedbackResolver(
            contract,
            family_id=wrong_family,
            stage="eval_near",
        )


def test_retrieved_not_injected_does_not_become_retrieved_unused(tmp_path) -> None:
    memory = "Use TSV for durable reports."
    registry, log, recorder = _environment(tmp_path, memory=memory)
    future = recorder.record_prompt_injection(
        registry,
        "Base prompt deliberately omits the durable entry.",
        namespace="user",
        parent_operation_ids=("op.learn-verification",),
    )
    recorder.record_use_and_outcome(
        future,
        used_artifact_ids=(),
        outcome_status=OperationStatus.FAILED,
        outcome_reason_code="task_failed",
    )
    graph = materialize_operation_graph(log.events)
    use = next(item for item in graph.operations if item.kind == OperationKind.USE)
    injection = next(
        item for item in graph.operations if item.kind == OperationKind.INJECTION
    )
    assert injection.reason_code == "retrieved_not_injected"
    assert use.reason_code == "not_exposed"
    report = DeterministicFirstAttributor(model_enabled=True).attribute(graph)
    assert report.records[0].reason_code == "model_attribution_disabled"
    registry.close()


def test_injected_but_unused_attributes_retrieval_and_use_only(tmp_path) -> None:
    memory = "Use TSV for durable reports."
    registry, log, recorder = _environment(tmp_path, memory=memory)
    future = recorder.record_prompt_injection(
        registry,
        f"Base prompt.\n{memory}",
        namespace="user",
        parent_operation_ids=("op.learn-verification",),
    )
    recorder.record_use_and_outcome(
        future,
        used_artifact_ids=(),
        outcome_status=OperationStatus.FAILED,
        outcome_reason_code="task_failed",
    )
    graph = materialize_operation_graph(log.events)
    report = DeterministicFirstAttributor().attribute(graph)
    assert len(report.records) == 1
    record = report.records[0]
    assert record.category == FailureCategory.RETRIEVED_BUT_UNUSED
    assert tuple(
        next(item for item in graph.operations if item.operation_id == operation_id).kind
        for operation_id in record.candidate_operation_ids
    ) == (OperationKind.RETRIEVAL, OperationKind.USE)
    registry.close()
