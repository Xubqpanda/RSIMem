from __future__ import annotations

import pytest

from rsimem.lifecycle import RawResourceUsage
from rsimem.memory.attribution import DeterministicFirstAttributor, FailureCategory
from rsimem.memory.backends import build_hermes_native_registry
from rsimem.memory.future_trace import (
    SemanticFeedbackContract,
    SemanticFeedbackResolver,
    SemanticFutureEvidence,
    SemanticFutureTraceRecorder,
)
from rsimem.memory.extraction_feedback import (
    DeploymentObservation,
    ObservableToolEvent,
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
