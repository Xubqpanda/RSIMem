from __future__ import annotations

from rsimem.lifecycle import RawResourceUsage
from rsimem.memory.attribution import DeterministicFirstAttributor, FailureCategory
from rsimem.memory.backends import build_hermes_native_registry
from rsimem.memory.future_trace import SemanticFutureTraceRecorder
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


def test_future_retrieval_miss_is_distinct_from_unexposed_use(tmp_path) -> None:
    registry, log, recorder = _environment(tmp_path, memory=None)
    future = recorder.record_prompt_injection(
        registry,
        "Base prompt without memory.",
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
