from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

import rsimem.memory.operation_graph as operation_graph_module
from rsimem.lifecycle import RawResourceUsage
from rsimem.memory.ingestion import InternalMemoryAction
from rsimem.memory.operation_graph import (
    AppendOnlyOperationEvidenceLog,
    ArtifactKind,
    ArtifactNode,
    AtomicOperationRecorder,
    MutationEdge,
    OperationContext,
    OperationKind,
    OperationRecord,
    OperationSpec,
    OperationStatus,
    TraceBudget,
    TracingLevel,
    audit_operation_evidence,
    build_artifact_id,
    build_operation_id,
    materialize_operation_graph,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _context() -> OperationContext:
    return OperationContext(
        run_id="run-fixture",
        episode_id="episode-learn",
        session_id="session-learn",
        task_id="SM01_preference_adoption",
        policy_version="mem0-flat-v1",
        prompt_version="prompt-v1",
        framework_version="framework-v1",
    )


def _artifact(
    context: OperationContext,
    kind: ArtifactKind,
    name: str,
    content: str,
    *,
    revision: str | None = None,
) -> ArtifactNode:
    digest = _sha(content)
    return ArtifactNode(
        artifact_id=build_artifact_id(
            kind,
            context,
            logical_name=name,
            content_digest=digest,
        ),
        kind=kind,
        artifact_schema_version="artifact-v1",
        content_digest=digest,
        byte_size=len(content.encode("utf-8")),
        token_size=max(1, len(content.split())),
        revision=revision,
        provenance_ref=f"prov.{name}",
    )


def _spec(
    context: OperationContext,
    kind: OperationKind,
    step: str,
    *,
    parents: tuple[str, ...] = (),
    inputs: tuple[str, ...] = (),
    retry: str = "attempt-0",
) -> OperationSpec:
    return OperationSpec(
        build_operation_id(
            kind,
            context,
            step_id=step,
            retry_identity=retry,
            parent_operation_ids=parents,
            input_artifact_ids=inputs,
        ),
        kind,
        context,
        parents,
        inputs,
        retry,
    )


def _record(
    recorder: AtomicOperationRecorder,
    spec: OperationSpec,
    *,
    outputs: tuple[str, ...] = (),
    status: OperationStatus = OperationStatus.SUCCESS,
    reason_code: str | None = None,
    usage: RawResourceUsage = RawResourceUsage(),
) -> str:
    with recorder.operation_scope(spec) as operation:
        operation.complete(
            output_artifact_ids=outputs,
            status=status,
            reason_code=reason_code,
            usage=usage,
        )
    return spec.operation_id


def _ingestion_graph_fixture(tmp_path: Path):
    context = _context()
    evidence_path = tmp_path / "operation-evidence.jsonl"
    log = AppendOnlyOperationEvidenceLog(evidence_path)
    recorder = AtomicOperationRecorder(log, tracing_level=TracingLevel.MINIMAL)
    backend_file = tmp_path / "MEMORY.md"
    backend_file.write_text("native state remains unchanged\n", encoding="utf-8")
    before = backend_file.read_bytes()

    source = _artifact(context, ArtifactKind.SOURCE_OBSERVATION, "source", "private source")
    fact_a = _artifact(context, ArtifactKind.EXTRACTED_FACT, "fact-a", "fact a")
    fact_b = _artifact(context, ArtifactKind.EXTRACTED_FACT, "fact-b", "fact b")
    related_a = _artifact(
        context,
        ArtifactKind.RELATED_MEMORY,
        "related-a",
        "old memory a",
        revision="revision-1",
    )
    related_b = _artifact(
        context,
        ArtifactKind.RELATED_MEMORY,
        "related-b",
        "old memory b",
        revision="revision-2",
    )
    proposal_a = _artifact(
        context,
        ArtifactKind.OPERATION_PROPOSAL,
        "proposal-a",
        "proposal a",
    )
    proposal_b = _artifact(
        context,
        ArtifactKind.OPERATION_PROPOSAL,
        "proposal-b",
        "proposal b",
    )
    stored = _artifact(
        context,
        ArtifactKind.MEMORY_ARTIFACT,
        "stored",
        "updated memory",
        revision="revision-3",
    )
    verified = _artifact(
        context,
        ArtifactKind.VALIDATION_RESULT,
        "verified",
        "verified digest only",
    )
    for artifact in (
        source,
        fact_a,
        fact_b,
        related_a,
        related_b,
        proposal_a,
        proposal_b,
        stored,
        verified,
    ):
        recorder.record_artifact(artifact)

    source_op = _record(
        recorder,
        _spec(context, OperationKind.SOURCE_OBSERVATION, "source-observation"),
        outputs=(source.artifact_id,),
    )
    extraction_op = _record(
        recorder,
        _spec(
            context,
            OperationKind.FACT_EXTRACTION,
            "fact-extraction",
            parents=(source_op,),
            inputs=(source.artifact_id,),
        ),
        outputs=(fact_a.artifact_id, fact_b.artifact_id),
        usage=RawResourceUsage(input_tokens=12, output_tokens=4, model_requests=1),
    )
    related_op = _record(
        recorder,
        _spec(
            context,
            OperationKind.RELATED_MEMORY_RETRIEVAL,
            "related-retrieval",
            parents=(extraction_op,),
            inputs=(fact_a.artifact_id,),
        ),
        outputs=(related_a.artifact_id, related_b.artifact_id),
    )
    decision_a = _record(
        recorder,
        _spec(
            context,
            OperationKind.INTERNAL_OPERATION_DECISION,
            "decision-a",
            parents=(extraction_op, related_op),
            inputs=(fact_a.artifact_id, related_a.artifact_id, related_b.artifact_id),
        ),
        outputs=(proposal_a.artifact_id,),
    )
    decision_b = _record(
        recorder,
        _spec(
            context,
            OperationKind.INTERNAL_OPERATION_DECISION,
            "decision-b",
            parents=(extraction_op, related_op),
            inputs=(fact_b.artifact_id, related_a.artifact_id),
        ),
        outputs=(proposal_b.artifact_id,),
    )
    resolution_op = _record(
        recorder,
        _spec(
            context,
            OperationKind.TARGET_RESOLUTION,
            "target-resolution",
            parents=(decision_a, decision_b),
            inputs=(proposal_a.artifact_id, proposal_b.artifact_id),
        ),
    )
    validation_op = _record(
        recorder,
        _spec(
            context,
            OperationKind.VALIDATION,
            "validation",
            parents=(resolution_op,),
            inputs=(proposal_a.artifact_id, proposal_b.artifact_id),
        ),
    )
    mutation_op = _record(
        recorder,
        _spec(
            context,
            OperationKind.MUTATION,
            "fixture-mutation",
            parents=(validation_op,),
            inputs=(related_a.artifact_id, proposal_a.artifact_id, proposal_b.artifact_id),
        ),
        outputs=(stored.artifact_id,),
    )
    recorder.record_mutation(MutationEdge(
        mutation_id="mutation.fixture-1",
        operation_id=mutation_op,
        proposal_operation_ids=(decision_a, decision_b),
        action=InternalMemoryAction.UPDATE,
        target_artifact_id=related_a.artifact_id,
        expected_revision="revision-1",
        before_digest=related_a.content_digest,
        after_digest=stored.content_digest,
        receipt_id="receipt.fixture-1",
    ))
    verification_op = _record(
        recorder,
        _spec(
            context,
            OperationKind.REREAD_VERIFICATION,
            "reread-verification",
            parents=(mutation_op,),
            inputs=(stored.artifact_id,),
        ),
        outputs=(verified.artifact_id,),
    )
    assert backend_file.read_bytes() == before
    return context, log, recorder, stored, verification_op, evidence_path


def test_deterministic_fixture_rebuilds_atomic_ingestion_subgraph(tmp_path: Path) -> None:
    context, log, recorder, stored, verification_op, evidence_path = (
        _ingestion_graph_fixture(tmp_path)
    )
    del context, recorder, stored
    graph = materialize_operation_graph(log.events)
    kinds = [item.kind for item in graph.operations]
    assert OperationKind.SOURCE_OBSERVATION in kinds
    assert OperationKind.FACT_EXTRACTION in kinds
    assert OperationKind.RELATED_MEMORY_RETRIEVAL in kinds
    assert OperationKind.INTERNAL_OPERATION_DECISION in kinds
    assert OperationKind.TARGET_RESOLUTION in kinds
    assert OperationKind.VALIDATION in kinds
    assert OperationKind.MUTATION in kinds
    assert OperationKind.REREAD_VERIFICATION in kinds
    assert sum(item.kind == ArtifactKind.EXTRACTED_FACT for item in graph.artifacts) == 2
    assert sum(item.kind == ArtifactKind.RELATED_MEMORY for item in graph.artifacts) == 2
    assert len(graph.mutations) == 1
    assert len(graph.mutations[0].proposal_operation_ids) == 2
    assert graph.policy_target_join() == ((
        "mem0-flat-v1",
        graph.mutations[0].target_artifact_id,
        "revision-1",
    ),)
    subgraph = graph.operation_subgraph(verification_op)
    assert len(subgraph.operations) == len(graph.operations)
    assert evidence_path.read_text(encoding="utf-8").count("\n") == len(log.events)


def test_future_use_connects_repeated_retrieval_injection_use_and_outcome(
    tmp_path: Path,
) -> None:
    context, log, recorder, stored, verification_op, _ = _ingestion_graph_fixture(tmp_path)
    query_a = _artifact(context, ArtifactKind.QUERY, "query-a", "private future query a")
    query_b = _artifact(context, ArtifactKind.QUERY, "query-b", "private future query b")
    retrieval = _artifact(context, ArtifactKind.RETRIEVAL_RESULT, "retrieval", "rank evidence")
    injection = _artifact(context, ArtifactKind.INJECTION, "injection", "injection evidence")
    use = _artifact(context, ArtifactKind.USE_EVIDENCE, "use", "use evidence")
    outcome = _artifact(context, ArtifactKind.OUTCOME, "outcome", "outcome evidence")
    for artifact in (query_a, query_b, retrieval, injection, use, outcome):
        recorder.record_artifact(artifact)

    query_op = _record(
        recorder,
        _spec(
            context,
            OperationKind.FUTURE_QUERY,
            "future-query-a",
            parents=(verification_op,),
            inputs=(query_a.artifact_id,),
        ),
    )
    retrieval_a = _record(
        recorder,
        _spec(
            context,
            OperationKind.RETRIEVAL,
            "retrieval-a",
            parents=(query_op,),
            inputs=(query_a.artifact_id, stored.artifact_id),
        ),
        outputs=(retrieval.artifact_id,),
    )
    retrieval_b = _record(
        recorder,
        _spec(
            context,
            OperationKind.RETRIEVAL,
            "retrieval-b",
            parents=(verification_op,),
            inputs=(query_b.artifact_id, stored.artifact_id),
        ),
        status=OperationStatus.NONE,
        reason_code="not_retrieved",
    )
    injection_op = _record(
        recorder,
        _spec(
            context,
            OperationKind.INJECTION,
            "injection",
            parents=(retrieval_a,),
            inputs=(stored.artifact_id, retrieval.artifact_id),
        ),
        outputs=(injection.artifact_id,),
    )
    use_op = _record(
        recorder,
        _spec(
            context,
            OperationKind.USE,
            "use",
            parents=(injection_op,),
            inputs=(stored.artifact_id, injection.artifact_id),
        ),
        outputs=(use.artifact_id,),
    )
    _record(
        recorder,
        _spec(
            context,
            OperationKind.DOWNSTREAM_OUTCOME,
            "outcome",
            parents=(use_op, retrieval_b),
            inputs=(use.artifact_id,),
        ),
        outputs=(outcome.artifact_id,),
    )

    graph = materialize_operation_graph(log.events)
    retrievals = [item for item in graph.operations if item.kind == OperationKind.RETRIEVAL]
    assert len(retrievals) == 2
    assert all(stored.artifact_id in item.input_artifact_ids for item in retrievals)
    assert {item.status for item in retrievals} == {
        OperationStatus.SUCCESS,
        OperationStatus.NONE,
    }
    assert graph.operations[-1].kind == OperationKind.DOWNSTREAM_OUTCOME


def test_parallel_retry_none_rejection_failure_and_restart_keep_distinct_identity(
    tmp_path: Path,
) -> None:
    context = _context()
    root = _spec(context, OperationKind.SOURCE_OBSERVATION, "root")
    parallel_a = _spec(
        context,
        OperationKind.FACT_EXTRACTION,
        "parallel-a",
        parents=(root.operation_id,),
    )
    parallel_b = _spec(
        context,
        OperationKind.FACT_EXTRACTION,
        "parallel-b",
        parents=(root.operation_id,),
    )
    retry = _spec(
        context,
        OperationKind.FACT_EXTRACTION,
        "parallel-a",
        parents=(root.operation_id,),
        retry="attempt-1",
    )
    assert len({root.operation_id, parallel_a.operation_id, parallel_b.operation_id, retry.operation_id}) == 4

    path = tmp_path / "restart.jsonl"
    first_log = AppendOnlyOperationEvidenceLog(path)
    first = AtomicOperationRecorder(first_log)
    root_record = OperationRecord(
        root.operation_id,
        root.kind,
        context,
        (),
        (),
        (),
        root.retry_identity,
        OperationStatus.SUCCESS,
        None,
        0,
        RawResourceUsage(),
    )
    first.record_operation(root_record)
    _record(first, parallel_a, status=OperationStatus.NONE, reason_code="no_facts")
    _record(first, parallel_b, status=OperationStatus.REJECTED, reason_code="invalid_output")
    _record(first, retry, status=OperationStatus.FAILED, reason_code="provider_timeout")

    restarted_log = AppendOnlyOperationEvidenceLog(path)
    restarted = AtomicOperationRecorder(restarted_log)
    restarted.record_operation(root_record)
    assert len(restarted_log.events) == 4
    graph = materialize_operation_graph(restarted_log.events)
    assert graph.failure_groups().keys() == {"invalid_output", "provider_timeout"}
    assert len({item.operation_id for item in graph.operations}) == 4


def test_append_only_conflict_fails_closed() -> None:
    context = _context()
    log = AppendOnlyOperationEvidenceLog()
    recorder = AtomicOperationRecorder(log)
    record = OperationRecord(
        "op.conflict",
        OperationKind.VALIDATION,
        context,
        (),
        (),
        (),
        "attempt-0",
        OperationStatus.SUCCESS,
        None,
        0,
        RawResourceUsage(),
    )
    recorder.record_operation(record)
    conflict = copy.deepcopy(log.events[0])
    conflict["payload"]["latency_ms"] = 1
    with pytest.raises(ValueError, match="conflicting operation evidence"):
        log.append(conflict)


class _FailingSink:
    def append(self, event):
        raise OSError("SENTINEL_PRIVATE_WRITER_FAILURE")


def test_observer_failure_does_not_change_control_result_and_records_gap() -> None:
    context = _context()
    artifact = _artifact(context, ArtifactKind.EXTRACTED_FACT, "fact", "private fact")

    def execute(recorder=None):
        result = ("planned_mutation", artifact.content_digest)
        if recorder is not None:
            recorder.record_artifact(artifact)
        return result

    control = execute()
    failing = AtomicOperationRecorder(_FailingSink())
    observed = execute(failing)
    assert observed == control
    assert len(failing.observer_failures) == 1
    assert failing.observer_failures[0].reason_code == "observer_append_failed"
    assert len(failing.attribution_gaps) == 1
    assert "SENTINEL" not in repr(failing.observer_failures)


def test_digest_and_serialization_failures_are_observer_only(monkeypatch) -> None:
    context = _context()
    artifact = _artifact(context, ArtifactKind.EXTRACTED_FACT, "fact", "private fact")

    def fail_serialization(value):
        raise TypeError("SENTINEL_SERIALIZATION_FAILURE")

    serialization = AtomicOperationRecorder(
        AppendOnlyOperationEvidenceLog(),
        serializer=fail_serialization,
    )
    serialization.record_artifact(artifact)
    assert len(serialization.observer_failures) == 1
    assert serialization.metrics.event_count == 0

    original = operation_graph_module._stable_id

    def fail_event_digest(prefix, value):
        if prefix == "oev":
            raise RuntimeError("SENTINEL_DIGEST_FAILURE")
        return original(prefix, value)

    monkeypatch.setattr(operation_graph_module, "_stable_id", fail_event_digest)
    digest = AtomicOperationRecorder(AppendOnlyOperationEvidenceLog())
    digest.record_artifact(artifact)
    assert len(digest.observer_failures) == 1
    assert digest.observer_failures[0].failed_event_id.startswith("oev.failure.")
    assert "SENTINEL" not in repr(digest.observer_failures)


def test_tracing_levels_privacy_and_overhead_budget_are_explicit() -> None:
    context = _context()
    sentinel = "SENTINEL_RAW_SOURCE_31f9"
    artifact = _artifact(context, ArtifactKind.SOURCE_OBSERVATION, "source", sentinel)

    disabled_log = AppendOnlyOperationEvidenceLog()
    disabled = AtomicOperationRecorder(disabled_log, tracing_level=TracingLevel.DISABLED)
    disabled.record_artifact(artifact)
    assert disabled.metrics.event_count == 0

    sampled_out_log = AppendOnlyOperationEvidenceLog()
    sampled_out = AtomicOperationRecorder(
        sampled_out_log,
        tracing_level=TracingLevel.SAMPLED,
        sample_rate=0.0,
    )
    sampled_out.record_artifact(artifact)
    assert sampled_out.metrics.event_count == 0

    minimal_log = AppendOnlyOperationEvidenceLog()
    minimal = AtomicOperationRecorder(minimal_log, tracing_level=TracingLevel.MINIMAL)
    minimal.record_artifact(artifact)
    assert minimal.metrics.event_count == 1

    diagnostic_log = AppendOnlyOperationEvidenceLog()
    diagnostic = AtomicOperationRecorder(
        diagnostic_log,
        tracing_level=TracingLevel.DIAGNOSTIC,
        budget=TraceBudget(
            max_events=0,
            max_serialized_bytes=10_000_000,
            max_wall_time_ms=60_000,
            max_peak_memory_bytes=100_000_000,
        ),
    )
    diagnostic.record_artifact(artifact)
    report = diagnostic.overhead_report()
    assert report["event_count"] == 1
    assert report["serialized_bytes"] > 0
    assert report["cpu_time_ms"] >= 0
    assert report["wall_time_ms"] >= 0
    assert report["peak_memory_bytes"] >= 0
    assert report["effective_level"] == "minimal"
    assert diagnostic.attribution_gaps[0].reason_code == "trace_budget_exceeded"

    assert audit_operation_evidence(minimal_log.events, forbidden_values=(sentinel,)) == ()
    assert sentinel not in str(minimal_log.events)
    malicious = ({
        "content": sentinel,
        "credential": "sk-not-a-real-fixture-secret",
        "source_path": "/private/absolute/path",
    },)
    assert set(audit_operation_evidence(malicious, forbidden_values=(sentinel,))) == {
        "absolute_path",
        "credential_pattern",
        "forbidden_value",
        "raw_field",
    }


def test_operation_scope_preserves_exception_and_records_failed_operation() -> None:
    context = _context()
    log = AppendOnlyOperationEvidenceLog()
    recorder = AtomicOperationRecorder(log)
    spec = _spec(context, OperationKind.FACT_EXTRACTION, "exception")
    with pytest.raises(RuntimeError, match="fixture failure"):
        with recorder.operation_scope(spec):
            raise RuntimeError("fixture failure")
    graph = materialize_operation_graph(log.events)
    assert graph.operations[0].status == OperationStatus.FAILED
    assert graph.operations[0].reason_code == "operation_exception"


def test_rejected_proposal_failed_mutation_and_none_are_not_merged() -> None:
    context = _context()
    log = AppendOnlyOperationEvidenceLog()
    recorder = AtomicOperationRecorder(log)
    root = _spec(context, OperationKind.SOURCE_OBSERVATION, "root-status")
    root_id = _record(recorder, root)
    rejected = _spec(
        context,
        OperationKind.INTERNAL_OPERATION_DECISION,
        "rejected-proposal",
        parents=(root_id,),
    )
    rejected_id = _record(
        recorder,
        rejected,
        status=OperationStatus.REJECTED,
        reason_code="invalid_proposal",
    )
    failed = _spec(
        context,
        OperationKind.MUTATION,
        "failed-mutation",
        parents=(rejected_id,),
    )
    _record(
        recorder,
        failed,
        status=OperationStatus.FAILED,
        reason_code="storage_failure",
    )
    none = _spec(
        context,
        OperationKind.MUTATION,
        "none-mutation",
        parents=(rejected_id,),
    )
    _record(
        recorder,
        none,
        status=OperationStatus.NONE,
        reason_code="duplicate_memory",
    )
    recorder.record_mutation(MutationEdge(
        mutation_id="mutation.none",
        operation_id=none.operation_id,
        proposal_operation_ids=(rejected_id,),
        action=InternalMemoryAction.NONE,
        target_artifact_id=None,
        expected_revision=None,
        before_digest=None,
        after_digest=None,
        receipt_id=None,
    ))
    graph = materialize_operation_graph(log.events)
    assert len({item.operation_id for item in graph.operations}) == 4
    assert graph.failure_groups().keys() == {"invalid_proposal", "storage_failure"}
    assert graph.mutations[0].action == InternalMemoryAction.NONE


def test_contract_rejects_raw_path_identity_and_invalid_mutation_shapes() -> None:
    context = _context()
    with pytest.raises(ValueError, match="provenance reference"):
        ArtifactNode(
            "artifact.invalid",
            ArtifactKind.SOURCE_OBSERVATION,
            "artifact-v1",
            _sha("value"),
            5,
            1,
            None,
            "/absolute/source/path",
        )
    with pytest.raises(ValueError, match="UPDATE mutation evidence"):
        MutationEdge(
            "mutation.invalid",
            "op.invalid",
            ("op.proposal",),
            InternalMemoryAction.UPDATE,
            "artifact.target",
            None,
            _sha("before"),
            _sha("after"),
            "receipt.invalid",
        )
    del context
