from __future__ import annotations

import hashlib
import json

from rsimem.hermes_integration import capture_native_hermes_system_prompt
from rsimem.ledger import MemoryLedgerObserver
from rsimem.lifecycle import RawResourceUsage
from rsimem.memory import MemoryExperience, MemoryKind, MemoryMessage, MemoryQuery
from rsimem.memory.backends import build_hermes_native_registry
from rsimem.memory.executor import MutationExecutionStatus, TransactionalMutationExecutor
from rsimem.memory.future_trace import SemanticFutureTraceRecorder
from rsimem.memory.feedback_dataset import (
    DelayedFeedbackConfig,
    DelayedFeedbackDatasetBuilder,
    FeedbackLabel,
    FeedbackObservationWindow,
    audit_feedback_dataset,
)
from rsimem.memory.attribution import DeterministicFirstAttributor
from rsimem.memory.ingestion import (
    InternalMemoryAction,
    SemanticIngestionCoordinator,
    SemanticPolicyRegistry,
    build_semantic_ingest_request,
)
from rsimem.memory.operation_graph import OperationContext
from rsimem.memory.operation_graph import (
    AppendOnlyOperationEvidenceLog,
    ArtifactKind,
    AtomicOperationRecorder,
    OperationKind,
    OperationStatus,
    audit_operation_evidence,
    materialize_operation_graph,
)
from rsimem.memory.receipt_audit import audit_mutation_receipts
from rsimem.memory.receipts import JsonMutationReceiptStore, MutationReceiptStatus
from rsimem.memory.runtime import MemoryBackendRegistry
from rsimem.memory.semantic_loop import SEMANTIC_LOOP_SCHEMA_VERSION, SemanticWritebackLoop
from rsimem.memory.validation import MutationValidator
from rsimem.memory_systems.mem0_flat import (
    POLICY_FACT_EXTRACTION_PROMPT,
    POLICY_INTERNAL_OPERATION_PROMPT,
    FakeCompletionClient,
    FlatSemanticCandidateReader,
    Mem0FlatSemanticPolicy,
)
from test_semantic_ingestion_contracts import _ingestion_fixture


PREFERENCE = "Use TSV with owner, priority, task, and due_date."


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _loop_environment(tmp_path, *, enabled=True, fact_response=None):
    home = tmp_path / "hermes-home"
    home.mkdir(parents=True)
    registry = build_hermes_native_registry(home)
    receipt_store = JsonMutationReceiptStore(tmp_path / "receipts.json")
    client = FakeCompletionClient(
        {
            POLICY_FACT_EXTRACTION_PROMPT.artifact.prompt_id: (
                fact_response
                if fact_response is not None
                else json.dumps({"facts": [PREFERENCE]})
            ),
            POLICY_INTERNAL_OPERATION_PROMPT.artifact.prompt_id: json.dumps({
                "operations": [{
                    "fact_index": 0,
                    "action": "add",
                    "candidate_id": None,
                }],
            }),
        },
        usage=RawResourceUsage(
            input_tokens=9,
            output_tokens=4,
            model_requests=1,
            duration_ms=2,
        ),
    )
    policy = Mem0FlatSemanticPolicy(client)
    candidates = FlatSemanticCandidateReader(
        registry,
        ownership=receipt_store,
    )
    policies = SemanticPolicyRegistry()
    policies.register(policy)
    coordinator = SemanticIngestionCoordinator(
        policies,
        provider=policy.descriptor.provider,
    )
    validator = MutationValidator(registry, target_resolver=receipt_store)
    executor = TransactionalMutationExecutor(
        registry,
        validator,
        receipt_store,
        enabled=enabled,
        isolated_fixture=enabled,
    )
    operation_log = AppendOnlyOperationEvidenceLog(tmp_path / "sm01-operations.jsonl")
    operation_recorder = AtomicOperationRecorder(operation_log)
    observer = MemoryLedgerObserver(
        run_id="run-sm01",
        variant="static-rsimem",
        trace_id="trace-sm01",
        episode_id="episode-1",
        session_id="session-1",
        task_id="task-1",
        family_id="SM01_preference_adoption",
        stage="e2e",
        snapshot_id="snapshot-1",
        execution_mode="native+rsimem+ledger",
        output_path=tmp_path / "sm01-ledger.jsonl",
    )
    loop = SemanticWritebackLoop(
        coordinator,
        policy,
        candidates,
        executor,
        observer=observer,
        operation_recorder=operation_recorder,
    )
    return (
        home,
        registry,
        receipt_store,
        client,
        policy,
        loop,
        observer,
        operation_log,
    )


def _sm01_request(policy):
    snapshot, plan = _ingestion_fixture()
    experience = MemoryExperience(
        experience_id="experience-sm01-learn-real",
        session_id=snapshot.session_id,
        task_id=snapshot.task_id,
        outcome="completed",
        messages=tuple(
            MemoryMessage(segment.role, segment.content)
            for segment in snapshot.segments
        ),
    )
    request = build_semantic_ingest_request(
        snapshot,
        plan,
        experience,
        policy_version=policy.descriptor.policy_version,
        framework_version=policy.descriptor.framework_version,
    )
    return snapshot, request


def _future_task_model(system_prompt: str) -> str:
    """Deterministic model fixture sees only the real Hermes system prompt."""

    required = ("TSV", "owner", "priority", "task", "due_date")
    return "tsv_preference_used" if all(item in system_prompt for item in required) else "cold_default"


def test_sm01_learn_mutate_restart_native_inject_and_use(tmp_path) -> None:
    assert SEMANTIC_LOOP_SCHEMA_VERSION == 1
    home, registry, store, client, policy, loop, observer, operation_log = (
        _loop_environment(tmp_path)
    )
    snapshot, request = _sm01_request(policy)
    OperationContext(
        run_id="run-sm01",
        episode_id="episode-sm01",
        session_id="session-sm01",
        task_id="task-sm01",
        policy_version=policy.descriptor.policy_version,
        prompt_version=policy.descriptor.prompt_version,
        framework_version=policy.descriptor.framework_version,
    )
    result = loop.run(
        request,
        current_source_revision=snapshot.context_revision,
    )
    assert result.ingestion is not None
    assert result.executions[0].status == MutationExecutionStatus.COMMITTED
    assert result.logical_exit is True
    assert result.source_retained is False
    assert result.reason_code == "all_mutations_committed"
    assert len(client.calls) == 2
    assert store.all()[0].status == MutationReceiptStatus.COMMITTED
    assert store.all()[0].provenance.snapshot_id == snapshot.snapshot_id
    assert store.all()[0].provenance.operation_id == result.ingestion.operations[0].operation_id
    registry.close()

    restarted_registry = build_hermes_native_registry(home)
    backend = restarted_registry.resolve(MemoryKind.SEMANTIC)
    hits = backend.query(MemoryQuery(
        MemoryKind.SEMANTIC,
        "",
        namespace="user",
        limit=10,
    ))
    assert len(hits) == 1
    artifact = hits[0].artifact
    assert artifact.content == PREFERENCE
    assert _sha(artifact.content) == store.all()[0].applied_content_digest
    assert artifact.revision == store.all()[0].applied_revision

    base_message = "Future task: summarize action items without an explicit format hint."
    assert PREFERENCE not in base_message
    system_prompt = capture_native_hermes_system_prompt(
        home,
        base_system_message=base_message,
        observer=observer,
    )
    assert PREFERENCE in system_prompt
    assert _future_task_model(system_prompt) == "tsv_preference_used"
    assert audit_mutation_receipts(store, restarted_registry).ok is True

    ingestion_graph = materialize_operation_graph(operation_log.events)
    verification = next(
        operation
        for operation in ingestion_graph.operations
        if operation.kind == OperationKind.REREAD_VERIFICATION
    )
    learn_context = policy.operation_trace(request.idempotency_key).context
    future_context = OperationContext(
        learn_context.run_id,
        "episode-sm01-future",
        "session-sm01-future",
        "task-sm01-future",
        learn_context.policy_version,
        learn_context.prompt_version,
        learn_context.framework_version,
    )
    future_recorder = SemanticFutureTraceRecorder(
        loop.operation_recorder,
        future_context,
    )
    future = future_recorder.record_prompt_injection(
        restarted_registry,
        system_prompt,
        namespace="user",
        parent_operation_ids=(verification.operation_id,),
    )
    downstream = future_recorder.record_use_and_outcome(
        future,
        used_artifact_ids=(artifact.artifact_id,),
        outcome_status=OperationStatus.SUCCESS,
    )
    assert downstream.used_artifact_ids == (artifact.artifact_id,)

    evidence = json.dumps(result.observer_evidence(), sort_keys=True)
    assert PREFERENCE not in evidence
    assert "Future task" not in evidence
    events = observer.events
    assert [event["kind"] for event in events] == [
        "mutation_requested",
        "mutation_committed",
        "query",
        "retrieved",
        "query",
        "retrieved",
        "injected",
    ]
    committed = events[1]
    injected = events[-1]
    assert committed["data"]["artifactIds"] == [artifact.artifact_id]
    assert injected["data"]["artifactIds"] == [artifact.artifact_id]
    assert committed["data"]["attributes"]["operation_id"] == (
        result.ingestion.operations[0].operation_id
    )
    serialized_ledger = (tmp_path / "sm01-ledger.jsonl").read_text(encoding="utf-8")
    assert PREFERENCE not in serialized_ledger
    assert base_message not in serialized_ledger
    graph = materialize_operation_graph(operation_log.events)
    assert [operation.kind for operation in graph.operations] == [
        OperationKind.SOURCE_OBSERVATION,
        OperationKind.FACT_EXTRACTION,
        OperationKind.RELATED_MEMORY_RETRIEVAL,
        OperationKind.INTERNAL_OPERATION_DECISION,
        OperationKind.TARGET_RESOLUTION,
        OperationKind.VALIDATION,
        OperationKind.MUTATION,
        OperationKind.REREAD_VERIFICATION,
        OperationKind.FUTURE_QUERY,
        OperationKind.RETRIEVAL,
        OperationKind.INJECTION,
        OperationKind.USE,
        OperationKind.DOWNSTREAM_OUTCOME,
    ]
    memory_nodes = [
        item for item in graph.artifacts if item.artifact_id == artifact.artifact_id
    ]
    assert len(memory_nodes) == 1
    assert memory_nodes[0].revision == artifact.revision
    assert graph.operations[-1].input_artifact_ids
    parameters = {
        item.artifact_id: item
        for item in graph.artifacts
        if item.kind == ArtifactKind.POLICY_PARAMETER
    }
    owned_parameters = {
        operation.kind: tuple(
            parameters[artifact_id].provenance_ref
            for artifact_id in operation.input_artifact_ids
            if artifact_id in parameters
        )
        for operation in graph.operations
    }
    assert owned_parameters[OperationKind.FACT_EXTRACTION] == (
        "mem0-flat.fact-extraction",
    )
    assert owned_parameters[OperationKind.RELATED_MEMORY_RETRIEVAL] == (
        "flat-retrieval-v1",
    )
    assert owned_parameters[OperationKind.INTERNAL_OPERATION_DECISION] == (
        "mem0-flat.internal-operation",
    )
    successful_attribution = DeterministicFirstAttributor().attribute(graph)
    assert successful_attribution.records == ()
    assert successful_attribution.model_call_count == 0
    feedback_window = FeedbackObservationWindow.create(
        graph,
        complete=True,
    )
    feedback = DelayedFeedbackDatasetBuilder(DelayedFeedbackConfig(
        learn_context.policy_version,
        policy.descriptor.feature_schema_version,
    )).build(
        graph,
        feedback_window,
        attribution_reports=(successful_attribution,),
    )
    assert len(feedback.examples) == 1
    feedback_example = feedback.examples[0]
    assert feedback_example.label == FeedbackLabel.POSITIVE
    assert feedback_example.memory_artifact_id == artifact.artifact_id
    assert feedback_example.memory_revision == artifact.revision
    assert feedback_example.mutation_operation_id in feedback_example.operation_ids
    assert future.retrieval_operation_id in feedback_example.retrieval_operation_ids
    assert future.injection_operation_id in feedback_example.injection_operation_ids
    assert downstream.use_operation_id in feedback_example.use_operation_ids
    assert downstream.outcome_operation_id in feedback_example.outcome_operation_ids
    assert audit_feedback_dataset(feedback, graph).ok is True
    serialized_feedback = json.dumps(feedback.payload(), sort_keys=True)
    assert PREFERENCE not in serialized_feedback
    assert base_message not in serialized_feedback
    assert audit_operation_evidence(
        operation_log.events,
        forbidden_values=(PREFERENCE, base_message),
    ) == ()
    restarted_registry.close()


def test_disabled_loop_restores_direct_native_no_write_behavior(tmp_path) -> None:
    home, registry, store, _, policy, loop, _, _ = _loop_environment(
        tmp_path,
        enabled=False,
    )
    snapshot, request = _sm01_request(policy)
    result = loop.run(
        request,
        current_source_revision=snapshot.context_revision,
    )
    assert result.executions[0].status == MutationExecutionStatus.DISABLED
    assert result.logical_exit is False
    assert result.source_retained is True
    assert store.all() == ()
    assert not (home / "memories" / "USER.md").exists()
    prompt = capture_native_hermes_system_prompt(
        home,
        base_system_message="Future task without memory.",
    )
    assert PREFERENCE not in prompt
    assert _future_task_model(prompt) == "cold_default"
    registry.close()


def test_ingestion_failure_retains_source_and_never_creates_receipt(tmp_path) -> None:
    home, registry, store, _, policy, loop, _, _ = _loop_environment(
        tmp_path,
        fact_response="not json",
    )
    snapshot, request = _sm01_request(policy)
    result = loop.run(
        request,
        current_source_revision=snapshot.context_revision,
    )
    assert result.ingestion is not None
    assert result.ingestion.status.value == "rejected"
    assert result.executions == ()
    assert result.logical_exit is False
    assert result.source_retained is True
    assert store.all() == ()
    assert not (home / "memories" / "USER.md").exists()
    registry.close()


def test_stale_source_is_rejected_before_model_or_mutation(tmp_path) -> None:
    _, registry, store, client, policy, loop, _, _ = _loop_environment(tmp_path)
    _, request = _sm01_request(policy)
    try:
        loop.run(request, current_source_revision="rev-newer")
    except ValueError as exc:
        assert "stale" in str(exc)
    else:
        raise AssertionError("stale source should be rejected")
    assert client.calls == ()
    assert store.all() == ()
    registry.close()
