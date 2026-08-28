from __future__ import annotations

import copy
import hashlib
from dataclasses import replace

import pytest

from rsimem.audit import summarize_ingestion_usage
from rsimem.ledger import LifecycleLedgerObserver
from rsimem.lifecycle import (
    DeterministicPreferenceEvaluator,
    EvaluationTrigger,
    HermesMessage,
    HermesSnapshotCollector,
    RawResourceUsage,
    SegmentKind,
    TaskLifecycleState,
    WritebackCoordinator,
    run_sm01_preference_fixture,
    snapshot_to_evaluation_request,
)
from rsimem.memory import (
    MemoryAccessMode,
    MemoryExperience,
    MemoryKind,
    MemoryMessage,
)
from rsimem.memory.backends import HermesSemanticBackend
from rsimem.memory.ingestion import (
    BoundSemanticPolicy,
    ContextExitSemantics,
    DeterministicPassThroughIngestor,
    ExistingMemoryCandidate,
    FixedMemoryRoute,
    FixedMemoryRouter,
    HERMES_NATIVE_ROUTES,
    INGESTION_CONTRACT_SCHEMA_VERSION,
    InternalMemoryAction,
    InternalOperationProposal,
    InvalidPolicyOutputError,
    MemoryIngestOutcome,
    MemoryIngestRequest,
    MemoryIngestStatus,
    PolicyExecutionError,
    PolicyCapability,
    SemanticIngestRequest,
    SemanticIngestionCoordinator,
    SemanticPolicyDecision,
    SemanticPolicyDescriptor,
    SemanticPolicyRegistry,
    build_completed_task_semantic_ingest_request,
    build_semantic_ingest_request,
    mem0_flat_policy,
)
from rsimem.memory.extraction_source import ExtractionSourceProjector


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _Candidates:
    def __init__(self, values=(), *, resolved=None) -> None:
        self.values = tuple(values)
        self.resolved = dict(resolved or {item.candidate_id: item for item in self.values})

    def candidates(self, request):
        return self.values

    def resolve(self, candidate_id):
        return self.resolved.get(candidate_id)


def _experience(snapshot=None) -> MemoryExperience:
    snapshot = snapshot or _ingestion_fixture()[0]
    return ExtractionSourceProjector().project(snapshot).to_experience(snapshot)


def _ingestion_fixture():
    source = run_sm01_preference_fixture().snapshot
    messages = tuple(
        HermesMessage(
            segment.source_message_id,
            segment.role,
            segment.content,
            segment.turn_id,
            segment.token_count,
            kind=segment.kind,
            completed=segment.completed,
            tool_call_id=segment.tool_call_id,
            metadata=segment.metadata,
        )
        for segment in source.segments
    )
    snapshot = HermesSnapshotCollector().collect(
        messages,
        run_id=source.run_id,
        episode_id=source.episode_id,
        session_id=source.session_id,
        task_id=source.task_id,
        current_turn_id=None,
        task_state=TaskLifecycleState.COMPLETED,
        lifecycle_state="task_completed",
        source_ref=source.provenance.source_ref,
    )
    evaluation = DeterministicPreferenceEvaluator(
        snapshot.segments[0].segment_id,
        policy_version="source-policy-v1",
    ).evaluate(snapshot_to_evaluation_request(
        snapshot,
        evaluation_id="evaluation-natural-exit",
    ))
    plan = WritebackCoordinator().create_plans(snapshot, evaluation)[0]
    return snapshot, plan


def _request(
    policy_version="policy-v1",
    framework_version="framework-v1",
    *,
    experience=None,
) -> SemanticIngestRequest:
    snapshot, plan = _ingestion_fixture()
    return build_semantic_ingest_request(
        snapshot,
        plan,
        experience or _experience(snapshot),
        policy_version=policy_version,
        framework_version=framework_version,
    )


def _descriptor(
    provider="fake_semantic",
    policy_version="policy-v1",
    *,
    operations=frozenset(InternalMemoryAction),
    add_time_update=True,
) -> SemanticPolicyDescriptor:
    return SemanticPolicyDescriptor(
        provider=provider,
        policy_version=policy_version,
        framework_version="framework-v1",
        prompt_version="prompt-v1",
        feature_schema_version="features-v1",
        capability=PolicyCapability(operations, add_time_update),
    )


def _coordinator(policy) -> SemanticIngestionCoordinator:
    registry = SemanticPolicyRegistry()
    registry.register(policy)
    return SemanticIngestionCoordinator(registry, provider=policy.descriptor.provider)


def test_fixed_router_preserves_three_native_routes_and_only_enables_semantic() -> None:
    router = FixedMemoryRouter()
    assert set(HERMES_NATIVE_ROUTES) == set(MemoryKind)
    assert router.semantic.backend == "hermes-native-semantic"
    assert router.semantic.access_mode == MemoryAccessMode.EAGER
    assert router.semantic.policy_enabled is True
    assert router.resolve(MemoryKind.EPISODIC).backend == "hermes-native-episodic"
    assert router.resolve(MemoryKind.PROCEDURAL).backend == "hermes-native-procedural"
    assert not router.resolve(MemoryKind.EPISODIC).policy_enabled
    assert not router.resolve(MemoryKind.PROCEDURAL).policy_enabled

    with pytest.raises(ValueError, match="requires semantic, episodic, and procedural"):
        FixedMemoryRouter({MemoryKind.SEMANTIC: router.semantic})


def test_external_contract_has_fixed_route_and_no_operation_or_target_fields() -> None:
    request = _request()
    assert isinstance(request, MemoryIngestRequest)
    assert request.fixed_route == HERMES_NATIVE_ROUTES[MemoryKind.SEMANTIC]
    assert request.trigger == EvaluationTrigger.TASK_COMPLETED
    assert set(request.__dataclass_fields__) == {
        "source_experience",
        "source_projection",
        "fixed_route",
        "exit_evidence",
        "scope",
        "validity",
        "policy_version",
        "framework_version",
        "provenance",
        "idempotency_key",
        "trigger",
        "schema_version",
    }
    with pytest.raises(TypeError):
        replace(request, action="add")
    with pytest.raises(TypeError):
        InternalOperationProposal(
            InternalMemoryAction.ADD,
            "candidate_fact",
            new_content_digest=_sha("fact"),
            backend="forged",
        )


def test_request_rejects_route_override_hidden_score_and_operation_metadata() -> None:
    request = _request()
    episodic = FixedMemoryRoute(
        MemoryKind.EPISODIC,
        "hermes-native-episodic",
        MemoryAccessMode.SEARCH,
        False,
    )
    with pytest.raises(ValueError, match="fixed Hermes semantic route"):
        replace(request, fixed_route=episodic)
    with pytest.raises(ValueError, match="score cannot enter"):
        replace(
            request,
            source_experience=replace(request.source_experience, score=1.0),
        )
    with pytest.raises(ValueError, match="cannot predeclare"):
        replace(
            request,
            source_experience=replace(
                request.source_experience,
                metadata={"target": "artifact-forged"},
            ),
        )
    with pytest.raises(ValueError, match="requires task_completed"):
        replace(request, trigger=EvaluationTrigger.TURN_INTERVAL)


def test_trusted_builder_rejects_active_unresolved_and_open_tool_context() -> None:
    snapshot, plan = _ingestion_fixture()
    experience = _experience(snapshot)
    with pytest.raises(ValueError, match="requires completed task"):
        build_semantic_ingest_request(
            replace(snapshot, task_state=TaskLifecycleState.ACTIVE),
            plan,
            experience,
            policy_version="policy-v1",
            framework_version="framework-v1",
        )
    with pytest.raises(ValueError, match="requires completed task"):
        build_completed_task_semantic_ingest_request(
            replace(snapshot, task_state=TaskLifecycleState.FAILED),
            policy_version="policy-v1",
            framework_version="framework-v1",
        )
    unresolved_segments = (
        replace(snapshot.segments[0], completed=False),
        *snapshot.segments[1:],
    )
    with pytest.raises(ValueError, match="unresolved"):
        build_semantic_ingest_request(
            replace(snapshot, segments=unresolved_segments),
            plan,
            experience,
            policy_version="policy-v1",
            framework_version="framework-v1",
        )
    with pytest.raises(ValueError, match="unresolved"):
        build_completed_task_semantic_ingest_request(
            replace(snapshot, segments=unresolved_segments),
            policy_version="policy-v1",
            framework_version="framework-v1",
        )

    with pytest.raises(ValueError, match="active/current context"):
        build_completed_task_semantic_ingest_request(
            replace(snapshot, current_turn_id=snapshot.segments[0].turn_id),
            policy_version="policy-v1",
            framework_version="framework-v1",
        )

    open_snapshot = HermesSnapshotCollector().collect(
        (
            HermesMessage("user", "user", "Completed preference.", "turn-1", 3, completed=True),
            HermesMessage(
                "call", "assistant", '{"name":"tool"}', "turn-1", 3,
                kind=SegmentKind.TOOL_CALL, completed=True, tool_call_id="call-1",
            ),
        ),
        run_id="run",
        episode_id="episode",
        session_id=snapshot.session_id,
        task_id=snapshot.task_id,
        current_turn_id=None,
        task_state=TaskLifecycleState.COMPLETED,
        lifecycle_state="task_completed",
        source_ref="fixture:open-tool",
    )
    with pytest.raises(ValueError, match="open tool closure"):
        build_semantic_ingest_request(
            open_snapshot,
            plan,
            experience,
            policy_version="policy-v1",
            framework_version="framework-v1",
        )
    with pytest.raises(ValueError, match="open tool closure"):
        build_completed_task_semantic_ingest_request(
            open_snapshot,
            policy_version="policy-v1",
            framework_version="framework-v1",
        )


def test_fake_and_mem0_flat_are_interchangeable_behind_policy_interface() -> None:
    decision = SemanticPolicyDecision(
        MemoryIngestStatus.SUCCESS,
        (InternalOperationProposal(
            InternalMemoryAction.ADD,
            "candidate_fact",
            new_content_digest=_sha("durable fact"),
        ),),
    )
    fake = BoundSemanticPolicy(_descriptor(), lambda request, candidates: decision)
    mem0 = mem0_flat_policy(
        lambda request, candidates: decision,
        policy_version="policy-v1",
        framework_version="framework-v1",
        prompt_version="prompt-v1",
        feature_schema_version="features-v1",
    )

    fake_result = _coordinator(fake).ingest(_request(), _Candidates())
    mem0_result = _coordinator(mem0).ingest(_request(), _Candidates())
    assert fake_result is not None and mem0_result is not None
    assert fake_result.operations[0].action == InternalMemoryAction.ADD
    assert mem0_result.operations[0].action == InternalMemoryAction.ADD
    assert fake_result.policy_provider == "fake_semantic"
    assert mem0_result.policy_provider == "mem0_flat"
    assert fake_result.execution_id != mem0_result.execution_id


def test_update_and_delete_targets_are_bound_by_trusted_candidate_reader() -> None:
    candidate = ExistingMemoryCandidate(
        "candidate-1",
        "artifact-real",
        "revision-7",
        _sha("old fact"),
    )
    decision = SemanticPolicyDecision(
        MemoryIngestStatus.SUCCESS,
        (
            InternalOperationProposal(
                InternalMemoryAction.UPDATE,
                "superseded_fact",
                candidate_id="candidate-1",
                new_content_digest=_sha("new fact"),
            ),
            InternalOperationProposal(
                InternalMemoryAction.DELETE,
                "expired_fact",
                candidate_id="candidate-1",
            ),
        ),
    )
    policy = BoundSemanticPolicy(_descriptor(), lambda request, candidates: decision)
    result = _coordinator(policy).ingest(_request(), _Candidates((candidate,)))
    assert result is not None
    assert [item.target_artifact_id for item in result.operations] == [
        "artifact-real",
        "artifact-real",
    ]
    assert [item.expected_revision for item in result.operations] == [
        "revision-7",
        "revision-7",
    ]
    assert result.operations[0].old_content_digest == _sha("old fact")
    assert all(item.transaction_required for item in result.operations)
    assert all(item.recovery_receipt_required for item in result.operations)


def test_unknown_stale_duplicate_and_unsupported_operations_fail_closed() -> None:
    candidate = ExistingMemoryCandidate(
        "candidate-1", "artifact-real", "revision-1", _sha("old"),
    )
    update = InternalOperationProposal(
        InternalMemoryAction.UPDATE,
        "update_fact",
        candidate_id="candidate-1",
        new_content_digest=_sha("new"),
    )
    policy = BoundSemanticPolicy(
        _descriptor(),
        lambda request, candidates: SemanticPolicyDecision(
            MemoryIngestStatus.SUCCESS, (update,),
        ),
    )
    with pytest.raises(ValueError, match="unknown candidate"):
        _coordinator(policy).ingest(_request(), _Candidates())
    stale = replace(candidate, revision="revision-2")
    with pytest.raises(ValueError, match="ownership or revision is stale"):
        _coordinator(policy).ingest(
            _request(),
            _Candidates((candidate,), resolved={"candidate-1": stale}),
        )

    ambiguous = ExistingMemoryCandidate(
        "candidate-1", "artifact-other", "revision-1", _sha("other"),
    )
    with pytest.raises(ValueError, match="ambiguous candidate"):
        _coordinator(policy).ingest(
            _request(),
            _Candidates((candidate, ambiguous), resolved={"candidate-1": ambiguous}),
        )

    duplicate_policy = BoundSemanticPolicy(
        _descriptor(),
        lambda request, candidates: SemanticPolicyDecision(
            MemoryIngestStatus.SUCCESS, (update, update),
        ),
    )
    with pytest.raises(ValueError, match="duplicate operation"):
        _coordinator(duplicate_policy).ingest(_request(), _Candidates((candidate,)))

    distinct_adds = (
        InternalOperationProposal(
            InternalMemoryAction.ADD,
            "new_fact",
            new_content_digest=_sha("first fact"),
        ),
        InternalOperationProposal(
            InternalMemoryAction.ADD,
            "new_fact",
            new_content_digest=_sha("second fact"),
        ),
    )
    multi_fact_policy = BoundSemanticPolicy(
        _descriptor(),
        lambda request, candidates: SemanticPolicyDecision(
            MemoryIngestStatus.SUCCESS,
            (*distinct_adds, *(InternalOperationProposal(
                InternalMemoryAction.NONE,
                "duplicate_fact",
            ) for _ in range(2))),
        ),
    )
    multi_fact = _coordinator(multi_fact_policy).ingest(_request(), _Candidates())
    assert multi_fact is not None
    assert tuple(operation.action for operation in multi_fact.operations) == (
        InternalMemoryAction.ADD,
        InternalMemoryAction.ADD,
        InternalMemoryAction.NONE,
        InternalMemoryAction.NONE,
    )

    duplicate_add_policy = BoundSemanticPolicy(
        _descriptor(),
        lambda request, candidates: SemanticPolicyDecision(
            MemoryIngestStatus.SUCCESS,
            (distinct_adds[0], distinct_adds[0]),
        ),
    )
    with pytest.raises(ValueError, match="duplicate operation"):
        _coordinator(duplicate_add_policy).ingest(_request(), _Candidates())

    limited = BoundSemanticPolicy(
        _descriptor(
            operations=frozenset({InternalMemoryAction.ADD, InternalMemoryAction.NONE}),
            add_time_update=False,
        ),
        lambda request, candidates: SemanticPolicyDecision(
            MemoryIngestStatus.SUCCESS, (update,),
        ),
    )
    with pytest.raises(ValueError, match="unsupported operation"):
        _coordinator(limited).ingest(_request(), _Candidates((candidate,)))


def test_none_failure_idempotency_and_disabled_mode_have_distinct_semantics() -> None:
    calls = 0

    def decide(request, candidates):
        nonlocal calls
        calls += 1
        return SemanticPolicyDecision(
            MemoryIngestStatus.SUCCESS,
            (InternalOperationProposal(InternalMemoryAction.NONE, "duplicate_fact"),),
        )

    policy = BoundSemanticPolicy(_descriptor(), decide)
    coordinator = _coordinator(policy)
    request = _request()
    first = coordinator.ingest(request, _Candidates())
    replay = coordinator.ingest(request, _Candidates())
    assert first is replay
    assert calls == 1
    assert first is not None and first.status == MemoryIngestStatus.SUCCESS
    assert first.operations[0].action == InternalMemoryAction.NONE
    assert first.operations[0].transaction_required is False
    assert first.operations[0].recovery_receipt_required is False

    failed_policy = BoundSemanticPolicy(
        _descriptor(provider="failed_semantic"),
        lambda request, candidates: SemanticPolicyDecision(
            MemoryIngestStatus.FAILED, (), reason_codes=("framework_exception",),
        ),
    )
    failed = _coordinator(failed_policy).ingest(request, _Candidates())
    assert failed is not None and failed.status == MemoryIngestStatus.FAILED
    assert failed.outcome == MemoryIngestOutcome.FAILED
    assert failed.operations == ()

    disabled = SemanticIngestionCoordinator(
        SemanticPolicyRegistry(),
        provider="not-resolved",
        enabled=False,
    )
    assert disabled.ingest(request, _Candidates()) is None


def test_policy_version_changes_decision_identity_not_source_snapshot() -> None:
    fixture = run_sm01_preference_fixture()
    source_snapshot_id = fixture.snapshot.snapshot_id
    decision = SemanticPolicyDecision(
        MemoryIngestStatus.SUCCESS,
        (InternalOperationProposal(InternalMemoryAction.NONE, "no_change"),),
    )
    results = []
    for version in ("policy-v1", "policy-v2"):
        policy = BoundSemanticPolicy(
            _descriptor(policy_version=version),
            lambda request, candidates: decision,
        )
        results.append(_coordinator(policy).ingest(_request(version), _Candidates()))
    assert all(result is not None for result in results)
    assert results[0].execution_id != results[1].execution_id
    assert fixture.snapshot.snapshot_id == source_snapshot_id


def test_deterministic_request_is_restart_stable_across_coordinators() -> None:
    decision = SemanticPolicyDecision(
        MemoryIngestStatus.SUCCESS,
        (InternalOperationProposal(
            InternalMemoryAction.ADD,
            "new_fact",
            new_content_digest=_sha("fact"),
        ),),
    )
    policy = BoundSemanticPolicy(_descriptor(), lambda request, candidates: decision)
    request = _request()
    first = _coordinator(policy).ingest(request, _Candidates())
    second = _coordinator(policy).ingest(request, _Candidates())
    assert first is not None and second is not None
    assert first.execution_id == second.execution_id
    assert first.operations[0].operation_id == second.operations[0].operation_id
    assert first.source_digest == second.source_digest


def test_unknown_provider_policy_mismatch_and_non_natural_exit_fail_closed() -> None:
    registry = SemanticPolicyRegistry()
    with pytest.raises(KeyError, match="unknown semantic policy provider"):
        SemanticIngestionCoordinator(registry, provider="missing").ingest(
            _request(), _Candidates(),
        )
    policy = BoundSemanticPolicy(_descriptor(), lambda request, candidates: None)
    registry.register(policy)
    with pytest.raises(ValueError, match="policy version differs"):
        SemanticIngestionCoordinator(registry, provider="fake_semantic").ingest(
            _request("another-version"), _Candidates(),
        )
    with pytest.raises(ValueError, match="natural context exit only"):
        SemanticIngestionCoordinator(
            registry,
            provider="fake_semantic",
            exit_semantics=ContextExitSemantics.PHYSICAL,
        )


def test_request_identity_is_order_stable_and_covers_framework_evidence() -> None:
    first_experience = replace(
        _experience(),
        metadata={"fixture": {"z": 2, "a": 1}},
    )
    second_experience = replace(
        _experience(),
        metadata={"fixture": {"a": 1, "z": 2}},
    )
    first = _request(experience=first_experience)
    second = _request(experience=second_experience)
    assert first.idempotency_key == second.idempotency_key
    assert first.canonical_payload() == second.canonical_payload()

    changed_source = _request(experience=replace(
        second_experience,
        metadata={"fixture": {"a": 1, "z": 3}},
    ))
    assert changed_source.idempotency_key == first.idempotency_key

    changed_evidence = replace(
        first.exit_evidence,
        reusable_facts=("durable preference candidate",),
    )
    rebound = SemanticIngestRequest.create(
        source_experience=first.source_experience,
        source_projection=first.source_projection,
        fixed_route=first.fixed_route,
        exit_evidence=changed_evidence,
        scope=first.scope,
        validity=first.validity,
        policy_version=first.policy_version,
        framework_version=first.framework_version,
        provenance=first.provenance,
        trigger=first.trigger,
    )
    assert rebound.idempotency_key != first.idempotency_key

    framework_changed = _request(framework_version="framework-v2")
    assert framework_changed.idempotency_key != _request().idempotency_key


def test_request_rejects_missing_versions_noncanonical_identity_and_stale_source() -> None:
    request = _request()
    assert INGESTION_CONTRACT_SCHEMA_VERSION == 1
    with pytest.raises(ValueError, match="unsupported semantic ingest request schema"):
        replace(request, schema_version=2)
    with pytest.raises(ValueError, match="policy, framework"):
        replace(request, framework_version="")
    with pytest.raises(ValueError, match="compilation_id and base_revision"):
        replace(request, provenance=replace(request.provenance, base_revision=""))
    with pytest.raises(ValueError, match="experience_id"):
        replace(
            request,
            source_experience=replace(request.source_experience, experience_id=""),
        )
    with pytest.raises(ValueError, match="not canonical"):
        replace(request, idempotency_key="forged")
    with pytest.raises(TypeError):
        InternalOperationProposal(
            InternalMemoryAction.ADD,
            "new_fact",
            new_content_digest=_sha("fact"),
            resources=[{"path": "/forged"}],
        )
    with pytest.raises(TypeError):
        InternalOperationProposal(
            InternalMemoryAction.ADD,
            "new_fact",
            new_content_digest=_sha("fact"),
            openai_messages=[{"role": "system"}],
        )

    policy = BoundSemanticPolicy(
        _descriptor(),
        lambda request, candidates: SemanticPolicyDecision(
            MemoryIngestStatus.SUCCESS,
            (InternalOperationProposal(InternalMemoryAction.NONE, "no_change"),),
        ),
    )
    with pytest.raises(ValueError, match="source snapshot is stale"):
        _coordinator(policy).ingest(
            request,
            _Candidates(),
            current_source_revision="rev_newer",
        )


@pytest.mark.parametrize(
    ("error", "status", "outcome", "reason_code"),
    (
        (
            TimeoutError("SENTINEL_PRIVATE_TIMEOUT"),
            MemoryIngestStatus.FAILED,
            MemoryIngestOutcome.FAILED,
            "policy_timeout",
        ),
        (
            InvalidPolicyOutputError("SENTINEL_INVALID_JSON"),
            MemoryIngestStatus.REJECTED,
            MemoryIngestOutcome.REJECTED,
            "invalid_policy_json",
        ),
        (
            RuntimeError("SENTINEL_PRIVATE_EXCEPTION"),
            MemoryIngestStatus.FAILED,
            MemoryIngestOutcome.FAILED,
            "policy_exception",
        ),
    ),
)
def test_policy_failures_are_structured_without_source_or_exception_text(
    error,
    status,
    outcome,
    reason_code,
) -> None:
    def fail(request, candidates):
        raise error

    policy = BoundSemanticPolicy(_descriptor(), fail)
    result = _coordinator(policy).ingest(_request(), _Candidates())
    assert result is not None
    assert result.status == status
    assert result.outcome == outcome
    assert result.reason_codes == (reason_code,)
    assert result.operations == ()
    evidence = repr(result.observer_evidence())
    assert "SENTINEL" not in evidence
    assert "Always use TSV" not in evidence


def test_structured_policy_error_preserves_content_free_usage() -> None:
    usage = RawResourceUsage(
        input_tokens=11,
        output_tokens=2,
        model_requests=1,
        duration_ms=19,
    )

    def fail(request, candidates):
        raise PolicyExecutionError("provider_timeout", usage)

    policy = BoundSemanticPolicy(_descriptor(), fail)
    result = _coordinator(policy).ingest(_request(), _Candidates())
    assert result is not None
    assert result.status == MemoryIngestStatus.FAILED
    assert result.reason_codes == ("provider_timeout",)
    assert result.usage == usage


@pytest.mark.parametrize(
    "proposal",
    (
        InternalOperationProposal(
            InternalMemoryAction.ADD,
            "new_fact",
            new_content_digest=_sha("new"),
        ),
        InternalOperationProposal(
            InternalMemoryAction.UPDATE,
            "updated_fact",
            candidate_id="candidate-1",
            new_content_digest=_sha("updated"),
        ),
        InternalOperationProposal(
            InternalMemoryAction.DELETE,
            "expired_fact",
            candidate_id="candidate-1",
        ),
        InternalOperationProposal(InternalMemoryAction.NONE, "duplicate_fact"),
    ),
)
def test_deterministic_pass_through_ingestor_supports_all_internal_actions_without_mutation(
    tmp_path,
    proposal,
) -> None:
    memories = tmp_path / "memories"
    memories.mkdir()
    memory_file = memories / "MEMORY.md"
    memory_file.write_text("existing native memory\n", encoding="utf-8")
    before = memory_file.read_bytes()
    backend = HermesSemanticBackend(memories)

    usage = RawResourceUsage(duration_ms=3)
    ingestor = DeterministicPassThroughIngestor((proposal,), usage=usage)
    registry = SemanticPolicyRegistry()
    registry.register(ingestor)
    coordinator = SemanticIngestionCoordinator(
        registry,
        provider=ingestor.descriptor.provider,
    )
    candidate = ExistingMemoryCandidate(
        "candidate-1",
        "artifact-real",
        "revision-1",
        _sha("old"),
    )
    request = _request(
        ingestor.descriptor.policy_version,
        ingestor.descriptor.framework_version,
    )
    result = coordinator.ingest(request, _Candidates((candidate,)))
    replay = SemanticIngestionCoordinator(
        registry,
        provider=ingestor.descriptor.provider,
    ).ingest(request, _Candidates((candidate,)))

    assert result is not None and replay is not None
    assert ingestor.fixture_only is True
    assert result.execution_id == replay.execution_id
    assert result.operations[0].operation_id == replay.operations[0].operation_id
    assert result.operations[0].action == proposal.action
    assert result.usage == usage
    assert result.outcome == (
        MemoryIngestOutcome.NO_CHANGE
        if proposal.action == InternalMemoryAction.NONE
        else MemoryIngestOutcome.PLANNED_MUTATION
    )
    expected_digests = {
        InternalMemoryAction.ADD: (_sha("new"),),
        InternalMemoryAction.UPDATE: (_sha("old"), _sha("updated")),
        InternalMemoryAction.DELETE: (_sha("old"),),
        InternalMemoryAction.NONE: (),
    }
    assert result.content_digests == expected_digests[proposal.action]
    assert memory_file.read_bytes() == before
    assert backend.get("missing-artifact") is None


def test_ingestion_success_and_failure_usage_enter_content_free_ledger(tmp_path) -> None:
    usage = RawResourceUsage(
        input_tokens=13,
        output_tokens=5,
        cache_read_tokens=2,
        cache_write_tokens=1,
        reasoning_tokens=3,
        model_requests=1,
        retry_count=1,
        duration_ms=21,
        storage_bytes=0,
    )
    ingestor = DeterministicPassThroughIngestor(
        (InternalOperationProposal(InternalMemoryAction.NONE, "duplicate_fact"),),
        usage=usage,
    )
    registry = SemanticPolicyRegistry()
    registry.register(ingestor)
    request = _request(
        ingestor.descriptor.policy_version,
        ingestor.descriptor.framework_version,
    )
    result = SemanticIngestionCoordinator(
        registry,
        provider=ingestor.descriptor.provider,
    ).ingest(request, _Candidates())
    assert result is not None

    path = tmp_path / "lifecycle.jsonl"
    observer = LifecycleLedgerObserver(
        variant="fixture",
        trace_id="trace-ingestion",
        family_id="SM01",
        stage="learn",
        output_path=path,
    )
    observer.record_ingestion(request, result)
    event = observer.events[0]
    assert event["kind"] == "memory_ingestion"
    assert event["data"]["resources"] == {
        "schemaVersion": 1,
        "inputTokens": 13,
        "outputTokens": 5,
        "cacheReadTokens": 2,
        "cacheWriteTokens": 1,
        "reasoningTokens": 3,
        "modelRequests": 1,
        "retryCount": 1,
        "durationMs": 21,
        "storageBytes": 0,
    }
    serialized = path.read_text(encoding="utf-8")
    assert "Always use TSV" not in serialized
    assert "existing native memory" not in serialized

    failure_usage = RawResourceUsage(
        input_tokens=7,
        output_tokens=0,
        model_requests=1,
        duration_ms=8,
    )

    def fail(failed_request, candidates):
        raise PolicyExecutionError("provider_timeout", failure_usage)

    failed_policy = BoundSemanticPolicy(_descriptor(), fail)
    failed_request = _request()
    failed = _coordinator(failed_policy).ingest(failed_request, _Candidates())
    assert failed is not None
    observer.record_ingestion(failed_request, failed)
    assert observer.events[-1]["data"]["resources"]["inputTokens"] == 7
    assert observer.events[-1]["data"]["status"] == "failed"
    summary = summarize_ingestion_usage(list(observer.events))
    assert summary["uniqueExecutions"] == 2
    assert summary["inputTokens"] == 20
    assert summary["modelRequests"] == 2
    assert summary["retries"] == 1
    assert summary["statuses"] == {"success": 1, "failed": 1}

    duplicate_summary = summarize_ingestion_usage([
        *observer.events,
        observer.events[0],
    ])
    assert duplicate_summary["duplicateViews"] == 1
    conflict = copy.deepcopy(observer.events[0])
    conflict["data"]["resources"]["inputTokens"] = 99
    with pytest.raises(ValueError, match="conflicting memory ingestion execution"):
        summarize_ingestion_usage([observer.events[0], conflict])

    restarted = LifecycleLedgerObserver(
        variant="fixture",
        trace_id="trace-ingestion",
        family_id="SM01",
        stage="learn",
        output_path=path,
    )
    restarted.record_ingestion(request, result)
    restarted.record_ingestion(failed_request, failed)
    assert len(restarted.events) == 2
