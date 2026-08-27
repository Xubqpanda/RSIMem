from __future__ import annotations

import hashlib
import re
from dataclasses import FrozenInstanceError, replace

import pytest

from rsimem.lifecycle import RawResourceUsage
from rsimem.memory import MemoryKind, MemoryQuery
from rsimem.memory.backends import HermesSemanticBackend
from rsimem.memory.ingestion import (
    InternalMemoryAction,
    MemoryIngestStatus,
    SemanticIngestRequest,
    SemanticIngestionCoordinator,
    SemanticPolicyRegistry,
)
from rsimem.memory.operation_graph import AtomicOperationRecorder
from rsimem.memory.operation_graph import (
    AppendOnlyOperationEvidenceLog,
    ArtifactKind,
    OperationKind,
    materialize_operation_graph,
)
from rsimem.memory.utility import (
    MEM0_CONSOLIDATION_UPDATE_PARAMETER_ID,
    MEM0_UTILITY_PARAMETER_IDS,
    UtilityTarget,
)
from rsimem.memory.attribution import DeterministicFirstAttributor, FailureCategory
from rsimem.memory.runtime import MemoryBackendRegistry
from rsimem.memory.validation import (
    InMemoryTargetOwnershipRegistry,
    TrustedTargetBinding,
    ValidationProvenance,
)
from rsimem.memory_systems.mem0_flat import (
    FACT_EXTRACTION_PROMPT,
    INTERNAL_OPERATION_PROMPT,
    MEM0_FLAT_POLICY_SCHEMA_VERSION,
    FakeCompletionClient,
    FlatRetrievalConfig,
    FlatSemanticCandidateReader,
    Mem0FlatSemanticPolicy,
    FrozenMem0UtilityConfig,
    FrozenMem0UtilityGate,
    build_validation_candidate,
)
from rsimem.memory.utility import (
    StaticUtilityPolicy,
    UtilityDisposition,
    UtilityTarget,
)
from test_semantic_ingestion_contracts import _request


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _candidate_id(prompt_text: str) -> str:
    values = re.findall(r"candidate\.[0-9a-f]{40}", prompt_text)
    assert values
    return values[0]


def _operation_response(action: InternalMemoryAction, *, use_candidate: bool):
    def respond(prompt):
        candidate_id = _candidate_id(prompt.text) if use_candidate else None
        return (
            '{"operations": [{"fact_index": 0, "action": "'
            + action.value
            + '", "candidate_id": '
            + (f'"{candidate_id}"' if candidate_id is not None else "null")
            + "}]}"
        )

    return respond


def _setup(
    tmp_path,
    *,
    entries=(),
    owned=False,
    facts='{"facts": ["Always use TSV with four columns."]}',
    operation=None,
    usage=RawResourceUsage(input_tokens=7, output_tokens=3, model_requests=1, duration_ms=2),
    operation_recorder=None,
    utility_gate=None,
    policy_version=None,
):
    memories = tmp_path / "memories"
    memories.mkdir(parents=True)
    if entries:
        (memories / "USER.md").write_text("\n§\n".join(entries), encoding="utf-8")
    backend = HermesSemanticBackend(memories)
    registry = MemoryBackendRegistry()
    registry.register(backend)
    ownership = InMemoryTargetOwnershipRegistry()
    if owned:
        for hit in backend.query(MemoryQuery(
            MemoryKind.SEMANTIC,
            "",
            namespace="user",
            limit=100,
        )):
            artifact = hit.artifact
            ownership.register(TrustedTargetBinding(
                backend=backend.descriptor.name,
                artifact_id=artifact.artifact_id,
                revision=artifact.revision,
                kind=artifact.kind,
                namespace=artifact.namespace,
                content_digest=_sha(artifact.content),
                owner_run_id="run-sm01-fixture",
            ))
    responses = {FACT_EXTRACTION_PROMPT.artifact.prompt_id: facts}
    if operation is not None:
        responses[INTERNAL_OPERATION_PROMPT.artifact.prompt_id] = operation
    client = FakeCompletionClient(responses, usage=usage)
    policy = Mem0FlatSemanticPolicy(
        client,
        operation_recorder=operation_recorder,
        utility_gate=utility_gate,
        policy_version=policy_version,
    )
    request = _request(
        policy.descriptor.policy_version,
        policy.descriptor.framework_version,
    )
    reader = FlatSemanticCandidateReader(
        registry,
        ownership=ownership,
    )
    policies = SemanticPolicyRegistry()
    policies.register(policy)
    coordinator = SemanticIngestionCoordinator(
        policies,
        provider=policy.descriptor.provider,
    )
    return backend, registry, ownership, client, policy, request, reader, coordinator


def _provenance(result, request, operation_index=0):
    operation = result.operations[operation_index]
    source = request.provenance.source
    return ValidationProvenance(
        run_id=source.run_id,
        episode_id=source.episode_id,
        session_id=source.session_id,
        task_id=source.task_id,
        snapshot_id=source.snapshot_id,
        execution_id=result.execution_id,
        operation_id=operation.operation_id,
        source_digest=result.source_digest,
    )


def test_sm01_fact_extraction_generates_minimal_add_candidate(tmp_path) -> None:
    assert MEM0_FLAT_POLICY_SCHEMA_VERSION == 1
    (
        backend,
        _,
        _,
        client,
        policy,
        request,
        reader,
        coordinator,
    ) = _setup(
        tmp_path,
        operation=_operation_response(InternalMemoryAction.ADD, use_candidate=False),
    )
    result = coordinator.ingest(request, reader)
    assert result is not None and result.status == MemoryIngestStatus.SUCCESS
    assert len(result.operations) == 1
    assert result.operations[0].action == InternalMemoryAction.ADD
    assert result.usage.input_tokens == 14
    assert result.usage.output_tokens == 6
    assert result.usage.model_requests == 2
    assert len(client.calls) == 2
    assert backend.query(MemoryQuery(
        MemoryKind.SEMANTIC,
        "",
        namespace="user",
    )) == ()

    candidate = build_validation_candidate(
        result,
        0,
        policy,
        _provenance(result, request),
    )
    assert candidate.content == "Always use TSV with four columns."
    assert candidate.category.value == "preference"
    assert candidate.namespace == "user"
    assert candidate.metadata["source_operation_id"] == result.operations[0].operation_id


def test_flat_retrieval_contract_is_restart_stable_and_revision_bound(tmp_path) -> None:
    entry = "Always use CSV with four columns."
    backend, registry, ownership, _, policy, request, reader, _ = _setup(
        tmp_path,
        entries=(entry,),
        owned=True,
        operation=_operation_response(InternalMemoryAction.NONE, use_candidate=False),
    )
    config = FlatRetrievalConfig()
    assert config.embedding_model == "rsimem-token-hash-cosine-v1"
    assert config.top_k == 5
    assert config.threshold == 0.12
    assert config.rebuild_semantics == "snapshot-rebuild-per-ingest-v1"
    assert config.digest[:16] in policy.descriptor.policy_version

    first = reader.search(request, "Always use TSV with four columns.")
    restarted = FlatSemanticCandidateReader(
        registry,
        ownership=ownership,
        config=config,
    )
    second = restarted.search(request, "Always use TSV with four columns.")
    assert first == second
    assert reader.index_revision == restarted.index_revision
    assert first[0].mutable is True
    old_revision = reader.index_revision

    (tmp_path / "memories" / "USER.md").write_text(
        entry + "\n§\nA second durable preference.",
        encoding="utf-8",
    )
    rebuilt = FlatSemanticCandidateReader(registry, ownership=ownership, config=config)
    rebuilt.search(request, "Always use TSV with four columns.")
    assert rebuilt.index_revision != old_revision
    assert backend.get(first[0].candidate.artifact_id) is not None


def test_frozen_utility_gate_preserves_cadence_and_accepts_high_utility_add(
    tmp_path,
) -> None:
    gate = FrozenMem0UtilityGate()
    setup = _setup(
        tmp_path,
        operation=_operation_response(InternalMemoryAction.ADD, use_candidate=False),
        utility_gate=gate,
    )
    result = setup[-1].ingest(setup[5], setup[6])

    assert result is not None and result.status == MemoryIngestStatus.SUCCESS
    assert result.operations[0].action == InternalMemoryAction.ADD
    assert len(setup[3].calls) == 2
    decisions = gate.decisions(setup[5].idempotency_key)
    assert [item.target for item in decisions] == [
        UtilityTarget.GENERATION,
        UtilityTarget.INTERNAL_OPERATION,
    ]
    assert all(item.disposition == UtilityDisposition.ACCEPT for item in decisions)
    assert gate.feature_schema == result.feature_schema_version
    assert gate.digest[:16] in result.policy_version


def test_frozen_utility_gate_defers_low_utility_without_skipping_prompts(tmp_path) -> None:
    gate = FrozenMem0UtilityGate()
    setup = _setup(
        tmp_path,
        operation=_operation_response(InternalMemoryAction.ADD, use_candidate=False),
        utility_gate=gate,
    )
    request = setup[5]
    low_evidence = replace(
        request.exit_evidence,
        utility_estimate=0.1,
        confidence=0.9,
    )
    low_request = SemanticIngestRequest.create(
        source_experience=request.source_experience,
        source_projection=request.source_projection,
        fixed_route=request.fixed_route,
        exit_evidence=low_evidence,
        scope=request.scope,
        validity=request.validity,
        policy_version=request.policy_version,
        framework_version=request.framework_version,
        provenance=request.provenance,
        trigger=request.trigger,
    )
    result = setup[-1].ingest(low_request, setup[6])

    assert result is not None and result.status == MemoryIngestStatus.SUCCESS
    assert result.operations[0].action == InternalMemoryAction.NONE
    assert result.operations[0].reason_code == "utility_deferred"
    assert len(setup[3].calls) == 2
    assert [item.target for item in gate.decisions(low_request.idempotency_key)] == [
        UtilityTarget.GENERATION,
        UtilityTarget.INTERNAL_OPERATION,
    ]


def test_frozen_utility_gate_ranks_related_with_shared_objective(tmp_path) -> None:
    gate = FrozenMem0UtilityGate()
    high = "Always use TSV with four columns."
    medium = "Always use CSV with four columns."
    low = "Always " + " ".join(f"unrelated{index}" for index in range(10))
    observed_prompt = []

    def operation(prompt):
        observed_prompt.append(prompt.text)
        return _operation_response(
            InternalMemoryAction.NONE,
            use_candidate=False,
        )(prompt)

    setup = _setup(
        tmp_path,
        entries=(low, medium, high),
        owned=True,
        operation=operation,
        utility_gate=gate,
    )
    result = setup[-1].ingest(setup[5], setup[6])

    assert result is not None and len(setup[3].calls) == 2
    assert high in observed_prompt[0]
    assert medium in observed_prompt[0]
    assert low not in observed_prompt[0]
    retrieval = [
        item
        for item in gate.decisions(setup[5].idempotency_key)
        if item.target == UtilityTarget.RETRIEVAL
    ]
    accepted = [
        item for item in retrieval if item.disposition == UtilityDisposition.ACCEPT
    ]
    assert [item.score for item in accepted] == sorted(
        (item.score for item in accepted),
        reverse=True,
    )
    assert len(accepted) == 2
    assert [item.disposition for item in retrieval].count(
        UtilityDisposition.DEFER
    ) == 1
    assert all(item.feature_schema == gate.feature_schema for item in retrieval)
    evidence = repr(gate.observer_evidence(setup[5].idempotency_key))
    assert all(content not in evidence for content in (high, medium, low))


def test_frozen_utility_gate_replay_is_deterministic_and_content_free(tmp_path) -> None:
    gate = FrozenMem0UtilityGate()
    setup = _setup(
        tmp_path,
        entries=("Always use CSV with four columns.",),
        owned=True,
        operation=_operation_response(InternalMemoryAction.NONE, use_candidate=False),
        utility_gate=gate,
    )

    first = setup[4].ingest(setup[5], setup[6])
    first_evidence = gate.observer_evidence(setup[5].idempotency_key)
    second = setup[4].ingest(setup[5], setup[6])
    second_evidence = gate.observer_evidence(setup[5].idempotency_key)

    assert first == second
    assert first_evidence == second_evidence
    assert len(setup[3].calls) == 4
    assert "Always use CSV with four columns." not in repr(second_evidence)


def test_operation_graph_binds_stable_runtime_owned_utility_parameters(
    tmp_path,
) -> None:
    log = AppendOnlyOperationEvidenceLog()
    setup = _setup(
        tmp_path,
        operation=_operation_response(InternalMemoryAction.ADD, use_candidate=False),
        utility_gate=FrozenMem0UtilityGate(),
        operation_recorder=AtomicOperationRecorder(log),
    )
    setup[-1].ingest(setup[5], setup[6])
    graph = materialize_operation_graph(log.events)
    parameters = {
        artifact.artifact_id: artifact
        for artifact in graph.artifacts
        if artifact.kind == ArtifactKind.POLICY_PARAMETER
    }
    expected = {
        *MEM0_UTILITY_PARAMETER_IDS.values(),
        MEM0_CONSOLIDATION_UPDATE_PARAMETER_ID,
    }
    assert expected.issubset(parameters)
    assert all(parameters[value].provenance_ref == value for value in expected)
    by_kind = {operation.kind: operation for operation in graph.operations}
    assert MEM0_UTILITY_PARAMETER_IDS[UtilityTarget.GENERATION] in (
        by_kind[OperationKind.FACT_EXTRACTION].input_artifact_ids
    )
    assert MEM0_UTILITY_PARAMETER_IDS[UtilityTarget.RETRIEVAL] in (
        by_kind[OperationKind.RELATED_MEMORY_RETRIEVAL].input_artifact_ids
    )
    decision_inputs = by_kind[
        OperationKind.INTERNAL_OPERATION_DECISION
    ].input_artifact_ids
    assert MEM0_UTILITY_PARAMETER_IDS[UtilityTarget.INTERNAL_OPERATION] in (
        decision_inputs
    )
    assert MEM0_CONSOLIDATION_UPDATE_PARAMETER_ID in decision_inputs


def test_utility_policy_version_does_not_change_route_or_invocation_count(tmp_path) -> None:
    first = _setup(
        tmp_path / "first",
        operation=_operation_response(InternalMemoryAction.ADD, use_candidate=False),
        utility_gate=FrozenMem0UtilityGate(),
        policy_version="static-policy-a",
    )
    second = _setup(
        tmp_path / "second",
        operation=_operation_response(InternalMemoryAction.ADD, use_candidate=False),
        utility_gate=FrozenMem0UtilityGate(),
        policy_version="static-policy-b",
    )
    first_result = first[-1].ingest(first[5], first[6])
    second_result = second[-1].ingest(second[5], second[6])

    assert first_result.fixed_route == second_result.fixed_route
    assert [item.action for item in first_result.operations] == [
        item.action for item in second_result.operations
    ]
    assert len(first[3].calls) == len(second[3].calls) == 2
    assert first_result.policy_version != second_result.policy_version
    threshold = _setup(
        tmp_path / "threshold",
        operation=_operation_response(InternalMemoryAction.ADD, use_candidate=False),
        utility_gate=FrozenMem0UtilityGate(
            policy=StaticUtilityPolicy(accept_threshold=0.4),
        ),
    )
    assert threshold[4].descriptor.policy_version != first[4].descriptor.policy_version
    assert threshold[5].fixed_route == first[5].fixed_route
    with pytest.raises(FrozenInstanceError):
        FrozenMem0UtilityConfig().recency = 0.0
    with pytest.raises(FrozenInstanceError):
        first[4].utility_gate.config = FrozenMem0UtilityConfig(recency=0.5)
    with pytest.raises(FrozenInstanceError):
        first[4].utility_gate.policy = StaticUtilityPolicy(accept_threshold=0.4)


@pytest.mark.parametrize(
    ("action", "existing", "fact"),
    (
        (InternalMemoryAction.ADD, None, "Always use TSV with four columns."),
        (InternalMemoryAction.UPDATE, "Always use CSV with four columns.", "Always use TSV with four columns."),
        (InternalMemoryAction.DELETE, "Always use TSV with four columns.", "Delete my TSV preference."),
        (InternalMemoryAction.NONE, "Always use TSV with four columns.", "Always use TSV with four columns."),
    ),
)
def test_internal_operation_matrix_is_host_bound(tmp_path, action, existing, fact) -> None:
    entries = (existing,) if existing is not None else ()
    use_candidate = action in {InternalMemoryAction.UPDATE, InternalMemoryAction.DELETE}
    (
        _,
        _,
        _,
        _,
        policy,
        request,
        reader,
        coordinator,
    ) = _setup(
        tmp_path,
        entries=entries,
        owned=use_candidate,
        facts='{"facts": ["' + fact + '"]}',
        operation=_operation_response(action, use_candidate=use_candidate),
    )
    result = coordinator.ingest(request, reader)
    assert result is not None and result.status == MemoryIngestStatus.SUCCESS
    operation = result.operations[0]
    assert operation.action == action
    if use_candidate:
        assert operation.target_artifact_id is not None
        assert operation.expected_revision is not None
    else:
        assert operation.target_artifact_id is None
    if action == InternalMemoryAction.UPDATE:
        candidate = build_validation_candidate(
            result,
            0,
            policy,
            _provenance(result, request),
        )
        assert candidate.metadata["replaces_artifact_id"] == operation.target_artifact_id


def test_duplicate_native_memory_uses_none_without_claiming_ownership(tmp_path) -> None:
    entry = "Always use TSV with four columns."
    setup = _setup(
        tmp_path,
        entries=(entry,),
        owned=False,
        operation=_operation_response(InternalMemoryAction.NONE, use_candidate=False),
    )
    result = setup[-1].ingest(setup[5], setup[6])
    assert result is not None and result.status == MemoryIngestStatus.SUCCESS
    assert result.operations[0].action == InternalMemoryAction.NONE
    assert result.operations[0].target_artifact_id is None


@pytest.mark.parametrize(
    ("operation", "reason"),
    (
        (_operation_response(InternalMemoryAction.UPDATE, use_candidate=True), "unknown_owner_target"),
        (
            lambda prompt: '{"operations": [{"fact_index": 0, "action": "update", "candidate_id": "candidate.' + "0" * 40 + '"}]}',
            "hallucinated_candidate_target",
        ),
    ),
)
def test_unknown_owner_and_hallucinated_target_are_rejected(tmp_path, operation, reason) -> None:
    log = AppendOnlyOperationEvidenceLog()
    setup = _setup(
        tmp_path,
        entries=("Always use CSV with four columns.",),
        owned=False,
        operation=operation,
        operation_recorder=AtomicOperationRecorder(log),
    )
    result = setup[-1].ingest(setup[5], setup[6])
    assert result is not None and result.status == MemoryIngestStatus.REJECTED
    assert result.reason_codes == (reason,)
    assert result.operations == ()
    report = DeterministicFirstAttributor().attribute(
        materialize_operation_graph(log.events)
    )
    assert len(report.records) == 1
    assert report.records[0].category == FailureCategory.WRONG_UPDATE_TARGET
    assert report.records[0].policy_parameter_ids


def test_temporary_and_unresolved_sources_are_rejected_without_mutation(tmp_path) -> None:
    temporary = _setup(
        tmp_path / "temporary",
        facts='{"facts": ["For this task, use TSV temporarily."]}',
        operation=_operation_response(InternalMemoryAction.ADD, use_candidate=False),
    )
    result = temporary[-1].ingest(temporary[5], temporary[6])
    assert result is not None and result.status == MemoryIngestStatus.REJECTED
    assert result.reason_codes == ("non_durable_fact",)
    assert len(temporary[3].calls) == 1

    unresolved = _setup(
        tmp_path / "unresolved",
        operation=_operation_response(InternalMemoryAction.ADD, use_candidate=False),
    )
    request = unresolved[5]
    evidence = replace(request.exit_evidence, unresolved_state="pending_confirmation")
    unresolved_request = SemanticIngestRequest.create(
        source_experience=request.source_experience,
        source_projection=request.source_projection,
        fixed_route=request.fixed_route,
        exit_evidence=evidence,
        scope=request.scope,
        validity=request.validity,
        policy_version=request.policy_version,
        framework_version=request.framework_version,
        provenance=request.provenance,
        trigger=request.trigger,
    )
    result = unresolved[-1].ingest(unresolved_request, unresolved[6])
    assert result is not None and result.status == MemoryIngestStatus.REJECTED
    assert result.reason_codes == ("unresolved_source",)
    assert unresolved[3].calls == ()


@pytest.mark.parametrize(
    ("fact_response", "operation_response", "expected_reason", "status"),
    (
        ("not json", None, "invalid_policy_json", MemoryIngestStatus.REJECTED),
        (
            '{"facts": ["Always use TSV."]}',
            "not json",
            "invalid_policy_json",
            MemoryIngestStatus.REJECTED,
        ),
    ),
)
def test_malformed_prompt_output_fails_closed(
    tmp_path,
    fact_response,
    operation_response,
    expected_reason,
    status,
) -> None:
    setup = _setup(
        tmp_path,
        facts=fact_response,
        operation=operation_response,
    )
    result = setup[-1].ingest(setup[5], setup[6])
    assert result is not None and result.status == status
    assert result.reason_codes == (expected_reason,)


def test_completion_timeout_becomes_structured_failure(tmp_path) -> None:
    setup = _setup(
        tmp_path,
        facts=TimeoutError("SENTINEL_PRIVATE_TIMEOUT"),
        operation=_operation_response(InternalMemoryAction.ADD, use_candidate=False),
    )
    result = setup[-1].ingest(setup[5], setup[6])
    assert result is not None and result.status == MemoryIngestStatus.FAILED
    assert result.reason_codes == ("policy_timeout",)
    assert "SENTINEL" not in repr(result.observer_evidence())


def test_operation_observer_failure_does_not_change_mem0_policy_result(tmp_path) -> None:
    class FailingSink:
        def append(self, event):
            del event
            raise OSError("SENTINEL_PRIVATE_OBSERVER_FAILURE")

    operation = _operation_response(InternalMemoryAction.ADD, use_candidate=False)
    control = _setup(tmp_path / "control", operation=operation)
    recorder = AtomicOperationRecorder(FailingSink())
    observed = _setup(
        tmp_path / "observed",
        operation=operation,
        operation_recorder=recorder,
    )
    control_result = control[-1].ingest(control[5], control[6])
    observed_result = observed[-1].ingest(observed[5], observed[6])

    assert observed_result.observer_evidence() == control_result.observer_evidence()
    assert recorder.observer_failures
    assert recorder.attribution_gaps
    assert "SENTINEL" not in repr(recorder.observer_failures)
