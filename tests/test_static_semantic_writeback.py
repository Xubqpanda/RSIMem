from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rsimem.ledger import LifecycleLedgerObserver
from rsimem.lifecycle import (
    DryRunStatus,
    EvaluationTrigger,
    HermesLifecycleConfig,
    HermesLifecycleDryRunRuntime,
    TaskLifecycleState,
)
from rsimem.memory import (
    MemoryArtifact,
    MemoryKind,
    MemoryMutation,
    MemoryMutationAction,
    MemoryQuery,
)
from rsimem.memory.live_writeback import (
    MATCHED_EXTRACTION_CANDIDATE_ID,
    LEGACY_ADAPTIVE_UTILITY_ID,
    LEGACY_STATIC_UTILITY_ID,
    STATIC_EXTRACTION_PARENT_ID,
    STATIC_SEMANTIC_WRITEBACK_SCHEMA_VERSION,
    ExtractionPromptRuntimeScope,
    StaticSemanticWritebackConfig,
    StaticSemanticWritebackMode,
    StaticSemanticWritebackRuntime,
)
from rsimem.memory.adaptive_mem0_binding import TrustedAdaptiveMem0Parameter
from rsimem.memory.adaptive_policy import AdaptiveParameterName
from rsimem.memory.adaptive_policy_store import JsonAdaptivePolicyStore
from rsimem.memory.operation_graph import (
    ArtifactKind,
    audit_operation_evidence,
    materialize_operation_graph,
)
from rsimem.memory.receipts import MutationReceiptStatus
from rsimem.memory_systems.mem0_flat import (
    MEM0_FLAT_EXTRACTION_SLOT_ID,
    Mem0FlatPromptAdapter,
    FakeCompletionClient,
    FrozenMem0UtilityGate,
    POLICY_FACT_EXTRACTION_PROMPT,
    POLICY_INTERNAL_OPERATION_PROMPT,
)
from test_extraction_offline_validation import _candidate


PREFERENCE = "Use TSV with owner, priority, task, and due_date."


def _lifecycle(tmp_path):
    runtime = HermesLifecycleDryRunRuntime(
        HermesLifecycleConfig(evaluator_mode="deterministic"),
        run_id="run-static",
        episode_id="episode-static",
        session_id="session-static",
        task_id="SM01-static",
        variant="static-rsimem",
        trace_id="trace-static",
        receipt_path=tmp_path / "lifecycle-receipts.json",
        evidence_path=tmp_path / "lifecycle-evidence.jsonl",
        family_id="SM01_preference_adoption",
        stage="learn",
    )
    return runtime.process(
        (
            {"id": 1, "role": "user", "content": "Always use TSV output."},
            {"id": 2, "role": "assistant", "content": "Understood."},
        ),
        trigger=EvaluationTrigger.TASK_COMPLETED,
        task_state=TaskLifecycleState.COMPLETED,
        source_ref="hermes_state:session:static",
    )


def _client(*, fact_response: str | None = None) -> FakeCompletionClient:
    return FakeCompletionClient({
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
    })


def _runtime(tmp_path, client):
    return StaticSemanticWritebackRuntime(
        tmp_path / "hermes-home",
        client,
        operation_evidence_path=tmp_path / "episode" / "operations.jsonl",
        mutation_receipt_path=tmp_path / "hermes-home" / "rsimem-receipts.json",
    )


def test_static_config_is_default_disabled_and_strict() -> None:
    assert STATIC_SEMANTIC_WRITEBACK_SCHEMA_VERSION == 2
    assert StaticSemanticWritebackConfig().enabled is False
    assert StaticSemanticWritebackConfig.from_mapping({
        "mode": "static",
        "timeout_seconds": 12,
        "max_output_tokens": 512,
    }).mode == StaticSemanticWritebackMode.STATIC
    plain = StaticSemanticWritebackConfig.from_mapping({"mode": "static"})
    assert plain.plain_extraction_parent is True
    assert plain.utility_enabled is False
    assert plain.adaptive_enabled is False
    assert plain.method_identity == STATIC_EXTRACTION_PARENT_ID
    matched = StaticSemanticWritebackConfig.from_mapping({
        "mode": "static",
        "extraction_runtime_scope": "matched_validation",
        "extraction_runtime_config_path": "/attempt/extraction-trial.json",
    })
    assert matched.matched_extraction_enabled is True
    assert matched.plain_extraction_parent is False
    assert matched.method_identity == MATCHED_EXTRACTION_CANDIDATE_ID
    offline = StaticSemanticWritebackConfig.from_mapping({
        "mode": "static",
        "extraction_runtime_scope": "offline_validation",
        "extraction_runtime_config_path": "/attempt/extraction-offline-validation.json",
    })
    assert offline.offline_extraction_enabled is True
    assert offline.validated_extraction_enabled is True
    assert offline.plain_extraction_parent is False
    assert offline.method_identity == MATCHED_EXTRACTION_CANDIDATE_ID
    utility = StaticSemanticWritebackConfig.from_mapping({
        "mode": "static_utility",
    })
    assert utility.enabled is True
    assert utility.utility_enabled is True
    assert utility.plain_extraction_parent is False
    assert utility.method_identity == LEGACY_STATIC_UTILITY_ID
    adaptive = StaticSemanticWritebackConfig.from_mapping({
        "mode": "adaptive_utility",
        "adaptive_policy_store_path": ".rsimem/adaptive-policies.json",
        "adaptive_trusted_roots": ["mem0-flat.parent-v1"],
        "adaptive_parameters": [{
            "parameter_id": "parameter.retrieval",
            "name": "retrieval_accept_threshold",
            "prompt_ref": "mem0-flat.retrieval",
            "baseline_value": 0.35,
        }],
    })
    assert adaptive.adaptive_enabled is True
    assert adaptive.utility_enabled is True
    assert adaptive.method_identity == LEGACY_ADAPTIVE_UTILITY_ID
    feedback = StaticSemanticWritebackConfig.from_mapping({
        "mode": "static_utility",
        "feedback_contract": "sm01_tsv_v1",
    })
    assert feedback.feedback_contract.value == "sm01_tsv_v1"
    sm03_feedback = StaticSemanticWritebackConfig.from_mapping({
        "mode": "static",
        "feedback_contract": "sm03_fact_correction_v1",
    })
    assert sm03_feedback.feedback_contract.value == "sm03_fact_correction_v1"
    with pytest.raises(ValueError, match="configuration is incomplete"):
        StaticSemanticWritebackConfig.from_mapping({"mode": "adaptive_utility"})
    with pytest.raises(ValueError, match="require adaptive_utility"):
        StaticSemanticWritebackConfig.from_mapping({
            "mode": "static_utility",
            "adaptive_policy_store_path": "policy.json",
        })
    with pytest.raises(ValueError, match="unknown static semantic"):
        StaticSemanticWritebackConfig.from_mapping({"provider_seed": 7})
    with pytest.raises(ValueError, match="requires writeback mode"):
        StaticSemanticWritebackConfig.from_mapping({
            "mode": "disabled",
            "feedback_contract": "sm01_tsv_v1",
        })
    with pytest.raises(ValueError, match="explicit config path"):
        StaticSemanticWritebackConfig.from_mapping({
            "mode": "static",
            "extraction_runtime_scope": "matched_validation",
        })
    with pytest.raises(ValueError, match="plain static writeback"):
        StaticSemanticWritebackConfig.from_mapping({
            "mode": "static_utility",
            "extraction_runtime_scope": "matched_validation",
            "extraction_runtime_config_path": "/attempt/extraction-trial.json",
        })
    with pytest.raises(ValueError, match="requires matched_validation"):
        StaticSemanticWritebackConfig.from_mapping({
            "mode": "static",
            "extraction_runtime_config_path": "/attempt/extraction-trial.json",
        })


def test_matched_extraction_runtime_binds_candidate_and_is_restart_stable(
    tmp_path,
) -> None:
    parent = Mem0FlatPromptAdapter().export_root_policy_artifact(
        MEM0_FLAT_EXTRACTION_SLOT_ID
    )
    candidate = _candidate(parent=parent)
    bindings = []
    clients = []
    for name in ("first", "restart"):
        client = _client()
        clients.append(client)
        runtime = StaticSemanticWritebackRuntime(
            tmp_path / name / "hermes-home",
            client,
            operation_evidence_path=tmp_path / name / "operations.jsonl",
            mutation_receipt_path=tmp_path / name / "receipts.json",
            extraction_policy_artifact=candidate,
            expected_extraction_policy_artifact_id=candidate.artifact_id,
            expected_extraction_policy_artifact_digest=candidate.artifact_digest,
            extraction_runtime_scope=(
                ExtractionPromptRuntimeScope.MATCHED_VALIDATION
            ),
            extraction_trial_id="extraction-trial.test-restart",
        )
        result = runtime.process(_lifecycle(tmp_path / name))[0]
        assert result.writeback is not None
        assert client.calls[0]["binding_fingerprint"] == (
            runtime.extraction_runtime_binding.binding_id
        )
        assert runtime.policy.semantic_manifest.extraction_component_id == (
            runtime.extraction_runtime_binding.component_artifact_id
        )
        bindings.append(runtime.extraction_runtime_binding.payload())
        runtime.close()

    assert bindings[0] == bindings[1]
    assert bindings[0]["policy_artifact_id"] == candidate.artifact_id
    assert bindings[0]["policy_artifact_digest"] == candidate.artifact_digest
    assert bindings[0]["deployment_scope"] == "matched_validation"
    assert all(len(client.calls) == 2 for client in clients)


def test_offline_extraction_runtime_binds_candidate_without_trial_activation(tmp_path) -> None:
    parent = Mem0FlatPromptAdapter().export_root_policy_artifact(
        MEM0_FLAT_EXTRACTION_SLOT_ID
    )
    candidate = _candidate(parent=parent)
    runtime = StaticSemanticWritebackRuntime(
        tmp_path / "offline" / "hermes-home",
        _client(),
        operation_evidence_path=tmp_path / "offline" / "operations.jsonl",
        mutation_receipt_path=tmp_path / "offline" / "receipts.json",
        extraction_policy_artifact=candidate,
        expected_extraction_policy_artifact_id=candidate.artifact_id,
        expected_extraction_policy_artifact_digest=candidate.artifact_digest,
        extraction_runtime_scope=ExtractionPromptRuntimeScope.OFFLINE_VALIDATION,
        extraction_trial_id="sm03-heldout-v1",
    )
    assert runtime.extraction_runtime_binding.deployment_scope == (
        ExtractionPromptRuntimeScope.OFFLINE_VALIDATION
    )
    assert runtime.extraction_runtime_binding.trial_id == "sm03-heldout-v1"
    assert runtime.extraction_runtime_binding.payload()["deployment_scope"] == (
        "offline_validation"
    )
    runtime.close()


@pytest.mark.parametrize(
    ("actual", "expected_id", "expected_digest", "scope", "trial_id", "message"),
    (
        ("root", "candidate", "candidate", "matched_validation", "trial", "configured and loaded"),
        ("candidate", "candidate", "wrong", "matched_validation", "trial", "configured and loaded"),
        ("candidate", "candidate", "candidate", "root_static", None, "cannot bind a trial"),
        ("candidate", None, None, "matched_validation", "trial", "expected artifact identity"),
    ),
)
def test_extraction_runtime_identity_mismatch_fails_before_model_call(
    tmp_path,
    actual,
    expected_id,
    expected_digest,
    scope,
    trial_id,
    message,
) -> None:
    adapter = Mem0FlatPromptAdapter()
    root = adapter.export_root_policy_artifact(MEM0_FLAT_EXTRACTION_SLOT_ID)
    candidate = _candidate(parent=root)
    artifact = root if actual == "root" else candidate
    resolved_id = (
        candidate.artifact_id if expected_id == "candidate" else expected_id
    )
    resolved_digest = (
        candidate.artifact_digest
        if expected_digest == "candidate"
        else "0" * 64
        if expected_digest == "wrong"
        else expected_digest
    )
    client = _client()

    with pytest.raises(ValueError, match=message):
        StaticSemanticWritebackRuntime(
            tmp_path / "hermes-home",
            client,
            operation_evidence_path=tmp_path / "operations.jsonl",
            mutation_receipt_path=tmp_path / "receipts.json",
            extraction_policy_artifact=artifact,
            expected_extraction_policy_artifact_id=resolved_id,
            expected_extraction_policy_artifact_digest=resolved_digest,
            extraction_runtime_scope=ExtractionPromptRuntimeScope(scope),
            extraction_trial_id=trial_id,
        )
    assert client.calls == ()


def test_explicit_adaptive_runtime_rejects_empty_store_before_model_call(
    tmp_path,
) -> None:
    client = _client()
    store = JsonAdaptivePolicyStore(
        tmp_path / "adaptive-policies.json",
        trusted_root_policy_versions=("mem0-flat.parent-v1",),
    )
    parameter = TrustedAdaptiveMem0Parameter(
        parameter_id="parameter.retrieval",
        name=AdaptiveParameterName.RETRIEVAL_ACCEPT_THRESHOLD,
        prompt_ref="mem0-flat.retrieval",
        baseline_value=0.35,
    )
    with pytest.raises(ValueError, match="requires an active policy"):
        StaticSemanticWritebackRuntime(
            tmp_path / "hermes-home",
            client,
            operation_evidence_path=tmp_path / "episode" / "operations.jsonl",
            mutation_receipt_path=tmp_path / "hermes-home" / "receipts.json",
            adaptive_policy_store=store,
            adaptive_parameters=(parameter,),
            require_adaptive_policy=True,
        )
    assert client.calls == ()


def test_plain_runtime_identifies_static_extraction_parent(tmp_path) -> None:
    runtime = _runtime(tmp_path, _client())
    assert runtime.static_parent_identity == STATIC_EXTRACTION_PARENT_ID
    assert runtime.utility_gate is None
    assert runtime.adaptive_binding is None
    runtime.close()


def test_static_utility_runtime_preserves_boundary_and_invocation_count(
    tmp_path,
) -> None:
    lifecycle = _lifecycle(tmp_path)
    client = _client()
    gate = FrozenMem0UtilityGate()
    observer = LifecycleLedgerObserver(
        variant="static-utility",
        trace_id="trace-static",
        output_path=tmp_path / "episode" / "lifecycle.jsonl",
    )
    runtime = StaticSemanticWritebackRuntime(
        tmp_path / "hermes-home",
        client,
        operation_evidence_path=tmp_path / "episode" / "operations.jsonl",
        mutation_receipt_path=tmp_path / "hermes-home" / "rsimem-receipts.json",
        ingestion_observer=observer,
        utility_gate=gate,
    )

    result = runtime.process(lifecycle)[0]

    assert result.writeback.logical_exit is True
    execution = result.writeback.executions[0]
    assert execution.context_exit.physical_rewrite is False
    assert execution.context_exit.saved_tokens is None
    assert len(client.calls) == 2
    assert client.calls[0]["binding_fingerprint"] == (
        runtime.extraction_binding.binding_id
    )
    request_id = result.writeback.ingestion.idempotency_key
    assert [item.target.value for item in gate.decisions(request_id)] == [
        "generation",
        "internal_operation",
    ]
    utility_events = [
        item for item in observer.events if item["kind"] == "static_utility_decisions"
    ]
    assert len(utility_events) == 1
    assert utility_events[0]["data"]["decisionCount"] == 2
    assert PREFERENCE not in json.dumps(utility_events[0], sort_keys=True)
    runtime.close()


def test_static_and_utility_modes_share_route_boundary_and_raw_usage(
    tmp_path,
) -> None:
    baseline_lifecycle = _lifecycle(tmp_path / "baseline")
    utility_lifecycle = _lifecycle(tmp_path / "utility")
    baseline_client = _client()
    utility_client = _client()
    baseline = _runtime(tmp_path / "baseline", baseline_client)
    utility = StaticSemanticWritebackRuntime(
        tmp_path / "utility" / "hermes-home",
        utility_client,
        operation_evidence_path=tmp_path / "utility" / "episode" / "operations.jsonl",
        mutation_receipt_path=(
            tmp_path / "utility" / "hermes-home" / "rsimem-receipts.json"
        ),
        utility_gate=FrozenMem0UtilityGate(),
    )

    baseline_result = baseline.process(baseline_lifecycle)[0]
    utility_result = utility.process(utility_lifecycle)[0]
    baseline_ingestion = baseline_result.writeback.ingestion
    utility_ingestion = utility_result.writeback.ingestion
    baseline_execution = baseline_result.writeback.executions[0]
    utility_execution = utility_result.writeback.executions[0]

    assert baseline_lifecycle.snapshot == utility_lifecycle.snapshot
    assert baseline_result.snapshot_id == utility_result.snapshot_id
    assert baseline_ingestion.fixed_route == utility_ingestion.fixed_route
    assert baseline_ingestion.usage == utility_ingestion.usage
    assert [item.action for item in baseline_ingestion.operations] == [
        item.action for item in utility_ingestion.operations
    ]
    assert len(baseline_client.calls) == len(utility_client.calls) == 2
    assert baseline_ingestion.policy_version != utility_ingestion.policy_version
    assert baseline_ingestion.feature_schema_version != (
        utility_ingestion.feature_schema_version
    )
    assert baseline_execution.context_exit == utility_execution.context_exit
    assert baseline_execution.context_exit.natural_exit is True
    assert baseline_execution.context_exit.logical_exit is True
    assert baseline_execution.context_exit.physical_rewrite is False
    assert baseline_execution.context_exit.saved_tokens is None
    baseline.close()
    utility.close()


def test_static_runtime_commits_restart_duplicate_and_emits_content_free_evidence(
    tmp_path,
) -> None:
    lifecycle = _lifecycle(tmp_path)
    first = _runtime(tmp_path, _client())
    result = first.process(lifecycle)[0]

    assert result.snapshot_id == lifecycle.snapshot.snapshot_id
    assert result.plan_id == lifecycle.plans[0].plan_id
    assert result.writeback.logical_exit is True
    assert result.writeback.source_retained is False
    assert result.writeback.executions[0].status.value == "committed"
    assert first.receipts.all()[0].status == MutationReceiptStatus.COMMITTED
    hits = first.registry.resolve(MemoryKind.SEMANTIC).query(MemoryQuery(
        MemoryKind.SEMANTIC,
        "",
        namespace="user",
        limit=10,
    ))
    assert [hit.artifact.content for hit in hits] == [PREFERENCE]
    assert PREFERENCE not in json.dumps(result.observer_evidence(), sort_keys=True)
    assert audit_operation_evidence(
        first.operation_log.events,
        forbidden_values=(PREFERENCE, "Always use TSV output."),
    ) == ()
    first.close()
    first.close()

    restarted = _runtime(tmp_path, _client())
    replay = restarted.process(lifecycle)[0]
    assert replay.writeback.logical_exit is True
    assert replay.writeback.executions[0].status.value == "duplicate"
    assert len(restarted.receipts.all()) == 1
    serialized = (tmp_path / "hermes-home" / "rsimem-receipts.json").read_text(
        encoding="utf-8"
    )
    assert PREFERENCE not in serialized
    assert "Always use TSV output." not in serialized
    restarted.close()


def test_static_runtime_rejects_unvalidated_or_mismatched_lifecycle(tmp_path) -> None:
    lifecycle = _lifecycle(tmp_path)
    runtime = _runtime(tmp_path, _client())
    rejected = replace(lifecycle.receipts[0], status=DryRunStatus.REJECTED)
    with pytest.raises(ValueError, match="requires a validated plan"):
        runtime.process(replace(lifecycle, receipts=(rejected,)))
    with pytest.raises(ValueError, match="one-to-one"):
        runtime.process(replace(lifecycle, receipts=()))
    assert runtime.receipts.all() == ()
    runtime.close()


def test_static_runtime_policy_failure_retains_source_without_mutation(tmp_path) -> None:
    lifecycle = _lifecycle(tmp_path)
    runtime = _runtime(tmp_path, _client(fact_response="not json"))
    result = runtime.process(lifecycle)[0]

    assert result.writeback.logical_exit is False
    assert result.writeback.source_retained is True
    assert result.writeback.reason_code == "ingestion_not_successful"
    assert result.writeback.executions == ()
    assert runtime.receipts.all() == ()
    assert not (tmp_path / "hermes-home" / "memories" / "USER.md").exists()
    runtime.close()


def test_completed_snapshot_compiles_once_without_eviction_plan(tmp_path) -> None:
    snapshot = _lifecycle(tmp_path).snapshot
    client = _client()
    runtime = _runtime(tmp_path, client)

    first = runtime.process_completed_snapshot(snapshot)
    replay = runtime.process_completed_snapshot(snapshot)

    assert first == replay
    assert len(first) == 1
    assert first[0].compilation_id.startswith("semantic_compilation_")
    assert first[0].writeback.logical_exit is True
    assert first[0].receipt is not None
    assert first[0].writeback.ingestion is not None
    projection_digest = first[0].receipt.source_projection_digest
    assert first[0].writeback.ingestion.source_digest == projection_digest
    graph = materialize_operation_graph(runtime.operation_log.events)
    source_artifacts = tuple(
        artifact for artifact in graph.artifacts
        if artifact.kind == ArtifactKind.SOURCE_OBSERVATION
    )
    assert len(source_artifacts) == 1
    assert source_artifacts[0].content_digest == projection_digest
    component_artifacts = {
        artifact.artifact_id: artifact for artifact in graph.artifacts
        if artifact.kind == ArtifactKind.POLICY_PARAMETER
    }
    manifest = runtime.policy.semantic_manifest
    assert component_artifacts[manifest.extraction_component_id].content_digest == (
        manifest.extraction_component_digest
    )
    assert component_artifacts[manifest.update_component_id].content_digest == (
        manifest.update_component_digest
    )
    assert component_artifacts[manifest.retrieval_component_id].content_digest == (
        manifest.retrieval_component_digest
    )
    assert len(client.calls) == 2
    context_exit = first[0].writeback.executions[0].context_exit
    assert context_exit.physical_rewrite is False
    assert context_exit.saved_tokens is None
    runtime.close()

    replay_client = _client()
    restarted = _runtime(tmp_path, replay_client)
    restarted_result = restarted.process_completed_snapshot(snapshot)[0]
    assert restarted_result.duplicate is True
    assert restarted_result.writeback is None
    assert restarted_result.receipt == first[0].receipt
    assert replay_client.calls == ()
    restarted.close()


def test_dry_run_observer_does_not_change_direct_compilation(tmp_path) -> None:
    snapshot = _lifecycle(tmp_path / "snapshot").snapshot
    plain_client = _client()
    observed_client = _client()
    plain = _runtime(tmp_path / "plain", plain_client)
    observer = LifecycleLedgerObserver(
        variant="static-observed",
        trace_id="trace-observed",
        output_path=tmp_path / "observed" / "lifecycle.jsonl",
    )
    observed = StaticSemanticWritebackRuntime(
        tmp_path / "observed" / "hermes-home",
        observed_client,
        operation_evidence_path=tmp_path / "observed" / "operations.jsonl",
        mutation_receipt_path=(
            tmp_path / "observed" / "hermes-home" / "rsimem-receipts.json"
        ),
        ingestion_observer=observer,
    )

    plain_result = plain.process_completed_snapshot(snapshot)[0]
    observed_result = observed.process_completed_snapshot(snapshot)[0]

    assert plain_client.calls == observed_client.calls
    assert plain_result.compilation_id == observed_result.compilation_id
    assert plain_result.writeback is not None
    assert observed_result.writeback is not None
    assert plain_result.writeback.ingestion == observed_result.writeback.ingestion
    assert (
        tmp_path / "plain" / "hermes-home" / "memories" / "USER.md"
    ).read_bytes() == (
        tmp_path / "observed" / "hermes-home" / "memories" / "USER.md"
    ).read_bytes()
    assert len([event for event in observer.events if event["kind"] == "memory_ingestion"]) == 1
    plain.close()
    observed.close()


def test_static_runtime_fails_audit_on_semantic_mutation_without_receipt(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path, _client())
    backend = runtime.registry.resolve(MemoryKind.SEMANTIC)
    result = backend.mutate(MemoryMutation(
        MemoryMutationAction.ADD,
        MemoryKind.SEMANTIC,
        artifact=MemoryArtifact(
            "native-candidate",
            MemoryKind.SEMANTIC,
            "Unreceipted native mutation.",
            namespace="memory",
        ),
    ))
    assert result.accepted is True

    with pytest.raises(ValueError, match="semantic mutation audit failed"):
        runtime.close()
    assert runtime.mutation_audit_report is not None
    issue = next(
        issue for issue in runtime.mutation_audit_report.issues
        if issue.kind == "semantic_state_changed_without_receipt"
    )
    assert issue.writer_identity.value == "native_hermes"
