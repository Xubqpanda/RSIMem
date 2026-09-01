from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsimem.hermes_integration import (
    HermesAdapterExecutionError,
    HermesAdapterFailurePolicy,
    HermesEquivalenceProbe,
    HermesExecutionSurface,
    HermesExecutionMode,
    HermesExecutionRoute,
    HermesExperimentConfig,
    build_configured_hermes_runtime,
    run_hermes_execution_equivalence_variants,
    run_hermes_equivalence_variants,
    _bound_hermes_memory_dir,
    _bound_hermes_skills_dir,
)
from rsimem.hermes_past_bridge import HermesPastBenchBridge
from rsimem.extraction_validation_runtime import (
    EXTRACTION_TRIAL_CONFIG_FILE,
    prepare_extraction_matched_trial_runtime,
)
from rsimem.ledger import LifecycleLedgerObserver
from rsimem.lifecycle import (
    HermesLifecycleConfig,
    RawResourceUsage,
    run_sm01_preference_fixture,
)
from rsimem.memory import (
    OpportunityEvidence,
    OpportunitySurface,
    MemoryArtifact,
    MemoryHit,
    MemoryKind,
    MemoryQuery,
    MemoryResource,
)
from rsimem.memory.live_writeback import (
    StaticSemanticBoundaryResult,
    StaticSemanticWritebackConfig,
)
from rsimem.memory.extraction_feedback import (
    ExtractedFactEvidence,
    ExtractionSetStatus,
    ExtractionSourceEvidence,
    FactDisposition,
)
from rsimem.memory.artifact_set import ArtifactSetSemanticBinding
from rsimem.memory.evidence_planes import EvidencePlane, EvidenceSourceKind
from rsimem.memory.revocation import JsonRevocationRegistry, RevocationEntry
from rsimem.memory.extraction_projection import (
    ExtractionSourceRecord,
    JsonExtractionSourceRecordStore,
)
from rsimem.memory.pure_extraction import (
    JsonPureExtractionFeedbackRecordStore,
    JsonPureExtractionSourceRecordStore,
)
from rsimem.memory.pure_process import JsonPureProcessEventArchive, PureProcessCorpus
from rsimem.memory.process_feedback import ProcessEventKind
from rsimem.memory.tool_exact_join import ToolJoinResolutionStatus, resolve_tool_call_result
from rsimem.memory.extraction_optimizer_capture import (
    ExtractionOptimizerFeedbackCapture,
    ExtractionOptimizerSourceCapture,
    JsonExtractionOptimizerCaptureLog,
)
from rsimem.memory.operation_graph import (
    AppendOnlyOperationEvidenceLog,
    OperationKind,
    OperationStatus,
    materialize_operation_graph,
)
from rsimem.memory.future_trace import SemanticFutureEvidence, SemanticOutcomeEvidence
from rsimem.memory.use_attribution import (
    MemoryUseEvidence,
    MemoryUseResolutionStatus,
    OutcomeEvidenceKind,
    resolve_memory_use,
)
from rsimem.memory.adaptive_policy import AdaptiveParameterName
from rsimem.memory.adaptive_mem0_binding import TrustedAdaptiveMem0Parameter
from rsimem.memory_systems.mem0_flat import (
    FakeCompletionClient,
    POLICY_FACT_EXTRACTION_PROMPT,
    POLICY_INTERNAL_OPERATION_PROMPT,
)
from test_extraction_matched_activation import _offline_decision
from test_extraction_offline_validation import _candidate, _parent
from extraction_fingerprint_support import extraction_activation_fixture


PRIVATE_PREFERENCE = "Use TSV with owner, priority, task, and due_date."


def _fixture_application_schema() -> dict[str, object]:
    from rsimem.memory.opportunity import ApplicationOpportunitySchema

    contract = ApplicationOpportunitySchema.create(
        schema_id="fixture.application.v1",
        version="v1",
        requirement_ids=("application.fixture.preference",),
    )
    return {
        "schema_id": contract.schema_id,
        "schema_version": 1,
        "application_contract": contract.payload(),
        "opportunities": [{
            "semantic_key": "application.fixture.preference",
            "surface": "tool_schema",
            "tool_name": "fixture_apply",
            "required_parameter": "preference",
        }],
    }


def test_pure_observation_window_is_stable_across_physical_replays() -> None:
    first = OpportunityEvidence.create(
        source_surface=OpportunitySurface.TOOL_SCHEMA,
        semantic_requirement="application.fixture.preference",
        observation_time="2026-08-31T00:00:00Z",
        operation_id="opportunity.physical.one",
        provenance_id="provenance.physical.one",
        source_payload={"tool_name": "fixture_apply"},
    )
    replay = OpportunityEvidence.create(
        source_surface=OpportunitySurface.TOOL_SCHEMA,
        semantic_requirement="application.fixture.preference",
        observation_time="2026-08-31T00:00:00Z",
        operation_id="opportunity.physical.two",
        provenance_id="provenance.physical.two",
        source_payload={"tool_name": "fixture_apply"},
    )
    first_window = HermesPastBenchBridge._pure_observation_window(
        "Apply the saved preference.",
        (first,),
    )
    replay_window = HermesPastBenchBridge._pure_observation_window(
        "Apply the saved preference.",
        (replay,),
    )
    assert first_window == replay_window
    assert first_window != HermesPastBenchBridge._pure_observation_window(
        "Apply a different preference.",
        (replay,),
    )


def _hermes_home(path: Path) -> Path:
    memories = path / "memories"
    memories.mkdir(parents=True)
    (memories / "MEMORY.md").write_text(
        f"{PRIVATE_PREFERENCE}\n§\nAlways include a header row.",
        encoding="utf-8",
    )
    (memories / "USER.md").write_text(
        "The user prefers concise status updates.",
        encoding="utf-8",
    )

    skill = path / "skills" / "operations" / "task-table"
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: task-table\ndescription: Format a task table\n---\nUse the requested columns.",
        encoding="utf-8",
    )
    (skill / "references" / "columns.md").write_text(
        "owner, priority, task, due_date\n",
        encoding="utf-8",
    )

    from hermes_state import SessionDB

    db = SessionDB(path / "state.db")
    db.create_session("session-1", "cli", model="fixture-model")
    db.append_message("session-1", "user", "Please format the project tasks.")
    db.append_message("session-1", "assistant", "I used the requested task table.")
    db.append_message("session-1", "user", "The task table looks correct.")
    db.close()

    connection = sqlite3.connect(path / "state.db")
    connection.execute(
        "UPDATE sessions SET started_at = ?, title = ? WHERE id = ?",
        (0.0, "Task table fixture", "session-1"),
    )
    connection.executemany(
        "UPDATE messages SET timestamp = ? WHERE id = ?",
        [(1.0, 1), (2.0, 2), (3.0, 3)],
    )
    connection.commit()
    connection.close()
    return path


def test_config_defaults_to_direct_native_and_requires_three_routes(tmp_path: Path) -> None:
    config = HermesExperimentConfig()
    assert config.mode == HermesExecutionMode.NATIVE
    assert config.uses_adapter is False
    assert config.ledger_enabled is False
    assert config.adapter_failure_policy == HermesAdapterFailurePolicy.FAIL_CLOSED
    assert set(config.routes) == set(MemoryKind)

    with pytest.raises(ValueError, match="one route per memory kind"):
        HermesExperimentConfig(routes={MemoryKind.SEMANTIC: "semantic"})

    unsupported = HermesExperimentConfig(routes={
        MemoryKind.SEMANTIC: "external-semantic",
        MemoryKind.EPISODIC: "hermes-native-episodic",
        MemoryKind.PROCEDURAL: "hermes-native-procedural",
    })
    with pytest.raises(ValueError, match="unregistered Hermes backend routes"):
        build_configured_hermes_runtime(tmp_path, unsupported)


def test_native_ledger_and_adapter_views_are_equivalent(tmp_path: Path) -> None:
    home = _hermes_home(tmp_path)
    report = run_hermes_equivalence_variants(
        home,
        HermesEquivalenceProbe(episodic_query="task table"),
    )

    assert report.equivalent is True
    assert [variant.mode for variant in report.variants] == list(HermesExecutionMode)
    assert all(variant.equivalent_to_native for variant in report.variants)
    assert [variant.ledger_enabled for variant in report.variants] == [False, True, True]
    native = report.variants[0]
    native_ledger = report.variants[1]
    adapter = next(
        item for item in report.variants
        if item.mode == HermesExecutionMode.ADAPTER_LEDGER
    )
    assert {check.memory_kind for check in adapter.checks} == set(MemoryKind)
    assert adapter.memory_event_count == 12
    assert native.memory_event_count == 0
    assert native_ledger.memory_event_count == 12
    assert native.ledger_event_count == 0
    assert native_ledger.ledger_event_count == 12
    assert adapter.ledger_event_count == 12
    assert native_ledger.ledger_event_kinds == adapter.ledger_event_kinds
    assert native_ledger.memory_event_kinds == adapter.memory_event_kinds
    assert native_ledger.checks == native.checks
    semantic = next(
        check for check in adapter.checks if check.memory_kind == MemoryKind.SEMANTIC
    )
    assert semantic.native_item_count == semantic.candidate_item_count == 3
    serialized = json.dumps(asdict(report), default=str)
    assert PRIVATE_PREFERENCE not in serialized
    assert "requested task table" not in serialized


def test_real_hermes_execution_surfaces_are_equivalent_across_variants(
    tmp_path: Path,
) -> None:
    from tools import memory_tool, skills_tool

    home = _hermes_home(tmp_path)
    original_memory_dir = memory_tool.MEMORY_DIR
    original_skills_dir = skills_tool.SKILLS_DIR
    original_hermes_home = os.environ.get("HERMES_HOME")
    report = run_hermes_execution_equivalence_variants(
        home,
        HermesEquivalenceProbe(
            episodic_query="task table",
            procedural_skill_name="task-table",
            procedural_resource_path="references/columns.md",
        ),
    )

    assert report.equivalent is True
    assert [variant.mode for variant in report.variants] == list(HermesExecutionMode)
    assert all(variant.equivalent_to_native for variant in report.variants)
    assert all(
        {check.surface for check in variant.checks} == set(HermesExecutionSurface)
        for variant in report.variants
    )
    assert all(
        check.native_content_chars == check.candidate_content_chars
        for variant in report.variants
        for check in variant.checks
    )
    assert [variant.memory_event_count for variant in report.variants] == [0, 15, 15]
    assert [variant.ledger_event_count for variant in report.variants] == [0, 15, 15]
    assert report.variants[1].memory_event_kinds == report.variants[2].memory_event_kinds
    serialized = json.dumps(asdict(report), default=str)
    assert PRIVATE_PREFERENCE not in serialized
    assert "concise status updates" not in serialized
    assert "requested task table" not in serialized
    assert "Use the requested columns" not in serialized
    assert "owner, priority, task, due_date" not in serialized
    assert memory_tool.MEMORY_DIR == original_memory_dir
    assert skills_tool.SKILLS_DIR == original_skills_dir
    assert os.environ.get("HERMES_HOME") == original_hermes_home


def test_execution_surfaces_and_artifact_ids_survive_runtime_restart(
    tmp_path: Path,
) -> None:
    home = _hermes_home(tmp_path)
    probe = HermesEquivalenceProbe(
        episodic_query="task table",
        procedural_skill_name="task-table",
        procedural_resource_path="references/columns.md",
    )

    first_report = run_hermes_execution_equivalence_variants(home, probe)
    second_report = run_hermes_execution_equivalence_variants(home, probe)

    def artifact_ids() -> dict[str, tuple[str, ...]]:
        runtime = build_configured_hermes_runtime(
            home,
            HermesExperimentConfig(HermesExecutionMode.ADAPTER_LEDGER),
        )
        try:
            return {
                "memory": tuple(hit.artifact.artifact_id for hit in runtime.query(
                    MemoryQuery(MemoryKind.SEMANTIC, "", namespace="memory", limit=100)
                )),
                "user": tuple(hit.artifact.artifact_id for hit in runtime.query(
                    MemoryQuery(MemoryKind.SEMANTIC, "", namespace="user", limit=100)
                )),
                "episodic": tuple(hit.artifact.artifact_id for hit in runtime.query(
                    MemoryQuery(MemoryKind.EPISODIC, probe.episodic_query, limit=5)
                )),
                "procedural": tuple(hit.artifact.artifact_id for hit in runtime.query(
                    MemoryQuery(MemoryKind.PROCEDURAL, "", limit=100)
                )),
            }
        finally:
            runtime.close()

    first_ids = artifact_ids()
    second_ids = artifact_ids()
    assert first_report == second_report
    assert first_ids == second_ids


def test_adapter_failure_policy_is_explicit_and_content_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rsimem.hermes_integration as integration

    home = _hermes_home(tmp_path)
    probe = HermesEquivalenceProbe(
        episodic_query="task table",
        procedural_skill_name="task-table",
        procedural_resource_path="references/columns.md",
    )

    def fail_session_search(*args, **kwargs):
        raise RuntimeError("private adapter failure detail")

    monkeypatch.setattr(
        integration,
        "_capture_adapter_session_search",
        fail_session_search,
    )
    with pytest.raises(
        HermesAdapterExecutionError,
        match=r"session_search \(RuntimeError\)",
    ):
        run_hermes_execution_equivalence_variants(home, probe)

    report = run_hermes_execution_equivalence_variants(
        home,
        probe,
        adapter_failure_policy=HermesAdapterFailurePolicy.BYPASS_NATIVE,
    )
    adapter = next(
        variant
        for variant in report.variants
        if variant.mode == HermesExecutionMode.ADAPTER_LEDGER
    )
    checks = {check.surface: check for check in adapter.checks}

    assert report.equivalent is True
    assert checks[HermesExecutionSurface.SESSION_SEARCH].route == (
        HermesExecutionRoute.NATIVE_BYPASS
    )
    assert checks[HermesExecutionSurface.SESSION_SEARCH].adapter_failure == "RuntimeError"
    assert checks[HermesExecutionSurface.SYSTEM_PROMPT].route == HermesExecutionRoute.ADAPTER
    assert "private adapter failure detail" not in json.dumps(asdict(report), default=str)


def test_past_bench_bridge_routes_real_hermes_read_surfaces(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from hermes_state import SessionDB
    from tools.registry import registry

    home = _hermes_home(tmp_path)
    evidence_path = tmp_path / "artifacts" / "rsimem_memory_events.jsonl"
    db = SessionDB(home / "state.db")
    with _bound_hermes_memory_dir(home) as memory_tool:
        store = memory_tool.MemoryStore()
        store.load_from_disk()
    native_memory = store.format_for_system_prompt("memory")
    native_search = db.search_messages(query="task table", limit=50, offset=0)

    bridge = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(
            HermesExecutionMode.ADAPTER_LEDGER,
            verify_native_projection=True,
        ),
        evidence_path=evidence_path,
        run_id="run-bridge",
        trace_id="trace-bridge",
        episode_id="episode-bridge",
        session_id="session-bridge",
        task_id="task-bridge",
        experiment_variant="with_persistence",
    )
    agent = SimpleNamespace(_memory_store=store, _session_db=db)
    bridge.attach(agent)
    wrapped_skill_handler = registry._tools["skill_view"].handler
    try:
        assert agent._memory_store.format_for_system_prompt("memory") == native_memory
        updated_preference = "Use CSV with an explicit owner column."
        (home / "memories" / "MEMORY.md").write_text(
            updated_preference,
            encoding="utf-8",
        )
        assert store.format_for_system_prompt("memory") == native_memory
        assert agent._memory_store.format_for_system_prompt("memory") == native_memory
        assert agent._session_db.search_messages(
            query="task table",
            limit=50,
            offset=0,
        ) == native_search
        assert agent._session_db.get_messages_as_conversation("session-1") == (
            db.get_messages_as_conversation("session-1")
        )
        with _bound_hermes_skills_dir(home / "skills"):
            skills = registry.dispatch("skills_list", {})
            skill = registry.dispatch("skill_view", {"name": "task-table"})
        assert json.loads(skills)["count"] == 1
        assert json.loads(skill)["name"] == "task-table"
        persisted_before_close = evidence_path.read_text(encoding="utf-8")
        assert '"kind": "query"' in persisted_before_close
        assert '"kind": "injected"' in persisted_before_close
        policy_path = evidence_path.with_name("rsimem_policy_decisions.jsonl")
        policy_events = [
            json.loads(line)
            for line in policy_path.read_text(encoding="utf-8").splitlines()
        ]
        assert any(event["layer"] == "exposure" for event in policy_events)
        process_path = evidence_path.with_name("rsimem_process_feedback.jsonl")
        process_events = [
            json.loads(line)
            for line in process_path.read_text(encoding="utf-8").splitlines()
        ]
        assert process_events
        from rsimem.memory.process_feedback import PROCESS_FEEDBACK_SCHEMA

        assert all(event["schema"] == PROCESS_FEEDBACK_SCHEMA for event in process_events)
        assert all(event["source_revision"] for event in process_events)
        assert {event["policy_layer"] for event in process_events if event["policy_layer"]} == {
            "exposure",
        }
        # Process feedback is content-free and must not copy the native skill,
        # conversation or memory body across the audit boundary.
        assert "Use CSV with an explicit owner column." not in process_path.read_text(encoding="utf-8")
        assert all(event["sourceRevision"] for event in policy_events)
    finally:
        bridge.close()
        db.close()

    assert registry._tools["skill_view"].handler is not wrapped_skill_handler
    serialized = evidence_path.read_text(encoding="utf-8")
    assert '"kind": "query"' in serialized
    assert '"kind": "injected"' in serialized
    assert '"kind": "projection_check"' in serialized
    assert PRIVATE_PREFERENCE not in serialized
    assert "requested task table" not in serialized
    assert "Use the requested columns" not in serialized


def test_past_bench_bridge_records_runtime_opportunity_without_family_leakage(
    tmp_path: Path,
) -> None:
    home = _hermes_home(tmp_path)

    def provider(result):
        assert result["messages"][0]["role"] == "user"
        return (OpportunityEvidence.create(
            source_surface=OpportunitySurface.CURRENT_INPUT,
            semantic_requirement="resource.share.recipient_policy",
            observation_time="2026-08-30T01:02:03Z",
            operation_id="op.current-input.v1",
            provenance_id="provenance.run-bridge.v1",
            source_payload={"input_digest": "a" * 64},
        ),)

    bridge = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.ADAPTER_LEDGER),
        evidence_path=tmp_path / "artifacts" / "events.jsonl",
        run_id="run-bridge-opportunity",
        trace_id="trace-bridge-opportunity",
        episode_id="episode-bridge-opportunity",
        session_id="session-bridge-opportunity",
        task_id="task-bridge-opportunity",
        experiment_variant="native+adapter+ledger",
        family_id="SM02_constraint_retention",
        stage="eval_near",
        opportunity_evidence_provider=provider,
    )
    try:
        recorded = bridge.record_opportunity_evidence(provider({
            "messages": [{"role": "user", "content": "visible request"}],
        }))
        assert len(recorded) == 1
        assert bridge.opportunity_evidence == recorded
        payload = json.dumps(recorded[0].payload(), ensure_ascii=True)
        assert "family_id" not in payload
        assert "stage" not in payload
    finally:
        bridge.close()

    restarted = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.ADAPTER_LEDGER),
        evidence_path=tmp_path / "artifacts" / "events.jsonl",
        run_id="run-bridge-opportunity",
        trace_id="trace-bridge-opportunity",
        episode_id="episode-bridge-opportunity",
        session_id="session-bridge-opportunity",
        task_id="task-bridge-opportunity",
        experiment_variant="native+adapter+ledger",
        family_id="SM02_constraint_retention",
        stage="eval_near",
    )
    try:
        assert restarted.opportunity_evidence == recorded
    finally:
        restarted.close()


def test_past_bench_runtime_observation_does_not_run_family_semantic_parser(
    tmp_path: Path,
) -> None:
    def opportunity_provider(result):
        assert result["messages"][0]["role"] == "user"
        return (OpportunityEvidence.create(
            source_surface=OpportunitySurface.CURRENT_INPUT,
            semantic_requirement="resource.share.recipient_policy",
            observation_time="2026-08-30T01:02:03Z",
            operation_id="op.runtime-current-input.v1",
            provenance_id="provenance.runtime-current-input.v1",
            source_payload={"input_digest": "a" * 64},
        ),)

    bridge = HermesPastBenchBridge(
        _hermes_home(tmp_path),
        HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
        evidence_path=tmp_path / "artifacts" / "events.jsonl",
        run_id="run-runtime-observation",
        trace_id="trace-runtime-observation",
        episode_id="episode-runtime-observation",
        session_id="session-runtime-observation",
        task_id="task-runtime-observation",
        experiment_variant="native+adapter+ledger",
        family_id="SM01_preference_adoption",
        stage="eval_near",
        opportunity_evidence_provider=opportunity_provider,
        static_writeback_config=StaticSemanticWritebackConfig(
            mode="static",
            feedback_contract="sm01_tsv_v1",
        ),
        static_completion_client=FakeCompletionClient({}),
    )
    result = {
        "messages": [{
            "role": "user",
            "content": "Use TSV with owner, priority, task, and due_date for this report.",
        }],
        "completed": True,
        "final_response": "",
    }
    try:
        runtime = bridge._semantic_deployment_observation(result)
        assert runtime.current_input_semantic_keys == (
            "resource.share.recipient_policy",
        )
        assert runtime.task_semantic_keys == (
            "resource.share.recipient_policy",
        )

        audit = bridge._semantic_audit_observation(
            runtime,
            current_input="Use TSV with owner, priority, task, and due_date for this report.",
        )
        assert audit.current_input_semantic_keys
        assert "resource.share.recipient_policy" in audit.current_input_semantic_keys
        assert audit.task_semantic_keys
    finally:
        bridge.close()


def test_runtime_opportunity_provider_receives_scope_free_input(tmp_path: Path) -> None:
    seen: list[object] = []

    def provider(result):
        seen.append(result)
        return ()

    bridge = HermesPastBenchBridge(
        _hermes_home(tmp_path),
        HermesExperimentConfig(HermesExecutionMode.ADAPTER_LEDGER),
        evidence_path=tmp_path / "artifacts" / "events.jsonl",
        run_id="run-scope-free-provider",
        trace_id="trace-scope-free-provider",
        episode_id="episode-scope-free-provider",
        session_id="session-scope-free-provider",
        task_id="task-scope-free-provider",
        experiment_variant="native+adapter+ledger",
        family_id="SM02_constraint_retention",
        stage="eval_near",
        opportunity_evidence_provider=provider,
    )
    try:
        bridge._record_runtime_opportunities({
            "family_id": "SM02_constraint_retention",
            "stage": "eval_near",
            "messages": [{
                "role": "user",
                "content": "visible request",
                "metadata": {
                    "familyId": "nested-family",
                    "FAMILY_ID": "nested-family-upper",
                    "benchmarkStage": "nested-stage",
                },
            }],
            "resource_state": {"available": True},
        })
    finally:
        bridge.close()
    assert seen == [{
        "messages": [{
            "role": "user",
            "content": "visible request",
            "metadata": {},
        }],
        "resource_state": {"available": True},
    }]


def test_runtime_opportunities_can_be_materialized_from_explicit_result_field(
    tmp_path: Path,
) -> None:
    bridge = HermesPastBenchBridge(
        _hermes_home(tmp_path),
        HermesExperimentConfig(HermesExecutionMode.ADAPTER_LEDGER),
        evidence_path=tmp_path / "artifacts" / "events.jsonl",
        run_id="run-explicit-opportunity",
        trace_id="trace-explicit-opportunity",
        episode_id="episode-explicit-opportunity",
        session_id="session-explicit-opportunity",
        task_id="task-explicit-opportunity",
        experiment_variant="native+adapter+ledger",
    )
    evidence = OpportunityEvidence.create(
        source_surface=OpportunitySurface.USER_REQUEST,
        semantic_requirement="preference.output.concise",
        observation_time="2026-08-30T01:02:03Z",
        operation_id="op.explicit-opportunity.v1",
        provenance_id="provenance.explicit-opportunity.v1",
        source_payload={"request_digest": "a" * 64},
    )
    try:
        recorded = bridge._record_runtime_opportunities({
            "family_id": "SM01_preference_adoption",
            "stage": "eval_near",
            "rsimem_opportunities": [evidence.payload()],
        })
    finally:
        bridge.close()
    assert recorded == (evidence,)


def test_past_bench_bridge_records_generic_memory_use_join(tmp_path: Path) -> None:
    bridge = HermesPastBenchBridge(
        _hermes_home(tmp_path),
        HermesExperimentConfig(HermesExecutionMode.ADAPTER_LEDGER),
        evidence_path=tmp_path / "artifacts" / "events.jsonl",
        run_id="run-memory-use",
        trace_id="trace-memory-use",
        episode_id="episode-memory-use",
        session_id="session-memory-use",
        task_id="task-memory-use",
        experiment_variant="native+adapter+ledger",
    )
    future = SemanticFutureEvidence(
        "op.query.memory-use",
        "op.retrieval.memory-use",
        "op.injection.memory-use",
        ("artifact.memory-use.v1",),
        ("revision.memory-use.v1",),
        "artifact.injection.memory-use",
        ("artifact.memory-use.v1",),
    )
    outcome = SemanticOutcomeEvidence(
        "op.use.memory-use",
        "op.outcome.memory-use",
        ("artifact.memory-use.v1",),
        OperationStatus.SUCCESS,
    )
    try:
        bridge._record_memory_use_evidence(
            future,
            outcome,
            {"completed": True, "observed_at": "2026-08-30T01:02:03Z"},
        )
        evidence = bridge.memory_use_evidence
        assert len(evidence) == 1
        assert resolve_memory_use(evidence[0]).status == MemoryUseResolutionStatus.ATTRIBUTABLE_USE
        serialized = json.dumps(evidence[0].payload(), ensure_ascii=True)
        assert "SM02" not in serialized
        assert "stage" not in serialized
    finally:
        bridge.close()


def test_malformed_tool_outcome_does_not_create_failure_attribution(
    tmp_path: Path,
) -> None:
    bridge = HermesPastBenchBridge(
        _hermes_home(tmp_path),
        HermesExperimentConfig(HermesExecutionMode.ADAPTER_LEDGER),
        evidence_path=tmp_path / "artifacts" / "events.jsonl",
        run_id="run-memory-use-malformed",
        trace_id="trace-memory-use-malformed",
        episode_id="episode-memory-use-malformed",
        session_id="session-memory-use-malformed",
        task_id="task-memory-use-malformed",
        experiment_variant="native+adapter+ledger",
    )
    future = SemanticFutureEvidence(
        "op.query.malformed",
        "op.retrieval.malformed",
        "op.injection.malformed",
        ("artifact.memory-use-malformed.v1",),
        ("revision.memory-use-malformed.v1",),
        "artifact.injection.memory-use-malformed",
        ("artifact.memory-use-malformed.v1",),
    )
    outcome = SemanticOutcomeEvidence(
        "op.use.malformed",
        "op.outcome.malformed",
        ("artifact.memory-use-malformed.v1",),
        OperationStatus.SUCCESS,
    )
    try:
        bridge._record_memory_use_evidence(
            future,
            outcome,
            {
                "completed": True,
                "observed_at": "2026-08-30T01:02:03Z",
                "messages": [{
                    "role": "tool",
                    "content": '{"success": "false"}',
                }],
            },
        )
        evidence = bridge.memory_use_evidence[0]
    finally:
        bridge.close()

    assert evidence.outcome_kind is None
    assert evidence.outcome_success is None
    assert resolve_memory_use(evidence).status is (
        MemoryUseResolutionStatus.BEHAVIORAL_CONSISTENCY
    )


def test_past_bench_bridge_validates_artifact_set_ownership(tmp_path: Path) -> None:
    source = ExtractionSourceEvidence(
        source_id="source.set.v1",
        source_projection_digest="a" * 64,
        extraction_set_id="set.v1",
        status=ExtractionSetStatus.NONEMPTY,
        available_semantic_keys=("preference.summary.tsv",),
        facts=(
            ExtractedFactEvidence(
                "fact.set.a.v1",
                ("preference.summary.tsv",),
                FactDisposition.PERSISTED,
                artifact_id="artifact.set.a.v1",
            ),
            ExtractedFactEvidence(
                "fact.set.b.v1",
                ("preference.summary.tsv",),
                FactDisposition.PERSISTED,
                artifact_id="artifact.set.b.v1",
            ),
        ),
    )
    binding = ArtifactSetSemanticBinding.create(
        semantic_unit_id="semantic.preference.tsv.v1",
        semantic_key="preference.summary.tsv",
        member_artifact_ids=("artifact.set.a.v1", "artifact.set.b.v1"),
        member_fact_ids=("fact.set.a.v1", "fact.set.b.v1"),
        complete=True,
        source_digest=source.source_projection_digest,
        provenance_id="provenance.set.v1",
    )
    bridge = HermesPastBenchBridge(
        _hermes_home(tmp_path),
        HermesExperimentConfig(HermesExecutionMode.ADAPTER_LEDGER),
        evidence_path=tmp_path / "artifacts" / "events.jsonl",
        run_id="run-set",
        trace_id="trace-set",
        episode_id="episode-set",
        session_id="session-set",
        task_id="task-set",
        experiment_variant="native+adapter+ledger",
        artifact_set_binding_provider=lambda _: (binding,),
    )
    try:
        bridge._record_artifact_set_bindings(source)
        assert bridge.artifact_set_bindings == (binding,)
        # Use a valid foreign binding to exercise source ownership.
        foreign = ArtifactSetSemanticBinding.create(
            semantic_unit_id="semantic.foreign.v1",
            member_artifact_ids=("artifact.foreign.v1",),
            member_fact_ids=("fact.foreign.v1",),
            complete=True,
            source_digest="b" * 64,
            provenance_id="provenance.foreign.v1",
        )
        bridge._artifact_set_binding_provider = lambda _: (foreign,)
        with pytest.raises(ValueError, match="source digest mismatch"):
            bridge._record_artifact_set_bindings(source)
    finally:
        bridge.close()


def test_past_bench_bridge_failure_policy_controls_native_bypass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    class Store:
        def format_for_system_prompt(self, target: str) -> str:
            return "native prompt"

        def _render_block(self, target: str, entries: list[str]) -> str:
            return "adapter prompt"

    home = _hermes_home(tmp_path)

    def bridge(policy: HermesAdapterFailurePolicy) -> HermesPastBenchBridge:
        return HermesPastBenchBridge(
            home,
            HermesExperimentConfig(
                HermesExecutionMode.ADAPTER_LEDGER,
                adapter_failure_policy=policy,
            ),
            evidence_path=tmp_path / policy.value / "events.jsonl",
            run_id=f"run-{policy.value}",
            trace_id=f"trace-{policy.value}",
            episode_id="episode",
            session_id="session",
            task_id="task",
            experiment_variant="with_persistence",
        )

    fail_closed = bridge(HermesAdapterFailurePolicy.FAIL_CLOSED)
    monkeypatch.setattr(
        fail_closed.runtime,
        "query",
        lambda query: (_ for _ in ()).throw(RuntimeError("private failure")),
    )
    closed_agent = SimpleNamespace(_memory_store=Store(), _session_db=None)
    fail_closed.attach(closed_agent)
    try:
        with pytest.raises(HermesAdapterExecutionError, match="system_prompt"):
            closed_agent._memory_store.format_for_system_prompt("memory")
    finally:
        fail_closed.close()

    bypass = bridge(HermesAdapterFailurePolicy.BYPASS_NATIVE)
    monkeypatch.setattr(
        bypass.runtime,
        "query",
        lambda query: (_ for _ in ()).throw(RuntimeError("private failure")),
    )
    bypass_agent = SimpleNamespace(_memory_store=Store(), _session_db=None)
    bypass.attach(bypass_agent)
    try:
        assert bypass_agent._memory_store.format_for_system_prompt("memory") == (
            "native prompt"
        )
    finally:
        bypass.close()

    serialized = (tmp_path / "bypass_native" / "events.jsonl").read_text(
        encoding="utf-8"
    )
    assert "adapter_failure_native_bypass" in serialized
    assert "RuntimeError" in serialized
    assert "private failure" not in serialized

    native_failure = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(
            HermesExecutionMode.ADAPTER_LEDGER,
            adapter_failure_policy=HermesAdapterFailurePolicy.BYPASS_NATIVE,
        ),
        evidence_path=tmp_path / "bypass_native_failure" / "events.jsonl",
        run_id="run-bypass-native-failure",
        trace_id="trace-bypass-native-failure",
        episode_id="episode",
        session_id="session",
        task_id="task",
        experiment_variant="with_persistence",
    )
    try:
        with pytest.raises(RuntimeError, match="native failure"):
            native_failure.adapter_call(
                "session_search",
                lambda: (_ for _ in ()).throw(RuntimeError("adapter failure")),
                lambda: (_ for _ in ()).throw(RuntimeError("native failure")),
            )
    finally:
        native_failure.close()
    native_failure_events = (
        tmp_path / "bypass_native_failure" / "rsimem_process_feedback.jsonl"
    ).read_text(encoding="utf-8")
    assert "retrieval_failure" in native_failure_events
    assert '"status": "failed"' in native_failure_events


def test_past_bench_bridge_fails_closed_on_projection_mismatch(tmp_path: Path) -> None:
    from types import SimpleNamespace

    class Store:
        def format_for_system_prompt(self, target: str) -> str:
            return "native prompt"

        def _render_block(self, target: str, entries: list[str]) -> str:
            return f"adapter prompt: {entries[0]}"

    bridge = HermesPastBenchBridge(
        _hermes_home(tmp_path),
        HermesExperimentConfig(
            HermesExecutionMode.ADAPTER_LEDGER,
            verify_native_projection=True,
        ),
        evidence_path=tmp_path / "artifacts" / "events.jsonl",
        run_id="run-mismatch",
        trace_id="trace-mismatch",
        episode_id="episode-mismatch",
        session_id="session-mismatch",
        task_id="task-mismatch",
        experiment_variant="with_persistence",
    )
    artifact = MemoryArtifact(
        artifact_id="adapter-only",
        kind=MemoryKind.SEMANTIC,
        namespace="memory",
        content="adapter-only content",
    )
    bridge.runtime.query = lambda query: (
        MemoryHit(artifact, rank=1, score=1.0, backend="hermes-native-semantic"),
    )
    agent = SimpleNamespace(_memory_store=Store(), _session_db=None)
    bridge.attach(agent)
    try:
        with pytest.raises(HermesAdapterExecutionError, match="projection mismatch"):
            agent._memory_store.format_for_system_prompt("memory")
    finally:
        bridge.close()

    serialized = (tmp_path / "artifacts" / "events.jsonl").read_text(encoding="utf-8")
    assert '"kind": "projection_check"' in serialized
    assert '"equivalent": false' in serialized
    assert "adapter-only content" not in serialized
    assert "native prompt" not in serialized


def test_past_bench_bridge_skill_reads_come_from_adapter_projection(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from tools.registry import registry

    home = _hermes_home(tmp_path)
    bridge = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.ADAPTER_LEDGER),
        evidence_path=tmp_path / "artifacts" / "events.jsonl",
        run_id="run-procedural",
        trace_id="trace-procedural",
        episode_id="episode-procedural",
        session_id="session-procedural",
        task_id="task-procedural",
        experiment_variant="with_persistence",
    )
    original_query = bridge.runtime.query
    artifact = MemoryArtifact(
        artifact_id="adapter-only-skill",
        kind=MemoryKind.PROCEDURAL,
        namespace="operations",
        title="adapter-only",
        content=(
            "---\nname: adapter-only\ndescription: Adapter projected skill\n---\n"
            "Read this only through RSIMem."
        ),
        metadata={
            "skill_name": "adapter-only",
            "description": "Adapter projected skill",
            "relative_path": "operations/adapter-only",
        },
        resources=(MemoryResource(
            "references/source.md",
            b"adapter projected resource\n",
            "text/markdown",
        ),),
    )
    hit = MemoryHit(artifact, rank=1, score=1.0, backend="adapter-procedural")

    def query(value: MemoryQuery):
        if value.kind == MemoryKind.PROCEDURAL:
            if not value.text or "adapter-only" in value.text:
                return (hit,)
            return ()
        return original_query(value)

    bridge.runtime.query = query
    bridge.attach(SimpleNamespace(_memory_store=None, _session_db=None))
    try:
        listed = json.loads(registry.dispatch("skills_list", {}))
        viewed = json.loads(registry.dispatch(
            "skill_view",
            {"name": "adapter-only"},
        ))
        resource = json.loads(registry.dispatch(
            "skill_view",
            {"name": "adapter-only", "file_path": "references/source.md"},
        ))
    finally:
        bridge.close()

    assert [skill["name"] for skill in listed["skills"]] == ["adapter-only"]
    assert viewed["content"].endswith("Read this only through RSIMem.")
    assert resource["content"] == "adapter projected resource\n"
    assert "task-table" not in json.dumps((listed, viewed, resource))


def test_past_bench_bridge_historical_sessions_come_from_adapter_projection(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    class NativeDb:
        def search_messages(self, **kwargs):
            raise AssertionError("native search must not run")

        def get_session(self, session_id):
            raise AssertionError("native historical session must not run")

        def get_messages_as_conversation(self, session_id):
            raise AssertionError("native historical conversation must not run")

    home = _hermes_home(tmp_path)
    bridge = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.ADAPTER_LEDGER),
        evidence_path=tmp_path / "artifacts" / "events.jsonl",
        run_id="run-episodic",
        trace_id="trace-episodic",
        episode_id="episode-episodic",
        session_id="session-episodic",
        task_id="task-episodic",
        experiment_variant="with_persistence",
    )
    original_query = bridge.runtime.query
    conversation = (
        {"role": "user", "content": "Adapter-owned history."},
        {"role": "assistant", "content": "Recovered without native state."},
    )
    artifact = MemoryArtifact(
        artifact_id="adapter-episode-42",
        kind=MemoryKind.EPISODIC,
        namespace="adapter-session",
        title="historical message",
        content="Adapter-owned history.",
        metadata={
            "message_id": 42,
            "role": "user",
            "source": "remote-backend",
            "snippet": ">>>Adapter-owned<<< history.",
            "timestamp": 1.0,
            "tool_name": None,
            "model": "fixture-model",
            "session_started": 1.0,
            "context": (("user", "Adapter-owned history."),),
            "session_lineage": ((
                "adapter-session",
                {
                    "id": "adapter-session",
                    "source": "remote-backend",
                    "model": "fixture-model",
                    "parent_session_id": None,
                },
                conversation,
            ),),
        },
    )
    hit = MemoryHit(artifact, rank=1, score=1.0, backend="adapter-episodic")

    def query(value: MemoryQuery):
        if value.kind == MemoryKind.EPISODIC:
            assert value.filters["offset"] == 0
            assert value.filters["role_filter"] == ["user"]
            return (hit,)
        return original_query(value)

    bridge.runtime.query = query
    agent = SimpleNamespace(
        _memory_store=None,
        _session_db=NativeDb(),
        session_id="current-session",
    )
    bridge.attach(agent)
    try:
        results = agent._session_db.search_messages(
            query="adapter history",
            role_filter=["user"],
            limit=5,
            offset=0,
        )
        projected_session = agent._session_db.get_session("adapter-session")
        projected_conversation = agent._session_db.get_messages_as_conversation(
            "adapter-session"
        )
    finally:
        bridge.close()

    assert results[0]["id"] == 42
    assert results[0]["source"] == "remote-backend"
    assert projected_session["model"] == "fixture-model"
    assert projected_conversation == list(conversation)


def test_lifecycle_events_join_ledger_without_context_content(tmp_path: Path) -> None:
    fixture = run_sm01_preference_fixture()

    def build() -> LifecycleLedgerObserver:
        observer = LifecycleLedgerObserver(
            variant="native+adapter+ledger",
            trace_id="trace-sm01-fixture",
            family_id="SM01",
            stage="learn",
        )
        observer.record_snapshot(fixture.snapshot)
        for event in fixture.events:
            observer.record(event)
        return observer

    first = build()
    second = build()
    assert first.events == second.events
    assert [event["kind"] for event in first.events] == [
        "context_snapshot",
        "plan_created",
        "plan_validated",
        "dry_run_mutation",
    ]
    assert all(event["episodeId"] == fixture.snapshot.episode_id for event in first.events)
    assert all(event["sessionId"] == fixture.snapshot.session_id for event in first.events)
    assert all(event["snapshotId"] == fixture.snapshot.snapshot_id for event in first.events)
    assert first.events[-1]["data"]["mutationId"] == fixture.receipts[0].mutation_id
    event_count = len(first.events)
    first.record(fixture.events[-1])
    assert len(first.events) == event_count
    with pytest.raises(ValueError, match="conflicting lifecycle ledger event"):
        first.record(replace(
            fixture.events[-1],
            resources=RawResourceUsage(input_tokens=1, model_requests=1),
        ))

    plan_only = LifecycleLedgerObserver(
        variant="native+adapter+ledger",
        trace_id="trace-sm01-fixture",
        family_id="SM01",
        stage="learn",
    )
    plan_only.record(fixture.events[-1])
    assert plan_only.events[0]["eventId"] == first.events[-1]["eventId"]

    usage_observer = LifecycleLedgerObserver(
        variant="native+adapter+ledger",
        trace_id="trace-sm01-fixture",
        family_id="SM01",
        stage="learn",
    )
    usage_observer.record(replace(
        fixture.events[0],
        resources=RawResourceUsage(
            input_tokens=100,
            output_tokens=20,
            cache_read_tokens=30,
            cache_write_tokens=4,
            reasoning_tokens=7,
            model_requests=2,
            retry_count=1,
        ),
    ))
    assert usage_observer.events[0]["data"]["resources"] == {
        "schemaVersion": 1,
        "inputTokens": 100,
        "outputTokens": 20,
        "cacheReadTokens": 30,
        "cacheWriteTokens": 4,
        "reasoningTokens": 7,
        "modelRequests": 2,
        "retryCount": 1,
        "durationMs": None,
        "storageBytes": 0,
    }

    output = tmp_path / "lifecycle.jsonl"
    first.write(output)
    serialized = output.read_text(encoding="utf-8")
    assert PRIVATE_PREFERENCE not in serialized
    assert "current task is complete" not in serialized
    assert "/mnt/" not in serialized


def test_lifecycle_ledger_appends_incrementally_and_resumes(tmp_path: Path) -> None:
    fixture = run_sm01_preference_fixture()
    output = tmp_path / "lifecycle.jsonl"

    first = LifecycleLedgerObserver(
        variant="native+adapter+ledger",
        trace_id="trace-sm01-fixture",
        family_id="SM01",
        stage="learn",
        output_path=output,
    )
    first.record_snapshot(fixture.snapshot)
    first.record(fixture.events[0])
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2

    resumed = LifecycleLedgerObserver(
        variant="native+adapter+ledger",
        trace_id="trace-sm01-fixture",
        family_id="SM01",
        stage="learn",
        output_path=output,
    )
    assert resumed.events == first.events
    resumed.record(fixture.events[0])
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
    resumed.record(fixture.events[1])
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3

    output.write_text(output.read_text(encoding="utf-8") + "not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed lifecycle ledger event"):
        LifecycleLedgerObserver(
            variant="native+adapter+ledger",
            trace_id="trace-sm01-fixture",
            output_path=output,
        )


def test_lifecycle_ledger_reloads_external_append_and_rejects_symlink_lock(
    tmp_path: Path,
) -> None:
    fixture = run_sm01_preference_fixture()
    output = tmp_path / "lifecycle-reload.jsonl"
    first = LifecycleLedgerObserver(
        variant="native+adapter+ledger",
        trace_id="trace-lifecycle-reload",
        output_path=output,
    )
    first.record(fixture.events[0])
    second = LifecycleLedgerObserver(
        variant="native+adapter+ledger",
        trace_id="trace-lifecycle-reload",
        output_path=output,
    )
    second.record(fixture.events[1])
    assert len(first.events) == 2

    output.unlink()
    assert first.events == ()

    lock_target = tmp_path / "lock-target"
    lock_target.write_text("", encoding="utf-8")
    output.with_name(output.name + ".lock").unlink()
    output.with_name(output.name + ".lock").symlink_to(lock_target)
    with pytest.raises(ValueError, match="lock.*symlink"):
        first.events


def test_lifecycle_ledger_restart_rejects_schema_invalid_object(tmp_path: Path) -> None:
    output = tmp_path / "lifecycle.jsonl"
    output.write_text(
        json.dumps({"eventId": "evt.invalid", "kind": "context_snapshot"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="malformed lifecycle ledger event"):
        LifecycleLedgerObserver(
            variant="native+adapter+ledger",
            trace_id="trace-invalid",
            output_path=output,
        )


def test_lifecycle_ledger_records_content_free_pre_snapshot_rejection(tmp_path: Path) -> None:
    output = tmp_path / "lifecycle.jsonl"
    observer = LifecycleLedgerObserver(
        variant="native+adapter+ledger",
        trace_id="trace-host-failure",
        output_path=output,
    )
    observer.record_boundary_rejection(
        run_id="run-host-failure",
        episode_id="episode-host-failure",
        session_id="session-host-failure",
        task_id="task-host-failure",
        boundary_id="boundary-stable",
        trigger="task_completed",
        reason_code="host_valueerror",
    )
    event = observer.events[0]
    assert event["kind"] == "boundary_rejected"
    assert event["snapshotId"] is None
    assert event["data"] == {
        "evaluationId": "boundary-stable",
        "boundaryId": "boundary-stable",
        "trigger": "task_completed",
        "status": "rejected",
        "reasonCodes": ["host_valueerror"],
    }


def test_live_bridge_persists_pre_snapshot_failure_without_failing_task(tmp_path: Path) -> None:
    home = _hermes_home(tmp_path / "home")
    lifecycle_path = tmp_path / "artifacts" / "lifecycle.jsonl"
    bridge = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.ADAPTER_LEDGER),
        evidence_path=tmp_path / "artifacts" / "memory.jsonl",
        run_id="run-host-failure",
        trace_id="trace-host-failure",
        episode_id="episode-host-failure",
        session_id="session-host-failure",
        task_id="task-host-failure",
        experiment_variant="with_persistence",
        lifecycle_config=HermesLifecycleConfig(evaluator_mode="deterministic"),
        lifecycle_evidence_path=lifecycle_path,
        lifecycle_receipt_path=tmp_path / "artifacts" / "receipts.json",
    )
    bridge.attach(SimpleNamespace(
        _memory_store=None,
        _session_db=None,
        session_id="native-session",
    ))
    bridge.on_task_completed({"completed": True})
    bridge.close()

    assert bridge.lifecycle_results == ()
    assert bridge.lifecycle_failures == (
        ("task_completed", "ValueError"),
    )
    events = [
        json.loads(line)
        for line in lifecycle_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["kind"] for event in events] == ["boundary_rejected"]
    assert all(event["snapshotId"] is None for event in events)
    assert PRIVATE_PREFERENCE not in json.dumps(events, ensure_ascii=True)


def test_pure_process_archive_is_incremental_and_restart_recoverable(
    tmp_path: Path,
) -> None:
    """A process event survives before task completion or bridge shutdown."""

    home = _hermes_home(tmp_path)
    evidence_path = tmp_path / "artifacts" / "process-events.jsonl"
    bridge = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
        evidence_path=evidence_path,
        run_id="run-archive-incremental",
        trace_id="trace-archive-incremental",
        episode_id="episode-archive-incremental",
        session_id="session-archive-incremental",
        task_id="task-archive-incremental",
        experiment_variant="native+ledger",
    )
    bridge._record_process_observation(
        kind=ProcessEventKind.TASK_OUTCOME,
        status="success",
        host_event_id="event.archive-incremental",
        source_revision="revision.archive-incremental",
        input_payload={"task_id": "task-archive-incremental"},
        output_payload={"outcome_digest": "a" * 64},
        reason_codes=("task_completed",),
        execution_receipt_ids=("receipt.archive-incremental",),
    )

    archive_path = home / ".rsimem" / "pure_process_event_archive.jsonl"
    archived = JsonPureProcessEventArchive(archive_path).records()
    assert len(archived) == 1
    event_id = archived[0].event_id
    bridge.close()

    # Simulate a crash window in which the run-scoped ledger reached disk but
    # the shared archive did not.  Constructor recovery must project the
    # durable ledger event before any future-task join is attempted.
    archive_path.unlink()

    # Reopening the run replays its process ledger and reconciles the shared
    # archive idempotently; the event remains available for delayed joins.
    restarted = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
        evidence_path=evidence_path,
        run_id="run-archive-incremental",
        trace_id="trace-archive-incremental",
        episode_id="episode-archive-incremental",
        session_id="session-archive-incremental",
        task_id="task-archive-incremental",
        experiment_variant="native+ledger",
    )
    try:
        replayed = JsonPureProcessEventArchive(archive_path).records()
        assert tuple(event.event_id for event in replayed) == (event_id,)
    finally:
        restarted.close()


def test_live_bridge_trigger_and_source_decisions_replay_from_persisted_evidence(tmp_path: Path) -> None:
    from hermes_state import SessionDB

    home = _hermes_home(tmp_path / "home")
    db = SessionDB(home / "state.db")
    session_id = "session-policy-replay"
    db.create_session(session_id, "past_bench", model="fixture-model")
    db.append_message(session_id, "user", "Replay this completed task.")
    db.append_message(session_id, "assistant", "Done.")
    artifacts = tmp_path / "artifacts"
    bridge = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
        evidence_path=artifacts / "memory.jsonl",
        run_id="run-policy-replay",
        trace_id="trace-policy-replay",
        episode_id="episode-policy-replay",
        session_id=session_id,
        task_id="task-policy-replay",
        experiment_variant="with_persistence",
        lifecycle_config=HermesLifecycleConfig(evaluator_mode="deterministic"),
        lifecycle_evidence_path=artifacts / "lifecycle.jsonl",
        lifecycle_receipt_path=artifacts / "lifecycle-receipts.json",
    )
    agent = SimpleNamespace(_memory_store=None, _session_db=db, session_id=session_id)
    bridge.attach(agent)
    bridge.on_task_completed({"completed": True})
    first = bridge.policy_evidence
    first_process = bridge.process_feedback
    first_process_ids = bridge.process_feedback_event_ids
    first_process_digest = bridge.process_feedback_digest
    bridge.close()
    assert {event["layer"] for event in first} == {"trigger", "source_selection"}
    assert {event["evidencePlane"] for event in first} == {"pure_process"}
    assert {event["evidenceSource"] for event in first} == {"runtime_observation"}

    restarted = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
        evidence_path=artifacts / "memory.jsonl",
        run_id="run-policy-replay",
        trace_id="trace-policy-replay",
        episode_id="episode-policy-replay",
        session_id=session_id,
        task_id="task-policy-replay",
        experiment_variant="with_persistence",
        lifecycle_config=HermesLifecycleConfig(evaluator_mode="deterministic"),
        lifecycle_evidence_path=artifacts / "lifecycle.jsonl",
        lifecycle_receipt_path=artifacts / "lifecycle-receipts.json",
    )
    restarted.attach(agent)
    restarted.on_task_completed({"completed": True})
    assert restarted.policy_evidence == first
    assert restarted.process_feedback == first_process
    assert restarted.process_feedback_event_ids == first_process_ids
    assert restarted.process_feedback_digest == first_process_digest
    restarted.close()
    db.close()


def test_live_bridge_static_writeback_runs_only_at_task_completion(tmp_path: Path) -> None:
    from hermes_state import SessionDB

    home = _hermes_home(tmp_path / "home")
    db = SessionDB(home / "state.db")
    session_id = "session-static-live"
    db.create_session(session_id, "past_bench", model="fixture-model")
    db.append_message(session_id, "user", "Always use pipe-delimited output.")
    db.append_message(session_id, "assistant", "Understood.")
    artifacts = tmp_path / "artifacts"
    client = FakeCompletionClient({
        POLICY_FACT_EXTRACTION_PROMPT.artifact.prompt_id: json.dumps({
            "facts": ["Use pipe-delimited output for future tables."],
        }),
        POLICY_INTERNAL_OPERATION_PROMPT.artifact.prompt_id: json.dumps({
            "operations": [{
                "fact_index": 0,
                "action": "add",
                "candidate_id": None,
            }],
        }),
    })
    bridge = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
        evidence_path=artifacts / "memory.jsonl",
        run_id="run-static-live",
        trace_id="trace-static-live",
        episode_id="episode-static-live",
        session_id=session_id,
        task_id="SM01-static-live",
        experiment_variant="static-rsimem",
        lifecycle_config=HermesLifecycleConfig(evaluator_mode="deterministic"),
        lifecycle_evidence_path=artifacts / "lifecycle.jsonl",
        lifecycle_receipt_path=artifacts / "lifecycle-receipts.json",
        static_writeback_config=StaticSemanticWritebackConfig(
            mode="static_utility"
        ),
        static_completion_client=client,
    )
    bridge.attach(SimpleNamespace(
        _memory_store=None,
        _session_db=db,
        session_id=session_id,
    ))

    bridge.on_task_completed({"completed": False})
    assert bridge.static_results == ()
    assert client.calls == ()
    bridge.on_task_completed({"completed": True})
    assert len(bridge.static_results) == 1
    assert bridge.static_results[0].writeback.logical_exit is True
    assert bridge.static_failures == ()
    lifecycle_events = [
        json.loads(line)
        for line in (artifacts / "lifecycle.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["kind"] for event in lifecycle_events].count(
        "memory_ingestion"
    ) == 1
    utility_events = [
        event
        for event in lifecycle_events
        if event["kind"] == "static_utility_decisions"
    ]
    assert len(utility_events) == 1
    assert utility_events[0]["data"]["decisionCount"] == 3
    assert [
        decision["target"]
        for decision in utility_events[0]["data"]["decisions"]
    ] == ["generation", "retrieval", "internal_operation"]
    assert "pipe-delimited" not in json.dumps(utility_events[0], sort_keys=True)
    bridge.close()

    assert len(bridge.lifecycle_results) == 1
    assert len(bridge.trigger_observations) == 1
    assert len(bridge.source_selection_decisions) == 1
    assert bridge.source_selection_decisions[0].source_revision == bridge.lifecycle_results[0].snapshot.context_revision
    policy_events = [
        json.loads(line)
        for line in (artifacts / "rsimem_policy_decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {event["layer"] for event in policy_events} >= {
        "trigger",
        "source_selection",
        "extraction",
        "admission",
        "commit",
    }
    assert all(event["lineageId"] for event in policy_events)
    process_events = [
        json.loads(line)
        for line in (artifacts / "rsimem_process_feedback.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {event["policy_layer"] for event in process_events if event["policy_layer"]} >= {
        "trigger", "source_selection", "extraction", "admission", "commit",
    }
    assert len(bridge.static_results) == 1
    assert len(client.calls) == 2
    serialized = (artifacts / "memory.jsonl").read_text(encoding="utf-8")
    assert '"kind": "mutation_requested"' in serialized
    assert '"kind": "mutation_committed"' in serialized
    assert "pipe-delimited" not in serialized
    operations = (artifacts / "rsimem_semantic_operations.jsonl").read_text(
        encoding="utf-8"
    )
    assert "pipe-delimited" not in operations
    assert (home / ".rsimem" / "semantic_mutation_receipts.json").exists()

    restarted = build_configured_hermes_runtime(
        home,
        HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
    )
    hits = restarted.query(MemoryQuery(
        MemoryKind.SEMANTIC,
        "",
        namespace="user",
        limit=100,
    ))
    assert any("pipe-delimited" in hit.artifact.content for hit in hits)
    restarted.close()
    db.close()


def test_live_bridge_does_not_compile_reflection_episode(tmp_path: Path) -> None:
    """Reflection is a review episode, not a second extraction boundary."""

    from hermes_state import SessionDB

    home = _hermes_home(tmp_path / "home")
    db = SessionDB(home / "state.db")
    session_id = "session-reflection"
    db.create_session(session_id, "past_bench", model="fixture-model")
    db.append_message(session_id, "user", "Reflect on the completed task.")
    db.append_message(session_id, "assistant", "Nothing to save.")
    client = FakeCompletionClient({
        POLICY_FACT_EXTRACTION_PROMPT.artifact.prompt_id: json.dumps({
            "facts": ["This must never be extracted from reflection."],
        }),
        POLICY_INTERNAL_OPERATION_PROMPT.artifact.prompt_id: json.dumps({
            "operations": [],
        }),
    })
    artifacts = tmp_path / "reflection-artifacts"
    bridge = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
        evidence_path=artifacts / "memory.jsonl",
        run_id="run-reflection",
        trace_id="trace-reflection",
        episode_id="episode-reflection",
        session_id=session_id,
        task_id="SM01_LEARN_B_001_REFLECT",
        experiment_variant="static-extraction-rsimem",
        family_id="SM01_preference_adoption",
        stage="reflection",
        static_writeback_config=StaticSemanticWritebackConfig(
            mode="static",
            feedback_contract="sm01_tsv_v1",
        ),
        static_completion_client=client,
    )
    bridge.attach(SimpleNamespace(
        _memory_store=None,
        _session_db=db,
        session_id=session_id,
    ))

    bridge.on_task_completed({"completed": True})

    assert bridge.static_results == ()
    assert bridge.static_failures == ()
    assert client.calls == ()
    assert not (home / ".rsimem" / "extraction_sources.jsonl").exists()
    bridge.close()
    db.close()


def test_live_bridge_compiles_completed_task_without_lifecycle_evaluator(
    tmp_path: Path,
) -> None:
    from hermes_state import SessionDB

    home = _hermes_home(tmp_path / "home")
    db = SessionDB(home / "state.db")
    session_id = "session-direct-compilation"
    db.create_session(session_id, "past_bench", model="fixture-model")
    db.append_message(session_id, "user", "Always use pipe-delimited output.")
    db.append_message(session_id, "assistant", "Understood.")
    client = FakeCompletionClient({
        POLICY_FACT_EXTRACTION_PROMPT.artifact.prompt_id: json.dumps({
            "facts": ["Use pipe-delimited output for future tables."],
        }),
        POLICY_INTERNAL_OPERATION_PROMPT.artifact.prompt_id: json.dumps({
            "operations": [{
                "fact_index": 0,
                "action": "add",
                "candidate_id": None,
            }],
        }),
    })
    bridge = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
        evidence_path=tmp_path / "artifacts" / "memory.jsonl",
        run_id="run-direct-compilation",
        trace_id="trace-direct-compilation",
        episode_id="episode-direct-compilation",
        session_id=session_id,
        task_id="SM01-direct-compilation",
        experiment_variant="static-rsimem",
        static_writeback_config=StaticSemanticWritebackConfig(mode="static"),
        static_completion_client=client,
    )
    bridge.attach(SimpleNamespace(
        _memory_store=None,
        _session_db=db,
        session_id=session_id,
    ))

    bridge.on_task_completed({"completed": True})
    bridge.on_task_completed({"completed": True})
    bridge.close()

    assert bridge.lifecycle is None
    assert bridge.lifecycle_results == ()
    assert bridge.lifecycle_failures == ()
    assert len(bridge.static_results) == 1
    assert len(bridge.trigger_observations) == 1
    assert len(bridge.source_selection_decisions) == 1
    assert bridge.source_selection_decisions[0].action.value == "RUN"
    assert bridge.static_results[0].writeback.logical_exit is True
    assert len(client.calls) == 2
    pure_sources = JsonPureExtractionSourceRecordStore(
        home / ".rsimem" / "pure_extraction_sources.jsonl"
    ).records()
    assert len(pure_sources) == 1
    assert pure_sources[0].activation.compilation_id == bridge.static_results[0].compilation_id
    assert bridge.unbound_process_signal_cases
    assert bridge.unbound_process_signal_cases[0].analysis_protocol_id is None
    db.close()


def test_pure_feedback_ambiguous_evidence_stays_unresolved(
    tmp_path: Path,
) -> None:
    """Multiple joins never become authoritative by append order or overlap."""

    from hermes_state import SessionDB

    home = _hermes_home(tmp_path / "home")
    db = SessionDB(home / "state.db")
    session_id = "session-ambiguous-source"
    db.create_session(session_id, "past_bench", model="fixture-model")
    db.append_message(session_id, "user", "Remember this durable preference.")

    client = FakeCompletionClient({
        POLICY_FACT_EXTRACTION_PROMPT.artifact.prompt_id: json.dumps({
            "facts": ["Always use pipe-delimited output."],
        }),
        POLICY_INTERNAL_OPERATION_PROMPT.artifact.prompt_id: json.dumps({
            "operations": [{"fact_index": 0, "action": "add", "candidate_id": None}],
        }),
    })
    source_bridge = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
        evidence_path=tmp_path / "source" / "memory.jsonl",
        run_id="run.ambiguous-source",
        trace_id="trace.ambiguous-source",
        episode_id="episode.ambiguous-source",
        session_id=session_id,
        task_id="task.ambiguous-source",
        experiment_variant="fixture",
        static_writeback_config=StaticSemanticWritebackConfig(mode="static"),
        static_completion_client=client,
    )
    source_bridge.attach(SimpleNamespace(
        _memory_store=None,
        _session_db=db,
        session_id=session_id,
    ))
    source_bridge.on_task_completed({
        "completed": True,
        "messages": [{"role": "user", "content": "Remember this durable preference."}],
    })
    source = source_bridge.pure_extraction_sources[0]
    artifact_id = next(
        fact.artifact_id for fact in source.source.facts if fact.artifact_id is not None
    )
    source_bridge.close()
    db.close()

    future_db = SessionDB(home / "state.db")
    future_session = "session-ambiguous-future"
    future_db.create_session(future_session, "past_bench", model="fixture-model")
    future_bridge = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
        evidence_path=tmp_path / "future" / "memory.jsonl",
        run_id="run.ambiguous-source",
        trace_id="trace.ambiguous-future",
        episode_id="episode.ambiguous-future",
        session_id=future_session,
        task_id="task.ambiguous-future",
        experiment_variant="fixture",
        static_writeback_config=StaticSemanticWritebackConfig(mode="static"),
        static_completion_client=client,
    )

    # Two opportunities, two use records and two set bindings share the same
    # source provenance.  A valid resolver must retain the source record but
    # leave every selected join empty rather than choosing the first row.
    provenance = source.provenance_id
    for suffix in ("a", "b"):
        future_bridge._opportunity_evidence_log.append(OpportunityEvidence.create(
            source_surface=OpportunitySurface.TOOL_SCHEMA,
            semantic_requirement="preference.summary.tsv",
            observation_time="2026-08-31T00:00:00Z",
            operation_id=f"op.ambiguous-opportunity.{suffix}",
            provenance_id=provenance,
            source_payload={"tool_name": "fixture_apply", "variant": suffix},
        ))
        future_bridge._memory_use_evidence_log.append(MemoryUseEvidence.create(
            artifact_ids=(artifact_id,),
            retrieval_operation_id=f"op.ambiguous-retrieval.{suffix}",
            retrieved_artifact_ids=(artifact_id,),
            injection_operation_id=f"op.ambiguous-injection.{suffix}",
            injected_artifact_ids=(artifact_id,),
            downstream_operation_id=f"op.ambiguous-use.{suffix}",
            used_artifact_ids=(artifact_id,),
            outcome_operation_id=f"op.ambiguous-outcome.{suffix}",
            outcome_kind=OutcomeEvidenceKind.STATE_TRANSITION,
            outcome_success=True,
            observation_cutoff="2026-08-31T00:01:00Z",
            provenance_id=provenance,
        ))
        future_bridge._artifact_set_binding_log.append(
            ArtifactSetSemanticBinding.create(
                semantic_unit_id=f"semantic-unit.ambiguous.{suffix}",
                semantic_key="preference.summary.tsv",
                member_artifact_ids=(artifact_id,),
                member_fact_ids=tuple(
                    fact.fact_id
                    for fact in source.source.facts
                    if fact.artifact_id is not None
                ),
                complete=True,
                source_digest=source.source_projection_digest,
                provenance_id=provenance,
            )
        )

    future_bridge._record_pure_extraction_feedback({
        "completed": True,
        "messages": [],
    })
    records = future_bridge.pure_extraction_feedback_store.records()
    assert len(records) == 1
    feedback = records[0]
    assert feedback.attribution.value == "unresolved"
    assert feedback.opportunity is None
    assert feedback.memory_use is None
    assert feedback.artifact_set_binding is None
    # Opportunity resolution is evaluated first, so the unresolved record
    # carries that deterministic reason while still proving that no memory
    # join was selected.
    assert feedback.reason_codes == ("opportunity_not_observed",)
    future_bridge.close()
    future_db.close()


def test_memory_use_recorder_rejects_ambiguous_artifact_sets(
    tmp_path: Path,
) -> None:
    """Future use evidence is suppressed when set ownership is ambiguous."""

    home = _hermes_home(tmp_path / "home")
    bridge = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
        evidence_path=tmp_path / "future" / "memory.jsonl",
        run_id="run.ambiguous-set",
        trace_id="trace.ambiguous-set",
        episode_id="episode.ambiguous-set",
        session_id="session.ambiguous-set",
        task_id="task.ambiguous-set",
        experiment_variant="fixture",
    )
    artifact_id = "artifact.ambiguous-set.v1"
    for suffix in ("a", "b"):
        bridge._artifact_set_binding_log.append(
            ArtifactSetSemanticBinding.create(
                semantic_unit_id=f"semantic-unit.recorder.{suffix}",
                semantic_key="preference.summary.tsv",
                member_artifact_ids=(artifact_id,),
                member_fact_ids=(f"fact.recorder.{suffix}",),
                complete=True,
                source_digest="a" * 64,
                provenance_id="provenance.recorder.v1",
            )
        )
    future = SemanticFutureEvidence(
        query_operation_id="op.recorder.query",
        retrieval_operation_id="op.recorder.retrieve",
        injection_operation_id="op.recorder.inject",
        memory_artifact_ids=(artifact_id,),
        memory_revisions=("revision.recorder.v1",),
        injection_artifact_id="artifact.injection.recorder",
        injected_artifact_ids=(artifact_id,),
    )
    outcome = SemanticOutcomeEvidence(
        use_operation_id="op.recorder.use",
        outcome_operation_id="op.recorder.outcome",
        used_artifact_ids=(artifact_id,),
        outcome_status=OperationStatus.SUCCESS,
    )
    bridge._record_memory_use_evidence(future, outcome, {"completed": True})
    assert bridge.memory_use_evidence == ()
    bridge.close()


def test_pure_process_runtime_fixture_closes_extraction_to_use_and_tool_outcome(
    tmp_path: Path,
) -> None:
    """Exercise the deployment-only chain without family/grader semantics.

    The application provider is deliberately tiny and deterministic: a public
    tool schema identifies the future opportunity, while the owner-controlled
    matcher binds both extracted facts to one semantic unit.  The bridge must
    persist source, artifact-set, retrieval/exposure, exact tool closure and
    pure feedback across a learn -> future restart.
    """

    from hermes_state import SessionDB
    from rsimem.memory.opportunity import ApplicationOpportunitySchema
    from rsimem.memory.tool_exact_join import resolve_tool_call_result

    key = "application.fixture.preference"
    raw_schema = _fixture_application_schema()
    application_schema = ApplicationOpportunitySchema.from_payload(
        raw_schema["application_contract"]
    )

    def visible_messages(result: dict[str, object]) -> tuple[dict[str, object], ...]:
        raw = result.get("messages")
        return tuple(item for item in raw if isinstance(item, dict)) if isinstance(raw, (list, tuple)) else ()

    def opportunity_provider(result: dict[str, object]):
        messages = visible_messages(result)
        source_provenance = result.get("rsimem_source_provenance_id")
        source_records = result.get("rsimem_source_records")
        has_apply_call = any(
            isinstance(call, dict)
            and isinstance(call.get("function"), dict)
            and call["function"].get("name") == "fixture_apply"
            for message in messages
            for call in (message.get("tool_calls") or ())
            if isinstance(message.get("tool_calls"), (list, tuple))
        )
        if isinstance(source_provenance, str) and any(
            "durable preference" in str(message.get("content") or "").casefold()
            for message in messages
            if message.get("role") == "user"
        ):
            return (OpportunityEvidence.create(
                source_surface=OpportunitySurface.CURRENT_INPUT,
                semantic_requirement=key,
                observation_time="2026-08-31T00:00:00Z",
                operation_id="opportunity.fixture.source",
                provenance_id=source_provenance,
                source_payload={"visible_input": "durable preference"},
            ),)
        if not has_apply_call or not isinstance(source_records, (list, tuple)):
            return ()
        values = []
        for record in source_records:
            if not isinstance(record, dict) or key not in tuple(record.get("semantic_keys") or ()):
                continue
            provenance = record.get("provenance_id")
            if not isinstance(provenance, str):
                continue
            values.append(OpportunityEvidence.create(
                source_surface=OpportunitySurface.APPLICATION_SCHEMA,
                semantic_requirement=key,
                observation_time="2026-08-31T00:00:00Z",
                operation_id=(
                    "opportunity.fixture.future."
                    + str(result.get("boundary") or "initial")
                ),
                provenance_id=provenance,
                source_payload={"tool_name": "fixture_apply"},
                application_schema=application_schema,
            ))
        return tuple(values)

    matcher_calls = []

    def fact_keys_provider(
        record,
        *,
        fact_contents=None,
        application_schema=None,
    ):
        matcher_calls.append((fact_contents, application_schema))
        assert isinstance(fact_contents, dict)
        assert set(fact_contents) == {
            fact.fact_id
            for fact in record.source.facts
        }
        assert application_schema == raw_schema
        return {
            fact.fact_id: (key,)
            for fact in record.source.facts
            if fact.artifact_id is not None
        }

    def binding_provider(source, *, provenance_id=None):
        facts = tuple(
            fact for fact in source.facts
            if fact.artifact_id is not None and fact.disposition.value == "persisted"
        )
        if provenance_id is None or len(facts) < 2 or len(facts) != len(source.facts):
            return ()
        if any(fact.semantic_keys != (key,) for fact in facts):
            return ()
        return (ArtifactSetSemanticBinding.create(
            semantic_unit_id="fixture.semantic-unit.preference",
            semantic_key=key,
            member_artifact_ids=tuple(fact.artifact_id for fact in facts),
            member_fact_ids=tuple(fact.fact_id for fact in facts),
            complete=True,
            source_digest=source.source_projection_digest,
            provenance_id=provenance_id,
        ),)

    class NativeStore:
        def format_for_system_prompt(self, target: str) -> str | None:
            if target != "user":
                return None
            content = (home / "memories" / "USER.md").read_text(encoding="utf-8").strip()
            return "MEMORY\n" + content if content else None

    home = tmp_path / "home"
    (home / "memories").mkdir(parents=True)
    (home / "memories" / "MEMORY.md").write_text("", encoding="utf-8")
    (home / "memories" / "USER.md").write_text("", encoding="utf-8")

    def client_for(facts: tuple[str, ...], operations: list[dict[str, object]]):
        return FakeCompletionClient({
            POLICY_FACT_EXTRACTION_PROMPT.artifact.prompt_id: json.dumps({"facts": list(facts)}),
            POLICY_INTERNAL_OPERATION_PROMPT.artifact.prompt_id: json.dumps({"operations": operations}),
        })

    learn_db = SessionDB(home / "state.db")
    learn_session = "session.fixture.learn"
    learn_db.create_session(learn_session, "fixture", model="fixture-model")
    learn_db.append_message(learn_session, "user", "Remember this durable preference.")
    learn_db.append_message(learn_session, "assistant", "Saved.")
    facts = ("Prefer fixture format.", "Apply the preference only to fixture tasks.")
    learn_client = client_for(
        facts,
        [{"fact_index": index, "action": "add", "candidate_id": None} for index in range(2)],
    )
    learn_bridge = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
        evidence_path=tmp_path / "learn" / "memory.jsonl",
        run_id="run.fixture",
        trace_id="trace.fixture.learn",
        episode_id="episode.fixture.learn",
        session_id=learn_session,
        task_id="task.fixture.learn",
        experiment_variant="fixture",
        static_writeback_config=StaticSemanticWritebackConfig(mode="static"),
        static_completion_client=learn_client,
        opportunity_evidence_provider=opportunity_provider,
        artifact_set_binding_provider=binding_provider,
        pure_extraction_fact_semantic_keys_provider=fact_keys_provider,
        application_opportunity_schema=raw_schema,
    )
    learn_bridge.attach(SimpleNamespace(
        _memory_store=NativeStore(), _session_db=learn_db, session_id=learn_session,
    ))
    learn_bridge.on_task_completed({
        "completed": True,
        "messages": [{"role": "user", "content": "Remember this durable preference."}],
    })
    assert len(matcher_calls) == 1
    assert all(isinstance(value, str) and value for value in matcher_calls[0][0].values())
    learn_bridge.close()
    learn_db.close()

    source_records = JsonPureExtractionSourceRecordStore(
        home / ".rsimem" / "pure_extraction_sources.jsonl"
    ).records()
    assert len(source_records) == 1
    assert source_records[0].source.available_semantic_keys == (key,)
    assert all(fact.semantic_keys == (key,) for fact in source_records[0].source.facts)
    bindings = learn_bridge.artifact_set_bindings
    assert len(bindings) == 1
    member_ids = bindings[0].member_artifact_ids

    future_db = SessionDB(home / "state.db")
    future_session = "session.fixture.future"
    future_db.create_session(future_session, "fixture", model="fixture-model")
    future_db.append_message(future_session, "user", "Apply the saved preference.")
    future_db.append_message(future_session, "assistant", "Applied.")
    future_client = client_for((), [])
    future_bridge = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
        evidence_path=tmp_path / "future" / "memory.jsonl",
        run_id="run.fixture",
        trace_id="trace.fixture.future",
        episode_id="episode.fixture.future",
        session_id=future_session,
        task_id="task.fixture.future",
        experiment_variant="fixture",
        static_writeback_config=StaticSemanticWritebackConfig(mode="static"),
        static_completion_client=future_client,
        opportunity_evidence_provider=opportunity_provider,
        artifact_set_binding_provider=binding_provider,
        pure_extraction_fact_semantic_keys_provider=fact_keys_provider,
        memory_use_attribution_provider=lambda future, result: (
            future.memory_artifact_ids if result.get("memory_used") is True else ()
        ),
        application_opportunity_schema=raw_schema,
    )
    future_bridge.attach(SimpleNamespace(
        _memory_store=NativeStore(), _session_db=future_db, session_id=future_session,
    ))
    prompt = future_bridge._agent._memory_store.format_for_system_prompt("user")
    assert prompt is not None
    future_bridge.on_task_completed({
        "completed": True,
        "memory_used": True,
        "messages": [
            {"role": "user", "content": "Apply the saved preference."},
            {"role": "assistant", "tool_calls": [{"id": "call.fixture", "function": {"name": "fixture_apply", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call.fixture", "id": "result.fixture", "content": '{"success": true}'},
        ],
    })
    joins = future_bridge.tool_call_result_joins
    assert len(joins) == 1
    assert resolve_tool_call_result(joins[0]).status is ToolJoinResolutionStatus.COMPLETE
    from rsimem.memory.opportunity import JsonApplicationOpportunitySchemaRegistry

    feedback = JsonPureExtractionFeedbackRecordStore(
        tmp_path / "future" / "rsimem_pure_extraction_feedback.jsonl",
        schema_registry=JsonApplicationOpportunitySchemaRegistry(
            home / ".rsimem" / "application_opportunity_schemas.jsonl"
        ),
    ).records()
    assert len(feedback) == 1
    assert feedback[0].attribution.value == "attributable_success", (
        feedback[0].reason_codes,
        [item.payload() for item in future_bridge.opportunity_evidence],
        future_bridge.memory_use_evidence,
        [item.payload() for item in future_bridge.artifact_set_bindings],
        source_records[0].payload(),
    )
    assert feedback[0].artifact_set_binding is not None
    assert feedback[0].artifact_set_binding.member_artifact_ids == member_ids
    assert feedback[0].tool_joins[0].memory_use_operation_id is not None
    assert set(future_bridge.memory_use_evidence[0].retrieved_artifact_ids) == set(member_ids)
    joined_cases = tuple(
        case for case in future_bridge.unbound_process_signal_cases
        if case.source_observed and case.retrieval_observed
    )
    assert joined_cases, "source/future pure-process join was not materialized"
    assert any(
        case.extraction_observed
        and case.persistence_observed
        and case.outcome_observed
        and case.extraction_attributable
        for case in joined_cases
    )

    # A later future boundary for the same source must use only the evidence
    # emitted at that boundary.  Historical opportunity/use rows remain in
    # the append-only logs for audit, but cannot make the new join ambiguous
    # or silently reuse an old memory-use observation.
    future_bridge._record_pure_extraction_feedback(
        {
            "completed": True,
            "boundary": "second",
            "messages": [
                {"role": "user", "content": "Apply the saved preference."},
                {"role": "assistant", "tool_calls": [{
                    "id": "call.fixture.second",
                    "function": {"name": "fixture_apply", "arguments": "{}"},
                }]},
                {"role": "tool", "tool_call_id": "call.fixture.second",
                 "id": "result.fixture.second", "content": '{"success": true}'},
            ],
        },
        current_memory_uses=(),
    )
    feedback_rows = JsonPureExtractionFeedbackRecordStore(
        tmp_path / "future" / "rsimem_pure_extraction_feedback.jsonl",
        schema_registry=JsonApplicationOpportunitySchemaRegistry(
            home / ".rsimem" / "application_opportunity_schemas.jsonl"
        ),
    ).records()
    assert len(feedback_rows) == 2
    assert feedback_rows[1].opportunity is not None
    assert feedback_rows[1].opportunity.operation_id.endswith(".second")
    assert feedback_rows[1].memory_use is None

    pure_events = PureProcessCorpus.create(
        tuple(learn_bridge.process_feedback) + tuple(future_bridge.process_feedback)
    )
    assert any(event.kind is ProcessEventKind.EXTRACTION for event in pure_events.events)
    assert any(event.kind is ProcessEventKind.TOOL_RESULT for event in pure_events.events)
    assert all(event.evidence_plane.value == "pure_process" for event in pure_events.events)
    from rsimem.memory.process_feedback import audit_process_events
    assert audit_process_events(future_bridge.process_feedback) == ()
    future_bridge.close()
    future_db.close()


def test_hermes_tool_identity_gaps_are_type_mismatch_not_exact_closure(
    tmp_path: Path,
) -> None:
    """Synthetic IDs preserve diagnostics but cannot grant attribution."""

    home = _hermes_home(tmp_path / "home")
    bridge = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
        evidence_path=tmp_path / "artifacts" / "memory.jsonl",
        run_id="run.tool-identity-gap",
        trace_id="trace.tool-identity-gap",
        episode_id="episode.tool-identity-gap",
        session_id="session.tool-identity-gap",
        task_id="task.tool-identity-gap",
        experiment_variant="fixture",
    )
    bridge._record_tool_call_results({
        "messages": [
            {
                "role": "assistant",
                # Missing host call ID: bridge may retain a synthetic ID for
                # deterministic diagnostics, but must mark the join unsafe.
                "tool_calls": [{
                    "function": {"name": "fixture_apply", "arguments": "{}"},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "tool-call-1",
                # Missing host result ID: same fail-closed rule applies.
                "content": '{"success": true}',
            },
        ],
    })
    assert len(bridge.tool_call_result_joins) == 1
    join = bridge.tool_call_result_joins[0]
    assert join.type_mismatch is True
    assert resolve_tool_call_result(join).status is ToolJoinResolutionStatus.TYPE_MISMATCH
    bridge.close()


def test_live_bridge_records_content_free_sm01_future_feedback(tmp_path: Path) -> None:
    from hermes_state import SessionDB

    home = _hermes_home(tmp_path / "home")
    (home / "memories" / "MEMORY.md").write_text(
        PRIVATE_PREFERENCE,
        encoding="utf-8",
    )
    (home / "memories" / "USER.md").write_text("", encoding="utf-8")
    db = SessionDB(home / "state.db")
    artifacts = tmp_path / "artifacts"
    operations_path = artifacts / "operations.jsonl"
    client = FakeCompletionClient({
        POLICY_FACT_EXTRACTION_PROMPT.artifact.prompt_id: json.dumps({"facts": []}),
        POLICY_INTERNAL_OPERATION_PROMPT.artifact.prompt_id: json.dumps({
            "operations": [],
        }),
    })

    class NativeStore:
        def format_for_system_prompt(self, target: str) -> str | None:
            if target != "memory":
                return None
            return f"MEMORY\n{PRIVATE_PREFERENCE}"

    agent = SimpleNamespace(
        _memory_store=NativeStore(),
        _session_db=db,
        session_id="session-1",
    )
    bridge = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
        evidence_path=artifacts / "memory.jsonl",
        run_id="run-future-live",
        trace_id="trace-future-live",
        episode_id="episode-future-live",
        session_id="session-1",
        task_id="SM01_EVAL_NEAR_001",
        experiment_variant="static-rsimem",
        family_id="SM01_preference_adoption",
        stage="eval_near",
        lifecycle_config=HermesLifecycleConfig(evaluator_mode="deterministic"),
        static_writeback_config=StaticSemanticWritebackConfig(
            mode="static_utility",
            feedback_contract="sm01_tsv_v1",
        ),
        static_completion_client=client,
        static_operation_evidence_path=operations_path,
    )
    bridge.attach(agent)
    prompt = agent._memory_store.format_for_system_prompt("memory")
    assert prompt is not None
    bridge.on_task_completed({
        "completed": True,
        "final_response": (
            "owner\tpriority\ttask\tdue_date\n"
            "Iris Chen\tHigh\tFix drift\t2026/04/28"
        ),
        "messages": [
            {
                "role": "user",
                "content": "Extract today's action items and share the source note.",
            },
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-share",
                    "function": {
                        "name": "notes_share",
                        "arguments": json.dumps({
                            "note_id": "note-1",
                            "recipients": ["Iris Chen"],
                        }),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call-share",
                "content": json.dumps({"success": True}),
            },
        ],
    })
    bridge.close()
    db.close()

    graph = materialize_operation_graph(
        AppendOnlyOperationEvidenceLog(operations_path).events
    )
    future = {
        operation.kind: operation
        for operation in graph.operations
        if operation.kind in {
            OperationKind.FUTURE_QUERY,
            OperationKind.RETRIEVAL,
            OperationKind.INJECTION,
            OperationKind.USE,
            OperationKind.DOWNSTREAM_OUTCOME,
        }
    }
    assert set(future) == {
        OperationKind.FUTURE_QUERY,
        OperationKind.RETRIEVAL,
        OperationKind.INJECTION,
        OperationKind.USE,
        OperationKind.DOWNSTREAM_OUTCOME,
    }
    assert future[OperationKind.FUTURE_QUERY].status == OperationStatus.SUCCESS
    assert future[OperationKind.RETRIEVAL].status == OperationStatus.SUCCESS
    assert future[OperationKind.INJECTION].status == OperationStatus.SUCCESS
    # No application-owned attribution provider is installed in this
    # fixture.  The benchmark feedback resolver may classify the audit plane,
    # but it must not mark the pure-process USE operation as successful.
    assert future[OperationKind.USE].status == OperationStatus.NONE
    assert future[OperationKind.USE].reason_code == "retrieved_but_unused"
    assert future[OperationKind.DOWNSTREAM_OUTCOME].status == OperationStatus.SUCCESS
    memory_ids = tuple(
        artifact.artifact_id
        for artifact in graph.artifacts
        if artifact.kind.value == "memory_artifact"
    )
    assert len(memory_ids) == 1
    injected_ids = set(future[OperationKind.INJECTION].input_artifact_ids)
    assert set(memory_ids).issubset(injected_ids)
    assert future[OperationKind.USE].output_artifact_ids == ()
    serialized = operations_path.read_text(encoding="utf-8")
    assert PRIVATE_PREFERENCE not in serialized


def test_live_bridge_joins_restarted_source_to_future_feedback(tmp_path: Path) -> None:
    from hermes_state import SessionDB

    home = _hermes_home(tmp_path / "home")

    class NativeStore:
        def format_for_system_prompt(self, target: str) -> str | None:
            path = home / "memories" / f"{target.upper()}.md"
            content = path.read_text(encoding="utf-8").strip()
            return content or None

    learn_db = SessionDB(home / "state.db")
    learn_session = "session-feedback-learn"
    learn_db.create_session(learn_session, "past_bench", model="fixture-model")
    learn_db.append_message(
        learn_session,
        "user",
        "Always use TSV with owner, priority, task, and due_date columns.",
    )
    learn_db.append_message(learn_session, "assistant", "Understood.")
    learn_client = FakeCompletionClient({
        POLICY_FACT_EXTRACTION_PROMPT.artifact.prompt_id: json.dumps({
            "facts": [
                "Always use TSV with owner, priority, task, and due_date columns."
            ],
        }),
        POLICY_INTERNAL_OPERATION_PROMPT.artifact.prompt_id: json.dumps({
            "operations": [{
                "fact_index": 0,
                "action": "add",
                "candidate_id": None,
            }],
        }),
    })
    learn_bridge = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
        evidence_path=tmp_path / "learn" / "memory.jsonl",
        run_id="run-feedback-sequence",
        trace_id="trace-feedback-learn",
        episode_id="episode-feedback-learn",
        session_id=learn_session,
        task_id="SM01_LEARN_A_001",
        experiment_variant="static-extraction-rsimem",
        family_id="SM01_preference_adoption",
        stage="learn_a",
        static_writeback_config=StaticSemanticWritebackConfig(
            mode="static",
            feedback_contract="sm01_tsv_v1",
        ),
        static_completion_client=learn_client,
    )
    learn_bridge.attach(SimpleNamespace(
        _memory_store=NativeStore(),
        _session_db=learn_db,
        session_id=learn_session,
    ))
    learn_bridge.on_task_completed({"completed": True, "messages": []})
    learn_bridge.close()
    learn_db.close()

    source_path = home / ".rsimem" / "extraction_sources.jsonl"
    assert source_path.exists()
    source_serialized = source_path.read_text(encoding="utf-8")
    assert "Always use TSV" not in source_serialized
    source_store = JsonExtractionSourceRecordStore(source_path)
    learned_source = source_store.records()[0]
    learn_capture_path = tmp_path / "learn" / "extraction_optimizer_capture.jsonl"
    private_captures = JsonExtractionOptimizerCaptureLog(
        learn_capture_path
    ).records()
    assert len(private_captures) == 1
    assert isinstance(private_captures[0], ExtractionOptimizerSourceCapture)
    assert private_captures[0].source_record_id == learned_source.record_id
    assert private_captures[0].projection.projection_digest == (
        learned_source.source.source_projection_digest
    )
    assert private_captures[0].fact_contents[0].content.startswith(
        "Always use TSV"
    )
    assert learn_capture_path.stat().st_mode & 0o777 == 0o600
    assert not (home / ".rsimem" / "extraction_optimizer_capture.jsonl").exists()
    unrelated_empty = ExtractionSourceRecord.create(
        family_id="SM01_preference_adoption",
        stage="learn_b",
        run_id="run-feedback-sequence",
        episode_id="episode-feedback-learn-b",
        session_id="session-feedback-learn-b",
        task_id="SM01_LEARN_B_001",
        compilation_id="compilation.feedback-unrelated-empty",
        extraction_artifact_id=learned_source.extraction_artifact_id,
        extraction_artifact_digest=learned_source.extraction_artifact_digest,
        extraction_output_digest="d" * 64,
        source=ExtractionSourceEvidence(
            "source.feedback-unrelated-empty",
            "e" * 64,
            "extraction-set.feedback-unrelated-empty",
            ExtractionSetStatus.EMPTY,
            (),
            (),
        ),
        activation=extraction_activation_fixture(
            compilation_id="compilation.feedback-unrelated-empty",
            extraction_operation_id="extraction-set.feedback-unrelated-empty",
            component_artifact_id=learned_source.extraction_artifact_id,
            component_artifact_digest=learned_source.extraction_artifact_digest,
            parsed_output_digest="d" * 64,
        ),
    )
    source_store.append(unrelated_empty)

    eval_db = SessionDB(home / "state.db")
    eval_session = "session-feedback-eval"
    eval_db.create_session(eval_session, "past_bench", model="fixture-model")
    eval_db.append_message(
        eval_session,
        "user",
        "Extract today's action items and share the source note.",
    )
    eval_db.append_message(eval_session, "assistant", "Completed the report.")
    eval_client = FakeCompletionClient({
        POLICY_FACT_EXTRACTION_PROMPT.artifact.prompt_id: json.dumps({"facts": []}),
        POLICY_INTERNAL_OPERATION_PROMPT.artifact.prompt_id: json.dumps({
            "operations": [],
        }),
    })
    eval_operations = tmp_path / "eval" / "operations.jsonl"
    eval_bridge = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
        evidence_path=tmp_path / "eval" / "memory.jsonl",
        run_id="run-feedback-sequence",
        trace_id="trace-feedback-eval",
        episode_id="episode-feedback-eval",
        session_id=eval_session,
        task_id="SM01_EVAL_NEAR_001",
        experiment_variant="static-extraction-rsimem",
        family_id="SM01_preference_adoption",
        stage="eval_near",
        static_writeback_config=StaticSemanticWritebackConfig(
            mode="static",
            feedback_contract="sm01_tsv_v1",
        ),
        static_completion_client=eval_client,
        static_operation_evidence_path=eval_operations,
    )
    eval_agent = SimpleNamespace(
        _memory_store=NativeStore(),
        _session_db=eval_db,
        session_id=eval_session,
    )
    eval_bridge.attach(eval_agent)
    assert eval_agent._memory_store.format_for_system_prompt("user") is not None
    eval_bridge.on_task_completed({
        "completed": True,
        "final_response": (
            "owner\tpriority\ttask\tdue_date\n"
            "Iris Chen\tHigh\tFix drift\t2026/04/28"
        ),
        "messages": [
            {
                "role": "user",
                "content": "Extract today's action items and share the source note.",
            },
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-share",
                    "function": {
                        "name": "notes_share",
                        "arguments": json.dumps({
                            "note_id": "note-1",
                            "recipients": ["Iris Chen"],
                        }),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call-share",
                "content": json.dumps({"success": True}),
            },
        ],
    })
    eval_bridge.close()
    eval_db.close()

    pure_feedback = JsonPureExtractionFeedbackRecordStore(
        tmp_path / "eval" / "rsimem_pure_extraction_feedback.jsonl"
    ).records()
    # The runtime now materializes deployment-only feedback automatically;
    # without an application-owned opportunity provider these records remain
    # unresolved rather than inheriting the benchmark audit label.
    assert len(pure_feedback) == 1
    assert all(item.attribution.value == "unresolved" for item in pure_feedback)

    graph = materialize_operation_graph(
        AppendOnlyOperationEvidenceLog(eval_operations).events
    )
    operations = {operation.kind: operation for operation in graph.operations}
    feedback_path = tmp_path / "eval" / "rsimem_extraction_feedback.jsonl"
    records = [
        json.loads(line)
        for line in feedback_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 2
    assert all(record["run_id"] == "run-feedback-sequence" for record in records)
    assert all(record["episode_id"] == "episode-feedback-eval" for record in records)
    assert all(record["task_id"] == "SM01_EVAL_NEAR_001" for record in records)
    by_source = {record["source_record_id"]: record for record in records}
    primary = next(
        example
        for example in by_source[learned_source.record_id]["dataset"]["examples"]
        if example["primary"]
    )
    assert primary["label"] == "useful"
    assert primary["opportunity_operation_id"] == operations[
        OperationKind.FUTURE_QUERY
    ].operation_id
    assert primary["use_operation_id"] == operations[OperationKind.USE].operation_id
    assert primary["outcome_operation_id"] == operations[
        OperationKind.DOWNSTREAM_OUTCOME
    ].operation_id
    empty_primary = next(
        example
        for example in by_source[unrelated_empty.record_id]["dataset"]["examples"]
        if example["primary"]
    )
    assert empty_primary["label"] == "unresolved"
    assert empty_primary["reason_codes"] == ["use_not_bound_to_memory"]
    serialized = feedback_path.read_text(encoding="utf-8")
    assert "Always use TSV" not in serialized
    assert not any(token in serialized for token in (
        "task_score",
        "grader",
        "answer_key",
        "reasoning_tokens",
    ))
    eval_capture_path = tmp_path / "eval" / "extraction_optimizer_capture.jsonl"
    private_captures = (
        JsonExtractionOptimizerCaptureLog(learn_capture_path).records()
        + JsonExtractionOptimizerCaptureLog(eval_capture_path).records()
    )
    assert eval_capture_path.stat().st_mode & 0o777 == 0o600
    source_captures = tuple(
        value for value in private_captures
        if isinstance(value, ExtractionOptimizerSourceCapture)
    )
    feedback_captures = tuple(
        value for value in private_captures
        if isinstance(value, ExtractionOptimizerFeedbackCapture)
    )
    assert len(source_captures) == 2
    assert {value.source_record_id for value in source_captures} == {
        learned_source.record_id,
        next(
            value.record_id
            for value in JsonExtractionSourceRecordStore(source_path).records()
            if value.task_id == "SM01_EVAL_NEAR_001"
        ),
    }
    assert len(feedback_captures) == 2
    captured_useful = next(
        value for value in feedback_captures
        if value.source_record_id == learned_source.record_id
    )
    assert captured_useful.observation.final_response.startswith("owner\tpriority")
    assert captured_useful.current_input == (
        "Extract today's action items and share the source note."
    )


def test_live_bridge_derives_missed_from_empty_past_extraction(tmp_path: Path) -> None:
    from hermes_state import SessionDB

    home = _hermes_home(tmp_path / "home")
    (home / "memories" / "MEMORY.md").write_text("", encoding="utf-8")
    (home / "memories" / "USER.md").write_text("", encoding="utf-8")
    source = ExtractionSourceEvidence(
        "source.live-missed",
        "a" * 64,
        "extraction-set.live-missed",
        ExtractionSetStatus.EMPTY,
        ("preference.summary.tsv",),
        (),
    )
    source_record = ExtractionSourceRecord.create(
        family_id="SM01_preference_adoption",
        stage="learn_a",
        run_id="run-live-missed",
        episode_id="episode-live-missed-learn",
        session_id="session-live-missed-learn",
        task_id="SM01_LEARN_A_001",
        compilation_id="compilation.live-missed",
        extraction_artifact_id="prompt-component.live-missed",
        extraction_artifact_digest="b" * 64,
        extraction_output_digest="c" * 64,
        source=source,
        activation=extraction_activation_fixture(
            compilation_id="compilation.live-missed",
            extraction_operation_id=source.extraction_set_id,
            component_artifact_id="prompt-component.live-missed",
            component_artifact_digest="b" * 64,
            parsed_output_digest="c" * 64,
        ),
    )
    JsonExtractionSourceRecordStore(
        home / ".rsimem" / "extraction_sources.jsonl"
    ).append(source_record)

    class EmptyNativeStore:
        def format_for_system_prompt(self, target: str) -> str | None:
            return None

    db = SessionDB(home / "state.db")
    session_id = "session-live-missed-eval"
    db.create_session(session_id, "past_bench", model="fixture-model")
    db.append_message(
        session_id,
        "user",
        "Extract today's action items and share the source note.",
    )
    db.append_message(session_id, "assistant", "I could not prepare the report.")
    client = FakeCompletionClient({
        POLICY_FACT_EXTRACTION_PROMPT.artifact.prompt_id: json.dumps({"facts": []}),
        POLICY_INTERNAL_OPERATION_PROMPT.artifact.prompt_id: json.dumps({
            "operations": [],
        }),
    })
    operations_path = tmp_path / "eval" / "operations.jsonl"
    bridge = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
        evidence_path=tmp_path / "eval" / "memory.jsonl",
        run_id="run-live-missed",
        trace_id="trace-live-missed-eval",
        episode_id="episode-live-missed-eval",
        session_id=session_id,
        task_id="SM01_EVAL_NEAR_001",
        experiment_variant="static-extraction-rsimem",
        family_id="SM01_preference_adoption",
        stage="eval_near",
        static_writeback_config=StaticSemanticWritebackConfig(
            mode="static",
            feedback_contract="sm01_tsv_v1",
        ),
        static_completion_client=client,
        static_operation_evidence_path=operations_path,
    )
    agent = SimpleNamespace(
        _memory_store=EmptyNativeStore(),
        _session_db=db,
        session_id=session_id,
    )
    bridge.attach(agent)
    assert agent._memory_store.format_for_system_prompt("user") is None
    bridge.on_task_completed({
        "completed": False,
        "final_response": "I could not prepare the report.",
        "messages": [{
            "role": "user",
            "content": "Extract today's action items and share the source note.",
        }],
    })
    bridge.close()
    db.close()

    feedback_path = tmp_path / "eval" / "rsimem_extraction_feedback.jsonl"
    records = [
        json.loads(line)
        for line in feedback_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    primary = next(
        example
        for example in records[0]["dataset"]["examples"]
        if example["primary"]
    )
    assert primary["label"] == "missed"
    assert primary["reason_codes"] == ["high_confidence_missed_extraction"]
    assert primary["source_id"] == source.source_id
    assert primary["opportunity_operation_id"] is not None
    assert primary["outcome_operation_id"] is not None
    serialized = feedback_path.read_text(encoding="utf-8")
    assert "Always use TSV" not in serialized
    assert "Extract today's action items" not in serialized


def test_feedback_bridge_rejects_duplicate_without_source_record(tmp_path: Path) -> None:
    home = _hermes_home(tmp_path / "home")
    bridge = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
        evidence_path=tmp_path / "artifacts" / "memory.jsonl",
        run_id="run-missing-source",
        trace_id="trace-missing-source",
        episode_id="episode-missing-source",
        session_id="session-missing-source",
        task_id="SM01_LEARN_A_001",
        experiment_variant="static-extraction-rsimem",
        family_id="SM01_preference_adoption",
        stage="learn_a",
        static_writeback_config=StaticSemanticWritebackConfig(
            mode="static",
            feedback_contract="sm01_tsv_v1",
        ),
        static_completion_client=FakeCompletionClient({}),
    )
    try:
        with pytest.raises(ValueError, match="no source evidence"):
            bridge._record_extraction_source(StaticSemanticBoundaryResult(
                "snapshot.missing",
                "compilation.missing",
                None,
                duplicate=True,
            ))
    finally:
        bridge.close()


def test_static_writeback_bridge_requires_native_ledger_not_lifecycle(tmp_path: Path) -> None:
    home = _hermes_home(tmp_path / "home")
    client = FakeCompletionClient({})
    kwargs = {
        "evidence_path": tmp_path / "artifacts" / "events.jsonl",
        "run_id": "run-static-contract",
        "trace_id": "trace-static-contract",
        "episode_id": "episode-static-contract",
        "session_id": "session-static-contract",
        "task_id": "task-static-contract",
        "experiment_variant": "static-rsimem",
        "static_writeback_config": StaticSemanticWritebackConfig(mode="static"),
        "static_completion_client": client,
    }
    with pytest.raises(ValueError, match=r"native\+ledger"):
        HermesPastBenchBridge(
            home,
            HermesExperimentConfig(HermesExecutionMode.ADAPTER_LEDGER),
            lifecycle_config=HermesLifecycleConfig(evaluator_mode="deterministic"),
            **kwargs,
        )
    bridge = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
        **kwargs,
    )
    assert bridge.lifecycle is None
    bridge.close()


def test_matched_extraction_bridge_loads_attempt_local_active_candidate(
    tmp_path: Path,
) -> None:
    home = _hermes_home(tmp_path / "home")
    artifacts = tmp_path / "artifacts"
    trial = artifacts / "extraction-trial"
    parent = _parent()
    candidate = _candidate(parent=parent)
    prepare_extraction_matched_trial_runtime(
        parent=parent,
        candidate=candidate,
        offline_decision=_offline_decision(parent, candidate),
        output_root=trial,
    )
    client = FakeCompletionClient({})
    bridge = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
        evidence_path=artifacts / "events.jsonl",
        run_id="run-extraction-trial",
        trace_id="trace-extraction-trial",
        episode_id="episode-extraction-trial",
        session_id="session-extraction-trial",
        task_id="SM01-extraction-trial",
        experiment_variant="adaptive-extraction-rsimem",
        static_writeback_config=StaticSemanticWritebackConfig(
            mode="static",
            extraction_runtime_scope="matched_validation",
            extraction_runtime_config_path=str(
                trial / EXTRACTION_TRIAL_CONFIG_FILE
            ),
        ),
        static_completion_client=client,
    )
    try:
        assert bridge.static_writeback is not None
        binding = bridge.static_writeback.extraction_runtime_binding
        assert binding.policy_artifact_id == candidate.artifact_id
        assert binding.policy_artifact_digest == candidate.artifact_digest
        assert binding.deployment_scope.value == "matched_validation"
        assert binding.trial_id is not None
        assert bridge.static_writeback.static_parent_identity is None
    finally:
        bridge.close()
    assert client.calls == ()


def test_matched_extraction_bridge_rejects_non_attempt_local_config(
    tmp_path: Path,
) -> None:
    home = _hermes_home(tmp_path / "home")
    outside = tmp_path / "outside" / EXTRACTION_TRIAL_CONFIG_FILE
    client = FakeCompletionClient({})
    config = StaticSemanticWritebackConfig(
        mode="static",
        extraction_runtime_scope="matched_validation",
        extraction_runtime_config_path=str(outside),
    )

    with pytest.raises(ValueError, match="inside capture artifacts"):
        HermesPastBenchBridge(
            home,
            HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
            evidence_path=tmp_path / "artifacts" / "events.jsonl",
            run_id="run-extraction-path",
            trace_id="trace-extraction-path",
            episode_id="episode-extraction-path",
            session_id="session-extraction-path",
            task_id="SM01-extraction-path",
            experiment_variant="adaptive-extraction-rsimem",
            static_writeback_config=config,
            static_completion_client=client,
        )


def test_matched_extraction_bridge_rechecks_configured_revocation_registry(
    tmp_path: Path,
) -> None:
    home = _hermes_home(tmp_path / "home")
    artifacts = tmp_path / "artifacts"
    trial = artifacts / "extraction-trial"
    parent = _parent()
    candidate = _candidate(parent=parent)
    prepare_extraction_matched_trial_runtime(
        parent=parent,
        candidate=candidate,
        offline_decision=_offline_decision(parent, candidate),
        output_root=trial,
    )
    registry = JsonRevocationRegistry(home / ".rsimem" / "revocations.jsonl")
    registry.initialize()
    registry.append(RevocationEntry.create(
        artifact_id=candidate.artifact_id,
        artifact_schema_version=candidate.schema_version,
        artifact_digest=candidate.artifact_digest,
        evidence_plane=EvidencePlane.PURE_PROCESS,
        evidence_source=EvidenceSourceKind.RUNTIME_OBSERVATION,
        revoked_at="2026-08-31T00:00:00Z",
        reason_code="stale_schema",
    ))
    config = StaticSemanticWritebackConfig(
        mode="static",
        extraction_runtime_scope="matched_validation",
        extraction_runtime_config_path=str(trial / EXTRACTION_TRIAL_CONFIG_FILE),
        revocation_registry_path=".rsimem/revocations.jsonl",
    )
    with pytest.raises(ValueError, match="formal validated extraction runtime requires"):
        HermesPastBenchBridge(
            home,
            HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
            evidence_path=artifacts / "missing-registry-events.jsonl",
            run_id="run-extraction-formal-missing-registry",
            trace_id="trace-extraction-formal-missing-registry",
            episode_id="episode-extraction-formal-missing-registry",
            session_id="session-extraction-formal-missing-registry",
            task_id="task-extraction-formal-missing-registry",
            experiment_variant="adaptive-extraction-rsimem",
            family_id="SM01_preference_adoption",
            stage="validation",
            static_writeback_config=StaticSemanticWritebackConfig(
                mode="static",
                extraction_runtime_scope="matched_validation",
                extraction_runtime_config_path=str(trial / EXTRACTION_TRIAL_CONFIG_FILE),
            ),
            static_completion_client=FakeCompletionClient({}),
        )
    with pytest.raises(ValueError, match="artifact is revoked"):
        HermesPastBenchBridge(
            home,
            HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
            evidence_path=artifacts / "events.jsonl",
            run_id="run-extraction-revoked",
            trace_id="trace-extraction-revoked",
            episode_id="episode-extraction-revoked",
            session_id="session-extraction-revoked",
            task_id="task-extraction-revoked",
            experiment_variant="adaptive-extraction-rsimem",
            static_writeback_config=config,
            static_completion_client=FakeCompletionClient({}),
        )


@pytest.mark.parametrize(
    "store_path",
    ("/tmp/adaptive-policies.json", "../adaptive-policies.json"),
)
def test_adaptive_writeback_bridge_rejects_store_outside_hermes_home(
    tmp_path: Path,
    store_path: str,
) -> None:
    home = _hermes_home(tmp_path / "home")
    client = FakeCompletionClient({})
    config = StaticSemanticWritebackConfig(
        mode="adaptive_utility",
        adaptive_policy_store_path=store_path,
        adaptive_trusted_roots=("mem0-flat.parent-v1",),
        adaptive_parameters=(TrustedAdaptiveMem0Parameter(
            parameter_id="parameter.retrieval",
            name=AdaptiveParameterName.RETRIEVAL_ACCEPT_THRESHOLD,
            prompt_ref="mem0-flat.retrieval",
            baseline_value=0.35,
        ),),
    )

    with pytest.raises(ValueError, match="adaptive policy store"):
        HermesPastBenchBridge(
            home,
            HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
            evidence_path=tmp_path / "artifacts" / "events.jsonl",
            run_id="run-adaptive-path",
            trace_id="trace-adaptive-path",
            episode_id="episode-adaptive-path",
            session_id="session-adaptive-path",
            task_id="task-adaptive-path",
            experiment_variant="adaptive-rsimem",
            lifecycle_config=HermesLifecycleConfig(evaluator_mode="deterministic"),
            static_writeback_config=config,
            static_completion_client=client,
        )
    assert client.calls == ()


def test_adaptive_writeback_bridge_requires_active_attempt_local_policy(
    tmp_path: Path,
) -> None:
    home = _hermes_home(tmp_path / "home")
    client = FakeCompletionClient({})
    relative_store = Path(".rsimem/adaptive-policies.json")
    config = StaticSemanticWritebackConfig(
        mode="adaptive_utility",
        adaptive_policy_store_path=str(relative_store),
        adaptive_trusted_roots=("mem0-flat.parent-v1",),
        adaptive_parameters=(TrustedAdaptiveMem0Parameter(
            parameter_id="parameter.retrieval",
            name=AdaptiveParameterName.RETRIEVAL_ACCEPT_THRESHOLD,
            prompt_ref="mem0-flat.retrieval",
            baseline_value=0.35,
        ),),
    )

    with pytest.raises(ValueError, match="requires an active policy"):
        HermesPastBenchBridge(
            home,
            HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
            evidence_path=tmp_path / "artifacts" / "events.jsonl",
            run_id="run-adaptive-empty",
            trace_id="trace-adaptive-empty",
            episode_id="episode-adaptive-empty",
            session_id="session-adaptive-empty",
            task_id="task-adaptive-empty",
            experiment_variant="adaptive-rsimem",
            lifecycle_config=HermesLifecycleConfig(evaluator_mode="deterministic"),
            static_writeback_config=config,
            static_completion_client=client,
        )
    store_path = home / relative_store
    assert store_path.parent.is_dir()
    assert store_path.with_suffix(".json.lock").exists()
    assert not store_path.exists()
    assert client.calls == ()
