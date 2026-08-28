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
from rsimem.ledger import LifecycleLedgerObserver
from rsimem.lifecycle import (
    HermesLifecycleConfig,
    RawResourceUsage,
    run_sm01_preference_fixture,
)
from rsimem.memory import (
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
    ExtractionSetStatus,
    ExtractionSourceEvidence,
)
from rsimem.memory.extraction_projection import (
    ExtractionSourceRecord,
    JsonExtractionSourceRecordStore,
)
from rsimem.memory.operation_graph import (
    AppendOnlyOperationEvidenceLog,
    OperationKind,
    OperationStatus,
    materialize_operation_graph,
)
from rsimem.memory.adaptive_policy import AdaptiveParameterName
from rsimem.memory.adaptive_mem0_binding import TrustedAdaptiveMem0Parameter
from rsimem.memory_systems.mem0_flat import (
    FakeCompletionClient,
    POLICY_FACT_EXTRACTION_PROMPT,
    POLICY_INTERNAL_OPERATION_PROMPT,
)


PRIVATE_PREFERENCE = "Use TSV with owner, priority, task, and due_date."


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
    assert bridge.static_results[0].writeback.logical_exit is True
    assert len(client.calls) == 2
    db.close()


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
    assert all(operation.status == OperationStatus.SUCCESS for operation in future.values())
    memory_ids = tuple(
        artifact.artifact_id
        for artifact in graph.artifacts
        if artifact.kind.value == "memory_artifact"
    )
    assert len(memory_ids) == 1
    injected_ids = set(future[OperationKind.INJECTION].input_artifact_ids)
    used_ids = set(future[OperationKind.USE].input_artifact_ids)
    assert set(memory_ids).issubset(injected_ids)
    assert set(memory_ids).issubset(used_ids)
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
