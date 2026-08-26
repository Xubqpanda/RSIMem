from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, replace
from pathlib import Path

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
from rsimem.lifecycle import RawResourceUsage, run_sm01_preference_fixture
from rsimem.memory import (
    MemoryArtifact,
    MemoryHit,
    MemoryKind,
    MemoryQuery,
    MemoryResource,
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
        HermesExperimentConfig(HermesExecutionMode.ADAPTER_LEDGER),
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
