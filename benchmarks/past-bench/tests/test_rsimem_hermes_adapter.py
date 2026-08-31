from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from past_bench.models.content import TextBlock
from past_bench.models.message import Message
from past_bench.runtime.adapters.hermes import HermesAdapter, _past_bench_opportunity_provider
from past_bench.runtime.protocol import (
    RuntimeConfigPayload,
    RuntimeModelConfig,
    StartSessionRequest,
    StepRequest,
)
from past_bench.runtime.registry import AgentSpec
from rsimem.memory.process_feedback import audit_process_events


PRIVATE_MEMORY = "Use TSV with owner, priority, task, and due_date."


def test_past_opportunity_provider_ignores_benchmark_scope_metadata() -> None:
    base = {
        "messages": [{
            "role": "user",
            "content": "Please produce a TSV with owner, priority, task, and due_date.",
        }],
        "rsimem_source_provenance_id": "provenance.provider.scope-v1",
    }
    first = _past_bench_opportunity_provider({
        **base,
        "family_id": "SM01_preference_adoption",
        "stage": "eval_near",
    })
    second = _past_bench_opportunity_provider({
        **base,
        "family_id": "SM05_weak_trigger_preference_adoption",
        "stage": "eval_far",
    })
    assert first == second
    assert len(first) == 1
    assert first[0].evidence_plane.value == "pure_process"
    assert first[0].evidence_source.value == "runtime_observation"


def _home(path: Path) -> Path:
    memories = path / "memories"
    memories.mkdir(parents=True)
    (memories / "MEMORY.md").write_text(PRIVATE_MEMORY, encoding="utf-8")
    (memories / "USER.md").write_text(
        "The user prefers concise updates.",
        encoding="utf-8",
    )
    skill = path / "skills" / "operations" / "task-table"
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: task-table\ndescription: Format task tables\n---\nUse four columns.",
        encoding="utf-8",
    )
    (skill / "references" / "columns.md").write_text(
        "owner, priority, task, due_date\n",
        encoding="utf-8",
    )

    from hermes_state import SessionDB

    db = SessionDB(path / "state.db")
    db.create_session("past-session", "cli", model="fixture-model")
    db.append_message("past-session", "user", "Please format the task table.")
    db.append_message("past-session", "assistant", "The task table is ready.")
    db.close()
    return path


def _request(home: Path, artifacts: Path, mode: str) -> StartSessionRequest:
    # ``+`` is reserved for the experiment-variant label; host/session IDs
    # use the stricter policy-contract identifier grammar.
    session_id = f"session-{mode.replace('+', '-')}"
    return StartSessionRequest(
        session_id=session_id,
        agent_name="hermes",
        task_id="SM01_fixture",
        task_name="Matched RSIMem fixture",
        max_turns=1,
        timeout_seconds=60,
        initial_messages=[Message(
            role="user",
            content=[TextBlock(text="Recall the task table preference.")],
        )],
        model=RuntimeModelConfig(
            model_id="fixture-model",
            extra_body={"hermes": {
                "persistence_enabled": True,
                "session_search_enabled": True,
                "home_dir": str(home),
                "capture_artifacts_dir": str(artifacts),
                "background_review_wait_s": 0,
                "config_overrides": {"memory": {
                    "memory_enabled": True,
                    "user_profile_enabled": True,
                }},
                "rsimem": {
                    "mode": mode,
                    "adapter_failure_policy": "fail_closed",
                    "verify_native_projection": True,
                    "evidence_path": str(artifacts / "rsimem_memory_events.jsonl"),
                },
            }},
        ),
        runtime_config=RuntimeConfigPayload(metadata={
            "run_id": "run-matched",
            "trace_id": f"trace-{mode}",
            "episode_id": "episode-matched",
            "family_id": "SM01",
            "stage": "eval_near",
            "experiment_variant": "with_persistence",
        }),
    )


def test_past_bench_agent_loop_matches_native_ledger_and_adapter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from hermes_state import SessionDB
    from tools import memory_tool, skills_tool
    from tools.registry import registry

    home = _home(tmp_path / "home")
    monkeypatch.setattr(memory_tool, "MEMORY_DIR", home / "memories")
    monkeypatch.setattr(skills_tool, "HERMES_HOME", home)
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", home / "skills")

    fake_run_agent = types.ModuleType("run_agent")

    class FixtureAgent:
        def __init__(self, **kwargs):
            self._session_db = kwargs["session_db"]
            self.session_id = "current-session"
            self.session_log_file = None
            self._memory_store = memory_tool.MemoryStore()
            self._memory_store.load_from_disk()

        def _execute_recorded_model_call(self, *args, **kwargs):
            return None

        async def _execute_recorded_async_model_call(self, *args, **kwargs):
            return None

        def run_conversation(self, **kwargs):
            memory = self._memory_store.format_for_system_prompt("memory")
            user = self._memory_store.format_for_system_prompt("user")
            search = self._session_db.search_messages(
                query="task table",
                limit=50,
                offset=0,
            )
            conversation = self._session_db.get_messages_as_conversation(
                "past-session"
            )
            skills = registry.dispatch("skills_list", {})
            skill = registry.dispatch("skill_view", {"name": "task-table"})
            final = json.dumps({
                "memory": memory,
                "user": user,
                "search": search,
                "conversation": conversation,
                "skills": json.loads(skills),
                "skill": json.loads(skill),
            }, ensure_ascii=True, sort_keys=True)
            return {"final_response": final, "input_tokens": 0, "output_tokens": 0}

        def wait_for_background_reviews(self, timeout=0):
            return True

    fake_run_agent.AIAgent = FixtureAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(
        HermesAdapter,
        "_reload_hermes_modules_if_needed",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "hermes_state.DEFAULT_DB_PATH",
        home / "state.db",
    )

    results = {}
    evidence = {}
    responses = {}
    for mode in ("native", "native+ledger", "native+adapter+ledger"):
        artifacts = tmp_path / mode.replace("+", "_")
        request = _request(home, artifacts, mode)
        adapter = HermesAdapter(
            AgentSpec(name="hermes", adapter="hermes"),
            request,
        )
        try:
            response = adapter.step(StepRequest(
                session_id=request.session_id,
                step_id=0,
            ))
        finally:
            adapter.close("fixture complete")
        assert response.status == "finished", response.error
        responses[mode] = response
        results[mode] = response.final_output
        evidence_path = artifacts / "rsimem_memory_events.jsonl"
        evidence[mode] = (
            evidence_path.read_text(encoding="utf-8")
            if evidence_path.exists()
            else None
        )

    assert results["native"] == results["native+ledger"]
    assert results["native+ledger"] == results["native+adapter+ledger"]
    assert evidence["native"] is None
    for serialized in (
        evidence["native+ledger"],
        evidence["native+adapter+ledger"],
    ):
        assert serialized is not None
        assert '"kind": "query"' in serialized
        assert '"kind": "injected"' in serialized
        assert PRIVATE_MEMORY not in serialized
        assert "task table is ready" not in serialized.lower()
    assert '"kind": "projection_check"' in evidence["native+adapter+ledger"]
    # The runtime transports only content-free process-corpus identity.  It
    # must be an exact projection of the persisted event IDs and must not
    # expose benchmark scores or grader fields to the learner-facing response.
    for mode, response in responses.items():
        process_path = tmp_path / mode.replace("+", "_") / "rsimem_process_feedback.jsonl"
        process_ids = [
            json.loads(line)["event_id"]
            for line in process_path.read_text(encoding="utf-8").splitlines()
        ] if process_path.exists() else []
        process_ids.sort()
        assert response.process_feedback_event_ids == process_ids
        if not process_path.exists():
            assert response.process_feedback_digest is None
            continue
        expected_digest = hashlib.sha256(
            json.dumps(process_ids, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        assert response.process_feedback_digest == expected_digest
        assert "score" not in response.model_dump()
        assert "grader" not in response.model_dump()
    assert not any(
        (tmp_path / mode.replace("+", "_") / "rsimem_lifecycle_events.jsonl").exists()
        for mode in ("native+ledger", "native+adapter+ledger")
    )


def test_rsimem_bridge_receives_automatic_task_completion_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The production Hermes runner must invoke the bridge after one-shot execution.

    Bridge-level tests exercise source/feedback projection directly.  This
    check covers the host wiring itself: a completed Hermes result is passed
    to ``on_task_completed`` without a launcher-side opt-in callback.
    """

    import past_bench.runtime.adapters.hermes as hermes_module
    import rsimem.hermes_past_bridge as bridge_module

    calls: dict[str, object] = {"completed": [], "closed": False}

    class Bridge:
        process_feedback_event_ids = ()
        process_feedback_digest = "0" * 64

        def __init__(self, *args, **kwargs):
            calls["bridge_kwargs"] = kwargs

        def attach(self, agent):
            calls["agent"] = agent

        def on_task_completed(self, result):
            calls["completed"].append(dict(result))

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr(bridge_module, "HermesPastBenchBridge", Bridge)
    monkeypatch.setattr(
        hermes_module.HermesAdapter,
        "_register_past_bench_tools",
        lambda self: None,
    )
    monkeypatch.setattr(
        hermes_module.HermesAdapter,
        "_capture_hermes_artifacts",
        staticmethod(lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        hermes_module.HermesAdapter,
        "_reload_hermes_modules_if_needed",
        staticmethod(lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        hermes_module.HermesAdapter,
        "_set_session_title_if_missing",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        hermes_module.HermesAdapter,
        "_isolate_rsimem_semantic_writer",
        lambda *args, **kwargs: None,
    )

    fake_run_agent = types.ModuleType("run_agent")

    class FixtureAgent:
        def __init__(self, **kwargs):
            self.session_log_file = None
            self.model_call_usage_records = []

        def _execute_recorded_model_call(self, request, **kwargs):
            return request()

        async def _execute_recorded_async_model_call(self, request, **kwargs):
            return request()

        def run_conversation(self, **kwargs):
            return {
                "completed": True,
                "final_response": "done",
                "messages": [{"role": "user", "content": "finish"}],
                "input_tokens": 0,
                "output_tokens": 0,
            }

        def wait_for_background_reviews(self, timeout=0):
            return True

    fake_run_agent.AIAgent = FixtureAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    request = _request(_home(tmp_path / "home"), tmp_path / "artifacts", "native+ledger")
    request.model.extra_body["hermes"]["session_search_enabled"] = False
    request.model.extra_body["hermes"]["rsimem"]["semantic_writeback"] = {
        "mode": "static",
        "feedback_contract": "disabled",
    }
    adapter = HermesAdapter(AgentSpec(name="hermes", adapter="hermes"), request)
    try:
        response = adapter.step(StepRequest(session_id=request.session_id, step_id=0))
    finally:
        adapter.close("automatic boundary fixture")

    assert response.status == "finished", response.error
    assert calls["agent"] is not None
    bridge_kwargs = calls["bridge_kwargs"]
    assert callable(bridge_kwargs["opportunity_evidence_provider"])
    assert callable(bridge_kwargs["artifact_set_binding_provider"])
    assert callable(bridge_kwargs["pure_extraction_fact_semantic_keys_provider"])
    assert calls["completed"] == [{
        "completed": True,
        "final_response": "done",
        "messages": [{"role": "user", "content": "finish"}],
        "input_tokens": 0,
        "output_tokens": 0,
    }]
    assert calls["closed"] is True


def test_past_bench_error_response_keeps_process_identity(monkeypatch) -> None:
    """A post-bridge failure must not sever the process-corpus join."""

    adapter = HermesAdapter.__new__(HermesAdapter)
    adapter._completed_response = None
    adapter._rsimem_bridge = types.SimpleNamespace(
        process_feedback_event_ids=("process-event.example",),
        process_feedback_digest="a" * 64,
    )

    def fail_run_agent():
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(adapter, "_run_agent", fail_run_agent)
    response = adapter.step(StepRequest(session_id="session-error", step_id=0))
    assert response.status == "error"
    assert response.process_feedback_event_ids == ["process-event.example"]
    assert response.process_feedback_digest == "a" * 64


def test_past_bench_emits_explicit_lifecycle_boundaries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from hermes_state import SessionDB

    home = tmp_path / "home"
    home.mkdir()
    artifacts = tmp_path / "artifacts"
    fake_run_agent = types.ModuleType("run_agent")

    class LifecycleAgent:
        def __init__(self, **kwargs):
            self._session_db = kwargs["session_db"]
            self.session_id = "native-lifecycle-session"
            self.session_log_file = None
            self._memory_store = None
            self._session_db.create_session(
                self.session_id,
                "past_bench",
                model="fixture-model",
            )

        def _execute_recorded_model_call(self, *args, **kwargs):
            return None

        async def _execute_recorded_async_model_call(self, *args, **kwargs):
            return None

        def run_conversation(self, **kwargs):
            self._session_db.append_message(
                self.session_id,
                "user",
                "Always use TSV output.",
            )
            self._session_db.append_message(
                self.session_id,
                "assistant",
                "Understood.",
            )
            return {
                "final_response": "Understood.",
                "messages": [],
                "completed": True,
                "input_tokens": 0,
                "output_tokens": 0,
            }

        def wait_for_background_reviews(self, timeout=0):
            return True

    fake_run_agent.AIAgent = LifecycleAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(
        HermesAdapter,
        "_reload_hermes_modules_if_needed",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")

    request = _request(home, artifacts, "native+adapter+ledger")
    request.model.extra_body["hermes"]["rsimem"]["lifecycle"] = {
        "evaluator_mode": "deterministic",
        "policy_version": "phase1-fixture-v1",
        "compiler_version": "uncompiled-v0",
    }
    request.model.extra_body["hermes"]["session_search_enabled"] = False
    adapter = HermesAdapter(
        AgentSpec(name="hermes", adapter="hermes"),
        request,
    )
    try:
        response = adapter.step(StepRequest(
            session_id=request.session_id,
            step_id=0,
        ))
    finally:
        adapter.close("fixture session end")

    assert response.status == "finished", response.error
    events = [
        json.loads(line)
        for line in (artifacts / "rsimem_lifecycle_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["kind"] for event in events].count("context_snapshot") == 1
    assert [event["kind"] for event in events].count("evaluation_accepted") == 1
    assert [event["kind"] for event in events].count("dry_run_mutation") == 1
    assert {event["data"].get("status") for event in events} >= {
        "accepted",
        "created",
        "valid",
        None,
    }
    serialized = json.dumps(events, ensure_ascii=True)
    assert "Always use TSV output." not in serialized
    assert str(home) not in serialized
    assert (artifacts / "rsimem_lifecycle_receipts.json").exists()


@pytest.mark.parametrize(
    ("configured_toolsets", "expected_toolsets"),
    [
        (None, []),
        ([], []),
        (["memory", "skills"], ["skills"]),
    ],
)
def test_past_bench_static_writeback_disables_native_writer_and_persists(
    tmp_path: Path,
    monkeypatch,
    configured_toolsets,
    expected_toolsets,
) -> None:
    from agent import auxiliary_client
    from tools import memory_tool

    home = tmp_path / "home"
    (home / "memories").mkdir(parents=True)
    artifacts = tmp_path / "artifacts"
    fake_run_agent = types.ModuleType("run_agent")
    captured: dict[str, object] = {}

    class StaticAgent:
        def __init__(self, **kwargs):
            captured["enabled_toolsets"] = kwargs["enabled_toolsets"]
            captured["disabled_toolsets"] = kwargs["disabled_toolsets"]
            captured["skip_memory"] = kwargs["skip_memory"]
            self.tools = [
                {"type": "function", "function": {"name": "memory"}},
                {"type": "function", "function": {"name": "skills_list"}},
            ]
            self.valid_tool_names = {"memory", "skills_list"}
            self._memory_nudge_interval = 1
            self._skill_nudge_interval = 1
            self._session_db = kwargs["session_db"]
            self.session_id = "native-static-session"
            self.session_log_file = None
            self.model_call_usage_records = []
            self.model_usage_callback = kwargs["model_usage_callback"]
            self._sequence = 0
            self._memory_store = memory_tool.MemoryStore()
            self._memory_store.load_from_disk()
            self._session_db.create_session(
                self.session_id,
                "past_bench",
                model="fixture-model",
            )

        def _execute_recorded_model_call(self, request, **kwargs):
            response = request()
            self._sequence += 1
            record = {
                "call_id": f"static-call-{self._sequence}",
                "sequence": self._sequence,
                "component": kwargs["component"],
                "purpose": kwargs.get("purpose") or kwargs["component"],
                "provider": kwargs.get("provider"),
                "model": kwargs.get("model"),
                "api_mode": kwargs.get("api_mode"),
                "attempt": kwargs.get("attempt", 1),
                "status": "success",
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 5,
                    "cache_read_tokens": 2,
                    "cache_write_tokens": 0,
                    "reasoning_tokens": 1,
                    "request_count": 1,
                    "retry_count": 0,
                    "usage_complete": True,
                },
                "usage_available": True,
                "duration_ms": 3.0,
                "http_status": 200,
                "error_category": None,
            }
            self.model_call_usage_records.append(record)
            self.model_usage_callback(record.copy())
            return response

        async def _execute_recorded_async_model_call(self, *args, **kwargs):
            return None

        def run_conversation(self, **kwargs):
            captured["model_tools"] = tuple(
                tool["function"]["name"] for tool in self.tools
            )
            captured["valid_tool_names"] = set(self.valid_tool_names)
            captured["memory_nudge_interval"] = self._memory_nudge_interval
            captured["skill_nudge_interval"] = self._skill_nudge_interval
            assert self._memory_store.format_for_system_prompt("user") is None
            self._session_db.append_message(
                self.session_id,
                "user",
                "Always use TSV output.",
            )
            self._session_db.append_message(
                self.session_id,
                "assistant",
                "Understood.",
            )
            return {
                "final_response": "Understood.",
                "completed": True,
                "input_tokens": 0,
                "output_tokens": 0,
            }

        def wait_for_background_reviews(self, timeout=0):
            captured["background_review_requests"] = sum(
                record["component"] == "memory_controller"
                for record in self.model_call_usage_records
            )
            return True

    def call_llm(**kwargs):
        content = (
            json.dumps({"facts": [PRIVATE_MEMORY]})
            if kwargs["task"] == "semantic_fact_extraction"
            else json.dumps({
                "operations": [{
                    "fact_index": 0,
                    "action": "add",
                    "candidate_id": None,
                }],
            })
        )
        response = types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=content),
        )])
        return kwargs["request_executor"](
            lambda: response,
            attempt=1,
            purpose=kwargs["task"],
            provider="custom",
            model="fixture-model",
            api_mode="chat_completions",
        )

    fake_run_agent.AIAgent = StaticAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(auxiliary_client, "call_llm", call_llm)
    monkeypatch.setattr(memory_tool, "MEMORY_DIR", home / "memories")
    monkeypatch.setattr(
        HermesAdapter,
        "_reload_hermes_modules_if_needed",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")

    request = _request(home, artifacts, "native+ledger")
    request.session_id = "session-static-writeback"
    rsimem = request.model.extra_body["hermes"]["rsimem"]
    rsimem["lifecycle"] = {"evaluator_mode": "disabled"}
    rsimem["semantic_writeback"] = {
        "mode": "static_utility",
        "timeout_seconds": 10.0,
        "max_output_tokens": 512,
    }
    request.model.extra_body["hermes"]["enabled_toolsets"] = configured_toolsets
    request.model.extra_body["hermes"]["session_search_enabled"] = False
    adapter = HermesAdapter(
        AgentSpec(name="hermes", adapter="hermes"),
        request,
    )
    try:
        response = adapter.step(StepRequest(
            session_id=request.session_id,
            step_id=0,
        ))
        captured["static_failures"] = adapter._rsimem_bridge.static_failures
        captured["static_results"] = tuple(
            result.observer_evidence()
            for result in adapter._rsimem_bridge.static_results
        )
    finally:
        adapter.close("static fixture complete")

    assert response.status == "finished", response.error
    assert captured["enabled_toolsets"] == expected_toolsets
    assert captured["disabled_toolsets"] == ["memory"]
    assert captured["skip_memory"] is False
    assert captured["model_tools"] == ("skills_list",)
    assert captured["valid_tool_names"] == {"skills_list"}
    assert captured["memory_nudge_interval"] == 0
    assert captured["skill_nudge_interval"] == 0
    assert captured["background_review_requests"] == 0
    assert captured["static_failures"] == ()
    assert captured["static_results"][0]["writeback"]["logical_exit"] is True
    assert [record.component for record in response.model_calls] == [
        "semantic_fact_extraction",
        "semantic_operation_decision",
    ]
    assert response.usage.request_count == 2
    assert response.usage.input_tokens == 22
    assert response.usage.output_tokens == 10
    assert PRIVATE_MEMORY in (home / "memories" / "USER.md").read_text(
        encoding="utf-8"
    )
    ledger = (artifacts / "rsimem_memory_events.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"kind": "mutation_committed"' in ledger
    assert PRIVATE_MEMORY not in ledger
    assert PRIVATE_MEMORY not in (
        artifacts / "rsimem_semantic_operations.jsonl"
    ).read_text(encoding="utf-8")
    # A deployment without an output evaluator still closes the process
    # feedback loop: formation and exposure receipts are observable without
    # importing any benchmark grader or score.
    process_path = artifacts / "rsimem_process_feedback.jsonl"
    process_events = [
        json.loads(line) for line in process_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {event["kind"] for event in process_events} >= {
        "trigger",
        "source_selection",
        "extraction",
        "admission",
        "commit",
        "exposure",
        "task_outcome",
    }
    assert all("score" not in event and "grader" not in event for event in process_events)
    assert response.process_feedback_event_ids == sorted(event["event_id"] for event in process_events)
    assert audit_process_events(process_path) == ()
