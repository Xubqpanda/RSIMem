from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from past_bench.models.content import TextBlock
from past_bench.models.message import Message
from past_bench.runtime.adapters.hermes import HermesAdapter
from past_bench.runtime.protocol import (
    RuntimeConfigPayload,
    RuntimeModelConfig,
    StartSessionRequest,
    StepRequest,
)
from past_bench.runtime.registry import AgentSpec


PRIVATE_MEMORY = "Use TSV with owner, priority, task, and due_date."


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
    return StartSessionRequest(
        session_id=f"session-{mode}",
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
    for mode in ("native", "native+ledger", "native+adapter+ledger"):
        artifacts = tmp_path / mode.replace("+", "_")
        adapter = HermesAdapter(
            AgentSpec(name="hermes", adapter="hermes"),
            _request(home, artifacts, mode),
        )
        try:
            response = adapter.step(StepRequest(
                session_id=f"session-{mode}",
                step_id=0,
            ))
        finally:
            adapter.close("fixture complete")
        assert response.status == "finished", response.error
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
