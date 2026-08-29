from __future__ import annotations

import json
from types import SimpleNamespace

from rsimem.hermes_past_bridge import HermesPastBenchBridge
from rsimem.hermes_integration import HermesExecutionMode, HermesExperimentConfig
from rsimem.lifecycle import HermesLifecycleConfig, TaskLifecycleState
from rsimem.memory.tool_exact_join import ToolJoinResolutionStatus, resolve_tool_call_result


def _home(tmp_path):
    memories = tmp_path / "memories"
    memories.mkdir(parents=True)
    (memories / "MEMORY.md").write_text("durable fixture", encoding="utf-8")
    (memories / "USER.md").write_text("user fixture", encoding="utf-8")
    skills = tmp_path / "skills" / "operations" / "fixture"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("---\nname: fixture\ndescription: fixture\n---\nrun", encoding="utf-8")
    return tmp_path


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def get_messages(self, session_id):
        return list(self.rows)


def _bridge(tmp_path, rows):
    home = _home(tmp_path / "home")
    bridge = HermesPastBenchBridge(
        home,
        HermesExperimentConfig(HermesExecutionMode.NATIVE_LEDGER),
        evidence_path=tmp_path / "artifacts" / "memory.jsonl",
        run_id="run-shadow-boundaries",
        trace_id="trace-shadow-boundaries",
        episode_id="episode-shadow-boundaries",
        session_id="session-shadow-boundaries",
        task_id="task-shadow-boundaries",
        experiment_variant="native+ledger",
        lifecycle_config=HermesLifecycleConfig(),
    )
    bridge.attach(SimpleNamespace(
        _memory_store=None,
        _session_db=_Rows(rows),
        session_id="session-shadow-boundaries",
    ))
    return bridge


def test_real_shadow_boundaries_are_observed_without_evaluator_or_mutation(tmp_path) -> None:
    rows = [
        {"id": 1, "role": "user", "content": "turn", "token_count": 1},
        {"id": 2, "role": "assistant", "content": "done", "token_count": 1},
    ]
    bridge = _bridge(tmp_path, rows)
    try:
        bridge.on_session_end(task_state=TaskLifecycleState.COMPLETED)
        bridge.on_turn_interval(turn_index=1)
        bridge.on_context_pressure(context_tokens=4096)
        bridge.on_manual_trigger()
    finally:
        bridge.close()
    assert len(bridge.trigger_observations) == 4
    assert all(item.shadow_only for item in bridge.trigger_observations)
    assert all(item.decision.action.value == "SKIP" for item in bridge.trigger_observations)
    assert bridge.static_results == ()


def test_tool_boundary_preserves_open_closure_and_cannot_run_source_selection(tmp_path) -> None:
    rows = [
        {"id": 1, "role": "user", "content": "call tool", "token_count": 1},
        {
            "id": 2,
            "role": "assistant",
            "content": "",
            "token_count": 1,
            "tool_calls": [{"id": "call-1", "function": {"name": "inspect", "arguments": "{}"}}],
        },
    ]
    bridge = _bridge(tmp_path, rows)
    try:
        bridge.on_tool_boundary(turn_index=1)
    finally:
        bridge.close()
    assert len(bridge.trigger_observations) == 1
    assert bridge.trigger_observations[0].event.supported is True
    assert bridge.trigger_observations[0].decision.action.value == "SKIP"
    assert bridge.source_selection_decisions[0].selected_segment_ids == ()


def test_task_result_projects_every_tool_call_and_result_without_content(tmp_path) -> None:
    bridge = _bridge(tmp_path, [])
    result = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-inspect-1",
                    "function": {"name": "inspect", "arguments": "{\"path\": \"/private\"}"},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call-inspect-1",
                "content": "{\"success\": true, \"content\": \"secret\"}",
            },
        ],
        "completed": True,
    }
    try:
        bridge._record_tool_call_results(
            result,
            memory_use_operation_id="op.use.fixture.v1",
        )
        events = bridge.process_feedback
    finally:
        bridge.close()
    calls = [event for event in events if event.kind.value == "tool_call"]
    results = [event for event in events if event.kind.value == "tool_result"]
    assert len(calls) == 1
    assert len(results) == 1
    assert calls[0].tool_call_id == "call-inspect-1"
    assert results[0].tool_result_id is not None
    assert bridge.tool_call_result_joins[0].memory_use_operation_id == (
        "op.use.fixture.v1"
    )
    serialized = json.dumps([event.payload() for event in events], ensure_ascii=True)
    assert "private" not in serialized
    assert "secret" not in serialized


def test_duplicate_tool_call_id_is_marked_duplicate_and_fails_closed(tmp_path) -> None:
    bridge = _bridge(tmp_path, [])
    result = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-duplicate-1",
                    "function": {"name": "inspect", "arguments": "{}"},
                }],
            },
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-duplicate-1",
                    "function": {"name": "inspect", "arguments": "{}"},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call-duplicate-1",
                "content": "{\"success\": true}",
            },
        ],
    }
    try:
        bridge._record_tool_call_results(result)
    finally:
        bridge.close()
    joins = bridge.tool_call_result_joins
    assert len(joins) == 1
    assert joins[0].duplicate_call is True
    resolution = resolve_tool_call_result(joins[0])
    assert resolution.status is ToolJoinResolutionStatus.DUPLICATE


def test_duplicate_tool_call_id_without_result_remains_duplicate_missing_closure(tmp_path) -> None:
    bridge = _bridge(tmp_path, [])
    result = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-duplicate-open",
                    "function": {"name": "inspect", "arguments": "{}"},
                }],
            },
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-duplicate-open",
                    "function": {"name": "inspect", "arguments": "{}"},
                }],
            },
        ],
    }
    try:
        bridge._record_tool_call_results(result)
    finally:
        bridge.close()
    joins = bridge.tool_call_result_joins
    assert len(joins) == 1
    assert joins[0].duplicate_call is True
    assert joins[0].result_present is False
    resolution = resolve_tool_call_result(joins[0])
    assert resolution.status is ToolJoinResolutionStatus.DUPLICATE
