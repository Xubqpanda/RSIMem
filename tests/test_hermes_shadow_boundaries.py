from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

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


@pytest.mark.parametrize("content", ("not-json", '{"success": "false"}'))
def test_malformed_tool_result_is_type_mismatch_not_tool_failure(
    tmp_path,
    content,
) -> None:
    bridge = _bridge(tmp_path, [])
    result = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-malformed-result",
                    "function": {"name": "inspect", "arguments": "{}"},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call-malformed-result",
                "content": content,
            },
        ],
    }
    try:
        bridge._record_tool_call_results(result)
        join = bridge.tool_call_result_joins[-1]
        events = bridge.process_feedback
    finally:
        bridge.close()

    assert join.type_mismatch is True
    assert join.success is None
    assert resolve_tool_call_result(join).status is ToolJoinResolutionStatus.TYPE_MISMATCH
    tool_results = [event for event in events if event.kind.value == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0].status.value == "unknown"
    assert tool_results[0].reason_codes == ("schema_failure",)
    assert "tool_failure" not in tool_results[0].reason_codes


def test_cross_task_tool_closure_is_not_attributed_to_current_task(tmp_path) -> None:
    bridge = _bridge(tmp_path, [])
    result = {
        "messages": [
            {
                "role": "assistant",
                "task_id": "task.other",
                "tool_calls": [{
                    "id": "call-cross-task",
                    "function": {"name": "inspect", "arguments": "{}"},
                }],
            },
            {
                "role": "tool",
                "task_id": "task.other",
                "tool_call_id": "call-cross-task",
                "content": '{"success": true}',
            },
        ],
    }
    try:
        bridge._record_tool_call_results(result)
        join = bridge.tool_call_result_joins[-1]
    finally:
        bridge.close()

    assert join.cross_task is True
    resolution = resolve_tool_call_result(join)
    assert resolution.status is ToolJoinResolutionStatus.CROSS_TASK
    assert resolution.exact is False


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
    assert len(joins) == 2
    assert all(join.duplicate_call for join in joins)
    assert sum(join.result_present for join in joins) == 1
    assert sum(event.kind.value == "tool_call" for event in bridge.process_feedback) == 2
    assert sum(event.kind.value == "tool_result" for event in bridge.process_feedback) == 1
    assert all(
        resolve_tool_call_result(join).status is ToolJoinResolutionStatus.DUPLICATE
        for join in joins
    )


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
    assert len(joins) == 2
    assert all(join.duplicate_call for join in joins)
    assert all(join.result_present is False for join in joins)
    assert all(
        resolve_tool_call_result(join).status is ToolJoinResolutionStatus.DUPLICATE
        for join in joins
    )


def test_replayed_tool_identity_is_marked_duplicate_across_bridge_calls(tmp_path) -> None:
    bridge = _bridge(tmp_path, [])
    result = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-replayed-1",
                    "function": {"name": "inspect", "arguments": "{}"},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call-replayed-1",
                "id": "result-replayed-1",
                "content": "{\"success\": true}",
            },
        ],
    }
    try:
        bridge._record_tool_call_results(result)
        bridge._record_tool_call_results(result)
        joins = bridge.tool_call_result_joins
        assert len(joins) == 2
        assert any(not join.duplicate_call and not join.duplicate_result for join in joins)
        assert any(join.duplicate_call and join.duplicate_result for join in joins)
        statuses = tuple(resolve_tool_call_result(join).status for join in joins)
        assert statuses.count(ToolJoinResolutionStatus.COMPLETE) == 1
        assert statuses.count(ToolJoinResolutionStatus.DUPLICATE) == 1
    finally:
        bridge.close()
