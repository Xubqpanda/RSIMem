from __future__ import annotations

from rsimem.lifecycle import (
    HermesMessage,
    HermesSnapshotCollector,
    SegmentKind,
    TaskLifecycleState,
)
from rsimem.memory.extraction_source import (
    EXTRACTION_SOURCE_METADATA_ALLOWLIST,
    EXTRACTION_SOURCE_SCHEMA,
    EXTRACTION_SOURCE_SCHEMA_VERSION,
    ExtractionSourceProjector,
)
from rsimem.memory.ingestion import build_completed_task_semantic_ingest_request
from rsimem.memory_systems.mem0_flat import POLICY_FACT_EXTRACTION_PROMPT


def _messages(*, changed: str = "Keep TSV output.") -> tuple[HermesMessage, ...]:
    return (
        HermesMessage(
            "hidden-system",
            "system",
            "SENTINEL_HIDDEN_GRADER",
            "turn-0",
            4,
            completed=True,
            metadata={"benchmark": "must-not-project"},
        ),
        HermesMessage("user-1", "user", changed, "turn-1", 4, completed=True),
        HermesMessage(
            "tool-call-1",
            "assistant",
            '{"function":{"name":"inspect"}}',
            "turn-1",
            5,
            kind=SegmentKind.TOOL_CALL,
            completed=True,
            tool_call_id="call-1",
        ),
        HermesMessage(
            "tool-result-1",
            "tool",
            "inspection succeeded",
            "turn-1",
            4,
            kind=SegmentKind.TOOL_RESULT,
            completed=True,
            tool_call_id="call-1",
        ),
        HermesMessage(
            "assistant-1",
            "assistant",
            "Task completed.",
            "turn-1",
            3,
            completed=True,
        ),
    )


def _snapshot(messages=None):
    return HermesSnapshotCollector().collect(
        messages or _messages(),
        run_id="run-projection",
        episode_id="episode-projection",
        session_id="session-projection",
        task_id="task-projection",
        current_turn_id=None,
        task_state=TaskLifecycleState.COMPLETED,
        lifecycle_state="task_completed",
        source_ref="fixture:projection",
    )


def test_projection_is_stable_bounded_and_exactly_matches_prompt_input() -> None:
    snapshot = _snapshot()
    projection = ExtractionSourceProjector().project(snapshot)
    replay = ExtractionSourceProjector().project(snapshot)

    assert projection == replay
    assert projection.schema == EXTRACTION_SOURCE_SCHEMA
    assert projection.schema_version == EXTRACTION_SOURCE_SCHEMA_VERSION
    assert projection.source_message_ids == (
        "user-1",
        "tool-call-1",
        "tool-result-1",
        "assistant-1",
    )
    assert projection.source_message_ids == tuple(
        value["source_message_id"] for value in projection.prompt_messages()
    )
    assert set(projection.prompt_messages()[0]) == set(
        EXTRACTION_SOURCE_METADATA_ALLOWLIST
    ) | {"role", "content", "content_truncated"}
    assert snapshot.segments[0].segment_id in projection.omitted_segment_ids
    assert "SENTINEL_HIDDEN_GRADER" not in str(projection.prompt_messages())
    assert "benchmark" not in str(projection.prompt_messages())

    request = build_completed_task_semantic_ingest_request(
        snapshot,
        policy_version="policy-v1",
        framework_version="framework-v1",
    )
    assert request.provenance.source.segment_ids == projection.source_segment_ids
    assert request.source_projection.projection_digest == projection.projection_digest
    rendered = POLICY_FACT_EXTRACTION_PROMPT.render({
        "source_messages": request.source_projection.prompt_messages(),
        "source_projection_digest": projection.projection_digest,
        "exit_evidence": request.exit_evidence.compiler_input_payload(),
    })
    assert projection.projection_digest in rendered.text
    assert "SENTINEL_HIDDEN_GRADER" not in rendered.text


def test_projection_and_request_identity_cover_add_remove_reorder_and_content() -> None:
    baseline = _snapshot()
    variants = (
        _snapshot(_messages(changed="Keep CSV output.")),
        _snapshot((*_messages(), HermesMessage(
            "assistant-2", "assistant", "Additional outcome.", "turn-2", 3,
            completed=True,
        ))),
        _snapshot(tuple(reversed(_messages()))),
        _snapshot(_messages()[1:]),
    )
    baseline_request = build_completed_task_semantic_ingest_request(
        baseline,
        policy_version="policy-v1",
        framework_version="framework-v1",
    )
    for snapshot in variants:
        request = build_completed_task_semantic_ingest_request(
            snapshot,
            policy_version="policy-v1",
            framework_version="framework-v1",
        )
        assert request.source_projection.projection_digest != (
            baseline_request.source_projection.projection_digest
        )
        assert request.idempotency_key != baseline_request.idempotency_key


def test_budget_truncation_keeps_tool_call_and_result_atomic() -> None:
    snapshot = _snapshot((
        HermesMessage("older", "user", "u" * 20, "turn-1", 5, completed=True),
        HermesMessage(
            "call", "assistant", "c" * 20, "turn-1", 5,
            kind=SegmentKind.TOOL_CALL, completed=True, tool_call_id="call-1",
        ),
        HermesMessage(
            "result", "tool", "r" * 20, "turn-1", 5,
            kind=SegmentKind.TOOL_RESULT, completed=True, tool_call_id="call-1",
        ),
        HermesMessage("newest", "assistant", "done!", "turn-1", 2, completed=True),
    ))
    projection = ExtractionSourceProjector(max_content_chars=25).project(snapshot)

    assert projection.projected_content_chars == 25
    assert projection.source_message_ids == ("call", "result", "newest")
    assert set(projection.truncated_segment_ids) == {
        snapshot.segments[1].segment_id,
        snapshot.segments[2].segment_id,
    }
    assert snapshot.segments[0].segment_id in projection.omitted_segment_ids
    assert {
        message.tool_call_id for message in projection.messages
        if message.segment_kind in {SegmentKind.TOOL_CALL, SegmentKind.TOOL_RESULT}
    } == {"call-1"}
