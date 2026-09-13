from __future__ import annotations

import hashlib
import json

import pytest

from rsimem.memory.hook_contract import (
    MemoryHookDecision,
    MemoryHookEvent,
    MemoryHookPoint,
    MemoryHookType,
)
from rsimem.memory.process_feedback import ProcessEvent, ProcessEventKind, ProcessEventStatus
from rsimem.memory.surface_policy import (
    MemorySurface,
    RuntimeSurfacePolicy,
    SurfaceOperation,
    capability_check,
)


def test_surface_policy_keeps_expected_signal_separate_from_availability() -> None:
    base = RuntimeSurfacePolicy.from_hermes_flags(
        memory_enabled=True,
        user_profile_enabled=True,
        skills_enabled=True,
        session_search_enabled=True,
    )
    skill = base.for_expected_signal("skill")
    assert base.enabled(MemorySurface.SEMANTIC)
    assert not skill.enabled(MemorySurface.SEMANTIC)
    assert skill.enabled(MemorySurface.PROCEDURAL)
    assert capability_check(skill, MemorySurface.PROCEDURAL, SurfaceOperation.RETRIEVE)
    assert not capability_check(skill, MemorySurface.EPISODIC, SurfaceOperation.RETRIEVE)


def test_all_memory_off_policy_is_fail_closed_and_round_trips() -> None:
    policy = RuntimeSurfacePolicy.all_memory_off()
    assert all(not policy.enabled(surface) for surface in MemorySurface)
    assert all(
        not capability_check(policy, surface, SurfaceOperation.RETRIEVE)
        for surface in MemorySurface
    )
    assert RuntimeSurfacePolicy.from_payload(policy.payload()) == policy


def test_surface_policy_rejects_all_memory_off_with_any_enabled_switch() -> None:
    with pytest.raises(ValueError, match="AllMemoryOff"):
        RuntimeSurfacePolicy.from_hermes_flags(
            memory_enabled=True,
            user_profile_enabled=False,
            skills_enabled=False,
            session_search_enabled=False,
            all_memory_off=True,
        )


def test_hook_event_has_stable_content_free_identity() -> None:
    provenance = hashlib.sha256(b"fixture provenance").hexdigest()
    event = MemoryHookEvent.create(
        run_id="run.test",
        episode_id="episode.test",
        session_id="session.test",
        task_id="task.test",
        memory_type=MemoryHookType.SEMANTIC,
        hook_point=MemoryHookPoint.RETRIEVAL,
        operation="retrieve",
        decision=MemoryHookDecision.OBSERVED,
        result_code="retrieval_miss",
        provenance_digest=provenance,
        surface_policy_id="mem0-semantic-v1",
    )
    payload = event.payload()
    assert payload["event_id"] == event.event_id
    assert "content" not in json.dumps(payload)
    assert MemoryHookEvent(**{**event.identity_payload(), "event_id": event.event_id}) == event


def test_hook_event_rejects_non_digest_provenance() -> None:
    with pytest.raises(ValueError, match="provenance digest"):
        MemoryHookEvent.create(
            run_id="run.test", episode_id="episode.test", session_id="session.test",
            task_id="task.test", memory_type="semantic", hook_point="outcome",
            operation="outcome", decision="observed", result_code="task_completed",
            provenance_digest="not-a-digest", surface_policy_id="mem0-semantic-v1",
        )


def test_hook_event_bridges_to_and_from_process_evidence_without_content() -> None:
    process = ProcessEvent.create(
        kind=ProcessEventKind.RETRIEVAL,
        status=ProcessEventStatus.SUCCESS,
        run_id="run.bridge",
        variant="mem0",
        trace_id="trace.bridge",
        episode_id="episode.bridge",
        session_id="session.bridge",
        task_id="task.bridge",
        host_event_id="host.bridge",
        source_revision="revision.bridge",
        input_payload={"query_digest": "query"},
        output_payload={"hit_count": 1},
    )
    hook = MemoryHookEvent.from_process_event(
        process,
        memory_type=MemoryHookType.SEMANTIC,
        hook_point=MemoryHookPoint.RETRIEVAL,
        operation="memory_search",
        result_code="retrieval_hit",
        surface_policy_id="mem0-semantic-v1",
    )
    projected = hook.to_process_event(
        variant="mem0",
        trace_id="trace.bridge.hook",
        host_event_id="host.bridge.hook",
        source_revision="revision.bridge",
    )
    assert process.event_id in hook.parent_event_ids
    assert hook.provenance_digest == hashlib.sha256(
        json.dumps(process.payload(), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert hook.event_id in projected.execution_receipt_ids
    assert "content" not in json.dumps(projected.payload())
