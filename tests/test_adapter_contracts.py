from __future__ import annotations

import hashlib
import json

import pytest

from rsimem.adapter_contracts import (
    AdapterResult,
    AdapterStatus,
    BenchmarkPublicEvent,
    BenchmarkSplit,
    BenchmarkTaskRequest,
    CanonicalHostEvent,
    DeterministicHostAdapter,
    DeterministicMemoryMethodAdapter,
    FeedbackCondition,
    FeedbackView,
    FinalEvaluationRecord,
    HostCapabilities,
    HostEventKind,
    MemoryMethodAdapter,
    MethodCapabilities,
    MethodRunIdentity,
    MethodStateSnapshot,
    MethodUpdate,
    content_digest,
)
from rsimem.memory import MemoryKind
from rsimem.memory.evidence_planes import EvidencePlane
from rsimem.memory.lifecycle_surfaces import MemoryLifecycleSurface


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def test_benchmark_boundary_accepts_public_event_and_rejects_grader_fields() -> None:
    event = BenchmarkPublicEvent(
        event_id="benchmark-event.1",
        case_id="case.1",
        stage="task",
        event_type="turn.completed",
        public_state_digest=_sha({"state": "public"}),
        attributes={"turn_index": 1},
    )
    assert event.case_id == "case.1"
    with pytest.raises(ValueError, match="forbidden"):
        BenchmarkPublicEvent(
            event_id="benchmark-event.2",
            case_id="case.1",
            stage="task",
            event_type="turn.completed",
            public_state_digest=_sha({"state": "public"}),
            attributes={"nested": {"official_score": 1.0}},
        )


def test_benchmark_request_and_final_score_are_separate() -> None:
    request = BenchmarkTaskRequest(
        case_id="case.1",
        split=BenchmarkSplit.TRAIN,
        task_template_id="template.1",
        seed="seed.1",
        tool_budget=4,
        max_turns=8,
    )
    score = FinalEvaluationRecord(
        evaluation_id="evaluation.1",
        case_id=request.case_id,
        metric_id="past_bench.official_task_metric.v1",
        score_digest=_sha({"score": 1.0}),
    )
    assert request.split is BenchmarkSplit.TRAIN
    assert score.evidence_plane is EvidencePlane.FINAL_EVALUATION
    with pytest.raises(ValueError, match="final_evaluation"):
        FinalEvaluationRecord(
            evaluation_id="evaluation.2",
            case_id=request.case_id,
            metric_id="metric.1",
            score_digest=_sha({"score": 0.0}),
            evidence_plane=EvidencePlane.PURE_PROCESS,
        )


def test_host_contract_declares_three_memory_kinds_and_content_free_event() -> None:
    capabilities = HostCapabilities(
        memory_kinds=tuple(MemoryKind),
        tool_call_result_closure=True,
        usage_accounting=True,
        restart=True,
        context_snapshot=True,
        native_bypass=True,
    )
    assert set(capabilities.memory_kinds) == set(MemoryKind)
    event = CanonicalHostEvent(
        event_id="host-event.1",
        session_id="session.1",
        task_id="task.1",
        kind=HostEventKind.MEMORY_EXPOSURE,
        revision="revision.1",
        memory_kind=MemoryKind.SEMANTIC,
        surface=MemoryLifecycleSurface.RETRIEVAL_EXPOSURE,
        attributes={"artifact_ids": ["artifact.1"]},
    )
    assert event.memory_kind is MemoryKind.SEMANTIC
    with pytest.raises(ValueError, match="forbidden"):
        CanonicalHostEvent(
            event_id="host-event.2",
            session_id="session.1",
            task_id="task.1",
            kind=HostEventKind.TASK_COMPLETED,
            revision="revision.1",
            attributes={"answer": "hidden"},
        )


def test_feedback_conditions_are_strictly_allowlisted() -> None:
    f0 = FeedbackView(FeedbackCondition.F0, "2026-09-02T00:00:00Z")
    assert f0.allowed_fields == frozenset()
    f1 = FeedbackView(
        FeedbackCondition.F1,
        "2026-09-02T00:00:00Z",
        {"terminal_outcome": "completed"},
    )
    assert f1.value_digest != f0.value_digest
    with pytest.raises(ValueError, match="not allowed"):
        FeedbackView(
            FeedbackCondition.F1,
            "2026-09-02T00:00:00Z",
            {"trajectory": {"event": "hidden"}},
        )
    with pytest.raises(ValueError, match="forbidden"):
        FeedbackView(
            FeedbackCondition.F4,
            "2026-09-02T00:00:00Z",
            {"provenance_joins": {"family_id": "SM01"}},
        )
    with pytest.raises(ValueError, match="forbidden"):
        FeedbackView(
            FeedbackCondition.F5,
            "2026-09-02T00:00:00Z",
            {"counterfactual_replay": {"metadata": {"pointer": "F4"}}},
        )


def test_method_update_and_capabilities_are_typed_and_content_free() -> None:
    capabilities = MethodCapabilities(
        method_id="method.semantic.v1",
        primary_kind=MemoryKind.SEMANTIC,
        secondary_kind=None,
        transform=None,
        owned_surfaces=(MemoryLifecycleSurface.CONSTRUCTION,),
        required_feedback=(FeedbackCondition.F3,),
        required_host_capabilities=("context_snapshot", "restart"),
        state_schema="method.state.v1",
        lineage_schema="method.lineage.v1",
        online_update=True,
        validation=True,
        rollback=True,
    )
    assert capabilities.payload()["primary_kind"] == "semantic"
    update = MethodUpdate(
        update_id="update.semantic.1",
        target_surface=MemoryLifecycleSurface.CONSTRUCTION,
        affected_artifact_ids=("artifact.1",),
        base_revision="revision.1",
        observation_cutoff="2026-09-02T00:00:00Z",
        expected_behavior_change="construction_scope.v2",
        state_digest=_sha({"state": "candidate"}),
    )
    assert update.target_surface is MemoryLifecycleSurface.CONSTRUCTION
    with pytest.raises(ValueError, match="paired"):
        MethodCapabilities(
            method_id="method.invalid.v1",
            primary_kind=MemoryKind.SEMANTIC,
            secondary_kind=MemoryKind.EPISODIC,
            transform=None,
            owned_surfaces=(),
            required_feedback=(),
            required_host_capabilities=(),
            state_schema="state.v1",
            lineage_schema="lineage.v1",
            online_update=False,
            validation=False,
            rollback=False,
        )


def test_method_protocol_can_be_satisfied_without_benchmark_or_host_types() -> None:
    class FakeMethod:
        def describe_capabilities(self):
            return MethodCapabilities(
                method_id="method.fake.v1",
                primary_kind=MemoryKind.SEMANTIC,
                secondary_kind=None,
                transform=None,
                owned_surfaces=(),
                required_feedback=(),
                required_host_capabilities=(),
                state_schema="state.v1",
                lineage_schema="lineage.v1",
                online_update=False,
                validation=False,
                rollback=False,
            )

        def prepare_run(self, run): return AdapterResult(AdapterStatus.SUPPORTED, "operation.prepare")
        def start_episode(self, run): return AdapterResult(AdapterStatus.SUPPORTED, "operation.start")
        def observe_event(self, event): return AdapterResult(AdapterStatus.SUPPORTED, "operation.observe")
        def finalize_episode(self, run): return AdapterResult(AdapterStatus.SUPPORTED, "operation.finalize")
        def snapshot_state(self): return MethodStateSnapshot("state.1", "revision.1", "state.v1", _sha({}), False)
        def propose_update(self, feedback): return AdapterResult(AdapterStatus.UNSUPPORTED, "operation.propose", "unsupported"), None
        def validate_update(self, update): return AdapterResult(AdapterStatus.UNSUPPORTED, "operation.validate", "unsupported")
        def activate_update(self, update): return AdapterResult(AdapterStatus.UNSUPPORTED, "operation.activate", "unsupported")
        def rollback_update(self, update): return AdapterResult(AdapterStatus.UNSUPPORTED, "operation.rollback", "unsupported")

    assert isinstance(FakeMethod(), MemoryMethodAdapter)
    assert FakeMethod().snapshot_state().state_schema == "state.v1"


def test_deterministic_method_adapter_is_kind_scoped_and_revision_safe() -> None:
    method = DeterministicMemoryMethodAdapter(MethodCapabilities(
        method_id="method.semantic.fixture.v1",
        primary_kind=MemoryKind.SEMANTIC,
        secondary_kind=None,
        transform=None,
        owned_surfaces=(MemoryLifecycleSurface.CONSTRUCTION,),
        required_feedback=(FeedbackCondition.F2,),
        required_host_capabilities=(),
        state_schema="method.state.fixture.v1",
        lineage_schema="method.lineage.fixture.v1",
        online_update=True,
        validation=True,
        rollback=True,
    ))
    run = MethodRunIdentity("run.fixture.v1", "session.fixture.v1", "task.fixture.v1", "revision.initial")
    assert method.prepare_run(run).status is AdapterStatus.SUPPORTED
    assert method.start_episode(run).status is AdapterStatus.SUPPORTED
    event = CanonicalHostEvent(
        event_id="host-event.semantic.v1",
        session_id=run.session_id,
        task_id=run.task_id,
        kind=HostEventKind.MEMORY_RETRIEVAL,
        revision="revision.initial",
        memory_kind=MemoryKind.SEMANTIC,
    )
    assert method.observe_event(event).status is AdapterStatus.SUPPORTED
    assert method.observe_event(event).reason_code == "duplicate_event"
    episodic = CanonicalHostEvent(
        event_id="host-event.episodic.v1",
        session_id=run.session_id,
        task_id=run.task_id,
        kind=HostEventKind.MEMORY_RETRIEVAL,
        revision="revision.initial",
        memory_kind=MemoryKind.EPISODIC,
    )
    assert method.observe_event(episodic).reason_code == "kind_mismatch"

    f1 = FeedbackView(FeedbackCondition.F1, "2026-09-02T00:00:00Z", {"terminal_outcome": "completed"})
    rejected, update = method.propose_update(f1)
    assert rejected.status is AdapterStatus.REJECTED
    assert update is None
    f2 = FeedbackView(FeedbackCondition.F2, "2026-09-02T00:00:00Z", {"terminal_outcome": "completed", "trajectory": {"digest": _sha({"t": 1})}})
    accepted, update = method.propose_update(f2)
    assert accepted.status is AdapterStatus.ACCEPTED
    assert update is not None
    assert method.validate_update(update).status is AdapterStatus.ACCEPTED
    assert method.activate_update(update).status is AdapterStatus.ACCEPTED
    assert method.activate_update(update).reason_code == "duplicate_activation"
    assert method.validate_update(update).status is AdapterStatus.STALE
    assert method.rollback_update(update).status is AdapterStatus.ACCEPTED


def test_deterministic_host_adapter_preserves_restart_identity_and_rejects_cross_session() -> None:
    host = DeterministicHostAdapter(HostCapabilities(
        memory_kinds=tuple(MemoryKind),
        tool_call_result_closure=True,
        usage_accounting=True,
        restart=True,
        context_snapshot=True,
        native_bypass=True,
    ))
    run = MethodRunIdentity("run.host.v1", "session.host.v1", "task.host.v1", "revision.initial")
    assert host.prepare_session(run).status is AdapterStatus.SUPPORTED
    event = CanonicalHostEvent(
        event_id="host-event.turn.v1",
        session_id=run.session_id,
        task_id=run.task_id,
        kind=HostEventKind.TURN_COMPLETED,
        revision="revision.initial",
    )
    assert host.observe_event(event).status is AdapterStatus.SUPPORTED
    before = host.snapshot_state()
    assert host.restart(run).status is AdapterStatus.SUPPORTED
    assert host.snapshot_state() == before
    assert host.observe_event(event).reason_code == "duplicate_event"
    cross = CanonicalHostEvent(
        event_id="host-event.cross.v1",
        session_id="session.other.v1",
        task_id=run.task_id,
        kind=HostEventKind.TURN_COMPLETED,
        revision="revision.initial",
    )
    assert host.observe_event(cross).reason_code == "session_task_mismatch"
