from __future__ import annotations

import json

import pytest

from rsimem.lifecycle import (
    ContextAction,
    ContextEvaluationRequest,
    ContextSegment,
    ConservativeContextEvaluator,
    EvaluationCadence,
    EvaluationEvent,
    EvaluationScheduler,
    EvaluationTrigger,
    JsonLlmContextEvaluator,
    LifecycleController,
    AllowlistedUpdateTargetResolver,
    WritebackCoordinator,
    WritebackPlanValidator,
    WritebackAction,
    run_sm01_preference_fixture,
    snapshot_to_evaluation_request,
)
from rsimem.memory import (
    MemoryAccessMode,
    MemoryArtifact,
    MemoryBackendDescriptor,
    MemoryBackendRegistry,
    MemoryKind,
    MemoryKindCapability,
)


class _TrustedSemanticBackend:
    artifact = MemoryArtifact(
        "trusted-artifact",
        MemoryKind.SEMANTIC,
        "Stored preference.",
        revision="trusted-revision-3",
    )

    @property
    def descriptor(self) -> MemoryBackendDescriptor:
        return MemoryBackendDescriptor(
            "hermes-native-semantic",
            (MemoryKindCapability(
                MemoryKind.SEMANTIC,
                MemoryAccessMode.SEARCH,
            ),),
        )

    def get(self, artifact_id: str) -> MemoryArtifact | None:
        return self.artifact if artifact_id == self.artifact.artifact_id else None

    def query(self, query):
        return ()

    def mutate(self, mutation):
        raise AssertionError("resolver tests must not mutate memory")

    def close(self) -> None:
        return None


def _request(trigger: EvaluationTrigger = EvaluationTrigger.TASK_COMPLETED) -> ContextEvaluationRequest:
    return ContextEvaluationRequest(
        evaluation_id="eval-1",
        session_id="session-1",
        task_id="task-1",
        trigger=trigger,
        turn_index=4,
        context_tokens=1200,
        active_segment_ids=("current",),
        segments=(
            ContextSegment("done", "tool", "Completed deployment output.", completed=True),
            ContextSegment("current", "user", "Continue with the active task."),
        ),
    )


def test_scheduler_supports_event_driven_and_periodic_cadence() -> None:
    scheduler = EvaluationScheduler(EvaluationCadence(
        on_task_completed=False,
        context_pressure_tokens=1000,
        every_n_turns=3,
    ))
    assert not scheduler.should_evaluate(EvaluationEvent(EvaluationTrigger.TASK_COMPLETED, 1))
    assert scheduler.should_evaluate(EvaluationEvent(EvaluationTrigger.CONTEXT_PRESSURE, 2, 1200))
    scheduler.mark_evaluated(EvaluationEvent(EvaluationTrigger.CONTEXT_PRESSURE, 2, 1200), "eval-1")
    assert not scheduler.should_evaluate(EvaluationEvent(EvaluationTrigger.TURN_INTERVAL, 2))
    assert scheduler.should_evaluate(EvaluationEvent(EvaluationTrigger.TURN_INTERVAL, 6))


def test_controller_rejects_eviction_of_active_context() -> None:
    class UnsafeEvaluator(ConservativeContextEvaluator):
        def evaluate(self, request):
            result = super().evaluate(request)
            from dataclasses import replace

            return replace(result, signals=(
                result.signals[0],
                replace(result.signals[1], context_action=ContextAction.EVICT),
            ))

    with pytest.raises(ValueError, match="active segments"):
        LifecycleController(UnsafeEvaluator()).evaluate(_request())


def test_controller_evaluates_once_and_notifies_observer() -> None:
    seen = []

    class Observer:
        def record(self, evaluation):
            seen.append(evaluation)

    controller = LifecycleController(
        ConservativeContextEvaluator(),
        observers=(Observer(),),
    )
    request = _request()
    evaluation = controller.evaluate(request)
    assert evaluation is not None
    assert len(evaluation.signals) == 2
    assert controller.evaluate(request) is None
    assert len(seen) == 1


def test_json_llm_evaluator_requires_complete_typed_output() -> None:
    payload = {
        "policy_version": "llm-v1",
        "signals": [
            {
                "segment_id": "done",
                "context_action": "evict",
                "writeback_action": "add",
                "memory_kind": "episodic",
                "utility_estimate": 0.7,
                "confidence": 0.9,
                "reason_codes": ["completed", "future_reuse"],
                "completion_status": "completed",
                "completion_evidence": ["task_boundary"],
                "scope": "user",
                "temporal_validity": "durable",
                "reusable_facts": ["candidate preference"],
                "reusable_procedures": [],
                "update_hints": [],
            },
            {
                "segment_id": "current",
                "context_action": "retain",
                "writeback_action": "defer",
                "memory_kind": None,
                "utility_estimate": 0.2,
                "confidence": 0.8,
                "reason_codes": ["active"],
            },
        ],
    }
    evaluator = JsonLlmContextEvaluator(lambda prompt: json.dumps(payload))
    evaluation = evaluator.evaluate(_request())
    assert evaluation.signals[0].writeback_action == WritebackAction.ADD
    assert evaluation.signals[0].memory_kind.value == "episodic"
    assert evaluation.signals[0].completion_status.value == "completed"
    assert evaluation.signals[0].scope.value == "user"
    assert evaluation.signals[0].safe_to_evict is None
    assert evaluation.signals[0].provenance == ()
    assert '"content": "Completed deployment output."' in evaluator.build_prompt(_request())


def test_json_evaluator_prompt_carries_host_request_identity() -> None:
    request = _request()
    from dataclasses import replace

    request = replace(
        request,
        context_revision="rev-host-7",
        metadata={"snapshot_id": "snapshot-7", "policy_version": "host-policy-7"},
    )
    prompt = json.loads(JsonLlmContextEvaluator(lambda _: "{}").build_prompt(request))
    assert prompt["evaluation_id"] == "eval-1"
    assert prompt["context_revision"] == "rev-host-7"
    assert prompt["turn_index"] == 4
    assert prompt["protected_segment_ids"] == ["current"]
    assert prompt["host_policy_version"] == "host-policy-7"


def test_json_llm_evaluator_rejects_missing_segment() -> None:
    evaluator = JsonLlmContextEvaluator(lambda _: '{"signals": []}')
    with pytest.raises(ValueError, match="omitted segment IDs"):
        evaluator.evaluate(_request())


def test_llm_update_requires_trusted_target_resolution() -> None:
    fixture = run_sm01_preference_fixture()
    request = snapshot_to_evaluation_request(
        fixture.snapshot,
        evaluation_id="llm-update",
    )
    preference_id = fixture.snapshot.segments[0].segment_id
    payload = {
        "policy_version": "llm-update-v1",
        "signals": [
            {
                "segment_id": segment.segment_id,
                "context_action": "evict" if segment.segment_id == preference_id else "retain",
                "writeback_action": "update" if segment.segment_id == preference_id else "defer",
                "memory_kind": "semantic" if segment.segment_id == preference_id else None,
                "utility_estimate": 0.9,
                "confidence": 0.8,
                "completion_status": "completed",
                "completion_evidence": ["model_predicted_complete"],
                "scope": "user",
                "temporal_validity": "durable",
                "reusable_facts": ["candidate preference"],
                "reusable_procedures": ["candidate formatting procedure"],
                "update_hints": (
                    ["replace_preference", "preserve_columns"]
                    if segment.segment_id == preference_id
                    else []
                ),
                "update_mode": (
                    "replace" if segment.segment_id == preference_id else None
                ),
                "compiler_version": (
                    "compiler-v4"
                    if segment.segment_id == preference_id
                    else "uncompiled-v0"
                ),
                "safe_to_evict": False,
                "target_backend": "untrusted-backend",
                "target_artifact_id": "untrusted-artifact",
                "expected_memory_revision": "untrusted-revision",
                "reason_codes": ["preference_update"],
            }
            for segment in request.segments
        ],
    }
    evaluation = JsonLlmContextEvaluator(
        lambda _: json.dumps(payload),
        compiler_version="compiler-v4",
    ).evaluate(request)
    signal = evaluation.signals[0]
    assert signal.target_backend is None
    assert signal.target_artifact_id is None
    assert signal.expected_memory_revision is None
    assert signal.safe_to_evict is None
    assert signal.compiler_version == "compiler-v4"

    registry = MemoryBackendRegistry()
    registry.register(_TrustedSemanticBackend())
    resolver = AllowlistedUpdateTargetResolver(
        registry,
        lambda snapshot, candidate: ("trusted-artifact",),
        allowed_backends={
            "hermes-native-semantic": frozenset({MemoryKind.SEMANTIC}),
        },
    )
    coordinator = WritebackCoordinator(
        target_resolver=resolver,
        validator=WritebackPlanValidator(updatable_backends={
            "hermes-native-semantic": frozenset({MemoryKind.SEMANTIC}),
        }),
    )
    plan = coordinator.create_plans(fixture.snapshot, evaluation)[0]
    assert plan.target_backend == "hermes-native-semantic"
    assert plan.target_artifact_id == "trusted-artifact"
    assert plan.expected_memory_revision == "trusted-revision-3"
    assert plan.exit_evidence.safe_to_evict is True
    assert plan.exit_evidence.completion_status.value == "completed"
    assert plan.exit_evidence.completion_evidence == ("model_predicted_complete",)
    assert plan.exit_evidence.reusable_facts == ("candidate preference",)
    assert plan.exit_evidence.reusable_procedures == (
        "candidate formatting procedure",
    )
    assert plan.exit_evidence.update_hints == (
        "replace_preference",
        "preserve_columns",
    )
