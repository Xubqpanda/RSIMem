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
    WritebackAction,
)


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
    assert '"content": "Completed deployment output."' in evaluator.build_prompt(_request())


def test_json_llm_evaluator_rejects_missing_segment() -> None:
    evaluator = JsonLlmContextEvaluator(lambda _: '{"signals": []}')
    with pytest.raises(ValueError, match="omitted segment IDs"):
        evaluator.evaluate(_request())
