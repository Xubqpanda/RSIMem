"""Orchestration for scheduled context evaluation."""

from __future__ import annotations

from collections.abc import Iterable

from .contracts import (
    ContextAction,
    ContextEvaluation,
    ContextEvaluationRequest,
    ContextEvaluator,
    EvaluationEvent,
    EvaluationObserver,
)
from .scheduler import EvaluationScheduler


class LifecycleController:
    """Run a pluggable evaluator at configured lifecycle boundaries.

    This controller intentionally does not mutate a memory backend. Applying a
    signal requires a compiler, provenance, validation, and rollback policy;
    those belong to the writeback coordinator built on this contract.
    """

    def __init__(
        self,
        evaluator: ContextEvaluator,
        *,
        scheduler: EvaluationScheduler | None = None,
        observers: Iterable[EvaluationObserver] = (),
    ) -> None:
        self.evaluator = evaluator
        self.scheduler = scheduler or EvaluationScheduler()
        self.observers = tuple(observers)

    def evaluate(
        self,
        request: ContextEvaluationRequest,
        *,
        event: EvaluationEvent | None = None,
        force: bool = False,
    ) -> ContextEvaluation | None:
        event = event or EvaluationEvent(
            trigger=request.trigger,
            turn_index=request.turn_index,
            context_tokens=request.context_tokens,
            evaluation_id=request.evaluation_id,
        )
        if not force and not self.scheduler.should_evaluate(event):
            return None
        evaluation = self.evaluator.evaluate(request)
        self._validate(evaluation, request)
        self.scheduler.mark_evaluated(event, evaluation.evaluation_id)
        for observer in self.observers:
            observer.record(evaluation)
        return evaluation

    @staticmethod
    def _validate(evaluation: ContextEvaluation, request: ContextEvaluationRequest) -> None:
        if evaluation.evaluation_id != request.evaluation_id:
            raise ValueError("evaluator returned the wrong evaluation_id")
        expected = {segment.segment_id for segment in request.segments}
        actual = {signal.segment_id for signal in evaluation.signals}
        if actual != expected:
            raise ValueError("evaluation must contain exactly one signal per segment")
        active = set(request.active_segment_ids)
        evicted_active = {
            signal.segment_id
            for signal in evaluation.signals
            if signal.segment_id in active and signal.context_action == ContextAction.EVICT
        }
        if evicted_active:
            raise ValueError(
                "evaluation attempted to evict active segments: "
                + ", ".join(sorted(evicted_active))
            )
