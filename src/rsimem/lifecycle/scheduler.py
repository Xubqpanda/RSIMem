"""Cadence decisions for context evaluation."""

from __future__ import annotations

from .contracts import EvaluationCadence, EvaluationEvent, EvaluationTrigger, SchedulerState


class EvaluationScheduler:
    """Decide when an evaluator may run, without knowing how it evaluates."""

    def __init__(self, cadence: EvaluationCadence | None = None) -> None:
        self.cadence = cadence or EvaluationCadence()
        self.state = SchedulerState()

    def should_evaluate(self, event: EvaluationEvent) -> bool:
        if (
            self.state.last_evaluated_turn is not None
            and event.turn_index - self.state.last_evaluated_turn
            < self.cadence.min_turns_between_evaluations
        ):
            return False
        if event.trigger == EvaluationTrigger.TASK_COMPLETED:
            return self.cadence.on_task_completed
        if event.trigger == EvaluationTrigger.SESSION_END:
            return self.cadence.on_session_end
        if event.trigger == EvaluationTrigger.TOOL_BOUNDARY:
            return self.cadence.on_tool_boundary
        if event.trigger == EvaluationTrigger.CONTEXT_PRESSURE:
            return (
                self.cadence.on_context_pressure
                and self.cadence.context_pressure_tokens is not None
                and event.context_tokens is not None
                and event.context_tokens >= self.cadence.context_pressure_tokens
            )
        if event.trigger == EvaluationTrigger.TURN_INTERVAL:
            return (
                self.cadence.every_n_turns is not None
                and event.turn_index > 0
                and event.turn_index % self.cadence.every_n_turns == 0
            )
        return event.trigger == EvaluationTrigger.MANUAL

    def mark_evaluated(self, event: EvaluationEvent, evaluation_id: str) -> None:
        self.state = SchedulerState(
            last_evaluated_turn=event.turn_index,
            last_evaluation_id=evaluation_id,
        )
