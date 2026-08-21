"""Contracts for evaluating context at lifecycle boundaries.

The lifecycle layer deliberately knows neither the host protocol nor a memory
backend. It turns a context snapshot into typed, auditable decisions that a
later writeback coordinator can apply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

from ..memory.contracts import MemoryKind


class EvaluationTrigger(StrEnum):
    TASK_COMPLETED = "task_completed"
    SESSION_END = "session_end"
    CONTEXT_PRESSURE = "context_pressure"
    TURN_INTERVAL = "turn_interval"
    TOOL_BOUNDARY = "tool_boundary"
    MANUAL = "manual"


class ContextAction(StrEnum):
    RETAIN = "retain"
    EVICT = "evict"


class WritebackAction(StrEnum):
    DEFER = "defer"
    DISCARD = "discard"
    ADD = "add"
    UPDATE = "update"


def _frozen_metadata(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True, slots=True)
class ContextSegment:
    """A candidate unit that can leave the active context."""

    segment_id: str
    role: str
    content: str
    token_count: int | None = None
    completed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.segment_id.strip() or not self.role.strip():
            raise ValueError("segment_id and role must not be empty")
        if not self.content.strip():
            raise ValueError("segment content must not be empty")
        if self.token_count is not None and self.token_count < 0:
            raise ValueError("token_count must not be negative")
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ContextEvaluationRequest:
    """Immutable input supplied to one context evaluation."""

    evaluation_id: str
    session_id: str
    segments: tuple[ContextSegment, ...]
    trigger: EvaluationTrigger
    task_id: str | None = None
    turn_index: int = 0
    context_revision: str = ""
    active_segment_ids: tuple[str, ...] = ()
    context_tokens: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evaluation_id.strip() or not self.session_id.strip():
            raise ValueError("evaluation_id and session_id must not be empty")
        if not self.segments:
            raise ValueError("context evaluation requires at least one segment")
        if self.turn_index < 0:
            raise ValueError("turn_index must not be negative")
        if self.context_tokens is not None and self.context_tokens < 0:
            raise ValueError("context_tokens must not be negative")
        ids = [segment.segment_id for segment in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("context segment IDs must be unique")
        if not set(self.active_segment_ids).issubset(ids):
            raise ValueError("active segments must be present in the snapshot")
        object.__setattr__(self, "trigger", EvaluationTrigger(self.trigger))
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class EvaluationSignal:
    """One joint context and memory decision for a context segment."""

    segment_id: str
    context_action: ContextAction
    writeback_action: WritebackAction
    utility_estimate: float
    confidence: float
    memory_kind: MemoryKind | None = None
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.segment_id.strip():
            raise ValueError("signal segment_id must not be empty")
        if not 0.0 <= self.utility_estimate <= 1.0:
            raise ValueError("utility_estimate must be between 0 and 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "context_action", ContextAction(self.context_action))
        object.__setattr__(self, "writeback_action", WritebackAction(self.writeback_action))
        if self.memory_kind is not None:
            object.__setattr__(self, "memory_kind", MemoryKind(self.memory_kind))
        if self.writeback_action in {WritebackAction.ADD, WritebackAction.UPDATE}:
            if self.memory_kind is None:
                raise ValueError("memory_kind is required for add/update")
        if self.context_action == ContextAction.RETAIN and self.writeback_action == WritebackAction.DISCARD:
            raise ValueError("retained context cannot be marked for discard")


@dataclass(frozen=True, slots=True)
class ContextEvaluation:
    """Validated output of one evaluator invocation."""

    evaluation_id: str
    evaluator: str
    trigger: EvaluationTrigger
    signals: tuple[EvaluationSignal, ...]
    policy_version: str = "initial"
    input_chars: int = 0

    def __post_init__(self) -> None:
        if not self.evaluation_id.strip() or not self.evaluator.strip():
            raise ValueError("evaluation_id and evaluator must not be empty")
        if self.input_chars < 0:
            raise ValueError("input_chars must not be negative")
        ids = [signal.segment_id for signal in self.signals]
        if len(ids) != len(set(ids)):
            raise ValueError("evaluation signal IDs must be unique")
        object.__setattr__(self, "trigger", EvaluationTrigger(self.trigger))


@runtime_checkable
class ContextEvaluator(Protocol):
    """Pluggable evaluator, implemented by an LLM or a small local model."""

    @property
    def name(self) -> str: ...

    def evaluate(self, request: ContextEvaluationRequest) -> ContextEvaluation: ...


@runtime_checkable
class EvaluationObserver(Protocol):
    """Content-free observer for later ledger integration."""

    def record(self, evaluation: ContextEvaluation) -> None: ...


@dataclass(frozen=True, slots=True)
class EvaluationCadence:
    """Events that are allowed to trigger an evaluation."""

    on_task_completed: bool = True
    on_session_end: bool = True
    on_context_pressure: bool = True
    on_tool_boundary: bool = False
    context_pressure_tokens: int | None = None
    every_n_turns: int | None = None
    min_turns_between_evaluations: int = 1

    def __post_init__(self) -> None:
        if self.context_pressure_tokens is not None and self.context_pressure_tokens < 1:
            raise ValueError("context_pressure_tokens must be positive")
        if self.every_n_turns is not None and self.every_n_turns < 1:
            raise ValueError("every_n_turns must be positive")
        if self.min_turns_between_evaluations < 0:
            raise ValueError("min_turns_between_evaluations must not be negative")


@dataclass(frozen=True, slots=True)
class SchedulerState:
    last_evaluated_turn: int | None = None
    last_evaluation_id: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationEvent:
    """A host lifecycle event presented to the cadence policy."""

    trigger: EvaluationTrigger
    turn_index: int
    context_tokens: int | None = None
    evaluation_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "trigger", EvaluationTrigger(self.trigger))
        if self.turn_index < 0:
            raise ValueError("turn_index must not be negative")
        if self.context_tokens is not None and self.context_tokens < 0:
            raise ValueError("context_tokens must not be negative")
