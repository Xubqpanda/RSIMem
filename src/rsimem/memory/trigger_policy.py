"""Deterministic trigger policy and host-event boundary for RSIMem 2B.

The adapter in this module carries only host facts.  Trigger strategy remains
in :class:`DeterministicTriggerPolicy`; consequently shadow observations do
not accidentally invoke extraction or mutate memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .policy_contracts import (
    DecisionAction,
    ExecutionStatus,
    PolicyArtifactIdentity,
    PolicyArtifactKind,
    PolicyLayer,
    TriggerDecision,
    TriggerEvent,
)


SUPPORTED_TRIGGER_TYPES = frozenset(
    {
        "task_completed",
        "session_end",
        "turn_interval",
        "tool_boundary",
        "context_pressure",
        "manual",
    }
)


@dataclass(frozen=True, slots=True)
class TriggerPolicyConfig:
    policy_version: str = "fixed.trigger.parent.v1"
    min_turns_between_runs: int = 0
    task_completed_enabled: bool = True
    shadow_only_types: tuple[str, ...] = (
        "session_end",
        "turn_interval",
        "tool_boundary",
        "context_pressure",
        "manual",
    )

    def __post_init__(self) -> None:
        if not self.policy_version.strip():
            raise ValueError("trigger policy version must not be empty")
        if type(self.min_turns_between_runs) is not int or self.min_turns_between_runs < 0:
            raise ValueError("minimum turns between trigger runs must be non-negative")
        if type(self.task_completed_enabled) is not bool:
            raise ValueError("task_completed_enabled must be bool")
        values = tuple(self.shadow_only_types)
        if len(values) != len(set(values)) or any(value not in SUPPORTED_TRIGGER_TYPES for value in values):
            raise ValueError("shadow trigger types are invalid or duplicated")
        object.__setattr__(self, "shadow_only_types", values)


@dataclass(frozen=True, slots=True)
class TriggerPolicyState:
    last_run_source_revision: str | None = None
    last_run_turn: int | None = None
    seen_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.last_run_turn is not None and (type(self.last_run_turn) is not int or self.last_run_turn < 0):
            raise ValueError("last run turn must be non-negative")
        if len(self.seen_event_ids) != len(set(self.seen_event_ids)):
            raise ValueError("seen trigger event IDs must be unique")


@dataclass(frozen=True, slots=True)
class TriggerObservation:
    event: TriggerEvent
    decision: TriggerDecision
    shadow_only: bool

    def __post_init__(self) -> None:
        if self.decision.trigger_event_id != self.event.event_id:
            raise ValueError("trigger decision is bound to another event")
        if type(self.shadow_only) is not bool:
            raise ValueError("shadow_only must be bool")


class DeterministicTriggerPolicy:
    """Fixed parent trigger policy with explicit shadow-only observations."""

    layer = PolicyLayer.TRIGGER

    def __init__(self, config: TriggerPolicyConfig | None = None) -> None:
        self.config = config or TriggerPolicyConfig()
        self.state = TriggerPolicyState()
        self._observations: list[TriggerObservation] = []

    @property
    def artifact_identity(self) -> PolicyArtifactIdentity:
        return PolicyArtifactIdentity.create(
            policy_version=self.config.policy_version,
            kind=PolicyArtifactKind.FIXED,
            layers=(PolicyLayer.TRIGGER,),
        )

    @property
    def observations(self) -> tuple[TriggerObservation, ...]:
        return tuple(self._observations)

    def decide(self, event: TriggerEvent) -> TriggerObservation:
        if event.event_id in self.state.seen_event_ids:
            decision = self._decision(
                event,
                DecisionAction.SKIP,
                "duplicate_event",
            )
            observation = TriggerObservation(event, decision, event.event_type != "task_completed")
            self._record(observation)
            return observation
        if not event.supported or event.event_type not in SUPPORTED_TRIGGER_TYPES:
            decision = self._decision(event, DecisionAction.SKIP, "unsupported_trigger")
            observation = TriggerObservation(event, decision, True)
            self._record(observation)
            return observation
        if event.event_type in self.config.shadow_only_types:
            decision = self._decision(event, DecisionAction.SKIP, "shadow_only")
            observation = TriggerObservation(event, decision, True)
            self._record(observation)
            return observation
        if event.event_type == "task_completed" and not self.config.task_completed_enabled:
            decision = self._decision(event, DecisionAction.SKIP, "parent_disabled")
            observation = TriggerObservation(event, decision, False)
            self._record(observation)
            return observation
        if event.event_type == "task_completed" and event.metadata.get("task_state") not in {None, "completed", "TaskLifecycleState.COMPLETED"}:
            decision = self._decision(event, DecisionAction.SKIP, "task_not_completed")
            observation = TriggerObservation(event, decision, False)
            self._record(observation)
            return observation
        if event.source_revision == self.state.last_run_source_revision:
            decision = self._decision(event, DecisionAction.SKIP, "duplicate_source_revision")
            observation = TriggerObservation(event, decision, False)
            self._record(observation)
            return observation
        if (
            self.state.last_run_turn is not None
            and event.turn_index is not None
            and event.turn_index - self.state.last_run_turn < self.config.min_turns_between_runs
        ):
            next_boundary = f"turn:{self.state.last_run_turn + self.config.min_turns_between_runs}"
            decision = self._decision(event, DecisionAction.DEFER, "minimum_interval", next_boundary)
            observation = TriggerObservation(event, decision, False)
            self._record(observation)
            return observation
        decision = self._decision(event, DecisionAction.RUN, "task_completed_parent", status=ExecutionStatus.PENDING)
        observation = TriggerObservation(event, decision, False)
        self._record(observation, mark_run=True)
        return observation

    def decide_trigger(self, event: TriggerEvent) -> TriggerDecision:
        """Protocol-friendly shorthand returning only the decision envelope."""

        return self.decide(event).decision

    def _decision(
        self,
        event: TriggerEvent,
        action: DecisionAction,
        reason: str,
        next_boundary: str | None = None,
        *,
        status: ExecutionStatus | None = None,
    ) -> TriggerDecision:
        output = {
            "action": action.value,
            "reason_code": reason,
            "next_eligible_boundary": next_boundary,
        }
        return TriggerDecision.create(
            policy_version=self.config.policy_version,
            source_revision=event.source_revision,
            input_payload={"event_id": event.event_id, "event_digest": event.input_digest},
            output_payload=output,
            action=action,
            execution_status=status or {
                DecisionAction.SKIP: ExecutionStatus.SKIPPED,
                DecisionAction.DEFER: ExecutionStatus.DEFERRED,
            }[action],
            reason_codes=(reason,),
            lineage_id=f"lineage.{event.event_id}",
            trigger_event_id=event.event_id,
            next_eligible_boundary=next_boundary,
        )

    def _record(self, observation: TriggerObservation, *, mark_run: bool = False) -> None:
        self._observations.append(observation)
        seen = self.state.seen_event_ids
        if observation.event.event_id not in seen:
            seen = seen + (observation.event.event_id,)
        self.state = TriggerPolicyState(
            last_run_source_revision=(observation.event.source_revision if mark_run else self.state.last_run_source_revision),
            last_run_turn=(observation.event.turn_index if mark_run and observation.event.turn_index is not None else self.state.last_run_turn),
            seen_event_ids=seen,
        )


class HostTriggerAdapter:
    """Map an observed host event to a typed event without inventing support."""

    def __init__(self, *, adapter_id: str = "host-trigger-adapter.v1") -> None:
        self.adapter_id = adapter_id

    def event(
        self,
        event_type: str,
        *,
        source_revision: str,
        payload: object,
        session_id: str | None = None,
        task_id: str | None = None,
        turn_id: str | None = None,
        turn_index: int | None = None,
        supported: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TriggerEvent:
        normalized = event_type.strip()
        if not normalized:
            raise ValueError("host event type must not be empty")
        actual_supported = normalized in SUPPORTED_TRIGGER_TYPES if supported is None else supported
        metadata_value = dict(metadata or {})
        metadata_value.setdefault("adapter_id", self.adapter_id)
        if normalized not in SUPPORTED_TRIGGER_TYPES:
            metadata_value.setdefault("unsupported_reason", "unknown_host_event")
        return TriggerEvent.create(
            event_type=normalized,
            source_revision=source_revision,
            input_payload=payload,
            session_id=session_id,
            task_id=task_id,
            turn_id=turn_id,
            turn_index=turn_index,
            supported=actual_supported,
            metadata=metadata_value,
        )


class HermesTriggerEventAdapter(HostTriggerAdapter):
    """Project a real RSIMem/Hermes snapshot into a trigger event.

    The adapter reads only stable snapshot identity and lifecycle facts.  It
    never infers an event that the snapshot does not carry: context-pressure
    events without an observed token count are explicitly unsupported.
    """

    def from_snapshot(
        self,
        snapshot: object,
        event_type: str,
        *,
        context_tokens: int | None = None,
        turn_index: int | None = None,
        tool_boundary_observed: bool | None = None,
        manual_authorized: bool = False,
    ) -> TriggerEvent:
        try:
            source_revision = str(snapshot.context_revision)
            snapshot_id = str(snapshot.snapshot_id)
            session_id = str(snapshot.session_id)
            task_id = str(snapshot.task_id)
            task_state = str(snapshot.task_state)
        except AttributeError as exc:
            raise ValueError("Hermes trigger requires a context snapshot") from exc
        if not source_revision.strip() or not snapshot_id.strip():
            raise ValueError("Hermes trigger snapshot identity is incomplete")
        normalized_event = event_type.strip()
        supported = normalized_event in SUPPORTED_TRIGGER_TYPES
        metadata: dict[str, Any] = {
            "snapshot_id": snapshot_id,
            "task_state": task_state,
        }
        if normalized_event == "task_completed":
            supported = task_state in {"completed", "TaskLifecycleState.COMPLETED"}
            if not supported:
                metadata["unsupported_reason"] = "task_not_completed"
        elif normalized_event == "session_end":
            supported = task_state not in {"active", "TaskLifecycleState.ACTIVE"}
            if not supported:
                metadata["unsupported_reason"] = "session_task_still_active"
        elif normalized_event == "context_pressure":
            if context_tokens is None:
                supported = False
                metadata["unsupported_reason"] = "context_tokens_unobserved"
            else:
                metadata["context_tokens"] = context_tokens
        elif normalized_event == "turn_interval":
            supported = turn_index is not None
            if not supported:
                metadata["unsupported_reason"] = "turn_index_unobserved"
        elif normalized_event == "tool_boundary":
            supported = tool_boundary_observed is True
            if not supported:
                metadata["unsupported_reason"] = "tool_boundary_unobserved"
        elif normalized_event == "manual":
            supported = manual_authorized is True
            if not supported:
                metadata["unsupported_reason"] = "manual_authorization_unobserved"
        return self.event(
            normalized_event,
            source_revision=source_revision,
            payload={
                "snapshot_id": snapshot_id,
                "context_revision": source_revision,
                "task_state": task_state,
                "context_tokens": context_tokens,
            },
            session_id=session_id,
            task_id=task_id,
            turn_id=getattr(snapshot, "current_turn_id", None),
            turn_index=turn_index,
            supported=supported,
            metadata=metadata,
        )


__all__ = [
    "SUPPORTED_TRIGGER_TYPES",
    "TriggerPolicyConfig",
    "TriggerPolicyState",
    "TriggerObservation",
    "DeterministicTriggerPolicy",
    "HostTriggerAdapter",
    "HermesTriggerEventAdapter",
]
