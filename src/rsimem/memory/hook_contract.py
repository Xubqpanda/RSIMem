"""Stable, content-free lifecycle hook evidence contract."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping


MEMORY_HOOK_SCHEMA = "rsimem-memory-hook-event-v1"
MEMORY_HOOK_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,255}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class MemoryHookPoint(StrEnum):
    FORMATION = "formation"
    PERSISTENCE = "persistence"
    RETRIEVAL = "retrieval"
    EXPOSURE = "exposure"
    USE = "use"
    OUTCOME = "outcome"


class MemoryHookDecision(StrEnum):
    ALLOWED = "allowed"
    DENIED = "denied"
    NOT_ATTEMPTED = "not_attempted"
    FAILED = "failed"
    OBSERVED = "observed"


class MemoryHookType(StrEnum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    PROFILE = "profile"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _id(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")
    return value


@dataclass(frozen=True, slots=True)
class MemoryHookEvent:
    """One hook observation joined to one run/episode execution identity."""

    event_id: str
    run_id: str
    episode_id: str
    session_id: str
    task_id: str
    memory_type: MemoryHookType
    hook_point: MemoryHookPoint
    operation: str
    decision: MemoryHookDecision
    result_code: str
    provenance_digest: str
    surface_policy_id: str
    failure_code: str | None = None
    parent_event_ids: tuple[str, ...] = ()
    schema_version: int = MEMORY_HOOK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MEMORY_HOOK_SCHEMA_VERSION:
            raise ValueError("unsupported Memory hook schema")
        for value, name in (
            (self.run_id, "run ID"), (self.episode_id, "episode ID"),
            (self.session_id, "session ID"), (self.task_id, "task ID"),
            (self.operation, "operation"), (self.result_code, "result code"),
            (self.surface_policy_id, "surface policy ID"),
        ):
            _id(value, name)
        _id(self.event_id, "event ID")
        object.__setattr__(self, "memory_type", MemoryHookType(self.memory_type))
        object.__setattr__(self, "hook_point", MemoryHookPoint(self.hook_point))
        object.__setattr__(self, "decision", MemoryHookDecision(self.decision))
        if _DIGEST.fullmatch(self.provenance_digest) is None:
            raise ValueError("hook provenance digest must be sha256")
        if self.failure_code is not None:
            _id(self.failure_code, "failure code")
        object.__setattr__(self, "parent_event_ids", tuple(self.parent_event_ids))
        if len(set(self.parent_event_ids)) != len(self.parent_event_ids):
            raise ValueError("parent hook event IDs must be unique")
        for value in self.parent_event_ids:
            _id(value, "parent hook event ID")
        if self.event_id != f"memory-hook.{_digest(self.identity_payload())[:40]}":
            raise ValueError("hook event ID does not match its identity")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "memory_type": self.memory_type.value,
            "hook_point": self.hook_point.value,
            "operation": self.operation,
            "decision": self.decision.value,
            "result_code": self.result_code,
            "provenance_digest": self.provenance_digest,
            "surface_policy_id": self.surface_policy_id,
            "failure_code": self.failure_code,
            "parent_event_ids": list(self.parent_event_ids),
        }

    def payload(self) -> dict[str, object]:
        return {"schema": MEMORY_HOOK_SCHEMA, "event_id": self.event_id, **self.identity_payload()}

    @classmethod
    def from_process_event(
        cls,
        process_event: "ProcessEvent",
        *,
        memory_type: MemoryHookType | str,
        hook_point: MemoryHookPoint | str,
        operation: str,
        result_code: str,
        surface_policy_id: str,
        failure_code: str | None = None,
        parent_event_ids: tuple[str, ...] = (),
    ) -> "MemoryHookEvent":
        """Project process evidence into a content-free memory lifecycle hook."""
        from .process_feedback import ProcessEvent

        if not isinstance(process_event, ProcessEvent):
            raise TypeError("process_event must be a ProcessEvent")
        return cls.create(
            run_id=process_event.run_id,
            episode_id=process_event.episode_id,
            session_id=process_event.session_id,
            task_id=process_event.task_id,
            memory_type=memory_type,
            hook_point=hook_point,
            operation=operation,
            decision=_decision_for_process_status(process_event.status),
            result_code=result_code,
            provenance_digest=_process_event_digest(process_event),
            surface_policy_id=surface_policy_id,
            failure_code=failure_code,
            parent_event_ids=(process_event.event_id, *parent_event_ids),
        )

    def to_process_event(
        self,
        *,
        variant: str,
        trace_id: str,
        host_event_id: str,
        source_revision: str,
        family_id: str | None = None,
        stage: str | None = None,
    ) -> "ProcessEvent":
        """Project this hook into pure process evidence without exposing content."""
        from .process_feedback import ProcessEvent, ProcessEventKind, ProcessEventStatus

        return ProcessEvent.create(
            kind=_process_kind_for_hook_point(self.hook_point),
            status=_process_status_for_decision(self.decision),
            run_id=self.run_id,
            variant=variant,
            trace_id=trace_id,
            episode_id=self.episode_id,
            session_id=self.session_id,
            task_id=self.task_id,
            host_event_id=host_event_id,
            source_revision=source_revision,
            input_payload={
                "memory_hook_event_id": self.event_id,
                "memory_type": self.memory_type.value,
                "hook_point": self.hook_point.value,
                "operation": self.operation,
            },
            output_payload={
                "decision": self.decision.value,
                "result_code": self.result_code,
                "failure_code": self.failure_code,
            },
            execution_receipt_ids=(self.event_id, *self.parent_event_ids),
            reason_codes=("adapter_failure",) if self.decision is MemoryHookDecision.FAILED else ("decision_observed",),
            family_id=family_id,
            stage=stage,
        )

    @classmethod
    def create(cls, **values: object) -> "MemoryHookEvent":
        identity = dict(values)
        identity.setdefault("schema_version", MEMORY_HOOK_SCHEMA_VERSION)
        identity["memory_type"] = MemoryHookType(identity["memory_type"]).value
        identity["hook_point"] = MemoryHookPoint(identity["hook_point"]).value
        identity["decision"] = MemoryHookDecision(identity["decision"]).value
        identity["parent_event_ids"] = list(identity.get("parent_event_ids", ()))
        identity["failure_code"] = identity.get("failure_code")
        return cls(event_id=f"memory-hook.{_digest(identity)[:40]}", **values)


def _process_event_digest(process_event: "ProcessEvent") -> str:
    return _digest(process_event.payload())


def _decision_for_process_status(status: object) -> MemoryHookDecision:
    from .process_feedback import ProcessEventStatus

    value = ProcessEventStatus(status)
    if value is ProcessEventStatus.FAILED:
        return MemoryHookDecision.FAILED
    if value is ProcessEventStatus.REJECTED:
        return MemoryHookDecision.DENIED
    if value is ProcessEventStatus.SKIPPED:
        return MemoryHookDecision.NOT_ATTEMPTED
    return MemoryHookDecision.OBSERVED


def _process_kind_for_hook_point(hook_point: MemoryHookPoint):
    from .process_feedback import ProcessEventKind

    return {
        MemoryHookPoint.FORMATION: ProcessEventKind.EXTRACTION,
        MemoryHookPoint.PERSISTENCE: ProcessEventKind.COMMIT,
        MemoryHookPoint.RETRIEVAL: ProcessEventKind.RETRIEVAL,
        MemoryHookPoint.EXPOSURE: ProcessEventKind.EXPOSURE,
        MemoryHookPoint.USE: ProcessEventKind.TOOL_RESULT,
        MemoryHookPoint.OUTCOME: ProcessEventKind.TASK_OUTCOME,
    }[hook_point]


def _process_status_for_decision(decision: MemoryHookDecision):
    from .process_feedback import ProcessEventStatus

    return {
        MemoryHookDecision.ALLOWED: ProcessEventStatus.EXECUTED,
        MemoryHookDecision.OBSERVED: ProcessEventStatus.EXECUTED,
        MemoryHookDecision.DENIED: ProcessEventStatus.REJECTED,
        MemoryHookDecision.NOT_ATTEMPTED: ProcessEventStatus.SKIPPED,
        MemoryHookDecision.FAILED: ProcessEventStatus.FAILED,
    }[decision]


__all__ = [
    "MemoryHookDecision", "MemoryHookEvent", "MemoryHookPoint", "MemoryHookType",
    "MEMORY_HOOK_SCHEMA", "MEMORY_HOOK_SCHEMA_VERSION",
]
