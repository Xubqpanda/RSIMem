"""Content-free exact joins for tool call/result process evidence."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from .evidence_planes import EvidencePlane, EvidenceSourceKind, validate_plane_source
from .process_feedback import ProcessEvent, ProcessEventKind, ProcessEventStatus


TOOL_EXACT_JOIN_SCHEMA_VERSION = 1
TOOL_EXACT_JOIN_SCHEMA = "rsimem-tool-call-result-join-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,255}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _id(value: object, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")


def _digest_field(value: object, name: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be sha256")


class ToolJoinResolutionStatus(StrEnum):
    COMPLETE = "complete"
    MISSING = "missing"
    DUPLICATE = "duplicate"
    ORPHANED = "orphaned"
    CROSS_TASK = "cross_task"
    TYPE_MISMATCH = "type_mismatch"
    CENSORED = "censored"


@dataclass(frozen=True, slots=True)
class ToolCallResultJoin:
    """A closure between one tool call and exactly one result.

    The contract carries no arguments or return content.  Those remain in an
    owner-controlled trace; this object is safe to project into the public
    process corpus.
    """

    join_id: str
    call_id: str | None
    result_id: str | None
    tool_name_digest: str
    success: bool | None
    retry_identity: str
    run_id: str
    variant: str
    trace_id: str
    episode_id: str
    session_id: str
    task_id: str
    source_revision: str
    host_event_id: str
    policy_lineage_id: str | None = None
    memory_use_operation_id: str | None = None
    call_receipt_id: str | None = None
    result_receipt_id: str | None = None
    call_present: bool = True
    result_present: bool = True
    duplicate_call: bool = False
    duplicate_result: bool = False
    orphan_result: bool = False
    cross_task: bool = False
    type_mismatch: bool = False
    observation_complete: bool = True
    schema_version: int = TOOL_EXACT_JOIN_SCHEMA_VERSION
    evidence_plane: EvidencePlane = EvidencePlane.PURE_PROCESS
    evidence_source: EvidenceSourceKind = EvidenceSourceKind.RUNTIME_OBSERVATION

    def __post_init__(self) -> None:
        if self.schema_version != TOOL_EXACT_JOIN_SCHEMA_VERSION:
            raise ValueError("unsupported tool exact-join schema")
        plane, source = validate_plane_source(self.evidence_plane, self.evidence_source)
        if plane != EvidencePlane.PURE_PROCESS or source != EvidenceSourceKind.RUNTIME_OBSERVATION:
            raise ValueError("tool exact join must be pure_process runtime evidence")
        object.__setattr__(self, "evidence_plane", plane)
        object.__setattr__(self, "evidence_source", source)
        _id(self.join_id, "tool join ID")
        for value, name in (
            (self.call_id, "tool call ID"),
            (self.result_id, "tool result ID"),
            (self.retry_identity, "tool retry identity"),
            (self.run_id, "run ID"),
            (self.variant, "variant"),
            (self.trace_id, "trace ID"),
            (self.episode_id, "episode ID"),
            (self.session_id, "session ID"),
            (self.task_id, "task ID"),
            (self.source_revision, "source revision"),
            (self.host_event_id, "host event ID"),
            (self.policy_lineage_id, "policy lineage ID"),
            (self.memory_use_operation_id, "memory-use operation ID"),
            (self.call_receipt_id, "call receipt ID"),
            (self.result_receipt_id, "result receipt ID"),
        ):
            if value is not None:
                _id(value, name)
        _digest_field(self.tool_name_digest, "tool name digest")
        if self.success is not None and type(self.success) is not bool:
            raise TypeError("tool result success must be bool or None")
        for value, name in (
            (self.call_present, "call presence"),
            (self.result_present, "result presence"),
            (self.duplicate_call, "duplicate call"),
            (self.duplicate_result, "duplicate result"),
            (self.orphan_result, "orphan result"),
            (self.cross_task, "cross-task flag"),
            (self.type_mismatch, "tool type mismatch"),
            (self.observation_complete, "observation completeness"),
        ):
            if type(value) is not bool:
                raise TypeError(f"{name} must be bool")
        if self.call_present and self.call_id is None:
            raise ValueError("present tool call requires call ID")
        if not self.call_present and self.call_id is not None and not self.orphan_result:
            raise ValueError("absent tool call cannot carry a call ID")
        if self.result_present and self.result_id is None:
            raise ValueError("present tool result requires result ID")
        if not self.result_present and self.result_id is not None:
            raise ValueError("absent tool result cannot carry a result ID")
        identity = self._identity_payload()
        if self.join_id != f"tool-join.{_digest(identity)[:40]}":
            raise ValueError("tool join ID mismatch")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "call_id": self.call_id,
            "result_id": self.result_id,
            "tool_name_digest": self.tool_name_digest,
            "success": self.success,
            "retry_identity": self.retry_identity,
            "run_id": self.run_id,
            "variant": self.variant,
            "trace_id": self.trace_id,
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "source_revision": self.source_revision,
            "host_event_id": self.host_event_id,
            "policy_lineage_id": self.policy_lineage_id,
            "memory_use_operation_id": self.memory_use_operation_id,
            "call_receipt_id": self.call_receipt_id,
            "result_receipt_id": self.result_receipt_id,
            "call_present": self.call_present,
            "result_present": self.result_present,
            "duplicate_call": self.duplicate_call,
            "duplicate_result": self.duplicate_result,
            "orphan_result": self.orphan_result,
            "cross_task": self.cross_task,
            "type_mismatch": self.type_mismatch,
            "observation_complete": self.observation_complete,
            "evidence_plane": self.evidence_plane.value,
            "evidence_source": self.evidence_source.value,
        }

    @classmethod
    def create(
        cls,
        *,
        call_id: str | None,
        result_id: str | None,
        tool_name_digest: str,
        success: bool | None,
        retry_identity: str,
        run_id: str,
        variant: str,
        trace_id: str,
        episode_id: str,
        session_id: str,
        task_id: str,
        source_revision: str,
        host_event_id: str,
        policy_lineage_id: str | None = None,
        memory_use_operation_id: str | None = None,
        call_receipt_id: str | None = None,
        result_receipt_id: str | None = None,
        call_present: bool = True,
        result_present: bool = True,
        duplicate_call: bool = False,
        duplicate_result: bool = False,
        orphan_result: bool = False,
        cross_task: bool = False,
        type_mismatch: bool = False,
        observation_complete: bool = True,
    ) -> "ToolCallResultJoin":
        values: dict[str, object] = {
            "call_id": call_id,
            "result_id": result_id,
            "tool_name_digest": tool_name_digest,
            "success": success,
            "retry_identity": retry_identity,
            "run_id": run_id,
            "variant": variant,
            "trace_id": trace_id,
            "episode_id": episode_id,
            "session_id": session_id,
            "task_id": task_id,
            "source_revision": source_revision,
            "host_event_id": host_event_id,
            "policy_lineage_id": policy_lineage_id,
            "memory_use_operation_id": memory_use_operation_id,
            "call_receipt_id": call_receipt_id,
            "result_receipt_id": result_receipt_id,
            "call_present": call_present,
            "result_present": result_present,
            "duplicate_call": duplicate_call,
            "duplicate_result": duplicate_result,
            "orphan_result": orphan_result,
            "cross_task": cross_task,
            "type_mismatch": type_mismatch,
            "observation_complete": observation_complete,
            "schema_version": TOOL_EXACT_JOIN_SCHEMA_VERSION,
            "evidence_plane": EvidencePlane.PURE_PROCESS,
            "evidence_source": EvidenceSourceKind.RUNTIME_OBSERVATION,
        }
        return cls(join_id=f"tool-join.{_digest(values)[:40]}", **values)

    def payload(self) -> dict[str, object]:
        return {"schema": TOOL_EXACT_JOIN_SCHEMA, "join_id": self.join_id, **self._identity_payload()}

    @classmethod
    def from_payload(cls, value: object) -> "ToolCallResultJoin":
        expected = {
            "schema", "join_id", "schema_version", "call_id", "result_id", "tool_name_digest",
            "success", "retry_identity", "run_id", "variant", "trace_id", "episode_id",
            "session_id", "task_id", "source_revision", "host_event_id", "policy_lineage_id",
            "memory_use_operation_id", "call_receipt_id", "result_receipt_id", "call_present",
            "result_present", "duplicate_call", "duplicate_result", "orphan_result", "cross_task",
            "type_mismatch", "observation_complete", "evidence_plane", "evidence_source",
        }
        if not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != TOOL_EXACT_JOIN_SCHEMA:
            raise ValueError("malformed tool exact join")
        try:
            return cls(
                join_id=value["join_id"], call_id=value["call_id"], result_id=value["result_id"],
                tool_name_digest=value["tool_name_digest"], success=value["success"],
                retry_identity=value["retry_identity"], run_id=value["run_id"], variant=value["variant"],
                trace_id=value["trace_id"], episode_id=value["episode_id"], session_id=value["session_id"],
                task_id=value["task_id"], source_revision=value["source_revision"],
                host_event_id=value["host_event_id"], policy_lineage_id=value["policy_lineage_id"],
                memory_use_operation_id=value["memory_use_operation_id"], call_receipt_id=value["call_receipt_id"],
                result_receipt_id=value["result_receipt_id"], call_present=value["call_present"],
                result_present=value["result_present"], duplicate_call=value["duplicate_call"],
                duplicate_result=value["duplicate_result"], orphan_result=value["orphan_result"],
                cross_task=value["cross_task"], type_mismatch=value["type_mismatch"],
                observation_complete=value["observation_complete"], schema_version=value["schema_version"],
                evidence_plane=value["evidence_plane"], evidence_source=value["evidence_source"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed tool exact join") from exc

    def process_events(
        self,
        *,
        family_id: str | None = None,
        stage: str | None = None,
    ) -> tuple[ProcessEvent, ...]:
        """Project the call and result independently into process events.

        The join remains host-neutral and does not carry benchmark scope.
        Callers supply scope at projection time so process events can still
        participate in the cross-ledger identity audit.
        """

        events: list[ProcessEvent] = []
        call = None
        if self.call_present and self.call_id is not None:
            call = ProcessEvent.create(
            kind=ProcessEventKind.TOOL_CALL,
            status=ProcessEventStatus.EXECUTED,
            run_id=self.run_id,
            variant=self.variant,
            trace_id=self.trace_id,
            episode_id=self.episode_id,
            session_id=self.session_id,
            task_id=self.task_id,
            host_event_id=self.host_event_id,
            source_revision=self.source_revision,
            input_payload={
                "call_id": self.call_id,
                "tool_name_digest": self.tool_name_digest,
                "retry_identity": self.retry_identity,
                "memory_use_operation_id": self.memory_use_operation_id,
            },
            output_payload={"receipt_id": self.call_receipt_id},
            lineage_id=self.policy_lineage_id,
            execution_receipt_ids=((self.call_receipt_id,) if self.call_receipt_id else ()),
            reason_codes=("decision_observed",),
                tool_call_id=self.call_id,
                tool_name_digest=self.tool_name_digest,
                retry_identity=self.retry_identity,
                family_id=family_id,
                stage=stage,
            )
            events.append(call)
        if not self.result_present or self.result_id is None:
            return tuple(events)
        result_call_id = self.call_id or (
            "orphan-call."
            + hashlib.sha256(self.result_id.encode("utf-8")).hexdigest()[:24]
        )
        result = ProcessEvent.create(
            kind=ProcessEventKind.TOOL_RESULT,
            status=(
                ProcessEventStatus.SUCCESS
                if self.success is True
                else ProcessEventStatus.FAILED
                if self.success is False
                else ProcessEventStatus.UNKNOWN
            ),
            run_id=self.run_id,
            variant=self.variant,
            trace_id=self.trace_id,
            episode_id=self.episode_id,
            session_id=self.session_id,
            task_id=self.task_id,
            host_event_id=self.host_event_id,
            source_revision=self.source_revision,
            input_payload={
                "result_id": self.result_id,
                "call_id": self.call_id,
                "tool_name_digest": self.tool_name_digest,
                "retry_identity": self.retry_identity,
            },
            output_payload={"success": self.success, "receipt_id": self.result_receipt_id},
            lineage_id=self.policy_lineage_id,
            execution_receipt_ids=((self.result_receipt_id,) if self.result_receipt_id else ()),
            reason_codes=(
                ("schema_failure",)
                if self.type_mismatch
                else
                ("decision_observed",)
                if self.success is True
                else ("tool_failure",)
                if self.success is False
                else ("decision_observed",)
            ),
            tool_call_id=result_call_id,
            tool_result_id=self.result_id,
            tool_name_digest=self.tool_name_digest,
            retry_identity=self.retry_identity,
            tool_success=self.success,
            family_id=family_id,
            stage=stage,
        )
        events.append(result)
        return tuple(events)


@dataclass(frozen=True, slots=True)
class ToolJoinResolution:
    join_id: str
    status: ToolJoinResolutionStatus
    reason_code: str
    exact: bool
    observation_complete: bool

    def __post_init__(self) -> None:
        _id(self.join_id, "tool join ID")
        object.__setattr__(self, "status", ToolJoinResolutionStatus(self.status))
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.reason_code):
            raise ValueError("tool join reason is invalid")
        if type(self.exact) is not bool or type(self.observation_complete) is not bool:
            raise TypeError("tool join resolution flags must be bool")

    def payload(self) -> dict[str, object]:
        return {
            "join_id": self.join_id,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "exact": self.exact,
            "observation_complete": self.observation_complete,
        }


def resolve_tool_call_result(join: ToolCallResultJoin) -> ToolJoinResolution:
    if not isinstance(join, ToolCallResultJoin):
        raise TypeError("tool join resolver requires ToolCallResultJoin")
    if not join.observation_complete:
        return ToolJoinResolution(join.join_id, ToolJoinResolutionStatus.CENSORED, "observation_censored", False, False)
    checks = (
        (join.duplicate_call or join.duplicate_result, ToolJoinResolutionStatus.DUPLICATE, "duplicate_tool_identity"),
        (join.cross_task, ToolJoinResolutionStatus.CROSS_TASK, "cross_task_join"),
        (join.type_mismatch, ToolJoinResolutionStatus.TYPE_MISMATCH, "tool_type_mismatch"),
        (join.orphan_result, ToolJoinResolutionStatus.ORPHANED, "orphan_result"),
        (not join.call_present or not join.result_present, ToolJoinResolutionStatus.MISSING, "missing_call_or_result"),
    )
    for condition, status, reason in checks:
        if condition:
            return ToolJoinResolution(join.join_id, status, reason, False, True)
    if join.call_id is None or join.result_id is None or join.call_receipt_id is None or join.result_receipt_id is None:
        return ToolJoinResolution(join.join_id, ToolJoinResolutionStatus.MISSING, "missing_receipt_identity", False, True)
    return ToolJoinResolution(join.join_id, ToolJoinResolutionStatus.COMPLETE, "exact_call_result_join", True, True)


__all__ = [
    "TOOL_EXACT_JOIN_SCHEMA",
    "TOOL_EXACT_JOIN_SCHEMA_VERSION",
    "ToolCallResultJoin",
    "ToolJoinResolution",
    "ToolJoinResolutionStatus",
    "resolve_tool_call_result",
]
