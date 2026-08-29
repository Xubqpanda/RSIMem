"""Host-neutral process feedback events and a crash-safe evidence ledger.

Process feedback is deliberately weaker than an application quality label.  It
records what the runtime observed (a trigger, a retrieval miss, a tool failure,
etc.) and binds that observation to the host event, policy decision, source
revision and execution receipts.  The payload is content-free: only digests and
stable identifiers cross the audit boundary.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .policy_contracts import PolicyDecision, PolicyLayer, content_digest


PROCESS_FEEDBACK_SCHEMA_VERSION = 2
PROCESS_FEEDBACK_SCHEMA = "rsimem-process-feedback-v2"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,255}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REASON = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _id(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be sha256")
    return value


def _strings(
    values: Sequence[str],
    name: str,
    *,
    allow_empty: bool = False,
    pattern: re.Pattern[str] = _REASON,
) -> tuple[str, ...]:
    result = tuple(values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    for value in result:
        if not isinstance(value, str) or pattern.fullmatch(value) is None:
            raise ValueError(f"{name} contains an invalid identifier")
    return result


class ProcessEventKind(StrEnum):
    HOST_LIFECYCLE = "host_lifecycle"
    TRIGGER = "trigger"
    SOURCE_SELECTION = "source_selection"
    EXTRACTION = "extraction"
    ADMISSION = "admission"
    COMMIT = "commit"
    RETRIEVAL = "retrieval"
    EXPOSURE = "exposure"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TASK_OUTCOME = "task_outcome"
    RECOVERY = "recovery"


class ProcessEventStatus(StrEnum):
    PENDING = "pending"
    EXECUTED = "executed"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    DEFERRED = "deferred"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


# Keep these names stable.  They intentionally distinguish stages that are
# often incorrectly collapsed into an ``extraction_failed`` label.
PROCESS_REASON_CODES = frozenset({
    "decision_observed",
    "absence",
    "non_use",
    "tool_failure",
    "retrieval_miss",
    "retrieval_failure",
    "injection_failure",
    "task_failure",
    "task_completed",
    "schema_failure",
    "mutation_rejected",
    "revision_conflict",
    "rollback",
    "recovery_failure",
    "unsupported_boundary",
    "observation_censored",
    "adapter_failure",
})

# Policy ledgers retain the full layer-specific reason.  This bounded mapping
# keeps the process corpus vocabulary stable while preserving the distinction
# between no eligible work and an unsupported host boundary.
PROCESS_POLICY_REASON_MAP = {
    "trigger_not_run": "absence",
    "no_eligible_source": "absence",
    "task_not_completed": "absence",
    "duplicate_event": "absence",
    "duplicate_source_revision": "absence",
    "shadow_only": "unsupported_boundary",
    "unsupported_trigger": "unsupported_boundary",
    "parent_disabled": "unsupported_boundary",
}


def _layer_kind(layer: PolicyLayer) -> ProcessEventKind:
    return {
        PolicyLayer.TRIGGER: ProcessEventKind.TRIGGER,
        PolicyLayer.SOURCE_SELECTION: ProcessEventKind.SOURCE_SELECTION,
        PolicyLayer.EXTRACTION: ProcessEventKind.EXTRACTION,
        PolicyLayer.ADMISSION: ProcessEventKind.ADMISSION,
        PolicyLayer.COMMIT: ProcessEventKind.COMMIT,
        PolicyLayer.EXPOSURE: ProcessEventKind.EXPOSURE,
    }[PolicyLayer(layer)]


@dataclass(frozen=True, slots=True)
class ProcessEvent:
    """One content-free observation in the runtime process corpus."""

    event_id: str
    kind: ProcessEventKind
    status: ProcessEventStatus
    run_id: str
    variant: str
    trace_id: str
    episode_id: str
    session_id: str
    task_id: str
    host_event_id: str
    source_revision: str
    input_digest: str
    output_digest: str
    policy_decision_id: str | None = None
    policy_layer: PolicyLayer | None = None
    lineage_id: str | None = None
    execution_receipt_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ("decision_observed",)
    tool_call_id: str | None = None
    tool_result_id: str | None = None
    tool_name_digest: str | None = None
    retry_identity: str | None = None
    tool_success: bool | None = None
    family_id: str | None = None
    stage: str | None = None
    schema_version: int = PROCESS_FEEDBACK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROCESS_FEEDBACK_SCHEMA_VERSION:
            raise ValueError("unsupported process feedback schema version")
        for value, name in (
            (self.event_id, "process event ID"),
            (self.run_id, "run ID"),
            (self.variant, "variant"),
            (self.trace_id, "trace ID"),
            (self.episode_id, "episode ID"),
            (self.session_id, "session ID"),
            (self.task_id, "task ID"),
            (self.host_event_id, "host event ID"),
            (self.source_revision, "source revision"),
        ):
            _id(value, name)
        object.__setattr__(self, "kind", ProcessEventKind(self.kind))
        object.__setattr__(self, "status", ProcessEventStatus(self.status))
        _digest(self.input_digest, "process event input digest")
        _digest(self.output_digest, "process event output digest")
        if self.policy_decision_id is not None:
            _id(self.policy_decision_id, "policy decision ID")
        if self.policy_layer is not None:
            object.__setattr__(self, "policy_layer", PolicyLayer(self.policy_layer))
        if self.lineage_id is not None:
            _id(self.lineage_id, "lineage ID")
        for value, name in (
            (self.tool_call_id, "tool call ID"),
            (self.tool_result_id, "tool result ID"),
            (self.retry_identity, "tool retry identity"),
        ):
            if value is not None:
                _id(value, name)
        if self.tool_name_digest is not None:
            _digest(self.tool_name_digest, "tool name digest")
        if self.tool_success is not None and type(self.tool_success) is not bool:
            raise TypeError("tool success must be bool or None")
        if self.family_id is not None:
            _id(self.family_id, "family ID")
        if self.stage is not None:
            _id(self.stage, "stage")
        object.__setattr__(
            self,
            "execution_receipt_ids",
            _strings(
                self.execution_receipt_ids,
                "execution receipt IDs",
                allow_empty=True,
                pattern=_IDENTIFIER,
            ),
        )
        reasons = _strings(self.reason_codes, "process reason codes")
        unknown = set(reasons).difference(PROCESS_REASON_CODES)
        if unknown:
            raise ValueError("unknown process reason code: " + ",".join(sorted(unknown)))
        object.__setattr__(self, "reason_codes", reasons)
        expected = f"process-event.{content_digest(self.identity_payload())[:40]}"
        if self.event_id != expected:
            raise ValueError("process event ID mismatch")
        if self.policy_layer is None and self.policy_decision_id is not None:
            raise ValueError("policy decision requires policy layer")
        if self.kind is ProcessEventKind.TOOL_CALL:
            if any(value is not None for value in (
                self.tool_call_id, self.tool_result_id, self.tool_name_digest,
                self.retry_identity, self.tool_success,
            )) and (
                self.tool_call_id is None
                or self.tool_name_digest is None
                or self.retry_identity is None
                or self.tool_result_id is not None
                or self.tool_success is not None
            ):
                raise ValueError("tool call event has result fields")
        elif self.kind is ProcessEventKind.TOOL_RESULT:
            if any(value is not None for value in (
                self.tool_call_id, self.tool_result_id, self.tool_name_digest,
                self.retry_identity, self.tool_success,
            )) and (
                self.tool_call_id is None
                or self.tool_result_id is None
                or self.tool_name_digest is None
                or self.retry_identity is None
            ):
                raise ValueError("tool result event lacks exact identity")
        elif any(value is not None for value in (
            self.tool_call_id, self.tool_result_id, self.tool_name_digest,
            self.retry_identity, self.tool_success,
        )):
            raise ValueError("non-tool process event has tool identity")

    @classmethod
    def create(
        cls,
        *,
        kind: ProcessEventKind | str,
        status: ProcessEventStatus | str,
        run_id: str,
        variant: str,
        trace_id: str,
        episode_id: str,
        session_id: str,
        task_id: str,
        host_event_id: str,
        source_revision: str,
        input_payload: object,
        output_payload: object,
        input_digest: str | None = None,
        output_digest: str | None = None,
        policy_decision_id: str | None = None,
        policy_layer: PolicyLayer | str | None = None,
        lineage_id: str | None = None,
        execution_receipt_ids: Sequence[str] = (),
        reason_codes: Sequence[str] = ("decision_observed",),
        tool_call_id: str | None = None,
        tool_result_id: str | None = None,
        tool_name_digest: str | None = None,
        retry_identity: str | None = None,
        tool_success: bool | None = None,
        family_id: str | None = None,
        stage: str | None = None,
    ) -> "ProcessEvent":
        input_digest = input_digest or content_digest(input_payload)
        output_digest = output_digest or content_digest(output_payload)
        _digest(input_digest, "process event input digest")
        _digest(output_digest, "process event output digest")
        identity = {
            "kind": ProcessEventKind(kind).value,
            "status": ProcessEventStatus(status).value,
            "run_id": run_id,
            "variant": variant,
            "trace_id": trace_id,
            "episode_id": episode_id,
            "session_id": session_id,
            "task_id": task_id,
            "host_event_id": host_event_id,
            "source_revision": source_revision,
            "input_digest": input_digest,
            "output_digest": output_digest,
            "policy_decision_id": policy_decision_id,
            "policy_layer": PolicyLayer(policy_layer).value if policy_layer is not None else None,
            "lineage_id": lineage_id,
            "execution_receipt_ids": list(execution_receipt_ids),
            "reason_codes": list(reason_codes),
            "tool_call_id": tool_call_id,
            "tool_result_id": tool_result_id,
            "tool_name_digest": tool_name_digest,
            "retry_identity": retry_identity,
            "tool_success": tool_success,
            "family_id": family_id,
            "stage": stage,
        }
        return cls(
            event_id=f"process-event.{content_digest(identity)[:40]}",
            kind=kind,
            status=status,
            run_id=run_id,
            variant=variant,
            trace_id=trace_id,
            episode_id=episode_id,
            session_id=session_id,
            task_id=task_id,
            host_event_id=host_event_id,
            source_revision=source_revision,
            input_digest=input_digest,
            output_digest=output_digest,
            policy_decision_id=policy_decision_id,
            policy_layer=policy_layer,
            lineage_id=lineage_id,
            execution_receipt_ids=tuple(execution_receipt_ids),
            reason_codes=tuple(reason_codes),
            tool_call_id=tool_call_id,
            tool_result_id=tool_result_id,
            tool_name_digest=tool_name_digest,
            retry_identity=retry_identity,
            tool_success=tool_success,
            family_id=family_id,
            stage=stage,
        )

    @classmethod
    def from_policy_decision(
        cls,
        decision: PolicyDecision,
        *,
        run_id: str,
        variant: str,
        trace_id: str,
        episode_id: str,
        session_id: str,
        task_id: str,
        host_event_id: str,
        family_id: str | None = None,
        stage: str | None = None,
        execution_receipt_ids: Sequence[str] = (),
    ) -> "ProcessEvent":
        """Project a policy decision without copying its input/output content."""

        receipts = tuple(execution_receipt_ids)
        if decision.execution_receipt_id and decision.execution_receipt_id not in receipts:
            receipts = (*receipts, decision.execution_receipt_id)
        # Policy decisions have layer-specific reason codes.  Keep the process
        # corpus vocabulary intentionally small and stable, while the richer
        # policy reason remains available in the policy ledger.
        mapped = tuple(
            value
            if value in PROCESS_REASON_CODES
            else PROCESS_POLICY_REASON_MAP.get(value, "decision_observed")
            for value in decision.reason_codes
        )
        reason_codes = tuple(dict.fromkeys(mapped)) or ("decision_observed",)
        return cls.create(
            kind=_layer_kind(decision.layer),
            status=ProcessEventStatus(decision.execution_status.value),
            run_id=run_id,
            variant=variant,
            trace_id=trace_id,
            episode_id=episode_id,
            session_id=session_id,
            task_id=task_id,
            host_event_id=host_event_id,
            source_revision=decision.source_revision,
            input_payload={},
            output_payload={},
            input_digest=decision.input_digest,
            output_digest=decision.output_digest,
            policy_decision_id=decision.decision_id,
            policy_layer=decision.layer,
            lineage_id=decision.lineage_id,
            execution_receipt_ids=receipts,
            reason_codes=reason_codes,
            family_id=family_id,
            stage=stage,
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "status": self.status.value,
            "run_id": self.run_id,
            "variant": self.variant,
            "trace_id": self.trace_id,
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "host_event_id": self.host_event_id,
            "source_revision": self.source_revision,
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
            "policy_decision_id": self.policy_decision_id,
            "policy_layer": self.policy_layer.value if self.policy_layer is not None else None,
            "lineage_id": self.lineage_id,
            "execution_receipt_ids": list(self.execution_receipt_ids),
            "reason_codes": list(self.reason_codes),
            "tool_call_id": self.tool_call_id,
            "tool_result_id": self.tool_result_id,
            "tool_name_digest": self.tool_name_digest,
            "retry_identity": self.retry_identity,
            "tool_success": self.tool_success,
            "family_id": self.family_id,
            "stage": self.stage,
        }

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "schema": PROCESS_FEEDBACK_SCHEMA,
            "event_id": self.event_id,
            **self.identity_payload(),
        }

    @classmethod
    def from_payload(cls, value: object) -> "ProcessEvent":
        fields = {
            "schema_version", "schema", "event_id", "kind", "status", "run_id",
            "variant", "trace_id", "episode_id", "session_id", "task_id",
            "host_event_id", "source_revision", "input_digest", "output_digest",
            "policy_decision_id", "policy_layer", "lineage_id", "execution_receipt_ids",
            "reason_codes", "tool_call_id", "tool_result_id", "tool_name_digest",
            "retry_identity", "tool_success", "family_id", "stage",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("malformed process feedback event")
        if value["schema"] != PROCESS_FEEDBACK_SCHEMA:
            raise ValueError("unsupported process feedback schema")
        for field in ("execution_receipt_ids", "reason_codes"):
            if not isinstance(value[field], list):
                raise ValueError("malformed process feedback event collections")
        try:
            event = cls(
                event_id=value["event_id"], kind=value["kind"], status=value["status"],
                run_id=value["run_id"], variant=value["variant"], trace_id=value["trace_id"],
                episode_id=value["episode_id"], session_id=value["session_id"], task_id=value["task_id"],
                host_event_id=value["host_event_id"], source_revision=value["source_revision"],
                input_digest=value["input_digest"], output_digest=value["output_digest"],
                policy_decision_id=value["policy_decision_id"], policy_layer=value["policy_layer"],
                lineage_id=value["lineage_id"], execution_receipt_ids=tuple(value["execution_receipt_ids"]),
                reason_codes=tuple(value["reason_codes"]),
                tool_call_id=value["tool_call_id"], tool_result_id=value["tool_result_id"],
                tool_name_digest=value["tool_name_digest"], retry_identity=value["retry_identity"],
                tool_success=value["tool_success"], family_id=value["family_id"],
                stage=value["stage"], schema_version=value["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed process feedback event") from exc
        if event.payload() != dict(value):
            raise ValueError("non-canonical process feedback event")
        return event


class JsonProcessFeedbackLedger:
    """Append-only, idempotent and restart-safe process event storage."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    @contextmanager
    def _lock(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read(self) -> dict[str, str]:
        records: dict[str, str] = {}
        if not self.path.exists():
            return records
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                event = ProcessEvent.from_payload(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"malformed process feedback at line {line_number}") from exc
            canonical = json.dumps(event.payload(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            previous = records.get(event.event_id)
            if previous is not None and previous != canonical:
                raise ValueError("conflicting process feedback event")
            records[event.event_id] = canonical
        return records

    @property
    def events(self) -> tuple[ProcessEvent, ...]:
        with self._lock():
            records = self._read()
        return tuple(ProcessEvent.from_payload(json.loads(value)) for value in records.values())

    def record(self, event: ProcessEvent) -> tuple[ProcessEvent, bool]:
        if not isinstance(event, ProcessEvent):
            raise TypeError("process feedback event has the wrong type")
        canonical = json.dumps(event.payload(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        with self._lock():
            records = self._read()
            previous = records.get(event.event_id)
            if previous is not None:
                if previous != canonical:
                    raise ValueError("conflicting process feedback event")
                return event, False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
            try:
                payload = [json.loads(value) for value in records.values()]
                payload.append(event.payload())
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    for item in payload:
                        handle.write(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return event, True


def audit_process_events(
    events: Iterable[ProcessEvent] | Path,
    *,
    required_identity: Mapping[str, str | None] | None = None,
    policy_decision_ids: Iterable[str] = (),
    policy_trigger_event_ids: Mapping[str, str] | None = None,
    source_revisions: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return deterministic errors for a process corpus.

    This audit is intentionally independent of benchmark graders and memory
    text.  A policy event must point at a known decision and every event must
    carry a source revision; optional maps let an adapter enforce cross-ledger
    joins when both ledgers are available.
    """

    if isinstance(events, Path):
        ledger = JsonProcessFeedbackLedger(events)
        items = ledger.events
    else:
        items = tuple(events)
    errors: list[str] = []
    known_decisions = set(policy_decision_ids)
    expected = dict(required_identity or {})
    seen: set[str] = set()
    for event in items:
        if event.event_id in seen:
            errors.append(f"{event.event_id}: duplicate event")
        seen.add(event.event_id)
        for field, value in expected.items():
            if value is not None and getattr(event, field) != value:
                errors.append(f"{event.event_id}: {field} does not match expected identity")
        if event.policy_decision_id is not None and known_decisions and event.policy_decision_id not in known_decisions:
            errors.append(f"{event.event_id}: policy decision is absent")
        if event.policy_decision_id is not None and policy_trigger_event_ids is not None:
            expected_host_event = policy_trigger_event_ids.get(event.policy_decision_id)
            if expected_host_event is not None and event.host_event_id != expected_host_event:
                errors.append(f"{event.event_id}: host event does not match policy trigger")
        if source_revisions is not None:
            expected_revision = source_revisions.get(event.host_event_id)
            if expected_revision is not None and expected_revision != event.source_revision:
                errors.append(f"{event.event_id}: source revision does not match host event")
        if event.status in {
            ProcessEventStatus.EXECUTED,
            ProcessEventStatus.SUCCESS,
            ProcessEventStatus.FAILED,
            ProcessEventStatus.REJECTED,
        } and not event.execution_receipt_ids:
            # A failed observation may be a host event with no runtime receipt;
            # it must then explain the failure explicitly rather than looking
            # like a successful execution.
            if event.kind not in {ProcessEventKind.HOST_LIFECYCLE, ProcessEventKind.TASK_OUTCOME}:
                errors.append(f"{event.event_id}: terminal process event lacks receipt")
        if event.status in {
            ProcessEventStatus.SKIPPED,
            ProcessEventStatus.DEFERRED,
        } and event.policy_decision_id is None and not {
            "absence",
            "unsupported_boundary",
        }.intersection(event.reason_codes):
            errors.append(
                f"{event.event_id}: non-executing process event lacks decision or reason"
            )
        if event.kind is ProcessEventKind.RETRIEVAL and "retrieval_miss" in event.reason_codes and event.status is not ProcessEventStatus.FAILED:
            errors.append(f"{event.event_id}: retrieval miss must be failed")
        if event.kind is ProcessEventKind.TOOL_RESULT and "tool_failure" in event.reason_codes and event.status is not ProcessEventStatus.FAILED:
            errors.append(f"{event.event_id}: tool failure must be failed")
        if event.kind is ProcessEventKind.EXPOSURE and "injection_failure" in event.reason_codes and event.status is not ProcessEventStatus.FAILED:
            errors.append(f"{event.event_id}: injection failure must be failed")
        if event.kind is ProcessEventKind.TASK_OUTCOME and "task_failure" in event.reason_codes and event.status is not ProcessEventStatus.FAILED:
            errors.append(f"{event.event_id}: task failure must be failed")
    return tuple(dict.fromkeys(errors))


__all__ = [
    "PROCESS_FEEDBACK_SCHEMA_VERSION",
    "PROCESS_FEEDBACK_SCHEMA",
    "ProcessEventKind",
    "ProcessEventStatus",
    "PROCESS_REASON_CODES",
    "ProcessEvent",
    "JsonProcessFeedbackLedger",
    "audit_process_events",
]
