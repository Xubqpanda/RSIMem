"""Content-free, crash-safe evidence ledger for six-layer policy decisions."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .policy_contracts import PolicyDecision, content_digest


POLICY_EVIDENCE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PolicyDecisionEvidence:
    event_id: str
    run_id: str
    episode_id: str
    session_id: str
    task_id: str
    snapshot_id: str
    layer: str
    decision_id: str
    policy_version: str
    source_revision: str
    action: str
    execution_status: str
    reason_codes: tuple[str, ...]
    input_digest: str
    output_digest: str
    lineage_id: str
    trigger_event_id: str | None
    execution_receipt_id: str | None
    mutation_receipt_ids: tuple[str, ...] = ()
    injection_receipt_ids: tuple[str, ...] = ()
    future_feedback_ids: tuple[str, ...] = ()
    schema_version: int = POLICY_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported policy evidence schema version")
        for value, name in (
            (self.event_id, "policy evidence event ID"),
            (self.run_id, "run ID"),
            (self.episode_id, "episode ID"),
            (self.session_id, "session ID"),
            (self.task_id, "task ID"),
            (self.snapshot_id, "snapshot ID"),
            (self.layer, "policy layer"),
            (self.decision_id, "decision ID"),
            (self.policy_version, "policy version"),
            (self.source_revision, "source revision"),
            (self.action, "policy action"),
            (self.execution_status, "execution status"),
            (self.lineage_id, "lineage ID"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if self.trigger_event_id is not None and not self.trigger_event_id.strip():
            raise ValueError("trigger event ID must not be empty")
        if self.execution_receipt_id is not None and not self.execution_receipt_id.strip():
            raise ValueError("execution receipt ID must not be empty")
        for value in (self.input_digest, self.output_digest):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("policy evidence digest must be sha256")
        reasons = tuple(self.reason_codes)
        if not reasons or len(reasons) != len(set(reasons)) or any(not item.strip() for item in reasons):
            raise ValueError("policy evidence reason codes are invalid")
        object.__setattr__(self, "reason_codes", reasons)
        for name in ("mutation_receipt_ids", "injection_receipt_ids", "future_feedback_ids"):
            values = tuple(getattr(self, name))
            if len(values) != len(set(values)) or any(not item.strip() for item in values):
                raise ValueError(f"policy evidence {name} are invalid")
            object.__setattr__(self, name, values)

    @classmethod
    def from_decision(
        cls,
        decision: PolicyDecision,
        *,
        run_id: str,
        episode_id: str,
        session_id: str,
        task_id: str,
        snapshot_id: str,
        mutation_receipt_ids: Sequence[str] = (),
        injection_receipt_ids: Sequence[str] = (),
        future_feedback_ids: Sequence[str] = (),
    ) -> "PolicyDecisionEvidence":
        identity = {
            "run_id": run_id,
            "episode_id": episode_id,
            "session_id": session_id,
            "task_id": task_id,
            "snapshot_id": snapshot_id,
            "layer": decision.layer.value,
            "decision_id": decision.decision_id,
            "lineage_id": decision.lineage_id,
            "source_revision": decision.source_revision,
            "action": decision.action.value,
            "execution_status": decision.execution_status.value,
            "reason_codes": list(decision.reason_codes),
            "input_digest": decision.input_digest,
            "output_digest": decision.output_digest,
            "execution_receipt_id": decision.execution_receipt_id,
            "mutation_receipt_ids": list(mutation_receipt_ids),
            "injection_receipt_ids": list(injection_receipt_ids),
            "future_feedback_ids": list(future_feedback_ids),
        }
        return cls(
            event_id=f"policy-event.{content_digest(identity)[:40]}",
            run_id=run_id,
            episode_id=episode_id,
            session_id=session_id,
            task_id=task_id,
            snapshot_id=snapshot_id,
            layer=decision.layer.value,
            decision_id=decision.decision_id,
            policy_version=decision.policy_version,
            source_revision=decision.source_revision,
            action=decision.action.value,
            execution_status=decision.execution_status.value,
            reason_codes=decision.reason_codes,
            input_digest=decision.input_digest,
            output_digest=decision.output_digest,
            lineage_id=decision.lineage_id,
            trigger_event_id=decision.trigger_event_id,
            execution_receipt_id=decision.execution_receipt_id,
            mutation_receipt_ids=tuple(mutation_receipt_ids),
            injection_receipt_ids=tuple(injection_receipt_ids),
            future_feedback_ids=tuple(future_feedback_ids),
        )

    def payload(self) -> dict[str, object]:
        return {
            "schemaVersion": self.schema_version,
            "eventId": self.event_id,
            "runId": self.run_id,
            "episodeId": self.episode_id,
            "sessionId": self.session_id,
            "taskId": self.task_id,
            "snapshotId": self.snapshot_id,
            "layer": self.layer,
            "decisionId": self.decision_id,
            "policyVersion": self.policy_version,
            "sourceRevision": self.source_revision,
            "action": self.action,
            "executionStatus": self.execution_status,
            "reasonCodes": list(self.reason_codes),
            "inputDigest": self.input_digest,
            "outputDigest": self.output_digest,
            "lineageId": self.lineage_id,
            "triggerEventId": self.trigger_event_id,
            "executionReceiptId": self.execution_receipt_id,
            "mutationReceiptIds": list(self.mutation_receipt_ids),
            "injectionReceiptIds": list(self.injection_receipt_ids),
            "futureFeedbackIds": list(self.future_feedback_ids),
        }

    @classmethod
    def from_payload(cls, value: object) -> "PolicyDecisionEvidence":
        fields = {
            "schemaVersion", "eventId", "runId", "episodeId", "sessionId", "taskId",
            "snapshotId", "layer", "decisionId", "policyVersion", "sourceRevision",
            "action", "executionStatus", "reasonCodes", "inputDigest", "outputDigest",
            "lineageId", "triggerEventId", "executionReceiptId", "mutationReceiptIds",
            "injectionReceiptIds", "futureFeedbackIds",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("malformed policy evidence event")
        try:
            result = cls(
                event_id=value["eventId"], run_id=value["runId"], episode_id=value["episodeId"],
                session_id=value["sessionId"], task_id=value["taskId"], snapshot_id=value["snapshotId"],
                layer=value["layer"], decision_id=value["decisionId"], policy_version=value["policyVersion"],
                source_revision=value["sourceRevision"], action=value["action"],
                execution_status=value["executionStatus"], reason_codes=tuple(value["reasonCodes"]),
                input_digest=value["inputDigest"], output_digest=value["outputDigest"],
                lineage_id=value["lineageId"], trigger_event_id=value["triggerEventId"],
                execution_receipt_id=value["executionReceiptId"],
                mutation_receipt_ids=tuple(value["mutationReceiptIds"]),
                injection_receipt_ids=tuple(value["injectionReceiptIds"]),
                future_feedback_ids=tuple(value["futureFeedbackIds"]),
                schema_version=value["schemaVersion"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed policy evidence event") from exc
        if result.payload() != value:
            raise ValueError("non-canonical policy evidence event")
        event_identity = {
            'run_id': result.run_id,
            'episode_id': result.episode_id,
            'session_id': result.session_id,
            'task_id': result.task_id,
            'snapshot_id': result.snapshot_id,
            'layer': result.layer,
            'decision_id': result.decision_id,
            'lineage_id': result.lineage_id,
            'source_revision': result.source_revision,
            'action': result.action,
            'execution_status': result.execution_status,
            'reason_codes': list(result.reason_codes),
            'input_digest': result.input_digest,
            'output_digest': result.output_digest,
            'execution_receipt_id': result.execution_receipt_id,
            'mutation_receipt_ids': list(result.mutation_receipt_ids),
            'injection_receipt_ids': list(result.injection_receipt_ids),
            'future_feedback_ids': list(result.future_feedback_ids),
        }
        expected_event_id = f"policy-event.{content_digest(event_identity)[:40]}"
        if result.event_id != expected_event_id:
            raise ValueError("malformed policy evidence event ID mismatch")
        return result


class JsonPolicyDecisionLedger:
    """Append-only policy evidence with idempotent event IDs and conflict checks."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self._events: dict[str, str] = {}
        self._load()

    @property
    def events(self) -> tuple[dict[str, object], ...]:
        return tuple(json.loads(value) for value in self._events.values())

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed policy evidence at line {line_number}") from exc
            validated = PolicyDecisionEvidence.from_payload(value)
            value = validated.payload()
            canonical = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            previous = self._events.get(value["eventId"])
            if previous is not None and previous != canonical:
                raise ValueError("conflicting policy evidence event")
            self._events[value["eventId"]] = canonical

    def record(self, evidence: PolicyDecisionEvidence) -> None:
        canonical = json.dumps(evidence.payload(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        with self._lock():
            self._load()
            previous = self._events.get(evidence.event_id)
            if previous is not None:
                if previous != canonical:
                    raise ValueError("conflicting policy evidence event")
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
            try:
                # Preserve the append-only logical set while atomically replacing
                # the file, so a crash cannot leave a partial JSON line.
                payload = [json.loads(value) for value in self._events.values()]
                payload.append(evidence.payload())
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    for item in payload:
                        handle.write(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            self._events[evidence.event_id] = canonical

    def record_decision(self, decision: PolicyDecision, **kwargs: object) -> PolicyDecisionEvidence:
        evidence = PolicyDecisionEvidence.from_decision(decision, **kwargs)  # type: ignore[arg-type]
        self.record(evidence)
        return evidence

    @contextmanager
    def _lock(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


__all__ = [
    "POLICY_EVIDENCE_SCHEMA_VERSION",
    "PolicyDecisionEvidence",
    "JsonPolicyDecisionLedger",
]
