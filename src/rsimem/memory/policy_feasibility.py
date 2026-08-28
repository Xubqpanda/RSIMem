"""Deterministic feasibility census for the six RSIMem policy layers.

The census is deliberately weaker than an effect experiment: it measures
whether a layer is observable, controllable, replayable, and connected to an
outcome signal.  It never turns missing evidence into a negative label and it
does not read benchmark graders or resource cost fields.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from .policy_contracts import PolicyArtifactIdentity, PolicyLayer, PolicyDecision, content_digest
from .policy_replay import PolicyReplayResult


class FeasibilityOutcome(StrEnum):
    USEFUL = "useful"
    HARMFUL = "harmful"
    MISSED = "missed"
    UNRESOLVED = "unresolved"
    CENSORED = "censored"


class FeasibilityStatus(StrEnum):
    OPTIMIZATION_READY = "optimization-ready"
    DIAGNOSTIC_ONLY = "diagnostic-only"
    VALIDATION_ONLY = "validation-only"


@dataclass(frozen=True, slots=True)
class FeedbackChain:
    """Evidence IDs for an attributable outcome.

    Useful requires opportunity/use/outcome.  Missed requires source/demand/
    absence/outcome.  IDs are opaque and content-free.
    """

    opportunity_id: str | None = None
    use_id: str | None = None
    outcome_id: str | None = None
    source_id: str | None = None
    demand_id: str | None = None
    absence_id: str | None = None

    def __post_init__(self) -> None:
        for value in (
            self.opportunity_id,
            self.use_id,
            self.outcome_id,
            self.source_id,
            self.demand_id,
            self.absence_id,
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError("feedback chain IDs must be non-empty strings")

    @property
    def complete_useful(self) -> bool:
        return all((self.opportunity_id, self.use_id, self.outcome_id))

    @property
    def complete_missed(self) -> bool:
        return all((self.source_id, self.demand_id, self.absence_id, self.outcome_id))

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(
            value
            for value in (
                self.opportunity_id,
                self.use_id,
                self.outcome_id,
                self.source_id,
                self.demand_id,
                self.absence_id,
            )
            if value is not None
        )


@dataclass(frozen=True, slots=True)
class ProcessFeedback:
    """Content-free before/after evidence for one layer intervention."""

    feedback_id: str
    event_id: str
    source_revision: str
    target_layer: PolicyLayer
    parent_decision_id: str
    candidate_decision_id: str
    parent_execution_receipt_ids: tuple[str, ...] = ()
    candidate_execution_receipt_ids: tuple[str, ...] = ()
    observed_before_digest: str = ""
    observed_after_digest: str = ""
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_layer", PolicyLayer(self.target_layer))
        for value, name in (
            (self.feedback_id, "process feedback ID"),
            (self.event_id, "process feedback event ID"),
            (self.source_revision, "process feedback source revision"),
            (self.parent_decision_id, "process feedback parent decision ID"),
            (self.candidate_decision_id, "process feedback candidate decision ID"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        for value, name in (
            (self.observed_before_digest, "process feedback before digest"),
            (self.observed_after_digest, "process feedback after digest"),
        ):
            if not isinstance(value, str) or len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError(f"{name} must be sha256")
        for values, name in (
            (self.parent_execution_receipt_ids, "parent execution receipts"),
            (self.candidate_execution_receipt_ids, "candidate execution receipts"),
            (self.reason_codes, "process feedback reason codes"),
        ):
            values = tuple(values)
            if len(values) != len(set(values)) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(f"{name} must be unique non-empty strings")
            attribute = {
                "parent execution receipts": "parent_execution_receipt_ids",
                "candidate execution receipts": "candidate_execution_receipt_ids",
                "process feedback reason codes": "reason_codes",
            }[name]
            object.__setattr__(self, attribute, values)

    @classmethod
    def create(
        cls,
        *,
        event_id: str,
        source_revision: str,
        target_layer: PolicyLayer,
        parent_decision_id: str,
        candidate_decision_id: str,
        parent_execution_receipt_ids: Sequence[str] = (),
        candidate_execution_receipt_ids: Sequence[str] = (),
        observed_before_digest: str,
        observed_after_digest: str,
        reason_codes: Sequence[str] = (),
    ) -> "ProcessFeedback":
        identity = {
            "event_id": event_id,
            "source_revision": source_revision,
            "target_layer": PolicyLayer(target_layer).value,
            "parent_decision_id": parent_decision_id,
            "candidate_decision_id": candidate_decision_id,
            "parent_execution_receipt_ids": list(parent_execution_receipt_ids),
            "candidate_execution_receipt_ids": list(candidate_execution_receipt_ids),
            "observed_before_digest": observed_before_digest,
            "observed_after_digest": observed_after_digest,
            "reason_codes": list(reason_codes),
        }
        return cls(
            f"process-feedback.{content_digest(identity)[:40]}",
            event_id,
            source_revision,
            target_layer,
            parent_decision_id,
            candidate_decision_id,
            tuple(parent_execution_receipt_ids),
            tuple(candidate_execution_receipt_ids),
            observed_before_digest,
            observed_after_digest,
            tuple(reason_codes),
        )

    def payload(self) -> dict[str, object]:
        return {
            "feedback_id": self.feedback_id,
            "event_id": self.event_id,
            "source_revision": self.source_revision,
            "target_layer": self.target_layer.value,
            "parent_decision_id": self.parent_decision_id,
            "candidate_decision_id": self.candidate_decision_id,
            "parent_execution_receipt_ids": list(self.parent_execution_receipt_ids),
            "candidate_execution_receipt_ids": list(self.candidate_execution_receipt_ids),
            "observed_before_digest": self.observed_before_digest,
            "observed_after_digest": self.observed_after_digest,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class LayerIntervention:
    case_id: str
    target_layer: PolicyLayer
    parent: PolicyReplayResult
    candidate: PolicyReplayResult
    parent_artifact: PolicyArtifactIdentity
    candidate_artifact: PolicyArtifactIdentity
    process_signal: bool
    outcome: FeasibilityOutcome
    feedback: FeedbackChain = FeedbackChain()
    reason_codes: tuple[str, ...] = ()
    process_feedback: ProcessFeedback | None = None

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("feasibility case ID must not be empty")
        object.__setattr__(self, "target_layer", PolicyLayer(self.target_layer))
        if self.parent.event != self.candidate.event:
            raise ValueError("parent and candidate must use the same trigger event")
        if self.parent.lineage.trigger_event_id != self.candidate.lineage.trigger_event_id:
            raise ValueError("parent and candidate must share trigger lineage")
        if not self.parent.audit.ok or not self.candidate.audit.ok:
            raise ValueError("parent and candidate replay audit must pass")
        if self.parent_artifact.kind.value == "joint" or self.candidate_artifact.kind.value == "joint":
            raise ValueError("joint artifacts are not valid for single-layer feasibility cases")
        if self.candidate_artifact.layers != (self.target_layer,):
            raise ValueError("candidate artifact must open exactly the target layer")
        if self.parent_artifact.layers != (self.target_layer,):
            raise ValueError("parent artifact must identify exactly the target layer")
        if self.parent_artifact.artifact_id == self.candidate_artifact.artifact_id:
            raise ValueError("parent and candidate artifacts must be distinct")
        if type(self.process_signal) is not bool:
            raise ValueError("process_signal must be bool")
        outcome = FeasibilityOutcome(self.outcome)
        reasons = tuple(self.reason_codes)
        if not reasons or len(reasons) != len(set(reasons)) or any(not item.strip() for item in reasons):
            raise ValueError("feasibility reason codes are required and unique")
        # A caller may construct a provisional useful/missed label before all
        # delayed evidence has arrived.  Never let that incomplete label enter
        # a reward denominator: fail closed to unresolved and retain an
        # explicit diagnostic reason for the census/audit layer.
        if outcome == FeasibilityOutcome.USEFUL and not self.feedback.complete_useful:
            outcome = FeasibilityOutcome.UNRESOLVED
            if "incomplete_useful_feedback" not in reasons:
                reasons = (*reasons, "incomplete_useful_feedback")
        elif outcome == FeasibilityOutcome.MISSED and not self.feedback.complete_missed:
            outcome = FeasibilityOutcome.UNRESOLVED
            if "incomplete_missed_feedback" not in reasons:
                reasons = (*reasons, "incomplete_missed_feedback")
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "reason_codes", reasons)
        self._validate_feedback(outcome)
        if self.action_changed is False:
            raise ValueError("candidate must change the target-layer decision fingerprint")
        if self.process_signal:
            process_feedback = self.process_feedback or self._derive_process_feedback()
            self._validate_process_feedback(process_feedback)
            object.__setattr__(self, "process_feedback", process_feedback)
        elif self.process_feedback is not None:
            raise ValueError("process feedback cannot be present when process_signal is false")

    def _validate_feedback(self, outcome: FeasibilityOutcome) -> None:
        if outcome == FeasibilityOutcome.USEFUL and not self.feedback.complete_useful:
            raise ValueError("useful feedback requires opportunity/use/outcome chain")
        if outcome == FeasibilityOutcome.MISSED and not self.feedback.complete_missed:
            raise ValueError("missed feedback requires source/demand/absence/outcome chain")
        if outcome in {FeasibilityOutcome.UNRESOLVED, FeasibilityOutcome.CENSORED} and (
            self.feedback.complete_useful or self.feedback.complete_missed
        ):
            raise ValueError("unresolved/censored feedback cannot carry a complete reward chain")

    def _derive_process_feedback(self) -> ProcessFeedback:
        parent = self.parent_decision
        candidate = self.candidate_decision
        if parent is None or candidate is None:
            raise ValueError("process feedback requires target-layer decisions")
        return ProcessFeedback.create(
            event_id=self.parent.event.event_id,
            source_revision=self.parent.event.source_revision,
            target_layer=self.target_layer,
            parent_decision_id=parent.decision_id,
            candidate_decision_id=candidate.decision_id,
            parent_execution_receipt_ids=tuple(
                decision.execution_receipt_id
                for decision in self.parent.decisions
                if decision.execution_receipt_id
            ),
            candidate_execution_receipt_ids=tuple(
                decision.execution_receipt_id
                for decision in self.candidate.decisions
                if decision.execution_receipt_id
            ),
            observed_before_digest=parent.output_digest,
            observed_after_digest=candidate.output_digest,
            reason_codes=("decision_observed",),
        )

    def _validate_process_feedback(self, value: ProcessFeedback) -> None:
        if not isinstance(value, ProcessFeedback):
            raise ValueError("process feedback has the wrong type")
        parent = self.parent_decision
        candidate = self.candidate_decision
        if parent is None or candidate is None:
            raise ValueError("process feedback requires target-layer decisions")
        if (
            value.event_id != self.parent.event.event_id
            or value.source_revision != self.parent.event.source_revision
            or value.target_layer != self.target_layer
            or value.parent_decision_id != parent.decision_id
            or value.candidate_decision_id != candidate.decision_id
            or value.observed_before_digest != parent.output_digest
            or value.observed_after_digest != candidate.output_digest
        ):
            raise ValueError("process feedback does not match replay decisions")

    @property
    def parent_decision(self) -> PolicyDecision | None:
        return next((item for item in self.parent.decisions if item.layer == self.target_layer), None)

    @property
    def candidate_decision(self) -> PolicyDecision | None:
        return next((item for item in self.candidate.decisions if item.layer == self.target_layer), None)

    @property
    def action_changed(self) -> bool:
        parent = self.parent_decision
        candidate = self.candidate_decision
        if parent is None or candidate is None:
            return parent != candidate
        return (
            parent.decision_id != candidate.decision_id
            or parent.input_digest != candidate.input_digest
            or parent.output_digest != candidate.output_digest
            or parent.action != candidate.action
        )

    @property
    def outcome_resolved(self) -> bool:
        return self.outcome in {FeasibilityOutcome.USEFUL, FeasibilityOutcome.HARMFUL, FeasibilityOutcome.MISSED}

    @property
    def intervention_fingerprint(self) -> str:
        return content_digest({
            "case_id": self.case_id,
            "target_layer": self.target_layer.value,
            "parent_artifact": self.parent_artifact.artifact_id,
            "candidate_artifact": self.candidate_artifact.artifact_id,
            "parent_decision": self.parent_decision.decision_id if self.parent_decision else None,
            "candidate_decision": self.candidate_decision.decision_id if self.candidate_decision else None,
        })

    @property
    def replay_payload(self) -> dict[str, object]:
        """Content-free identity used to persist and replay one intervention."""

        return {
            "case_id": self.case_id,
            "target_layer": self.target_layer.value,
            "parent_artifact_id": self.parent_artifact.artifact_id,
            "candidate_artifact_id": self.candidate_artifact.artifact_id,
            "parent_event_id": self.parent.event.event_id,
            "candidate_event_id": self.candidate.event.event_id,
            "parent_source_revision": self.parent.event.source_revision,
            "candidate_source_revision": self.candidate.event.source_revision,
            "parent_decision_ids": [item.decision_id for item in self.parent.decisions],
            "candidate_decision_ids": [item.decision_id for item in self.candidate.decisions],
            "parent_lineage_id": self.parent.lineage.lineage_id,
            "candidate_lineage_id": self.candidate.lineage.lineage_id,
            "parent_audit_ok": self.parent.audit.ok,
            "candidate_audit_ok": self.candidate.audit.ok,
            "process_signal": self.process_signal,
            "outcome": self.outcome.value,
            "feedback_ids": list(self.feedback.ids),
            "reason_codes": list(self.reason_codes),
            "intervention_fingerprint": self.intervention_fingerprint,
            "process_feedback_id": (
                self.process_feedback.feedback_id
                if self.process_feedback is not None
                else None
            ),
        }


FEASIBILITY_EVIDENCE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FeasibilityEvidenceRecord:
    """Content-free durable identity for one replayable intervention."""

    record_id: str
    replay_payload: Mapping[str, object]
    schema_version: int = FEASIBILITY_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FEASIBILITY_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported feasibility evidence schema version")
        if not isinstance(self.record_id, str) or not self.record_id.strip():
            raise ValueError("feasibility evidence record ID must not be empty")
        payload = dict(self.replay_payload)
        required = {
            "case_id", "target_layer", "parent_artifact_id", "candidate_artifact_id",
            "parent_event_id", "candidate_event_id", "parent_source_revision",
            "candidate_source_revision", "parent_decision_ids", "candidate_decision_ids",
            "parent_lineage_id", "candidate_lineage_id", "parent_audit_ok",
            "candidate_audit_ok", "process_signal", "outcome", "feedback_ids",
            "reason_codes", "intervention_fingerprint", "process_feedback_id",
        }
        if set(payload) != required:
            raise ValueError("malformed feasibility replay payload")
        if not isinstance(payload["case_id"], str) or not payload["case_id"].strip():
            raise ValueError("feasibility replay case ID must not be empty")
        if payload["target_layer"] not in {layer.value for layer in PolicyLayer}:
            raise ValueError("feasibility replay target layer is invalid")
        if payload["process_feedback_id"] is not None and (
            not isinstance(payload["process_feedback_id"], str)
            or not payload["process_feedback_id"].strip()
        ):
            raise ValueError("feasibility process feedback ID is invalid")
        for field in ("parent_decision_ids", "candidate_decision_ids", "feedback_ids", "reason_codes"):
            values = payload[field]
            if not isinstance(values, list) or len(values) != len(set(values)) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError("feasibility replay ID lists are invalid")
        for field in ("parent_audit_ok", "candidate_audit_ok", "process_signal"):
            if type(payload[field]) is not bool:
                raise ValueError("feasibility replay flags are invalid")
        outcome = FeasibilityOutcome(payload["outcome"])
        payload["outcome"] = outcome.value
        object.__setattr__(self, "replay_payload", MappingProxyType(payload))
        expected = f"feasibility-record.{content_digest(payload)[:40]}"
        if self.record_id != expected:
            raise ValueError("feasibility evidence record ID mismatch")

    @classmethod
    def from_case(cls, case: LayerIntervention) -> "FeasibilityEvidenceRecord":
        payload = case.replay_payload
        return cls(f"feasibility-record.{content_digest(payload)[:40]}", payload)

    def payload(self) -> dict[str, object]:
        return {
            "schemaVersion": self.schema_version,
            "recordId": self.record_id,
            "replayPayload": dict(self.replay_payload),
        }

    @classmethod
    def from_payload(cls, value: object) -> "FeasibilityEvidenceRecord":
        if not isinstance(value, dict) or set(value) != {
            "schemaVersion", "recordId", "replayPayload",
        }:
            raise ValueError("malformed feasibility evidence record")
        try:
            return cls(
                value["recordId"],
                value["replayPayload"],
                value["schemaVersion"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed feasibility evidence record") from exc


class JsonFeasibilityEvidenceLedger:
    """Crash-safe, idempotent storage for replay identities."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self._records: dict[str, str] = {}
        self._load()

    @property
    def records(self) -> tuple[FeasibilityEvidenceRecord, ...]:
        return tuple(
            FeasibilityEvidenceRecord.from_payload(json.loads(value))
            for value in self._records.values()
        )

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                record = FeasibilityEvidenceRecord.from_payload(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"malformed feasibility evidence at line {line_number}") from exc
            canonical = json.dumps(record.payload(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            previous = self._records.get(record.record_id)
            if previous is not None and previous != canonical:
                raise ValueError("conflicting feasibility evidence record")
            self._records[record.record_id] = canonical

    def record(self, record: FeasibilityEvidenceRecord) -> None:
        canonical = json.dumps(record.payload(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        with self._lock():
            self._load()
            previous = self._records.get(record.record_id)
            if previous is not None:
                if previous != canonical:
                    raise ValueError("conflicting feasibility evidence record")
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
            try:
                payload = [json.loads(value) for value in self._records.values()]
                payload.append(record.payload())
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    for item in payload:
                        handle.write(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            self._records[record.record_id] = canonical

    def record_case(self, case: LayerIntervention) -> FeasibilityEvidenceRecord:
        record = FeasibilityEvidenceRecord.from_case(case)
        self.record(record)
        return record

    @contextmanager
    def _lock(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True, slots=True)
class LayerFeasibilityCensus:
    layer: PolicyLayer
    case_count: int
    signal_count: int
    action_variation_count: int
    outcome_variation_count: int
    outcome_counts: Mapping[str, int]
    unknown_count: int
    complete_feedback_count: int
    status: FeasibilityStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer", PolicyLayer(self.layer))
        for value, name in (
            (self.case_count, "case count"),
            (self.signal_count, "signal count"),
            (self.action_variation_count, "action variation count"),
            (self.outcome_variation_count, "outcome variation count"),
            (self.unknown_count, "unknown count"),
            (self.complete_feedback_count, "complete feedback count"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if any(value > self.case_count for value in (
            self.signal_count,
            self.action_variation_count,
            self.outcome_variation_count,
            self.unknown_count,
            self.complete_feedback_count,
        )):
            raise ValueError("census counts cannot exceed case count")
        object.__setattr__(self, "outcome_counts", MappingProxyType(dict(self.outcome_counts)))
        reasons = tuple(self.reason_codes)
        if len(reasons) != len(set(reasons)) or any(not isinstance(item, str) or not item.strip() for item in reasons):
            raise ValueError("census reason codes must be unique non-empty strings")
        object.__setattr__(self, "reason_codes", reasons)

    @property
    def signal_coverage(self) -> float | None:
        return self.signal_count / self.case_count if self.case_count else None

    @property
    def action_variation(self) -> float | None:
        return self.action_variation_count / self.case_count if self.case_count else None

    @property
    def outcome_variation(self) -> float | None:
        return self.outcome_variation_count / self.case_count if self.case_count else None

    def payload(self) -> dict[str, object]:
        return {
            "layer": self.layer.value,
            "caseCount": self.case_count,
            "signalCount": self.signal_count,
            "signalCoverage": self.signal_coverage,
            "actionVariationCount": self.action_variation_count,
            "actionVariation": self.action_variation,
            "outcomeVariationCount": self.outcome_variation_count,
            "outcomeVariation": self.outcome_variation,
            "outcomeCounts": dict(self.outcome_counts),
            "unknownCount": self.unknown_count,
            "completeFeedbackCount": self.complete_feedback_count,
            "status": self.status.value,
            "reasonCodes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class PolicyFeasibilityReport:
    cases: tuple[LayerIntervention, ...]
    census: tuple[LayerFeasibilityCensus, ...]

    @property
    def ok(self) -> bool:
        return bool(self.cases) and all(item.status != FeasibilityStatus.DIAGNOSTIC_ONLY for item in self.census)

    @property
    def digest(self) -> str:
        """Stable digest for cross-process/restart report comparison."""

        return content_digest(self.payload())

    def payload(self) -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "ok": self.ok,
            "caseCount": len(self.cases),
            "cases": [
                {
                    "caseId": case.case_id,
                    "targetLayer": case.target_layer.value,
                    "outcome": case.outcome.value,
                    "processSignal": case.process_signal,
                    "actionChanged": case.action_changed,
                    "interventionFingerprint": case.intervention_fingerprint,
                    "reasonCodes": list(case.reason_codes),
                }
                for case in self.cases
            ],
            "layers": [item.payload() for item in self.census],
        }


def build_feasibility_report(cases: Iterable[LayerIntervention]) -> PolicyFeasibilityReport:
    normalized = tuple(cases)
    case_ids = tuple(case.case_id for case in normalized)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("feasibility case IDs must be unique")
    grouped: dict[PolicyLayer, list[LayerIntervention]] = {layer: [] for layer in PolicyLayer}
    for case in normalized:
        grouped[case.target_layer].append(case)
    census: list[LayerFeasibilityCensus] = []
    for layer in PolicyLayer:
        items = grouped[layer]
        outcomes: dict[str, int] = {}
        for case in items:
            outcomes[case.outcome.value] = outcomes.get(case.outcome.value, 0) + 1
        signal_count = sum(1 for case in items if case.process_signal)
        action_count = sum(1 for case in items if case.action_changed)
        outcome_variation_count = sum(1 for case in items if case.outcome_resolved)
        unknown_count = sum(
            1 for case in items
            if case.outcome in {FeasibilityOutcome.UNRESOLVED, FeasibilityOutcome.CENSORED}
        )
        complete_count = sum(
            1 for case in items
            if case.feedback.complete_useful or case.feedback.complete_missed
        )
        reasons: list[str] = []
        if not items:
            reasons.append("no_cases")
        if items and not signal_count:
            reasons.append("no_process_signal")
        if items and not action_count:
            reasons.append("no_action_variation")
        if items and len(outcomes) < 2:
            reasons.append("no_outcome_variation")
        if unknown_count:
            reasons.append("unresolved_or_censored_present")
        if not reasons:
            status = FeasibilityStatus.OPTIMIZATION_READY
        elif "no_process_signal" in reasons or "no_action_variation" in reasons or "no_cases" in reasons:
            status = FeasibilityStatus.DIAGNOSTIC_ONLY
        else:
            status = FeasibilityStatus.VALIDATION_ONLY
        census.append(LayerFeasibilityCensus(
            layer,
            len(items),
            signal_count,
            action_count,
            outcome_variation_count,
            outcomes,
            unknown_count,
            complete_count,
            status,
            tuple(reasons),
        ))
    return PolicyFeasibilityReport(normalized, tuple(census))


def validate_feasibility_case(case: LayerIntervention) -> None:
    """Raise if a case violates target-layer or evidence-chain boundaries."""

    if not case.action_changed:
        raise ValueError("feasibility candidate did not change target layer")
    if case.outcome in {FeasibilityOutcome.USEFUL, FeasibilityOutcome.MISSED} and not case.outcome_resolved:
        raise ValueError("resolved outcome is missing a complete evidence chain")


__all__ = [
    "FEASIBILITY_EVIDENCE_SCHEMA_VERSION",
    "FeasibilityOutcome",
    "FeasibilityStatus",
    "FeedbackChain",
    "ProcessFeedback",
    "LayerIntervention",
    "FeasibilityEvidenceRecord",
    "JsonFeasibilityEvidenceLedger",
    "LayerFeasibilityCensus",
    "PolicyFeasibilityReport",
    "build_feasibility_report",
    "validate_feasibility_case",
]
