"""Deterministic feasibility census for the six RSIMem policy layers.

The census is deliberately weaker than an effect experiment: it measures
whether a layer is observable, controllable, replayable, and connected to an
outcome signal.  It never turns missing evidence into a negative label and it
does not read benchmark graders or resource cost fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
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

    def _validate_feedback(self, outcome: FeasibilityOutcome) -> None:
        if outcome == FeasibilityOutcome.USEFUL and not self.feedback.complete_useful:
            raise ValueError("useful feedback requires opportunity/use/outcome chain")
        if outcome == FeasibilityOutcome.MISSED and not self.feedback.complete_missed:
            raise ValueError("missed feedback requires source/demand/absence/outcome chain")
        if outcome in {FeasibilityOutcome.UNRESOLVED, FeasibilityOutcome.CENSORED} and (
            self.feedback.complete_useful or self.feedback.complete_missed
        ):
            raise ValueError("unresolved/censored feedback cannot carry a complete reward chain")

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
        }


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
    "FeasibilityOutcome",
    "FeasibilityStatus",
    "FeedbackChain",
    "LayerIntervention",
    "LayerFeasibilityCensus",
    "PolicyFeasibilityReport",
    "build_feasibility_report",
    "validate_feasibility_case",
]
