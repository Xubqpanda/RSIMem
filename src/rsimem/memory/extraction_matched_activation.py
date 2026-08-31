"""Matched-trial activation and rollback for extraction prompt candidates."""

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
from typing import Mapping

from .extraction_offline_validation import (
    ExtractionOfflineDecisionStatus,
    ExtractionOfflineValidationDecision,
    OfflineRatioEvidence,
    extraction_ratio_evidence,
)
from .extraction_policy_artifact import ExtractionPromptPolicyArtifact
from .extraction_policy_store import (
    ExtractionPolicyState,
    JsonExtractionPolicyStore,
)
from .extraction_prompt_validation import (
    ExtractionAcceptanceCriteria,
    ExtractionPromptMatchedValidator,
    ExtractionPromptValidationSplit,
    ExtractionValidationDecision,
    ExtractionValidationObservation,
    ExtractionValidationSafetyEvidence,
    ExtractionValidationSplitRole,
    ExtractionValidationVariant,
)
from .prompt_components import canonical_json, content_digest
from .evidence_planes import EvidencePlane, EvidenceSourceKind
from .revocation import JsonRevocationRegistry


EXTRACTION_MATCHED_ACTIVATION_SCHEMA_VERSION = 1
EXTRACTION_MATCHED_DECISION_SCHEMA = "extraction-matched-trial-decision-v1"
EXTRACTION_ROLLBACK_EVIDENCE_SCHEMA = "extraction-rollback-evidence-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _require_id(value: object, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")


def _require_digest(value: object, name: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be sha256")


class ExtractionMatchedTrialStatus(StrEnum):
    REJECTED = "rejected"
    ACCEPTED_FOR_ACTIVATION = "accepted_for_activation"


class ExtractionMatchedConstraint(StrEnum):
    MATCHED_PAIRS = "matched_pairs"
    RESOLVED_EXAMPLES = "resolved_examples"
    USEFUL_RATE = "useful_rate"
    HARMFUL_RATE = "harmful_rate"
    NONEMPTY_COVERAGE = "nonempty_coverage"
    EMPTY_RATE = "empty_rate"
    MISSED_RATE = "missed_rate"
    SAFETY = "safety"
    EXTRACTION_INTERVENTION = "extraction_intervention"


@dataclass(frozen=True, slots=True)
class ExtractionMatchedConstraintResult:
    constraint: ExtractionMatchedConstraint
    passed: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "constraint",
            ExtractionMatchedConstraint(self.constraint),
        )
        if type(self.passed) is not bool:
            raise TypeError("matched constraint status must be bool")
        if self.passed != (not self.reason_codes):
            raise ValueError("matched constraint status and reasons disagree")
        for reason in self.reason_codes:
            _require_id(reason, "matched constraint reason")

    def payload(self) -> dict[str, object]:
        return {
            "constraint": self.constraint.value,
            "passed": self.passed,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionMatchedConstraintResult":
        if not isinstance(value, Mapping) or set(value) != {
            "constraint",
            "passed",
            "reason_codes",
        } or not isinstance(value["reason_codes"], list):
            raise ValueError("malformed extraction matched constraint")
        try:
            return cls(
                value["constraint"],
                value["passed"],
                tuple(value["reason_codes"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed extraction matched constraint") from exc


_CONSTRAINT_REASONS = {
    ExtractionMatchedConstraint.MATCHED_PAIRS: ("insufficient_matched_pairs",),
    ExtractionMatchedConstraint.RESOLVED_EXAMPLES: (
        "insufficient_resolved_examples",
    ),
    ExtractionMatchedConstraint.USEFUL_RATE: ("useful_rate_not_improved",),
    ExtractionMatchedConstraint.HARMFUL_RATE: (
        "harmful_rate_unknown",
        "harmful_rate_regression",
    ),
    ExtractionMatchedConstraint.NONEMPTY_COVERAGE: ("coverage_collapse",),
    ExtractionMatchedConstraint.EMPTY_RATE: ("empty_rate_exceeded",),
    ExtractionMatchedConstraint.MISSED_RATE: (
        "missed_rate_unknown",
        "missed_rate_regression",
    ),
    ExtractionMatchedConstraint.SAFETY: ("safety_failure",),
    ExtractionMatchedConstraint.EXTRACTION_INTERVENTION: (
        "no_extraction_intervention",
    ),
}


def _constraint_results(
    quality: ExtractionValidationDecision,
) -> tuple[ExtractionMatchedConstraintResult, ...]:
    reasons = set(quality.reason_codes)
    return tuple(
        ExtractionMatchedConstraintResult(
            constraint,
            not any(reason in reasons for reason in failure_reasons),
            tuple(reason for reason in failure_reasons if reason in reasons),
        )
        for constraint, failure_reasons in _CONSTRAINT_REASONS.items()
    )


def _ratio_from_payload(value: object) -> OfflineRatioEvidence:
    if not isinstance(value, Mapping) or set(value) != {
        "metric",
        "numerator",
        "denominator",
        "unknown_count",
        "value",
    }:
        raise ValueError("malformed extraction matched ratio")
    try:
        return OfflineRatioEvidence(
            value["metric"],
            value["numerator"],
            value["denominator"],
            value["unknown_count"],
            value["value"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("malformed extraction matched ratio") from exc


@dataclass(frozen=True, slots=True)
class ExtractionMatchedTrialDecision:
    decision_id: str
    status: ExtractionMatchedTrialStatus
    offline_decision_id: str
    parent_artifact_id: str
    parent_artifact_digest: str
    candidate_artifact_id: str
    candidate_artifact_digest: str
    parent_runtime_artifact_id: str
    candidate_runtime_artifact_id: str
    split_id: str
    criteria_digest: str
    quality_decision: ExtractionValidationDecision
    parent_ratios: tuple[OfflineRatioEvidence, ...]
    candidate_ratios: tuple[OfflineRatioEvidence, ...]
    constraint_results: tuple[ExtractionMatchedConstraintResult, ...]
    reason_codes: tuple[str, ...]
    decision_schema: str = EXTRACTION_MATCHED_DECISION_SCHEMA
    schema_version: int = EXTRACTION_MATCHED_ACTIVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != EXTRACTION_MATCHED_ACTIVATION_SCHEMA_VERSION
            or self.decision_schema != EXTRACTION_MATCHED_DECISION_SCHEMA
        ):
            raise ValueError("unsupported extraction matched trial decision")
        object.__setattr__(self, "status", ExtractionMatchedTrialStatus(self.status))
        for value in (
            self.decision_id,
            self.offline_decision_id,
            self.parent_artifact_id,
            self.candidate_artifact_id,
            self.parent_runtime_artifact_id,
            self.candidate_runtime_artifact_id,
            self.split_id,
        ):
            _require_id(value, "extraction matched decision identity")
        for value in (
            self.parent_artifact_digest,
            self.candidate_artifact_digest,
            self.criteria_digest,
        ):
            _require_digest(value, "extraction matched decision digest")
        accepted = (
            self.status == ExtractionMatchedTrialStatus.ACCEPTED_FOR_ACTIVATION
        )
        if accepted != (self.reason_codes == ("matched_trial_passed",)):
            raise ValueError("matched trial status and reasons disagree")
        if accepted != self.quality_decision.accepted:
            raise ValueError("matched trial status and quality disagree")
        if (
            self.quality_decision.split_id != self.split_id
            or self.quality_decision.criteria_digest != self.criteria_digest
            or self.quality_decision.parent_artifact_id
            != self.parent_runtime_artifact_id
            or self.quality_decision.proposal_artifact_id
            != self.candidate_runtime_artifact_id
        ):
            raise ValueError("matched trial quality decision join mismatch")
        if self.parent_ratios != extraction_ratio_evidence(
            self.quality_decision.parent_metrics
        ) or self.candidate_ratios != extraction_ratio_evidence(
            self.quality_decision.proposal_metrics
        ):
            raise ValueError("matched trial ratio evidence mismatch")
        expected_constraints = _constraint_results(self.quality_decision)
        if self.constraint_results != expected_constraints:
            raise ValueError("matched trial constraint results mismatch")
        if self.reason_codes != (
            ("matched_trial_passed",)
            if accepted
            else self.quality_decision.reason_codes
        ):
            raise ValueError("matched trial reasons differ from quality decision")
        expected = f"extraction-matched.{content_digest(self.identity_payload())[:40]}"
        if self.decision_id != expected:
            raise ValueError("extraction matched trial decision ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "decision_schema": self.decision_schema,
            "status": self.status.value,
            "offline_decision_id": self.offline_decision_id,
            "parent_artifact_id": self.parent_artifact_id,
            "parent_artifact_digest": self.parent_artifact_digest,
            "candidate_artifact_id": self.candidate_artifact_id,
            "candidate_artifact_digest": self.candidate_artifact_digest,
            "parent_runtime_artifact_id": self.parent_runtime_artifact_id,
            "candidate_runtime_artifact_id": self.candidate_runtime_artifact_id,
            "split_id": self.split_id,
            "criteria_digest": self.criteria_digest,
            "quality_decision": self.quality_decision.payload(),
            "parent_ratios": [value.payload() for value in self.parent_ratios],
            "candidate_ratios": [value.payload() for value in self.candidate_ratios],
            "constraint_results": [
                value.payload() for value in self.constraint_results
            ],
            "reason_codes": list(self.reason_codes),
        }

    def payload(self) -> dict[str, object]:
        return {"decision_id": self.decision_id, **self.identity_payload()}

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionMatchedTrialDecision":
        fields = {
            "decision_id",
            "schema_version",
            "decision_schema",
            "status",
            "offline_decision_id",
            "parent_artifact_id",
            "parent_artifact_digest",
            "candidate_artifact_id",
            "candidate_artifact_digest",
            "parent_runtime_artifact_id",
            "candidate_runtime_artifact_id",
            "split_id",
            "criteria_digest",
            "quality_decision",
            "parent_ratios",
            "candidate_ratios",
            "constraint_results",
            "reason_codes",
        }
        if not isinstance(value, Mapping) or set(value) != fields or not all(
            isinstance(value[field], list)
            for field in (
                "parent_ratios",
                "candidate_ratios",
                "constraint_results",
                "reason_codes",
            )
        ):
            raise ValueError("malformed extraction matched trial decision")
        try:
            return cls(
                decision_id=value["decision_id"],
                status=value["status"],
                offline_decision_id=value["offline_decision_id"],
                parent_artifact_id=value["parent_artifact_id"],
                parent_artifact_digest=value["parent_artifact_digest"],
                candidate_artifact_id=value["candidate_artifact_id"],
                candidate_artifact_digest=value["candidate_artifact_digest"],
                parent_runtime_artifact_id=value["parent_runtime_artifact_id"],
                candidate_runtime_artifact_id=value[
                    "candidate_runtime_artifact_id"
                ],
                split_id=value["split_id"],
                criteria_digest=value["criteria_digest"],
                quality_decision=ExtractionValidationDecision.from_payload(
                    value["quality_decision"]
                ),
                parent_ratios=tuple(
                    _ratio_from_payload(item) for item in value["parent_ratios"]
                ),
                candidate_ratios=tuple(
                    _ratio_from_payload(item) for item in value["candidate_ratios"]
                ),
                constraint_results=tuple(
                    ExtractionMatchedConstraintResult.from_payload(item)
                    for item in value["constraint_results"]
                ),
                reason_codes=tuple(value["reason_codes"]),
                decision_schema=value["decision_schema"],
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed extraction matched trial decision") from exc


class ExtractionMatchedTrialEvaluator:
    def evaluate(
        self,
        *,
        parent: ExtractionPromptPolicyArtifact,
        candidate: ExtractionPromptPolicyArtifact,
        offline_decision: ExtractionOfflineValidationDecision,
        split: ExtractionPromptValidationSplit,
        observations: tuple[ExtractionValidationObservation, ...],
        criteria: ExtractionAcceptanceCriteria,
        parent_runtime_artifact_id: str | None = None,
        candidate_runtime_artifact_id: str | None = None,
    ) -> ExtractionMatchedTrialDecision:
        if (
            offline_decision.status
            != ExtractionOfflineDecisionStatus.ACCEPTED_FOR_MATCHED_TRIAL
            or offline_decision.eligible_next_stage != "matched_trial"
        ):
            raise ValueError("matched trial requires offline accepted candidate")
        if (
            offline_decision.parent_artifact_id != parent.artifact_id
            or offline_decision.parent_artifact_digest != parent.artifact_digest
            or offline_decision.candidate_artifact_id != candidate.artifact_id
            or offline_decision.candidate_artifact_digest
            != candidate.artifact_digest
        ):
            raise ValueError("matched trial offline artifact join mismatch")
        if candidate.parent_artifact_id != parent.artifact_id:
            raise ValueError("matched trial candidate parent differs")
        if offline_decision.criteria_digest != criteria.digest:
            raise ValueError("matched trial criteria differ from offline gate")
        if offline_decision.split_id == split.split_id:
            raise ValueError("matched trial must use an independent split")
        if {value.role for value in split.assignments} != set(
            ExtractionValidationSplitRole
        ):
            raise ValueError("matched trial split roles are incomplete")
        parent_runtime_id = parent_runtime_artifact_id or parent.artifact_id
        candidate_runtime_id = candidate_runtime_artifact_id or candidate.artifact_id
        _require_id(parent_runtime_id, "matched parent runtime artifact")
        _require_id(candidate_runtime_id, "matched candidate runtime artifact")
        if parent_runtime_id == candidate_runtime_id:
            raise ValueError("matched trial runtime artifacts must differ")
        offline_quality = offline_decision.quality_decision
        if set(offline_quality.observation_ids) & {
            value.observation_id for value in observations
        } or set(offline_quality.pair_ids) & {
            value.pair_id for value in observations
        }:
            raise ValueError("matched trial reuses offline validation evidence")
        for observation in observations:
            if observation.extraction_artifact_id == parent_runtime_id:
                expected_digest = parent.body_digest
            elif observation.extraction_artifact_id == candidate_runtime_id:
                expected_digest = candidate.body_digest
            else:
                raise ValueError("matched trial observation artifact mismatch")
            if observation.extraction_artifact_digest != expected_digest:
                raise ValueError("matched trial observation body digest mismatch")
        quality = ExtractionPromptMatchedValidator().evaluate(
            split=split,
            observations=observations,
            parent_artifact_id=parent_runtime_id,
            proposal_artifact_id=candidate_runtime_id,
            criteria=criteria,
        )
        status = (
            ExtractionMatchedTrialStatus.ACCEPTED_FOR_ACTIVATION
            if quality.accepted
            else ExtractionMatchedTrialStatus.REJECTED
        )
        reasons = (
            ("matched_trial_passed",)
            if quality.accepted
            else quality.reason_codes
        )
        values = {
            "status": status,
            "offline_decision_id": offline_decision.decision_id,
            "parent_artifact_id": parent.artifact_id,
            "parent_artifact_digest": parent.artifact_digest,
            "candidate_artifact_id": candidate.artifact_id,
            "candidate_artifact_digest": candidate.artifact_digest,
            "parent_runtime_artifact_id": parent_runtime_id,
            "candidate_runtime_artifact_id": candidate_runtime_id,
            "split_id": split.split_id,
            "criteria_digest": criteria.digest,
            "quality_decision": quality,
            "parent_ratios": extraction_ratio_evidence(quality.parent_metrics),
            "candidate_ratios": extraction_ratio_evidence(quality.proposal_metrics),
            "constraint_results": _constraint_results(quality),
            "reason_codes": reasons,
            "decision_schema": EXTRACTION_MATCHED_DECISION_SCHEMA,
            "schema_version": EXTRACTION_MATCHED_ACTIVATION_SCHEMA_VERSION,
        }
        identity = {
            "schema_version": values["schema_version"],
            "decision_schema": values["decision_schema"],
            "status": status.value,
            "offline_decision_id": values["offline_decision_id"],
            "parent_artifact_id": values["parent_artifact_id"],
            "parent_artifact_digest": values["parent_artifact_digest"],
            "candidate_artifact_id": values["candidate_artifact_id"],
            "candidate_artifact_digest": values["candidate_artifact_digest"],
            "parent_runtime_artifact_id": values["parent_runtime_artifact_id"],
            "candidate_runtime_artifact_id": values[
                "candidate_runtime_artifact_id"
            ],
            "split_id": values["split_id"],
            "criteria_digest": values["criteria_digest"],
            "quality_decision": quality.payload(),
            "parent_ratios": [item.payload() for item in values["parent_ratios"]],
            "candidate_ratios": [
                item.payload() for item in values["candidate_ratios"]
            ],
            "constraint_results": [
                item.payload() for item in values["constraint_results"]
            ],
            "reason_codes": list(reasons),
        }
        return ExtractionMatchedTrialDecision(
            decision_id=f"extraction-matched.{content_digest(identity)[:40]}",
            **values,
        )


class JsonExtractionMatchedTrialDecisionStore:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    @contextmanager
    def _lock(self, operation: int):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".extraction-matched.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def put(
        self,
        decision: ExtractionMatchedTrialDecision,
    ) -> tuple[Path, bool]:
        path = self.root / f"{decision.decision_id}.json"
        serialized = canonical_json(decision.payload()) + "\n"
        with self._lock(fcntl.LOCK_EX):
            if path.exists():
                if path.read_text(encoding="utf-8") != serialized:
                    raise ValueError("extraction matched decision conflicts with its ID")
                return path, False
            file_descriptor, temporary = tempfile.mkstemp(
                prefix=".extraction-matched.",
                dir=self.root,
            )
            try:
                with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                directory = os.open(self.root, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            except BaseException:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
                raise
        return path, True

    def get(self, decision_id: str) -> ExtractionMatchedTrialDecision | None:
        _require_id(decision_id, "extraction matched decision ID")
        path = self.root / f"{decision_id}.json"
        with self._lock(fcntl.LOCK_SH):
            if not path.exists():
                return None
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("malformed extraction matched decision JSON") from exc
            decision = ExtractionMatchedTrialDecision.from_payload(raw)
            if decision.decision_id != decision_id:
                raise ValueError("extraction matched decision filename mismatch")
            return decision


class ExtractionRollbackTrigger(StrEnum):
    OPERATOR = "operator"
    SAFETY_VIOLATION = "safety_violation"


@dataclass(frozen=True, slots=True)
class ExtractionRollbackEvidence:
    evidence_id: str
    candidate_artifact_id: str
    candidate_artifact_digest: str
    matched_decision_id: str
    trigger: ExtractionRollbackTrigger
    source_evidence_id: str
    source_evidence_digest: str
    matched_observation_id: str | None
    violation_codes: tuple[str, ...]
    reason_codes: tuple[str, ...]
    evidence_schema: str = EXTRACTION_ROLLBACK_EVIDENCE_SCHEMA
    schema_version: int = EXTRACTION_MATCHED_ACTIVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != EXTRACTION_MATCHED_ACTIVATION_SCHEMA_VERSION
            or self.evidence_schema != EXTRACTION_ROLLBACK_EVIDENCE_SCHEMA
        ):
            raise ValueError("unsupported extraction rollback evidence")
        object.__setattr__(self, "trigger", ExtractionRollbackTrigger(self.trigger))
        for value in (
            self.evidence_id,
            self.candidate_artifact_id,
            self.matched_decision_id,
            self.source_evidence_id,
        ):
            _require_id(value, "extraction rollback identity")
        if self.matched_observation_id is not None:
            _require_id(
                self.matched_observation_id,
                "extraction rollback observation",
            )
        for value in (
            self.candidate_artifact_digest,
            self.source_evidence_digest,
        ):
            _require_digest(value, "extraction rollback digest")
        if self.trigger == ExtractionRollbackTrigger.OPERATOR:
            if (
                self.matched_observation_id is not None
                or self.violation_codes
                or self.reason_codes != ("operator_requested",)
            ):
                raise ValueError("operator rollback evidence is inconsistent")
        elif (
            self.matched_observation_id is None
            or not self.violation_codes
            or self.reason_codes != ("automatic_safety_violation",)
        ):
            raise ValueError("automatic rollback requires safety violations")
        if len(self.violation_codes) != len(set(self.violation_codes)):
            raise ValueError("rollback safety violations are duplicated")
        for value in (*self.violation_codes, *self.reason_codes):
            _require_id(value, "extraction rollback reason")
        expected = f"extraction-rollback.{content_digest(self.identity_payload())[:40]}"
        if self.evidence_id != expected:
            raise ValueError("extraction rollback evidence ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evidence_schema": self.evidence_schema,
            "candidate_artifact_id": self.candidate_artifact_id,
            "candidate_artifact_digest": self.candidate_artifact_digest,
            "matched_decision_id": self.matched_decision_id,
            "trigger": self.trigger.value,
            "source_evidence_id": self.source_evidence_id,
            "source_evidence_digest": self.source_evidence_digest,
            "matched_observation_id": self.matched_observation_id,
            "violation_codes": list(self.violation_codes),
            "reason_codes": list(self.reason_codes),
        }

    def payload(self) -> dict[str, object]:
        return {"evidence_id": self.evidence_id, **self.identity_payload()}

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionRollbackEvidence":
        fields = {
            "evidence_id",
            "schema_version",
            "evidence_schema",
            "candidate_artifact_id",
            "candidate_artifact_digest",
            "matched_decision_id",
            "trigger",
            "source_evidence_id",
            "source_evidence_digest",
            "matched_observation_id",
            "violation_codes",
            "reason_codes",
        }
        if not isinstance(value, Mapping) or set(value) != fields or not all(
            isinstance(value[field], list)
            for field in ("violation_codes", "reason_codes")
        ):
            raise ValueError("malformed extraction rollback evidence")
        try:
            return cls(
                evidence_id=value["evidence_id"],
                candidate_artifact_id=value["candidate_artifact_id"],
                candidate_artifact_digest=value["candidate_artifact_digest"],
                matched_decision_id=value["matched_decision_id"],
                trigger=value["trigger"],
                source_evidence_id=value["source_evidence_id"],
                source_evidence_digest=value["source_evidence_digest"],
                matched_observation_id=value["matched_observation_id"],
                violation_codes=tuple(value["violation_codes"]),
                reason_codes=tuple(value["reason_codes"]),
                evidence_schema=value["evidence_schema"],
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed extraction rollback evidence") from exc

    @classmethod
    def operator_requested(
        cls,
        *,
        candidate: ExtractionPromptPolicyArtifact,
        matched_decision_id: str,
        request_id: str,
        request_digest: str,
    ) -> "ExtractionRollbackEvidence":
        values = {
            "candidate_artifact_id": candidate.artifact_id,
            "candidate_artifact_digest": candidate.artifact_digest,
            "matched_decision_id": matched_decision_id,
            "trigger": ExtractionRollbackTrigger.OPERATOR,
            "source_evidence_id": request_id,
            "source_evidence_digest": request_digest,
            "matched_observation_id": None,
            "violation_codes": (),
            "reason_codes": ("operator_requested",),
            "evidence_schema": EXTRACTION_ROLLBACK_EVIDENCE_SCHEMA,
            "schema_version": EXTRACTION_MATCHED_ACTIVATION_SCHEMA_VERSION,
        }
        identity = {
            **values,
            "trigger": values["trigger"].value,
            "violation_codes": [],
            "reason_codes": ["operator_requested"],
        }
        return cls(
            evidence_id=f"extraction-rollback.{content_digest(identity)[:40]}",
            **values,
        )

    @classmethod
    def safety_violation(
        cls,
        *,
        candidate: ExtractionPromptPolicyArtifact,
        matched_decision_id: str,
        safety: ExtractionValidationSafetyEvidence,
        observation: ExtractionValidationObservation,
    ) -> "ExtractionRollbackEvidence":
        names = (
            "schema_failure",
            "safety_failure",
            "prompt_leakage",
            "native_writer_failure",
        )
        violations = tuple(
            name
            for name, count in zip(names, safety.failure_counts)
            if count > 0
        )
        if (
            not safety.complete
            or not violations
            or observation.variant != ExtractionValidationVariant.PROPOSAL
            or observation.extraction_artifact_id != candidate.artifact_id
            or observation.extraction_artifact_digest != candidate.body_digest
            or observation.failure_counts != safety.failure_counts
        ):
            raise ValueError("automatic rollback requires observed safety failure")
        values = {
            "candidate_artifact_id": candidate.artifact_id,
            "candidate_artifact_digest": candidate.artifact_digest,
            "matched_decision_id": matched_decision_id,
            "trigger": ExtractionRollbackTrigger.SAFETY_VIOLATION,
            "source_evidence_id": safety.evidence_id,
            "source_evidence_digest": safety.audit_digest,
            "matched_observation_id": observation.observation_id,
            "violation_codes": violations,
            "reason_codes": ("automatic_safety_violation",),
            "evidence_schema": EXTRACTION_ROLLBACK_EVIDENCE_SCHEMA,
            "schema_version": EXTRACTION_MATCHED_ACTIVATION_SCHEMA_VERSION,
        }
        identity = {
            **values,
            "trigger": values["trigger"].value,
            "violation_codes": list(violations),
            "reason_codes": ["automatic_safety_violation"],
        }
        return cls(
            evidence_id=f"extraction-rollback.{content_digest(identity)[:40]}",
            **values,
        )


class JsonExtractionRollbackEvidenceStore:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    @contextmanager
    def _lock(self, operation: int):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".extraction-rollback.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def put(self, evidence: ExtractionRollbackEvidence) -> tuple[Path, bool]:
        path = self.root / f"{evidence.evidence_id}.json"
        serialized = canonical_json(evidence.payload()) + "\n"
        with self._lock(fcntl.LOCK_EX):
            if path.exists():
                if path.read_text(encoding="utf-8") != serialized:
                    raise ValueError("extraction rollback evidence conflicts with its ID")
                return path, False
            file_descriptor, temporary = tempfile.mkstemp(
                prefix=".extraction-rollback.",
                dir=self.root,
            )
            try:
                with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                directory = os.open(self.root, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            except BaseException:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
                raise
        return path, True

    def get(self, evidence_id: str) -> ExtractionRollbackEvidence | None:
        _require_id(evidence_id, "extraction rollback evidence ID")
        path = self.root / f"{evidence_id}.json"
        with self._lock(fcntl.LOCK_SH):
            if not path.exists():
                return None
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("malformed extraction rollback evidence JSON") from exc
            evidence = ExtractionRollbackEvidence.from_payload(raw)
            if evidence.evidence_id != evidence_id:
                raise ValueError("extraction rollback evidence filename mismatch")
            return evidence


class ExtractionMatchedActivationCoordinator:
    def __init__(
        self,
        policy_store: JsonExtractionPolicyStore,
        decision_store: JsonExtractionMatchedTrialDecisionStore,
        rollback_store: JsonExtractionRollbackEvidenceStore,
        revocation_registry: JsonRevocationRegistry | None = None,
    ) -> None:
        # Activation is a production state transition.  Without an explicit
        # owner-controlled revocation registry, a stale parent/candidate
        # artifact cannot be distinguished from an active one; fail closed
        # before any ACTIVE pointer or rollback evidence is written.
        if revocation_registry is None:
            raise ValueError(
                "extraction activation requires a revocation registry"
            )
        if not isinstance(revocation_registry, JsonRevocationRegistry):
            raise TypeError("extraction activation revocation registry has the wrong type")
        self.policy_store = policy_store
        self.decision_store = decision_store
        self.rollback_store = rollback_store
        self.revocation_registry = revocation_registry

    @staticmethod
    def _transition_id(action: str, identity: str) -> str:
        return (
            f"extraction-transition.{action}."
            f"{content_digest(identity)[:24]}"
        )

    def apply(
        self,
        *,
        parent: ExtractionPromptPolicyArtifact,
        candidate: ExtractionPromptPolicyArtifact,
        offline_decision: ExtractionOfflineValidationDecision,
        decision: ExtractionMatchedTrialDecision,
        split: ExtractionPromptValidationSplit,
        observations: tuple[ExtractionValidationObservation, ...],
        criteria: ExtractionAcceptanceCriteria,
        parent_runtime_artifact_id: str | None = None,
        candidate_runtime_artifact_id: str | None = None,
    ) -> ExtractionPolicyState:
        for artifact in (parent, candidate):
            self.revocation_registry.assert_active(
                artifact_id=artifact.artifact_id,
                artifact_schema_version=artifact.schema_version,
                artifact_digest=artifact.artifact_digest,
                evidence_plane=EvidencePlane.PURE_PROCESS,
                evidence_source=EvidenceSourceKind.RUNTIME_OBSERVATION,
            )
        if parent != self.policy_store.trusted_root:
            raise ValueError("first extraction activation requires trusted root parent")
        replay = ExtractionMatchedTrialEvaluator().evaluate(
            parent=parent,
            candidate=candidate,
            offline_decision=offline_decision,
            split=split,
            observations=observations,
            criteria=criteria,
            parent_runtime_artifact_id=parent_runtime_artifact_id,
            candidate_runtime_artifact_id=candidate_runtime_artifact_id,
        )
        if replay != decision:
            raise ValueError("extraction matched activation replay mismatch")
        self.decision_store.put(decision)
        current, _ = self.policy_store.register(candidate)
        accepted = (
            decision.status
            == ExtractionMatchedTrialStatus.ACCEPTED_FOR_ACTIVATION
        )
        action = "matched-activate" if accepted else "matched-reject"
        target = (
            ExtractionPolicyState.ACTIVE
            if accepted
            else ExtractionPolicyState.REJECTED
        )
        transition_id = self._transition_id(action, decision.decision_id)
        reason_code = (
            "matched_trial_passed" if accepted else decision.reason_codes[0]
        )
        if current.state == target:
            if (
                current.last_transition_id != transition_id
                or current.reason_code != reason_code
            ):
                raise ValueError("extraction matched decision conflicts with lifecycle")
            return current.state
        if current.state != ExtractionPolicyState.PROPOSAL:
            raise ValueError("extraction matched activation requires proposal state")
        record, _ = self.policy_store.transition(
            candidate.artifact_id,
            to_state=target,
            transition_id=transition_id,
            reason_code=reason_code,
        )
        return record.state

    def rollback(
        self,
        *,
        candidate: ExtractionPromptPolicyArtifact,
        evidence: ExtractionRollbackEvidence,
    ) -> ExtractionPolicyState:
        if (
            evidence.candidate_artifact_id != candidate.artifact_id
            or evidence.candidate_artifact_digest != candidate.artifact_digest
        ):
            raise ValueError("extraction rollback candidate differs")
        decision = self.decision_store.get(evidence.matched_decision_id)
        if (
            decision is None
            or decision.status
            != ExtractionMatchedTrialStatus.ACCEPTED_FOR_ACTIVATION
            or decision.candidate_artifact_id != candidate.artifact_id
        ):
            raise ValueError("extraction rollback lacks accepted matched decision")
        self.rollback_store.put(evidence)
        snapshot = self.policy_store.snapshot()
        current = next(
            (
                record
                for record in snapshot.records
                if record.artifact_id == candidate.artifact_id
            ),
            None,
        )
        if current is None:
            raise ValueError("extraction rollback candidate is not registered")
        transition_id = self._transition_id("rollback", evidence.evidence_id)
        reason_code = (
            "automatic_safety_rollback"
            if evidence.trigger == ExtractionRollbackTrigger.SAFETY_VIOLATION
            else "operator_rollback"
        )
        if current.state == ExtractionPolicyState.ROLLED_BACK:
            if (
                current.last_transition_id != transition_id
                or current.reason_code != reason_code
            ):
                raise ValueError("extraction rollback evidence conflicts with lifecycle")
            return current.state
        if current.state != ExtractionPolicyState.ACTIVE:
            raise ValueError("extraction rollback requires active candidate")
        record, _ = self.policy_store.transition(
            candidate.artifact_id,
            to_state=ExtractionPolicyState.ROLLED_BACK,
            transition_id=transition_id,
            reason_code=reason_code,
        )
        if self.policy_store.active_or_root() != self.policy_store.trusted_root:
            raise ValueError("extraction rollback did not restore trusted root")
        return record.state
