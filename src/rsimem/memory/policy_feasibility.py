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
from .extraction_feedback import (
    ExtractionFeedbackExample,
    ExtractionFeedbackLabel,
    ExtractionFeedbackLevel,
)


def _require_nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _normalize_strings(values: object, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise ValueError(f"{name} must be a list or tuple of strings")
    result = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{name} must contain unique non-empty strings")
    if (not allow_empty and not result) or len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique non-empty strings")
    return result
from .extraction_optimizer_corpus import (
    ExtractionOptimizerCorpusExample,
    OptimizerComponentOwnership,
)
from .extraction_prompt_optimizer import (
    ExtractionOptimizerDecision,
    ExtractionOptimizerResult,
)


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


_LAYER_BENEFIT_MECHANISMS = {
    PolicyLayer.TRIGGER: (
        "boundary_eligibility",
        "Trigger variation changes whether downstream formation is eligible.",
    ),
    PolicyLayer.SOURCE_SELECTION: (
        "formation_input_scope",
        "Source variation changes which completed context enters formation.",
    ),
    PolicyLayer.EXTRACTION: (
        "candidate_fact_set",
        "Extraction variation changes the candidate durable-fact set.",
    ),
    PolicyLayer.ADMISSION: (
        "mutation_admission",
        "Admission variation changes mutation kind, target, or acceptance.",
    ),
    PolicyLayer.COMMIT: (
        "mutation_scheduling",
        "Commit variation changes mutation scheduling and receipt identity.",
    ),
    PolicyLayer.EXPOSURE: (
        "future_context_exposure",
        "Exposure variation changes which artifacts enter future context.",
    ),
}


@dataclass(frozen=True, slots=True)
class LayerBenefitExplanation:
    """Content-free explanation of what a layer intervention can change.

    This is a mechanism explanation, not a claim that a task improved.  The
    outcome status is kept explicit so unresolved/censored cases cannot be
    mistaken for a benefit or harm signal.
    """

    target_layer: PolicyLayer
    mechanism_code: str
    outcome: FeasibilityOutcome
    outcome_status: str

    def __post_init__(self) -> None:
        layer = PolicyLayer(self.target_layer)
        object.__setattr__(self, "target_layer", layer)
        outcome = FeasibilityOutcome(self.outcome)
        object.__setattr__(self, "outcome", outcome)
        mechanism = _LAYER_BENEFIT_MECHANISMS.get(layer)
        if mechanism is None or self.mechanism_code != mechanism[0]:
            raise ValueError("benefit explanation mechanism does not match target layer")
        expected_status = (
            "resolved" if outcome in {
                FeasibilityOutcome.USEFUL,
                FeasibilityOutcome.HARMFUL,
                FeasibilityOutcome.MISSED,
            } else "unresolved"
        )
        if self.outcome_status != expected_status:
            raise ValueError("benefit explanation outcome status is invalid")

    @classmethod
    def create(
        cls,
        *,
        target_layer: PolicyLayer,
        outcome: FeasibilityOutcome,
    ) -> "LayerBenefitExplanation":
        layer = PolicyLayer(target_layer)
        resolved = FeasibilityOutcome(outcome) in {
            FeasibilityOutcome.USEFUL,
            FeasibilityOutcome.HARMFUL,
            FeasibilityOutcome.MISSED,
        }
        return cls(
            target_layer=layer,
            mechanism_code=_LAYER_BENEFIT_MECHANISMS[layer][0],
            outcome=FeasibilityOutcome(outcome),
            outcome_status="resolved" if resolved else "unresolved",
        )

    @property
    def summary(self) -> str:
        return _LAYER_BENEFIT_MECHANISMS[self.target_layer][1]

    def payload(self) -> dict[str, object]:
        return {
            "target_layer": self.target_layer.value,
            "mechanism_code": self.mechanism_code,
            "outcome": self.outcome.value,
            "outcome_status": self.outcome_status,
        }

    @classmethod
    def from_payload(cls, value: object) -> "LayerBenefitExplanation":
        fields = {"target_layer", "mechanism_code", "outcome", "outcome_status"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("malformed layer benefit explanation")
        try:
            result = cls(
                target_layer=value["target_layer"],
                mechanism_code=value["mechanism_code"],
                outcome=value["outcome"],
                outcome_status=value["outcome_status"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed layer benefit explanation") from exc
        if result.payload() != dict(value):
            raise ValueError("non-canonical layer benefit explanation")
        return result


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
    def complete_harmful(self) -> bool:
        """Whether an observed harm has a complete opportunity/use/outcome chain."""

        return self.complete_useful

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


def feedback_chain_from_extraction_example(
    example: ExtractionFeedbackExample,
) -> FeedbackChain:
    """Project one strict extraction example into the feasibility chain.

    Only resolved extraction-set labels are projected.  Fact-level labels and
    unresolved/censored/harmful examples remain diagnostics because their
    chain cannot, by itself, prove a policy-layer outcome.
    """

    if not isinstance(example, ExtractionFeedbackExample):
        raise TypeError("feedback chain source must be an extraction feedback example")
    if not example.primary or example.level is not ExtractionFeedbackLevel.EXTRACTION_SET:
        return FeedbackChain()
    if example.label is ExtractionFeedbackLabel.USEFUL:
        if any(value is None for value in (
            example.opportunity_operation_id,
            example.use_operation_id,
            example.outcome_operation_id,
        )):
            return FeedbackChain()
        return FeedbackChain(
            opportunity_id=example.future_opportunity_id,
            use_id=example.use_operation_id,
            outcome_id=example.outcome_operation_id,
        )
    if example.label is ExtractionFeedbackLabel.MISSED:
        if any(value is None for value in (
            example.opportunity_operation_id,
            example.outcome_operation_id,
        )):
            return FeedbackChain()
        return FeedbackChain(
            source_id=example.source_id,
            demand_id=example.future_opportunity_id,
            # The corpus carries the absence attribution on the outcome
            # operation.  Keep a distinct logical absence identity so the
            # resulting hypothesis evidence tuple remains duplicate-free.
            absence_id=f"absence.{example.outcome_operation_id}",
            outcome_id=example.outcome_operation_id,
        )
    return FeedbackChain()


def build_extraction_feedback_interventions(
    examples: Iterable[ExtractionFeedbackExample],
    *,
    parent: PolicyReplayResult,
    candidate: PolicyReplayResult,
    parent_artifact: PolicyArtifactIdentity,
    candidate_artifact: PolicyArtifactIdentity,
    case_id_prefix: str = "case.feedback",
) -> tuple["LayerIntervention", ...]:
    """Create extraction interventions from primary strict-feedback examples.

    Non-primary fact/source projections are deliberately ignored here; they
    remain available to the owner-controlled corpus and diagnostics but cannot
    be counted as independent policy opportunities.
    """

    if not isinstance(case_id_prefix, str) or not case_id_prefix.strip():
        raise ValueError("feasibility case ID prefix must not be empty")
    result: list[LayerIntervention] = []
    seen: set[str] = set()
    for example in examples:
        if not isinstance(example, ExtractionFeedbackExample):
            raise TypeError("extraction feasibility examples have the wrong type")
        if not example.primary:
            continue
        if example.example_id in seen:
            raise ValueError("extraction feasibility example IDs must be unique")
        seen.add(example.example_id)
        result.append(LayerIntervention.from_extraction_feedback(
            case_id=f"{case_id_prefix}.{example.example_id}",
            parent=parent,
            candidate=candidate,
            parent_artifact=parent_artifact,
            candidate_artifact=candidate_artifact,
            example=example,
        ))
    if not result:
        raise ValueError("extraction feasibility requires at least one primary example")
    return tuple(result)


def build_optimizer_corpus_interventions(
    examples: Iterable[ExtractionOptimizerCorpusExample],
    *,
    parent: PolicyReplayResult,
    candidate: PolicyReplayResult,
    parent_artifact: PolicyArtifactIdentity,
    candidate_artifact: PolicyArtifactIdentity,
    case_id_prefix: str = "case.optimizer",
) -> tuple["LayerIntervention", ...]:
    """Project primary optimizer-corpus examples into extraction cases.

    The corpus is owner-controlled and content-bearing, but this projection is
    content-free: only stable example/source/operation IDs and the already
    resolved label are copied into the feasibility contract.  Examples owned by
    retrieval/application/outcome components cannot create extraction reward.
    """

    if not isinstance(case_id_prefix, str) or not case_id_prefix.strip():
        raise ValueError("feasibility case ID prefix must not be empty")
    result: list[LayerIntervention] = []
    seen: set[str] = set()
    for example in examples:
        if not isinstance(example, ExtractionOptimizerCorpusExample):
            raise TypeError("optimizer corpus examples have the wrong type")
        if not example.primary:
            continue
        if example.example_id in seen:
            raise ValueError("optimizer feasibility example IDs must be unique")
        seen.add(example.example_id)
        if example.label in {
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.HARMFUL,
            ExtractionFeedbackLabel.MISSED,
        } and example.component_ownership is not OptimizerComponentOwnership.EXTRACTION:
            raise ValueError("resolved optimizer example is not extraction-owned")
        delayed = example.delayed_evidence
        if example.label is ExtractionFeedbackLabel.USEFUL:
            chain = FeedbackChain(
                opportunity_id=delayed.future_opportunity_id,
                use_id=delayed.use_operation_id,
                outcome_id=delayed.outcome_operation_id,
            )
        elif example.label is ExtractionFeedbackLabel.MISSED:
            chain = FeedbackChain(
                source_id=example.audit_join.source_record_id,
                demand_id=delayed.future_opportunity_id,
                # The corpus stores the absence-attributed outcome operation,
                # not a separate absence receipt.  Derive a distinct stable
                # absence identity instead of duplicating the outcome ID in
                # the hypothesis evidence tuple.
                absence_id=(
                    f"absence.{delayed.outcome_operation_id}"
                    if delayed.outcome_operation_id is not None
                    else None
                ),
                outcome_id=delayed.outcome_operation_id,
            )
        else:
            chain = FeedbackChain()
        result.append(LayerIntervention(
            case_id=f"{case_id_prefix}.{example.example_id}",
            target_layer=PolicyLayer.EXTRACTION,
            parent=parent,
            candidate=candidate,
            parent_artifact=parent_artifact,
            candidate_artifact=candidate_artifact,
            process_signal=True,
            outcome={
                ExtractionFeedbackLabel.USEFUL: FeasibilityOutcome.USEFUL,
                ExtractionFeedbackLabel.HARMFUL: FeasibilityOutcome.HARMFUL,
                ExtractionFeedbackLabel.MISSED: FeasibilityOutcome.MISSED,
                ExtractionFeedbackLabel.UNRESOLVED: FeasibilityOutcome.UNRESOLVED,
                ExtractionFeedbackLabel.CENSORED: FeasibilityOutcome.CENSORED,
            }[example.label],
            feedback=chain,
            reason_codes=tuple(example.reason_codes) or ("optimizer_feedback_projected",),
        ))
    if not result:
        raise ValueError("optimizer feasibility requires at least one primary example")
    return tuple(result)


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
            _require_nonempty(value, name)
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
            values = _normalize_strings(
                values,
                name,
                allow_empty=name in {
                    "parent execution receipts",
                    "candidate execution receipts",
                },
            )
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

    @classmethod
    def from_payload(cls, value: object) -> "ProcessFeedback":
        fields = {
            "feedback_id", "event_id", "source_revision", "target_layer",
            "parent_decision_id", "candidate_decision_id",
            "parent_execution_receipt_ids", "candidate_execution_receipt_ids",
            "observed_before_digest", "observed_after_digest", "reason_codes",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("malformed process feedback")
        for field in (
            "parent_execution_receipt_ids",
            "candidate_execution_receipt_ids",
            "reason_codes",
        ):
            if not isinstance(value[field], list):
                raise ValueError("malformed process feedback")
        try:
            result = cls(
                value["feedback_id"],
                value["event_id"],
                value["source_revision"],
                value["target_layer"],
                value["parent_decision_id"],
                value["candidate_decision_id"],
                tuple(value["parent_execution_receipt_ids"]),
                tuple(value["candidate_execution_receipt_ids"]),
                value["observed_before_digest"],
                value["observed_after_digest"],
                tuple(value["reason_codes"]),
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("malformed process feedback") from exc
        identity = {
            "event_id": result.event_id,
            "source_revision": result.source_revision,
            "target_layer": result.target_layer.value,
            "parent_decision_id": result.parent_decision_id,
            "candidate_decision_id": result.candidate_decision_id,
            "parent_execution_receipt_ids": list(result.parent_execution_receipt_ids),
            "candidate_execution_receipt_ids": list(result.candidate_execution_receipt_ids),
            "observed_before_digest": result.observed_before_digest,
            "observed_after_digest": result.observed_after_digest,
            "reason_codes": list(result.reason_codes),
        }
        if result.feedback_id != f"process-feedback.{content_digest(identity)[:40]}":
            raise ValueError("process feedback ID mismatch")
        return result


@dataclass(frozen=True, slots=True)
class PolicyHypothesis:
    """A constrained N+1 proposal derived from past content-free feedback."""

    hypothesis_id: str
    parent_artifact_id: str
    candidate_artifact_id: str
    target_layer: PolicyLayer
    feedback_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_layer", PolicyLayer(self.target_layer))
        for value, name in (
            (self.hypothesis_id, "policy hypothesis ID"),
            (self.parent_artifact_id, "hypothesis parent artifact ID"),
            (self.candidate_artifact_id, "hypothesis candidate artifact ID"),
        ):
            _require_nonempty(value, name)
        if self.parent_artifact_id == self.candidate_artifact_id:
            raise ValueError("hypothesis parent and candidate artifacts must differ")
        for values, name in (
            (self.feedback_ids, "hypothesis feedback IDs"),
            (self.reason_codes, "hypothesis reason codes"),
        ):
            values = _normalize_strings(values, name)
            attribute = {
                "hypothesis feedback IDs": "feedback_ids",
                "hypothesis reason codes": "reason_codes",
            }[name]
            object.__setattr__(self, attribute, values)

    @classmethod
    def create(
        cls,
        *,
        parent_artifact_id: str,
        candidate_artifact_id: str,
        target_layer: PolicyLayer,
        feedback_ids: Sequence[str],
        reason_codes: Sequence[str] = ("feedback_bound_candidate",),
    ) -> "PolicyHypothesis":
        identity = {
            "parent_artifact_id": parent_artifact_id,
            "candidate_artifact_id": candidate_artifact_id,
            "target_layer": PolicyLayer(target_layer).value,
            "feedback_ids": list(feedback_ids),
            "reason_codes": list(reason_codes),
        }
        return cls(
            f"hypothesis.{content_digest(identity)[:40]}",
            parent_artifact_id,
            candidate_artifact_id,
            target_layer,
            tuple(feedback_ids),
            tuple(reason_codes),
        )

    def payload(self) -> dict[str, object]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "parent_artifact_id": self.parent_artifact_id,
            "candidate_artifact_id": self.candidate_artifact_id,
            "target_layer": self.target_layer.value,
            "feedback_ids": list(self.feedback_ids),
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_payload(cls, value: object) -> "PolicyHypothesis":
        fields = {
            "hypothesis_id", "parent_artifact_id", "candidate_artifact_id",
            "target_layer", "feedback_ids", "reason_codes",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("malformed policy hypothesis")
        for field in ("feedback_ids", "reason_codes"):
            if not isinstance(value[field], list):
                raise ValueError("malformed policy hypothesis")
        try:
            result = cls(
                value["hypothesis_id"],
                value["parent_artifact_id"],
                value["candidate_artifact_id"],
                value["target_layer"],
                tuple(value["feedback_ids"]),
                tuple(value["reason_codes"]),
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("malformed policy hypothesis") from exc
        expected = cls.create(
            parent_artifact_id=result.parent_artifact_id,
            candidate_artifact_id=result.candidate_artifact_id,
            target_layer=result.target_layer,
            feedback_ids=result.feedback_ids,
            reason_codes=result.reason_codes,
        )
        if result.hypothesis_id != expected.hypothesis_id:
            raise ValueError("policy hypothesis ID mismatch")
        return result


class OptimizerHypothesisDecision(StrEnum):
    NO_PROPOSAL = "NO_PROPOSAL"
    PROPOSE = "PROPOSE"


@dataclass(frozen=True, slots=True)
class OptimizerHypothesisProjection:
    """Content-free projection of one optimizer result into N+1 evidence."""

    projection_id: str
    result_id: str
    request_id: str
    decision: OptimizerHypothesisDecision
    target_layer: PolicyLayer
    parent_artifact_id: str
    candidate_artifact_id: str | None
    corpus_id: str
    corpus_digest: str
    evidence_example_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", OptimizerHypothesisDecision(self.decision))
        object.__setattr__(self, "target_layer", PolicyLayer(self.target_layer))
        for value, name in (
            (self.projection_id, "optimizer projection ID"),
            (self.result_id, "optimizer result ID"),
            (self.request_id, "optimizer request ID"),
            (self.parent_artifact_id, "optimizer parent artifact ID"),
            (self.corpus_id, "optimizer corpus ID"),
            (self.corpus_digest, "optimizer corpus digest"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if self.candidate_artifact_id is not None:
            _require_nonempty(self.candidate_artifact_id, "optimizer candidate artifact ID")
        object.__setattr__(self, "evidence_example_ids", _normalize_strings(
            self.evidence_example_ids,
            "optimizer projection evidence IDs",
            allow_empty=True,
        ))
        object.__setattr__(self, "reason_codes", _normalize_strings(
            self.reason_codes,
            "optimizer projection reason codes",
        ))
        if self.decision is OptimizerHypothesisDecision.NO_PROPOSAL and self.candidate_artifact_id is not None:
            raise ValueError("NO_PROPOSAL projection cannot carry a candidate")
        if self.decision is OptimizerHypothesisDecision.PROPOSE and self.candidate_artifact_id is None:
            raise ValueError("PROPOSE projection requires a candidate")
        expected = f"optimizer-hypothesis.{content_digest(self.identity_payload())[:40]}"
        if self.projection_id != expected:
            raise ValueError("optimizer projection ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "result_id": self.result_id,
            "request_id": self.request_id,
            "decision": self.decision.value,
            "target_layer": self.target_layer.value,
            "parent_artifact_id": self.parent_artifact_id,
            "candidate_artifact_id": self.candidate_artifact_id,
            "corpus_id": self.corpus_id,
            "corpus_digest": self.corpus_digest,
            "evidence_example_ids": list(self.evidence_example_ids),
            "reason_codes": list(self.reason_codes),
        }

    def payload(self) -> dict[str, object]:
        return {"projection_id": self.projection_id, **self.identity_payload()}

    @classmethod
    def from_payload(cls, value: object) -> "OptimizerHypothesisProjection":
        fields = {
            "projection_id", "result_id", "request_id", "decision",
            "target_layer", "parent_artifact_id", "candidate_artifact_id",
            "corpus_id", "corpus_digest", "evidence_example_ids", "reason_codes",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("malformed optimizer hypothesis projection")
        for field in ("evidence_example_ids", "reason_codes"):
            if not isinstance(value[field], list):
                raise ValueError("malformed optimizer hypothesis projection")
        try:
            result = cls(
                value["projection_id"],
                value["result_id"],
                value["request_id"],
                value["decision"],
                value["target_layer"],
                value["parent_artifact_id"],
                value["candidate_artifact_id"],
                value["corpus_id"],
                value["corpus_digest"],
                tuple(value["evidence_example_ids"]),
                tuple(value["reason_codes"]),
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("malformed optimizer hypothesis projection") from exc
        if result.projection_id != f"optimizer-hypothesis.{content_digest(result.identity_payload())[:40]}":
            raise ValueError("optimizer projection ID mismatch")
        return result


def project_optimizer_result(
    result: ExtractionOptimizerResult,
    corpus: "object",
    *,
    parent_artifact_id: str,
    target_layer: PolicyLayer = PolicyLayer.EXTRACTION,
) -> OptimizerHypothesisProjection:
    """Validate a constrained optimizer result without exposing corpus text."""

    from .extraction_optimizer_corpus import ExtractionOptimizerCorpus

    if not isinstance(result, ExtractionOptimizerResult):
        raise TypeError("optimizer projection result has the wrong type")
    if not isinstance(corpus, ExtractionOptimizerCorpus):
        raise TypeError("optimizer projection corpus has the wrong type")
    target = PolicyLayer(target_layer)
    if target is not PolicyLayer.EXTRACTION:
        raise ValueError("current optimizer projection only supports extraction")
    request = result.request
    if request.parent_artifact_id != parent_artifact_id:
        raise ValueError("optimizer result parent artifact differs")
    if request.corpus_id != corpus.corpus_id or request.corpus_digest != corpus.corpus_digest:
        raise ValueError("optimizer result corpus identity differs")
    actionable = {
        example.example_id
        for example in corpus.examples
        if example.primary
        and example.label in {
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.HARMFUL,
            ExtractionFeedbackLabel.MISSED,
        }
        and example.component_ownership is OptimizerComponentOwnership.EXTRACTION
        and example.attribution_confidence.value in {"high", "medium"}
    }
    cited = tuple(
        example_id
        for edit in result.edits
        for example_id in edit.evidence_example_ids
    )
    if len(cited) != len(set(cited)):
        raise ValueError("optimizer result cites duplicate evidence IDs")
    if not set(cited).issubset(actionable):
        raise ValueError("optimizer result cites ineligible evidence")
    candidate_id = None
    if result.decision is ExtractionOptimizerDecision.PROPOSE:
        if result.candidate is None:
            raise ValueError("optimizer proposal has no candidate artifact")
        if result.candidate.parent_artifact_id != parent_artifact_id:
            raise ValueError("optimizer candidate parent artifact differs")
        candidate_id = result.candidate.artifact_id
        if candidate_id == parent_artifact_id:
            raise ValueError("optimizer candidate artifact must differ from parent")
    elif result.candidate is not None or result.edits:
        raise ValueError("NO_PROPOSAL result carries candidate data")
    identity = {
        "result_id": result.result_id,
        "request_id": request.request_id,
        "decision": OptimizerHypothesisDecision(result.decision.value).value,
        "target_layer": target.value,
        "parent_artifact_id": parent_artifact_id,
        "candidate_artifact_id": candidate_id,
        "corpus_id": corpus.corpus_id,
        "corpus_digest": corpus.corpus_digest,
        "evidence_example_ids": list(cited),
        "reason_codes": list(result.reason_codes),
    }
    return OptimizerHypothesisProjection(
        projection_id=f"optimizer-hypothesis.{content_digest(identity)[:40]}",
        result_id=result.result_id,
        request_id=request.request_id,
        decision=result.decision.value,
        target_layer=target,
        parent_artifact_id=parent_artifact_id,
        candidate_artifact_id=candidate_id,
        corpus_id=corpus.corpus_id,
        corpus_digest=corpus.corpus_digest,
        evidence_example_ids=cited,
        reason_codes=result.reason_codes,
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
    process_feedback: ProcessFeedback | None = None
    hypothesis: PolicyHypothesis | None = None
    benefit_explanation: LayerBenefitExplanation | None = None

    @classmethod
    def from_extraction_feedback(
        cls,
        *,
        case_id: str,
        parent: PolicyReplayResult,
        candidate: PolicyReplayResult,
        parent_artifact: PolicyArtifactIdentity,
        candidate_artifact: PolicyArtifactIdentity,
        example: ExtractionFeedbackExample,
        process_signal: bool = True,
    ) -> "LayerIntervention":
        """Build an extraction intervention from a strict feedback example."""

        if not isinstance(example, ExtractionFeedbackExample):
            raise TypeError("extraction feasibility requires a feedback example")
        if not example.primary or example.level is not ExtractionFeedbackLevel.EXTRACTION_SET:
            raise ValueError("feasibility requires the primary extraction-set example")
        outcome = {
            ExtractionFeedbackLabel.USEFUL: FeasibilityOutcome.USEFUL,
            ExtractionFeedbackLabel.HARMFUL: FeasibilityOutcome.HARMFUL,
            ExtractionFeedbackLabel.MISSED: FeasibilityOutcome.MISSED,
            ExtractionFeedbackLabel.UNRESOLVED: FeasibilityOutcome.UNRESOLVED,
            ExtractionFeedbackLabel.CENSORED: FeasibilityOutcome.CENSORED,
        }[example.label]
        return cls(
            case_id=case_id,
            target_layer=PolicyLayer.EXTRACTION,
            parent=parent,
            candidate=candidate,
            parent_artifact=parent_artifact,
            candidate_artifact=candidate_artifact,
            process_signal=process_signal,
            outcome=outcome,
            feedback=feedback_chain_from_extraction_example(example),
            reason_codes=tuple(example.reason_codes) or ("feedback_projected",),
        )

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
        try:
            reasons = _normalize_strings(
                self.reason_codes,
                "feasibility reason codes",
            )
        except ValueError as exc:
            raise ValueError("feasibility reason codes are required and unique") from exc
        # A caller may construct a provisional useful/missed label before all
        # delayed evidence has arrived.  Never let that incomplete label enter
        # a reward denominator: fail closed to unresolved and retain an
        # explicit diagnostic reason for the census/audit layer.
        if outcome == FeasibilityOutcome.USEFUL and not self.feedback.complete_useful:
            outcome = FeasibilityOutcome.UNRESOLVED
            if "incomplete_useful_feedback" not in reasons:
                reasons = (*reasons, "incomplete_useful_feedback")
        elif outcome == FeasibilityOutcome.HARMFUL and not self.feedback.complete_harmful:
            # Harm is an outcome claim too.  Without an attributable chain it
            # must remain unresolved rather than becoming a negative reward.
            outcome = FeasibilityOutcome.UNRESOLVED
            if "incomplete_harmful_feedback" not in reasons:
                reasons = (*reasons, "incomplete_harmful_feedback")
        elif outcome == FeasibilityOutcome.MISSED and not self.feedback.complete_missed:
            outcome = FeasibilityOutcome.UNRESOLVED
            if "incomplete_missed_feedback" not in reasons:
                reasons = (*reasons, "incomplete_missed_feedback")
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "reason_codes", reasons)
        explanation = self.benefit_explanation
        if explanation is None:
            explanation = LayerBenefitExplanation.create(
                target_layer=self.target_layer,
                outcome=outcome,
            )
        if not isinstance(explanation, LayerBenefitExplanation):
            raise ValueError("benefit explanation has the wrong type")
        if (
            explanation.target_layer != self.target_layer
            or explanation.outcome != outcome
        ):
            raise ValueError("benefit explanation does not match intervention")
        object.__setattr__(self, "benefit_explanation", explanation)
        self._validate_feedback(outcome)
        if self.action_changed is False:
            raise ValueError("candidate must change the target-layer decision fingerprint")
        if self.process_signal:
            process_feedback = self.process_feedback or self._derive_process_feedback()
            self._validate_process_feedback(process_feedback)
            object.__setattr__(self, "process_feedback", process_feedback)
        elif self.process_feedback is not None:
            raise ValueError("process feedback cannot be present when process_signal is false")
        if self.process_signal:
            hypothesis = self.hypothesis or self._derive_hypothesis()
            self._validate_hypothesis(hypothesis)
            object.__setattr__(self, "hypothesis", hypothesis)
        elif self.hypothesis is not None:
            raise ValueError("policy hypothesis requires process signal")

    def _validate_feedback(self, outcome: FeasibilityOutcome) -> None:
        if outcome == FeasibilityOutcome.USEFUL and not self.feedback.complete_useful:
            raise ValueError("useful feedback requires opportunity/use/outcome chain")
        if outcome == FeasibilityOutcome.HARMFUL and not self.feedback.complete_harmful:
            raise ValueError("harmful feedback requires opportunity/use/outcome chain")
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

    def _derive_hypothesis(self) -> PolicyHypothesis:
        feedback_ids = self.feedback.ids
        if not feedback_ids and self.process_feedback is not None:
            feedback_ids = (self.process_feedback.feedback_id,)
        return PolicyHypothesis.create(
            parent_artifact_id=self.parent_artifact.artifact_id,
            candidate_artifact_id=self.candidate_artifact.artifact_id,
            target_layer=self.target_layer,
            feedback_ids=feedback_ids,
        )

    def _validate_hypothesis(self, value: PolicyHypothesis) -> None:
        if not isinstance(value, PolicyHypothesis):
            raise ValueError("policy hypothesis has the wrong type")
        if (
            value.parent_artifact_id != self.parent_artifact.artifact_id
            or value.candidate_artifact_id != self.candidate_artifact.artifact_id
            or value.target_layer != self.target_layer
        ):
            raise ValueError("policy hypothesis does not match intervention artifacts")
        allowed_feedback = set(self.feedback.ids)
        if self.process_feedback is not None:
            allowed_feedback.add(self.process_feedback.feedback_id)
        if not set(value.feedback_ids).issubset(allowed_feedback):
            raise ValueError("policy hypothesis feedback is not bound to intervention")

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
            "benefit_explanation": self.benefit_explanation.payload(),
            "intervention_fingerprint": self.intervention_fingerprint,
            "process_feedback": (
                self.process_feedback.payload()
                if self.process_feedback is not None
                else None
            ),
            "hypothesis": self.hypothesis.payload() if self.hypothesis else None,
        }


FEASIBILITY_EVIDENCE_SCHEMA_VERSION = 3
POLICY_FEASIBILITY_REPORT_SCHEMA_VERSION = 2


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
            "reason_codes", "benefit_explanation", "intervention_fingerprint", "process_feedback",
            "hypothesis",
        }
        if set(payload) != required:
            raise ValueError("malformed feasibility replay payload")
        if not isinstance(payload["case_id"], str) or not payload["case_id"].strip():
            raise ValueError("feasibility replay case ID must not be empty")
        if payload["target_layer"] not in {layer.value for layer in PolicyLayer}:
            raise ValueError("feasibility replay target layer is invalid")
        process_feedback = payload["process_feedback"]
        if process_feedback is not None:
            ProcessFeedback.from_payload(process_feedback)
        if payload["process_signal"] is True and process_feedback is None:
            raise ValueError("process signal requires process feedback")
        if payload["process_signal"] is False and process_feedback is not None:
            raise ValueError("process feedback requires process signal")
        hypothesis = payload["hypothesis"]
        if hypothesis is not None:
            PolicyHypothesis.from_payload(hypothesis)
        if payload["process_signal"] is True and hypothesis is None:
            raise ValueError("process signal requires a policy hypothesis")
        if payload["process_signal"] is False and hypothesis is not None:
            raise ValueError("policy hypothesis requires process signal")
        for field in ("parent_decision_ids", "candidate_decision_ids", "feedback_ids", "reason_codes"):
            values = payload[field]
            if not isinstance(values, list):
                raise ValueError("feasibility replay ID lists are invalid")
            try:
                _normalize_strings(
                    values,
                    f"feasibility replay {field}",
                    allow_empty=field == "feedback_ids",
                )
            except ValueError as exc:
                raise ValueError("feasibility replay ID lists are invalid") from exc
        for field in ("parent_audit_ok", "candidate_audit_ok", "process_signal"):
            if type(payload[field]) is not bool:
                raise ValueError("feasibility replay flags are invalid")
        outcome = FeasibilityOutcome(payload["outcome"])
        payload["outcome"] = outcome.value
        explanation = LayerBenefitExplanation.from_payload(payload["benefit_explanation"])
        if (
            explanation.target_layer.value != payload["target_layer"]
            or explanation.outcome is not outcome
        ):
            raise ValueError("benefit explanation does not match feasibility outcome")
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
        # Reload from the authoritative file rather than merging with a
        # possibly stale in-memory cache.  If the file was removed or replaced
        # between attempts, verification must fail closed.
        self._records = {}
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

    def verify_case(self, case: LayerIntervention) -> FeasibilityEvidenceRecord:
        """Recompute and compare a case identity after restart.

        The method never treats a missing or conflicting receipt as a new
        observation.  Callers must explicitly record a new case instead.
        """

        expected = FeasibilityEvidenceRecord.from_case(case)
        with self._lock():
            self._load()
            actual = self._records.get(expected.record_id)
        if actual is None:
            raise ValueError("feasibility evidence record is missing")
        canonical = json.dumps(expected.payload(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        if actual != canonical:
            raise ValueError("conflicting feasibility evidence record")
        return expected

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
    ambiguous_count: int
    status: FeasibilityStatus
    reason_codes: tuple[str, ...]
    benefit_explanation_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer", PolicyLayer(self.layer))
        for value, name in (
            (self.case_count, "case count"),
            (self.signal_count, "signal count"),
            (self.action_variation_count, "action variation count"),
            (self.outcome_variation_count, "outcome variation count"),
            (self.unknown_count, "unknown count"),
            (self.complete_feedback_count, "complete feedback count"),
            (self.ambiguous_count, "ambiguous count"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if any(value > self.case_count for value in (
            self.signal_count,
            self.action_variation_count,
            self.outcome_variation_count,
            self.unknown_count,
            self.complete_feedback_count,
            self.ambiguous_count,
        )):
            raise ValueError("census counts cannot exceed case count")
        object.__setattr__(self, "outcome_counts", MappingProxyType(dict(self.outcome_counts)))
        try:
            reasons = _normalize_strings(
                self.reason_codes,
                "census reason codes",
                allow_empty=True,
            )
        except ValueError as exc:
            raise ValueError("census reason codes must be unique non-empty strings") from exc
        object.__setattr__(self, "reason_codes", reasons)
        try:
            explanations = _normalize_strings(
                self.benefit_explanation_codes,
                "benefit explanation codes",
                allow_empty=True,
            )
        except ValueError as exc:
            raise ValueError(
                "benefit explanation codes must be unique non-empty strings"
            ) from exc
        object.__setattr__(self, "benefit_explanation_codes", explanations)

    @property
    def signal_coverage(self) -> float | None:
        return self.signal_count / self.case_count if self.case_count else None

    @property
    def action_variation(self) -> float | None:
        return self.action_variation_count / self.case_count if self.case_count else None

    @property
    def outcome_variation(self) -> float | None:
        return self.outcome_variation_count / self.case_count if self.case_count else None

    @property
    def unresolved_count(self) -> int:
        return int(self.outcome_counts.get(FeasibilityOutcome.UNRESOLVED.value, 0))

    @property
    def censored_count(self) -> int:
        return int(self.outcome_counts.get(FeasibilityOutcome.CENSORED.value, 0))

    @property
    def useful_count(self) -> int:
        return int(self.outcome_counts.get(FeasibilityOutcome.USEFUL.value, 0))

    @property
    def harmful_count(self) -> int:
        return int(self.outcome_counts.get(FeasibilityOutcome.HARMFUL.value, 0))

    @property
    def missed_count(self) -> int:
        return int(self.outcome_counts.get(FeasibilityOutcome.MISSED.value, 0))

    @property
    def resolved_useful_rate(self) -> float | None:
        denominator = self.useful_count + self.harmful_count
        return self.useful_count / denominator if denominator else None

    @property
    def unresolved_ratio(self) -> float | None:
        return self.unresolved_count / self.case_count if self.case_count else None

    @property
    def censored_ratio(self) -> float | None:
        return self.censored_count / self.case_count if self.case_count else None

    @property
    def ambiguous_ratio(self) -> float | None:
        return self.ambiguous_count / self.case_count if self.case_count else None

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
            "usefulCount": self.useful_count,
            "harmfulCount": self.harmful_count,
            "missedCount": self.missed_count,
            "resolvedUsefulRate": self.resolved_useful_rate,
            "unresolvedCount": self.unresolved_count,
            "unresolvedRatio": self.unresolved_ratio,
            "censoredCount": self.censored_count,
            "censoredRatio": self.censored_ratio,
            "ambiguousCount": self.ambiguous_count,
            "ambiguousRatio": self.ambiguous_ratio,
            "completeFeedbackCount": self.complete_feedback_count,
            "status": self.status.value,
            "reasonCodes": list(self.reason_codes),
            "benefitExplanationCodes": list(self.benefit_explanation_codes),
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
            "schemaVersion": POLICY_FEASIBILITY_REPORT_SCHEMA_VERSION,
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
                    "processFeedback": (
                        case.process_feedback.payload()
                        if case.process_feedback is not None
                        else None
                    ),
                    "benefitExplanation": {
                        **case.benefit_explanation.payload(),
                        "summary": case.benefit_explanation.summary,
                    },
                    "hypothesis": (
                        case.hypothesis.payload()
                        if case.hypothesis is not None
                        else None
                    ),
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
        resolved_outcomes = {
            case.outcome
            for case in items
            if case.outcome_resolved
        }
        # "Variation" means at least two distinct resolved outcomes.  A lone
        # useful/harmful/missed observation is coverage, not outcome variation.
        outcome_variation_count = (
            len(items) if len(resolved_outcomes) >= 2 else 0
        )
        unknown_count = sum(
            1 for case in items
            if case.outcome in {FeasibilityOutcome.UNRESOLVED, FeasibilityOutcome.CENSORED}
        )
        complete_count = sum(
            1 for case in items
            if (
                case.feedback.complete_useful
                or case.feedback.complete_harmful
                or case.feedback.complete_missed
            )
        )
        ambiguous_count = sum(
            1
            for case in items
            if any("ambiguous" in reason.casefold() for reason in case.reason_codes)
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
            ambiguous_count,
            status,
            tuple(reasons),
            tuple(sorted({
                case.benefit_explanation.mechanism_code
                for case in items
            })),
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
    "POLICY_FEASIBILITY_REPORT_SCHEMA_VERSION",
    "FeasibilityOutcome",
    "FeasibilityStatus",
    "LayerBenefitExplanation",
    "FeedbackChain",
    "feedback_chain_from_extraction_example",
    "build_extraction_feedback_interventions",
    "build_optimizer_corpus_interventions",
    "ProcessFeedback",
    "PolicyHypothesis",
    "OptimizerHypothesisDecision",
    "OptimizerHypothesisProjection",
    "project_optimizer_result",
    "LayerIntervention",
    "FeasibilityEvidenceRecord",
    "JsonFeasibilityEvidenceLedger",
    "LayerFeasibilityCensus",
    "PolicyFeasibilityReport",
    "build_feasibility_report",
    "validate_feasibility_case",
]
