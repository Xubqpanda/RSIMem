"""Matched held-out execution gate for adaptive policy activation."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from .adaptive_policy import AdaptivePolicyArtifact, AdaptivePolicyState
from .adaptive_policy_store import JsonAdaptivePolicyStore
from .adaptive_policy_validation import AdaptiveValidationSplit
from .feedback_dataset import FeedbackLabel


MATCHED_VALIDATION_SCHEMA_VERSION = 1
MATCHED_CRITERIA_VERSION = "semantic-adaptive-matched-criteria-v1"
MATCHED_DECISION_SCHEMA = "semantic-adaptive-matched-decision-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_REASON = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _require_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")


def _require_ids(values: tuple[str, ...], name: str) -> None:
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique identifiers")
    for value in values:
        _require_identifier(value, name)


def _finite(value: float, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


class MatchedPolicyVariant(StrEnum):
    STATIC = "static"
    PROPOSAL = "proposal"


@dataclass(frozen=True, slots=True)
class MatchedPolicyObservation:
    observation_id: str
    split_id: str
    example_id: str
    episode_id: str
    variant: MatchedPolicyVariant
    policy_version: str
    label: FeedbackLabel
    lifecycle_cost: float
    stability_failure: bool
    uncertainty: float
    evidence_id: str
    evidence_cutoff: int
    task_input_digest: str
    budget_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant", MatchedPolicyVariant(self.variant))
        object.__setattr__(self, "label", FeedbackLabel(self.label))
        for value in (
            self.observation_id,
            self.split_id,
            self.example_id,
            self.episode_id,
            self.policy_version,
            self.evidence_id,
            self.budget_id,
        ):
            _require_identifier(value, "matched policy observation identity")
        if _DIGEST.fullmatch(self.task_input_digest) is None:
            raise ValueError("matched task input digest is invalid")
        cost = _finite(self.lifecycle_cost, "matched lifecycle cost")
        uncertainty = _finite(self.uncertainty, "matched uncertainty")
        if cost <= 0 or not 0.0 <= uncertainty <= 1.0:
            raise ValueError("matched held-out metrics are invalid")
        if type(self.stability_failure) is not bool:
            raise TypeError("matched stability flag must be bool")
        if type(self.evidence_cutoff) is not int or self.evidence_cutoff < 0:
            raise ValueError("matched evidence cutoff is invalid")
        object.__setattr__(self, "lifecycle_cost", cost)
        object.__setattr__(self, "uncertainty", uncertainty)

    def payload(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "split_id": self.split_id,
            "example_id": self.example_id,
            "episode_id": self.episode_id,
            "variant": self.variant.value,
            "policy_version": self.policy_version,
            "label": self.label.value,
            "lifecycle_cost": self.lifecycle_cost,
            "stability_failure": self.stability_failure,
            "uncertainty": self.uncertainty,
            "evidence_id": self.evidence_id,
            "evidence_cutoff": self.evidence_cutoff,
            "task_input_digest": self.task_input_digest,
            "budget_id": self.budget_id,
        }


@dataclass(frozen=True, slots=True)
class MatchedAcceptanceCriteria:
    minimum_matched_examples: int = 1
    minimum_resolved_examples: int = 1
    minimum_quality_delta: float = 0.0
    maximum_cost_ratio: float = 1.0
    maximum_stability_failures: int = 0
    maximum_mean_uncertainty: float = 0.25
    criteria_version: str = MATCHED_CRITERIA_VERSION
    schema_version: int = MATCHED_VALIDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != MATCHED_VALIDATION_SCHEMA_VERSION
            or self.criteria_version != MATCHED_CRITERIA_VERSION
        ):
            raise ValueError("matched acceptance criteria identity is not frozen")
        if type(self.minimum_matched_examples) is not int or (
            self.minimum_matched_examples < 1
        ):
            raise ValueError("minimum matched examples must be positive")
        if type(self.minimum_resolved_examples) is not int or (
            self.minimum_resolved_examples < 1
        ):
            raise ValueError("minimum resolved examples must be positive")
        if type(self.maximum_stability_failures) is not int or (
            self.maximum_stability_failures < 0
        ):
            raise ValueError("matched stability criterion is invalid")
        quality = _finite(self.minimum_quality_delta, "matched quality criterion")
        cost = _finite(self.maximum_cost_ratio, "matched cost criterion")
        uncertainty = _finite(
            self.maximum_mean_uncertainty,
            "matched uncertainty criterion",
        )
        if (
            not -1.0 <= quality <= 1.0
            or cost < 0
            or not 0.0 <= uncertainty <= 1.0
        ):
            raise ValueError("matched acceptance criteria are out of range")
        object.__setattr__(self, "minimum_quality_delta", quality)
        object.__setattr__(self, "maximum_cost_ratio", cost)
        object.__setattr__(self, "maximum_mean_uncertainty", uncertainty)

    @property
    def digest(self) -> str:
        return _digest(self.payload())

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "criteria_version": self.criteria_version,
            "minimum_matched_examples": self.minimum_matched_examples,
            "minimum_resolved_examples": self.minimum_resolved_examples,
            "minimum_quality_delta": self.minimum_quality_delta,
            "maximum_cost_ratio": self.maximum_cost_ratio,
            "maximum_stability_failures": self.maximum_stability_failures,
            "maximum_mean_uncertainty": self.maximum_mean_uncertainty,
        }


@dataclass(frozen=True, slots=True)
class MatchedValidationDecision:
    decision_id: str
    artifact_id: str
    policy_version: str
    parent_policy_version: str
    split_id: str
    criteria_digest: str
    accepted: bool
    matched_example_count: int
    resolved_example_count: int
    static_quality: float
    proposal_quality: float
    quality_delta: float
    cost_ratio: float
    stability_failures: int
    mean_uncertainty: float
    reason_codes: tuple[str, ...]
    observation_ids: tuple[str, ...]
    used_policy_versions: tuple[str, str]
    content_digest: str
    decision_schema: str = MATCHED_DECISION_SCHEMA
    schema_version: int = MATCHED_VALIDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != MATCHED_VALIDATION_SCHEMA_VERSION
            or self.decision_schema != MATCHED_DECISION_SCHEMA
        ):
            raise ValueError("unsupported matched validation decision schema")
        for value in (
            self.decision_id,
            self.artifact_id,
            self.policy_version,
            self.parent_policy_version,
            self.split_id,
            self.decision_schema,
        ):
            _require_identifier(value, "matched validation decision identity")
        for digest in (self.criteria_digest, self.content_digest):
            if _DIGEST.fullmatch(digest) is None:
                raise ValueError("matched validation decision digest is invalid")
        if type(self.accepted) is not bool:
            raise TypeError("matched validation acceptance must be bool")
        if (
            type(self.matched_example_count) is not int
            or self.matched_example_count < 1
            or type(self.resolved_example_count) is not int
            or not 0 <= self.resolved_example_count <= self.matched_example_count
        ):
            raise ValueError("matched validation example count is invalid")
        if type(self.stability_failures) is not int or self.stability_failures < 0:
            raise ValueError("matched validation stability count is invalid")
        static_quality = _finite(self.static_quality, "matched static quality")
        proposal_quality = _finite(self.proposal_quality, "matched proposal quality")
        quality_delta = _finite(self.quality_delta, "matched quality delta")
        cost_ratio = _finite(self.cost_ratio, "matched cost ratio")
        uncertainty = _finite(self.mean_uncertainty, "matched mean uncertainty")
        if (
            not 0.0 <= static_quality <= 1.0
            or not 0.0 <= proposal_quality <= 1.0
            or not math.isclose(
                quality_delta,
                proposal_quality - static_quality,
                abs_tol=1e-12,
            )
            or cost_ratio < 0
            or not 0.0 <= uncertainty <= 1.0
        ):
            raise ValueError("matched validation decision metrics are inconsistent")
        if not self.reason_codes or any(
            _REASON.fullmatch(value) is None for value in self.reason_codes
        ):
            raise ValueError("matched validation decision requires reason codes")
        if self.accepted != (self.reason_codes == ("matched_criteria_passed",)):
            raise ValueError("matched validation acceptance reasons are inconsistent")
        _require_ids(self.observation_ids, "matched validation observations")
        if self.used_policy_versions != (
            self.parent_policy_version,
            self.policy_version,
        ):
            raise ValueError("matched validation policy versions are inconsistent")
        expected_digest = _digest(self.identity_payload())
        if self.content_digest != expected_digest:
            raise ValueError("matched validation decision digest mismatch")
        if self.decision_id != f"matched-validation.{expected_digest[:40]}":
            raise ValueError("matched validation decision ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "decision_schema": self.decision_schema,
            "artifact_id": self.artifact_id,
            "policy_version": self.policy_version,
            "parent_policy_version": self.parent_policy_version,
            "split_id": self.split_id,
            "criteria_digest": self.criteria_digest,
            "accepted": self.accepted,
            "matched_example_count": self.matched_example_count,
            "resolved_example_count": self.resolved_example_count,
            "static_quality": self.static_quality,
            "proposal_quality": self.proposal_quality,
            "quality_delta": self.quality_delta,
            "cost_ratio": self.cost_ratio,
            "stability_failures": self.stability_failures,
            "mean_uncertainty": self.mean_uncertainty,
            "reason_codes": list(self.reason_codes),
            "observation_ids": list(self.observation_ids),
            "used_policy_versions": list(self.used_policy_versions),
        }

    def payload(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "decision_id": self.decision_id,
            "content_digest": self.content_digest,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveRollbackEvidence:
    evidence_id: str
    policy_version: str
    automatic: bool
    reason_codes: tuple[str, ...]
    evidence_cutoff: int

    def __post_init__(self) -> None:
        _require_identifier(self.evidence_id, "adaptive rollback evidence ID")
        _require_identifier(self.policy_version, "adaptive rollback policy")
        if type(self.automatic) is not bool:
            raise TypeError("adaptive rollback automatic flag must be bool")
        if not self.reason_codes or any(
            _REASON.fullmatch(value) is None for value in self.reason_codes
        ):
            raise ValueError("adaptive rollback requires reason codes")
        if type(self.evidence_cutoff) is not int or self.evidence_cutoff < 0:
            raise ValueError("adaptive rollback cutoff is invalid")
        if self.evidence_id != f"rollback-evidence.{_digest(self.identity_payload())[:40]}":
            raise ValueError("adaptive rollback evidence identity is invalid")

    @classmethod
    def create(
        cls,
        *,
        policy_version: str,
        automatic: bool,
        reason_codes: tuple[str, ...],
        evidence_cutoff: int,
    ) -> "AdaptiveRollbackEvidence":
        core = {
            "policy_version": policy_version,
            "automatic": automatic,
            "reason_codes": list(reason_codes),
            "evidence_cutoff": evidence_cutoff,
        }
        return cls(
            evidence_id=f"rollback-evidence.{_digest(core)[:40]}",
            policy_version=policy_version,
            automatic=automatic,
            reason_codes=reason_codes,
            evidence_cutoff=evidence_cutoff,
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "automatic": self.automatic,
            "reason_codes": list(self.reason_codes),
            "evidence_cutoff": self.evidence_cutoff,
        }


class MatchedAdaptivePolicyValidator:
    def evaluate(
        self,
        artifact: AdaptivePolicyArtifact,
        split: AdaptiveValidationSplit,
        observations: tuple[MatchedPolicyObservation, ...],
        criteria: MatchedAcceptanceCriteria,
    ) -> MatchedValidationDecision:
        if artifact.training_example_ids != split.training_example_ids:
            raise ValueError("matched artifact training split mismatch")
        if (
            not observations
            or len({item.observation_id for item in observations}) != len(observations)
            or len({item.evidence_id for item in observations}) != len(observations)
        ):
            raise ValueError("matched observations must be unique")
        validation_ids = set(split.validation_example_ids)
        validation_episodes = set(split.validation_episode_ids)
        pairs: dict[str, dict[MatchedPolicyVariant, MatchedPolicyObservation]] = {}
        for observation in observations:
            if (
                observation.split_id != split.split_id
                or observation.example_id not in validation_ids
                or observation.episode_id not in validation_episodes
            ):
                raise ValueError("matched observation is outside validation split")
            expected_policy = (
                artifact.parent_policy_version
                if observation.variant == MatchedPolicyVariant.STATIC
                else artifact.policy_version
            )
            if observation.policy_version != expected_policy:
                raise ValueError("matched observation policy version mismatch")
            variants = pairs.setdefault(observation.example_id, {})
            if observation.variant in variants:
                raise ValueError("matched observation variant is duplicated")
            variants[observation.variant] = observation
        if set(pairs) != validation_ids or any(
            set(values) != set(MatchedPolicyVariant) for values in pairs.values()
        ):
            raise ValueError("matched validation requires complete observation pairs")
        for values in pairs.values():
            static = values[MatchedPolicyVariant.STATIC]
            proposal = values[MatchedPolicyVariant.PROPOSAL]
            if (
                static.episode_id != proposal.episode_id
                or static.task_input_digest != proposal.task_input_digest
                or static.budget_id != proposal.budget_id
                or static.evidence_cutoff != proposal.evidence_cutoff
            ):
                raise ValueError("matched validation pair identity differs")
        resolved_pairs = tuple(
            values
            for values in pairs.values()
            if values[MatchedPolicyVariant.STATIC].label
            in {FeedbackLabel.POSITIVE, FeedbackLabel.NEGATIVE}
            and values[MatchedPolicyVariant.PROPOSAL].label
            in {FeedbackLabel.POSITIVE, FeedbackLabel.NEGATIVE}
        )
        static_values = [
            values[MatchedPolicyVariant.STATIC] for values in resolved_pairs
        ]
        proposal_values = [
            values[MatchedPolicyVariant.PROPOSAL] for values in resolved_pairs
        ]

        def quality(values: list[MatchedPolicyObservation]) -> float:
            return (
                sum(item.label == FeedbackLabel.POSITIVE for item in values)
                / len(values)
                if values
                else 0.0
            )

        static_quality = quality(static_values)
        proposal_quality = quality(proposal_values)
        quality_delta = proposal_quality - static_quality
        all_static = [values[MatchedPolicyVariant.STATIC] for values in pairs.values()]
        all_proposal = [values[MatchedPolicyVariant.PROPOSAL] for values in pairs.values()]
        static_cost = sum(item.lifecycle_cost for item in all_static) / len(all_static)
        proposal_cost = sum(item.lifecycle_cost for item in all_proposal) / len(
            all_proposal
        )
        cost_ratio = proposal_cost / static_cost
        stability_failures = sum(item.stability_failure for item in all_proposal)
        uncertainty = sum(item.uncertainty for item in all_proposal) / len(
            all_proposal
        )
        reasons = []
        if len(pairs) < criteria.minimum_matched_examples:
            reasons.append("insufficient_matched_examples")
        if len(resolved_pairs) < criteria.minimum_resolved_examples:
            reasons.append("insufficient_resolved_examples")
        if quality_delta < criteria.minimum_quality_delta:
            reasons.append("quality_criterion_failed")
        if cost_ratio > criteria.maximum_cost_ratio:
            reasons.append("cost_criterion_failed")
        if stability_failures > criteria.maximum_stability_failures:
            reasons.append("stability_criterion_failed")
        if uncertainty > criteria.maximum_mean_uncertainty:
            reasons.append("uncertainty_criterion_failed")
        accepted = not reasons
        reason_codes = ("matched_criteria_passed",) if accepted else tuple(reasons)
        identity = {
            "schema_version": MATCHED_VALIDATION_SCHEMA_VERSION,
            "decision_schema": MATCHED_DECISION_SCHEMA,
            "artifact_id": artifact.artifact_id,
            "policy_version": artifact.policy_version,
            "parent_policy_version": artifact.parent_policy_version,
            "split_id": split.split_id,
            "criteria_digest": criteria.digest,
            "accepted": accepted,
            "matched_example_count": len(pairs),
            "resolved_example_count": len(resolved_pairs),
            "static_quality": static_quality,
            "proposal_quality": proposal_quality,
            "quality_delta": quality_delta,
            "cost_ratio": cost_ratio,
            "stability_failures": stability_failures,
            "mean_uncertainty": uncertainty,
            "reason_codes": list(reason_codes),
            "observation_ids": sorted(item.observation_id for item in observations),
            "used_policy_versions": [
                artifact.parent_policy_version,
                artifact.policy_version,
            ],
        }
        digest = _digest(identity)
        return MatchedValidationDecision(
            decision_id=f"matched-validation.{digest[:40]}",
            artifact_id=artifact.artifact_id,
            policy_version=artifact.policy_version,
            parent_policy_version=artifact.parent_policy_version,
            split_id=split.split_id,
            criteria_digest=criteria.digest,
            accepted=accepted,
            matched_example_count=len(pairs),
            resolved_example_count=len(resolved_pairs),
            static_quality=static_quality,
            proposal_quality=proposal_quality,
            quality_delta=quality_delta,
            cost_ratio=cost_ratio,
            stability_failures=stability_failures,
            mean_uncertainty=uncertainty,
            reason_codes=reason_codes,
            observation_ids=tuple(sorted(item.observation_id for item in observations)),
            used_policy_versions=(
                artifact.parent_policy_version,
                artifact.policy_version,
            ),
            content_digest=digest,
        )


class JsonMatchedValidationDecisionStore:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    @contextmanager
    def _lock(self, operation: int):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".matched.lock").open("w", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def put(self, decision: MatchedValidationDecision) -> tuple[Path, bool]:
        path = self.root / f"{decision.decision_id}.json"
        canonical = _canonical(decision.payload()) + "\n"
        with self._lock(fcntl.LOCK_EX):
            if path.exists():
                if path.read_text(encoding="utf-8") != canonical:
                    raise ValueError("matched decision conflicts with its ID")
                return path, False
            fd, temporary = tempfile.mkstemp(prefix=".matched.", dir=self.root)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(canonical)
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

    @staticmethod
    def _parse(value: object) -> MatchedValidationDecision:
        fields = {
            "schema_version",
            "decision_schema",
            "decision_id",
            "artifact_id",
            "policy_version",
            "parent_policy_version",
            "split_id",
            "criteria_digest",
            "accepted",
            "matched_example_count",
            "resolved_example_count",
            "static_quality",
            "proposal_quality",
            "quality_delta",
            "cost_ratio",
            "stability_failures",
            "mean_uncertainty",
            "reason_codes",
            "observation_ids",
            "used_policy_versions",
            "content_digest",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("malformed matched validation decision")
        try:
            return MatchedValidationDecision(
                decision_id=value["decision_id"],
                artifact_id=value["artifact_id"],
                policy_version=value["policy_version"],
                parent_policy_version=value["parent_policy_version"],
                split_id=value["split_id"],
                criteria_digest=value["criteria_digest"],
                accepted=value["accepted"],
                matched_example_count=value["matched_example_count"],
                resolved_example_count=value["resolved_example_count"],
                static_quality=value["static_quality"],
                proposal_quality=value["proposal_quality"],
                quality_delta=value["quality_delta"],
                cost_ratio=value["cost_ratio"],
                stability_failures=value["stability_failures"],
                mean_uncertainty=value["mean_uncertainty"],
                reason_codes=tuple(value["reason_codes"]),
                observation_ids=tuple(value["observation_ids"]),
                used_policy_versions=tuple(value["used_policy_versions"]),
                content_digest=value["content_digest"],
                decision_schema=value["decision_schema"],
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed matched validation decision") from exc

    def get(self, decision_id: str) -> MatchedValidationDecision | None:
        _require_identifier(decision_id, "matched validation decision ID")
        path = self.root / f"{decision_id}.json"
        with self._lock(fcntl.LOCK_SH):
            if not path.exists():
                return None
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("malformed matched validation decision JSON") from exc
            decision = self._parse(value)
            if decision.decision_id != decision_id:
                raise ValueError("matched validation decision filename mismatch")
            return decision


class MatchedAdaptivePolicyActivationCoordinator:
    def __init__(
        self,
        policy_store: JsonAdaptivePolicyStore,
        decision_store: JsonMatchedValidationDecisionStore,
    ) -> None:
        self.policy_store = policy_store
        self.decision_store = decision_store

    def apply(
        self,
        artifact: AdaptivePolicyArtifact,
        decision: MatchedValidationDecision,
        *,
        split: AdaptiveValidationSplit,
        observations: tuple[MatchedPolicyObservation, ...],
        criteria: MatchedAcceptanceCriteria,
    ) -> AdaptivePolicyState:
        if (
            decision.artifact_id != artifact.artifact_id
            or decision.policy_version != artifact.policy_version
        ):
            raise ValueError("matched decision and artifact differ")
        replay = MatchedAdaptivePolicyValidator().evaluate(
            artifact,
            split,
            observations,
            criteria,
        )
        if replay != decision:
            raise ValueError("matched activation decision replay mismatch")
        self.decision_store.put(decision)
        current, _ = self.policy_store.register(artifact)
        if current.state == AdaptivePolicyState.ACTIVE and decision.accepted:
            return current.state
        if current.state == AdaptivePolicyState.REJECTED and not decision.accepted:
            return current.state
        if current.state != AdaptivePolicyState.VALIDATED:
            raise ValueError("matched activation requires offline validated proposal")
        suffix = _digest(decision.decision_id)[:24]
        if not decision.accepted:
            record, _ = self.policy_store.transition(
                artifact.policy_version,
                to_state=AdaptivePolicyState.REJECTED,
                transition_id=f"policy-transition.matched-reject.{suffix}",
                reason_code=decision.reason_codes[0],
            )
            return record.state
        record, _ = self.policy_store.transition(
            artifact.policy_version,
            to_state=AdaptivePolicyState.ACTIVE,
            transition_id=f"policy-transition.matched-activate.{suffix}",
            reason_code="matched_validation_passed",
        )
        return record.state

    def rollback(
        self,
        artifact: AdaptivePolicyArtifact,
        evidence: AdaptiveRollbackEvidence,
    ) -> AdaptivePolicyState:
        if evidence.policy_version != artifact.policy_version:
            raise ValueError("rollback evidence policy mismatch")
        record, _ = self.policy_store.transition(
            artifact.policy_version,
            to_state=AdaptivePolicyState.ROLLED_BACK,
            transition_id=f"policy-transition.rollback.{_digest(evidence.evidence_id)[:24]}",
            reason_code=(
                "automatic_safety_rollback"
                if evidence.automatic
                else "operator_rollback"
            ),
        )
        return record.state
