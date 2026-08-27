"""Versioned static future-utility features and interpretable scoring."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from ..lifecycle import CompletionStatus, ExitEvidence, MemoryScope, TemporalValidity


STATIC_UTILITY_SCHEMA_VERSION = 1
STATIC_UTILITY_FEATURE_SCHEMA = "semantic-static-utility-features-v1"
STATIC_UTILITY_COST_SCHEMA = "semantic-lifecycle-cost-v1"
STATIC_UTILITY_POLICY_VERSION = "semantic-static-utility-policy-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")


class FeatureSource(StrEnum):
    HOST_OBSERVED = "host_observed"
    MODEL_PREDICTED = "model_predicted"
    DELAYED = "delayed"


class MissingReason(StrEnum):
    UNKNOWN = "unknown"
    NO_HISTORY = "no_history"
    NOT_OBSERVED = "not_observed"
    NOT_APPLICABLE = "not_applicable"


class UtilityFeatureName(StrEnum):
    COMPLETION_STATUS = "completion_status"
    UNRESOLVED = "unresolved"
    SCOPE = "scope"
    TEMPORAL_VALIDITY = "temporal_validity"
    RECENCY = "recency"
    REUSE_LIKELIHOOD = "reuse_likelihood"
    CONFLICT_RISK = "conflict_risk"
    RECOVERY_RISK = "recovery_risk"
    PREDICTED_BENEFIT = "predicted_benefit"
    CONFIDENCE = "confidence"


class LifecycleCostName(StrEnum):
    GENERATION_INPUT_TOKENS = "generation_input_tokens"
    GENERATION_OUTPUT_TOKENS = "generation_output_tokens"
    STORAGE_BYTES = "storage_bytes"
    RETRIEVAL_COUNT = "retrieval_count"
    INJECTION_TOKENS = "injection_tokens"
    RECOVERY_DURATION_MS = "recovery_duration_ms"


class UtilityTarget(StrEnum):
    GENERATION = "generation"
    INTERNAL_OPERATION = "internal_operation"
    RETRIEVAL = "retrieval"


class UtilityDisposition(StrEnum):
    ACCEPT = "accept"
    DEFER = "defer"
    REJECT = "reject"


_NUMERIC_FEATURES = {
    UtilityFeatureName.RECENCY,
    UtilityFeatureName.REUSE_LIKELIHOOD,
    UtilityFeatureName.CONFLICT_RISK,
    UtilityFeatureName.RECOVERY_RISK,
    UtilityFeatureName.PREDICTED_BENEFIT,
    UtilityFeatureName.CONFIDENCE,
}
_HOST_FEATURES = {
    UtilityFeatureName.COMPLETION_STATUS,
    UtilityFeatureName.UNRESOLVED,
    UtilityFeatureName.SCOPE,
    UtilityFeatureName.TEMPORAL_VALIDITY,
    UtilityFeatureName.RECENCY,
    UtilityFeatureName.RECOVERY_RISK,
}
_BENEFIT_NAMES = {
    "predicted_benefit",
    "reuse_likelihood",
    "completion",
    "scope",
    "validity",
    "recency",
}
_RISK_NAMES = {"unresolved", "conflict", "recovery", "uncertainty"}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class FeatureObservation:
    name: UtilityFeatureName
    source: FeatureSource
    available_at: int
    value: str | bool | float | None = None
    missing_reason: MissingReason | None = None
    schema_version: int = STATIC_UTILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STATIC_UTILITY_SCHEMA_VERSION:
            raise ValueError("unsupported utility feature observation schema")
        object.__setattr__(self, "name", UtilityFeatureName(self.name))
        object.__setattr__(self, "source", FeatureSource(self.source))
        if type(self.available_at) is not int or self.available_at < 0:
            raise ValueError("feature availability must be a non-negative integer")
        if self.missing_reason is not None:
            object.__setattr__(self, "missing_reason", MissingReason(self.missing_reason))
        if self.name in _HOST_FEATURES and self.source not in {
            FeatureSource.HOST_OBSERVED,
            FeatureSource.DELAYED,
        }:
            raise ValueError("deterministic utility feature requires host evidence")
        if self.value is None:
            if self.missing_reason is None:
                raise ValueError("missing feature requires an explicit reason")
            return
        if self.missing_reason is not None:
            raise ValueError("present feature cannot carry a missing reason")
        if self.name in _NUMERIC_FEATURES:
            if (
                not isinstance(self.value, (int, float))
                or isinstance(self.value, bool)
                or not math.isfinite(float(self.value))
                or not 0.0 <= float(self.value) <= 1.0
            ):
                raise ValueError("numeric utility feature must be finite in [0,1]")
            object.__setattr__(self, "value", float(self.value))
        elif self.name == UtilityFeatureName.UNRESOLVED:
            if type(self.value) is not bool:
                raise TypeError("unresolved utility feature must be bool")
        elif self.name == UtilityFeatureName.COMPLETION_STATUS:
            object.__setattr__(self, "value", CompletionStatus(str(self.value)).value)
        elif self.name == UtilityFeatureName.SCOPE:
            object.__setattr__(self, "value", MemoryScope(str(self.value)).value)
        elif self.name == UtilityFeatureName.TEMPORAL_VALIDITY:
            object.__setattr__(self, "value", TemporalValidity(str(self.value)).value)

    def payload(self) -> dict[str, object]:
        return {
            "name": self.name.value,
            "source": self.source.value,
            "available_at": self.available_at,
            "value": self.value,
            "missing_reason": (
                self.missing_reason.value if self.missing_reason is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class StaticUtilityFeatures:
    observations: tuple[FeatureObservation, ...]
    feature_schema: str = STATIC_UTILITY_FEATURE_SCHEMA
    schema_version: int = STATIC_UTILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STATIC_UTILITY_SCHEMA_VERSION:
            raise ValueError("unsupported static utility feature schema")
        if self.feature_schema != STATIC_UTILITY_FEATURE_SCHEMA:
            raise ValueError("unknown static utility feature schema identity")
        names = tuple(item.name for item in self.observations)
        if len(names) != len(set(names)) or set(names) != set(UtilityFeatureName):
            raise ValueError("static utility features require each declared feature once")

    def get(self, name: UtilityFeatureName) -> FeatureObservation:
        name = UtilityFeatureName(name)
        return next(item for item in self.observations if item.name == name)

    @property
    def digest(self) -> str:
        return _digest(self.payload())

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "feature_schema": self.feature_schema,
            "observations": [
                item.payload() for item in sorted(self.observations, key=lambda value: value.name.value)
            ],
        }


@dataclass(frozen=True, slots=True)
class CostObservation:
    name: LifecycleCostName
    source: FeatureSource
    available_at: int
    amount: float | None = None
    missing_reason: MissingReason | None = None
    schema_version: int = STATIC_UTILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STATIC_UTILITY_SCHEMA_VERSION:
            raise ValueError("unsupported lifecycle cost observation schema")
        object.__setattr__(self, "name", LifecycleCostName(self.name))
        object.__setattr__(self, "source", FeatureSource(self.source))
        if type(self.available_at) is not int or self.available_at < 0:
            raise ValueError("cost availability must be a non-negative integer")
        if self.missing_reason is not None:
            object.__setattr__(self, "missing_reason", MissingReason(self.missing_reason))
        if self.amount is None:
            if self.missing_reason is None:
                raise ValueError("missing lifecycle cost requires an explicit reason")
        elif (
            self.missing_reason is not None
            or not isinstance(self.amount, (int, float))
            or isinstance(self.amount, bool)
            or not math.isfinite(float(self.amount))
            or self.amount < 0
        ):
            raise ValueError("lifecycle cost must be a finite non-negative quantity")
        else:
            object.__setattr__(self, "amount", float(self.amount))

    def payload(self) -> dict[str, object]:
        return {
            "name": self.name.value,
            "source": self.source.value,
            "available_at": self.available_at,
            "amount": self.amount,
            "missing_reason": (
                self.missing_reason.value if self.missing_reason is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class LifecycleCostProfile:
    observations: tuple[CostObservation, ...]
    cost_schema: str = STATIC_UTILITY_COST_SCHEMA
    schema_version: int = STATIC_UTILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STATIC_UTILITY_SCHEMA_VERSION:
            raise ValueError("unsupported lifecycle cost profile schema")
        if self.cost_schema != STATIC_UTILITY_COST_SCHEMA:
            raise ValueError("unknown lifecycle cost schema identity")
        names = tuple(item.name for item in self.observations)
        if len(names) != len(set(names)) or set(names) != set(LifecycleCostName):
            raise ValueError("lifecycle cost profile requires every cost bucket once")

    def get(self, name: LifecycleCostName) -> CostObservation:
        name = LifecycleCostName(name)
        return next(item for item in self.observations if item.name == name)

    @property
    def digest(self) -> str:
        return _digest(self.payload())

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "cost_schema": self.cost_schema,
            "observations": [
                item.payload() for item in sorted(self.observations, key=lambda value: value.name.value)
            ],
        }


@dataclass(frozen=True, slots=True)
class StaticUtilityPolicy:
    policy_version: str = STATIC_UTILITY_POLICY_VERSION
    accept_threshold: float = 0.35
    minimum_confidence: float = 0.55
    conflict_reject_threshold: float = 0.8
    benefit_weights: tuple[tuple[str, float], ...] = (
        ("predicted_benefit", 0.50),
        ("reuse_likelihood", 0.25),
        ("completion", 0.10),
        ("scope", 0.05),
        ("validity", 0.05),
        ("recency", 0.05),
    )
    risk_weights: tuple[tuple[str, float], ...] = (
        ("unresolved", 0.35),
        ("conflict", 0.30),
        ("recovery", 0.20),
        ("uncertainty", 0.15),
    )
    cost_caps: tuple[tuple[LifecycleCostName, float], ...] = (
        (LifecycleCostName.GENERATION_INPUT_TOKENS, 20_000.0),
        (LifecycleCostName.GENERATION_OUTPUT_TOKENS, 2_000.0),
        (LifecycleCostName.STORAGE_BYTES, 4_096.0),
        (LifecycleCostName.RETRIEVAL_COUNT, 20.0),
        (LifecycleCostName.INJECTION_TOKENS, 2_000.0),
        (LifecycleCostName.RECOVERY_DURATION_MS, 30_000.0),
    )
    cost_weight: float = 0.30

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.policy_version):
            raise ValueError("static utility policy version is invalid")
        for value in (
            self.accept_threshold,
            self.minimum_confidence,
            self.conflict_reject_threshold,
            self.cost_weight,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("static utility policy thresholds must be in [0,1]")
        if any(weight < 0 for _, weight in (*self.benefit_weights, *self.risk_weights)):
            raise ValueError("static utility policy weights must not be negative")
        benefit_names = [name for name, _ in self.benefit_weights]
        risk_names = [name for name, _ in self.risk_weights]
        if set(benefit_names) != _BENEFIT_NAMES or len(benefit_names) != len(set(benefit_names)):
            raise ValueError("static utility policy benefit weights are incomplete")
        if set(risk_names) != _RISK_NAMES or len(risk_names) != len(set(risk_names)):
            raise ValueError("static utility policy risk weights are incomplete")
        cap_names = [name for name, _ in self.cost_caps]
        if set(cap_names) != set(LifecycleCostName) or len(cap_names) != len(set(cap_names)):
            raise ValueError("static utility policy cost caps are incomplete")
        if any(cap <= 0 for _, cap in self.cost_caps):
            raise ValueError("static utility policy cost caps must be positive")

    @property
    def digest(self) -> str:
        return _digest({
            "policy_version": self.policy_version,
            "accept_threshold": self.accept_threshold,
            "minimum_confidence": self.minimum_confidence,
            "conflict_reject_threshold": self.conflict_reject_threshold,
            "benefit_weights": list(self.benefit_weights),
            "risk_weights": list(self.risk_weights),
            "cost_caps": [
                (name.value, value) for name, value in self.cost_caps
            ],
            "cost_weight": self.cost_weight,
        })


@dataclass(frozen=True, slots=True)
class UtilityDecision:
    target: UtilityTarget
    disposition: UtilityDisposition
    score: float
    predicted_benefit: float
    lifecycle_cost: float
    risk: float
    contributions: tuple[tuple[str, float], ...]
    reason_codes: tuple[str, ...]
    feature_digest: str
    cost_digest: str
    feature_schema: str
    cost_schema: str
    policy_version: str
    cutoff: int
    schema_version: int = STATIC_UTILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STATIC_UTILITY_SCHEMA_VERSION:
            raise ValueError("unsupported utility decision schema")
        object.__setattr__(self, "target", UtilityTarget(self.target))
        object.__setattr__(self, "disposition", UtilityDisposition(self.disposition))
        if (
            not -1.0 <= self.score <= 1.0
            or any(not 0.0 <= value <= 1.0 for value in (
                self.predicted_benefit,
                self.lifecycle_cost,
                self.risk,
            ))
        ):
            raise ValueError("utility score must be in [-1,1]")
        if not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in (
            self.feature_digest,
            self.cost_digest,
        )):
            raise ValueError("utility decision input digests must be sha256")
        if (
            not _IDENTIFIER.fullmatch(self.policy_version)
            or not _IDENTIFIER.fullmatch(self.feature_schema)
            or not _IDENTIFIER.fullmatch(self.cost_schema)
        ):
            raise ValueError("utility decision schema and policy identity is invalid")
        if type(self.cutoff) is not int or self.cutoff < 0:
            raise ValueError("utility decision cutoff must be non-negative")
        contribution_names = [name for name, _ in self.contributions]
        if len(contribution_names) != len(set(contribution_names)) or any(
            not name or not math.isfinite(value)
            for name, value in self.contributions
        ):
            raise ValueError("utility decision contributions are invalid")
        if not self.reason_codes or any(
            re.fullmatch(r"[a-z][a-z0-9_]{0,63}", reason) is None
            for reason in self.reason_codes
        ):
            raise ValueError("utility decision requires machine-readable reasons")

    def observer_evidence(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "target": self.target.value,
            "disposition": self.disposition.value,
            "score": self.score,
            "predicted_benefit": self.predicted_benefit,
            "lifecycle_cost": self.lifecycle_cost,
            "risk": self.risk,
            "contributions": dict(self.contributions),
            "reason_codes": list(self.reason_codes),
            "feature_digest": self.feature_digest,
            "cost_digest": self.cost_digest,
            "feature_schema": self.feature_schema,
            "cost_schema": self.cost_schema,
            "policy_version": self.policy_version,
            "cutoff": self.cutoff,
        }


class StaticUtilityFeatureExtractor:
    """Build the fixed decision-time feature set without arbitrary metadata."""

    def extract(
        self,
        exit_evidence: ExitEvidence,
        *,
        available_at: int,
        recency: float | None,
        reuse_likelihood: float | None,
        conflict_risk: float | None,
        recovery_risk: float | None,
        predicted_benefit: float | None = None,
        confidence: float | None = None,
        no_history: bool = False,
    ) -> StaticUtilityFeatures:
        def numeric(
            name: UtilityFeatureName,
            value: float | None,
            source: FeatureSource,
            missing: MissingReason = MissingReason.UNKNOWN,
        ) -> FeatureObservation:
            return FeatureObservation(
                name,
                source,
                available_at,
                value,
                missing if value is None else None,
            )

        resolved_benefit = (
            exit_evidence.utility_estimate
            if predicted_benefit is None
            else predicted_benefit
        )
        resolved_confidence = (
            exit_evidence.confidence if confidence is None else confidence
        )
        return StaticUtilityFeatures((
            FeatureObservation(
                UtilityFeatureName.COMPLETION_STATUS,
                FeatureSource.HOST_OBSERVED,
                available_at,
                exit_evidence.completion_status.value,
            ),
            FeatureObservation(
                UtilityFeatureName.UNRESOLVED,
                FeatureSource.HOST_OBSERVED,
                available_at,
                exit_evidence.unresolved_state is not None,
            ),
            FeatureObservation(
                UtilityFeatureName.SCOPE,
                FeatureSource.HOST_OBSERVED,
                available_at,
                exit_evidence.scope.value if exit_evidence.scope is not None else None,
                MissingReason.UNKNOWN if exit_evidence.scope is None else None,
            ),
            FeatureObservation(
                UtilityFeatureName.TEMPORAL_VALIDITY,
                FeatureSource.HOST_OBSERVED,
                available_at,
                (
                    exit_evidence.temporal_validity.value
                    if exit_evidence.temporal_validity is not None
                    else None
                ),
                MissingReason.UNKNOWN if exit_evidence.temporal_validity is None else None,
            ),
            numeric(UtilityFeatureName.RECENCY, recency, FeatureSource.HOST_OBSERVED),
            numeric(
                UtilityFeatureName.REUSE_LIKELIHOOD,
                reuse_likelihood,
                FeatureSource.MODEL_PREDICTED,
                MissingReason.NO_HISTORY if no_history else MissingReason.UNKNOWN,
            ),
            numeric(
                UtilityFeatureName.CONFLICT_RISK,
                conflict_risk,
                FeatureSource.MODEL_PREDICTED,
            ),
            numeric(
                UtilityFeatureName.RECOVERY_RISK,
                recovery_risk,
                FeatureSource.HOST_OBSERVED,
            ),
            numeric(
                UtilityFeatureName.PREDICTED_BENEFIT,
                resolved_benefit,
                FeatureSource.MODEL_PREDICTED,
            ),
            numeric(
                UtilityFeatureName.CONFIDENCE,
                resolved_confidence,
                FeatureSource.MODEL_PREDICTED,
            ),
        ))


class InterpretableStaticUtilityScorer:
    """Apply one frozen monotone benefit-risk-cost objective to every target."""

    def __init__(self, policy: StaticUtilityPolicy = StaticUtilityPolicy()) -> None:
        self.policy = policy

    @staticmethod
    def _present(features: StaticUtilityFeatures, name: UtilityFeatureName) -> float:
        value = features.get(name).value
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0

    def score(
        self,
        features: StaticUtilityFeatures,
        costs: LifecycleCostProfile,
        *,
        target: UtilityTarget,
        cutoff: int,
    ) -> UtilityDecision:
        target = UtilityTarget(target)
        if type(cutoff) is not int or cutoff < 0:
            raise ValueError("utility cutoff must be a non-negative integer")
        observations = (*features.observations, *costs.observations)
        if any(item.available_at > cutoff for item in observations):
            raise ValueError("future-dated utility evidence is forbidden")
        if any(item.source == FeatureSource.DELAYED for item in observations):
            raise ValueError("delayed evidence cannot enter a static decision")

        missing_features = [item for item in features.observations if item.value is None]
        missing_costs = [item for item in costs.observations if item.amount is None]
        completion = features.get(UtilityFeatureName.COMPLETION_STATUS).value
        unresolved = features.get(UtilityFeatureName.UNRESOLVED).value is True
        scope = features.get(UtilityFeatureName.SCOPE).value
        validity = features.get(UtilityFeatureName.TEMPORAL_VALIDITY).value
        confidence = self._present(features, UtilityFeatureName.CONFIDENCE)
        conflict = self._present(features, UtilityFeatureName.CONFLICT_RISK)

        benefit_values = {
            "predicted_benefit": self._present(features, UtilityFeatureName.PREDICTED_BENEFIT),
            "reuse_likelihood": self._present(features, UtilityFeatureName.REUSE_LIKELIHOOD),
            "completion": 1.0 if completion == CompletionStatus.COMPLETED.value else 0.0,
            "scope": {
                MemoryScope.TURN.value: 0.0,
                MemoryScope.TASK.value: 0.25,
                MemoryScope.SESSION.value: 0.50,
                MemoryScope.USER.value: 0.75,
                MemoryScope.GLOBAL.value: 1.0,
            }.get(str(scope), 0.0),
            "validity": 1.0 if validity == TemporalValidity.DURABLE.value else 0.0,
            "recency": self._present(features, UtilityFeatureName.RECENCY),
        }
        risk_values = {
            "unresolved": 1.0 if unresolved else 0.0,
            "conflict": conflict,
            "recovery": self._present(features, UtilityFeatureName.RECOVERY_RISK),
            "uncertainty": 1.0 - confidence,
        }
        benefit_contributions = {
            name: benefit_values[name] * weight for name, weight in self.policy.benefit_weights
        }
        risk_contributions = {
            name: risk_values[name] * weight for name, weight in self.policy.risk_weights
        }
        caps = dict(self.policy.cost_caps)
        cost_contributions = {
            item.name.value: (
                0.0
                if item.amount is None
                else min(item.amount / caps[item.name], 1.0)
                * self.policy.cost_weight
                / len(LifecycleCostName)
            )
            for item in costs.observations
        }
        predicted_benefit = sum(benefit_contributions.values())
        risk = sum(risk_contributions.values())
        lifecycle_cost = sum(cost_contributions.values())
        utility = max(-1.0, min(1.0, predicted_benefit - risk - lifecycle_cost))

        reasons: list[str] = []
        disposition = UtilityDisposition.ACCEPT
        if completion != CompletionStatus.COMPLETED.value or unresolved:
            disposition = UtilityDisposition.REJECT
            reasons.append("unsafe_incomplete_source")
        elif validity in {None, TemporalValidity.EXPIRED.value, TemporalValidity.TRANSIENT.value}:
            disposition = UtilityDisposition.REJECT
            reasons.append("non_durable_or_unknown_validity")
        elif conflict >= self.policy.conflict_reject_threshold:
            disposition = UtilityDisposition.REJECT
            reasons.append("high_conflict_risk")
        elif missing_costs:
            disposition = UtilityDisposition.DEFER
            reasons.append("unknown_lifecycle_cost")
        elif any(item.missing_reason == MissingReason.NO_HISTORY for item in missing_features):
            disposition = UtilityDisposition.DEFER
            reasons.append("no_history")
        elif missing_features:
            disposition = UtilityDisposition.DEFER
            reasons.append("missing_feature")
        elif confidence < self.policy.minimum_confidence:
            disposition = UtilityDisposition.DEFER
            reasons.append("low_confidence")
        elif utility < self.policy.accept_threshold:
            disposition = UtilityDisposition.DEFER
            reasons.append("utility_below_threshold")
        else:
            reasons.append("utility_accepted")

        contributions = tuple(sorted({
            **{f"benefit.{key}": value for key, value in benefit_contributions.items()},
            **{f"risk.{key}": -value for key, value in risk_contributions.items()},
            **{f"cost.{key}": -value for key, value in cost_contributions.items()},
        }.items()))
        return UtilityDecision(
            target,
            disposition,
            utility,
            predicted_benefit,
            lifecycle_cost,
            risk,
            contributions,
            tuple(reasons),
            features.digest,
            costs.digest,
            features.feature_schema,
            costs.cost_schema,
            self.policy.policy_version,
            cutoff,
        )


def known_lifecycle_costs(
    *,
    available_at: int,
    values: Mapping[LifecycleCostName, float],
    source: FeatureSource = FeatureSource.MODEL_PREDICTED,
) -> LifecycleCostProfile:
    if set(values) != set(LifecycleCostName):
        raise ValueError("known lifecycle costs require every raw cost bucket")
    return LifecycleCostProfile(tuple(
        CostObservation(name, source, available_at, values[name])
        for name in LifecycleCostName
    ))
