"""Deterministic semantic policy learning from frozen delayed feedback."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum

from .feedback_dataset import (
    DELAYED_FEEDBACK_DATASET_VERSION,
    DELAYED_FEEDBACK_LABEL_SCHEMA,
    DELAYED_FEEDBACK_WINDOW_VERSION,
    DelayedFeedbackDataset,
    DelayedFeedbackExample,
    FeedbackDatasetStageGate,
    FeedbackLabel,
)
from .utility import STATIC_UTILITY_FEATURE_SCHEMA


ADAPTIVE_POLICY_SCHEMA_VERSION = 1
ADAPTIVE_POLICY_ARTIFACT_SCHEMA = "semantic-adaptive-policy-artifact-v1"
ADAPTIVE_POLICY_TRAINING_SCHEMA = "semantic-adaptive-policy-training-v1"
ADAPTIVE_POLICY_LEARNER_VERSION = "semantic-parameter-bayesian-v1"
ADAPTIVE_POLICY_OBJECTIVE = "delayed-future-utility-per-cost-v1"
FIXED_SEMANTIC_ROUTE = "hermes-native-semantic"
FIXED_INVOCATION_BOUNDARY = "task-completion-or-session-end-v1"
ADAPTIVE_POLICY_FEATURES = (
    "label",
    "exposure_state",
    "candidate_disposition",
    "policy_parameter_ids",
    "operation_membership",
    "failure_subgraph_operation_ids",
    "raw_resource_usage",
)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, value: object) -> str:
    return f"{prefix}.{_digest(value)[:40]}"


def _require_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")


def _require_ids(values: tuple[str, ...], name: str) -> None:
    if len(values) != len(set(values)) or any(
        not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None
        for value in values
    ):
        raise ValueError(f"{name} must be unique stable identifiers")


def _finite(value: float, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


class AdaptivePolicyState(StrEnum):
    PROPOSAL = "proposal"
    VALIDATED = "validated"
    ACTIVE = "active"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class AdaptiveFallbackReason(StrEnum):
    NONE = "none"
    LOW_SAMPLE = "low_sample"
    MISSING_FEATURE = "missing_feature"
    DISTRIBUTION_SHIFT = "distribution_shift"


class AdaptiveParameterName(StrEnum):
    EXTRACTION_ACCEPT_THRESHOLD = "extraction_accept_threshold"
    INTERNAL_OPERATION_ACCEPT_THRESHOLD = "internal_operation_accept_threshold"
    CONSOLIDATION_UPDATE_THRESHOLD = "consolidation_update_threshold"
    RETRIEVAL_ACCEPT_THRESHOLD = "retrieval_accept_threshold"


_PARAMETER_COMPONENT = {
    AdaptiveParameterName.EXTRACTION_ACCEPT_THRESHOLD: "extraction",
    AdaptiveParameterName.INTERNAL_OPERATION_ACCEPT_THRESHOLD: "internal_operation",
    AdaptiveParameterName.CONSOLIDATION_UPDATE_THRESHOLD: "consolidation",
    AdaptiveParameterName.RETRIEVAL_ACCEPT_THRESHOLD: "retrieval",
}


@dataclass(frozen=True, slots=True)
class AdaptiveParameterSpec:
    parameter_id: str
    name: AdaptiveParameterName
    baseline_value: float
    prompt_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", AdaptiveParameterName(self.name))
        _require_identifier(self.parameter_id, "adaptive parameter owner")
        _require_identifier(self.prompt_ref, "adaptive prompt reference")
        baseline = _finite(self.baseline_value, "adaptive parameter baseline")
        if not 0.0 <= baseline <= 1.0:
            raise ValueError("adaptive parameter baseline must be in [0,1]")
        object.__setattr__(self, "baseline_value", baseline)

    @property
    def component(self) -> str:
        return _PARAMETER_COMPONENT[self.name]

    def payload(self) -> dict[str, object]:
        return {
            "parameter_id": self.parameter_id,
            "name": self.name.value,
            "component": self.component,
            "baseline_value": self.baseline_value,
            "prompt_ref": self.prompt_ref,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveTrainingConfig:
    parent_policy_version: str
    seed: int
    parameters: tuple[AdaptiveParameterSpec, ...]
    training_example_ids: tuple[str, ...] = ()
    minimum_resolved_examples: int = 2
    maximum_missing_propensity_rate: float = 0.25
    prior_positive: float = 1.0
    prior_negative: float = 1.0
    l2_regularization: float = 1.0
    maximum_parameter_delta: float = 0.25
    learner_version: str = ADAPTIVE_POLICY_LEARNER_VERSION
    objective: str = ADAPTIVE_POLICY_OBJECTIVE
    feature_names: tuple[str, ...] = ADAPTIVE_POLICY_FEATURES
    feature_schema: str = STATIC_UTILITY_FEATURE_SCHEMA
    label_schema: str = DELAYED_FEEDBACK_LABEL_SCHEMA
    dataset_version: str = DELAYED_FEEDBACK_DATASET_VERSION
    window_version: str = DELAYED_FEEDBACK_WINDOW_VERSION
    route_backend: str = FIXED_SEMANTIC_ROUTE
    invocation_boundary: str = FIXED_INVOCATION_BOUNDARY
    training_schema: str = ADAPTIVE_POLICY_TRAINING_SCHEMA
    schema_version: int = ADAPTIVE_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ADAPTIVE_POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported adaptive training schema")
        for name in (
            "parent_policy_version",
            "learner_version",
            "objective",
            "feature_schema",
            "label_schema",
            "dataset_version",
            "window_version",
            "route_backend",
            "invocation_boundary",
            "training_schema",
        ):
            _require_identifier(getattr(self, name), f"adaptive training {name}")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("adaptive training seed must be a non-negative integer")
        if (
            type(self.minimum_resolved_examples) is not int
            or self.minimum_resolved_examples < 1
        ):
            raise ValueError("adaptive minimum resolved examples must be positive")
        if not self.parameters:
            raise ValueError("adaptive training requires trusted semantic parameters")
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "training_example_ids", tuple(self.training_example_ids))
        _require_ids(self.training_example_ids, "adaptive training examples")
        parameter_ids = tuple(item.parameter_id for item in self.parameters)
        parameter_names = tuple(item.name for item in self.parameters)
        if (
            len(parameter_ids) != len(set(parameter_ids))
            or len(parameter_names) != len(set(parameter_names))
        ):
            raise ValueError("adaptive training parameters must be unique")
        if (
            self.learner_version != ADAPTIVE_POLICY_LEARNER_VERSION
            or self.objective != ADAPTIVE_POLICY_OBJECTIVE
            or self.feature_names != ADAPTIVE_POLICY_FEATURES
            or self.feature_schema != STATIC_UTILITY_FEATURE_SCHEMA
            or self.label_schema != DELAYED_FEEDBACK_LABEL_SCHEMA
            or self.dataset_version != DELAYED_FEEDBACK_DATASET_VERSION
            or self.window_version != DELAYED_FEEDBACK_WINDOW_VERSION
            or self.route_backend != FIXED_SEMANTIC_ROUTE
            or self.invocation_boundary != FIXED_INVOCATION_BOUNDARY
            or self.training_schema != ADAPTIVE_POLICY_TRAINING_SCHEMA
        ):
            raise ValueError("adaptive training identity is not frozen")
        missing_rate = _finite(
            self.maximum_missing_propensity_rate,
            "adaptive missing propensity rate",
        )
        prior_positive = _finite(self.prior_positive, "adaptive positive prior")
        prior_negative = _finite(self.prior_negative, "adaptive negative prior")
        regularization = _finite(
            self.l2_regularization,
            "adaptive L2 regularization",
        )
        maximum_delta = _finite(
            self.maximum_parameter_delta,
            "adaptive maximum parameter delta",
        )
        if not 0.0 <= missing_rate <= 1.0 or not 0.0 <= maximum_delta <= 1.0:
            raise ValueError("adaptive training rates must be in [0,1]")
        if prior_positive <= 0 or prior_negative <= 0 or regularization < 0:
            raise ValueError("adaptive training regularization is invalid")
        object.__setattr__(self, "maximum_missing_propensity_rate", missing_rate)
        object.__setattr__(self, "prior_positive", prior_positive)
        object.__setattr__(self, "prior_negative", prior_negative)
        object.__setattr__(self, "l2_regularization", regularization)
        object.__setattr__(self, "maximum_parameter_delta", maximum_delta)

    @property
    def digest(self) -> str:
        return _digest(self.payload())

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "training_schema": self.training_schema,
            "learner_version": self.learner_version,
            "parent_policy_version": self.parent_policy_version,
            "seed": self.seed,
            "objective": self.objective,
            "parameters": [item.payload() for item in self.parameters],
            "training_example_ids": list(self.training_example_ids),
            "minimum_resolved_examples": self.minimum_resolved_examples,
            "maximum_missing_propensity_rate": (
                self.maximum_missing_propensity_rate
            ),
            "prior_positive": self.prior_positive,
            "prior_negative": self.prior_negative,
            "l2_regularization": self.l2_regularization,
            "maximum_parameter_delta": self.maximum_parameter_delta,
            "feature_names": list(self.feature_names),
            "feature_schema": self.feature_schema,
            "label_schema": self.label_schema,
            "dataset_version": self.dataset_version,
            "window_version": self.window_version,
            "route_backend": self.route_backend,
            "invocation_boundary": self.invocation_boundary,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveParameterUpdate:
    parameter_id: str
    name: AdaptiveParameterName
    component: str
    baseline_value: float
    proposed_value: float
    delta: float
    posterior_negative_rate: float
    positive_count: int
    negative_count: int
    fallback_reason: AdaptiveFallbackReason
    attributed_example_ids: tuple[str, ...]
    failure_subgraph_operation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", AdaptiveParameterName(self.name))
        object.__setattr__(
            self,
            "fallback_reason",
            AdaptiveFallbackReason(self.fallback_reason),
        )
        _require_identifier(self.parameter_id, "adaptive parameter update owner")
        _require_identifier(self.component, "adaptive parameter update component")
        if self.component != _PARAMETER_COMPONENT[self.name]:
            raise ValueError("adaptive parameter component is inconsistent")
        values = tuple(
            _finite(getattr(self, name), f"adaptive update {name}")
            for name in (
                "baseline_value",
                "proposed_value",
                "delta",
                "posterior_negative_rate",
            )
        )
        baseline, proposed, delta, posterior = values
        if (
            not 0.0 <= baseline <= 1.0
            or not 0.0 <= proposed <= 1.0
            or not -1.0 <= delta <= 1.0
            or not 0.0 <= posterior <= 1.0
            or round(proposed - baseline, 12) != round(delta, 12)
        ):
            raise ValueError("adaptive parameter update values are inconsistent")
        if any(type(value) is not int or value < 0 for value in (
            self.positive_count,
            self.negative_count,
        )):
            raise ValueError("adaptive parameter counts must be non-negative integers")
        _require_ids(self.attributed_example_ids, "adaptive attributed examples")
        _require_ids(
            self.failure_subgraph_operation_ids,
            "adaptive failure subgraph",
        )
        if self.positive_count + self.negative_count > 0 and not (
            self.attributed_example_ids
        ):
            raise ValueError("adaptive parameter update provenance is missing")
        if len(self.attributed_example_ids) != (
            self.positive_count + self.negative_count
        ):
            raise ValueError("adaptive parameter update counts and provenance differ")
        if self.negative_count > 0 and not self.failure_subgraph_operation_ids:
            raise ValueError("negative adaptive update requires failure provenance")
        if self.fallback_reason != AdaptiveFallbackReason.NONE and delta != 0.0:
            raise ValueError("adaptive fallback cannot modify a parameter")
        object.__setattr__(self, "baseline_value", baseline)
        object.__setattr__(self, "proposed_value", proposed)
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "posterior_negative_rate", posterior)

    @property
    def resolved_count(self) -> int:
        return self.positive_count + self.negative_count

    def payload(self) -> dict[str, object]:
        return {
            "parameter_id": self.parameter_id,
            "name": self.name.value,
            "component": self.component,
            "baseline_value": self.baseline_value,
            "proposed_value": self.proposed_value,
            "delta": self.delta,
            "posterior_negative_rate": self.posterior_negative_rate,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "fallback_reason": self.fallback_reason.value,
            "attributed_example_ids": list(self.attributed_example_ids),
            "failure_subgraph_operation_ids": list(
                self.failure_subgraph_operation_ids
            ),
        }


@dataclass(frozen=True, slots=True)
class AdaptiveTrainingMetrics:
    observation_count: int
    positive_count: int
    negative_count: int
    unresolved_count: int
    censored_count: int
    missing_propensity_count: int
    missing_propensity_rate: float

    def __post_init__(self) -> None:
        counts = (
            self.observation_count,
            self.positive_count,
            self.negative_count,
            self.unresolved_count,
            self.censored_count,
            self.missing_propensity_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("adaptive training metric counts are invalid")
        if self.observation_count != sum(counts[1:5]):
            raise ValueError("adaptive training label counts are inconsistent")
        if self.missing_propensity_count > self.observation_count:
            raise ValueError("adaptive missing propensity count is inconsistent")
        rate = _finite(
            self.missing_propensity_rate,
            "adaptive missing propensity metric",
        )
        expected = (
            self.missing_propensity_count / self.observation_count
            if self.observation_count
            else 0.0
        )
        if round(rate, 12) != round(expected, 12):
            raise ValueError("adaptive missing propensity rate is inconsistent")
        object.__setattr__(self, "missing_propensity_rate", rate)

    def payload(self) -> dict[str, object]:
        return {
            "observation_count": self.observation_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "unresolved_count": self.unresolved_count,
            "censored_count": self.censored_count,
            "missing_propensity_count": self.missing_propensity_count,
            "missing_propensity_rate": self.missing_propensity_rate,
        }


@dataclass(frozen=True, slots=True)
class AdaptivePolicyArtifact:
    artifact_id: str
    policy_version: str
    parent_policy_version: str
    dataset_id: str
    dataset_payload_digest: str
    dataset_version: str
    feature_schema: str
    label_schema: str
    window_version: str
    training_config_digest: str
    training_seed: int
    objective: str
    regularization: float
    route_backend: str
    invocation_boundary: str
    parameters: tuple[AdaptiveParameterUpdate, ...]
    prompt_refs: tuple[str, ...]
    training_example_ids: tuple[str, ...]
    metrics: AdaptiveTrainingMetrics
    provenance_example_ids: tuple[str, ...]
    provenance_operation_ids: tuple[str, ...]
    state: AdaptivePolicyState
    content_digest: str
    artifact_schema: str = ADAPTIVE_POLICY_ARTIFACT_SCHEMA
    schema_version: int = ADAPTIVE_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != ADAPTIVE_POLICY_SCHEMA_VERSION
            or self.artifact_schema != ADAPTIVE_POLICY_ARTIFACT_SCHEMA
        ):
            raise ValueError("unsupported adaptive policy artifact schema")
        object.__setattr__(self, "state", AdaptivePolicyState(self.state))
        if self.state != AdaptivePolicyState.PROPOSAL:
            raise ValueError("new adaptive policy artifact must be a proposal")
        for name in (
            "artifact_id",
            "policy_version",
            "parent_policy_version",
            "dataset_id",
            "dataset_version",
            "feature_schema",
            "label_schema",
            "window_version",
            "objective",
            "route_backend",
            "invocation_boundary",
            "artifact_schema",
        ):
            _require_identifier(getattr(self, name), f"adaptive artifact {name}")
        for digest in (
            self.dataset_payload_digest,
            self.training_config_digest,
            self.content_digest,
        ):
            if _DIGEST.fullmatch(digest) is None:
                raise ValueError("adaptive policy content digest is invalid")
        if type(self.training_seed) is not int or self.training_seed < 0:
            raise ValueError("adaptive artifact seed is invalid")
        regularization = _finite(self.regularization, "adaptive regularization")
        if regularization < 0:
            raise ValueError("adaptive regularization is invalid")
        object.__setattr__(self, "regularization", regularization)
        if not self.parameters:
            raise ValueError("adaptive artifact requires parameters")
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "prompt_refs", tuple(self.prompt_refs))
        parameter_ids = tuple(item.parameter_id for item in self.parameters)
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("adaptive artifact parameters must be unique")
        if len(self.prompt_refs) != len(self.parameters):
            raise ValueError("adaptive artifact prompt refs are incomplete")
        for reference in self.prompt_refs:
            _require_identifier(reference, "adaptive artifact prompt reference")
        _require_ids(self.training_example_ids, "adaptive artifact training examples")
        _require_ids(self.provenance_example_ids, "adaptive artifact examples")
        _require_ids(self.provenance_operation_ids, "adaptive artifact operations")
        expected_examples = tuple(sorted({
            example_id
            for update in self.parameters
            for example_id in update.attributed_example_ids
        }))
        expected_operations = tuple(sorted({
            operation_id
            for update in self.parameters
            for operation_id in update.failure_subgraph_operation_ids
        }))
        if (
            self.provenance_example_ids != expected_examples
            or self.provenance_operation_ids != expected_operations
            or not set(self.provenance_example_ids).issubset(self.training_example_ids)
        ):
            raise ValueError("adaptive artifact provenance is incomplete")
        expected_digest = _digest(self.identity_payload())
        if self.content_digest != expected_digest:
            raise ValueError("adaptive policy content digest mismatch")
        if self.policy_version != self.expected_policy_version():
            raise ValueError("adaptive policy version conflicts with its content")
        if self.artifact_id != f"policy-artifact.{expected_digest[:40]}":
            raise ValueError("adaptive artifact ID conflicts with its content")

    def version_basis(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_schema": self.artifact_schema,
            "parent_policy_version": self.parent_policy_version,
            "dataset_id": self.dataset_id,
            "dataset_payload_digest": self.dataset_payload_digest,
            "dataset_version": self.dataset_version,
            "feature_schema": self.feature_schema,
            "label_schema": self.label_schema,
            "window_version": self.window_version,
            "training_config_digest": self.training_config_digest,
            "training_seed": self.training_seed,
            "objective": self.objective,
            "regularization": self.regularization,
            "route_backend": self.route_backend,
            "invocation_boundary": self.invocation_boundary,
            "parameters": [item.payload() for item in self.parameters],
            "prompt_refs": list(self.prompt_refs),
            "training_example_ids": list(self.training_example_ids),
            "metrics": self.metrics.payload(),
            "provenance_example_ids": list(self.provenance_example_ids),
            "provenance_operation_ids": list(self.provenance_operation_ids),
            "state": self.state.value,
        }

    def expected_policy_version(self) -> str:
        return f"semantic-adaptive.{_digest(self.version_basis())[:24]}"

    def identity_payload(self) -> dict[str, object]:
        return {**self.version_basis(), "policy_version": self.policy_version}

    def payload(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "artifact_id": self.artifact_id,
            "content_digest": self.content_digest,
        }

    @classmethod
    def create(
        cls,
        *,
        dataset: DelayedFeedbackDataset,
        gate: FeedbackDatasetStageGate,
        config: AdaptiveTrainingConfig,
        parameters: tuple[AdaptiveParameterUpdate, ...],
        metrics: AdaptiveTrainingMetrics,
    ) -> "AdaptivePolicyArtifact":
        provenance_examples = tuple(sorted({
            example_id
            for update in parameters
            for example_id in update.attributed_example_ids
        }))
        provenance_operations = tuple(sorted({
            operation_id
            for update in parameters
            for operation_id in update.failure_subgraph_operation_ids
        }))
        prompt_refs = tuple(spec.prompt_ref for spec in config.parameters)
        training_example_ids = tuple(
            config.training_example_ids
            or (example.example_id for example in dataset.examples)
        )
        basis = {
            "schema_version": ADAPTIVE_POLICY_SCHEMA_VERSION,
            "artifact_schema": ADAPTIVE_POLICY_ARTIFACT_SCHEMA,
            "parent_policy_version": config.parent_policy_version,
            "dataset_id": dataset.dataset_id,
            "dataset_payload_digest": gate.dataset_payload_digest,
            "dataset_version": dataset.config.dataset_version,
            "feature_schema": dataset.config.feature_schema,
            "label_schema": dataset.config.label_schema,
            "window_version": dataset.window.version,
            "training_config_digest": config.digest,
            "training_seed": config.seed,
            "objective": config.objective,
            "regularization": config.l2_regularization,
            "route_backend": config.route_backend,
            "invocation_boundary": config.invocation_boundary,
            "parameters": [item.payload() for item in parameters],
            "prompt_refs": list(prompt_refs),
            "training_example_ids": list(training_example_ids),
            "metrics": metrics.payload(),
            "provenance_example_ids": list(provenance_examples),
            "provenance_operation_ids": list(provenance_operations),
            "state": AdaptivePolicyState.PROPOSAL.value,
        }
        version = f"semantic-adaptive.{_digest(basis)[:24]}"
        identity = {**basis, "policy_version": version}
        content_digest = _digest(identity)
        return cls(
            artifact_id=f"policy-artifact.{content_digest[:40]}",
            policy_version=version,
            parent_policy_version=config.parent_policy_version,
            dataset_id=dataset.dataset_id,
            dataset_payload_digest=gate.dataset_payload_digest,
            dataset_version=dataset.config.dataset_version,
            feature_schema=dataset.config.feature_schema,
            label_schema=dataset.config.label_schema,
            window_version=dataset.window.version,
            training_config_digest=config.digest,
            training_seed=config.seed,
            objective=config.objective,
            regularization=config.l2_regularization,
            route_backend=config.route_backend,
            invocation_boundary=config.invocation_boundary,
            parameters=parameters,
            prompt_refs=prompt_refs,
            training_example_ids=training_example_ids,
            metrics=metrics,
            provenance_example_ids=provenance_examples,
            provenance_operation_ids=provenance_operations,
            state=AdaptivePolicyState.PROPOSAL,
            content_digest=content_digest,
        )


@dataclass(frozen=True, slots=True)
class AdaptivePolicyAudit:
    ok: bool
    issues: tuple[str, ...]
    artifact_id: str
    replay_artifact_id: str | None


def _dataset_identity(dataset: DelayedFeedbackDataset) -> str:
    payload = dataset.payload()
    payload.pop("dataset_id")
    return _stable_id("feedback-dataset", payload)


def _component_operation_ids(
    example: DelayedFeedbackExample,
    component: str,
) -> set[str]:
    if component == "extraction":
        return set(example.extraction_operation_ids)
    if component == "internal_operation":
        return set((
            *example.decision_operation_ids,
            *example.target_resolution_operation_ids,
        ))
    if component == "consolidation":
        return set((
            *example.related_retrieval_operation_ids,
            *example.validation_operation_ids,
            example.mutation_operation_id,
            *example.supersession_operation_ids,
        ))
    if component == "retrieval":
        return set((
            *example.query_operation_ids,
            *example.retrieval_operation_ids,
            *example.injection_operation_ids,
            *example.use_operation_ids,
            *example.outcome_operation_ids,
        ))
    raise ValueError("unknown adaptive semantic component")


class DeterministicAdaptivePolicyLearner:
    """Fit bounded threshold changes using only trusted delayed evidence."""

    @staticmethod
    def _validate_inputs(
        dataset: DelayedFeedbackDataset,
        gate: FeedbackDatasetStageGate,
        config: AdaptiveTrainingConfig,
    ) -> None:
        if (
            _dataset_identity(dataset) != dataset.dataset_id
            or gate.dataset_id != dataset.dataset_id
            or gate.dataset_payload_digest != _digest(dataset.payload())
            or gate.replay_dataset_id != dataset.dataset_id
            or gate.actual_config_digest != dataset.config.digest
            or gate.expected_config_digest != dataset.config.digest
        ):
            raise ValueError("adaptive learner dataset evidence mismatch")
        if not gate.ok or gate.issues or not gate.audit.ok:
            raise ValueError("adaptive learner requires an accepted dataset gate")
        if config.parent_policy_version != dataset.config.policy_version:
            raise ValueError("adaptive parent policy is unknown")
        if (
            config.feature_schema != dataset.config.feature_schema
            or config.label_schema != dataset.config.label_schema
            or config.dataset_version != dataset.config.dataset_version
            or config.window_version != dataset.window.version
        ):
            raise ValueError("adaptive learner frozen dataset schema mismatch")
        owned_parameters = {
            parameter_id
            for example in dataset.examples
            for parameter_id in example.policy_parameter_ids
        }
        if any(
            spec.parameter_id not in owned_parameters
            for spec in config.parameters
        ):
            raise ValueError("adaptive learner has an unknown policy parameter")
        available_examples = {example.example_id for example in dataset.examples}
        if config.training_example_ids and not set(
            config.training_example_ids
        ).issubset(available_examples):
            raise ValueError("adaptive learner training membership is invalid")

    def learn(
        self,
        dataset: DelayedFeedbackDataset,
        gate: FeedbackDatasetStageGate,
        config: AdaptiveTrainingConfig,
    ) -> AdaptivePolicyArtifact:
        self._validate_inputs(dataset, gate, config)
        selected_ids = set(config.training_example_ids)
        examples = tuple(
            example
            for example in dataset.examples
            if not selected_ids or example.example_id in selected_ids
        )
        label_counts = {label: 0 for label in FeedbackLabel}
        for example in examples:
            label_counts[example.label] += 1
        missing_propensity_count = sum(
            example.selection_propensity is None for example in examples
        )
        missing_propensity_rate = (
            missing_propensity_count / len(examples)
            if examples
            else 0.0
        )
        distribution_shift = (
            missing_propensity_rate > config.maximum_missing_propensity_rate
        )
        updates = []
        for spec in config.parameters:
            positive_ids = set()
            negative_ids = set()
            failure_ids = set()
            for example in examples:
                if spec.parameter_id not in example.policy_parameter_ids:
                    continue
                if example.label == FeedbackLabel.POSITIVE:
                    positive_ids.add(example.example_id)
                elif example.label == FeedbackLabel.NEGATIVE:
                    matched = set(example.failure_subgraph_operation_ids) & (
                        _component_operation_ids(example, spec.component)
                    )
                    if matched:
                        negative_ids.add(example.example_id)
                        failure_ids.update(matched)
            resolved = len(positive_ids) + len(negative_ids)
            posterior_negative = (
                len(negative_ids) + config.prior_negative
            ) / (
                resolved + config.prior_positive + config.prior_negative
            )
            if distribution_shift:
                fallback = AdaptiveFallbackReason.DISTRIBUTION_SHIFT
            elif resolved == 0:
                fallback = AdaptiveFallbackReason.MISSING_FEATURE
            elif resolved < config.minimum_resolved_examples:
                fallback = AdaptiveFallbackReason.LOW_SAMPLE
            else:
                fallback = AdaptiveFallbackReason.NONE
            delta = 0.0
            if fallback == AdaptiveFallbackReason.NONE:
                raw_delta = (
                    posterior_negative - 0.5
                ) / (1.0 + config.l2_regularization)
                delta = max(
                    -config.maximum_parameter_delta,
                    min(config.maximum_parameter_delta, raw_delta),
                )
                delta = max(-spec.baseline_value, min(1.0 - spec.baseline_value, delta))
            proposed = spec.baseline_value + delta
            updates.append(AdaptiveParameterUpdate(
                parameter_id=spec.parameter_id,
                name=spec.name,
                component=spec.component,
                baseline_value=spec.baseline_value,
                proposed_value=round(proposed, 12),
                delta=round(delta, 12),
                posterior_negative_rate=round(posterior_negative, 12),
                positive_count=len(positive_ids),
                negative_count=len(negative_ids),
                fallback_reason=fallback,
                attributed_example_ids=tuple(sorted(positive_ids | negative_ids)),
                failure_subgraph_operation_ids=tuple(sorted(failure_ids)),
            ))
        metrics = AdaptiveTrainingMetrics(
            observation_count=len(examples),
            positive_count=label_counts[FeedbackLabel.POSITIVE],
            negative_count=label_counts[FeedbackLabel.NEGATIVE],
            unresolved_count=label_counts[FeedbackLabel.UNRESOLVED],
            censored_count=label_counts[FeedbackLabel.CENSORED],
            missing_propensity_count=missing_propensity_count,
            missing_propensity_rate=round(missing_propensity_rate, 12),
        )
        return AdaptivePolicyArtifact.create(
            dataset=dataset,
            gate=gate,
            config=config,
            parameters=tuple(updates),
            metrics=metrics,
        )

    def audit(
        self,
        artifact: AdaptivePolicyArtifact,
        dataset: DelayedFeedbackDataset,
        gate: FeedbackDatasetStageGate,
        config: AdaptiveTrainingConfig,
    ) -> AdaptivePolicyAudit:
        issues = set()
        replay = None
        try:
            replay = self.learn(dataset, gate, config)
        except (TypeError, ValueError):
            issues.add("artifact_replay_failed")
        if replay is not None and artifact.payload() != replay.payload():
            issues.add("artifact_replay_mismatch")
        return AdaptivePolicyAudit(
            ok=not issues,
            issues=tuple(sorted(issues)),
            artifact_id=artifact.artifact_id,
            replay_artifact_id=None if replay is None else replay.artifact_id,
        )
