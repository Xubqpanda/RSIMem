"""Held-out validation and lifecycle coordination for adaptive policies."""

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

from .adaptive_policy import (
    AdaptiveFallbackReason,
    AdaptivePolicyArtifact,
    AdaptivePolicyState,
    AdaptiveParameterName,
    AdaptiveParameterUpdate,
    DeterministicAdaptivePolicyLearner,
    _component_operation_ids,
)
from .adaptive_policy_store import (
    AdaptivePolicyLifecycleRecord,
    JsonAdaptivePolicyStore,
)
from .feedback_dataset import (
    DelayedFeedbackDataset,
    FeedbackDatasetStageGate,
    FeedbackLabel,
)


ADAPTIVE_VALIDATION_SCHEMA_VERSION = 1
ADAPTIVE_SPLIT_SCHEMA = "semantic-adaptive-time-split-v1"
ADAPTIVE_CRITERIA_VERSION = "semantic-adaptive-acceptance-v1"
ADAPTIVE_DECISION_SCHEMA = "semantic-adaptive-validation-decision-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


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


class AdaptiveSplitStrategy(StrEnum):
    TIME_ORDERED_EPISODE = "time_ordered_episode"


@dataclass(frozen=True, slots=True)
class AdaptiveSplitConfig:
    validation_group_count: int = 1
    strategy: AdaptiveSplitStrategy = AdaptiveSplitStrategy.TIME_ORDERED_EPISODE
    split_schema: str = ADAPTIVE_SPLIT_SCHEMA
    schema_version: int = ADAPTIVE_VALIDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy", AdaptiveSplitStrategy(self.strategy))
        if self.schema_version != ADAPTIVE_VALIDATION_SCHEMA_VERSION:
            raise ValueError("unsupported adaptive split schema")
        if self.split_schema != ADAPTIVE_SPLIT_SCHEMA:
            raise ValueError("adaptive split identity is not frozen")
        if type(self.validation_group_count) is not int or (
            self.validation_group_count < 1
        ):
            raise ValueError("adaptive validation group count must be positive")

    @property
    def digest(self) -> str:
        return _digest(self.payload())

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "split_schema": self.split_schema,
            "strategy": self.strategy.value,
            "validation_group_count": self.validation_group_count,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveValidationSplit:
    split_id: str
    dataset_id: str
    dataset_payload_digest: str
    config_digest: str
    strategy: AdaptiveSplitStrategy
    training_example_ids: tuple[str, ...]
    validation_example_ids: tuple[str, ...]
    validation_membership: tuple[tuple[str, str], ...]
    training_episode_ids: tuple[str, ...]
    validation_episode_ids: tuple[str, ...]
    training_task_ids: tuple[str, ...]
    validation_task_ids: tuple[str, ...]
    training_cutoff_example_id: str
    observation_cutoff_operation_id: str
    split_digest: str
    split_schema: str = ADAPTIVE_SPLIT_SCHEMA
    schema_version: int = ADAPTIVE_VALIDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy", AdaptiveSplitStrategy(self.strategy))
        if (
            self.schema_version != ADAPTIVE_VALIDATION_SCHEMA_VERSION
            or self.split_schema != ADAPTIVE_SPLIT_SCHEMA
        ):
            raise ValueError("unsupported adaptive validation split schema")
        for value in (
            self.split_id,
            self.dataset_id,
            self.training_cutoff_example_id,
            self.observation_cutoff_operation_id,
            self.split_schema,
        ):
            _require_identifier(value, "adaptive split identity")
        for digest in (
            self.dataset_payload_digest,
            self.config_digest,
            self.split_digest,
        ):
            if _DIGEST.fullmatch(digest) is None:
                raise ValueError("adaptive split digest is invalid")
        for values in (
            self.training_example_ids,
            self.validation_example_ids,
            self.training_episode_ids,
            self.validation_episode_ids,
            self.training_task_ids,
            self.validation_task_ids,
        ):
            _require_ids(values, "adaptive split membership")
        if not self.training_example_ids or not self.validation_example_ids:
            raise ValueError("adaptive split requires train and validation members")
        if set(self.training_example_ids) & set(self.validation_example_ids):
            raise ValueError("adaptive train and validation examples overlap")
        if set(self.training_episode_ids) & set(self.validation_episode_ids):
            raise ValueError("adaptive train and validation episodes overlap")
        membership_examples = tuple(item[0] for item in self.validation_membership)
        if membership_examples != self.validation_example_ids:
            raise ValueError("adaptive validation example membership is inconsistent")
        for example_id, episode_id in self.validation_membership:
            _require_identifier(example_id, "adaptive validation example")
            _require_identifier(episode_id, "adaptive validation episode")
        if self.training_cutoff_example_id != self.training_example_ids[-1]:
            raise ValueError("adaptive training cutoff is inconsistent")
        if self.split_digest != _digest(self.identity_payload()):
            raise ValueError("adaptive split digest conflicts with membership")
        if self.split_id != f"adaptive-split.{self.split_digest[:40]}":
            raise ValueError("adaptive split ID conflicts with membership")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "split_schema": self.split_schema,
            "dataset_id": self.dataset_id,
            "dataset_payload_digest": self.dataset_payload_digest,
            "config_digest": self.config_digest,
            "strategy": self.strategy.value,
            "training_example_ids": list(self.training_example_ids),
            "validation_example_ids": list(self.validation_example_ids),
            "validation_membership": [
                list(item) for item in self.validation_membership
            ],
            "training_episode_ids": list(self.training_episode_ids),
            "validation_episode_ids": list(self.validation_episode_ids),
            "training_task_ids": list(self.training_task_ids),
            "validation_task_ids": list(self.validation_task_ids),
            "training_cutoff_example_id": self.training_cutoff_example_id,
            "observation_cutoff_operation_id": (
                self.observation_cutoff_operation_id
            ),
        }

    def payload(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "split_id": self.split_id,
            "split_digest": self.split_digest,
        }

    @classmethod
    def from_payload(cls, value: object) -> "AdaptiveValidationSplit":
        expected = {
            "schema_version",
            "split_schema",
            "split_id",
            "dataset_id",
            "dataset_payload_digest",
            "config_digest",
            "strategy",
            "training_example_ids",
            "validation_example_ids",
            "validation_membership",
            "training_episode_ids",
            "validation_episode_ids",
            "training_task_ids",
            "validation_task_ids",
            "training_cutoff_example_id",
            "observation_cutoff_operation_id",
            "split_digest",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("malformed adaptive validation split")
        try:
            return cls(
                split_id=value["split_id"],
                dataset_id=value["dataset_id"],
                dataset_payload_digest=value["dataset_payload_digest"],
                config_digest=value["config_digest"],
                strategy=AdaptiveSplitStrategy(value["strategy"]),
                training_example_ids=tuple(value["training_example_ids"]),
                validation_example_ids=tuple(value["validation_example_ids"]),
                validation_membership=tuple(
                    tuple(item) for item in value["validation_membership"]
                ),
                training_episode_ids=tuple(value["training_episode_ids"]),
                validation_episode_ids=tuple(value["validation_episode_ids"]),
                training_task_ids=tuple(value["training_task_ids"]),
                validation_task_ids=tuple(value["validation_task_ids"]),
                training_cutoff_example_id=value["training_cutoff_example_id"],
                observation_cutoff_operation_id=(
                    value["observation_cutoff_operation_id"]
                ),
                split_digest=value["split_digest"],
                split_schema=value["split_schema"],
                schema_version=value["schema_version"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed adaptive validation split") from exc


class TimeOrderedAdaptiveSplitter:
    def split(
        self,
        dataset: DelayedFeedbackDataset,
        gate: FeedbackDatasetStageGate,
        config: AdaptiveSplitConfig = AdaptiveSplitConfig(),
    ) -> AdaptiveValidationSplit:
        if (
            not gate.ok
            or gate.dataset_id != dataset.dataset_id
            or gate.replay_dataset_id != dataset.dataset_id
            or gate.dataset_payload_digest != _digest(dataset.payload())
        ):
            raise ValueError("adaptive split requires accepted dataset evidence")
        runs_by_episode: dict[str, set[str]] = {}
        for example in dataset.examples:
            runs_by_episode.setdefault(example.source_episode_id, set()).add(
                example.run_id
            )

        def group_id(example: object) -> str:
            episode_id = example.source_episode_id
            return (
                f"{example.run_id}:{episode_id}"
                if len(runs_by_episode[episode_id]) > 1
                else episode_id
            )

        groups: dict[str, list[object]] = {}
        for example in dataset.examples:
            groups.setdefault(group_id(example), []).append(example)
        ordered_groups = tuple(groups)
        if len(ordered_groups) <= config.validation_group_count:
            raise ValueError("adaptive split has insufficient chronological groups")
        validation_groups = set(ordered_groups[-config.validation_group_count :])
        training = tuple(
            example
            for example in dataset.examples
            if group_id(example) not in validation_groups
        )
        validation = tuple(
            example
            for example in dataset.examples
            if group_id(example) in validation_groups
        )
        training_episodes = tuple(dict.fromkeys(
            group_id(example) for example in training
        ))
        validation_episodes = tuple(dict.fromkeys(
            group_id(example) for example in validation
        ))
        training_tasks = tuple(dict.fromkeys(
            task_id for example in training for task_id in example.task_ids
        ))
        validation_tasks = tuple(dict.fromkeys(
            task_id for example in validation for task_id in example.task_ids
        ))
        identity = {
            "schema_version": ADAPTIVE_VALIDATION_SCHEMA_VERSION,
            "split_schema": ADAPTIVE_SPLIT_SCHEMA,
            "dataset_id": dataset.dataset_id,
            "dataset_payload_digest": gate.dataset_payload_digest,
            "config_digest": config.digest,
            "strategy": config.strategy.value,
            "training_example_ids": [item.example_id for item in training],
            "validation_example_ids": [item.example_id for item in validation],
            "validation_membership": [
                [item.example_id, group_id(item)] for item in validation
            ],
            "training_episode_ids": list(training_episodes),
            "validation_episode_ids": list(validation_episodes),
            "training_task_ids": list(training_tasks),
            "validation_task_ids": list(validation_tasks),
            "training_cutoff_example_id": training[-1].example_id,
            "observation_cutoff_operation_id": dataset.window.cutoff_operation_id,
        }
        digest = _digest(identity)
        return AdaptiveValidationSplit(
            split_id=f"adaptive-split.{digest[:40]}",
            dataset_id=dataset.dataset_id,
            dataset_payload_digest=gate.dataset_payload_digest,
            config_digest=config.digest,
            strategy=config.strategy,
            training_example_ids=tuple(item.example_id for item in training),
            validation_example_ids=tuple(item.example_id for item in validation),
            validation_membership=tuple(
                (item.example_id, group_id(item)) for item in validation
            ),
            training_episode_ids=training_episodes,
            validation_episode_ids=validation_episodes,
            training_task_ids=training_tasks,
            validation_task_ids=validation_tasks,
            training_cutoff_example_id=training[-1].example_id,
            observation_cutoff_operation_id=dataset.window.cutoff_operation_id,
            split_digest=digest,
        )


@dataclass(frozen=True, slots=True)
class AdaptiveAcceptanceCriteria:
    minimum_quality_delta: float = 0.0
    maximum_cost_delta: float = 0.15
    maximum_stability_delta: float = 0.25
    maximum_uncertainty: float = 0.75
    minimum_validation_examples: int = 1
    resource_cost_cap: float = 100_000.0
    allow_fallback_parameters: bool = False
    criteria_version: str = ADAPTIVE_CRITERIA_VERSION
    schema_version: int = ADAPTIVE_VALIDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ADAPTIVE_VALIDATION_SCHEMA_VERSION:
            raise ValueError("unsupported adaptive acceptance schema")
        if self.criteria_version != ADAPTIVE_CRITERIA_VERSION:
            raise ValueError("adaptive acceptance criteria identity is not frozen")
        quality = _finite(self.minimum_quality_delta, "minimum quality delta")
        cost = _finite(self.maximum_cost_delta, "maximum cost delta")
        stability = _finite(
            self.maximum_stability_delta,
            "maximum stability delta",
        )
        uncertainty = _finite(self.maximum_uncertainty, "maximum uncertainty")
        cap = _finite(self.resource_cost_cap, "resource cost cap")
        if (
            not -1.0 <= quality <= 1.0
            or not 0.0 <= cost <= 1.0
            or not 0.0 <= stability <= 1.0
            or not 0.0 <= uncertainty <= 1.0
            or cap <= 0
        ):
            raise ValueError("adaptive acceptance criteria bounds are invalid")
        if (
            type(self.minimum_validation_examples) is not int
            or self.minimum_validation_examples < 1
        ):
            raise ValueError("minimum validation examples must be positive")
        if type(self.allow_fallback_parameters) is not bool:
            raise TypeError("allow fallback parameters must be bool")
        object.__setattr__(self, "minimum_quality_delta", quality)
        object.__setattr__(self, "maximum_cost_delta", cost)
        object.__setattr__(self, "maximum_stability_delta", stability)
        object.__setattr__(self, "maximum_uncertainty", uncertainty)
        object.__setattr__(self, "resource_cost_cap", cap)

    @property
    def digest(self) -> str:
        return _digest(self.payload())

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "criteria_version": self.criteria_version,
            "minimum_quality_delta": self.minimum_quality_delta,
            "maximum_cost_delta": self.maximum_cost_delta,
            "maximum_stability_delta": self.maximum_stability_delta,
            "maximum_uncertainty": self.maximum_uncertainty,
            "minimum_validation_examples": self.minimum_validation_examples,
            "resource_cost_cap": self.resource_cost_cap,
            "allow_fallback_parameters": self.allow_fallback_parameters,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveValidationMetrics:
    validation_example_count: int
    resolved_comparison_count: int
    parent_quality: float | None
    proposal_quality: float | None
    quality_delta: float | None
    observed_lifecycle_cost: float
    proposal_change_cost: float
    cost_delta: float
    stability_delta: float
    uncertainty: float
    missing_resource_count: int
    fallback_parameter_count: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in (
            self.validation_example_count,
            self.resolved_comparison_count,
            self.missing_resource_count,
            self.fallback_parameter_count,
        )):
            raise ValueError("adaptive validation counts are invalid")
        optional = (self.parent_quality, self.proposal_quality, self.quality_delta)
        if self.resolved_comparison_count == 0:
            if any(value is not None for value in optional):
                raise ValueError("unresolved validation cannot carry quality")
        elif any(value is None for value in optional):
            raise ValueError("resolved validation requires quality metrics")
        for value in optional:
            if value is not None and not -1.0 <= _finite(value, "quality metric") <= 1.0:
                raise ValueError("adaptive quality metric is out of range")
        for name in (
            "observed_lifecycle_cost",
            "proposal_change_cost",
            "cost_delta",
            "stability_delta",
            "uncertainty",
        ):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError("adaptive validation metric is out of range")

    def payload(self) -> dict[str, object]:
        return {
            "validation_example_count": self.validation_example_count,
            "resolved_comparison_count": self.resolved_comparison_count,
            "parent_quality": self.parent_quality,
            "proposal_quality": self.proposal_quality,
            "quality_delta": self.quality_delta,
            "observed_lifecycle_cost": self.observed_lifecycle_cost,
            "proposal_change_cost": self.proposal_change_cost,
            "cost_delta": self.cost_delta,
            "stability_delta": self.stability_delta,
            "uncertainty": self.uncertainty,
            "missing_resource_count": self.missing_resource_count,
            "fallback_parameter_count": self.fallback_parameter_count,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveValidationDecision:
    decision_id: str
    artifact_id: str
    policy_version: str
    parent_policy_version: str
    dataset_id: str
    split_id: str
    criteria_digest: str
    training_example_ids: tuple[str, ...]
    validation_example_ids: tuple[str, ...]
    used_policy_versions: tuple[str, str]
    metrics: AdaptiveValidationMetrics
    accepted: bool
    resulting_state: AdaptivePolicyState
    reason_codes: tuple[str, ...]
    content_digest: str
    decision_schema: str = ADAPTIVE_DECISION_SCHEMA
    schema_version: int = ADAPTIVE_VALIDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "resulting_state", AdaptivePolicyState(
            self.resulting_state
        ))
        if (
            self.schema_version != ADAPTIVE_VALIDATION_SCHEMA_VERSION
            or self.decision_schema != ADAPTIVE_DECISION_SCHEMA
        ):
            raise ValueError("unsupported adaptive validation decision schema")
        for value in (
            self.decision_id,
            self.artifact_id,
            self.policy_version,
            self.parent_policy_version,
            self.dataset_id,
            self.split_id,
            self.decision_schema,
        ):
            _require_identifier(value, "adaptive validation decision identity")
        for digest in (self.criteria_digest, self.content_digest):
            if _DIGEST.fullmatch(digest) is None:
                raise ValueError("adaptive validation decision digest is invalid")
        _require_ids(self.training_example_ids, "adaptive decision training members")
        _require_ids(self.validation_example_ids, "adaptive decision validation members")
        if set(self.training_example_ids) & set(self.validation_example_ids):
            raise ValueError("adaptive decision memberships overlap")
        if type(self.accepted) is not bool:
            raise TypeError("adaptive validation acceptance must be bool")
        expected_state = (
            AdaptivePolicyState.VALIDATED
            if self.accepted
            else AdaptivePolicyState.REJECTED
        )
        if self.resulting_state != expected_state:
            raise ValueError("adaptive validation result state is inconsistent")
        if self.used_policy_versions != (
            self.parent_policy_version,
            self.policy_version,
        ):
            raise ValueError("adaptive validation compared policy versions differ")
        if not self.reason_codes or any(
            _REASON_CODE.fullmatch(value) is None for value in self.reason_codes
        ):
            raise ValueError("adaptive validation reasons must be machine-readable")
        expected_digest = _digest(self.identity_payload())
        if self.content_digest != expected_digest:
            raise ValueError("adaptive validation decision digest mismatch")
        if self.decision_id != f"adaptive-validation.{expected_digest[:40]}":
            raise ValueError("adaptive validation decision ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "decision_schema": self.decision_schema,
            "artifact_id": self.artifact_id,
            "policy_version": self.policy_version,
            "parent_policy_version": self.parent_policy_version,
            "dataset_id": self.dataset_id,
            "split_id": self.split_id,
            "criteria_digest": self.criteria_digest,
            "training_example_ids": list(self.training_example_ids),
            "validation_example_ids": list(self.validation_example_ids),
            "used_policy_versions": list(self.used_policy_versions),
            "metrics": self.metrics.payload(),
            "accepted": self.accepted,
            "resulting_state": self.resulting_state.value,
            "reason_codes": list(self.reason_codes),
        }

    def payload(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "decision_id": self.decision_id,
            "content_digest": self.content_digest,
        }


def _resource_cost(dataset: DelayedFeedbackDataset, ids: set[str], cap: float) -> tuple[float, int]:
    total = 0
    missing = 0
    for example in dataset.examples:
        if example.example_id not in ids:
            continue
        usage = example.resources
        for name in (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "duration_ms",
        ):
            value = getattr(usage, name)
            if value is None:
                missing += 1
            else:
                total += value
        total += (
            usage.model_requests
            + usage.retry_count
            + usage.storage_bytes
        )
    return min(1.0, total / cap), missing


class AdaptivePolicyValidator:
    """Replay one predeclared acceptance decision without official scores."""

    def evaluate(
        self,
        artifact: AdaptivePolicyArtifact,
        dataset: DelayedFeedbackDataset,
        gate: FeedbackDatasetStageGate,
        split: AdaptiveValidationSplit,
        criteria: AdaptiveAcceptanceCriteria,
    ) -> AdaptiveValidationDecision:
        if (
            not gate.ok
            or gate.dataset_id != dataset.dataset_id
            or gate.dataset_payload_digest != _digest(dataset.payload())
            or split.dataset_id != dataset.dataset_id
            or split.dataset_payload_digest != gate.dataset_payload_digest
            or artifact.dataset_id != dataset.dataset_id
            or artifact.dataset_payload_digest != gate.dataset_payload_digest
        ):
            raise ValueError("adaptive validation dataset binding mismatch")
        all_ids = {example.example_id for example in dataset.examples}
        if set(split.training_example_ids) | set(split.validation_example_ids) != all_ids:
            raise ValueError("adaptive validation split membership is incomplete")
        if artifact.training_example_ids != split.training_example_ids:
            raise ValueError("adaptive artifact was not trained on split membership")
        validation_ids = set(split.validation_example_ids)
        parent_errors = []
        proposal_errors = []
        resolved_examples = set()
        missing_propensity = 0
        for example in dataset.examples:
            if example.example_id not in validation_ids:
                continue
            if example.selection_propensity is None:
                missing_propensity += 1
            for update in artifact.parameters:
                if update.parameter_id not in example.policy_parameter_ids:
                    continue
                target = None
                if example.label == FeedbackLabel.POSITIVE:
                    target = 0.0
                elif example.label == FeedbackLabel.NEGATIVE and (
                    set(example.failure_subgraph_operation_ids)
                    & _component_operation_ids(example, update.component)
                ):
                    target = 1.0
                if target is None:
                    continue
                resolved_examples.add(example.example_id)
                parent_errors.append((update.baseline_value - target) ** 2)
                proposal_errors.append((update.proposed_value - target) ** 2)
        resolved = len(parent_errors)
        parent_quality = (
            1.0 - sum(parent_errors) / resolved if resolved else None
        )
        proposal_quality = (
            1.0 - sum(proposal_errors) / resolved if resolved else None
        )
        quality_delta = (
            proposal_quality - parent_quality
            if parent_quality is not None and proposal_quality is not None
            else None
        )
        observed_cost, missing_resources = _resource_cost(
            dataset,
            validation_ids,
            criteria.resource_cost_cap,
        )
        change_cost = sum(
            abs(update.delta) for update in artifact.parameters
        ) / len(artifact.parameters)
        stability = max(abs(update.delta) for update in artifact.parameters)
        sample_uncertainty = 1.0 / math.sqrt(len(resolved_examples) + 1)
        propensity_uncertainty = (
            missing_propensity / len(validation_ids) if validation_ids else 1.0
        )
        resource_uncertainty = (
            min(1.0, missing_resources / (len(validation_ids) * 5))
            if validation_ids
            else 1.0
        )
        uncertainty = max(
            sample_uncertainty,
            propensity_uncertainty,
            resource_uncertainty,
        )
        fallback_count = sum(
            update.fallback_reason != AdaptiveFallbackReason.NONE
            for update in artifact.parameters
        )
        metrics = AdaptiveValidationMetrics(
            validation_example_count=len(validation_ids),
            resolved_comparison_count=resolved,
            parent_quality=(
                None if parent_quality is None else round(parent_quality, 12)
            ),
            proposal_quality=(
                None if proposal_quality is None else round(proposal_quality, 12)
            ),
            quality_delta=(
                None if quality_delta is None else round(quality_delta, 12)
            ),
            observed_lifecycle_cost=round(observed_cost, 12),
            proposal_change_cost=round(change_cost, 12),
            cost_delta=round(change_cost, 12),
            stability_delta=round(stability, 12),
            uncertainty=round(uncertainty, 12),
            missing_resource_count=missing_resources,
            fallback_parameter_count=fallback_count,
        )
        reasons = []
        if len(resolved_examples) < criteria.minimum_validation_examples:
            reasons.append("insufficient_validation")
        if quality_delta is None or quality_delta < criteria.minimum_quality_delta:
            reasons.append("quality_regression")
        if change_cost > criteria.maximum_cost_delta:
            reasons.append("cost_regression")
        if stability > criteria.maximum_stability_delta:
            reasons.append("stability_regression")
        if uncertainty > criteria.maximum_uncertainty:
            reasons.append("uncertainty_exceeded")
        if fallback_count and not criteria.allow_fallback_parameters:
            reasons.append("proposal_fallback")
        accepted = not reasons
        reason_codes = ("acceptance_criteria_passed",) if accepted else tuple(reasons)
        identity = {
            "schema_version": ADAPTIVE_VALIDATION_SCHEMA_VERSION,
            "decision_schema": ADAPTIVE_DECISION_SCHEMA,
            "artifact_id": artifact.artifact_id,
            "policy_version": artifact.policy_version,
            "parent_policy_version": artifact.parent_policy_version,
            "dataset_id": dataset.dataset_id,
            "split_id": split.split_id,
            "criteria_digest": criteria.digest,
            "training_example_ids": list(split.training_example_ids),
            "validation_example_ids": list(split.validation_example_ids),
            "used_policy_versions": [
                artifact.parent_policy_version,
                artifact.policy_version,
            ],
            "metrics": metrics.payload(),
            "accepted": accepted,
            "resulting_state": (
                AdaptivePolicyState.VALIDATED.value
                if accepted
                else AdaptivePolicyState.REJECTED.value
            ),
            "reason_codes": list(reason_codes),
        }
        digest = _digest(identity)
        return AdaptiveValidationDecision(
            decision_id=f"adaptive-validation.{digest[:40]}",
            artifact_id=artifact.artifact_id,
            policy_version=artifact.policy_version,
            parent_policy_version=artifact.parent_policy_version,
            dataset_id=dataset.dataset_id,
            split_id=split.split_id,
            criteria_digest=criteria.digest,
            training_example_ids=split.training_example_ids,
            validation_example_ids=split.validation_example_ids,
            used_policy_versions=(
                artifact.parent_policy_version,
                artifact.policy_version,
            ),
            metrics=metrics,
            accepted=accepted,
            resulting_state=(
                AdaptivePolicyState.VALIDATED
                if accepted
                else AdaptivePolicyState.REJECTED
            ),
            reason_codes=reason_codes,
            content_digest=digest,
        )

    def replay_matches(
        self,
        decision: AdaptiveValidationDecision,
        artifact: AdaptivePolicyArtifact,
        dataset: DelayedFeedbackDataset,
        gate: FeedbackDatasetStageGate,
        split: AdaptiveValidationSplit,
        criteria: AdaptiveAcceptanceCriteria,
    ) -> bool:
        return decision == self.evaluate(
            artifact,
            dataset,
            gate,
            split,
            criteria,
        )


class JsonAdaptiveValidationDecisionStore:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    @contextmanager
    def _lock(self, operation: int):
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".validation.lock"
        with lock_path.open("w", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def put(self, decision: AdaptiveValidationDecision) -> tuple[Path, bool]:
        path = self.root / f"{decision.decision_id}.json"
        canonical = _canonical(decision.payload()) + "\n"
        with self._lock(fcntl.LOCK_EX):
            if path.exists():
                if path.read_text(encoding="utf-8") != canonical:
                    raise ValueError("adaptive validation decision conflicts with its ID")
                return path, False
            fd, temporary = tempfile.mkstemp(prefix=".validation.", dir=self.root)
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
    def _parse(value: object) -> AdaptiveValidationDecision:
        if not isinstance(value, Mapping):
            raise ValueError("malformed adaptive validation decision")
        expected = {
            "schema_version",
            "decision_schema",
            "decision_id",
            "artifact_id",
            "policy_version",
            "parent_policy_version",
            "dataset_id",
            "split_id",
            "criteria_digest",
            "training_example_ids",
            "validation_example_ids",
            "used_policy_versions",
            "metrics",
            "accepted",
            "resulting_state",
            "reason_codes",
            "content_digest",
        }
        if set(value) != expected or not isinstance(value["metrics"], Mapping):
            raise ValueError("malformed adaptive validation decision")
        metric_fields = {
            "validation_example_count",
            "resolved_comparison_count",
            "parent_quality",
            "proposal_quality",
            "quality_delta",
            "observed_lifecycle_cost",
            "proposal_change_cost",
            "cost_delta",
            "stability_delta",
            "uncertainty",
            "missing_resource_count",
            "fallback_parameter_count",
        }
        if set(value["metrics"]) != metric_fields:
            raise ValueError("malformed adaptive validation metrics")
        try:
            return AdaptiveValidationDecision(
                decision_id=value["decision_id"],
                artifact_id=value["artifact_id"],
                policy_version=value["policy_version"],
                parent_policy_version=value["parent_policy_version"],
                dataset_id=value["dataset_id"],
                split_id=value["split_id"],
                criteria_digest=value["criteria_digest"],
                training_example_ids=tuple(value["training_example_ids"]),
                validation_example_ids=tuple(value["validation_example_ids"]),
                used_policy_versions=tuple(value["used_policy_versions"]),
                metrics=AdaptiveValidationMetrics(**value["metrics"]),
                accepted=value["accepted"],
                resulting_state=AdaptivePolicyState(value["resulting_state"]),
                reason_codes=tuple(value["reason_codes"]),
                content_digest=value["content_digest"],
                decision_schema=value["decision_schema"],
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed adaptive validation decision") from exc

    def get(self, decision_id: str) -> AdaptiveValidationDecision | None:
        _require_identifier(decision_id, "adaptive validation decision ID")
        path = self.root / f"{decision_id}.json"
        with self._lock(fcntl.LOCK_SH):
            if not path.exists():
                return None
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("malformed adaptive validation decision JSON") from exc
            decision = self._parse(raw)
            if decision.decision_id != decision_id:
                raise ValueError("adaptive validation decision filename mismatch")
            return decision

    def all(self) -> tuple[AdaptiveValidationDecision, ...]:
        with self._lock(fcntl.LOCK_SH):
            paths = tuple(sorted(self.root.glob("adaptive-validation.*.json")))
            decisions = []
            for path in paths:
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError("malformed adaptive validation decision JSON") from exc
                decision = self._parse(raw)
                if path.name != f"{decision.decision_id}.json":
                    raise ValueError("adaptive validation decision filename mismatch")
                decisions.append(decision)
        ids = tuple(item.decision_id for item in decisions)
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate adaptive validation decision")
        return tuple(decisions)


@dataclass(frozen=True, slots=True)
class AdaptiveRuntimePolicyBinding:
    decision_id: str
    actual_policy_version: str
    adaptive: bool

    def __post_init__(self) -> None:
        _require_identifier(self.decision_id, "runtime policy decision")
        _require_identifier(self.actual_policy_version, "runtime policy version")
        if type(self.adaptive) is not bool:
            raise TypeError("runtime adaptive flag must be bool")

    def observer_evidence(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "actual_policy_version": self.actual_policy_version,
            "adaptive": self.adaptive,
        }


class AdaptivePolicyLifecycleCoordinator:
    """Apply deterministic offline screening; matched validation owns activation."""

    def __init__(
        self,
        policy_store: JsonAdaptivePolicyStore,
        decision_store: JsonAdaptiveValidationDecisionStore,
    ) -> None:
        self.policy_store = policy_store
        self.decision_store = decision_store

    def apply(
        self,
        artifact: AdaptivePolicyArtifact,
        decision: AdaptiveValidationDecision,
    ) -> AdaptivePolicyLifecycleRecord:
        if (
            decision.artifact_id != artifact.artifact_id
            or decision.policy_version != artifact.policy_version
            or decision.parent_policy_version != artifact.parent_policy_version
        ):
            raise ValueError("adaptive validation decision and artifact differ")
        self.decision_store.put(decision)
        self.policy_store.register(artifact)
        if not decision.accepted:
            record, _ = self.policy_store.transition(
                artifact.policy_version,
                to_state=AdaptivePolicyState.REJECTED,
                transition_id=f"{decision.decision_id}.rejected",
                reason_code="held_out_validation_failed",
            )
            return record
        self.policy_store.transition(
            artifact.policy_version,
            to_state=AdaptivePolicyState.VALIDATED,
            transition_id=f"{decision.decision_id}.validated",
            reason_code="held_out_validation_passed",
        )
        record = next(
            item
            for item in self.policy_store.snapshot().records
            if item.policy_version == artifact.policy_version
        )
        return record

    def bind_runtime_decision(
        self,
        decision_id: str,
        *,
        fallback_policy_version: str,
    ) -> AdaptiveRuntimePolicyBinding:
        _require_identifier(decision_id, "runtime decision ID")
        _require_identifier(fallback_policy_version, "runtime fallback policy")
        active = self.policy_store.snapshot().active
        return AdaptiveRuntimePolicyBinding(
            decision_id=decision_id,
            actual_policy_version=(
                active.policy_version if active is not None else fallback_policy_version
            ),
            adaptive=active is not None,
        )
