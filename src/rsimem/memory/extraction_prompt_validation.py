"""Prompt-oriented matched validation for extraction policy artifacts."""

from __future__ import annotations

import hashlib
import fcntl
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

from .extraction_feedback import ExtractionFeedbackLabel, ExtractionSetStatus


EXTRACTION_VALIDATION_SCHEMA_VERSION = 1
EXTRACTION_VALIDATION_SCHEMA = "extraction-prompt-validation-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_METRICS = {"harmful_rate", "missed_rate"}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _require_id(value: str, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")


def _require_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be sha256")


def _finite(value: float, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


class ExtractionValidationVariant(StrEnum):
    PARENT = "parent"
    PROPOSAL = "proposal"


class ExtractionValidationSplitRole(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    FINAL = "final"


@dataclass(frozen=True, slots=True)
class ExtractionSplitAssignment:
    role: ExtractionValidationSplitRole
    family_id: str
    task_template_group_id: str
    task_manifest_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", ExtractionValidationSplitRole(self.role))
        _require_id(self.family_id, "split family ID")
        _require_id(self.task_template_group_id, "task template group ID")
        _require_digest(self.task_manifest_digest, "task manifest digest")


@dataclass(frozen=True, slots=True)
class ExtractionPromptValidationSplit:
    split_id: str
    assignments: tuple[ExtractionSplitAssignment, ...]
    schema_version: int = EXTRACTION_VALIDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_id(self.split_id, "extraction validation split ID")
        if self.schema_version != EXTRACTION_VALIDATION_SCHEMA_VERSION:
            raise ValueError("unsupported extraction validation split schema")
        if not self.assignments:
            raise ValueError("extraction validation split requires assignments")
        groups: dict[tuple[str, str], ExtractionValidationSplitRole] = {}
        manifests: dict[str, ExtractionValidationSplitRole] = {}
        for assignment in self.assignments:
            group = (assignment.family_id, assignment.task_template_group_id)
            previous_group = groups.get(group)
            if previous_group is not None:
                raise ValueError("task template group appears more than once")
            groups[group] = assignment.role
            previous_manifest = manifests.get(assignment.task_manifest_digest)
            if previous_manifest is not None and previous_manifest != assignment.role:
                raise ValueError("task manifest digest crosses validation split roles")
            manifests[assignment.task_manifest_digest] = assignment.role

    def permits(self, observation: "ExtractionValidationObservation") -> bool:
        return any(
            assignment.role == ExtractionValidationSplitRole.VALIDATION
            and assignment.family_id == observation.family_id
            and assignment.task_template_group_id
            == observation.task_template_group_id
            and assignment.task_manifest_digest == observation.task_manifest_digest
            for assignment in self.assignments
        )


@dataclass(frozen=True, slots=True)
class ExtractionValidationObservation:
    observation_id: str
    pair_id: str
    variant: ExtractionValidationVariant
    replicate: int
    family_id: str
    task_template_group_id: str
    task_id: str
    run_id: str
    episode_id: str
    extraction_set_id: str
    task_manifest_digest: str
    model_profile_digest: str
    budget_id: str
    persistence_state_digest: str
    extraction_artifact_id: str
    extraction_artifact_digest: str
    extraction_output_digest: str
    label: ExtractionFeedbackLabel
    extraction_status: ExtractionSetStatus
    missed_assessable: bool | None
    schema_failure_count: int = 0
    safety_failure_count: int = 0
    prompt_leakage_failure_count: int = 0
    native_writer_failure_count: int = 0
    schema_version: int = EXTRACTION_VALIDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXTRACTION_VALIDATION_SCHEMA_VERSION:
            raise ValueError("unsupported extraction validation observation schema")
        object.__setattr__(self, "variant", ExtractionValidationVariant(self.variant))
        object.__setattr__(self, "label", ExtractionFeedbackLabel(self.label))
        object.__setattr__(
            self,
            "extraction_status",
            ExtractionSetStatus(self.extraction_status),
        )
        for value, name in (
            (self.observation_id, "validation observation ID"),
            (self.pair_id, "validation pair ID"),
            (self.family_id, "validation family ID"),
            (self.task_template_group_id, "task template group ID"),
            (self.task_id, "validation task ID"),
            (self.run_id, "validation run ID"),
            (self.episode_id, "validation episode ID"),
            (self.extraction_set_id, "validation extraction set ID"),
            (self.budget_id, "validation budget ID"),
            (self.extraction_artifact_id, "extraction artifact ID"),
        ):
            _require_id(value, name)
        for value, name in (
            (self.task_manifest_digest, "task manifest digest"),
            (self.model_profile_digest, "model profile digest"),
            (self.persistence_state_digest, "persistence state digest"),
            (self.extraction_artifact_digest, "extraction artifact digest"),
            (self.extraction_output_digest, "extraction output digest"),
        ):
            _require_digest(value, name)
        if type(self.replicate) is not int or self.replicate < 1:
            raise ValueError("validation replicate must be positive")
        if self.missed_assessable is not None and type(
            self.missed_assessable
        ) is not bool:
            raise TypeError("validation missed-assessable flag must be bool or unknown")
        if self.label == ExtractionFeedbackLabel.MISSED and (
            self.missed_assessable is not True
        ):
            raise ValueError("missed label requires assessable source evidence")
        for count in (
            self.schema_failure_count,
            self.safety_failure_count,
            self.prompt_leakage_failure_count,
            self.native_writer_failure_count,
        ):
            if type(count) is not int or count < 0:
                raise ValueError("validation failure counts must be non-negative")
        pair_identity = self.pair_identity_payload()
        if self.pair_id != f"extraction-pair.{_digest(pair_identity)[:40]}":
            raise ValueError("extraction validation pair ID mismatch")
        identity = {
            **pair_identity,
            "variant": self.variant.value,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "extraction_set_id": self.extraction_set_id,
            "extraction_artifact_id": self.extraction_artifact_id,
            "extraction_artifact_digest": self.extraction_artifact_digest,
            "extraction_output_digest": self.extraction_output_digest,
            "label": self.label.value,
            "extraction_status": self.extraction_status.value,
            "missed_assessable": self.missed_assessable,
            "failure_counts": list(self.failure_counts),
        }
        if self.observation_id != f"extraction-observation.{_digest(identity)[:40]}":
            raise ValueError("extraction validation observation ID mismatch")

    @property
    def failure_counts(self) -> tuple[int, int, int, int]:
        return (
            self.schema_failure_count,
            self.safety_failure_count,
            self.prompt_leakage_failure_count,
            self.native_writer_failure_count,
        )

    def pair_identity_payload(self) -> dict[str, object]:
        return {
            "replicate": self.replicate,
            "family_id": self.family_id,
            "task_template_group_id": self.task_template_group_id,
            "task_id": self.task_id,
            "task_manifest_digest": self.task_manifest_digest,
            "model_profile_digest": self.model_profile_digest,
            "budget_id": self.budget_id,
            "persistence_state_digest": self.persistence_state_digest,
        }

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "observation_id": self.observation_id,
            "pair_id": self.pair_id,
            "variant": self.variant.value,
            "replicate": self.replicate,
            "family_id": self.family_id,
            "task_template_group_id": self.task_template_group_id,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "extraction_set_id": self.extraction_set_id,
            "task_manifest_digest": self.task_manifest_digest,
            "model_profile_digest": self.model_profile_digest,
            "budget_id": self.budget_id,
            "persistence_state_digest": self.persistence_state_digest,
            "extraction_artifact_id": self.extraction_artifact_id,
            "extraction_artifact_digest": self.extraction_artifact_digest,
            "extraction_output_digest": self.extraction_output_digest,
            "label": self.label.value,
            "extraction_status": self.extraction_status.value,
            "missed_assessable": self.missed_assessable,
            "failure_counts": list(self.failure_counts),
        }

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionValidationObservation":
        fields = {
            "schema_version",
            "observation_id",
            "pair_id",
            "variant",
            "replicate",
            "family_id",
            "task_template_group_id",
            "task_id",
            "run_id",
            "episode_id",
            "extraction_set_id",
            "task_manifest_digest",
            "model_profile_digest",
            "budget_id",
            "persistence_state_digest",
            "extraction_artifact_id",
            "extraction_artifact_digest",
            "extraction_output_digest",
            "label",
            "extraction_status",
            "missed_assessable",
            "failure_counts",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != fields
            or not isinstance(value["failure_counts"], list)
            or len(value["failure_counts"]) != 4
        ):
            raise ValueError("malformed extraction validation observation")
        try:
            return cls(
                observation_id=value["observation_id"],
                pair_id=value["pair_id"],
                variant=value["variant"],
                replicate=value["replicate"],
                family_id=value["family_id"],
                task_template_group_id=value["task_template_group_id"],
                task_id=value["task_id"],
                run_id=value["run_id"],
                episode_id=value["episode_id"],
                extraction_set_id=value["extraction_set_id"],
                task_manifest_digest=value["task_manifest_digest"],
                model_profile_digest=value["model_profile_digest"],
                budget_id=value["budget_id"],
                persistence_state_digest=value["persistence_state_digest"],
                extraction_artifact_id=value["extraction_artifact_id"],
                extraction_artifact_digest=value["extraction_artifact_digest"],
                extraction_output_digest=value["extraction_output_digest"],
                label=value["label"],
                extraction_status=value["extraction_status"],
                missed_assessable=value["missed_assessable"],
                schema_failure_count=value["failure_counts"][0],
                safety_failure_count=value["failure_counts"][1],
                prompt_leakage_failure_count=value["failure_counts"][2],
                native_writer_failure_count=value["failure_counts"][3],
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed extraction validation observation") from exc

    @classmethod
    def create(
        cls,
        *,
        variant: ExtractionValidationVariant,
        replicate: int,
        family_id: str,
        task_template_group_id: str,
        task_id: str,
        run_id: str,
        episode_id: str,
        extraction_set_id: str,
        task_manifest_digest: str,
        model_profile_digest: str,
        budget_id: str,
        persistence_state_digest: str,
        extraction_artifact_id: str,
        extraction_artifact_digest: str,
        extraction_output_digest: str,
        label: ExtractionFeedbackLabel,
        extraction_status: ExtractionSetStatus,
        missed_assessable: bool | None,
        failure_counts: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> "ExtractionValidationObservation":
        pair_identity = {
            "replicate": replicate,
            "family_id": family_id,
            "task_template_group_id": task_template_group_id,
            "task_id": task_id,
            "task_manifest_digest": task_manifest_digest,
            "model_profile_digest": model_profile_digest,
            "budget_id": budget_id,
            "persistence_state_digest": persistence_state_digest,
        }
        pair_id = f"extraction-pair.{_digest(pair_identity)[:40]}"
        identity = {
            **pair_identity,
            "variant": ExtractionValidationVariant(variant).value,
            "run_id": run_id,
            "episode_id": episode_id,
            "task_id": task_id,
            "extraction_set_id": extraction_set_id,
            "extraction_artifact_id": extraction_artifact_id,
            "extraction_artifact_digest": extraction_artifact_digest,
            "extraction_output_digest": extraction_output_digest,
            "label": ExtractionFeedbackLabel(label).value,
            "extraction_status": ExtractionSetStatus(extraction_status).value,
            "missed_assessable": missed_assessable,
            "failure_counts": list(failure_counts),
        }
        return cls(
            f"extraction-observation.{_digest(identity)[:40]}",
            pair_id,
            variant,
            replicate,
            family_id,
            task_template_group_id,
            task_id,
            run_id,
            episode_id,
            extraction_set_id,
            task_manifest_digest,
            model_profile_digest,
            budget_id,
            persistence_state_digest,
            extraction_artifact_id,
            extraction_artifact_digest,
            extraction_output_digest,
            label,
            extraction_status,
            missed_assessable,
            *failure_counts,
        )


@dataclass(frozen=True, slots=True)
class ExtractionValidationSafetyEvidence:
    evidence_id: str
    live_feedback_record_id: str
    source_record_id: str
    audit_id: str
    audit_digest: str
    evidence_cutoff_operation_id: str
    complete: bool
    schema_failure_count: int
    safety_failure_count: int
    prompt_leakage_failure_count: int
    native_writer_failure_count: int
    schema_version: int = EXTRACTION_VALIDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXTRACTION_VALIDATION_SCHEMA_VERSION:
            raise ValueError("unsupported extraction validation safety evidence")
        for value in (
            self.evidence_id,
            self.live_feedback_record_id,
            self.source_record_id,
            self.audit_id,
            self.evidence_cutoff_operation_id,
        ):
            _require_id(value, "validation safety evidence ID")
        _require_digest(self.audit_digest, "validation safety audit digest")
        if type(self.complete) is not bool:
            raise TypeError("validation safety completeness must be bool")
        for value in self.failure_counts:
            if type(value) is not int or value < 0:
                raise ValueError("validation safety failure counts are invalid")
        expected = _digest({
            "live_feedback_record_id": self.live_feedback_record_id,
            "source_record_id": self.source_record_id,
            "audit_id": self.audit_id,
            "audit_digest": self.audit_digest,
            "evidence_cutoff_operation_id": self.evidence_cutoff_operation_id,
            "complete": self.complete,
            "failure_counts": list(self.failure_counts),
        })
        if self.evidence_id != f"validation-safety.{expected[:40]}":
            raise ValueError("validation safety evidence ID mismatch")

    @property
    def failure_counts(self) -> tuple[int, int, int, int]:
        return (
            self.schema_failure_count,
            self.safety_failure_count,
            self.prompt_leakage_failure_count,
            self.native_writer_failure_count,
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evidence_id": self.evidence_id,
            "live_feedback_record_id": self.live_feedback_record_id,
            "source_record_id": self.source_record_id,
            "audit_id": self.audit_id,
            "audit_digest": self.audit_digest,
            "evidence_cutoff_operation_id": self.evidence_cutoff_operation_id,
            "complete": self.complete,
            "failure_counts": list(self.failure_counts),
        }

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionValidationSafetyEvidence":
        fields = {
            "schema_version",
            "evidence_id",
            "live_feedback_record_id",
            "source_record_id",
            "audit_id",
            "audit_digest",
            "evidence_cutoff_operation_id",
            "complete",
            "failure_counts",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != fields
            or not isinstance(value["failure_counts"], list)
            or len(value["failure_counts"]) != 4
        ):
            raise ValueError("malformed extraction validation safety evidence")
        try:
            return cls(
                value["evidence_id"],
                value["live_feedback_record_id"],
                value["source_record_id"],
                value["audit_id"],
                value["audit_digest"],
                value["evidence_cutoff_operation_id"],
                value["complete"],
                *value["failure_counts"],
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "malformed extraction validation safety evidence"
            ) from exc

    @classmethod
    def create(
        cls,
        *,
        live_feedback_record_id: str,
        source_record_id: str,
        audit_id: str,
        audit_digest: str,
        evidence_cutoff_operation_id: str,
        complete: bool,
        schema_failure_count: int,
        safety_failure_count: int,
        prompt_leakage_failure_count: int,
        native_writer_failure_count: int,
    ) -> "ExtractionValidationSafetyEvidence":
        counts = (
            schema_failure_count,
            safety_failure_count,
            prompt_leakage_failure_count,
            native_writer_failure_count,
        )
        identity = {
            "live_feedback_record_id": live_feedback_record_id,
            "source_record_id": source_record_id,
            "audit_id": audit_id,
            "audit_digest": audit_digest,
            "evidence_cutoff_operation_id": evidence_cutoff_operation_id,
            "complete": complete,
            "failure_counts": list(counts),
        }
        return cls(
            f"validation-safety.{_digest(identity)[:40]}",
            live_feedback_record_id,
            source_record_id,
            audit_id,
            audit_digest,
            evidence_cutoff_operation_id,
            complete,
            *counts,
        )


@dataclass(frozen=True, slots=True)
class ExtractionQualityMetrics:
    completed_source_count: int
    useful_count: int
    harmful_count: int
    missed_count: int
    unresolved_count: int
    censored_count: int
    nonempty_count: int
    empty_count: int
    missed_assessable_count: int
    missed_assessability_complete: bool
    resolved_useful_rate: float | None
    observed_harmful_rate: float | None
    nonempty_coverage: float | None
    empty_extraction_rate: float | None
    high_confidence_missed_rate: float | None
    safety_failure_count: int

    def __post_init__(self) -> None:
        counts = (
            self.completed_source_count,
            self.useful_count,
            self.harmful_count,
            self.missed_count,
            self.unresolved_count,
            self.censored_count,
            self.nonempty_count,
            self.empty_count,
            self.missed_assessable_count,
            self.safety_failure_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("extraction quality counts must be non-negative integers")
        if self.completed_source_count != sum((
            self.useful_count,
            self.harmful_count,
            self.missed_count,
            self.unresolved_count,
            self.censored_count,
        )):
            raise ValueError("extraction quality label counts do not cover sources")
        if self.nonempty_count > self.completed_source_count or (
            self.empty_count > self.completed_source_count
        ) or self.missed_assessable_count > self.completed_source_count:
            raise ValueError("extraction quality denominator counts are invalid")
        if type(self.missed_assessability_complete) is not bool:
            raise TypeError("missed assessability completeness must be bool")
        ratios = (
            self.resolved_useful_rate,
            self.observed_harmful_rate,
            self.nonempty_coverage,
            self.empty_extraction_rate,
            self.high_confidence_missed_rate,
        )
        if any(
            value is not None
            and (not math.isfinite(value) or not 0 <= value <= 1)
            for value in ratios
        ):
            raise ValueError("extraction quality ratios must be probabilities or unknown")
        expected_ratios = (
            (
                self.useful_count / self.resolved_count
                if self.resolved_count
                else None
            ),
            (
                self.harmful_count / self.nonempty_count
                if self.nonempty_count
                else None
            ),
            (
                self.nonempty_count / self.completed_source_count
                if self.completed_source_count
                else None
            ),
            (
                self.empty_count / self.completed_source_count
                if self.completed_source_count
                else None
            ),
            (
                self.missed_count / self.missed_assessable_count
                if self.missed_assessability_complete
                and self.missed_assessable_count
                else None
            ),
        )
        if ratios != expected_ratios:
            raise ValueError("extraction quality ratios do not match counts")

    @classmethod
    def from_observations(
        cls,
        observations: tuple[ExtractionValidationObservation, ...],
    ) -> "ExtractionQualityMetrics":
        total = len(observations)
        counts = {
            label: sum(value.label == label for value in observations)
            for label in ExtractionFeedbackLabel
        }
        nonempty = sum(
            value.extraction_status == ExtractionSetStatus.NONEMPTY
            for value in observations
        )
        empty = sum(
            value.extraction_status == ExtractionSetStatus.EMPTY
            for value in observations
        )
        assessability_complete = all(
            value.missed_assessable is not None for value in observations
        )
        assessable = sum(value.missed_assessable is True for value in observations)
        resolved = (
            counts[ExtractionFeedbackLabel.USEFUL]
            + counts[ExtractionFeedbackLabel.HARMFUL]
        )
        return cls(
            total,
            counts[ExtractionFeedbackLabel.USEFUL],
            counts[ExtractionFeedbackLabel.HARMFUL],
            counts[ExtractionFeedbackLabel.MISSED],
            counts[ExtractionFeedbackLabel.UNRESOLVED],
            counts[ExtractionFeedbackLabel.CENSORED],
            nonempty,
            empty,
            assessable,
            assessability_complete,
            (
                counts[ExtractionFeedbackLabel.USEFUL] / resolved
                if resolved
                else None
            ),
            (
                counts[ExtractionFeedbackLabel.HARMFUL] / nonempty
                if nonempty
                else None
            ),
            nonempty / total if total else None,
            empty / total if total else None,
            (
                counts[ExtractionFeedbackLabel.MISSED] / assessable
                if assessability_complete and assessable
                else None
            ),
            sum(sum(value.failure_counts) for value in observations),
        )

    @property
    def resolved_count(self) -> int:
        return self.useful_count + self.harmful_count

    def payload(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionQualityMetrics":
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("malformed extraction validation metrics")
        try:
            return cls(**value)
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed extraction validation metrics") from exc


@dataclass(frozen=True, slots=True)
class ExtractionAcceptanceCriteria:
    minimum_matched_pairs: int
    minimum_resolved_examples: int
    minimum_useful_rate_delta: float
    maximum_harmful_rate_delta: float
    minimum_coverage_ratio: float
    maximum_empty_rate: float
    maximum_missed_rate_delta: float
    required_metrics: tuple[str, ...] = ("harmful_rate",)
    proposal_budget_id: str = "proposal-budget.default-v1"
    maximum_proposal_generations: int = 1
    maximum_candidate_selections: int = 1

    def __post_init__(self) -> None:
        if type(self.minimum_matched_pairs) is not int or self.minimum_matched_pairs < 1:
            raise ValueError("minimum matched pairs must be positive")
        if (
            type(self.minimum_resolved_examples) is not int
            or self.minimum_resolved_examples < 1
        ):
            raise ValueError("minimum resolved examples must be positive")
        values = {
            "minimum_useful_rate_delta": _finite(
                self.minimum_useful_rate_delta,
                "minimum useful-rate delta",
            ),
            "maximum_harmful_rate_delta": _finite(
                self.maximum_harmful_rate_delta,
                "maximum harmful-rate delta",
            ),
            "minimum_coverage_ratio": _finite(
                self.minimum_coverage_ratio,
                "minimum coverage ratio",
            ),
            "maximum_empty_rate": _finite(
                self.maximum_empty_rate,
                "maximum empty rate",
            ),
            "maximum_missed_rate_delta": _finite(
                self.maximum_missed_rate_delta,
                "maximum missed-rate delta",
            ),
        }
        if values["minimum_useful_rate_delta"] <= 0:
            raise ValueError("minimum useful-rate delta must be strictly positive")
        if values["maximum_harmful_rate_delta"] < 0 or values[
            "maximum_missed_rate_delta"
        ] < 0:
            raise ValueError("maximum quality regressions must be non-negative")
        if not 0 <= values["minimum_coverage_ratio"] <= 1:
            raise ValueError("minimum coverage ratio must be between zero and one")
        if not 0 <= values["maximum_empty_rate"] <= 1:
            raise ValueError("maximum empty rate must be between zero and one")
        if (
            len(self.required_metrics) != len(set(self.required_metrics))
            or set(self.required_metrics) - _REQUIRED_METRICS
        ):
            raise ValueError("required extraction validation metrics are invalid")
        _require_id(self.proposal_budget_id, "proposal budget ID")
        if (
            type(self.maximum_proposal_generations) is not int
            or self.maximum_proposal_generations < 1
            or type(self.maximum_candidate_selections) is not int
            or self.maximum_candidate_selections < 1
        ):
            raise ValueError("proposal generation and selection budgets must be positive")
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def digest(self) -> str:
        return _digest({
            "minimum_matched_pairs": self.minimum_matched_pairs,
            "minimum_resolved_examples": self.minimum_resolved_examples,
            "minimum_useful_rate_delta": self.minimum_useful_rate_delta,
            "maximum_harmful_rate_delta": self.maximum_harmful_rate_delta,
            "minimum_coverage_ratio": self.minimum_coverage_ratio,
            "maximum_empty_rate": self.maximum_empty_rate,
            "maximum_missed_rate_delta": self.maximum_missed_rate_delta,
            "required_metrics": list(self.required_metrics),
            "proposal_budget_id": self.proposal_budget_id,
            "maximum_proposal_generations": self.maximum_proposal_generations,
            "maximum_candidate_selections": self.maximum_candidate_selections,
        })


@dataclass(frozen=True, slots=True)
class ExtractionValidationDecision:
    decision_id: str
    accepted: bool
    split_id: str
    parent_artifact_id: str
    proposal_artifact_id: str
    criteria_digest: str
    parent_metrics: ExtractionQualityMetrics
    proposal_metrics: ExtractionQualityMetrics
    useful_rate_delta: float | None
    harmful_rate_delta: float | None
    missed_rate_delta: float | None
    changed_extraction_count: int
    reason_codes: tuple[str, ...]
    pair_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    decision_schema: str = EXTRACTION_VALIDATION_SCHEMA
    schema_version: int = EXTRACTION_VALIDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != EXTRACTION_VALIDATION_SCHEMA_VERSION
            or self.decision_schema != EXTRACTION_VALIDATION_SCHEMA
        ):
            raise ValueError("unsupported extraction validation decision schema")
        for value in (
            self.decision_id,
            self.split_id,
            self.parent_artifact_id,
            self.proposal_artifact_id,
        ):
            _require_id(value, "extraction validation decision identity")
        _require_digest(self.criteria_digest, "validation criteria digest")
        if type(self.accepted) is not bool:
            raise TypeError("validation acceptance must be bool")
        if (
            type(self.changed_extraction_count) is not int
            or self.changed_extraction_count < 0
        ):
            raise ValueError("changed extraction count must be non-negative")
        if not self.reason_codes:
            raise ValueError("validation decision requires reason codes")
        if len(self.pair_ids) != len(set(self.pair_ids)) or len(
            self.observation_ids
        ) != len(set(self.observation_ids)):
            raise ValueError("validation decision evidence IDs must be unique")
        for value in (*self.pair_ids, *self.observation_ids):
            _require_id(value, "validation decision evidence ID")
        for value in (
            self.useful_rate_delta,
            self.harmful_rate_delta,
            self.missed_rate_delta,
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError("validation quality deltas must be finite or unknown")
        if self.accepted != (self.reason_codes == ("extraction_validation_passed",)):
            raise ValueError("validation acceptance and reason codes disagree")
        expected = _digest(self.identity_payload())
        if self.decision_id != f"extraction-validation.{expected[:40]}":
            raise ValueError("extraction validation decision ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "decision_schema": self.decision_schema,
            "accepted": self.accepted,
            "split_id": self.split_id,
            "parent_artifact_id": self.parent_artifact_id,
            "proposal_artifact_id": self.proposal_artifact_id,
            "criteria_digest": self.criteria_digest,
            "parent_metrics": self.parent_metrics.payload(),
            "proposal_metrics": self.proposal_metrics.payload(),
            "useful_rate_delta": self.useful_rate_delta,
            "harmful_rate_delta": self.harmful_rate_delta,
            "missed_rate_delta": self.missed_rate_delta,
            "changed_extraction_count": self.changed_extraction_count,
            "reason_codes": list(self.reason_codes),
            "pair_ids": list(self.pair_ids),
            "observation_ids": list(self.observation_ids),
        }

    def payload(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "decision_id": self.decision_id,
        }

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionValidationDecision":
        fields = {
            "schema_version",
            "decision_schema",
            "decision_id",
            "accepted",
            "split_id",
            "parent_artifact_id",
            "proposal_artifact_id",
            "criteria_digest",
            "parent_metrics",
            "proposal_metrics",
            "useful_rate_delta",
            "harmful_rate_delta",
            "missed_rate_delta",
            "changed_extraction_count",
            "reason_codes",
            "pair_ids",
            "observation_ids",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("malformed extraction validation decision")
        try:
            reason_codes = value["reason_codes"]
            pair_ids = value["pair_ids"]
            observation_ids = value["observation_ids"]
            if not all(isinstance(items, list) for items in (
                reason_codes,
                pair_ids,
                observation_ids,
            )):
                raise TypeError("decision collections must be lists")
            return cls(
                decision_id=value["decision_id"],
                accepted=value["accepted"],
                split_id=value["split_id"],
                parent_artifact_id=value["parent_artifact_id"],
                proposal_artifact_id=value["proposal_artifact_id"],
                criteria_digest=value["criteria_digest"],
                parent_metrics=ExtractionQualityMetrics.from_payload(
                    value["parent_metrics"]
                ),
                proposal_metrics=ExtractionQualityMetrics.from_payload(
                    value["proposal_metrics"]
                ),
                useful_rate_delta=value["useful_rate_delta"],
                harmful_rate_delta=value["harmful_rate_delta"],
                missed_rate_delta=value["missed_rate_delta"],
                changed_extraction_count=value["changed_extraction_count"],
                reason_codes=tuple(reason_codes),
                pair_ids=tuple(pair_ids),
                observation_ids=tuple(observation_ids),
                decision_schema=value["decision_schema"],
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed extraction validation decision") from exc


class ExtractionPromptMatchedValidator:
    def evaluate(
        self,
        *,
        split: ExtractionPromptValidationSplit,
        observations: tuple[ExtractionValidationObservation, ...],
        parent_artifact_id: str,
        proposal_artifact_id: str,
        criteria: ExtractionAcceptanceCriteria,
    ) -> ExtractionValidationDecision:
        _require_id(parent_artifact_id, "parent extraction artifact ID")
        _require_id(proposal_artifact_id, "proposal extraction artifact ID")
        if parent_artifact_id == proposal_artifact_id:
            raise ValueError("validation requires distinct extraction artifacts")
        if not observations or len({value.observation_id for value in observations}) != len(
            observations
        ):
            raise ValueError("validation observations must be nonempty and unique")
        pairs: dict[str, dict[ExtractionValidationVariant, ExtractionValidationObservation]] = {}
        for observation in observations:
            if not split.permits(observation):
                raise ValueError("observation is outside the validation split")
            expected_artifact = (
                parent_artifact_id
                if observation.variant == ExtractionValidationVariant.PARENT
                else proposal_artifact_id
            )
            if observation.extraction_artifact_id != expected_artifact:
                raise ValueError("observation extraction artifact mismatch")
            variants = pairs.setdefault(observation.pair_id, {})
            if observation.variant in variants:
                raise ValueError("validation pair contains a duplicate variant")
            variants[observation.variant] = observation
        if any(set(values) != set(ExtractionValidationVariant) for values in pairs.values()):
            raise ValueError("validation requires complete parent/proposal pairs")
        for values in pairs.values():
            parent = values[ExtractionValidationVariant.PARENT]
            proposal = values[ExtractionValidationVariant.PROPOSAL]
            if parent.pair_identity_payload() != proposal.pair_identity_payload():
                raise ValueError("validation pair identity differs")
        changed_extraction_count = sum(
            values[ExtractionValidationVariant.PARENT].extraction_output_digest
            != values[ExtractionValidationVariant.PROPOSAL].extraction_output_digest
            for values in pairs.values()
        )

        parent_values = tuple(
            values[ExtractionValidationVariant.PARENT] for values in pairs.values()
        )
        proposal_values = tuple(
            values[ExtractionValidationVariant.PROPOSAL] for values in pairs.values()
        )
        parent_metrics = ExtractionQualityMetrics.from_observations(parent_values)
        proposal_metrics = ExtractionQualityMetrics.from_observations(proposal_values)

        def delta(proposal: float | None, parent: float | None) -> float | None:
            return None if proposal is None or parent is None else proposal - parent

        useful_delta = delta(
            proposal_metrics.resolved_useful_rate,
            parent_metrics.resolved_useful_rate,
        )
        harmful_delta = delta(
            proposal_metrics.observed_harmful_rate,
            parent_metrics.observed_harmful_rate,
        )
        missed_delta = delta(
            proposal_metrics.high_confidence_missed_rate,
            parent_metrics.high_confidence_missed_rate,
        )
        reasons = []
        if len(pairs) < criteria.minimum_matched_pairs:
            reasons.append("insufficient_matched_pairs")
        if proposal_metrics.resolved_count < criteria.minimum_resolved_examples:
            reasons.append("insufficient_resolved_examples")
        if useful_delta is None or useful_delta < criteria.minimum_useful_rate_delta:
            reasons.append("useful_rate_not_improved")
        if "harmful_rate" in criteria.required_metrics and harmful_delta is None:
            reasons.append("harmful_rate_unknown")
        elif (
            harmful_delta is not None
            and harmful_delta > criteria.maximum_harmful_rate_delta
        ):
            reasons.append("harmful_rate_regression")
        parent_coverage = parent_metrics.nonempty_coverage
        proposal_coverage = proposal_metrics.nonempty_coverage
        if (
            parent_coverage is None
            or proposal_coverage is None
            or proposal_coverage < parent_coverage * criteria.minimum_coverage_ratio
        ):
            reasons.append("coverage_collapse")
        if (
            proposal_metrics.empty_extraction_rate is None
            or proposal_metrics.empty_extraction_rate > criteria.maximum_empty_rate
        ):
            reasons.append("empty_rate_exceeded")
        if "missed_rate" in criteria.required_metrics and missed_delta is None:
            reasons.append("missed_rate_unknown")
        elif missed_delta is not None and missed_delta > criteria.maximum_missed_rate_delta:
            reasons.append("missed_rate_regression")
        if proposal_metrics.safety_failure_count:
            reasons.append("safety_failure")
        if changed_extraction_count == 0:
            reasons.append("no_extraction_intervention")
        accepted = not reasons
        reason_codes = (
            ("extraction_validation_passed",)
            if accepted
            else tuple(dict.fromkeys(reasons))
        )
        values = {
            "accepted": accepted,
            "split_id": split.split_id,
            "parent_artifact_id": parent_artifact_id,
            "proposal_artifact_id": proposal_artifact_id,
            "criteria_digest": criteria.digest,
            "parent_metrics": parent_metrics,
            "proposal_metrics": proposal_metrics,
            "useful_rate_delta": useful_delta,
            "harmful_rate_delta": harmful_delta,
            "missed_rate_delta": missed_delta,
            "changed_extraction_count": changed_extraction_count,
            "reason_codes": reason_codes,
            "pair_ids": tuple(sorted(pairs)),
            "observation_ids": tuple(sorted(
                value.observation_id for value in observations
            )),
        }
        identity = {
            "schema_version": EXTRACTION_VALIDATION_SCHEMA_VERSION,
            "decision_schema": EXTRACTION_VALIDATION_SCHEMA,
            **{
                key: (
                    value.payload()
                    if isinstance(value, ExtractionQualityMetrics)
                    else list(value)
                    if isinstance(value, tuple)
                    else value
                )
                for key, value in values.items()
            },
        }
        return ExtractionValidationDecision(
            f"extraction-validation.{_digest(identity)[:40]}",
            accepted,
            split.split_id,
            parent_artifact_id,
            proposal_artifact_id,
            criteria.digest,
            parent_metrics,
            proposal_metrics,
            useful_delta,
            harmful_delta,
            missed_delta,
            changed_extraction_count,
            reason_codes,
            values["pair_ids"],
            values["observation_ids"],
        )


class JsonExtractionValidationDecisionStore:
    """Immutable extraction validation decisions keyed by logical identity."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    @contextmanager
    def _lock(self, operation: int):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".extraction-validation.lock").open(
            "a+",
            encoding="utf-8",
        ) as lock:
            fcntl.flock(lock.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def put(self, decision: ExtractionValidationDecision) -> tuple[Path, bool]:
        path = self.root / f"{decision.decision_id}.json"
        canonical = _canonical(decision.payload()) + "\n"
        with self._lock(fcntl.LOCK_EX):
            if path.exists():
                if path.read_text(encoding="utf-8") != canonical:
                    raise ValueError("extraction validation decision conflicts with its ID")
                return path, False
            file_descriptor, temporary = tempfile.mkstemp(
                prefix=".extraction-validation.",
                dir=self.root,
            )
            try:
                with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
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
    def _metrics(value: object) -> ExtractionQualityMetrics:
        return ExtractionQualityMetrics.from_payload(value)

    @classmethod
    def _parse(cls, value: object) -> ExtractionValidationDecision:
        return ExtractionValidationDecision.from_payload(value)

    def get(self, decision_id: str) -> ExtractionValidationDecision | None:
        _require_id(decision_id, "extraction validation decision ID")
        path = self.root / f"{decision_id}.json"
        with self._lock(fcntl.LOCK_SH):
            if not path.exists():
                return None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("malformed extraction validation decision JSON") from exc
            decision = self._parse(payload)
            if decision.decision_id != decision_id:
                raise ValueError("extraction validation decision filename mismatch")
            return decision


class JsonExtractionValidationObservationStore:
    """Crash-safe content-free storage for raw validation observations.

    Observations are persisted separately from the derived validation decision
    so that a decision can always be recomputed from the exact raw evidence.
    The store is bound to one frozen split and rejects observations from any
    other family/template/manifest before writing them.
    """

    def __init__(
        self,
        root: Path,
        *,
        split: ExtractionPromptValidationSplit,
    ) -> None:
        if not isinstance(split, ExtractionPromptValidationSplit):
            raise TypeError("validation observation store requires a validation split")
        self.root = root.expanduser().resolve()
        self.split = split

    @contextmanager
    def _lock(self, operation: int):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".extraction-observations.lock").open(
            "a+", encoding="utf-8"
        ) as lock:
            fcntl.flock(lock.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _path(root: Path, observation_id: str) -> Path:
        _require_id(observation_id, "validation observation ID")
        return root / f"{observation_id}.json"

    def put(
        self,
        observation: ExtractionValidationObservation,
    ) -> tuple[Path, bool]:
        if not isinstance(observation, ExtractionValidationObservation):
            raise TypeError("validation observation store requires observations")
        if not self.split.permits(observation):
            raise ValueError("validation observation is outside the frozen split")
        path = self._path(self.root, observation.observation_id)
        serialized = _canonical(observation.payload()) + "\n"
        with self._lock(fcntl.LOCK_EX):
            if path.exists():
                if path.read_text(encoding="utf-8") != serialized:
                    raise ValueError(
                        "validation observation conflicts with its ID"
                    )
                return path, False
            descriptor, temporary = tempfile.mkstemp(
                prefix=".extraction-observation.",
                dir=self.root,
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
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

    def get(self, observation_id: str) -> ExtractionValidationObservation | None:
        path = self._path(self.root, observation_id)
        with self._lock(fcntl.LOCK_SH):
            if not path.exists():
                return None
            try:
                observation = ExtractionValidationObservation.from_payload(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError("malformed extraction validation observation") from exc
            if observation.observation_id != observation_id:
                raise ValueError("validation observation filename mismatch")
            if not self.split.permits(observation):
                raise ValueError("stored validation observation is outside the frozen split")
            return observation

    def records(self) -> tuple[ExtractionValidationObservation, ...]:
        with self._lock(fcntl.LOCK_SH):
            paths = tuple(sorted(
                path for path in self.root.glob("extraction-observation.*.json")
                if path.is_file()
            ))
        values = tuple(
            observation
            for path in paths
            for observation in (self.get(path.stem),)
            if observation is not None
        )
        return tuple(sorted(values, key=lambda value: value.observation_id))


class ExtractionValidationReplay:
    """Require a stored decision to equal a fresh raw-observation evaluation."""

    def verify(
        self,
        decision: ExtractionValidationDecision,
        *,
        split: ExtractionPromptValidationSplit,
        observations: tuple[ExtractionValidationObservation, ...],
        parent_artifact_id: str,
        proposal_artifact_id: str,
        criteria: ExtractionAcceptanceCriteria,
    ) -> None:
        replay = ExtractionPromptMatchedValidator().evaluate(
            split=split,
            observations=observations,
            parent_artifact_id=parent_artifact_id,
            proposal_artifact_id=proposal_artifact_id,
            criteria=criteria,
        )
        if replay != decision:
            raise ValueError("extraction validation decision replay mismatch")
