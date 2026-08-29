"""Deterministic process-signal census for stage-two feasibility analysis.

The census consumes only pure-process identities and application-observable
stage flags.  It never reads benchmark labels, task scores, grader output or
cost fields.  A logical case is counted once; physical observations and
replicate conflicts remain visible diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Mapping

from .evidence_planes import EvidencePlane, EvidenceSourceKind, validate_plane_source


PROCESS_SIGNAL_SCHEMA_VERSION = 1
PROCESS_SIGNAL_SCHEMA = "rsimem-process-signal-case-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,255}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _id(value: object, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")


def _sha(value: object, name: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be sha256")


class ProcessSignalCaseStatus(StrEnum):
    OPTIMIZATION_SIGNAL = "optimization_signal"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    OBSERVABLE_ONLY = "observable_only"
    CENSORED = "censored"
    INVALID = "invalid"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class ProcessSignalCase:
    case_id: str
    logical_case_id: str
    physical_observation_ids: tuple[str, ...]
    source_observed: bool
    extraction_observed: bool
    persistence_observed: bool
    retrieval_observed: bool
    exposure_observed: bool
    outcome_observed: bool
    extraction_attributable: bool
    abstract_hypothesis_digest: str | None
    observation_complete: bool
    evidence_plane: EvidencePlane = EvidencePlane.PURE_PROCESS
    evidence_source: EvidenceSourceKind = EvidenceSourceKind.RUNTIME_OBSERVATION
    schema_version: int = PROCESS_SIGNAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROCESS_SIGNAL_SCHEMA_VERSION:
            raise ValueError("unsupported process-signal schema")
        plane, source = validate_plane_source(self.evidence_plane, self.evidence_source)
        if plane != EvidencePlane.PURE_PROCESS or source != EvidenceSourceKind.RUNTIME_OBSERVATION:
            raise ValueError("process signal case must be pure_process runtime evidence")
        object.__setattr__(self, "evidence_plane", plane)
        object.__setattr__(self, "evidence_source", source)
        for value, name in (
            (self.case_id, "process signal case ID"),
            (self.logical_case_id, "logical case ID"),
        ):
            _id(value, name)
        if not isinstance(self.physical_observation_ids, tuple) or not self.physical_observation_ids:
            raise ValueError("process signal case requires physical observations")
        if len(set(self.physical_observation_ids)) != len(self.physical_observation_ids):
            raise ValueError("physical observation IDs must be unique")
        for value in self.physical_observation_ids:
            _id(value, "physical observation ID")
        for value, name in (
            (self.source_observed, "source observed"),
            (self.extraction_observed, "extraction observed"),
            (self.persistence_observed, "persistence observed"),
            (self.retrieval_observed, "retrieval observed"),
            (self.exposure_observed, "exposure observed"),
            (self.outcome_observed, "outcome observed"),
            (self.extraction_attributable, "extraction attribution"),
            (self.observation_complete, "observation completeness"),
        ):
            if type(value) is not bool:
                raise TypeError(f"{name} must be bool")
        if self.abstract_hypothesis_digest is not None:
            _sha(self.abstract_hypothesis_digest, "abstract hypothesis digest")
        expected = f"process-signal-case.{_digest(self._identity_payload())[:40]}"
        if self.case_id != expected:
            raise ValueError("process signal case ID mismatch")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "logical_case_id": self.logical_case_id,
            "physical_observation_ids": list(self.physical_observation_ids),
            "source_observed": self.source_observed,
            "extraction_observed": self.extraction_observed,
            "persistence_observed": self.persistence_observed,
            "retrieval_observed": self.retrieval_observed,
            "exposure_observed": self.exposure_observed,
            "outcome_observed": self.outcome_observed,
            "extraction_attributable": self.extraction_attributable,
            "abstract_hypothesis_digest": self.abstract_hypothesis_digest,
            "observation_complete": self.observation_complete,
            "evidence_plane": self.evidence_plane.value,
            "evidence_source": self.evidence_source.value,
        }

    @classmethod
    def create(
        cls,
        *,
        logical_case_id: str,
        physical_observation_ids: tuple[str, ...],
        source_observed: bool,
        extraction_observed: bool,
        persistence_observed: bool,
        retrieval_observed: bool,
        exposure_observed: bool,
        outcome_observed: bool,
        extraction_attributable: bool,
        abstract_hypothesis_digest: str | None,
        observation_complete: bool,
    ) -> "ProcessSignalCase":
        values = {
            "logical_case_id": logical_case_id,
            "physical_observation_ids": tuple(physical_observation_ids),
            "source_observed": source_observed,
            "extraction_observed": extraction_observed,
            "persistence_observed": persistence_observed,
            "retrieval_observed": retrieval_observed,
            "exposure_observed": exposure_observed,
            "outcome_observed": outcome_observed,
            "extraction_attributable": extraction_attributable,
            "abstract_hypothesis_digest": abstract_hypothesis_digest,
            "observation_complete": observation_complete,
            "evidence_plane": EvidencePlane.PURE_PROCESS,
            "evidence_source": EvidenceSourceKind.RUNTIME_OBSERVATION,
            "schema_version": PROCESS_SIGNAL_SCHEMA_VERSION,
        }
        return cls(case_id=f"process-signal-case.{_digest(values)[:40]}", **values)

    @property
    def status(self) -> ProcessSignalCaseStatus:
        if not self.observation_complete:
            return ProcessSignalCaseStatus.CENSORED
        required_observation = (
            self.source_observed,
            self.extraction_observed,
            self.persistence_observed,
            self.retrieval_observed,
            self.exposure_observed,
            self.outcome_observed,
        )
        if not all(required_observation):
            return ProcessSignalCaseStatus.OBSERVABLE_ONLY
        if not self.extraction_attributable:
            return ProcessSignalCaseStatus.OBSERVABLE_ONLY
        if self.abstract_hypothesis_digest is None:
            return ProcessSignalCaseStatus.DIAGNOSTIC_ONLY
        return ProcessSignalCaseStatus.OPTIMIZATION_SIGNAL

    def payload(self) -> dict[str, object]:
        return {
            "schema": PROCESS_SIGNAL_SCHEMA,
            "case_id": self.case_id,
            **self._identity_payload(),
            "status": self.status.value,
        }

    @classmethod
    def from_payload(cls, value: object) -> "ProcessSignalCase":
        fields = {
            "schema", "case_id", "schema_version", "logical_case_id",
            "physical_observation_ids", "source_observed", "extraction_observed",
            "persistence_observed", "retrieval_observed", "exposure_observed",
            "outcome_observed", "extraction_attributable", "abstract_hypothesis_digest",
            "observation_complete", "evidence_plane", "evidence_source", "status",
        }
        if not isinstance(value, Mapping) or set(value) != fields or value.get("schema") != PROCESS_SIGNAL_SCHEMA:
            raise ValueError("malformed process signal case")
        if not isinstance(value["physical_observation_ids"], list):
            raise ValueError("malformed process signal observations")
        try:
            result = cls(
                case_id=value["case_id"], logical_case_id=value["logical_case_id"],
                physical_observation_ids=tuple(value["physical_observation_ids"]),
                source_observed=value["source_observed"], extraction_observed=value["extraction_observed"],
                persistence_observed=value["persistence_observed"], retrieval_observed=value["retrieval_observed"],
                exposure_observed=value["exposure_observed"], outcome_observed=value["outcome_observed"],
                extraction_attributable=value["extraction_attributable"],
                abstract_hypothesis_digest=value["abstract_hypothesis_digest"],
                observation_complete=value["observation_complete"],
                evidence_plane=value["evidence_plane"], evidence_source=value["evidence_source"],
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed process signal case") from exc
        if result.status.value != value["status"] or result.payload() != dict(value):
            raise ValueError("non-canonical process signal case")
        return result


@dataclass(frozen=True, slots=True)
class ProcessSignalCaseCensus:
    physical_observation_count: int
    logical_case_count: int
    status_counts: Mapping[str, int]
    conflict_case_count: int

    def __post_init__(self) -> None:
        if self.physical_observation_count < 0 or self.logical_case_count < 0:
            raise ValueError("process signal counts must be non-negative")
        if self.conflict_case_count < 0 or self.conflict_case_count > self.logical_case_count:
            raise ValueError("process signal conflict count is invalid")
        values = dict(self.status_counts)
        if any(not isinstance(key, str) or type(value) is not int or value < 0 for key, value in values.items()):
            raise ValueError("process signal status counts are invalid")
        if sum(values.values()) != self.logical_case_count:
            raise ValueError("process signal status counts do not cover cases")
        object.__setattr__(self, "status_counts", dict(sorted(values.items())))

    def payload(self) -> dict[str, object]:
        return {
            "physicalObservationCount": self.physical_observation_count,
            "logicalCaseCount": self.logical_case_count,
            "statusCounts": dict(self.status_counts),
            "conflictCaseCount": self.conflict_case_count,
        }


def census_process_signal_cases(cases: Iterable[ProcessSignalCase]) -> ProcessSignalCaseCensus:
    values = tuple(cases)
    if any(not isinstance(value, ProcessSignalCase) for value in values):
        raise TypeError("process signal census requires ProcessSignalCase values")
    by_logical: dict[str, list[ProcessSignalCase]] = {}
    for case in values:
        by_logical.setdefault(case.logical_case_id, []).append(case)
    statuses: dict[str, int] = {}
    conflicts = 0
    for logical_case_id, group in by_logical.items():
        labels = {case.status for case in group}
        status = ProcessSignalCaseStatus.AMBIGUOUS if len(labels) > 1 else next(iter(labels))
        statuses[status.value] = statuses.get(status.value, 0) + 1
        if len(labels) > 1:
            conflicts += 1
    return ProcessSignalCaseCensus(
        physical_observation_count=len(values),
        logical_case_count=len(by_logical),
        status_counts=statuses,
        conflict_case_count=conflicts,
    )


__all__ = [
    "PROCESS_SIGNAL_SCHEMA",
    "PROCESS_SIGNAL_SCHEMA_VERSION",
    "ProcessSignalCase",
    "ProcessSignalCaseCensus",
    "ProcessSignalCaseStatus",
    "census_process_signal_cases",
]
