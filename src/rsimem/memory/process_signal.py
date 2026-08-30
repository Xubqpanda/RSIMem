"""Deterministic process-signal census for stage-two feasibility analysis.

The census consumes only pure-process identities and application-observable
stage flags.  It never reads benchmark labels, task scores, grader output or
cost fields.  A logical case is counted once; physical observations and
replicate conflicts remain visible diagnostics.
"""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Mapping

from .evidence_planes import EvidencePlane, EvidenceSourceKind, validate_plane_source
from .process_feedback import ProcessEvent, ProcessEventKind, ProcessEventStatus
from .logical_case import LogicalCaseIdentity


PROCESS_SIGNAL_SCHEMA_VERSION = 2
PROCESS_SIGNAL_SCHEMA = "rsimem-process-signal-case-v2"
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
    stage_diagnosis_observed: bool = True
    invalid_reason_code: str | None = None
    evidence_plane: EvidencePlane = EvidencePlane.PURE_PROCESS
    evidence_source: EvidenceSourceKind = EvidenceSourceKind.RUNTIME_OBSERVATION
    analysis_protocol_id: str | None = None
    replicate_id: str | None = None
    observation_window: str | None = None
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
            (self.stage_diagnosis_observed, "stage diagnosis"),
        ):
            if type(value) is not bool:
                raise TypeError(f"{name} must be bool")
        if self.abstract_hypothesis_digest is not None:
            _sha(self.abstract_hypothesis_digest, "abstract hypothesis digest")
        if self.invalid_reason_code is not None and not re.fullmatch(
            r"[a-z][a-z0-9_]{0,63}", self.invalid_reason_code
        ):
            raise ValueError("invalid process signal reason is malformed")
        metadata = (
            self.analysis_protocol_id,
            self.replicate_id,
            self.observation_window,
        )
        if any(value is not None for value in metadata) and any(
            value is None for value in metadata
        ):
            raise ValueError("process signal protocol metadata must be complete")
        for value, name in (
            (self.analysis_protocol_id, "analysis protocol ID"),
            (self.replicate_id, "replicate ID"),
            (self.observation_window, "observation window"),
        ):
            if value is not None:
                _id(value, name)
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
            "stage_diagnosis_observed": self.stage_diagnosis_observed,
            "invalid_reason_code": self.invalid_reason_code,
            "evidence_plane": self.evidence_plane.value,
            "evidence_source": self.evidence_source.value,
            "analysis_protocol_id": self.analysis_protocol_id,
            "replicate_id": self.replicate_id,
            "observation_window": self.observation_window,
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
        stage_diagnosis_observed: bool = True,
        invalid_reason_code: str | None = None,
        analysis_protocol_id: str | None = None,
        replicate_id: str | None = None,
        observation_window: str | None = None,
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
            "stage_diagnosis_observed": stage_diagnosis_observed,
            "invalid_reason_code": invalid_reason_code,
            "analysis_protocol_id": analysis_protocol_id,
            "replicate_id": replicate_id,
            "observation_window": observation_window,
            "evidence_plane": EvidencePlane.PURE_PROCESS,
            "evidence_source": EvidenceSourceKind.RUNTIME_OBSERVATION,
            "schema_version": PROCESS_SIGNAL_SCHEMA_VERSION,
        }
        return cls(case_id=f"process-signal-case.{_digest(values)[:40]}", **values)

    @classmethod
    def from_process_events(
        cls,
        *,
        logical_case_id: str,
        physical_observation_ids: tuple[str, ...],
        events: Iterable[ProcessEvent],
        extraction_attributable: bool = False,
        abstract_hypothesis_digest: str | None = None,
        analysis_protocol_id: str | None = None,
        replicate_id: str | None = None,
        observation_window: str | None = None,
    ) -> "ProcessSignalCase":
        """Project stage coverage without inferring attribution from outcomes."""

        values = tuple(events)
        if any(not isinstance(event, ProcessEvent) for event in values):
            raise TypeError("process signal projection requires ProcessEvent values")
        terminal = {
            ProcessEventStatus.SUCCESS,
            ProcessEventStatus.FAILED,
            ProcessEventStatus.EXECUTED,
            ProcessEventStatus.REJECTED,
        }
        by_kind = {
            kind: any(event.kind is kind and event.status in terminal for event in values)
            for kind in ProcessEventKind
        }
        complete = bool(values) and not any(
            "observation_censored" in event.reason_codes
            or event.status is ProcessEventStatus.UNKNOWN
            for event in values
        )
        diagnosis_reasons = {
            "absence",
            "non_use",
            "retrieval_miss",
            "retrieval_failure",
            "injection_failure",
            "tool_failure",
            "adapter_failure",
        }
        return cls.create(
            logical_case_id=logical_case_id,
            physical_observation_ids=physical_observation_ids,
            source_observed=(
                by_kind[ProcessEventKind.SOURCE_SELECTION]
                or by_kind[ProcessEventKind.EXTRACTION]
            ),
            extraction_observed=by_kind[ProcessEventKind.EXTRACTION],
            persistence_observed=by_kind[ProcessEventKind.COMMIT],
            retrieval_observed=by_kind[ProcessEventKind.RETRIEVAL],
            exposure_observed=by_kind[ProcessEventKind.EXPOSURE],
            outcome_observed=(
                by_kind[ProcessEventKind.TASK_OUTCOME]
                or by_kind[ProcessEventKind.TOOL_RESULT]
            ),
            extraction_attributable=extraction_attributable,
            abstract_hypothesis_digest=abstract_hypothesis_digest,
            observation_complete=complete,
            stage_diagnosis_observed=any(
                diagnosis_reasons.intersection(event.reason_codes)
                for event in values
            ),
            analysis_protocol_id=analysis_protocol_id,
            replicate_id=replicate_id,
            observation_window=observation_window,
        )

    @property
    def status(self) -> ProcessSignalCaseStatus:
        if self.invalid_reason_code is not None:
            return ProcessSignalCaseStatus.INVALID
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
        if not self.stage_diagnosis_observed:
            return ProcessSignalCaseStatus.OBSERVABLE_ONLY
        if not self.extraction_attributable:
            return ProcessSignalCaseStatus.DIAGNOSTIC_ONLY
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
            "observation_complete", "stage_diagnosis_observed", "invalid_reason_code", "evidence_plane",
            "evidence_source", "status",
            "analysis_protocol_id", "replicate_id", "observation_window",
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
                stage_diagnosis_observed=value["stage_diagnosis_observed"],
                invalid_reason_code=value["invalid_reason_code"],
                evidence_plane=value["evidence_plane"], evidence_source=value["evidence_source"],
                analysis_protocol_id=value["analysis_protocol_id"],
                replicate_id=value["replicate_id"],
                observation_window=value["observation_window"],
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
        consistent = self.logical_case_count - self.conflict_case_count
        return {
            "physicalObservationCount": self.physical_observation_count,
            "logicalCaseCount": self.logical_case_count,
            "statusCounts": dict(self.status_counts),
            "conflictCaseCount": self.conflict_case_count,
            "conflictRate": (
                self.conflict_case_count / self.logical_case_count
                if self.logical_case_count else None
            ),
            "replicateConsistentCaseCount": consistent,
            "replicateConsistency": (
                consistent / self.logical_case_count
                if self.logical_case_count else None
            ),
        }


class JsonProcessSignalCaseStore:
    """Atomic, restart-safe storage for logical process-signal cases."""

    def __init__(self, path: Path) -> None:
        # Preserve the final path component so a symlink cannot redirect the
        # case store after a batch protocol has been frozen.
        self.path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    @staticmethod
    def _canonical(value: Mapping[str, object]) -> str:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    def _read_unlocked(self) -> dict[str, str]:
        if self.path.is_symlink():
            raise ValueError("process-signal case file cannot be a symlink")
        if not self.path.exists():
            return {}
        records: dict[str, str] = {}
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                case = ProcessSignalCase.from_payload(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"malformed process-signal case at line {line_number}"
                ) from exc
            canonical = self._canonical(case.payload())
            previous = records.get(case.case_id)
            if previous is not None and previous != canonical:
                raise ValueError("conflicting process-signal case")
            records[case.case_id] = canonical
        return records

    def records(self) -> tuple[ProcessSignalCase, ...]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            try:
                records = self._read_unlocked()
                return tuple(
                    ProcessSignalCase.from_payload(json.loads(value))
                    for _, value in sorted(records.items())
                )
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def append(self, case: ProcessSignalCase) -> bool:
        if not isinstance(case, ProcessSignalCase):
            raise TypeError("process-signal store accepts ProcessSignalCase only")
        serialized = self._canonical(case.payload())
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                records = self._read_unlocked()
                previous = records.get(case.case_id)
                if previous is not None:
                    if previous != serialized:
                        raise ValueError("conflicting process-signal case")
                    return False
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(serialized + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                return True
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def census(self) -> ProcessSignalCaseCensus:
        return census_process_signal_cases(self.records())


def census_process_signal_cases(cases: Iterable[ProcessSignalCase]) -> ProcessSignalCaseCensus:
    values = tuple(cases)
    if any(not isinstance(value, ProcessSignalCase) for value in values):
        raise TypeError("process signal census requires ProcessSignalCase values")
    by_logical: dict[str, list[ProcessSignalCase]] = {}
    physical_seen: set[str] = set()
    for case in values:
        overlap = physical_seen.intersection(case.physical_observation_ids)
        if overlap:
            raise ValueError("duplicate physical observation identity")
        physical_seen.update(case.physical_observation_ids)
        by_logical.setdefault(case.logical_case_id, []).append(case)
    statuses: dict[str, int] = {}
    conflicts = 0
    for logical_case_id, group in by_logical.items():
        labels = {case.status for case in group}
        hypotheses = {
            case.abstract_hypothesis_digest
            for case in group
            if case.status is ProcessSignalCaseStatus.OPTIMIZATION_SIGNAL
        }
        conflict = len(labels) > 1 or len(hypotheses) > 1
        status = (
            ProcessSignalCaseStatus.AMBIGUOUS
            if conflict
            else next(iter(labels))
        )
        statuses[status.value] = statuses.get(status.value, 0) + 1
        if conflict:
            conflicts += 1
    return ProcessSignalCaseCensus(
        physical_observation_count=len(physical_seen),
        logical_case_count=len(by_logical),
        status_counts=statuses,
        conflict_case_count=conflicts,
    )


def build_process_signal_cases(
    events: Iterable[ProcessEvent],
    *,
    frozen_policy_digest: str,
    source_task_template_id: str,
    future_task_template_id: str,
    observation_window: str,
    replicate_id: str,
    analysis_protocol_id: str | None = None,
) -> tuple[ProcessSignalCase, ...]:
    """Project one physical run into replay-stable task-level signal cases.

    The projection uses only process-event identity and stage status. It does
    not infer extraction attribution or a policy hypothesis; those remain
    ``False``/``None`` until a trusted attribution pass supplies them.
    Replicate identity affects only the physical observation ID.
    """

    values = tuple(events)
    _id(replicate_id, "replicate ID")
    if any(not isinstance(event, ProcessEvent) for event in values):
        raise TypeError("process signal projection requires ProcessEvent values")
    if not values:
        return ()
    grouped: dict[str, list[ProcessEvent]] = {}
    for event in values:
        grouped.setdefault(event.task_id, []).append(event)
    cases: list[ProcessSignalCase] = []
    for task_id, task_events in sorted(grouped.items()):
        source_set = "extraction-set." + _digest({
            "task_id": task_id,
            "source_task_template_id": source_task_template_id,
        })[:32]
        identity = LogicalCaseIdentity.create(
            frozen_policy_digest=frozen_policy_digest,
            source_task_template_id=source_task_template_id,
            source_extraction_set_id=source_set,
            future_task_template_id=future_task_template_id,
            observation_window=observation_window,
        )
        event_ids = tuple(sorted(event.event_id for event in task_events))
        physical_identity = {
            "logical_case_id": identity.logical_case_id,
            "replicate_id": replicate_id,
            "event_ids": list(event_ids),
        }
        physical_id = "physical-observation." + _digest(physical_identity)[:40]
        cases.append(ProcessSignalCase.from_process_events(
            logical_case_id=identity.logical_case_id,
            physical_observation_ids=(physical_id,),
            events=task_events,
            analysis_protocol_id=analysis_protocol_id,
            replicate_id=replicate_id if analysis_protocol_id is not None else None,
            observation_window=(
                observation_window if analysis_protocol_id is not None else None
            ),
        ))
    return tuple(cases)


__all__ = [
    "PROCESS_SIGNAL_SCHEMA",
    "PROCESS_SIGNAL_SCHEMA_VERSION",
    "ProcessSignalCase",
    "ProcessSignalCaseCensus",
    "JsonProcessSignalCaseStore",
    "ProcessSignalCaseStatus",
    "census_process_signal_cases",
    "build_process_signal_cases",
]
