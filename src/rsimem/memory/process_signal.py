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
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

from .evidence_planes import EvidencePlane, EvidenceSourceKind, validate_plane_source
from .process_feedback import ProcessEvent, ProcessEventKind, ProcessEventStatus
from .logical_case import LogicalCaseIdentity
from .pure_process import PureProcessEvent

if TYPE_CHECKING:
    from .pure_extraction import (
        PureExtractionFeedbackRecord,
        PureExtractionSourceRecord,
    )


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
        events: Iterable[ProcessEvent | PureProcessEvent],
        extraction_attributable: bool = False,
        abstract_hypothesis_digest: str | None = None,
        analysis_protocol_id: str | None = None,
        replicate_id: str | None = None,
        observation_window: str | None = None,
    ) -> "ProcessSignalCase":
        """Project stage coverage without inferring attribution from outcomes."""

        values = tuple(events)
        if any(
            not isinstance(event, (ProcessEvent, PureProcessEvent))
            for event in values
        ):
            raise TypeError(
                "process signal projection requires runtime process events"
            )
        if any(
            event.evidence_plane is not EvidencePlane.PURE_PROCESS
            or event.evidence_source is not EvidenceSourceKind.RUNTIME_OBSERVATION
            for event in values
        ):
            raise ValueError(
                "process signal projection requires pure_process runtime events"
            )
        # A logical case may aggregate replicate observations, but one
        # physical projection must never splice stages from different runtime
        # executions.  ``build_process_signal_cases`` groups by task before
        # reaching this method; keep the same invariant at the public
        # constructor boundary for hand-built/replayed inputs.
        contexts = {
            (
                event.run_id,
                event.variant,
                event.trace_id,
                event.episode_id,
                event.session_id,
                event.task_id,
            )
            for event in values
        }
        if len(contexts) > 1:
            raise ValueError(
                "process signal projection cannot cross execution contexts"
            )
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
    # Number of distinct, non-conflicting logical cases supporting each
    # abstract extraction hypothesis.  This is intentionally derived from
    # pure-process cases; it prevents a pair of replicates for one case from
    # being mistaken for two independent cases.
    optimization_hypothesis_case_counts: Mapping[str, int] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for value, name in (
            (self.physical_observation_count, "physical observation count"),
            (self.logical_case_count, "logical case count"),
            (self.conflict_case_count, "conflict case count"),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.physical_observation_count < self.logical_case_count:
            raise ValueError(
                "physical observation count cannot be less than logical case count"
            )
        if self.logical_case_count == 0 and self.physical_observation_count:
            # A census with no logical cases cannot claim unattached physical
            # observations; callers must retain them under a case first.
            raise ValueError(
                "physical observations require at least one logical case"
            )
        if self.conflict_case_count > self.logical_case_count:
            raise ValueError("process signal conflict count is invalid")
        if not isinstance(self.optimization_hypothesis_case_counts, Mapping):
            raise ValueError("process signal hypothesis counts are invalid")
        hypothesis_counts: dict[str, int] = {}
        for hypothesis, count in self.optimization_hypothesis_case_counts.items():
            _sha(hypothesis, "process signal hypothesis digest")
            if type(count) is not int or count < 1:
                raise ValueError("process signal hypothesis counts are invalid")
            hypothesis_counts[hypothesis] = count
        if sum(hypothesis_counts.values()) > self.logical_case_count:
            raise ValueError("process signal hypothesis counts exceed logical cases")
        values = dict(self.status_counts)
        allowed_statuses = {status.value for status in ProcessSignalCaseStatus}
        if any(key not in allowed_statuses for key in values):
            raise ValueError("process signal status key is invalid")
        if any(
            not isinstance(key, str)
            or type(value) is not int
            or value < 0
            for key, value in values.items()
        ):
            raise ValueError("process signal status counts are invalid")
        if sum(values.values()) != self.logical_case_count:
            raise ValueError("process signal status counts do not cover cases")
        optimization_count = values.get(
            ProcessSignalCaseStatus.OPTIMIZATION_SIGNAL.value,
            0,
        )
        if sum(hypothesis_counts.values()) > optimization_count:
            raise ValueError(
                "process signal hypothesis counts exceed optimization cases"
            )
        object.__setattr__(self, "status_counts", dict(sorted(values.items())))
        object.__setattr__(
            self,
            "optimization_hypothesis_case_counts",
            dict(sorted(hypothesis_counts.items())),
        )

    def payload(self) -> dict[str, object]:
        consistent = self.logical_case_count - self.conflict_case_count
        return {
            "physicalObservationCount": self.physical_observation_count,
            "logicalCaseCount": self.logical_case_count,
            "statusCounts": dict(self.status_counts),
            "conflictCaseCount": self.conflict_case_count,
            "optimizationHypothesisCaseCounts": dict(
                self.optimization_hypothesis_case_counts
            ),
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
        if self.lock_path.is_symlink():
            raise ValueError("process-signal case lock cannot be a symlink")
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
        if self.lock_path.is_symlink():
            raise ValueError("process-signal case lock cannot be a symlink")
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
    hypothesis_counts: dict[str, int] = {}
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
        elif status is ProcessSignalCaseStatus.OPTIMIZATION_SIGNAL:
            hypothesis = next(iter(hypotheses), None)
            if hypothesis is not None:
                hypothesis_counts[hypothesis] = hypothesis_counts.get(hypothesis, 0) + 1
    return ProcessSignalCaseCensus(
        physical_observation_count=len(physical_seen),
        logical_case_count=len(by_logical),
        status_counts=statuses,
        conflict_case_count=conflicts,
        optimization_hypothesis_case_counts=hypothesis_counts,
    )


def build_process_signal_cases(
    events: Iterable[ProcessEvent | PureProcessEvent],
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
    if any(
        not isinstance(event, (ProcessEvent, PureProcessEvent))
        for event in values
    ):
        raise TypeError(
            "process signal projection requires runtime process events"
        )
    if any(
        event.evidence_plane is not EvidencePlane.PURE_PROCESS
        or event.evidence_source is not EvidenceSourceKind.RUNTIME_OBSERVATION
        for event in values
    ):
        raise ValueError(
            "process signal projection requires pure_process runtime events"
        )
    if not values:
        return ()
    grouped: dict[str, list[ProcessEvent]] = {}
    for event in values:
        grouped.setdefault(event.task_id, []).append(event)
    cases: list[ProcessSignalCase] = []
    for task_id, task_events in sorted(grouped.items()):
        # ``task_id``/run identifiers are physical observation metadata and
        # must not split one semantic case across provider replicates.  Bind
        # the source extraction set to stable process digests instead.  When
        # an extraction event is present, its source revision and IO digests
        # identify the frozen source set; selection-only traces fall back to
        # the stable source-template identity.
        source_events = tuple(
            event
            for event in task_events
            if event.kind in {
                ProcessEventKind.SOURCE_SELECTION,
                ProcessEventKind.EXTRACTION,
            }
        )
        source_identity = [
            {
                "kind": event.kind.value,
                "source_revision": event.source_revision,
                "input_digest": event.input_digest,
                # Source-selection output identifies which source segments
                # entered formation.  Extraction output is deliberately not
                # part of the semantic case identity: model/provider
                # variation across replicates must remain one case with
                # multiple physical observations, not split the denominator.
                "output_digest": (
                    event.output_digest
                    if event.kind is ProcessEventKind.SOURCE_SELECTION
                    else None
                ),
            }
            for event in sorted(
                source_events,
                key=lambda event: (
                    event.kind.value,
                    event.source_revision,
                    event.input_digest,
                    event.output_digest,
                ),
            )
        ]
        source_set = "extraction-set." + _digest({
            "source_task_template_id": source_task_template_id,
            "source_events": source_identity,
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


def build_joined_process_signal_cases(
    events: Iterable[ProcessEvent | PureProcessEvent],
    feedback: Iterable[PureExtractionFeedbackRecord],
    *,
    sources: Iterable[PureExtractionSourceRecord] = (),
    frozen_policy_digest: str,
    source_task_template_id: str,
    future_task_template_id: str,
    observation_window: str,
    replicate_id: str,
    analysis_protocol_id: str | None = None,
) -> tuple[ProcessSignalCase, ...]:
    """Join source and future task observations through trusted feedback.

    ``build_process_signal_cases`` intentionally projects one execution
    context at a time.  A pure extraction feedback record is the only
    permitted bridge across a source task and a later future task.  This
    helper therefore requires deterministic link events carrying the
    feedback provenance as ``lineage_id``; task IDs, event counts, and result
    status are never used to infer an attribution.

    Link events are emitted by the Hermes bridge as ``RECOVERY`` observations
    with host IDs ``event.pure-source.<record-id>`` and
    ``event.pure-feedback.<record-id>``.  They contain no raw memory or
    benchmark fields and are only anchors used to recover the two execution
    contexts after restart.
    """

    values = tuple(events)
    records = tuple(feedback)
    source_records = tuple(sources)
    _id(replicate_id, "replicate ID")
    if any(not isinstance(event, (ProcessEvent, PureProcessEvent)) for event in values):
        raise TypeError("process signal projection requires runtime process events")
    if any(
        event.evidence_plane is not EvidencePlane.PURE_PROCESS
        or event.evidence_source is not EvidenceSourceKind.RUNTIME_OBSERVATION
        for event in values
    ):
        raise ValueError("process signal projection requires pure_process runtime events")
    events_by_id: dict[str, ProcessEvent | PureProcessEvent] = {}
    for event in values:
        previous = events_by_id.get(event.event_id)
        if previous is not None and previous != event:
            raise ValueError("conflicting joined process signal event identity")
        events_by_id[event.event_id] = event
    values = tuple(events_by_id[event_id] for event_id in sorted(events_by_id))

    # Import lazily to keep the process-signal module independent of the
    # extraction contract's import order.
    from .pure_extraction import (
        PureExtractionAttribution,
        PureExtractionFeedbackRecord,
        PureExtractionSourceRecord,
    )

    if any(not isinstance(item, PureExtractionFeedbackRecord) for item in records):
        raise TypeError("joined process signal requires pure extraction feedback records")
    feedback_by_id: dict[str, PureExtractionFeedbackRecord] = {}
    for item in records:
        previous = feedback_by_id.get(item.record_id)
        if previous is not None and previous != item:
            raise ValueError("conflicting joined process signal feedback identity")
        feedback_by_id[item.record_id] = item
    records = tuple(feedback_by_id[record_id] for record_id in sorted(feedback_by_id))
    if any(not isinstance(item, PureExtractionSourceRecord) for item in source_records):
        raise TypeError("joined process signal requires pure extraction source records")
    sources_by_id: dict[str, PureExtractionSourceRecord] = {}
    for item in source_records:
        previous = sources_by_id.get(item.record_id)
        if previous is not None and previous != item:
            raise ValueError("conflicting joined process signal source identity")
        sources_by_id[item.record_id] = item

    def context(event: ProcessEvent | PureProcessEvent) -> tuple[str, str, str, str, str, str]:
        return (
            event.run_id,
            event.variant,
            event.trace_id,
            event.episode_id,
            event.session_id,
            event.task_id,
        )

    def stage_flags(items: tuple[ProcessEvent | PureProcessEvent, ...]) -> dict[ProcessEventKind, bool]:
        # This is an observation census, not a success label: pending,
        # skipped and failed boundaries are still observed.  UNKNOWN remains
        # censored and is handled by ``observation_complete`` below.
        return {
            kind: any(event.kind is kind and event.status is not ProcessEventStatus.UNKNOWN for event in items)
            for kind in ProcessEventKind
        }

    diagnosis_reasons = {
        "absence", "non_use", "retrieval_miss", "retrieval_failure",
        "injection_failure", "tool_failure", "adapter_failure",
    }
    cases: list[ProcessSignalCase] = []
    for record in records:
        anchors = tuple(
            event for event in values
            if event.lineage_id == record.provenance_id
            and event.kind is ProcessEventKind.RECOVERY
            and (
                event.host_event_id.startswith("event.pure-source.")
                or event.host_event_id.startswith("event.pure-feedback.")
            )
        )
        source_anchors = tuple(
            event for event in anchors
            if event.host_event_id == "event.pure-source." + record.source_record_id
        )
        future_anchors = tuple(
            event for event in anchors
            if event.host_event_id == "event.pure-feedback." + record.record_id
        )
        # A missing, duplicated, or conflicting anchor is not a join.  Keep
        # the record available to the ordinary corpus, but fail closed here.
        if len(source_anchors) != 1 or len(future_anchors) != 1:
            continue
        source_record = sources_by_id.get(record.source_record_id)
        if source_record is None or source_record.provenance_id != record.provenance_id:
            # A provenance anchor without its typed source record is not
            # sufficient to establish an extraction observation.
            continue
        source_anchor = source_anchors[0]
        future_anchor = future_anchors[0]
        if (
            source_anchor.variant != future_anchor.variant
            or source_anchor.run_id != future_anchor.run_id
            or source_anchor.source_revision != source_record.context_revision
            or source_anchor.task_id == future_anchor.task_id
        ):
            # A source/future anchor must share the physical run lineage but
            # represent distinct task boundaries.  Any mismatch (including a
            # stale source revision) is rejected rather than guessed.
            continue
        source_context = context(source_anchor)
        future_context = context(future_anchor)
        source_events = tuple(event for event in values if context(event) == source_context)
        future_events = tuple(event for event in values if context(event) == future_context)
        if not source_events or not future_events:
            continue
        combined = source_events + tuple(event for event in future_events if event not in source_events)
        future_flags = stage_flags(future_events)
        attributable = record.attribution in {
            PureExtractionAttribution.ATTRIBUTABLE_SUCCESS,
            PureExtractionAttribution.ATTRIBUTABLE_FAILURE,
        }
        complete = record.observation_complete and not any(
            "observation_censored" in event.reason_codes
            or event.status is ProcessEventStatus.UNKNOWN
            for event in combined
        )
        source_set = "extraction-set." + _digest({
            "source_task_template_id": source_task_template_id,
            "source_projection_digest": record.source_projection_digest,
        })[:32]
        identity = LogicalCaseIdentity.create(
            frozen_policy_digest=frozen_policy_digest,
            source_task_template_id=source_task_template_id,
            source_extraction_set_id=source_set,
            future_task_template_id=future_task_template_id,
            observation_window=observation_window,
        )
        event_ids = tuple(sorted(event.event_id for event in combined))
        physical_id = "physical-observation." + _digest({
            "logical_case_id": identity.logical_case_id,
            "replicate_id": replicate_id,
            "feedback_record_id": record.record_id,
            "event_ids": list(event_ids),
        })[:40]
        cases.append(ProcessSignalCase.create(
            logical_case_id=identity.logical_case_id,
            physical_observation_ids=(physical_id,),
            source_observed=True,
            extraction_observed=True,
            persistence_observed=any(
                fact.disposition.value == "persisted"
                for fact in source_record.source.facts
            ),
            retrieval_observed=future_flags[ProcessEventKind.RETRIEVAL],
            exposure_observed=future_flags[ProcessEventKind.EXPOSURE],
            outcome_observed=(
                future_flags[ProcessEventKind.TASK_OUTCOME]
                or future_flags[ProcessEventKind.TOOL_RESULT]
            ),
            extraction_attributable=attributable,
            abstract_hypothesis_digest=None,
            observation_complete=complete,
            stage_diagnosis_observed=(
                attributable
                or any(
                    diagnosis_reasons.intersection(event.reason_codes)
                    for event in combined
                )
            ),
            analysis_protocol_id=analysis_protocol_id,
            replicate_id=replicate_id if analysis_protocol_id is not None else None,
            observation_window=observation_window if analysis_protocol_id is not None else None,
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
    "build_joined_process_signal_cases",
]
