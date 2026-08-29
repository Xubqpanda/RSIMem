"""Host-neutral memory-use and outcome attribution evidence.

This module deliberately does not know about benchmark families, stages or
grader outputs.  A use claim is only attributable when the same bound artifact
(or artifact set) can be joined through retrieval, injection, downstream
behaviour and an application-observable outcome inside a closed observation
window.  Exposure and behavioural consistency are retained as weaker signals;
neither is silently promoted to attributable use.
"""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from .evidence_planes import EvidencePlane, EvidenceSourceKind, validate_plane_source

if TYPE_CHECKING:
    from .artifact_set import ArtifactSetSemanticBinding
    from .operation_graph import OperationGraph


MEMORY_USE_EVIDENCE_SCHEMA_VERSION = 1
MEMORY_USE_EVIDENCE_SCHEMA = "rsimem-memory-use-evidence-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,255}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_REASON = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _require_id(value: object, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")


def _require_digest(value: object, name: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be sha256")


def _require_timestamp(value: object, name: str) -> None:
    if not isinstance(value, str) or _ISO_UTC.fullmatch(value) is None:
        raise ValueError(f"{name} must be an ISO UTC timestamp")
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO UTC timestamp") from exc


def _require_ids(values: tuple[str, ...], name: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")
    for value in values:
        _require_id(value, name)


class OutcomeEvidenceKind(StrEnum):
    """Application-observable outcome surfaces.

    ``WEAK_STRING_MATCH`` is intentionally non-attributable.  It can explain
    why a caller suspects reuse, but a final response string alone cannot prove
    that a particular memory artifact caused the behaviour.
    """

    TOOL_SUCCESS = "tool_success"
    TOOL_FAILURE = "tool_failure"
    STATE_TRANSITION = "state_transition"
    USER_CONFIRMATION = "user_confirmation"
    TASK_COMPLETION = "task_completion"
    WEAK_STRING_MATCH = "weak_string_match"


class MemoryUseResolutionStatus(StrEnum):
    UNRESOLVED = "unresolved"
    CENSORED = "censored"
    EXPOSURE_ONLY = "exposure_only"
    BEHAVIORAL_CONSISTENCY = "behavioral_consistency"
    ATTRIBUTABLE_USE = "attributable_use"


@dataclass(frozen=True, slots=True)
class MemoryUseEvidence:
    """Content-free joins for one observation of a memory artifact or set."""

    evidence_id: str
    artifact_ids: tuple[str, ...]
    artifact_set_id: str | None
    retrieval_operation_id: str | None
    retrieved_artifact_ids: tuple[str, ...]
    injection_operation_id: str | None
    injected_artifact_ids: tuple[str, ...]
    downstream_operation_id: str | None
    used_artifact_ids: tuple[str, ...]
    outcome_operation_id: str | None
    outcome_kind: OutcomeEvidenceKind | None
    outcome_success: bool | None
    observation_cutoff: str
    provenance_id: str
    retrieval_failure: bool = False
    injection_failure: bool = False
    observation_complete: bool = True
    behavioral_consistency: bool = False
    schema_version: int = MEMORY_USE_EVIDENCE_SCHEMA_VERSION
    evidence_plane: EvidencePlane = EvidencePlane.PURE_PROCESS
    evidence_source: EvidenceSourceKind = EvidenceSourceKind.RUNTIME_OBSERVATION

    def __post_init__(self) -> None:
        if self.schema_version != MEMORY_USE_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported memory-use evidence schema")
        plane, source = validate_plane_source(self.evidence_plane, self.evidence_source)
        if plane != EvidencePlane.PURE_PROCESS or source != EvidenceSourceKind.RUNTIME_OBSERVATION:
            raise ValueError("memory-use evidence must be pure_process runtime evidence")
        object.__setattr__(self, "evidence_plane", plane)
        object.__setattr__(self, "evidence_source", source)
        _require_id(self.evidence_id, "memory-use evidence ID")
        _require_ids(self.artifact_ids, "memory artifact IDs")
        _require_ids(self.retrieved_artifact_ids, "retrieved artifact IDs")
        _require_ids(self.injected_artifact_ids, "injected artifact IDs")
        _require_ids(self.used_artifact_ids, "used artifact IDs")
        if not self.artifact_ids and self.artifact_set_id is None:
            raise ValueError("memory-use evidence requires an artifact or artifact set")
        if self.artifact_set_id is not None:
            _require_id(self.artifact_set_id, "memory artifact set ID")
        for value, name in (
            (self.retrieval_operation_id, "retrieval operation ID"),
            (self.injection_operation_id, "injection operation ID"),
            (self.downstream_operation_id, "downstream operation ID"),
            (self.outcome_operation_id, "outcome operation ID"),
            (self.provenance_id, "memory-use provenance ID"),
        ):
            if value is not None:
                _require_id(value, name)
        if self.retrieved_artifact_ids and self.retrieval_operation_id is None:
            raise ValueError("retrieved artifacts require a retrieval operation")
        if self.injected_artifact_ids and self.injection_operation_id is None:
            raise ValueError("injected artifacts require an injection operation")
        if self.used_artifact_ids and self.downstream_operation_id is None:
            raise ValueError("used artifacts require a downstream operation")
        if self.artifact_ids:
            bound = set(self.artifact_ids)
            for values, name in (
                (self.retrieved_artifact_ids, "retrieved artifacts"),
                (self.injected_artifact_ids, "injected artifacts"),
                (self.used_artifact_ids, "used artifacts"),
            ):
                if not set(values).issubset(bound):
                    raise ValueError(f"{name} escape the bound artifact set")
        if type(self.observation_complete) is not bool:
            raise TypeError("observation completeness must be bool")
        if type(self.behavioral_consistency) is not bool:
            raise TypeError("behavioral consistency must be bool")
        if type(self.retrieval_failure) is not bool or type(self.injection_failure) is not bool:
            raise TypeError("retrieval/injection failure flags must be bool")
        if self.outcome_success is not None and type(self.outcome_success) is not bool:
            raise TypeError("outcome success must be bool or None")
        if self.outcome_kind is not None:
            object.__setattr__(self, "outcome_kind", OutcomeEvidenceKind(self.outcome_kind))
        _require_timestamp(self.observation_cutoff, "observation cutoff")
        digest = _digest(self._identity_payload())
        if self.evidence_id != f"memory-use-evidence.{digest[:40]}":
            raise ValueError("memory-use evidence ID mismatch")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_ids": list(self.artifact_ids),
            "artifact_set_id": self.artifact_set_id,
            "retrieval_operation_id": self.retrieval_operation_id,
            "retrieved_artifact_ids": list(self.retrieved_artifact_ids),
            "retrieval_failure": self.retrieval_failure,
            "injection_operation_id": self.injection_operation_id,
            "injected_artifact_ids": list(self.injected_artifact_ids),
            "injection_failure": self.injection_failure,
            "downstream_operation_id": self.downstream_operation_id,
            "used_artifact_ids": list(self.used_artifact_ids),
            "outcome_operation_id": self.outcome_operation_id,
            "outcome_kind": self.outcome_kind.value if self.outcome_kind is not None else None,
            "outcome_success": self.outcome_success,
            "observation_cutoff": self.observation_cutoff,
            "provenance_id": self.provenance_id,
            "observation_complete": self.observation_complete,
            "behavioral_consistency": self.behavioral_consistency,
            "evidence_plane": self.evidence_plane.value,
            "evidence_source": self.evidence_source.value,
        }

    @classmethod
    def create(
        cls,
        *,
        artifact_ids: tuple[str, ...] = (),
        artifact_set_id: str | None = None,
        retrieval_operation_id: str | None = None,
        retrieved_artifact_ids: tuple[str, ...] = (),
        retrieval_failure: bool = False,
        injection_operation_id: str | None = None,
        injected_artifact_ids: tuple[str, ...] = (),
        injection_failure: bool = False,
        downstream_operation_id: str | None = None,
        used_artifact_ids: tuple[str, ...] = (),
        outcome_operation_id: str | None = None,
        outcome_kind: OutcomeEvidenceKind | str | None = None,
        outcome_success: bool | None = None,
        observation_cutoff: str,
        provenance_id: str,
        observation_complete: bool = True,
        behavioral_consistency: bool = False,
    ) -> "MemoryUseEvidence":
        values: dict[str, object] = {
            "artifact_ids": tuple(artifact_ids),
            "artifact_set_id": artifact_set_id,
            "retrieval_operation_id": retrieval_operation_id,
            "retrieved_artifact_ids": tuple(retrieved_artifact_ids),
            "retrieval_failure": retrieval_failure,
            "injection_operation_id": injection_operation_id,
            "injected_artifact_ids": tuple(injected_artifact_ids),
            "injection_failure": injection_failure,
            "downstream_operation_id": downstream_operation_id,
            "used_artifact_ids": tuple(used_artifact_ids),
            "outcome_operation_id": outcome_operation_id,
            "outcome_kind": OutcomeEvidenceKind(outcome_kind) if outcome_kind is not None else None,
            "outcome_success": outcome_success,
            "observation_cutoff": observation_cutoff,
            "provenance_id": provenance_id,
            "observation_complete": observation_complete,
            "behavioral_consistency": behavioral_consistency,
            "schema_version": MEMORY_USE_EVIDENCE_SCHEMA_VERSION,
            "evidence_plane": EvidencePlane.PURE_PROCESS,
            "evidence_source": EvidenceSourceKind.RUNTIME_OBSERVATION,
        }
        digest = _digest(values)
        return cls(evidence_id=f"memory-use-evidence.{digest[:40]}", **values)

    def payload(self) -> dict[str, object]:
        return {"schema": MEMORY_USE_EVIDENCE_SCHEMA, "evidence_id": self.evidence_id, **self._identity_payload()}

    @classmethod
    def from_payload(cls, value: object) -> "MemoryUseEvidence":
        fields = {
            "schema", "evidence_id", "schema_version", "artifact_ids", "artifact_set_id",
            "retrieval_operation_id", "retrieved_artifact_ids", "retrieval_failure",
            "injection_operation_id", "injected_artifact_ids", "injection_failure",
            "downstream_operation_id", "used_artifact_ids",
            "outcome_operation_id", "outcome_kind", "outcome_success", "observation_cutoff",
            "provenance_id", "observation_complete", "behavioral_consistency",
            "evidence_plane", "evidence_source",
        }
        if not isinstance(value, Mapping) or set(value) != fields or value.get("schema") != MEMORY_USE_EVIDENCE_SCHEMA:
            raise ValueError("malformed memory-use evidence")
        list_fields = ("artifact_ids", "retrieved_artifact_ids", "injected_artifact_ids", "used_artifact_ids")
        if any(not isinstance(value[name], list) for name in list_fields):
            raise ValueError("malformed memory-use evidence collections")
        try:
            return cls(
                evidence_id=value["evidence_id"],
                artifact_ids=tuple(value["artifact_ids"]),
                artifact_set_id=value["artifact_set_id"],
                retrieval_operation_id=value["retrieval_operation_id"],
                retrieved_artifact_ids=tuple(value["retrieved_artifact_ids"]),
                retrieval_failure=value["retrieval_failure"],
                injection_operation_id=value["injection_operation_id"],
                injected_artifact_ids=tuple(value["injected_artifact_ids"]),
                injection_failure=value["injection_failure"],
                downstream_operation_id=value["downstream_operation_id"],
                used_artifact_ids=tuple(value["used_artifact_ids"]),
                outcome_operation_id=value["outcome_operation_id"],
                outcome_kind=value["outcome_kind"],
                outcome_success=value["outcome_success"],
                observation_cutoff=value["observation_cutoff"],
                provenance_id=value["provenance_id"],
                observation_complete=value["observation_complete"],
                behavioral_consistency=value["behavioral_consistency"],
                schema_version=value["schema_version"],
                evidence_plane=value["evidence_plane"],
                evidence_source=value["evidence_source"],
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("malformed memory-use evidence") from exc


class JsonMemoryUseEvidenceLog:
    """Crash-safe append-only log for generic runtime use evidence."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self._records: dict[str, str] = {}
        self._load()

    @staticmethod
    def _canonical(value: Mapping[str, object]) -> str:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                evidence = MemoryUseEvidence.from_payload(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"malformed memory-use evidence at line {line_number}"
                ) from exc
            canonical = self._canonical(evidence.payload())
            previous = self._records.get(evidence.evidence_id)
            if previous is not None and previous != canonical:
                raise ValueError("conflicting memory-use evidence")
            self._records[evidence.evidence_id] = canonical

    def records(self) -> tuple[MemoryUseEvidence, ...]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            try:
                self._records.clear()
                self._load()
                return tuple(
                    MemoryUseEvidence.from_payload(json.loads(value))
                    for value in self._records.values()
                )
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def append(self, evidence: MemoryUseEvidence) -> bool:
        if not isinstance(evidence, MemoryUseEvidence):
            raise TypeError("memory-use log accepts MemoryUseEvidence only")
        serialized = self._canonical(evidence.payload())
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                self._records.clear()
                self._load()
                previous = self._records.get(evidence.evidence_id)
                if previous is not None:
                    if previous != serialized:
                        raise ValueError("conflicting memory-use evidence")
                    return False
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(serialized + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                self._records[evidence.evidence_id] = serialized
                return True
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True, slots=True)
class MemoryUseResolution:
    evidence_id: str
    status: MemoryUseResolutionStatus
    reason_code: str
    exposure_observed: bool
    behavioral_consistency: bool
    attributable_use: bool
    outcome_success: bool | None
    observation_complete: bool

    def __post_init__(self) -> None:
        _require_id(self.evidence_id, "memory-use evidence ID")
        object.__setattr__(self, "status", MemoryUseResolutionStatus(self.status))
        if not _REASON.fullmatch(self.reason_code):
            raise ValueError("memory-use resolution reason is invalid")
        for value, name in (
            (self.exposure_observed, "exposure"),
            (self.behavioral_consistency, "behavioral consistency"),
            (self.attributable_use, "attributable use"),
            (self.observation_complete, "observation completeness"),
        ):
            if type(value) is not bool:
                raise TypeError(f"{name} flag must be bool")
        if self.outcome_success is not None and type(self.outcome_success) is not bool:
            raise TypeError("outcome success must be bool or None")

    def payload(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "exposure_observed": self.exposure_observed,
            "behavioral_consistency": self.behavioral_consistency,
            "attributable_use": self.attributable_use,
            "outcome_success": self.outcome_success,
            "observation_complete": self.observation_complete,
        }


def resolve_memory_use(
    evidence: MemoryUseEvidence,
    *,
    artifact_set_binding: "ArtifactSetSemanticBinding | None" = None,
    operation_graph: "OperationGraph | None" = None,
) -> MemoryUseResolution:
    """Resolve a use claim conservatively from exact operation joins.

    An ``artifact_set_id`` is only an opaque reference.  Before it can
    contribute to an attributable-use decision, callers must provide the
    corresponding trusted :class:`ArtifactSetSemanticBinding`; otherwise the
    resolver fails closed instead of treating arbitrary artifact IDs as a
    complete semantic unit.
    """

    if not isinstance(evidence, MemoryUseEvidence):
        raise TypeError("memory-use resolver requires MemoryUseEvidence")
    if operation_graph is not None:
        from .operation_graph import OperationGraph

        if not isinstance(operation_graph, OperationGraph):
            raise TypeError("operation graph has the wrong type")
        join_error = _operation_join_error(evidence, operation_graph)
        if join_error is not None:
            return MemoryUseResolution(
                evidence.evidence_id,
                MemoryUseResolutionStatus.UNRESOLVED,
                join_error,
                bool(evidence.injected_artifact_ids),
                evidence.behavioral_consistency,
                False,
                evidence.outcome_success,
                evidence.observation_complete,
            )
    exposure = bool(evidence.injected_artifact_ids)
    if not evidence.observation_complete:
        return MemoryUseResolution(
            evidence.evidence_id,
            MemoryUseResolutionStatus.CENSORED,
            "observation_censored",
            exposure,
            evidence.behavioral_consistency,
            False,
            evidence.outcome_success,
            False,
        )
    if evidence.retrieval_failure:
        return MemoryUseResolution(
            evidence.evidence_id,
            MemoryUseResolutionStatus.UNRESOLVED,
            "retrieval_failure",
            False,
            evidence.behavioral_consistency,
            False,
            evidence.outcome_success,
            True,
        )
    if evidence.artifact_set_id is not None:
        # Import lazily to keep the two contract modules independent at
        # runtime.  The binding itself carries the authoritative member set
        # and completeness bit; the ID alone is not sufficient evidence.
        from .artifact_set import ArtifactSetSemanticBinding

        if artifact_set_binding is None:
            return MemoryUseResolution(
                evidence.evidence_id,
                MemoryUseResolutionStatus.UNRESOLVED,
                "artifact_set_binding_missing",
                bool(evidence.injected_artifact_ids),
                evidence.behavioral_consistency,
                False,
                evidence.outcome_success,
                True,
            )
        if not isinstance(artifact_set_binding, ArtifactSetSemanticBinding):
            raise TypeError("artifact-set binding has the wrong type")
        if artifact_set_binding.binding_id != evidence.artifact_set_id:
            return MemoryUseResolution(
                evidence.evidence_id,
                MemoryUseResolutionStatus.UNRESOLVED,
                "artifact_set_binding_mismatch",
                bool(evidence.injected_artifact_ids),
                evidence.behavioral_consistency,
                False,
                evidence.outcome_success,
                True,
            )
        if not artifact_set_binding.complete:
            return MemoryUseResolution(
                evidence.evidence_id,
                MemoryUseResolutionStatus.UNRESOLVED,
                "artifact_set_binding_incomplete",
                bool(evidence.injected_artifact_ids),
                evidence.behavioral_consistency,
                False,
                evidence.outcome_success,
                True,
            )
        bound = set(artifact_set_binding.member_artifact_ids)
        if evidence.artifact_ids and set(evidence.artifact_ids) != bound:
            return MemoryUseResolution(
                evidence.evidence_id,
                MemoryUseResolutionStatus.UNRESOLVED,
                "artifact_set_member_mismatch",
                bool(evidence.injected_artifact_ids),
                evidence.behavioral_consistency,
                False,
                evidence.outcome_success,
                True,
            )
    else:
        if artifact_set_binding is not None:
            raise ValueError("artifact-set binding requires artifact_set_id")
        bound = set(evidence.artifact_ids)
    if not evidence.retrieval_operation_id or not evidence.retrieved_artifact_ids:
        return MemoryUseResolution(
            evidence.evidence_id,
            MemoryUseResolutionStatus.UNRESOLVED,
            "retrieval_exact_join_missing",
            exposure,
            evidence.behavioral_consistency,
            False,
            evidence.outcome_success,
            True,
        )
    if bound and set(evidence.retrieved_artifact_ids) != bound:
        return MemoryUseResolution(
            evidence.evidence_id,
            MemoryUseResolutionStatus.UNRESOLVED,
            "retrieval_artifact_set_incomplete",
            exposure,
            evidence.behavioral_consistency,
            False,
            evidence.outcome_success,
            True,
        )
    if evidence.injection_failure:
        return MemoryUseResolution(
            evidence.evidence_id,
            MemoryUseResolutionStatus.UNRESOLVED,
            "injection_failure",
            False,
            evidence.behavioral_consistency,
            False,
            evidence.outcome_success,
            True,
        )
    if not evidence.injection_operation_id or not evidence.injected_artifact_ids:
        return MemoryUseResolution(
            evidence.evidence_id,
            MemoryUseResolutionStatus.EXPOSURE_ONLY if evidence.retrieved_artifact_ids else MemoryUseResolutionStatus.UNRESOLVED,
            "injection_exact_join_missing" if evidence.retrieved_artifact_ids else "retrieval_miss",
            exposure,
            evidence.behavioral_consistency,
            False,
            evidence.outcome_success,
            True,
        )
    if bound and set(evidence.injected_artifact_ids) != bound:
        return MemoryUseResolution(
            evidence.evidence_id,
            MemoryUseResolutionStatus.UNRESOLVED,
            "injection_artifact_set_incomplete",
            True,
            evidence.behavioral_consistency,
            False,
            evidence.outcome_success,
            True,
        )
    if not evidence.downstream_operation_id or not evidence.used_artifact_ids:
        return MemoryUseResolution(
            evidence.evidence_id,
            MemoryUseResolutionStatus.BEHAVIORAL_CONSISTENCY if evidence.behavioral_consistency else MemoryUseResolutionStatus.EXPOSURE_ONLY,
            "downstream_use_not_observed",
            True,
            evidence.behavioral_consistency,
            False,
            evidence.outcome_success,
            True,
        )
    if bound and set(evidence.used_artifact_ids) != bound:
        return MemoryUseResolution(
            evidence.evidence_id,
            MemoryUseResolutionStatus.UNRESOLVED,
            "use_artifact_set_incomplete",
            True,
            evidence.behavioral_consistency,
            False,
            evidence.outcome_success,
            True,
        )
    if not evidence.outcome_operation_id or evidence.outcome_kind is None or evidence.outcome_success is None:
        return MemoryUseResolution(
            evidence.evidence_id,
            MemoryUseResolutionStatus.BEHAVIORAL_CONSISTENCY,
            "outcome_exact_join_missing",
            True,
            True,
            False,
            evidence.outcome_success,
            True,
        )
    if evidence.outcome_kind == OutcomeEvidenceKind.WEAK_STRING_MATCH:
        return MemoryUseResolution(
            evidence.evidence_id,
            MemoryUseResolutionStatus.BEHAVIORAL_CONSISTENCY,
            "weak_string_match_only",
            True,
            True,
            False,
            evidence.outcome_success,
            True,
        )
    if evidence.outcome_kind == OutcomeEvidenceKind.TOOL_FAILURE:
        return MemoryUseResolution(
            evidence.evidence_id,
            MemoryUseResolutionStatus.UNRESOLVED,
            "tool_failure_not_attributable",
            True,
            evidence.behavioral_consistency,
            False,
            evidence.outcome_success,
            True,
        )
    return MemoryUseResolution(
        evidence.evidence_id,
        MemoryUseResolutionStatus.ATTRIBUTABLE_USE,
        "attributable_use" if evidence.outcome_success else "attributable_use_failed_outcome",
        True,
        True,
        True,
        evidence.outcome_success,
        True,
    )


def _operation_join_error(
    evidence: MemoryUseEvidence,
    operation_graph: "OperationGraph",
) -> str | None:
    """Validate operation kinds/parentage when a graph is available.

    IDs in ``MemoryUseEvidence`` are intentionally content-free.  Supplying
    the owner-controlled operation graph lets callers prove that those IDs are
    the expected retrieval -> injection -> use -> outcome chain instead of
    merely matching arbitrary strings.
    """

    from .operation_graph import OperationKind

    operations = {item.operation_id: item for item in operation_graph.operations}

    def require(operation_id: str | None, kind: OperationKind) -> object | None:
        if operation_id is None:
            return None
        operation = operations.get(operation_id)
        if operation is None or operation.kind is not kind:
            return None
        return operation

    retrieval = require(evidence.retrieval_operation_id, OperationKind.RETRIEVAL)
    if evidence.retrieval_operation_id is not None and retrieval is None:
        return "operation_join_invalid"
    injection = require(evidence.injection_operation_id, OperationKind.INJECTION)
    if evidence.injection_operation_id is not None and injection is None:
        return "operation_join_invalid"
    downstream = require(evidence.downstream_operation_id, OperationKind.USE)
    if evidence.downstream_operation_id is not None and downstream is None:
        return "operation_join_invalid"
    outcome = require(evidence.outcome_operation_id, OperationKind.DOWNSTREAM_OUTCOME)
    if evidence.outcome_operation_id is not None and outcome is None:
        return "operation_join_invalid"

    if retrieval is not None and evidence.retrieved_artifact_ids and not set(
        evidence.retrieved_artifact_ids
    ).issubset(set(retrieval.output_artifact_ids) | set(retrieval.input_artifact_ids)):
        return "operation_join_invalid"
    if injection is not None and evidence.injected_artifact_ids and not set(
        evidence.injected_artifact_ids
    ).issubset(set(injection.output_artifact_ids) | set(injection.input_artifact_ids)):
        return "operation_join_invalid"
    if downstream is not None and evidence.used_artifact_ids and not set(
        evidence.used_artifact_ids
    ).issubset(set(downstream.input_artifact_ids) | set(downstream.output_artifact_ids)):
        return "operation_join_invalid"

    def has_parent(operation: object, parent_id: str | None) -> bool:
        return parent_id is None or parent_id in operation.parent_operation_ids

    if injection is not None and not has_parent(injection, evidence.retrieval_operation_id):
        return "operation_join_invalid"
    if downstream is not None and not has_parent(downstream, evidence.injection_operation_id):
        return "operation_join_invalid"
    if outcome is not None and not has_parent(outcome, evidence.downstream_operation_id):
        return "operation_join_invalid"
    return None


__all__ = [
    "MEMORY_USE_EVIDENCE_SCHEMA",
    "MEMORY_USE_EVIDENCE_SCHEMA_VERSION",
    "MemoryUseEvidence",
    "JsonMemoryUseEvidenceLog",
    "MemoryUseResolution",
    "MemoryUseResolutionStatus",
    "OutcomeEvidenceKind",
    "resolve_memory_use",
]
