"""Content-free atomic operation evidence and offline graph materialization."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import resource
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence, runtime_checkable

from ..lifecycle import RawResourceUsage
from .ingestion import InternalMemoryAction


OPERATION_GRAPH_SCHEMA_VERSION = 1
_MAX_EDGE_IDS = 64
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_FORBIDDEN_PAYLOAD_FIELDS = {
    "content",
    "memory",
    "query",
    "prompt",
    "rendered_prompt",
    "response",
    "credential",
    "api_key",
    "source_path",
    "absolute_path",
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, value: object) -> str:
    return f"{prefix}.{_digest(value)[:40]}"


def _require_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a stable identifier")


def _require_ids(values: tuple[str, ...], name: str) -> None:
    if len(values) > _MAX_EDGE_IDS:
        raise ValueError(f"{name} exceeds the bounded edge count")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")
    for value in values:
        _require_identifier(value, name)


class OperationKind(StrEnum):
    SOURCE_OBSERVATION = "source_observation"
    FACT_EXTRACTION = "fact_extraction"
    RELATED_MEMORY_RETRIEVAL = "related_memory_retrieval"
    INTERNAL_OPERATION_DECISION = "internal_operation_decision"
    TARGET_RESOLUTION = "target_resolution"
    VALIDATION = "validation"
    MUTATION = "mutation"
    REREAD_VERIFICATION = "reread_verification"
    FUTURE_QUERY = "future_query"
    RETRIEVAL = "retrieval"
    INJECTION = "injection"
    USE = "use"
    TOOL_BEHAVIOR = "tool_behavior"
    DOWNSTREAM_OUTCOME = "downstream_outcome"
    SUPERSESSION = "supersession"
    RECOVERY = "recovery"


class ArtifactKind(StrEnum):
    SOURCE_OBSERVATION = "source_observation"
    EXTRACTED_FACT = "extracted_fact"
    RELATED_MEMORY = "related_memory"
    OPERATION_PROPOSAL = "operation_proposal"
    VALIDATION_RESULT = "validation_result"
    MEMORY_ARTIFACT = "memory_artifact"
    QUERY = "query"
    RETRIEVAL_RESULT = "retrieval_result"
    INJECTION = "injection"
    USE_EVIDENCE = "use_evidence"
    OUTCOME = "outcome"
    POLICY_PARAMETER = "policy_parameter"


class OperationStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    REJECTED = "rejected"
    NONE = "none"


class TracingLevel(StrEnum):
    DISABLED = "disabled"
    MINIMAL = "minimal"
    SAMPLED = "sampled"
    DIAGNOSTIC = "diagnostic"


class EvidenceKind(StrEnum):
    ARTIFACT = "artifact"
    OPERATION = "operation"
    MUTATION = "mutation"


@dataclass(frozen=True, slots=True)
class OperationContext:
    run_id: str
    episode_id: str
    session_id: str
    task_id: str
    policy_version: str
    prompt_version: str
    framework_version: str

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "episode_id",
            "session_id",
            "task_id",
            "policy_version",
            "prompt_version",
            "framework_version",
        ):
            _require_identifier(getattr(self, name), f"operation context {name}")

    def to_payload(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "policy_version": self.policy_version,
            "prompt_version": self.prompt_version,
            "framework_version": self.framework_version,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> OperationContext:
        return cls(**{name: str(value[name]) for name in (
            "run_id",
            "episode_id",
            "session_id",
            "task_id",
            "policy_version",
            "prompt_version",
            "framework_version",
        )})


@dataclass(frozen=True, slots=True)
class ArtifactNode:
    artifact_id: str
    kind: ArtifactKind
    artifact_schema_version: str
    content_digest: str
    byte_size: int
    token_size: int | None
    revision: str | None
    provenance_ref: str
    schema_version: int = OPERATION_GRAPH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OPERATION_GRAPH_SCHEMA_VERSION:
            raise ValueError("unsupported artifact evidence schema version")
        object.__setattr__(self, "kind", ArtifactKind(self.kind))
        _require_identifier(self.artifact_id, "artifact_id")
        _require_identifier(self.artifact_schema_version, "artifact schema version")
        _require_identifier(self.provenance_ref, "artifact provenance reference")
        if not _DIGEST.fullmatch(self.content_digest):
            raise ValueError("artifact content_digest must be sha256")
        if self.byte_size < 0 or self.token_size is not None and self.token_size < 0:
            raise ValueError("artifact sizes must not be negative")
        if self.revision is not None:
            _require_identifier(self.revision, "artifact revision")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "kind": self.kind.value,
            "artifact_schema_version": self.artifact_schema_version,
            "content_digest": self.content_digest,
            "byte_size": self.byte_size,
            "token_size": self.token_size,
            "revision": self.revision,
            "provenance_ref": self.provenance_ref,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> ArtifactNode:
        return cls(
            artifact_id=str(value["artifact_id"]),
            kind=ArtifactKind(value["kind"]),
            artifact_schema_version=str(value["artifact_schema_version"]),
            content_digest=str(value["content_digest"]),
            byte_size=int(value["byte_size"]),
            token_size=(int(value["token_size"]) if value.get("token_size") is not None else None),
            revision=(str(value["revision"]) if value.get("revision") is not None else None),
            provenance_ref=str(value["provenance_ref"]),
            schema_version=int(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class OperationSpec:
    operation_id: str
    kind: OperationKind
    context: OperationContext
    parent_operation_ids: tuple[str, ...] = ()
    input_artifact_ids: tuple[str, ...] = ()
    retry_identity: str = "attempt-0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", OperationKind(self.kind))
        _require_identifier(self.operation_id, "operation_id")
        _require_identifier(self.retry_identity, "operation retry_identity")
        _require_ids(self.parent_operation_ids, "parent operation IDs")
        _require_ids(self.input_artifact_ids, "input artifact IDs")
        if self.operation_id in self.parent_operation_ids:
            raise ValueError("operation cannot be its own parent")


@dataclass(frozen=True, slots=True)
class OperationRecord:
    operation_id: str
    kind: OperationKind
    context: OperationContext
    parent_operation_ids: tuple[str, ...]
    input_artifact_ids: tuple[str, ...]
    output_artifact_ids: tuple[str, ...]
    retry_identity: str
    status: OperationStatus
    reason_code: str | None
    latency_ms: int
    usage: RawResourceUsage
    schema_version: int = OPERATION_GRAPH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OPERATION_GRAPH_SCHEMA_VERSION:
            raise ValueError("unsupported operation evidence schema version")
        object.__setattr__(self, "kind", OperationKind(self.kind))
        object.__setattr__(self, "status", OperationStatus(self.status))
        _require_identifier(self.operation_id, "operation_id")
        _require_identifier(self.retry_identity, "operation retry_identity")
        _require_ids(self.parent_operation_ids, "parent operation IDs")
        _require_ids(self.input_artifact_ids, "input artifact IDs")
        _require_ids(self.output_artifact_ids, "output artifact IDs")
        if self.latency_ms < 0:
            raise ValueError("operation latency must not be negative")
        if self.reason_code is not None and not _REASON_CODE.fullmatch(self.reason_code):
            raise ValueError("operation reason_code must be machine-readable")
        if self.status in {OperationStatus.FAILED, OperationStatus.REJECTED, OperationStatus.NONE}:
            if self.reason_code is None:
                raise ValueError("non-success operation requires a reason_code")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "operation_id": self.operation_id,
            "kind": self.kind.value,
            "context": self.context.to_payload(),
            "parent_operation_ids": list(self.parent_operation_ids),
            "input_artifact_ids": list(self.input_artifact_ids),
            "output_artifact_ids": list(self.output_artifact_ids),
            "retry_identity": self.retry_identity,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "latency_ms": self.latency_ms,
            "usage": self.usage.to_dict(),
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> OperationRecord:
        return cls(
            operation_id=str(value["operation_id"]),
            kind=OperationKind(value["kind"]),
            context=OperationContext.from_payload(value["context"]),
            parent_operation_ids=tuple(str(item) for item in value["parent_operation_ids"]),
            input_artifact_ids=tuple(str(item) for item in value["input_artifact_ids"]),
            output_artifact_ids=tuple(str(item) for item in value["output_artifact_ids"]),
            retry_identity=str(value["retry_identity"]),
            status=OperationStatus(value["status"]),
            reason_code=(str(value["reason_code"]) if value.get("reason_code") is not None else None),
            latency_ms=int(value["latency_ms"]),
            usage=RawResourceUsage(**dict(value["usage"])),
            schema_version=int(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class MutationEdge:
    mutation_id: str
    operation_id: str
    proposal_operation_ids: tuple[str, ...]
    action: InternalMemoryAction
    target_artifact_id: str | None
    expected_revision: str | None
    before_digest: str | None
    after_digest: str | None
    receipt_id: str | None
    schema_version: int = OPERATION_GRAPH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OPERATION_GRAPH_SCHEMA_VERSION:
            raise ValueError("unsupported mutation evidence schema version")
        object.__setattr__(self, "action", InternalMemoryAction(self.action))
        _require_identifier(self.mutation_id, "mutation_id")
        _require_identifier(self.operation_id, "mutation operation_id")
        _require_ids(self.proposal_operation_ids, "mutation proposal operation IDs")
        if not self.proposal_operation_ids:
            raise ValueError("mutation edge requires proposal operation IDs")
        for digest in (self.before_digest, self.after_digest):
            if digest is not None and not _DIGEST.fullmatch(digest):
                raise ValueError("mutation before/after digest must be sha256")
        if self.action == InternalMemoryAction.NONE:
            if any((self.target_artifact_id, self.expected_revision, self.before_digest, self.after_digest, self.receipt_id)):
                raise ValueError("NONE mutation edge cannot carry mutation state")
            return
        if self.target_artifact_id is None or self.receipt_id is None:
            raise ValueError("mutating edge requires target and receipt IDs")
        _require_identifier(self.target_artifact_id, "mutation target artifact_id")
        _require_identifier(self.receipt_id, "mutation receipt_id")
        if self.expected_revision is not None:
            _require_identifier(self.expected_revision, "mutation expected revision")
        if self.action == InternalMemoryAction.ADD:
            if self.expected_revision is not None or self.before_digest is not None or self.after_digest is None:
                raise ValueError("ADD mutation evidence shape is invalid")
        elif self.action == InternalMemoryAction.UPDATE:
            if self.expected_revision is None or self.before_digest is None or self.after_digest is None:
                raise ValueError("UPDATE mutation evidence shape is invalid")
        elif self.action == InternalMemoryAction.DELETE:
            if self.expected_revision is None or self.before_digest is None or self.after_digest is not None:
                raise ValueError("DELETE mutation evidence shape is invalid")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mutation_id": self.mutation_id,
            "operation_id": self.operation_id,
            "proposal_operation_ids": list(self.proposal_operation_ids),
            "action": self.action.value,
            "target_artifact_id": self.target_artifact_id,
            "expected_revision": self.expected_revision,
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
            "receipt_id": self.receipt_id,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> MutationEdge:
        return cls(
            mutation_id=str(value["mutation_id"]),
            operation_id=str(value["operation_id"]),
            proposal_operation_ids=tuple(str(item) for item in value["proposal_operation_ids"]),
            action=InternalMemoryAction(value["action"]),
            target_artifact_id=(str(value["target_artifact_id"]) if value.get("target_artifact_id") is not None else None),
            expected_revision=(str(value["expected_revision"]) if value.get("expected_revision") is not None else None),
            before_digest=(str(value["before_digest"]) if value.get("before_digest") is not None else None),
            after_digest=(str(value["after_digest"]) if value.get("after_digest") is not None else None),
            receipt_id=(str(value["receipt_id"]) if value.get("receipt_id") is not None else None),
            schema_version=int(value["schema_version"]),
        )


def build_operation_id(
    kind: OperationKind,
    context: OperationContext,
    *,
    step_id: str,
    retry_identity: str = "attempt-0",
    parent_operation_ids: tuple[str, ...] = (),
    input_artifact_ids: tuple[str, ...] = (),
) -> str:
    _require_identifier(step_id, "operation step_id")
    _require_identifier(retry_identity, "operation retry_identity")
    return _stable_id("op", {
        "schema_version": OPERATION_GRAPH_SCHEMA_VERSION,
        "kind": OperationKind(kind).value,
        "context": context.to_payload(),
        "step_id": step_id,
        "retry_identity": retry_identity,
        "parent_operation_ids": parent_operation_ids,
        "input_artifact_ids": input_artifact_ids,
    })


def build_artifact_id(
    kind: ArtifactKind,
    context: OperationContext,
    *,
    logical_name: str,
    content_digest: str,
) -> str:
    _require_identifier(logical_name, "artifact logical_name")
    if not _DIGEST.fullmatch(content_digest):
        raise ValueError("artifact identity content_digest must be sha256")
    return _stable_id("artifact", {
        "schema_version": OPERATION_GRAPH_SCHEMA_VERSION,
        "kind": ArtifactKind(kind).value,
        "context": context.to_payload(),
        "logical_name": logical_name,
        "content_digest": content_digest,
    })


@runtime_checkable
class OperationEvidenceSink(Protocol):
    def append(self, event: Mapping[str, Any]) -> bool: ...


_EVENT_FIELDS = {"schemaVersion", "eventId", "evidenceKind", "tracingLevel", "payload"}
_PAYLOAD_FIELDS = {
    EvidenceKind.ARTIFACT: set(ArtifactNode.__dataclass_fields__),
    EvidenceKind.OPERATION: set(OperationRecord.__dataclass_fields__),
    EvidenceKind.MUTATION: set(MutationEdge.__dataclass_fields__),
}


def _validate_event(event: Mapping[str, Any]) -> None:
    if set(event) != _EVENT_FIELDS:
        raise ValueError("operation evidence event fields are invalid")
    if event.get("schemaVersion") != OPERATION_GRAPH_SCHEMA_VERSION:
        raise ValueError("unsupported operation evidence event schema")
    _require_identifier(str(event.get("eventId") or ""), "operation evidence eventId")
    kind = EvidenceKind(event.get("evidenceKind"))
    TracingLevel(event.get("tracingLevel"))
    payload = event.get("payload")
    if not isinstance(payload, Mapping) or set(payload) != _PAYLOAD_FIELDS[kind]:
        raise ValueError("operation evidence payload fields are invalid")
    if kind == EvidenceKind.ARTIFACT:
        ArtifactNode.from_payload(payload)
    elif kind == EvidenceKind.OPERATION:
        OperationRecord.from_payload(payload)
    else:
        MutationEdge.from_payload(payload)


class AppendOnlyOperationEvidenceLog:
    """Idempotent append-only JSONL evidence; no mutable graph is stored."""

    def __init__(self, output_path: Path | None = None) -> None:
        self.output_path = (
            Path(os.path.abspath(os.path.expanduser(os.fspath(output_path))))
            if output_path
            else None
        )
        self._events: list[dict[str, Any]] = []
        self._canonical_by_id: dict[str, str] = {}
        self._lock = threading.RLock()
        if self.output_path is not None:
            if self.output_path.is_symlink():
                raise ValueError("operation evidence log cannot be a symlink")
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._load()

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            if self.output_path is not None:
                if self.output_path.is_symlink():
                    raise ValueError("operation evidence log cannot be a symlink")
                with self.output_path.open("a+", encoding="utf-8") as handle:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                    try:
                        self._events.clear()
                        self._canonical_by_id.clear()
                        self._load()
                    finally:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return tuple(json.loads(_canonical_json(event)) for event in self._events)

    def _load(self) -> None:
        assert self.output_path is not None
        if self.output_path.is_symlink():
            raise ValueError("operation evidence log cannot be a symlink")
        if not self.output_path.exists():
            return
        for line_number, line in enumerate(self.output_path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed operation evidence at line {line_number}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"operation evidence must be an object at line {line_number}")
            self._append_memory(event)

    def _append_memory(self, event: Mapping[str, Any]) -> bool:
        _validate_event(event)
        event_id = str(event["eventId"])
        canonical = _canonical_json(event)
        existing = self._canonical_by_id.get(event_id)
        if existing is not None:
            if existing != canonical:
                raise ValueError(f"conflicting operation evidence event: {event_id}")
            return False
        value = json.loads(canonical)
        self._canonical_by_id[event_id] = canonical
        self._events.append(value)
        return True

    def append(self, event: Mapping[str, Any]) -> bool:
        with self._lock:
            _validate_event(event)
            event_id = str(event["eventId"])
            canonical = _canonical_json(event)
            if self.output_path is not None:
                if self.output_path.is_symlink():
                    raise ValueError("operation evidence log cannot be a symlink")
                with self.output_path.open("a+", encoding="utf-8") as handle:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    try:
                        # Another process may have appended or replaced the
                        # file since this instance was constructed.  The
                        # on-disk file is authoritative under the lock.
                        self._events.clear()
                        self._canonical_by_id.clear()
                        self._load()
                        existing = self._canonical_by_id.get(event_id)
                        if existing is not None:
                            if existing != canonical:
                                raise ValueError(
                                    f"conflicting operation evidence event: {event_id}"
                                )
                            return False
                        if self.output_path.is_symlink():
                            raise ValueError("operation evidence log cannot be a symlink")
                        handle.write(canonical + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    finally:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            else:
                existing = self._canonical_by_id.get(event_id)
                if existing is not None:
                    if existing != canonical:
                        raise ValueError(f"conflicting operation evidence event: {event_id}")
                    return False
            value = json.loads(canonical)
            self._canonical_by_id[event_id] = canonical
            self._events.append(value)
            return True


@dataclass(frozen=True, slots=True)
class TraceBudget:
    max_events: int = 10_000
    max_serialized_bytes: int = 4_000_000
    max_wall_time_ms: int = 60_000
    max_peak_memory_bytes: int = 64_000_000

    def __post_init__(self) -> None:
        if any(value < 0 for value in (
            self.max_events,
            self.max_serialized_bytes,
            self.max_wall_time_ms,
            self.max_peak_memory_bytes,
        )):
            raise ValueError("trace budgets must not be negative")


@dataclass(frozen=True, slots=True)
class TraceMetrics:
    configured_level: TracingLevel
    effective_level: TracingLevel
    event_count: int
    serialized_bytes: int
    cpu_time_ms: int
    wall_time_ms: int
    peak_memory_bytes: int
    observer_failure_count: int
    attribution_gap_count: int


@dataclass(frozen=True, slots=True)
class ObserverFailureEvidence:
    failure_id: str
    reason_code: str
    failed_event_id: str
    failed_event_digest: str


@dataclass(frozen=True, slots=True)
class AttributionGap:
    gap_id: str
    reason_code: str
    configured_level: TracingLevel
    effective_level: TracingLevel


class _OperationHandle:
    def __init__(self) -> None:
        self.output_artifact_ids: tuple[str, ...] = ()
        self.status = OperationStatus.SUCCESS
        self.reason_code: str | None = None
        self.usage = RawResourceUsage()
        self.completed = False

    def complete(
        self,
        *,
        output_artifact_ids: tuple[str, ...] = (),
        status: OperationStatus = OperationStatus.SUCCESS,
        reason_code: str | None = None,
        usage: RawResourceUsage = RawResourceUsage(),
    ) -> None:
        if self.completed:
            raise RuntimeError("operation scope already completed")
        self.output_artifact_ids = output_artifact_ids
        self.status = OperationStatus(status)
        self.reason_code = reason_code
        self.usage = usage
        self.completed = True


class AtomicOperationRecorder:
    """Observer-only recorder with bounded evidence and explicit failure gaps."""

    def __init__(
        self,
        sink: OperationEvidenceSink,
        *,
        tracing_level: TracingLevel = TracingLevel.MINIMAL,
        sample_key: str = "default-sample",
        sample_rate: float = 1.0,
        budget: TraceBudget = TraceBudget(),
        serializer: Callable[[object], str] = _canonical_json,
    ) -> None:
        self.sink = sink
        self.configured_level = TracingLevel(tracing_level)
        self.effective_level = self.configured_level
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("trace sample_rate must be between zero and one")
        _require_identifier(sample_key, "trace sample_key")
        sample_value = int(hashlib.sha256(sample_key.encode("utf-8")).hexdigest()[:8], 16)
        self._sampled_in = sample_value / 0xFFFFFFFF < sample_rate
        self.budget = budget
        self._serializer = serializer
        self._event_count = 0
        self._serialized_bytes = 0
        self._observer_failures: list[ObserverFailureEvidence] = []
        self._attribution_gaps: list[AttributionGap] = []
        self._budget_gap_recorded = False
        self._wall_start = time.perf_counter_ns()
        self._cpu_start = time.process_time_ns()
        self._peak_rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    @property
    def observer_failures(self) -> tuple[ObserverFailureEvidence, ...]:
        return tuple(self._observer_failures)

    @property
    def attribution_gaps(self) -> tuple[AttributionGap, ...]:
        return tuple(self._attribution_gaps)

    @property
    def metrics(self) -> TraceMetrics:
        wall_ms = max(0, (time.perf_counter_ns() - self._wall_start) // 1_000_000)
        cpu_ms = max(0, (time.process_time_ns() - self._cpu_start) // 1_000_000)
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        return TraceMetrics(
            configured_level=self.configured_level,
            effective_level=self.effective_level,
            event_count=self._event_count,
            serialized_bytes=self._serialized_bytes,
            cpu_time_ms=cpu_ms,
            wall_time_ms=wall_ms,
            peak_memory_bytes=max(0, peak_rss - self._peak_rss_start),
            observer_failure_count=len(self._observer_failures),
            attribution_gap_count=len(self._attribution_gaps),
        )

    def overhead_report(self) -> dict[str, object]:
        value = self.metrics
        return {
            "configured_level": value.configured_level.value,
            "effective_level": value.effective_level.value,
            "event_count": value.event_count,
            "serialized_bytes": value.serialized_bytes,
            "cpu_time_ms": value.cpu_time_ms,
            "wall_time_ms": value.wall_time_ms,
            "peak_memory_bytes": value.peak_memory_bytes,
            "observer_failure_count": value.observer_failure_count,
            "attribution_gap_count": value.attribution_gap_count,
        }

    def _enabled(self) -> bool:
        if self.effective_level == TracingLevel.DISABLED:
            return False
        if self.effective_level == TracingLevel.SAMPLED and not self._sampled_in:
            return False
        return True

    def _check_budget(self) -> None:
        metrics = self.metrics
        exceeded = (
            metrics.event_count > self.budget.max_events
            or metrics.serialized_bytes > self.budget.max_serialized_bytes
            or metrics.wall_time_ms > self.budget.max_wall_time_ms
            or metrics.peak_memory_bytes > self.budget.max_peak_memory_bytes
        )
        if not exceeded or self._budget_gap_recorded:
            return
        self._budget_gap_recorded = True
        previous = self.effective_level
        if previous in {TracingLevel.DIAGNOSTIC, TracingLevel.SAMPLED}:
            self.effective_level = TracingLevel.MINIMAL
        gap_identity = {
            "reason_code": "trace_budget_exceeded",
            "configured_level": self.configured_level.value,
            "previous_level": previous.value,
            "effective_level": self.effective_level.value,
        }
        self._attribution_gaps.append(AttributionGap(
            _stable_id("gap", gap_identity),
            "trace_budget_exceeded",
            self.configured_level,
            self.effective_level,
        ))

    def _emit(self, kind: EvidenceKind, logical_id: str, payload: Mapping[str, Any]) -> None:
        if not self._enabled():
            return
        event_id: str | None = None
        canonical: str | None = None
        try:
            event_id = _stable_id("oev", {
                "schema_version": OPERATION_GRAPH_SCHEMA_VERSION,
                "evidence_kind": kind.value,
                "logical_id": logical_id,
            })
            event = {
                "schemaVersion": OPERATION_GRAPH_SCHEMA_VERSION,
                "eventId": event_id,
                "evidenceKind": kind.value,
                "tracingLevel": self.effective_level.value,
                "payload": dict(payload),
            }
            canonical = self._serializer(event)
            inserted = self.sink.append(event)
        except Exception:
            if event_id is None:
                fallback = f"{kind.value}:{logical_id}"
                event_id = f"oev.failure.{hashlib.sha256(fallback.encode('utf-8')).hexdigest()[:32]}"
            digest_source = canonical if canonical is not None else event_id
            failure_identity = {
                "reason_code": "observer_append_failed",
                "failed_event_id": event_id,
                "failed_event_digest": hashlib.sha256(
                    digest_source.encode("utf-8")
                ).hexdigest(),
            }
            failure_key = _canonical_json(failure_identity).encode("utf-8")
            failure_id = (
                "observer-failure."
                f"{hashlib.sha256(failure_key).hexdigest()[:40]}"
            )
            gap_id = f"gap.{hashlib.sha256(b'gap:' + failure_key).hexdigest()[:40]}"
            self._observer_failures.append(ObserverFailureEvidence(
                failure_id,
                "observer_append_failed",
                event_id,
                failure_identity["failed_event_digest"],
            ))
            self._attribution_gaps.append(AttributionGap(
                gap_id,
                "observer_append_failed",
                self.configured_level,
                self.effective_level,
            ))
            return
        if inserted:
            self._event_count += 1
            assert canonical is not None
            self._serialized_bytes += len(canonical.encode("utf-8")) + 1
        self._check_budget()

    def record_artifact(self, artifact: ArtifactNode) -> None:
        self._emit(EvidenceKind.ARTIFACT, artifact.artifact_id, artifact.to_payload())

    def record_mutation(self, mutation: MutationEdge) -> None:
        self._emit(EvidenceKind.MUTATION, mutation.mutation_id, mutation.to_payload())

    def record_operation(self, operation: OperationRecord) -> None:
        self._emit(EvidenceKind.OPERATION, operation.operation_id, operation.to_payload())

    @contextmanager
    def operation_scope(self, spec: OperationSpec) -> Iterator[_OperationHandle]:
        started = time.perf_counter_ns()
        handle = _OperationHandle()
        try:
            yield handle
        except Exception:
            latency = max(0, (time.perf_counter_ns() - started) // 1_000_000)
            self.record_operation(OperationRecord(
                operation_id=spec.operation_id,
                kind=spec.kind,
                context=spec.context,
                parent_operation_ids=spec.parent_operation_ids,
                input_artifact_ids=spec.input_artifact_ids,
                output_artifact_ids=(
                    handle.output_artifact_ids if handle.completed else ()
                ),
                retry_identity=spec.retry_identity,
                status=(
                    handle.status if handle.completed else OperationStatus.FAILED
                ),
                reason_code=(
                    handle.reason_code if handle.completed else "operation_exception"
                ),
                latency_ms=latency,
                usage=handle.usage,
            ))
            raise
        else:
            latency = max(0, (time.perf_counter_ns() - started) // 1_000_000)
            self.record_operation(OperationRecord(
                operation_id=spec.operation_id,
                kind=spec.kind,
                context=spec.context,
                parent_operation_ids=spec.parent_operation_ids,
                input_artifact_ids=spec.input_artifact_ids,
                output_artifact_ids=handle.output_artifact_ids,
                retry_identity=spec.retry_identity,
                status=handle.status,
                reason_code=handle.reason_code,
                latency_ms=latency,
                usage=handle.usage,
            ))


@dataclass(frozen=True, slots=True)
class OperationGraph:
    artifacts: tuple[ArtifactNode, ...]
    operations: tuple[OperationRecord, ...]
    mutations: tuple[MutationEdge, ...]

    def operation_subgraph(self, operation_id: str) -> OperationGraph:
        by_id = {item.operation_id: item for item in self.operations}
        if operation_id not in by_id:
            raise KeyError(f"unknown operation: {operation_id}")
        selected: set[str] = set()
        pending = [operation_id]
        while pending:
            current = pending.pop()
            if current in selected:
                continue
            selected.add(current)
            pending.extend(by_id[current].parent_operation_ids)
        operations = tuple(item for item in self.operations if item.operation_id in selected)
        artifact_ids = {
            artifact_id
            for item in operations
            for artifact_id in (*item.input_artifact_ids, *item.output_artifact_ids)
        }
        return OperationGraph(
            tuple(item for item in self.artifacts if item.artifact_id in artifact_ids),
            operations,
            tuple(item for item in self.mutations if item.operation_id in selected),
        )

    def failure_groups(self) -> Mapping[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = {}
        for operation in self.operations:
            if operation.status in {OperationStatus.FAILED, OperationStatus.REJECTED}:
                grouped.setdefault(operation.reason_code or "unknown_failure", []).append(
                    operation.operation_id
                )
        return MappingProxyType({key: tuple(value) for key, value in grouped.items()})

    def policy_target_join(self) -> tuple[tuple[str, str, str | None], ...]:
        by_operation = {item.operation_id: item for item in self.operations}
        return tuple(
            (
                by_operation[item.operation_id].context.policy_version,
                item.target_artifact_id or "none",
                item.expected_revision,
            )
            for item in self.mutations
        )


def materialize_operation_graph(events: Sequence[Mapping[str, Any]]) -> OperationGraph:
    artifacts: list[ArtifactNode] = []
    operations: list[OperationRecord] = []
    mutations: list[MutationEdge] = []
    seen_events: dict[str, str] = {}
    for event in events:
        _validate_event(event)
        event_id = str(event["eventId"])
        canonical = _canonical_json(event)
        existing = seen_events.get(event_id)
        if existing is not None:
            if existing != canonical:
                raise ValueError(f"conflicting operation evidence event: {event_id}")
            continue
        seen_events[event_id] = canonical
        kind = EvidenceKind(event["evidenceKind"])
        if kind == EvidenceKind.ARTIFACT:
            artifacts.append(ArtifactNode.from_payload(event["payload"]))
        elif kind == EvidenceKind.OPERATION:
            operations.append(OperationRecord.from_payload(event["payload"]))
        else:
            mutations.append(MutationEdge.from_payload(event["payload"]))

    artifact_by_id = {item.artifact_id: item for item in artifacts}
    operation_by_id = {item.operation_id: item for item in operations}
    mutation_by_id = {item.mutation_id: item for item in mutations}
    if len(artifact_by_id) != len(artifacts):
        raise ValueError("operation graph contains duplicate artifact identity")
    if len(operation_by_id) != len(operations):
        raise ValueError("operation graph contains duplicate operation identity")
    if len(mutation_by_id) != len(mutations):
        raise ValueError("operation graph contains duplicate mutation identity")
    for operation in operations:
        if not set(operation.parent_operation_ids).issubset(operation_by_id):
            raise ValueError("operation graph has a missing parent operation")
        referenced = set(operation.input_artifact_ids) | set(operation.output_artifact_ids)
        if not referenced.issubset(artifact_by_id):
            raise ValueError("operation graph has a missing artifact node")
    for mutation in mutations:
        if mutation.operation_id not in operation_by_id:
            raise ValueError("mutation graph edge has no operation")
        if not set(mutation.proposal_operation_ids).issubset(operation_by_id):
            raise ValueError("mutation graph edge has a missing proposal operation")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(operation_id: str) -> None:
        if operation_id in visiting:
            raise ValueError("operation graph contains a parent cycle")
        if operation_id in visited:
            return
        visiting.add(operation_id)
        for parent in operation_by_id[operation_id].parent_operation_ids:
            visit(parent)
        visiting.remove(operation_id)
        visited.add(operation_id)

    for operation_id in operation_by_id:
        visit(operation_id)
    return OperationGraph(tuple(artifacts), tuple(operations), tuple(mutations))


def audit_operation_evidence(
    events: Sequence[Mapping[str, Any]],
    *,
    forbidden_values: Sequence[str] = (),
) -> tuple[str, ...]:
    issues: set[str] = set()

    def inspect(value: Any, key: str | None = None) -> None:
        if key is not None and key.lower() in _FORBIDDEN_PAYLOAD_FIELDS:
            issues.add("raw_field")
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                inspect(child, str(child_key))
        elif isinstance(value, (list, tuple)):
            for child in value:
                inspect(child)
        elif isinstance(value, str):
            if value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value):
                issues.add("absolute_path")
            if re.search(r"\bsk-[A-Za-z0-9_-]{8,}\b", value):
                issues.add("credential_pattern")
            if any(item and item in value for item in forbidden_values):
                issues.add("forbidden_value")

    for event in events:
        inspect(event)
    return tuple(sorted(issues))
