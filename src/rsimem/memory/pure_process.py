"""Content-free pure-process evidence independent of benchmark metadata."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .evidence_planes import (
    EvidencePlane,
    EvidenceSourceKind,
    validate_plane_source,
    validate_pure_process_payload,
)
from .policy_contracts import PolicyLayer, content_digest
from .process_feedback import ProcessEvent, ProcessEventKind, ProcessEventStatus


PURE_PROCESS_SCHEMA_VERSION = 3
PURE_PROCESS_SCHEMA = "rsimem-pure-process-corpus-v3"


def _id(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"{name} must be sha256")
    return value


@dataclass(frozen=True, slots=True)
class PureProcessEvent:
    """A process event with benchmark identity deliberately projected out."""

    event_id: str
    kind: ProcessEventKind
    status: ProcessEventStatus
    run_id: str
    variant: str
    trace_id: str
    episode_id: str
    session_id: str
    task_id: str
    host_event_id: str
    source_revision: str
    input_digest: str
    output_digest: str
    policy_decision_id: str | None
    policy_layer: PolicyLayer | None
    lineage_id: str | None
    execution_receipt_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    tool_call_id: str | None = None
    tool_result_id: str | None = None
    tool_name_digest: str | None = None
    retry_identity: str | None = None
    tool_success: bool | None = None
    schema_version: int = PURE_PROCESS_SCHEMA_VERSION
    evidence_plane: EvidencePlane = EvidencePlane.PURE_PROCESS
    evidence_source: EvidenceSourceKind = EvidenceSourceKind.RUNTIME_OBSERVATION

    @classmethod
    def from_process_event(cls, event: ProcessEvent) -> "PureProcessEvent":
        if not isinstance(event, ProcessEvent):
            raise TypeError("process event has the wrong type")
        values = {
            "kind": event.kind,
            "status": event.status,
            "run_id": event.run_id,
            "variant": event.variant,
            "trace_id": event.trace_id,
            "episode_id": event.episode_id,
            "session_id": event.session_id,
            "task_id": event.task_id,
            "host_event_id": event.host_event_id,
            "source_revision": event.source_revision,
            "input_digest": event.input_digest,
            "output_digest": event.output_digest,
            "policy_decision_id": event.policy_decision_id,
            "policy_layer": event.policy_layer,
            "lineage_id": event.lineage_id,
            "execution_receipt_ids": event.execution_receipt_ids,
            "reason_codes": event.reason_codes,
            "tool_call_id": event.tool_call_id,
            "tool_result_id": event.tool_result_id,
            "tool_name_digest": event.tool_name_digest,
            "retry_identity": event.retry_identity,
            "tool_success": event.tool_success,
            "schema_version": PURE_PROCESS_SCHEMA_VERSION,
            "evidence_plane": EvidencePlane.PURE_PROCESS,
            "evidence_source": EvidenceSourceKind.RUNTIME_OBSERVATION,
        }
        identity = cls._identity(values)
        return cls(
            event_id=f"pure-process-event.{content_digest(identity)[:40]}",
            **values,
        )

    def __post_init__(self) -> None:
        if self.schema_version != PURE_PROCESS_SCHEMA_VERSION:
            raise ValueError("unsupported pure-process event schema")
        if self.evidence_plane != EvidencePlane.PURE_PROCESS:
            raise ValueError("pure-process event has the wrong evidence plane")
        plane, source = validate_plane_source(self.evidence_plane, self.evidence_source)
        if plane is not EvidencePlane.PURE_PROCESS or source is not EvidenceSourceKind.RUNTIME_OBSERVATION:
            raise ValueError("pure-process event has the wrong source identity")
        object.__setattr__(self, "evidence_plane", plane)
        object.__setattr__(self, "evidence_source", source)
        object.__setattr__(self, "kind", ProcessEventKind(self.kind))
        object.__setattr__(self, "status", ProcessEventStatus(self.status))
        for value, name in (
            (self.event_id, "pure-process event ID"),
            (self.run_id, "run ID"),
            (self.variant, "variant"),
            (self.trace_id, "trace ID"),
            (self.episode_id, "episode ID"),
            (self.session_id, "session ID"),
            (self.task_id, "task ID"),
            (self.host_event_id, "host event ID"),
            (self.source_revision, "source revision"),
        ):
            _id(value, name)
        _digest(self.input_digest, "pure-process input digest")
        _digest(self.output_digest, "pure-process output digest")
        if self.policy_decision_id is not None:
            _id(self.policy_decision_id, "policy decision ID")
        if self.policy_layer is not None:
            object.__setattr__(self, "policy_layer", PolicyLayer(self.policy_layer))
        if self.lineage_id is not None:
            _id(self.lineage_id, "lineage ID")
        if len(self.execution_receipt_ids) != len(set(self.execution_receipt_ids)) or any(
            not isinstance(value, str) or not value for value in self.execution_receipt_ids
        ):
            raise ValueError("pure-process receipt IDs must be unique stable identifiers")
        if len(self.reason_codes) != len(set(self.reason_codes)) or any(
            not isinstance(value, str) or not value for value in self.reason_codes
        ):
            raise ValueError("pure-process reason codes must be unique strings")
        if self.policy_decision_id is not None and self.policy_layer is None:
            raise ValueError("policy decision requires policy layer")
        for value, name in (
            (self.tool_call_id, "tool call ID"),
            (self.tool_result_id, "tool result ID"),
            (self.retry_identity, "tool retry identity"),
        ):
            if value is not None:
                _id(value, name)
        if self.tool_name_digest is not None:
            _digest(self.tool_name_digest, "tool name digest")
        if self.tool_success is not None and type(self.tool_success) is not bool:
            raise TypeError("tool success must be bool or None")
        if self.kind is ProcessEventKind.TOOL_CALL:
            if any(value is not None for value in (
                self.tool_call_id, self.tool_result_id, self.tool_name_digest,
                self.retry_identity, self.tool_success,
            )) and (
                self.tool_call_id is None
                or self.tool_name_digest is None
                or self.retry_identity is None
                or self.tool_result_id is not None
                or self.tool_success is not None
            ):
                raise ValueError("pure-process tool call has incomplete result fields")
        elif self.kind is ProcessEventKind.TOOL_RESULT:
            if any(value is not None for value in (
                self.tool_call_id, self.tool_result_id, self.tool_name_digest,
                self.retry_identity, self.tool_success,
            )) and (
                self.tool_call_id is None
                or self.tool_result_id is None
                or self.tool_name_digest is None
                or self.retry_identity is None
            ):
                raise ValueError("pure-process tool result lacks exact identity")
            if self.tool_success is True and self.status is not ProcessEventStatus.SUCCESS:
                raise ValueError("pure-process successful tool result has a non-success status")
            if self.tool_success is False and self.status is not ProcessEventStatus.FAILED:
                raise ValueError("pure-process failed tool result has a non-failed status")
        elif any(value is not None for value in (
            self.tool_call_id, self.tool_result_id, self.tool_name_digest,
            self.retry_identity, self.tool_success,
        )):
            raise ValueError("pure-process non-tool event has tool identity")
        expected = f"pure-process-event.{content_digest(self.identity_payload())[:40]}"
        if self.event_id != expected:
            raise ValueError("pure-process event ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return self._identity({
            "kind": self.kind,
            "status": self.status,
            "run_id": self.run_id,
            "variant": self.variant,
            "trace_id": self.trace_id,
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "host_event_id": self.host_event_id,
            "source_revision": self.source_revision,
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
            "policy_decision_id": self.policy_decision_id,
            "policy_layer": self.policy_layer,
            "lineage_id": self.lineage_id,
            "execution_receipt_ids": list(self.execution_receipt_ids),
            "reason_codes": list(self.reason_codes),
            "tool_call_id": self.tool_call_id,
            "tool_result_id": self.tool_result_id,
            "tool_name_digest": self.tool_name_digest,
            "retry_identity": self.retry_identity,
            "tool_success": self.tool_success,
            "evidence_plane": self.evidence_plane,
            "evidence_source": self.evidence_source,
        })

    @staticmethod
    def _identity(values: Mapping[str, object]) -> dict[str, object]:
        return {
            "kind": ProcessEventKind(values["kind"]).value,
            "status": ProcessEventStatus(values["status"]).value,
            "run_id": values["run_id"],
            "variant": values["variant"],
            "trace_id": values["trace_id"],
            "episode_id": values["episode_id"],
            "session_id": values["session_id"],
            "task_id": values["task_id"],
            "host_event_id": values["host_event_id"],
            "source_revision": values["source_revision"],
            "input_digest": values["input_digest"],
            "output_digest": values["output_digest"],
            "policy_decision_id": values["policy_decision_id"],
            "policy_layer": (
                PolicyLayer(values["policy_layer"]).value
                if values["policy_layer"] is not None else None
            ),
            "lineage_id": values["lineage_id"],
            "execution_receipt_ids": list(values["execution_receipt_ids"]),
            "reason_codes": list(values["reason_codes"]),
            "tool_call_id": values["tool_call_id"],
            "tool_result_id": values["tool_result_id"],
            "tool_name_digest": values["tool_name_digest"],
            "retry_identity": values["retry_identity"],
            "tool_success": values["tool_success"],
            "evidence_plane": EvidencePlane(values["evidence_plane"]).value,
            "evidence_source": EvidenceSourceKind(values["evidence_source"]).value,
        }

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "schema": PURE_PROCESS_SCHEMA,
            "event_id": self.event_id,
            **self.identity_payload(),
        }

    @classmethod
    def from_payload(cls, value: object) -> "PureProcessEvent":
        fields = {
            "schema_version", "schema", "event_id", "kind", "status", "run_id",
            "variant", "trace_id", "episode_id", "session_id", "task_id",
            "host_event_id", "source_revision", "input_digest", "output_digest",
            "policy_decision_id", "policy_layer", "lineage_id",
            "execution_receipt_ids", "reason_codes", "evidence_plane",
            "tool_call_id", "tool_result_id", "tool_name_digest", "retry_identity",
            "tool_success", "evidence_source",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("malformed pure-process event")
        if value["schema"] != PURE_PROCESS_SCHEMA:
            raise ValueError("unsupported pure-process event schema")
        if not isinstance(value["execution_receipt_ids"], list) or not isinstance(
            value["reason_codes"], list
        ):
            raise ValueError("malformed pure-process event collections")
        try:
            event = cls(
                value["event_id"], ProcessEventKind(value["kind"]),
                ProcessEventStatus(value["status"]), value["run_id"], value["variant"],
                value["trace_id"], value["episode_id"], value["session_id"],
                value["task_id"], value["host_event_id"], value["source_revision"],
                value["input_digest"], value["output_digest"],
                value["policy_decision_id"],
                PolicyLayer(value["policy_layer"]) if value["policy_layer"] else None,
                value["lineage_id"], tuple(value["execution_receipt_ids"]),
                tuple(value["reason_codes"]),
                value["tool_call_id"], value["tool_result_id"], value["tool_name_digest"],
                value["retry_identity"], value["tool_success"],
                value["schema_version"],
                EvidencePlane(value["evidence_plane"]),
                EvidenceSourceKind(value["evidence_source"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed pure-process event") from exc
        if event.payload() != dict(value):
            raise ValueError("non-canonical pure-process event")
        return event


def project_pure_process_events(events: Iterable[ProcessEvent]) -> tuple[PureProcessEvent, ...]:
    projected = tuple(
        event if isinstance(event, PureProcessEvent)
        else PureProcessEvent.from_process_event(event)
        for event in events
    )
    by_id: dict[str, PureProcessEvent] = {}
    for event in projected:
        previous = by_id.get(event.event_id)
        if previous is not None and previous != event:
            raise ValueError("conflicting pure-process event identity")
        by_id[event.event_id] = event
    return tuple(sorted(by_id.values(), key=lambda event: event.event_id))


@dataclass(frozen=True, slots=True)
class PureProcessCorpus:
    corpus_id: str
    events: tuple[PureProcessEvent, ...]
    schema_version: int = PURE_PROCESS_SCHEMA_VERSION
    evidence_plane: EvidencePlane = EvidencePlane.PURE_PROCESS
    evidence_source: EvidenceSourceKind = EvidenceSourceKind.RUNTIME_OBSERVATION

    @classmethod
    def create(cls, events: Iterable[ProcessEvent | PureProcessEvent]) -> "PureProcessCorpus":
        ordered = project_pure_process_events(events)
        identity = {"schema_version": PURE_PROCESS_SCHEMA_VERSION,
                    "evidence_plane": EvidencePlane.PURE_PROCESS.value,
                    "evidence_source": EvidenceSourceKind.RUNTIME_OBSERVATION.value,
                    "event_ids": [event.event_id for event in ordered]}
        return cls(
            f"pure-process-corpus.{content_digest(identity)[:40]}", ordered
        )

    def __post_init__(self) -> None:
        if self.schema_version != PURE_PROCESS_SCHEMA_VERSION:
            raise ValueError("unsupported pure-process corpus schema")
        if self.evidence_plane != EvidencePlane.PURE_PROCESS:
            raise ValueError("pure-process corpus has the wrong evidence plane")
        plane, source = validate_plane_source(self.evidence_plane, self.evidence_source)
        if plane is not EvidencePlane.PURE_PROCESS or source is not EvidenceSourceKind.RUNTIME_OBSERVATION:
            raise ValueError("pure-process corpus has the wrong source identity")
        object.__setattr__(self, "evidence_plane", plane)
        object.__setattr__(self, "evidence_source", source)
        if not self.events or len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError("pure-process corpus events must be nonempty and unique")
        expected = f"pure-process-corpus.{content_digest(self.identity_payload())[:40]}"
        if self.corpus_id != expected:
            raise ValueError("pure-process corpus ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evidence_plane": self.evidence_plane.value,
            "evidence_source": self.evidence_source.value,
            "event_ids": [event.event_id for event in self.events],
        }

    def payload(self) -> dict[str, object]:
        payload = {
            "schema": PURE_PROCESS_SCHEMA,
            "corpus_id": self.corpus_id,
            **self.identity_payload(),
            "events": [event.payload() for event in self.events],
        }
        validate_pure_process_payload(payload)
        return payload

    @classmethod
    def from_payload(cls, value: object) -> "PureProcessCorpus":
        fields = {
            "schema", "corpus_id", "schema_version", "evidence_plane",
            "evidence_source", "event_ids", "events",
        }
        if not isinstance(value, Mapping) or set(value) != fields or value["schema"] != PURE_PROCESS_SCHEMA:
            raise ValueError("malformed pure-process corpus")
        if not isinstance(value["event_ids"], list) or not isinstance(value["events"], list):
            raise ValueError("malformed pure-process corpus events")
        events = tuple(PureProcessEvent.from_payload(item) for item in value["events"])
        result = cls(
            value["corpus_id"], events, value["schema_version"],
            EvidencePlane(value["evidence_plane"]),
            EvidenceSourceKind(value["evidence_source"]),
        )
        if list(event.event_id for event in result.events) != value["event_ids"] or result.payload() != dict(value):
            raise ValueError("non-canonical pure-process corpus")
        return result


class JsonPureProcessCorpusStore:
    def __init__(self, path: Path) -> None:
        # Keep the final path component unresolved so a symlink cannot
        # redirect an owner-controlled corpus after the store is created.
        self.path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    def _assert_not_symlink(self) -> None:
        if self.path.is_symlink():
            raise ValueError("pure-process corpus file cannot be a symlink")

    @contextmanager
    def _lock(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_path.is_symlink():
            raise ValueError("pure-process corpus lock cannot be a symlink")
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def put(self, corpus: PureProcessCorpus) -> tuple[Path, bool]:
        if not isinstance(corpus, PureProcessCorpus):
            raise TypeError("pure-process corpus has the wrong type")
        serialized = json.dumps(corpus.payload(), ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock():
            self._assert_not_symlink()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                if self.path.read_text(encoding="utf-8") != serialized:
                    raise ValueError("pure-process corpus conflicts with existing file")
                return self.path, False
            fd, temporary = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return self.path, True

    def get(self) -> PureProcessCorpus | None:
        with self._lock():
            self._assert_not_symlink()
            if not self.path.exists():
                return None
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("malformed pure-process corpus JSON") from exc
            return PureProcessCorpus.from_payload(value)


class JsonPureProcessEventArchive:
    """Append-only archive for cross-task pure-process event joins.

    Unlike a batch corpus this store accepts events from many execution
    contexts.  It is used only for delayed source/future attribution and
    stores the canonical content-free :class:`PureProcessEvent` payload.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    def _read_unlocked(self) -> dict[str, str]:
        if self.path.is_symlink():
            raise ValueError("pure-process event archive cannot be a symlink")
        if not self.path.exists():
            return {}
        records: dict[str, str] = {}
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                raise ValueError(
                    f"malformed pure-process archive at line {line_number}"
                )
            try:
                event = PureProcessEvent.from_payload(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"malformed pure-process archive at line {line_number}") from exc
            canonical = json.dumps(event.payload(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            previous = records.get(event.event_id)
            if previous is not None and previous != canonical:
                raise ValueError("conflicting pure-process archive event")
            records[event.event_id] = canonical
        return records

    def append(self, events: Iterable[PureProcessEvent]) -> int:
        values = tuple(events)
        if any(not isinstance(event, PureProcessEvent) for event in values):
            raise TypeError("pure-process archive accepts PureProcessEvent values")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_path.is_symlink():
            raise ValueError("pure-process event archive lock cannot be a symlink")
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                records = self._read_unlocked()
                added = 0
                self.path.parent.mkdir(parents=True, exist_ok=True)
                for event in sorted(values, key=lambda value: value.event_id):
                    canonical = json.dumps(
                        event.payload(),
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    previous = records.get(event.event_id)
                    if previous is not None:
                        if previous != canonical:
                            raise ValueError("conflicting pure-process archive event")
                        continue
                    records[event.event_id] = canonical
                    added += 1
                if added:
                    # Do not append directly to the archive.  A process crash
                    # between write(2) calls can leave a truncated final JSONL
                    # record that poisons every later replay.  Rewriting the
                    # complete canonical set through a temporary file keeps
                    # the archive logically append-only while making each
                    # commit atomic and restart-safe.
                    descriptor, temporary = tempfile.mkstemp(
                        prefix=self.path.name + ".",
                        dir=self.path.parent,
                    )
                    try:
                        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                            for canonical in (
                                records[event_id]
                                for event_id in sorted(records)
                            ):
                                handle.write(canonical + "\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                        if self.path.is_symlink():
                            raise ValueError(
                                "pure-process event archive cannot be a symlink"
                            )
                        os.replace(temporary, self.path)
                        directory = os.open(self.path.parent, os.O_RDONLY)
                        try:
                            os.fsync(directory)
                        finally:
                            os.close(directory)
                    finally:
                        if os.path.exists(temporary):
                            os.unlink(temporary)
                return added
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def records(self) -> tuple[PureProcessEvent, ...]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_path.is_symlink():
            raise ValueError("pure-process event archive lock cannot be a symlink")
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            try:
                return tuple(
                    PureProcessEvent.from_payload(json.loads(value))
                    for _, value in sorted(self._read_unlocked().items())
                )
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


__all__ = [
    "JsonPureProcessEventArchive",
    "JsonPureProcessCorpusStore",
    "PureProcessCorpus",
    "PureProcessEvent",
    "project_pure_process_events",
]
