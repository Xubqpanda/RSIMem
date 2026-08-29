"""Content-free process corpus assembled separately from benchmark scoring.

The PAST-Bench adapter may expose an official score to its reporter, but the
policy learner receives only this bundle.  Keeping the two objects as
different types makes accidental grader/answer-key joins fail at the API
boundary rather than relying on convention.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .policy_contracts import content_digest
from .process_feedback import (
    JsonProcessFeedbackLedger,
    ProcessEvent,
)


PROCESS_CORPUS_SCHEMA_VERSION = 1
PROCESS_CORPUS_SCHEMA = "rsimem-process-corpus-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,255}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SPLITS = {"train", "validation", "final", "pilot"}


def _id(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be sha256")
    return value


@dataclass(frozen=True, slots=True)
class ProcessCorpus:
    corpus_id: str
    split_role: str
    family_id: str
    task_template_group_id: str
    task_manifest_digest: str
    events: tuple[ProcessEvent, ...]
    schema_version: int = PROCESS_CORPUS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROCESS_CORPUS_SCHEMA_VERSION:
            raise ValueError("unsupported process corpus schema version")
        if self.split_role not in _SPLITS:
            raise ValueError("invalid process corpus split role")
        for value, name in (
            (self.corpus_id, "process corpus ID"),
            (self.family_id, "process corpus family ID"),
            (self.task_template_group_id, "process corpus task template group ID"),
        ):
            _id(value, name)
        _digest(self.task_manifest_digest, "process corpus task manifest digest")
        items = tuple(self.events)
        if not items:
            raise ValueError("process corpus requires at least one event")
        if any(not isinstance(item, ProcessEvent) for item in items):
            raise TypeError("process corpus events have the wrong type")
        ids = tuple(item.event_id for item in items)
        if len(ids) != len(set(ids)):
            raise ValueError("process corpus event IDs must be unique")
        for item in items:
            if item.family_id not in {None, self.family_id}:
                raise ValueError("process event family differs from corpus")
        object.__setattr__(self, "events", items)
        expected = f"process-corpus.{content_digest(self.identity_payload())[:40]}"
        if self.corpus_id != expected:
            raise ValueError("process corpus ID mismatch")

    @classmethod
    def create(
        cls,
        events: Iterable[ProcessEvent],
        *,
        split_role: str,
        family_id: str,
        task_template_group_id: str,
        task_manifest_digest: str,
    ) -> "ProcessCorpus":
        # Shared-cold PAST traces can expose the same logical event through an
        # outer and a nested attempt directory.  Exact duplicates are
        # idempotent and collapsed here; an event ID with a different payload
        # is a conflict and must fail closed.
        canonical_by_id: dict[str, str] = {}
        unique: list[ProcessEvent] = []
        for event in events:
            if not isinstance(event, ProcessEvent):
                raise TypeError("process corpus events have the wrong type")
            canonical = json.dumps(event.payload(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            previous = canonical_by_id.get(event.event_id)
            if previous is not None:
                if previous != canonical:
                    raise ValueError("conflicting process corpus event identity")
                continue
            canonical_by_id[event.event_id] = canonical
            unique.append(event)
        ordered = tuple(sorted(unique, key=lambda event: event.event_id))
        identity = {
            "split_role": split_role,
            "family_id": family_id,
            "task_template_group_id": task_template_group_id,
            "task_manifest_digest": task_manifest_digest,
            "event_ids": [event.event_id for event in ordered],
        }
        return cls(
            corpus_id=f"process-corpus.{content_digest(identity)[:40]}",
            split_role=split_role,
            family_id=family_id,
            task_template_group_id=task_template_group_id,
            task_manifest_digest=task_manifest_digest,
            events=ordered,
        )

    @classmethod
    def from_ledger(
        cls,
        path: Path,
        *,
        split_role: str,
        family_id: str,
        task_template_group_id: str,
        task_manifest_digest: str,
    ) -> "ProcessCorpus":
        return cls.create(
            JsonProcessFeedbackLedger(path).events,
            split_role=split_role,
            family_id=family_id,
            task_template_group_id=task_template_group_id,
            task_manifest_digest=task_manifest_digest,
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "split_role": self.split_role,
            "family_id": self.family_id,
            "task_template_group_id": self.task_template_group_id,
            "task_manifest_digest": self.task_manifest_digest,
            "event_ids": [event.event_id for event in self.events],
        }

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "schema": PROCESS_CORPUS_SCHEMA,
            "corpus_id": self.corpus_id,
            **self.identity_payload(),
            "events": [event.payload() for event in self.events],
        }

    @classmethod
    def from_payload(cls, value: object) -> "ProcessCorpus":
        fields = {
            "schema_version", "schema", "corpus_id", "split_role", "family_id",
            "task_template_group_id", "task_manifest_digest", "event_ids", "events",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("malformed process corpus")
        if value["schema"] != PROCESS_CORPUS_SCHEMA:
            raise ValueError("unsupported process corpus schema")
        if not isinstance(value["event_ids"], list) or not isinstance(value["events"], list):
            raise ValueError("malformed process corpus events")
        try:
            events = tuple(ProcessEvent.from_payload(item) for item in value["events"])
            result = cls(
                corpus_id=value["corpus_id"],
                split_role=value["split_role"],
                family_id=value["family_id"],
                task_template_group_id=value["task_template_group_id"],
                task_manifest_digest=value["task_manifest_digest"],
                events=events,
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed process corpus") from exc
        if list(event.event_id for event in result.events) != value["event_ids"]:
            raise ValueError("process corpus event identity list mismatch")
        if result.payload() != dict(value):
            raise ValueError("non-canonical process corpus")
        return result


class JsonProcessCorpusStore:
    """Atomic owner-controlled storage for one process corpus."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    @contextmanager
    def _lock(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def put(self, corpus: ProcessCorpus) -> tuple[Path, bool]:
        if not isinstance(corpus, ProcessCorpus):
            raise TypeError("process corpus has the wrong type")
        canonical = json.dumps(corpus.payload(), ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                if self.path.read_text(encoding="utf-8") != canonical:
                    raise ValueError("process corpus conflicts with existing file")
                return self.path, False
            fd, temporary = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(canonical)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return self.path, True

    def get(self) -> ProcessCorpus | None:
        with self._lock():
            if not self.path.exists():
                return None
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("malformed process corpus JSON") from exc
            return ProcessCorpus.from_payload(value)


@dataclass(frozen=True, slots=True)
class ProcessSignalCensus:
    """Stage/process coverage summary for one content-free corpus."""

    event_count: int
    policy_bound_count: int
    receipt_bound_count: int
    host_event_count: int
    kind_counts: Mapping[str, int]
    status_counts: Mapping[str, int]
    reason_counts: Mapping[str, int]
    layer_counts: Mapping[str, int]
    distinct_source_revision_count: int

    def __post_init__(self) -> None:
        values = (
            self.event_count,
            self.policy_bound_count,
            self.receipt_bound_count,
            self.host_event_count,
            self.distinct_source_revision_count,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("process census counts must be non-negative integers")
        if any(value > self.event_count for value in values[1:4]):
            raise ValueError("process census counts cannot exceed event count")
        for mapping in (self.kind_counts, self.status_counts, self.reason_counts, self.layer_counts):
            if any(not isinstance(key, str) or type(value) is not int or value < 0 for key, value in mapping.items()):
                raise ValueError("process census buckets are invalid")
        object.__setattr__(self, "kind_counts", dict(sorted(self.kind_counts.items())))
        object.__setattr__(self, "status_counts", dict(sorted(self.status_counts.items())))
        object.__setattr__(self, "reason_counts", dict(sorted(self.reason_counts.items())))
        object.__setattr__(self, "layer_counts", dict(sorted(self.layer_counts.items())))

    @property
    def signal_coverage(self) -> float | None:
        return self.policy_bound_count / self.event_count if self.event_count else None

    @property
    def receipt_coverage(self) -> float | None:
        return self.receipt_bound_count / self.event_count if self.event_count else None

    def payload(self) -> dict[str, object]:
        return {
            "eventCount": self.event_count,
            "policyBoundCount": self.policy_bound_count,
            "policyBoundCoverage": self.signal_coverage,
            "receiptBoundCount": self.receipt_bound_count,
            "receiptBoundCoverage": self.receipt_coverage,
            "hostEventCount": self.host_event_count,
            "kindCounts": dict(self.kind_counts),
            "statusCounts": dict(self.status_counts),
            "reasonCounts": dict(self.reason_counts),
            "layerCounts": dict(self.layer_counts),
            "distinctSourceRevisionCount": self.distinct_source_revision_count,
        }


def census_process_events(events: Iterable[ProcessEvent]) -> ProcessSignalCensus:
    """Count process signals without using application score or content."""

    items = tuple(events)
    if any(not isinstance(item, ProcessEvent) for item in items):
        raise TypeError("process census requires ProcessEvent values")
    kinds: dict[str, int] = {}
    statuses: dict[str, int] = {}
    reasons: dict[str, int] = {}
    layers: dict[str, int] = {}
    for event in items:
        kinds[event.kind.value] = kinds.get(event.kind.value, 0) + 1
        statuses[event.status.value] = statuses.get(event.status.value, 0) + 1
        for reason in event.reason_codes:
            reasons[reason] = reasons.get(reason, 0) + 1
        if event.policy_layer is not None:
            layer = event.policy_layer.value
            layers[layer] = layers.get(layer, 0) + 1
    return ProcessSignalCensus(
        event_count=len(items),
        policy_bound_count=sum(item.policy_decision_id is not None for item in items),
        receipt_bound_count=sum(bool(item.execution_receipt_ids) for item in items),
        host_event_count=len({item.host_event_id for item in items}),
        kind_counts=kinds,
        status_counts=statuses,
        reason_counts=reasons,
        layer_counts=layers,
        distinct_source_revision_count=len({item.source_revision for item in items}),
    )


def ensure_process_corpus_has_no_evaluation_fields(value: object) -> None:
    """Reject score/grader/answer fields before learner ingestion."""

    # Accept no naming convention as a bypass: benchmark reporters commonly
    # use camelCase while runtime contracts use snake_case.  Delimiters are
    # removed before comparison, but a suffix such as ``score_digest`` is not
    # rejected unless the field itself is an evaluation field.
    forbidden = {
        "score", "taskscore", "officialscore", "officialevaluation", "grader",
        "answer", "answerkey", "hiddenexpectation", "judge", "expectation",
    }

    def normalized_key(key: object) -> str:
        return re.sub(r"[^a-z0-9]", "", str(key).lower())

    def walk(item: object) -> None:
        if isinstance(item, Mapping):
            overlap = forbidden.intersection(normalized_key(key) for key in item)
            if overlap:
                raise ValueError("evaluation-only field leaked into process corpus")
            for child in item.values():
                walk(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                walk(child)

    walk(value)


__all__ = [
    "PROCESS_CORPUS_SCHEMA_VERSION",
    "PROCESS_CORPUS_SCHEMA",
    "ProcessCorpus",
    "JsonProcessCorpusStore",
    "ProcessSignalCensus",
    "census_process_events",
    "ensure_process_corpus_has_no_evaluation_fields",
]
