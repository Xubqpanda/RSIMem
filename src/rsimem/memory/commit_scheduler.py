"""Crash-safe, backend-agnostic commit scheduling (RSIMem 2D.2)."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable, Protocol

from .policy_contracts import CommitDecision, CommitMode, content_digest


class CommitScheduleStatus(StrEnum):
    PENDING = "pending"
    COMMITTED = "committed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class CommitSchedule:
    schedule_id: str
    decision_id: str
    mutation_ids: tuple[str, ...]
    expected_revision: str
    execution_boundary: str
    status: CommitScheduleStatus = CommitScheduleStatus.PENDING
    receipt_id: str | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.schedule_id, "schedule ID"),
            (self.decision_id, "decision ID"),
            (self.expected_revision, "expected revision"),
            (self.execution_boundary, "execution boundary"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        mutation_ids = tuple(self.mutation_ids)
        if not mutation_ids or len(mutation_ids) != len(set(mutation_ids)) or any(not item.strip() for item in mutation_ids):
            raise ValueError("commit schedule mutation IDs are invalid")
        object.__setattr__(self, "mutation_ids", mutation_ids)
        object.__setattr__(self, "status", CommitScheduleStatus(self.status))
        if self.status == CommitScheduleStatus.COMMITTED and not self.receipt_id:
            raise ValueError("committed schedule requires receipt ID")
        if self.status == CommitScheduleStatus.FAILED and not self.failure_reason:
            raise ValueError("failed schedule requires failure reason")
        if self.receipt_id is not None and not self.receipt_id.strip():
            raise ValueError("schedule receipt ID must not be empty")

    def payload(self) -> dict[str, object]:
        return {
            "schedule_id": self.schedule_id,
            "decision_id": self.decision_id,
            "mutation_ids": list(self.mutation_ids),
            "expected_revision": self.expected_revision,
            "execution_boundary": self.execution_boundary,
            "status": self.status.value,
            "receipt_id": self.receipt_id,
            "failure_reason": self.failure_reason,
        }


class CommitScheduleStore(Protocol):
    def get(self, schedule_id: str) -> CommitSchedule | None: ...

    def put(self, schedule: CommitSchedule) -> None: ...

    def all(self) -> tuple[CommitSchedule, ...]: ...


class InMemoryCommitScheduleStore:
    def __init__(self) -> None:
        self._items: dict[str, CommitSchedule] = {}

    def get(self, schedule_id: str) -> CommitSchedule | None:
        return self._items.get(schedule_id)

    def put(self, schedule: CommitSchedule) -> None:
        previous = self._items.get(schedule.schedule_id)
        if previous is not None and not _same_schedule_identity(previous, schedule):
            raise ValueError("commit schedule identity already has a different payload")
        self._items[schedule.schedule_id] = schedule

    def all(self) -> tuple[CommitSchedule, ...]:
        return tuple(self._items.values())


class JsonCommitScheduleStore:
    """Small crash-safe JSON store for pending schedule metadata."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    def _read(self) -> dict[str, object]:
        if not self.path.exists():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("malformed commit schedule store") from exc
        if not isinstance(value, dict):
            raise ValueError("malformed commit schedule store")
        return value

    def get(self, schedule_id: str) -> CommitSchedule | None:
        with self._lock(shared=True):
            raw = self._read().get(schedule_id)
        if raw is None:
            return None
        return self._decode(raw)

    def all(self) -> tuple[CommitSchedule, ...]:
        with self._lock(shared=True):
            raw = self._read()
        return tuple(self._decode(value) for value in raw.values())

    def put(self, schedule: CommitSchedule) -> None:
        with self._lock(shared=False):
            payload = self._read()
            previous = payload.get(schedule.schedule_id)
            if previous is not None:
                existing = self._decode(previous)
                if not _same_schedule_identity(existing, schedule):
                    raise ValueError("commit schedule identity already has a different payload")
            payload[schedule.schedule_id] = schedule.payload()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

    @staticmethod
    def _decode(raw: object) -> CommitSchedule:
        if not isinstance(raw, dict):
            raise ValueError("malformed commit schedule entry")
        try:
            return CommitSchedule(
                schedule_id=raw["schedule_id"],
                decision_id=raw["decision_id"],
                mutation_ids=tuple(raw["mutation_ids"]),
                expected_revision=raw["expected_revision"],
                execution_boundary=raw["execution_boundary"],
                status=raw["status"],
                receipt_id=raw.get("receipt_id"),
                failure_reason=raw.get("failure_reason"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed commit schedule entry") from exc

    from contextlib import contextmanager

    @contextmanager
    def _lock(self, *, shared: bool):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


ApplyCommit = Callable[[tuple[str, ...]], str]


def _same_schedule_identity(first: CommitSchedule, second: CommitSchedule) -> bool:
    return (
        first.schedule_id,
        first.decision_id,
        first.mutation_ids,
        first.expected_revision,
        first.execution_boundary,
    ) == (
        second.schedule_id,
        second.decision_id,
        second.mutation_ids,
        second.expected_revision,
        second.execution_boundary,
    )


class CommitScheduler:
    def __init__(self, store: CommitScheduleStore | None = None) -> None:
        self.store = store or InMemoryCommitScheduleStore()

    def schedule(self, decision: CommitDecision, *, boundary: str) -> CommitSchedule | None:
        if decision.action.value != "RUN":
            if decision.mutation_ids:
                raise ValueError("non-run commit decision cannot schedule mutations")
            return None
        if not boundary.strip():
            raise ValueError("commit boundary must not be empty")
        if decision.expected_revision is None:
            raise ValueError("commit decision expected revision is missing")
        schedule_id = f"commit-schedule.{content_digest({'decision_id': decision.decision_id, 'boundary': boundary})[:40]}"
        candidate = CommitSchedule(
            schedule_id=schedule_id,
            decision_id=decision.decision_id,
            mutation_ids=decision.mutation_ids,
            expected_revision=decision.expected_revision,
            execution_boundary=boundary,
        )
        existing = self.store.get(schedule_id)
        if existing is not None:
            return existing
        self.store.put(candidate)
        return candidate

    def execute(
        self,
        schedule_id: str,
        *,
        current_revision: str,
        apply: ApplyCommit,
    ) -> CommitSchedule:
        schedule = self.store.get(schedule_id)
        if schedule is None:
            raise ValueError("unknown commit schedule")
        if schedule.status != CommitScheduleStatus.PENDING:
            return schedule
        if current_revision != schedule.expected_revision:
            failed = CommitSchedule(**{**schedule.payload(), "status": CommitScheduleStatus.FAILED, "failure_reason": "stale_revision"})
            self.store.put(failed)
            return failed
        try:
            receipt_id = apply(schedule.mutation_ids)
            if not isinstance(receipt_id, str) or not receipt_id.strip():
                raise ValueError("commit executor returned an invalid receipt")
            committed = CommitSchedule(**{**schedule.payload(), "status": CommitScheduleStatus.COMMITTED, "receipt_id": receipt_id, "failure_reason": None})
        except Exception as exc:
            committed = CommitSchedule(**{**schedule.payload(), "status": CommitScheduleStatus.FAILED, "failure_reason": type(exc).__name__})
        self.store.put(committed)
        return committed

    def cancel(self, schedule_id: str) -> CommitSchedule:
        schedule = self.store.get(schedule_id)
        if schedule is None:
            raise ValueError("unknown commit schedule")
        if schedule.status != CommitScheduleStatus.PENDING:
            return schedule
        cancelled = CommitSchedule(**{**schedule.payload(), "status": CommitScheduleStatus.CANCELLED, "failure_reason": "cancelled"})
        self.store.put(cancelled)
        return cancelled


__all__ = [
    "CommitScheduleStatus",
    "CommitSchedule",
    "CommitScheduleStore",
    "InMemoryCommitScheduleStore",
    "JsonCommitScheduleStore",
    "CommitScheduler",
]
