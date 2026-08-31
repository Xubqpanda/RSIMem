"""Final-evaluation reporter contract, isolated from learner evidence."""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import Mapping

from .evidence_planes import EvidencePlane, EvidenceSourceKind, validate_plane_source
from ._atomic_jsonl import replace_jsonl


FINAL_EVALUATION_SCHEMA_VERSION = 1
FINAL_EVALUATION_SCHEMA = "rsimem-final-evaluation-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,255}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


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


def _time(value: object, name: str) -> datetime:
    if not isinstance(value, str) or _ISO_UTC.fullmatch(value) is None:
        raise ValueError(f"{name} must be an ISO UTC timestamp")
    try:
        return datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO UTC timestamp") from exc


@dataclass(frozen=True, slots=True)
class FinalEvaluationRecord:
    report_id: str
    candidate_artifact_id: str
    run_id: str
    candidate_frozen_at: str
    run_completed_at: str
    score_read_at: str
    score_digest: str
    metric_name: str
    metric_value: float
    schema_version: int = FINAL_EVALUATION_SCHEMA_VERSION
    evidence_plane: EvidencePlane = EvidencePlane.FINAL_EVALUATION
    evidence_source: EvidenceSourceKind = EvidenceSourceKind.FINAL_REPORTER

    def __post_init__(self) -> None:
        if self.schema_version != FINAL_EVALUATION_SCHEMA_VERSION:
            raise ValueError("unsupported final evaluation schema")
        plane, source = validate_plane_source(self.evidence_plane, self.evidence_source)
        if plane != EvidencePlane.FINAL_EVALUATION or source != EvidenceSourceKind.FINAL_REPORTER:
            raise ValueError("final evaluation requires final reporter plane")
        object.__setattr__(self, "evidence_plane", plane)
        object.__setattr__(self, "evidence_source", source)
        for value, name in (
            (self.report_id, "final evaluation report ID"),
            (self.candidate_artifact_id, "candidate artifact ID"),
            (self.run_id, "evaluation run ID"),
            (self.metric_name, "evaluation metric name"),
        ):
            _id(value, name)
        for value, name in (
            (self.candidate_frozen_at, "candidate freeze timestamp"),
            (self.run_completed_at, "run completion timestamp"),
            (self.score_read_at, "score read timestamp"),
        ):
            _time(value, name)
        if _time(self.score_read_at, "score read timestamp") <= _time(self.candidate_frozen_at, "candidate freeze timestamp"):
            raise ValueError("final score must be read after candidate freeze")
        if _time(self.run_completed_at, "run completion timestamp") <= _time(
            self.candidate_frozen_at,
            "candidate freeze timestamp",
        ):
            raise ValueError("evaluation run must complete after candidate freeze")
        if _time(self.score_read_at, "score read timestamp") <= _time(self.run_completed_at, "run completion timestamp"):
            raise ValueError("final score must be read after run completion")
        _sha(self.score_digest, "final score digest")
        if isinstance(self.metric_value, bool) or not isinstance(self.metric_value, (int, float)):
            raise TypeError("final metric value must be numeric")
        if not (-1e12 < float(self.metric_value) < 1e12):
            raise ValueError("final metric value is outside the bounded range")
        expected = f"final-evaluation.{_digest(self._identity_payload())[:40]}"
        if self.report_id != expected:
            raise ValueError("final evaluation report ID mismatch")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "candidate_artifact_id": self.candidate_artifact_id,
            "run_id": self.run_id,
            "candidate_frozen_at": self.candidate_frozen_at,
            "run_completed_at": self.run_completed_at,
            "score_read_at": self.score_read_at,
            "score_digest": self.score_digest,
            "metric_name": self.metric_name,
            "metric_value": float(self.metric_value),
            "evidence_plane": self.evidence_plane.value,
            "evidence_source": self.evidence_source.value,
        }

    @classmethod
    def create(
        cls,
        *,
        candidate_artifact_id: str,
        run_id: str,
        candidate_frozen_at: str,
        run_completed_at: str,
        score_read_at: str,
        score_digest: str,
        metric_name: str,
        metric_value: float,
    ) -> "FinalEvaluationRecord":
        values = {
            "candidate_artifact_id": candidate_artifact_id,
            "run_id": run_id,
            "candidate_frozen_at": candidate_frozen_at,
            "run_completed_at": run_completed_at,
            "score_read_at": score_read_at,
            "score_digest": score_digest,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "schema_version": FINAL_EVALUATION_SCHEMA_VERSION,
            "evidence_plane": EvidencePlane.FINAL_EVALUATION,
            "evidence_source": EvidenceSourceKind.FINAL_REPORTER,
        }
        return cls(report_id=f"final-evaluation.{_digest(values)[:40]}", **values)

    def payload(self) -> dict[str, object]:
        return {"schema": FINAL_EVALUATION_SCHEMA, "report_id": self.report_id, **self._identity_payload()}

    @classmethod
    def from_payload(cls, value: object) -> "FinalEvaluationRecord":
        fields = {
            "schema", "report_id", "schema_version", "candidate_artifact_id", "run_id",
            "candidate_frozen_at", "run_completed_at", "score_read_at", "score_digest",
            "metric_name", "metric_value", "evidence_plane", "evidence_source",
        }
        if not isinstance(value, Mapping) or set(value) != fields or value.get("schema") != FINAL_EVALUATION_SCHEMA:
            raise ValueError("malformed final evaluation record")
        try:
            return cls(
                report_id=value["report_id"], candidate_artifact_id=value["candidate_artifact_id"],
                run_id=value["run_id"], candidate_frozen_at=value["candidate_frozen_at"],
                run_completed_at=value["run_completed_at"], score_read_at=value["score_read_at"],
                score_digest=value["score_digest"], metric_name=value["metric_name"],
                metric_value=value["metric_value"], schema_version=value["schema_version"],
                evidence_plane=value["evidence_plane"], evidence_source=value["evidence_source"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed final evaluation record") from exc


class JsonFinalEvaluationStore:
    """Crash-safe append-only storage owned exclusively by the final reporter.

    This store accepts only :class:`FinalEvaluationRecord` values.  Keeping a
    separate file and reader prevents score-bearing records from entering the
    pure-process or optimizer corpus by accident.
    """

    def __init__(self, path: Path) -> None:
        # Preserve the final component so final-evaluation state cannot be
        # redirected through a symlink to a learner-owned file.
        self.path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self._records: dict[str, str] = {}
        self._load()

    @staticmethod
    def _canonical(value: Mapping[str, object]) -> str:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    def _load(self) -> None:
        if self.path.is_symlink():
            raise ValueError("final evaluation store cannot be a symlink")
        if not self.path.exists():
            return
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                record = FinalEvaluationRecord.from_payload(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"malformed final evaluation record at line {line_number}"
                ) from exc
            canonical = self._canonical(record.payload())
            previous = self._records.get(record.report_id)
            if previous is not None and previous != canonical:
                raise ValueError("conflicting final evaluation record")
            self._records[record.report_id] = canonical

    def records(self) -> tuple[FinalEvaluationRecord, ...]:
        """Reload and return records in stable report-ID order."""

        if self.path.is_symlink():
            raise ValueError("final evaluation store cannot be a symlink")
        if self.lock_path.is_symlink():
            raise ValueError("final evaluation store lock cannot be a symlink")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            try:
                self._records.clear()
                self._load()
                return tuple(
                    FinalEvaluationRecord.from_payload(json.loads(value))
                    for _, value in sorted(self._records.items())
                )
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def append(self, record: FinalEvaluationRecord) -> bool:
        """Persist a record once; conflicting identity fails closed."""

        if not isinstance(record, FinalEvaluationRecord):
            raise TypeError("final evaluation store accepts FinalEvaluationRecord only")
        serialized = self._canonical(record.payload())
        if self.path.is_symlink():
            raise ValueError("final evaluation store cannot be a symlink")
        if self.lock_path.is_symlink():
            raise ValueError("final evaluation store lock cannot be a symlink")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                self._records.clear()
                self._load()
                previous = self._records.get(record.report_id)
                if previous is not None:
                    if previous != serialized:
                        raise ValueError("conflicting final evaluation record")
                    return False
                replace_jsonl(
                    self.path,
                    tuple(value for _, value in sorted(self._records.items()))
                    + (serialized,),
                    error_name="final evaluation store",
                )
                self._records[record.report_id] = serialized
                return True
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class FinalEvaluationReporter:
    """Read official scores only after a frozen candidate run completes.

    The reporter is intentionally the only convenience API that accepts a
    score reader.  It validates the chronology before invoking that callback,
    then persists the resulting record into the isolated final-evaluation
    store.  Learner/process-corpus code has no path through this class.
    """

    def __init__(self, store: JsonFinalEvaluationStore) -> None:
        if not isinstance(store, JsonFinalEvaluationStore):
            raise TypeError("final reporter requires a final evaluation store")
        self.store = store

    def read_after_completion(
        self,
        *,
        candidate_artifact_id: str,
        run_id: str,
        candidate_frozen_at: str,
        run_completed_at: str,
        score_read_at: str,
        metric_name: str,
        score_reader: Callable[[], object],
    ) -> FinalEvaluationRecord:
        """Invoke ``score_reader`` only after chronology checks pass.

        ``score_reader`` may return a numeric metric directly.  For reporters
        that have a canonical score payload, it may instead return a mapping
        with exactly ``metric_value`` and ``score_digest``; the latter digest
        is retained without exposing the payload to process evidence.
        """

        if not callable(score_reader):
            raise TypeError("final score reader must be callable")
        frozen = _time(candidate_frozen_at, "candidate freeze timestamp")
        completed = _time(run_completed_at, "run completion timestamp")
        read_at = _time(score_read_at, "score read timestamp")
        if completed <= frozen:
            raise ValueError("evaluation run must complete after candidate freeze")
        if read_at <= completed:
            raise ValueError("final score must be read after run completion")

        # Do not move this callback above the chronology checks: reading an
        # official score is the externally visible final-evaluation boundary.
        raw_score = score_reader()
        score_digest: str | None = None
        metric_value: object = raw_score
        if isinstance(raw_score, Mapping):
            if set(raw_score) != {"metric_value", "score_digest"}:
                raise ValueError("final score payload must contain metric value and digest")
            metric_value = raw_score["metric_value"]
            score_digest = raw_score["score_digest"]
        if isinstance(metric_value, bool) or not isinstance(metric_value, (int, float)):
            raise TypeError("final score reader must return a numeric metric")
        if score_digest is None:
            score_digest = _digest({
                "metric_name": metric_name,
                "metric_value": float(metric_value),
            })
        _sha(score_digest, "final score digest")
        record = FinalEvaluationRecord.create(
            candidate_artifact_id=candidate_artifact_id,
            run_id=run_id,
            candidate_frozen_at=candidate_frozen_at,
            run_completed_at=run_completed_at,
            score_read_at=score_read_at,
            score_digest=score_digest,
            metric_name=metric_name,
            metric_value=float(metric_value),
        )
        self.store.append(record)
        return record


def _score_file_reader(path: Path) -> Callable[[], object]:
    """Return a lazy score reader for the final-reporter CLI.

    The file is intentionally opened inside the callback.  The reporter has
    already checked candidate/run chronology before this callback executes.
    """

    resolved = path.expanduser().resolve()

    def read() -> object:
        try:
            value = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("final score file cannot be read") from exc
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        if isinstance(value, Mapping):
            return dict(value)
        raise ValueError("final score file must contain a number or score payload")

    return read


def main(argv: list[str] | None = None) -> int:
    """Record one final score without exposing it to process/optimizer stores."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--score-file", type=Path, required=True)
    parser.add_argument("--candidate-artifact-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-frozen-at", required=True)
    parser.add_argument("--run-completed-at", required=True)
    parser.add_argument("--score-read-at", required=True)
    parser.add_argument("--metric-name", required=True)
    args = parser.parse_args(argv)
    record = FinalEvaluationReporter(
        JsonFinalEvaluationStore(args.store)
    ).read_after_completion(
        candidate_artifact_id=args.candidate_artifact_id,
        run_id=args.run_id,
        candidate_frozen_at=args.candidate_frozen_at,
        run_completed_at=args.run_completed_at,
        score_read_at=args.score_read_at,
        metric_name=args.metric_name,
        score_reader=_score_file_reader(args.score_file),
    )
    print(json.dumps(record.payload(), ensure_ascii=True, sort_keys=True))
    return 0


__all__ = [
    "FINAL_EVALUATION_SCHEMA",
    "FINAL_EVALUATION_SCHEMA_VERSION",
    "FinalEvaluationRecord",
    "FinalEvaluationReporter",
    "JsonFinalEvaluationStore",
]


if __name__ == "__main__":
    raise SystemExit(main())
