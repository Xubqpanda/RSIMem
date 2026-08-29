"""Final-evaluation reporter contract, isolated from learner evidence."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from .evidence_planes import EvidencePlane, EvidenceSourceKind, validate_plane_source


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


__all__ = [
    "FINAL_EVALUATION_SCHEMA",
    "FINAL_EVALUATION_SCHEMA_VERSION",
    "FinalEvaluationRecord",
]
