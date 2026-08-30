"""Durable, content-free identity for a future N+1 policy intervention."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Mapping

from .policy_contracts import PolicyLayer, content_digest
from .policy_feasibility import (
    LayerIntervention,
    OptimizerHypothesisDecision,
    OptimizerHypothesisProjection,
)


FEASIBILITY_INTERVENTION_SCHEMA_VERSION = 1


class InterventionPathStatus(StrEnum):
    REPLAYED = "replayed"
    REJECTED = "rejected"


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _normalize_ids(values: object, name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple of strings")
    result = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must be unique")
    return result


@dataclass(frozen=True, slots=True)
class FeasibilityInterventionPath:
    """Join an optimizer projection to a replayed target-layer intervention.

    The path deliberately stores only logical identities and fingerprints.  It
    is safe to persist in a shared evidence ledger without copying source
    messages, prompts, memory text, or benchmark outcomes.
    """

    path_id: str
    projection_id: str
    target_layer: PolicyLayer
    parent_artifact_id: str
    candidate_artifact_id: str
    case_id: str
    intervention_fingerprint: str
    status: InterventionPathStatus
    reason_codes: tuple[str, ...]
    schema_version: int = FEASIBILITY_INTERVENTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FEASIBILITY_INTERVENTION_SCHEMA_VERSION:
            raise ValueError("unsupported feasibility intervention path schema")
        for value, name in (
            (self.path_id, "intervention path ID"),
            (self.projection_id, "intervention projection ID"),
            (self.parent_artifact_id, "intervention parent artifact ID"),
            (self.candidate_artifact_id, "intervention candidate artifact ID"),
            (self.case_id, "intervention case ID"),
            (self.intervention_fingerprint, "intervention fingerprint"),
        ):
            _require_text(value, name)
        object.__setattr__(self, "target_layer", PolicyLayer(self.target_layer))
        object.__setattr__(self, "status", InterventionPathStatus(self.status))
        reasons = _normalize_ids(self.reason_codes, "intervention path reason codes")
        object.__setattr__(self, "reason_codes", reasons)
        expected = f"feasibility-intervention.{content_digest(self.identity_payload())[:40]}"
        if self.path_id != expected:
            raise ValueError("feasibility intervention path ID mismatch")

    @classmethod
    def from_projection_case(
        cls,
        projection: OptimizerHypothesisProjection,
        case: LayerIntervention,
        *,
        reason_codes: Iterable[str] = ("candidate_replayed",),
    ) -> "FeasibilityInterventionPath":
        if not isinstance(projection, OptimizerHypothesisProjection):
            raise TypeError("intervention projection has the wrong type")
        if not isinstance(case, LayerIntervention):
            raise TypeError("intervention case has the wrong type")
        if projection.decision is not OptimizerHypothesisDecision.PROPOSE:
            raise ValueError("intervention path requires a proposal")
        if projection.target_layer is not case.target_layer:
            raise ValueError("intervention projection and case layers differ")
        if projection.candidate_artifact_id != case.candidate_artifact.artifact_id:
            raise ValueError("intervention candidate artifact differs from projection")
        if projection.parent_artifact_id != case.parent_artifact.artifact_id:
            raise ValueError("intervention parent artifact differs from projection")
        reasons = _normalize_ids(tuple(reason_codes), "intervention path reason codes")
        values = {
            "projection_id": projection.projection_id,
            "target_layer": projection.target_layer.value,
            "parent_artifact_id": projection.parent_artifact_id,
            "candidate_artifact_id": projection.candidate_artifact_id,
            "case_id": case.case_id,
            "intervention_fingerprint": case.intervention_fingerprint,
            "status": InterventionPathStatus.REPLAYED.value,
            "reason_codes": list(reasons),
            "schema_version": FEASIBILITY_INTERVENTION_SCHEMA_VERSION,
        }
        return cls(
            path_id=f"feasibility-intervention.{content_digest(values)[:40]}",
            projection_id=projection.projection_id,
            target_layer=projection.target_layer,
            parent_artifact_id=projection.parent_artifact_id,
            candidate_artifact_id=projection.candidate_artifact_id or "",
            case_id=case.case_id,
            intervention_fingerprint=case.intervention_fingerprint,
            status=InterventionPathStatus.REPLAYED,
            reason_codes=reasons,
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "projection_id": self.projection_id,
            "target_layer": self.target_layer.value,
            "parent_artifact_id": self.parent_artifact_id,
            "candidate_artifact_id": self.candidate_artifact_id,
            "case_id": self.case_id,
            "intervention_fingerprint": self.intervention_fingerprint,
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
        }

    def payload(self) -> dict[str, object]:
        return {"path_id": self.path_id, **self.identity_payload()}

    @classmethod
    def from_payload(cls, value: object) -> "FeasibilityInterventionPath":
        fields = {
            "path_id", "schema_version", "projection_id", "target_layer",
            "parent_artifact_id", "candidate_artifact_id", "case_id",
            "intervention_fingerprint", "status", "reason_codes",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("malformed feasibility intervention path")
        if not isinstance(value["reason_codes"], list):
            raise ValueError("malformed feasibility intervention path")
        try:
            return cls(
                path_id=value["path_id"],
                projection_id=value["projection_id"],
                target_layer=value["target_layer"],
                parent_artifact_id=value["parent_artifact_id"],
                candidate_artifact_id=value["candidate_artifact_id"],
                case_id=value["case_id"],
                intervention_fingerprint=value["intervention_fingerprint"],
                status=value["status"],
                reason_codes=tuple(value["reason_codes"]),
                schema_version=value["schema_version"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed feasibility intervention path") from exc


class JsonFeasibilityInterventionPathStore:
    """Crash-safe, idempotent JSONL store for future intervention paths."""

    def __init__(self, path: Path) -> None:
        self.path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    @contextmanager
    def _lock(self):
        if self.path.is_symlink():
            raise ValueError("feasibility intervention path cannot be a symlink")
        if self.lock_path.is_symlink():
            raise ValueError("feasibility intervention path lock cannot be a symlink")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read(self) -> dict[str, str]:
        if self.path.is_symlink():
            raise ValueError("feasibility intervention path cannot be a symlink")
        if not self.path.exists():
            return {}
        records: dict[str, str] = {}
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), 1
        ):
            try:
                record = FeasibilityInterventionPath.from_payload(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"malformed feasibility intervention path at line {line_number}"
                ) from exc
            canonical = json.dumps(
                record.payload(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
            )
            previous = records.get(record.path_id)
            if previous is not None and previous != canonical:
                raise ValueError("conflicting feasibility intervention path")
            records[record.path_id] = canonical
        return records

    @property
    def records(self) -> tuple[FeasibilityInterventionPath, ...]:
        with self._lock():
            records = self._read()
        return tuple(
            FeasibilityInterventionPath.from_payload(json.loads(value))
            for value in records.values()
        )

    def put(self, record: FeasibilityInterventionPath) -> tuple[FeasibilityInterventionPath, bool]:
        if not isinstance(record, FeasibilityInterventionPath):
            raise TypeError("intervention path record has the wrong type")
        canonical = json.dumps(
            record.payload(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        with self._lock():
            records = self._read()
            previous = records.get(record.path_id)
            if previous is not None:
                if previous != canonical:
                    raise ValueError("conflicting feasibility intervention path")
                return record, False
            payload = [json.loads(value) for value in records.values()]
            payload.append(record.payload())
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(
                prefix=f".{self.path.name}.", dir=self.path.parent
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    for item in payload:
                        handle.write(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return record, True


__all__ = [
    "FEASIBILITY_INTERVENTION_SCHEMA_VERSION",
    "InterventionPathStatus",
    "FeasibilityInterventionPath",
    "JsonFeasibilityInterventionPathStore",
]
