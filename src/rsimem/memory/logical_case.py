"""Stable logical-case identity and replicate-safe aggregation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Mapping


LOGICAL_CASE_SCHEMA_VERSION = 1
LOGICAL_CASE_SCHEMA = "rsimem-logical-case-v1"
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


class LogicalCaseResolutionStatus(StrEnum):
    UNRESOLVED = "unresolved"
    CONSISTENT = "consistent"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class LogicalCaseIdentity:
    logical_case_id: str
    frozen_policy_digest: str
    source_task_template_id: str
    source_extraction_set_id: str
    future_task_template_id: str
    observation_window: str
    schema_version: int = LOGICAL_CASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LOGICAL_CASE_SCHEMA_VERSION:
            raise ValueError("unsupported logical-case schema")
        _sha(self.frozen_policy_digest, "frozen policy digest")
        for value, name in (
            (self.source_task_template_id, "source task template ID"),
            (self.source_extraction_set_id, "source extraction set ID"),
            (self.future_task_template_id, "future task template ID"),
            (self.observation_window, "observation window"),
            (self.logical_case_id, "logical case ID"),
        ):
            _id(value, name)
        expected = f"logical-case.{_digest(self._identity_payload())[:40]}"
        if self.logical_case_id != expected:
            raise ValueError("logical case ID mismatch")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "frozen_policy_digest": self.frozen_policy_digest,
            "source_task_template_id": self.source_task_template_id,
            "source_extraction_set_id": self.source_extraction_set_id,
            "future_task_template_id": self.future_task_template_id,
            "observation_window": self.observation_window,
        }

    @classmethod
    def create(
        cls,
        *,
        frozen_policy_digest: str,
        source_task_template_id: str,
        source_extraction_set_id: str,
        future_task_template_id: str,
        observation_window: str,
    ) -> "LogicalCaseIdentity":
        values = {
            "frozen_policy_digest": frozen_policy_digest,
            "source_task_template_id": source_task_template_id,
            "source_extraction_set_id": source_extraction_set_id,
            "future_task_template_id": future_task_template_id,
            "observation_window": observation_window,
            "schema_version": LOGICAL_CASE_SCHEMA_VERSION,
        }
        return cls(logical_case_id=f"logical-case.{_digest(values)[:40]}", **values)

    def payload(self) -> dict[str, object]:
        return {"schema": LOGICAL_CASE_SCHEMA, "logical_case_id": self.logical_case_id, **self._identity_payload()}

    @classmethod
    def from_payload(cls, value: object) -> "LogicalCaseIdentity":
        fields = {
            "schema", "logical_case_id", "schema_version", "frozen_policy_digest",
            "source_task_template_id", "source_extraction_set_id",
            "future_task_template_id", "observation_window",
        }
        if not isinstance(value, Mapping) or set(value) != fields or value.get("schema") != LOGICAL_CASE_SCHEMA:
            raise ValueError("malformed logical case identity")
        try:
            return cls(
                logical_case_id=value["logical_case_id"],
                frozen_policy_digest=value["frozen_policy_digest"],
                source_task_template_id=value["source_task_template_id"],
                source_extraction_set_id=value["source_extraction_set_id"],
                future_task_template_id=value["future_task_template_id"],
                observation_window=value["observation_window"],
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed logical case identity") from exc


@dataclass(frozen=True, slots=True)
class PhysicalObservation:
    observation_id: str
    logical_case_id: str
    replicate_id: str
    result_label: str
    observation_digest: str
    schema_version: int = LOGICAL_CASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LOGICAL_CASE_SCHEMA_VERSION:
            raise ValueError("unsupported physical observation schema")
        for value, name in (
            (self.observation_id, "physical observation ID"),
            (self.logical_case_id, "logical case ID"),
            (self.replicate_id, "replicate ID"),
            (self.result_label, "observation result label"),
        ):
            _id(value, name)
        _sha(self.observation_digest, "physical observation digest")
        expected = f"physical-observation.{_digest(self._identity_payload())[:40]}"
        if self.observation_id != expected:
            raise ValueError("physical observation ID mismatch")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "logical_case_id": self.logical_case_id,
            "replicate_id": self.replicate_id,
            "result_label": self.result_label,
            "observation_digest": self.observation_digest,
        }

    @classmethod
    def create(
        cls,
        *,
        logical_case_id: str,
        replicate_id: str,
        result_label: str,
        observation_digest: str,
    ) -> "PhysicalObservation":
        values = {
            "logical_case_id": logical_case_id,
            "replicate_id": replicate_id,
            "result_label": result_label,
            "observation_digest": observation_digest,
            "schema_version": LOGICAL_CASE_SCHEMA_VERSION,
        }
        return cls(observation_id=f"physical-observation.{_digest(values)[:40]}", **values)

    def payload(self) -> dict[str, object]:
        return {"schema": LOGICAL_CASE_SCHEMA, "observation_id": self.observation_id, **self._identity_payload()}

    @classmethod
    def from_payload(cls, value: object) -> "PhysicalObservation":
        fields = {
            "schema", "observation_id", "schema_version", "logical_case_id",
            "replicate_id", "result_label", "observation_digest",
        }
        if not isinstance(value, Mapping) or set(value) != fields or value.get("schema") != LOGICAL_CASE_SCHEMA:
            raise ValueError("malformed physical observation")
        try:
            return cls(
                observation_id=value["observation_id"],
                logical_case_id=value["logical_case_id"],
                replicate_id=value["replicate_id"],
                result_label=value["result_label"],
                observation_digest=value["observation_digest"],
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed physical observation") from exc


@dataclass(frozen=True, slots=True)
class LogicalCaseResolution:
    logical_case_id: str
    status: LogicalCaseResolutionStatus
    observation_count: int
    replicate_ids: tuple[str, ...]
    labels: tuple[str, ...]
    conflict_rate: float

    def __post_init__(self) -> None:
        _id(self.logical_case_id, "logical case ID")
        object.__setattr__(self, "status", LogicalCaseResolutionStatus(self.status))
        if self.observation_count < 1 or self.observation_count != len(self.replicate_ids):
            raise ValueError("logical case observation count mismatch")
        if len(set(self.replicate_ids)) != len(self.replicate_ids):
            raise ValueError("logical case replicate IDs must be unique")
        if len(self.labels) < 1:
            raise ValueError("logical case requires at least one label")
        if not 0.0 <= self.conflict_rate <= 1.0:
            raise ValueError("logical case conflict rate must be in [0,1]")

    def payload(self) -> dict[str, object]:
        return {
            "logical_case_id": self.logical_case_id,
            "status": self.status.value,
            "observation_count": self.observation_count,
            "replicate_ids": list(self.replicate_ids),
            "labels": list(self.labels),
            "conflict_rate": self.conflict_rate,
        }


def resolve_logical_case(
    identity: LogicalCaseIdentity,
    observations: Iterable[PhysicalObservation],
) -> LogicalCaseResolution:
    values = tuple(observations)
    if not values:
        raise ValueError("logical case requires observations")
    if any(value.logical_case_id != identity.logical_case_id for value in values):
        return LogicalCaseResolution(identity.logical_case_id, LogicalCaseResolutionStatus.AMBIGUOUS, len(values), tuple(value.replicate_id for value in values), tuple(value.result_label for value in values), 1.0)
    labels = tuple(sorted({value.result_label for value in values}))
    conflicts = sum(value.result_label != values[0].result_label for value in values)
    conflict_rate = conflicts / len(values)
    status = LogicalCaseResolutionStatus.CONSISTENT if len(labels) == 1 else LogicalCaseResolutionStatus.AMBIGUOUS
    return LogicalCaseResolution(identity.logical_case_id, status, len(values), tuple(value.replicate_id for value in values), labels, conflict_rate)


__all__ = [
    "LOGICAL_CASE_SCHEMA",
    "LOGICAL_CASE_SCHEMA_VERSION",
    "LogicalCaseIdentity",
    "LogicalCaseResolution",
    "LogicalCaseResolutionStatus",
    "PhysicalObservation",
    "resolve_logical_case",
]
