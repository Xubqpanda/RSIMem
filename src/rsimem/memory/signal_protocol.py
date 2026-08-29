"""Pre-registered, result-independent protocol for process-signal census."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Mapping


SIGNAL_PROTOCOL_SCHEMA_VERSION = 1
SIGNAL_PROTOCOL_SCHEMA = "rsimem-process-signal-protocol-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,255}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_PROVIDER_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,255}$")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _id(value: object, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")


def _ids(values: tuple[str, ...], name: str) -> None:
    if not isinstance(values, tuple) or not values or len(values) != len(set(values)):
        raise ValueError(f"{name} must be nonempty and unique")
    for value in values:
        _id(value, name)


@dataclass(frozen=True, slots=True)
class ProcessSignalAnalysisProtocol:
    protocol_id: str
    training_family_ids: tuple[str, ...]
    task_template_group_ids: tuple[str, ...]
    provider_model: str
    replicate_count: int
    observation_window: str
    case_dedup_rule: str
    no_signal_case_id: str
    schema_version: int = SIGNAL_PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SIGNAL_PROTOCOL_SCHEMA_VERSION:
            raise ValueError("unsupported process signal protocol schema")
        _id(self.protocol_id, "process signal protocol ID")
        _ids(self.training_family_ids, "training family IDs")
        _ids(self.task_template_group_ids, "task template group IDs")
        if not isinstance(self.provider_model, str) or _PROVIDER_MODEL.fullmatch(self.provider_model) is None:
            raise ValueError("provider/model identity must be stable")
        _id(self.observation_window, "observation window")
        _id(self.case_dedup_rule, "case deduplication rule")
        _id(self.no_signal_case_id, "no-signal case ID")
        if type(self.replicate_count) is not int or self.replicate_count < 1:
            raise ValueError("replicate count must be positive")
        expected = f"signal-protocol.{_digest(self._identity_payload())[:40]}"
        if self.protocol_id != expected:
            raise ValueError("process signal protocol ID mismatch")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "training_family_ids": list(self.training_family_ids),
            "task_template_group_ids": list(self.task_template_group_ids),
            "provider_model": self.provider_model,
            "replicate_count": self.replicate_count,
            "observation_window": self.observation_window,
            "case_dedup_rule": self.case_dedup_rule,
            "no_signal_case_id": self.no_signal_case_id,
        }

    @classmethod
    def create(
        cls,
        *,
        training_family_ids: tuple[str, ...],
        task_template_group_ids: tuple[str, ...],
        provider_model: str,
        replicate_count: int,
        observation_window: str,
        case_dedup_rule: str,
        no_signal_case_id: str,
    ) -> "ProcessSignalAnalysisProtocol":
        values = {
            "training_family_ids": tuple(training_family_ids),
            "task_template_group_ids": tuple(task_template_group_ids),
            "provider_model": provider_model,
            "replicate_count": replicate_count,
            "observation_window": observation_window,
            "case_dedup_rule": case_dedup_rule,
            "no_signal_case_id": no_signal_case_id,
            "schema_version": SIGNAL_PROTOCOL_SCHEMA_VERSION,
        }
        return cls(protocol_id=f"signal-protocol.{_digest(values)[:40]}", **values)

    def payload(self) -> dict[str, object]:
        return {"schema": SIGNAL_PROTOCOL_SCHEMA, "protocol_id": self.protocol_id, **self._identity_payload()}

    @classmethod
    def from_payload(cls, value: object) -> "ProcessSignalAnalysisProtocol":
        fields = {
            "schema", "protocol_id", "schema_version", "training_family_ids",
            "task_template_group_ids", "provider_model", "replicate_count",
            "observation_window", "case_dedup_rule", "no_signal_case_id",
        }
        if not isinstance(value, Mapping) or set(value) != fields or value.get("schema") != SIGNAL_PROTOCOL_SCHEMA:
            raise ValueError("malformed process signal protocol")
        if not isinstance(value["training_family_ids"], list) or not isinstance(value["task_template_group_ids"], list):
            raise ValueError("malformed process signal protocol collections")
        try:
            return cls(
                protocol_id=value["protocol_id"],
                training_family_ids=tuple(value["training_family_ids"]),
                task_template_group_ids=tuple(value["task_template_group_ids"]),
                provider_model=value["provider_model"],
                replicate_count=value["replicate_count"],
                observation_window=value["observation_window"],
                case_dedup_rule=value["case_dedup_rule"],
                no_signal_case_id=value["no_signal_case_id"],
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed process signal protocol") from exc


__all__ = [
    "SIGNAL_PROTOCOL_SCHEMA",
    "SIGNAL_PROTOCOL_SCHEMA_VERSION",
    "ProcessSignalAnalysisProtocol",
]
