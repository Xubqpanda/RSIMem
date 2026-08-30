"""Pre-registered, result-independent protocol for process-signal census."""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


SIGNAL_PROTOCOL_SCHEMA_VERSION = 2
SIGNAL_PROTOCOL_SCHEMA = "rsimem-process-signal-protocol-v2"
PROCESS_SIGNAL_PROTOCOL_FILENAME = "process_signal_protocol.json"
PROCESS_SIGNAL_OBSERVATION_WINDOW = "completed-task.v1"
PROCESS_SIGNAL_CASE_DEDUP_RULE = "logical_case_v1"
PROCESS_SIGNAL_NO_SIGNAL_CASE_ID = "case.no_signal.v1"
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
    task_manifest_digest: str
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
        if not isinstance(self.task_manifest_digest, str) or _DIGEST.fullmatch(self.task_manifest_digest) is None:
            raise ValueError("task manifest digest must be a sha256 hex digest")
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
            "task_manifest_digest": self.task_manifest_digest,
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
        task_manifest_digest: str,
        provider_model: str,
        replicate_count: int,
        observation_window: str,
        case_dedup_rule: str,
        no_signal_case_id: str,
    ) -> "ProcessSignalAnalysisProtocol":
        values = {
            "training_family_ids": tuple(training_family_ids),
            "task_template_group_ids": tuple(task_template_group_ids),
            "task_manifest_digest": task_manifest_digest,
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
            "task_template_group_ids", "task_manifest_digest", "provider_model", "replicate_count",
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
                task_manifest_digest=value["task_manifest_digest"],
                provider_model=value["provider_model"],
                replicate_count=value["replicate_count"],
                observation_window=value["observation_window"],
                case_dedup_rule=value["case_dedup_rule"],
                no_signal_case_id=value["no_signal_case_id"],
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed process signal protocol") from exc


def _manifest_provider_model(manifest: Mapping[str, object]) -> str:
    """Resolve the immutable provider/model identity recorded by a manifest."""

    model = manifest.get("modelProfile")
    resolved = model.get("resolved") if isinstance(model, Mapping) else None
    base_url = resolved.get("providerBaseUrl") if isinstance(resolved, Mapping) else None
    model_id = resolved.get("modelId") if isinstance(resolved, Mapping) else None
    if (
        not isinstance(base_url, str)
        or not base_url
        or not isinstance(model_id, str)
        or not model_id
    ):
        raise ValueError("manifest provider/model identity is incomplete")
    return f"{base_url.rstrip('/')}/{model_id}"


def protocol_for_extraction_manifest(
    manifest: Mapping[str, object],
) -> ProcessSignalAnalysisProtocol:
    """Build the pre-registered protocol expected for one extraction batch.

    The protocol is deliberately derived from manifest identity and fixed
    process-census rules.  It contains no result, score, or model output.
    """

    split = manifest.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("manifest split identity is missing")
    family_id = split.get("familyId")
    template_id = split.get("taskTemplateGroupId")
    task_manifest_digest = split.get("taskManifestDigest")
    replicates = manifest.get("replicates")
    if (
        not isinstance(family_id, str)
        or not isinstance(template_id, str)
        or not isinstance(task_manifest_digest, str)
        or _DIGEST.fullmatch(task_manifest_digest) is None
        or type(replicates) is not int
        or replicates < 1
    ):
        raise ValueError("manifest process-signal identity is incomplete")
    return ProcessSignalAnalysisProtocol.create(
        training_family_ids=(family_id,),
        task_template_group_ids=(template_id,),
        task_manifest_digest=task_manifest_digest,
        provider_model=_manifest_provider_model(manifest),
        replicate_count=replicates,
        observation_window=PROCESS_SIGNAL_OBSERVATION_WINDOW,
        case_dedup_rule=PROCESS_SIGNAL_CASE_DEDUP_RULE,
        no_signal_case_id=PROCESS_SIGNAL_NO_SIGNAL_CASE_ID,
    )


def validate_protocol_for_extraction_manifest(
    protocol: ProcessSignalAnalysisProtocol,
    manifest: Mapping[str, object],
) -> ProcessSignalAnalysisProtocol:
    """Fail closed when a frozen protocol does not join its batch manifest."""

    if not isinstance(protocol, ProcessSignalAnalysisProtocol):
        raise TypeError("process signal protocol is malformed")
    expected = protocol_for_extraction_manifest(manifest)
    if protocol != expected:
        raise ValueError("process signal protocol does not match extraction manifest")
    return protocol


class JsonProcessSignalAnalysisProtocolStore:
    """Freeze one result-independent process-signal analysis protocol.

    The protocol is written before any census output is consumed.  A later
    write is allowed only when its canonical payload is identical, so a
    changed family/split/provider/window configuration cannot silently alter
    the analysis contract after results exist.
    """

    def __init__(self, path: Path) -> None:
        # Keep the final path component unresolved so a symlink cannot be
        # silently followed into a trusted-looking protocol file.
        self.path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    @staticmethod
    def _canonical(value: Mapping[str, object]) -> str:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    def _read_unlocked(self) -> ProcessSignalAnalysisProtocol | None:
        if self.path.is_symlink():
            raise ValueError("process signal protocol file cannot be a symlink")
        if not self.path.exists():
            return None
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return ProcessSignalAnalysisProtocol.from_payload(value)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError("malformed process signal protocol store") from exc

    def get(self) -> ProcessSignalAnalysisProtocol | None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            try:
                return self._read_unlocked()
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def freeze(self, protocol: ProcessSignalAnalysisProtocol) -> bool:
        """Persist a protocol once; return ``False`` for an exact replay."""

        if not isinstance(protocol, ProcessSignalAnalysisProtocol):
            raise TypeError("protocol store accepts ProcessSignalAnalysisProtocol only")
        serialized = self._canonical(protocol.payload()) + "\n"
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                existing = self._read_unlocked()
                if existing is not None:
                    if self._canonical(existing.payload()) + "\n" != serialized:
                        raise ValueError("process signal protocol is already frozen")
                    return False
                self.path.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=self.path.name + ".",
                    suffix=".tmp",
                    dir=self.path.parent,
                )
                try:
                    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                        handle.write(serialized)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary_name, self.path)
                finally:
                    if os.path.exists(temporary_name):
                        os.unlink(temporary_name)
                return True
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


__all__ = [
    "SIGNAL_PROTOCOL_SCHEMA",
    "SIGNAL_PROTOCOL_SCHEMA_VERSION",
    "PROCESS_SIGNAL_PROTOCOL_FILENAME",
    "PROCESS_SIGNAL_OBSERVATION_WINDOW",
    "PROCESS_SIGNAL_CASE_DEDUP_RULE",
    "PROCESS_SIGNAL_NO_SIGNAL_CASE_ID",
    "ProcessSignalAnalysisProtocol",
    "JsonProcessSignalAnalysisProtocolStore",
    "protocol_for_extraction_manifest",
    "validate_protocol_for_extraction_manifest",
]
