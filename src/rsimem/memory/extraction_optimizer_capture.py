"""Private, append-only inputs for reconstructing an optimizer corpus."""

from __future__ import annotations

import fcntl
import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Iterator, Mapping

from .extraction_feedback import DeploymentObservation
from .extraction_optimizer_builder import ExtractionFactContent
from .extraction_source import ExtractionSourceProjection
from .prompt_components import content_digest, text_digest


EXTRACTION_OPTIMIZER_CAPTURE_SCHEMA_VERSION = 1
EXTRACTION_OPTIMIZER_CAPTURE_SCHEMA = "extraction-optimizer-capture-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _require_id(value: object, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")


def _require_digest(value: object, name: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be sha256")


def _require_utc(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be an ISO UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO UTC timestamp") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{name} must be an ISO UTC timestamp")


def _strict(value: object, fields: set[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"malformed {name}")
    return value


def _fact_payload(value: ExtractionFactContent) -> dict[str, object]:
    return {
        "fact_id": value.fact_id,
        "content": value.content,
        "accepted": value.accepted,
        "reason_code": value.reason_code,
    }


class ExtractionOptimizerCaptureKind(StrEnum):
    SOURCE = "source"
    FEEDBACK = "feedback"


@dataclass(frozen=True, slots=True)
class ExtractionOptimizerSourceCapture:
    capture_id: str
    captured_at: str
    source_record_id: str
    source_record_digest: str
    projection: ExtractionSourceProjection
    fact_contents: tuple[ExtractionFactContent, ...]
    schema_version: int = EXTRACTION_OPTIMIZER_CAPTURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXTRACTION_OPTIMIZER_CAPTURE_SCHEMA_VERSION:
            raise ValueError("unsupported optimizer source capture schema")
        _require_id(self.capture_id, "optimizer source capture ID")
        _require_id(self.source_record_id, "optimizer source record ID")
        _require_digest(self.source_record_digest, "optimizer source record digest")
        _require_utc(self.captured_at, "optimizer source capture time")
        if not isinstance(self.projection, ExtractionSourceProjection):
            raise TypeError("optimizer source projection has the wrong type")
        if not isinstance(self.fact_contents, tuple) or any(
            not isinstance(value, ExtractionFactContent)
            for value in self.fact_contents
        ):
            raise TypeError("optimizer fact captures have the wrong type")
        fact_ids = tuple(value.fact_id for value in self.fact_contents)
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("optimizer fact capture IDs must be unique")
        expected = f"optimizer-source-capture.{content_digest(self.identity_payload())[:40]}"
        if self.capture_id != expected:
            raise ValueError("optimizer source capture ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "captured_at": self.captured_at,
            "source_record_id": self.source_record_id,
            "source_record_digest": self.source_record_digest,
            "projection": self.projection.payload(),
            "fact_contents": [
                _fact_payload(value) for value in self.fact_contents
            ],
        }

    def payload(self) -> dict[str, object]:
        return {"capture_id": self.capture_id, **self.identity_payload()}

    @classmethod
    def create(
        cls,
        *,
        captured_at: str,
        source_record_id: str,
        source_record_digest: str,
        projection: ExtractionSourceProjection,
        fact_contents: tuple[ExtractionFactContent, ...],
    ) -> "ExtractionOptimizerSourceCapture":
        identity = {
            "schema_version": EXTRACTION_OPTIMIZER_CAPTURE_SCHEMA_VERSION,
            "captured_at": captured_at,
            "source_record_id": source_record_id,
            "source_record_digest": source_record_digest,
            "projection": projection.payload(),
            "fact_contents": [_fact_payload(value) for value in fact_contents],
        }
        return cls(
            capture_id=(
                f"optimizer-source-capture.{content_digest(identity)[:40]}"
            ),
            captured_at=captured_at,
            source_record_id=source_record_id,
            source_record_digest=source_record_digest,
            projection=projection,
            fact_contents=fact_contents,
        )

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionOptimizerSourceCapture":
        payload = _strict(value, {
            "capture_id",
            "schema_version",
            "captured_at",
            "source_record_id",
            "source_record_digest",
            "projection",
            "fact_contents",
        }, "optimizer source capture")
        facts = payload["fact_contents"]
        if not isinstance(facts, list):
            raise ValueError("malformed optimizer fact captures")
        try:
            return cls(
                capture_id=payload["capture_id"],
                captured_at=payload["captured_at"],
                source_record_id=payload["source_record_id"],
                source_record_digest=payload["source_record_digest"],
                projection=ExtractionSourceProjection.from_payload(
                    payload["projection"]
                ),
                fact_contents=tuple(
                    ExtractionFactContent(
                        **_strict(item, {
                            "fact_id",
                            "content",
                            "accepted",
                            "reason_code",
                        }, "optimizer fact capture")
                    )
                    for item in facts
                ),
                schema_version=payload["schema_version"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed optimizer source capture") from exc


@dataclass(frozen=True, slots=True)
class ExtractionOptimizerFeedbackCapture:
    capture_id: str
    captured_at: str
    feedback_record_id: str
    source_record_id: str
    observation: DeploymentObservation
    current_input: str
    schema_version: int = EXTRACTION_OPTIMIZER_CAPTURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXTRACTION_OPTIMIZER_CAPTURE_SCHEMA_VERSION:
            raise ValueError("unsupported optimizer feedback capture schema")
        _require_id(self.capture_id, "optimizer feedback capture ID")
        _require_id(self.feedback_record_id, "optimizer feedback record ID")
        _require_id(self.source_record_id, "optimizer source record ID")
        _require_utc(self.captured_at, "optimizer feedback capture time")
        if not isinstance(self.observation, DeploymentObservation):
            raise TypeError("optimizer deployment observation has the wrong type")
        if not isinstance(self.current_input, str):
            raise TypeError("optimizer current input must be text")
        if text_digest(self.current_input) != (
            self.observation.current_input_projection_digest
        ):
            raise ValueError("optimizer current input digest mismatch")
        expected = f"optimizer-feedback-capture.{content_digest(self.identity_payload())[:40]}"
        if self.capture_id != expected:
            raise ValueError("optimizer feedback capture ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "captured_at": self.captured_at,
            "feedback_record_id": self.feedback_record_id,
            "source_record_id": self.source_record_id,
            "observation": self.observation.payload(),
            "current_input": self.current_input,
        }

    def payload(self) -> dict[str, object]:
        return {"capture_id": self.capture_id, **self.identity_payload()}

    @classmethod
    def create(
        cls,
        *,
        captured_at: str,
        feedback_record_id: str,
        source_record_id: str,
        observation: DeploymentObservation,
        current_input: str,
    ) -> "ExtractionOptimizerFeedbackCapture":
        identity = {
            "schema_version": EXTRACTION_OPTIMIZER_CAPTURE_SCHEMA_VERSION,
            "captured_at": captured_at,
            "feedback_record_id": feedback_record_id,
            "source_record_id": source_record_id,
            "observation": observation.payload(),
            "current_input": current_input,
        }
        return cls(
            capture_id=(
                f"optimizer-feedback-capture.{content_digest(identity)[:40]}"
            ),
            captured_at=captured_at,
            feedback_record_id=feedback_record_id,
            source_record_id=source_record_id,
            observation=observation,
            current_input=current_input,
        )

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionOptimizerFeedbackCapture":
        payload = _strict(value, {
            "capture_id",
            "schema_version",
            "captured_at",
            "feedback_record_id",
            "source_record_id",
            "observation",
            "current_input",
        }, "optimizer feedback capture")
        try:
            return cls(
                capture_id=payload["capture_id"],
                captured_at=payload["captured_at"],
                feedback_record_id=payload["feedback_record_id"],
                source_record_id=payload["source_record_id"],
                observation=DeploymentObservation.from_payload(
                    payload["observation"]
                ),
                current_input=payload["current_input"],
                schema_version=payload["schema_version"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed optimizer feedback capture") from exc


ExtractionOptimizerCapture = (
    ExtractionOptimizerSourceCapture | ExtractionOptimizerFeedbackCapture
)


class JsonExtractionOptimizerCaptureLog:
    """Persist raw optimizer-only evidence without exposing it to public logs."""

    def __init__(self, path: Path) -> None:
        self.path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        if self.path.is_symlink():
            raise ValueError("optimizer capture log cannot be a symlink")
        if self.lock_path.is_symlink():
            raise ValueError("optimizer capture lock cannot be a symlink")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.lock_path,
            os.O_CREAT | os.O_RDWR,
            0o600,
        )
        os.chmod(self.lock_path, 0o600)
        with os.fdopen(descriptor, "r+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def records(self) -> tuple[ExtractionOptimizerCapture, ...]:
        with self._locked():
            return self._read_unlocked()

    def append(self, value: ExtractionOptimizerCapture) -> bool:
        if not isinstance(value, (
            ExtractionOptimizerSourceCapture,
            ExtractionOptimizerFeedbackCapture,
        )):
            raise TypeError("optimizer capture has the wrong type")
        with self._locked():
            records = self._read_unlocked()
            logical_id = (
                value.source_record_id
                if isinstance(value, ExtractionOptimizerSourceCapture)
                else value.feedback_record_id
            )
            for existing in records:
                existing_id = (
                    existing.source_record_id
                    if isinstance(existing, ExtractionOptimizerSourceCapture)
                    else existing.feedback_record_id
                )
                if type(existing) is type(value) and existing_id == logical_id:
                    if self._equivalent(existing, value):
                        return False
                    raise ValueError("optimizer capture identity conflict")
            wrapper = self._wrapper(value)
            encoded = (
                json.dumps(
                    wrapper,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            if self.path.is_symlink():
                raise ValueError("optimizer capture log cannot be a symlink")
            descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_APPEND | os.O_WRONLY,
                0o600,
            )
            try:
                os.chmod(self.path, 0o600)
                offset = 0
                while offset < len(encoded):
                    offset += os.write(descriptor, encoded[offset:])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return True

    def _read_unlocked(self) -> tuple[ExtractionOptimizerCapture, ...]:
        if self.path.is_symlink():
            raise ValueError("optimizer capture log cannot be a symlink")
        if not self.path.exists():
            return ()
        if self.path.is_symlink():
            raise ValueError("optimizer capture log cannot be a symlink")
        if self.path.stat().st_mode & 0o077:
            raise PermissionError("optimizer capture permissions are too broad")
        records = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    raise ValueError("optimizer capture log contains an empty record")
                try:
                    records.append(self._from_wrapper(json.loads(line)))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError("malformed optimizer capture log") from exc
        by_identity: dict[tuple[type[object], str], ExtractionOptimizerCapture] = {}
        for value in records:
            logical_id = (
                value.source_record_id
                if isinstance(value, ExtractionOptimizerSourceCapture)
                else value.feedback_record_id
            )
            key = (type(value), logical_id)
            existing = by_identity.get(key)
            if existing is not None and not self._equivalent(existing, value):
                raise ValueError("optimizer capture identity conflict")
            by_identity[key] = value
        return tuple(records)

    @staticmethod
    def _equivalent(
        left: ExtractionOptimizerCapture,
        right: ExtractionOptimizerCapture,
    ) -> bool:
        if type(left) is not type(right):
            return False
        left_payload = left.identity_payload()
        right_payload = right.identity_payload()
        left_payload.pop("captured_at")
        right_payload.pop("captured_at")
        return left_payload == right_payload

    @staticmethod
    def _wrapper(value: ExtractionOptimizerCapture) -> dict[str, object]:
        kind = (
            ExtractionOptimizerCaptureKind.SOURCE
            if isinstance(value, ExtractionOptimizerSourceCapture)
            else ExtractionOptimizerCaptureKind.FEEDBACK
        )
        return {
            "schema_version": EXTRACTION_OPTIMIZER_CAPTURE_SCHEMA_VERSION,
            "capture_schema": EXTRACTION_OPTIMIZER_CAPTURE_SCHEMA,
            "capture_kind": kind.value,
            "capture": value.payload(),
        }

    @staticmethod
    def _from_wrapper(value: object) -> ExtractionOptimizerCapture:
        payload = _strict(value, {
            "schema_version",
            "capture_schema",
            "capture_kind",
            "capture",
        }, "optimizer capture wrapper")
        if (
            payload["schema_version"]
            != EXTRACTION_OPTIMIZER_CAPTURE_SCHEMA_VERSION
            or payload["capture_schema"] != EXTRACTION_OPTIMIZER_CAPTURE_SCHEMA
        ):
            raise ValueError("unsupported optimizer capture wrapper schema")
        try:
            kind = ExtractionOptimizerCaptureKind(payload["capture_kind"])
        except (TypeError, ValueError) as exc:
            raise ValueError("unknown optimizer capture kind") from exc
        if kind == ExtractionOptimizerCaptureKind.SOURCE:
            return ExtractionOptimizerSourceCapture.from_payload(payload["capture"])
        return ExtractionOptimizerFeedbackCapture.from_payload(payload["capture"])
