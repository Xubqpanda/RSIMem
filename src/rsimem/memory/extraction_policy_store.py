"""Crash-safe lifecycle store for extraction prompt policy artifacts."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from .extraction_policy_artifact import (
    ExtractionPromptPolicyArtifact,
    apply_extraction_rule_edits,
)
from .prompt_components import PromptSlotDescriptor, canonical_json


EXTRACTION_POLICY_STORE_SCHEMA_VERSION = 1
EXTRACTION_POLICY_STORE_SCHEMA = "extraction-prompt-policy-store-v1"


def _strict_mapping(
    value: object,
    fields: set[str],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"malformed {name}")
    return value


def _require_identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or not value[0].isalnum()
        or any(not (character.isalnum() or character in "_.:-") for character in value)
    ):
        raise ValueError(f"{name} must be a stable identifier")
    return value


class ExtractionPolicyState(StrEnum):
    ROOT = "root"
    PROPOSAL = "proposal"
    ACTIVE = "active"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True, slots=True)
class ExtractionPolicyLifecycleRecord:
    artifact_id: str
    state: ExtractionPolicyState
    revision: int
    last_transition_id: str
    reason_code: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", ExtractionPolicyState(self.state))
        _require_identifier(self.artifact_id, "extraction lifecycle artifact")
        _require_identifier(self.last_transition_id, "extraction lifecycle transition")
        _require_identifier(self.reason_code, "extraction lifecycle reason")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("extraction lifecycle revision is invalid")
        if self.state == ExtractionPolicyState.ROOT:
            if (
                self.revision != 0
                or self.last_transition_id != "extraction-transition.root"
                or self.reason_code != "trusted_root"
            ):
                raise ValueError("trusted extraction root record is invalid")
        elif self.revision == 0 and (
            self.state != ExtractionPolicyState.PROPOSAL
            or self.last_transition_id != "extraction-transition.registered"
            or self.reason_code != "artifact_registered"
        ):
            raise ValueError("initial extraction proposal record is invalid")

    def payload(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "state": self.state.value,
            "revision": self.revision,
            "last_transition_id": self.last_transition_id,
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionPolicyLifecycleRecord":
        payload = _strict_mapping(
            value,
            {
                "artifact_id",
                "state",
                "revision",
                "last_transition_id",
                "reason_code",
            },
            "extraction lifecycle record",
        )
        try:
            return cls(**payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed extraction lifecycle record") from exc


@dataclass(frozen=True, slots=True)
class ExtractionPolicyTransition:
    transition_id: str
    artifact_id: str
    from_state: ExtractionPolicyState
    to_state: ExtractionPolicyState
    reason_code: str
    resulting_revision: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "from_state", ExtractionPolicyState(self.from_state))
        object.__setattr__(self, "to_state", ExtractionPolicyState(self.to_state))
        for value, name in (
            (self.transition_id, "extraction transition ID"),
            (self.artifact_id, "extraction transition artifact"),
            (self.reason_code, "extraction transition reason"),
        ):
            _require_identifier(value, name)
        if type(self.resulting_revision) is not int or self.resulting_revision < 1:
            raise ValueError("extraction transition revision is invalid")

    def payload(self) -> dict[str, object]:
        return {
            "transition_id": self.transition_id,
            "artifact_id": self.artifact_id,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "reason_code": self.reason_code,
            "resulting_revision": self.resulting_revision,
        }

    @classmethod
    def from_payload(cls, value: object) -> "ExtractionPolicyTransition":
        payload = _strict_mapping(
            value,
            {
                "transition_id",
                "artifact_id",
                "from_state",
                "to_state",
                "reason_code",
                "resulting_revision",
            },
            "extraction policy transition",
        )
        try:
            return cls(**payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed extraction policy transition") from exc


@dataclass(frozen=True, slots=True)
class ExtractionPolicyStoreSnapshot:
    artifacts: tuple[ExtractionPromptPolicyArtifact, ...]
    records: tuple[ExtractionPolicyLifecycleRecord, ...]
    transitions: tuple[ExtractionPolicyTransition, ...]
    root_artifact_id: str
    active_artifact_id: str | None

    @property
    def root(self) -> ExtractionPromptPolicyArtifact:
        return next(
            artifact
            for artifact in self.artifacts
            if artifact.artifact_id == self.root_artifact_id
        )

    @property
    def active(self) -> ExtractionPromptPolicyArtifact | None:
        if self.active_artifact_id is None:
            return None
        return next(
            artifact
            for artifact in self.artifacts
            if artifact.artifact_id == self.active_artifact_id
        )


_ALLOWED_TRANSITIONS = {
    ExtractionPolicyState.PROPOSAL: {
        ExtractionPolicyState.ACTIVE,
        ExtractionPolicyState.REJECTED,
    },
    ExtractionPolicyState.ACTIVE: {ExtractionPolicyState.ROLLED_BACK},
    ExtractionPolicyState.REJECTED: set(),
    ExtractionPolicyState.ROLLED_BACK: set(),
    ExtractionPolicyState.ROOT: set(),
}


class JsonExtractionPolicyStore:
    """Persist one trusted root, immutable children, and one active pointer."""

    def __init__(
        self,
        path: Path,
        *,
        trusted_root: ExtractionPromptPolicyArtifact,
        slot: PromptSlotDescriptor,
    ) -> None:
        if trusted_root.parent_artifact_id is not None:
            raise ValueError("extraction policy store root must be a root artifact")
        trusted_root.to_prompt_component(slot)
        # Preserve the final component so ACTIVE policy state cannot be
        # redirected by a symlink into an unrelated store.
        self.path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
        self.trusted_root = trusted_root
        self.slot = slot

    @contextmanager
    def _lock(self, operation: int):
        if self.path.is_symlink():
            raise ValueError("extraction policy store cannot be a symlink")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        if lock_path.is_symlink():
            raise ValueError("extraction policy store lock cannot be a symlink")
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _empty(self) -> dict[str, object]:
        root_record = ExtractionPolicyLifecycleRecord(
            self.trusted_root.artifact_id,
            ExtractionPolicyState.ROOT,
            0,
            "extraction-transition.root",
            "trusted_root",
        )
        return {
            "schema_version": EXTRACTION_POLICY_STORE_SCHEMA_VERSION,
            "store_schema": EXTRACTION_POLICY_STORE_SCHEMA,
            "slot_id": self.slot.slot_id,
            "slot_contract_digest": self.slot.contract_digest,
            "frozen_wrapper_digest": self.slot.frozen_wrapper_digest,
            "root_artifact_id": self.trusted_root.artifact_id,
            "active_artifact_id": None,
            "artifacts": {self.trusted_root.artifact_id: self.trusted_root},
            "records": {self.trusted_root.artifact_id: root_record},
            "transitions": {},
        }

    def _read_unlocked(self) -> dict[str, object]:
        if self.path.is_symlink():
            raise ValueError("extraction policy store cannot be a symlink")
        if not self.path.exists():
            return self._empty()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("malformed extraction policy store JSON") from exc
        payload = _strict_mapping(
            raw,
            {
                "schema_version",
                "store_schema",
                "slot_id",
                "slot_contract_digest",
                "frozen_wrapper_digest",
                "root_artifact_id",
                "active_artifact_id",
                "artifacts",
                "records",
                "transitions",
            },
            "extraction policy store envelope",
        )
        if (
            payload["schema_version"] != EXTRACTION_POLICY_STORE_SCHEMA_VERSION
            or payload["store_schema"] != EXTRACTION_POLICY_STORE_SCHEMA
        ):
            raise ValueError("unsupported extraction policy store schema")
        if (
            payload["slot_id"] != self.slot.slot_id
            or payload["slot_contract_digest"] != self.slot.contract_digest
            or payload["frozen_wrapper_digest"] != self.slot.frozen_wrapper_digest
            or payload["root_artifact_id"] != self.trusted_root.artifact_id
        ):
            raise ValueError("extraction policy store runtime contract drifted")
        if not all(
            isinstance(payload[field], Mapping)
            for field in ("artifacts", "records", "transitions")
        ):
            raise ValueError("malformed extraction policy store collections")
        artifacts = {
            key: ExtractionPromptPolicyArtifact.from_payload(value)
            for key, value in payload["artifacts"].items()
        }
        records = {
            key: ExtractionPolicyLifecycleRecord.from_payload(value)
            for key, value in payload["records"].items()
        }
        transitions = {
            key: ExtractionPolicyTransition.from_payload(value)
            for key, value in payload["transitions"].items()
        }
        if set(artifacts) != set(records) or self.trusted_root.artifact_id not in artifacts:
            raise ValueError("extraction policy artifacts and records differ")
        if artifacts[self.trusted_root.artifact_id].payload() != self.trusted_root.payload():
            raise ValueError("extraction policy trusted root was modified")
        for key, artifact in artifacts.items():
            if key != artifact.artifact_id or records[key].artifact_id != key:
                raise ValueError("extraction policy store artifact identity mismatch")
            artifact.to_prompt_component(self.slot)
        self._validate_lineage(artifacts)
        self._validate_lifecycle(records, transitions)
        active_versions = tuple(
            artifact_id
            for artifact_id, record in records.items()
            if record.state == ExtractionPolicyState.ACTIVE
        )
        active = payload["active_artifact_id"]
        if active is not None:
            _require_identifier(active, "active extraction artifact")
        if (
            len(active_versions) > 1
            or (not active_versions and active is not None)
            or (active_versions and active != active_versions[0])
        ):
            raise ValueError("extraction policy store active pointer is inconsistent")
        return {
            **payload,
            "artifacts": artifacts,
            "records": records,
            "transitions": transitions,
        }

    def _validate_lineage(
        self,
        artifacts: Mapping[str, ExtractionPromptPolicyArtifact],
    ) -> None:
        root_id = self.trusted_root.artifact_id
        for artifact_id, artifact in artifacts.items():
            if artifact_id == root_id:
                continue
            parent_id = artifact.parent_artifact_id
            parent = artifacts.get(parent_id or "")
            if parent is None:
                raise ValueError("extraction policy store contains an unknown parent")
            if artifact.parent_spec_digest != parent.spec.spec_digest:
                raise ValueError("extraction policy parent spec digest differs")
            replayed = apply_extraction_rule_edits(parent.spec, artifact.edits)
            if replayed != artifact.spec:
                raise ValueError("extraction policy child does not replay from its parent")
        for artifact_id in artifacts:
            visited: set[str] = set()
            current = artifact_id
            while current != root_id:
                if current in visited:
                    raise ValueError("extraction policy lineage contains a cycle")
                visited.add(current)
                parent = artifacts[current].parent_artifact_id
                if parent is None or parent not in artifacts:
                    raise ValueError("extraction policy lineage does not reach its root")
                current = parent

    @staticmethod
    def _validate_lifecycle(
        records: Mapping[str, ExtractionPolicyLifecycleRecord],
        transitions: Mapping[str, ExtractionPolicyTransition],
    ) -> None:
        for key, transition in transitions.items():
            if transition.transition_id != key or transition.artifact_id not in records:
                raise ValueError("extraction policy transition identity mismatch")
        for artifact_id, record in records.items():
            if record.state == ExtractionPolicyState.ROOT:
                continue
            history = sorted(
                (
                    transition
                    for transition in transitions.values()
                    if transition.artifact_id == artifact_id
                ),
                key=lambda transition: transition.resulting_revision,
            )
            if [item.resulting_revision for item in history] != list(
                range(1, record.revision + 1)
            ):
                raise ValueError("extraction policy transition history is incomplete")
            state = ExtractionPolicyState.PROPOSAL
            for transition in history:
                if (
                    transition.from_state != state
                    or transition.to_state not in _ALLOWED_TRANSITIONS[state]
                ):
                    raise ValueError("extraction policy transition history is invalid")
                state = transition.to_state
            if record.revision > 0 and (
                not history
                or history[-1].transition_id != record.last_transition_id
                or history[-1].reason_code != record.reason_code
                or state != record.state
            ):
                raise ValueError("extraction lifecycle record differs from history")

    def _write_unlocked(self, payload: Mapping[str, object]) -> None:
        if self.path.is_symlink():
            raise ValueError("extraction policy store cannot be a symlink")
        serialized = {
            "schema_version": EXTRACTION_POLICY_STORE_SCHEMA_VERSION,
            "store_schema": EXTRACTION_POLICY_STORE_SCHEMA,
            "slot_id": self.slot.slot_id,
            "slot_contract_digest": self.slot.contract_digest,
            "frozen_wrapper_digest": self.slot.frozen_wrapper_digest,
            "root_artifact_id": self.trusted_root.artifact_id,
            "active_artifact_id": payload["active_artifact_id"],
            "artifacts": {
                key: value.payload()
                for key, value in sorted(payload["artifacts"].items())
            },
            "records": {
                key: value.payload()
                for key, value in sorted(payload["records"].items())
            },
            "transitions": {
                key: value.payload()
                for key, value in sorted(payload["transitions"].items())
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            dir=self.path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(canonical_json(serialized) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def initialize(self) -> ExtractionPolicyStoreSnapshot:
        with self._lock(fcntl.LOCK_EX):
            payload = self._read_unlocked()
            if not self.path.exists():
                self._write_unlocked(payload)
        return self._snapshot_from_payload(payload)

    def snapshot(self) -> ExtractionPolicyStoreSnapshot:
        with self._lock(fcntl.LOCK_SH):
            payload = self._read_unlocked()
        return self._snapshot_from_payload(payload)

    @staticmethod
    def _snapshot_from_payload(payload: Mapping[str, object]) -> ExtractionPolicyStoreSnapshot:
        return ExtractionPolicyStoreSnapshot(
            artifacts=tuple(
                payload["artifacts"][key] for key in sorted(payload["artifacts"])
            ),
            records=tuple(
                payload["records"][key] for key in sorted(payload["records"])
            ),
            transitions=tuple(
                payload["transitions"][key] for key in sorted(payload["transitions"])
            ),
            root_artifact_id=payload["root_artifact_id"],
            active_artifact_id=payload["active_artifact_id"],
        )

    def register(
        self,
        artifact: ExtractionPromptPolicyArtifact,
    ) -> tuple[ExtractionPolicyLifecycleRecord, bool]:
        if artifact.parent_artifact_id is None:
            raise ValueError("extraction policy registration requires a child artifact")
        with self._lock(fcntl.LOCK_EX):
            payload = self._read_unlocked()
            artifacts = payload["artifacts"]
            records = payload["records"]
            existing = artifacts.get(artifact.artifact_id)
            if existing is not None:
                if existing.payload() != artifact.payload():
                    raise ValueError("extraction artifact ID has conflicting content")
                return records[artifact.artifact_id], False
            parent_record = records.get(artifact.parent_artifact_id)
            if parent_record is None or parent_record.state not in {
                ExtractionPolicyState.ROOT,
                ExtractionPolicyState.ACTIVE,
            }:
                raise ValueError("extraction policy parent is unknown or inactive")
            artifact.to_prompt_component(self.slot)
            parent = artifacts[artifact.parent_artifact_id]
            if artifact.parent_spec_digest != parent.spec.spec_digest or (
                apply_extraction_rule_edits(parent.spec, artifact.edits) != artifact.spec
            ):
                raise ValueError("extraction policy child does not replay from its parent")
            record = ExtractionPolicyLifecycleRecord(
                artifact.artifact_id,
                ExtractionPolicyState.PROPOSAL,
                0,
                "extraction-transition.registered",
                "artifact_registered",
            )
            artifacts[artifact.artifact_id] = artifact
            records[artifact.artifact_id] = record
            self._write_unlocked(payload)
            return record, True

    def transition(
        self,
        artifact_id: str,
        *,
        to_state: ExtractionPolicyState,
        transition_id: str,
        reason_code: str,
    ) -> tuple[ExtractionPolicyLifecycleRecord, bool]:
        _require_identifier(artifact_id, "extraction transition artifact")
        _require_identifier(transition_id, "extraction transition ID")
        _require_identifier(reason_code, "extraction transition reason")
        to_state = ExtractionPolicyState(to_state)
        with self._lock(fcntl.LOCK_EX):
            payload = self._read_unlocked()
            records = payload["records"]
            transitions = payload["transitions"]
            existing = transitions.get(transition_id)
            if existing is not None:
                if (
                    existing.artifact_id != artifact_id
                    or existing.to_state != to_state
                    or existing.reason_code != reason_code
                ):
                    raise ValueError("extraction transition ID has conflicting content")
                return records[artifact_id], False
            current = records.get(artifact_id)
            if current is None or current.state == ExtractionPolicyState.ROOT:
                raise KeyError("unknown mutable extraction policy")
            if to_state not in _ALLOWED_TRANSITIONS[current.state]:
                raise ValueError("invalid extraction policy state transition")
            active = payload["active_artifact_id"]
            if to_state == ExtractionPolicyState.ACTIVE and active not in {
                None,
                artifact_id,
            }:
                raise ValueError("another extraction policy is already active")
            transition = ExtractionPolicyTransition(
                transition_id,
                artifact_id,
                current.state,
                to_state,
                reason_code,
                current.revision + 1,
            )
            updated = ExtractionPolicyLifecycleRecord(
                artifact_id,
                to_state,
                current.revision + 1,
                transition_id,
                reason_code,
            )
            transitions[transition_id] = transition
            records[artifact_id] = updated
            if to_state == ExtractionPolicyState.ACTIVE:
                payload["active_artifact_id"] = artifact_id
            elif current.state == ExtractionPolicyState.ACTIVE:
                payload["active_artifact_id"] = None
            self._write_unlocked(payload)
            return updated, True

    def active_or_root(self) -> ExtractionPromptPolicyArtifact:
        try:
            snapshot = self.snapshot()
        except (OSError, TypeError, ValueError):
            return self.trusted_root
        return snapshot.active or snapshot.root
