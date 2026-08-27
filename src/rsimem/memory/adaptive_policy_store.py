"""Crash-safe adaptive policy artifacts and lifecycle state."""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .adaptive_policy import (
    AdaptiveFallbackReason,
    AdaptiveParameterName,
    AdaptiveParameterUpdate,
    AdaptivePolicyArtifact,
    AdaptivePolicyState,
    AdaptiveTrainingMetrics,
)


ADAPTIVE_POLICY_STORE_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _require_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")


def _strict_mapping(
    value: object,
    fields: set[str],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"malformed {name}")
    return value


@dataclass(frozen=True, slots=True)
class AdaptivePolicyLifecycleRecord:
    policy_version: str
    artifact_id: str
    artifact_digest: str
    state: AdaptivePolicyState
    revision: int
    last_transition_id: str
    reason_code: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", AdaptivePolicyState(self.state))
        for value in (
            self.policy_version,
            self.artifact_id,
            self.last_transition_id,
        ):
            _require_identifier(value, "adaptive lifecycle identity")
        if _DIGEST.fullmatch(self.artifact_digest) is None:
            raise ValueError("adaptive lifecycle artifact digest is invalid")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("adaptive lifecycle revision is invalid")
        if _REASON_CODE.fullmatch(self.reason_code) is None:
            raise ValueError("adaptive lifecycle reason must be machine-readable")
        if self.revision == 0 and (
            self.state != AdaptivePolicyState.PROPOSAL
            or self.last_transition_id != "policy-transition.registered"
            or self.reason_code != "artifact_registered"
        ):
            raise ValueError("initial adaptive lifecycle record is invalid")

    def payload(self) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "artifact_id": self.artifact_id,
            "artifact_digest": self.artifact_digest,
            "state": self.state.value,
            "revision": self.revision,
            "last_transition_id": self.last_transition_id,
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_payload(cls, value: object) -> "AdaptivePolicyLifecycleRecord":
        payload = _strict_mapping(value, {
            "policy_version",
            "artifact_id",
            "artifact_digest",
            "state",
            "revision",
            "last_transition_id",
            "reason_code",
        }, "adaptive lifecycle record")
        try:
            return cls(**payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed adaptive lifecycle record") from exc


@dataclass(frozen=True, slots=True)
class AdaptivePolicyTransition:
    transition_id: str
    policy_version: str
    from_state: AdaptivePolicyState
    to_state: AdaptivePolicyState
    reason_code: str
    resulting_revision: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "from_state", AdaptivePolicyState(self.from_state))
        object.__setattr__(self, "to_state", AdaptivePolicyState(self.to_state))
        _require_identifier(self.transition_id, "adaptive transition ID")
        _require_identifier(self.policy_version, "adaptive transition policy")
        if _REASON_CODE.fullmatch(self.reason_code) is None:
            raise ValueError("adaptive transition reason must be machine-readable")
        if type(self.resulting_revision) is not int or self.resulting_revision < 1:
            raise ValueError("adaptive transition revision is invalid")

    def request_identity(self) -> tuple[str, AdaptivePolicyState, str]:
        return (self.policy_version, self.to_state, self.reason_code)

    def payload(self) -> dict[str, object]:
        return {
            "transition_id": self.transition_id,
            "policy_version": self.policy_version,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "reason_code": self.reason_code,
            "resulting_revision": self.resulting_revision,
        }

    @classmethod
    def from_payload(cls, value: object) -> "AdaptivePolicyTransition":
        payload = _strict_mapping(value, {
            "transition_id",
            "policy_version",
            "from_state",
            "to_state",
            "reason_code",
            "resulting_revision",
        }, "adaptive policy transition")
        try:
            return cls(**payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed adaptive policy transition") from exc


@dataclass(frozen=True, slots=True)
class AdaptivePolicyStoreSnapshot:
    artifacts: tuple[AdaptivePolicyArtifact, ...]
    records: tuple[AdaptivePolicyLifecycleRecord, ...]
    transitions: tuple[AdaptivePolicyTransition, ...]
    active_policy_version: str | None

    @property
    def active(self) -> AdaptivePolicyArtifact | None:
        if self.active_policy_version is None:
            return None
        return next(
            artifact
            for artifact in self.artifacts
            if artifact.policy_version == self.active_policy_version
        )


_ALLOWED_TRANSITIONS = {
    AdaptivePolicyState.PROPOSAL: {
        AdaptivePolicyState.VALIDATED,
        AdaptivePolicyState.REJECTED,
    },
    AdaptivePolicyState.VALIDATED: {
        AdaptivePolicyState.ACTIVE,
        AdaptivePolicyState.REJECTED,
    },
    AdaptivePolicyState.ACTIVE: {
        AdaptivePolicyState.ROLLED_BACK,
    },
    AdaptivePolicyState.REJECTED: set(),
    AdaptivePolicyState.ROLLED_BACK: set(),
}


def _parse_update(value: object) -> AdaptiveParameterUpdate:
    payload = _strict_mapping(value, {
        "parameter_id",
        "name",
        "component",
        "baseline_value",
        "proposed_value",
        "delta",
        "posterior_negative_rate",
        "positive_count",
        "negative_count",
        "fallback_reason",
        "attributed_example_ids",
        "failure_subgraph_operation_ids",
    }, "adaptive parameter update")
    try:
        return AdaptiveParameterUpdate(
            parameter_id=payload["parameter_id"],
            name=AdaptiveParameterName(payload["name"]),
            component=payload["component"],
            baseline_value=payload["baseline_value"],
            proposed_value=payload["proposed_value"],
            delta=payload["delta"],
            posterior_negative_rate=payload["posterior_negative_rate"],
            positive_count=payload["positive_count"],
            negative_count=payload["negative_count"],
            fallback_reason=AdaptiveFallbackReason(payload["fallback_reason"]),
            attributed_example_ids=tuple(payload["attributed_example_ids"]),
            failure_subgraph_operation_ids=tuple(
                payload["failure_subgraph_operation_ids"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("malformed adaptive parameter update") from exc


def _parse_metrics(value: object) -> AdaptiveTrainingMetrics:
    payload = _strict_mapping(value, {
        "observation_count",
        "positive_count",
        "negative_count",
        "unresolved_count",
        "censored_count",
        "missing_propensity_count",
        "missing_propensity_rate",
    }, "adaptive training metrics")
    try:
        return AdaptiveTrainingMetrics(**payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("malformed adaptive training metrics") from exc


def _parse_artifact(value: object) -> AdaptivePolicyArtifact:
    payload = _strict_mapping(value, {
        "schema_version",
        "artifact_schema",
        "artifact_id",
        "policy_version",
        "parent_policy_version",
        "dataset_id",
        "dataset_payload_digest",
        "dataset_version",
        "feature_schema",
        "label_schema",
        "window_version",
        "training_config_digest",
        "training_seed",
        "objective",
        "regularization",
        "route_backend",
        "invocation_boundary",
        "parameters",
        "prompt_refs",
        "training_example_ids",
        "metrics",
        "provenance_example_ids",
        "provenance_operation_ids",
        "state",
        "content_digest",
    }, "adaptive policy artifact")
    try:
        return AdaptivePolicyArtifact(
            artifact_id=payload["artifact_id"],
            policy_version=payload["policy_version"],
            parent_policy_version=payload["parent_policy_version"],
            dataset_id=payload["dataset_id"],
            dataset_payload_digest=payload["dataset_payload_digest"],
            dataset_version=payload["dataset_version"],
            feature_schema=payload["feature_schema"],
            label_schema=payload["label_schema"],
            window_version=payload["window_version"],
            training_config_digest=payload["training_config_digest"],
            training_seed=payload["training_seed"],
            objective=payload["objective"],
            regularization=payload["regularization"],
            route_backend=payload["route_backend"],
            invocation_boundary=payload["invocation_boundary"],
            parameters=tuple(_parse_update(item) for item in payload["parameters"]),
            prompt_refs=tuple(payload["prompt_refs"]),
            training_example_ids=tuple(payload["training_example_ids"]),
            metrics=_parse_metrics(payload["metrics"]),
            provenance_example_ids=tuple(payload["provenance_example_ids"]),
            provenance_operation_ids=tuple(payload["provenance_operation_ids"]),
            state=AdaptivePolicyState(payload["state"]),
            content_digest=payload["content_digest"],
            artifact_schema=payload["artifact_schema"],
            schema_version=payload["schema_version"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("malformed adaptive policy artifact") from exc


class JsonAdaptivePolicyStore:
    """Persist immutable policy artifacts and one atomic active pointer."""

    def __init__(
        self,
        path: Path,
        *,
        trusted_root_policy_versions: tuple[str, ...],
    ) -> None:
        self.path = path.expanduser().resolve()
        if not trusted_root_policy_versions:
            raise ValueError("adaptive policy store requires a trusted root")
        for value in trusted_root_policy_versions:
            _require_identifier(value, "adaptive policy trusted root")
        if len(trusted_root_policy_versions) != len(set(trusted_root_policy_versions)):
            raise ValueError("adaptive policy trusted roots must be unique")
        self.trusted_roots = tuple(sorted(trusted_root_policy_versions))

    @contextmanager
    def _lock(self, operation: int):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("w", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _empty(self) -> dict[str, object]:
        return {
            "schema_version": ADAPTIVE_POLICY_STORE_SCHEMA_VERSION,
            "trusted_root_policy_versions": list(self.trusted_roots),
            "artifacts": {},
            "records": {},
            "transitions": {},
            "active_policy_version": None,
        }

    def _read_unlocked(self) -> dict[str, object]:
        if not self.path.exists():
            return self._empty()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("malformed adaptive policy store JSON") from exc
        payload = _strict_mapping(raw, {
            "schema_version",
            "trusted_root_policy_versions",
            "artifacts",
            "records",
            "transitions",
            "active_policy_version",
        }, "adaptive policy store envelope")
        if payload["schema_version"] != ADAPTIVE_POLICY_STORE_SCHEMA_VERSION:
            raise ValueError("unsupported adaptive policy store schema")
        if tuple(payload["trusted_root_policy_versions"]) != self.trusted_roots:
            raise ValueError("adaptive policy store trusted roots changed")
        if not all(isinstance(payload[name], Mapping) for name in (
            "artifacts",
            "records",
            "transitions",
        )):
            raise ValueError("malformed adaptive policy store collections")
        artifacts = {
            key: _parse_artifact(value)
            for key, value in payload["artifacts"].items()
        }
        records = {
            key: AdaptivePolicyLifecycleRecord.from_payload(value)
            for key, value in payload["records"].items()
        }
        transitions = {
            key: AdaptivePolicyTransition.from_payload(value)
            for key, value in payload["transitions"].items()
        }
        for key in (*artifacts, *records):
            _require_identifier(key, "adaptive policy store key")
        for key in transitions:
            _require_identifier(key, "adaptive transition store key")
        if set(artifacts) != set(records):
            raise ValueError("adaptive policy artifacts and records differ")
        for version, artifact in artifacts.items():
            record = records[version]
            if (
                artifact.policy_version != version
                or record.policy_version != version
                or record.artifact_id != artifact.artifact_id
                or record.artifact_digest != artifact.content_digest
            ):
                raise ValueError("adaptive policy record identity mismatch")
            parent_known = (
                artifact.parent_policy_version in self.trusted_roots
                or artifact.parent_policy_version in artifacts
            )
            if not parent_known:
                raise ValueError("adaptive policy store contains an unknown parent")
        for key, transition in transitions.items():
            if transition.transition_id != key or transition.policy_version not in records:
                raise ValueError("adaptive policy transition identity mismatch")
        for version, record in records.items():
            history = sorted(
                (
                    transition
                    for transition in transitions.values()
                    if transition.policy_version == version
                ),
                key=lambda transition: transition.resulting_revision,
            )
            if [item.resulting_revision for item in history] != list(
                range(1, record.revision + 1)
            ):
                raise ValueError("adaptive policy transition history is incomplete")
            state = AdaptivePolicyState.PROPOSAL
            for transition in history:
                if (
                    transition.from_state != state
                    or transition.to_state not in _ALLOWED_TRANSITIONS[state]
                ):
                    raise ValueError("adaptive policy transition history is invalid")
                state = transition.to_state
            if record.revision > 0 and (
                not history
                or history[-1].transition_id != record.last_transition_id
                or history[-1].reason_code != record.reason_code
                or state != record.state
            ):
                raise ValueError("adaptive policy lifecycle record differs from history")
        active_versions = tuple(
            version
            for version, record in records.items()
            if record.state == AdaptivePolicyState.ACTIVE
        )
        active = payload["active_policy_version"]
        if active is not None:
            _require_identifier(active, "adaptive active policy")
        if (
            len(active_versions) > 1
            or (not active_versions and active is not None)
            or (active_versions and active != active_versions[0])
        ):
            raise ValueError("adaptive policy store active pointer is inconsistent")
        return {
            **payload,
            "artifacts": artifacts,
            "records": records,
            "transitions": transitions,
        }

    def _write_unlocked(self, payload: dict[str, object]) -> None:
        serialized = {
            "schema_version": ADAPTIVE_POLICY_STORE_SCHEMA_VERSION,
            "trusted_root_policy_versions": list(self.trusted_roots),
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
            "active_policy_version": payload["active_policy_version"],
        }
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            dir=self.path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(serialized, handle, ensure_ascii=True, sort_keys=True)
                handle.write("\n")
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

    def snapshot(self) -> AdaptivePolicyStoreSnapshot:
        with self._lock(fcntl.LOCK_SH):
            payload = self._read_unlocked()
        return AdaptivePolicyStoreSnapshot(
            artifacts=tuple(
                payload["artifacts"][key]
                for key in sorted(payload["artifacts"])
            ),
            records=tuple(
                payload["records"][key]
                for key in sorted(payload["records"])
            ),
            transitions=tuple(
                payload["transitions"][key]
                for key in sorted(payload["transitions"])
            ),
            active_policy_version=payload["active_policy_version"],
        )

    def register(
        self,
        artifact: AdaptivePolicyArtifact,
    ) -> tuple[AdaptivePolicyLifecycleRecord, bool]:
        if artifact.state != AdaptivePolicyState.PROPOSAL:
            raise ValueError("adaptive policy registration requires a proposal")
        with self._lock(fcntl.LOCK_EX):
            payload = self._read_unlocked()
            artifacts = payload["artifacts"]
            records = payload["records"]
            existing = artifacts.get(artifact.policy_version)
            if existing is not None:
                if existing.payload() != artifact.payload():
                    raise ValueError("adaptive policy version has conflicting content")
                return records[artifact.policy_version], False
            if artifact.parent_policy_version in self.trusted_roots:
                parent_allowed = True
            else:
                parent_record = records.get(artifact.parent_policy_version)
                parent_allowed = (
                    parent_record is not None
                    and parent_record.state == AdaptivePolicyState.ACTIVE
                )
            if not parent_allowed:
                raise ValueError("adaptive policy parent is unknown or inactive")
            record = AdaptivePolicyLifecycleRecord(
                policy_version=artifact.policy_version,
                artifact_id=artifact.artifact_id,
                artifact_digest=artifact.content_digest,
                state=AdaptivePolicyState.PROPOSAL,
                revision=0,
                last_transition_id="policy-transition.registered",
                reason_code="artifact_registered",
            )
            artifacts[artifact.policy_version] = artifact
            records[artifact.policy_version] = record
            self._write_unlocked(payload)
            return record, True

    def transition(
        self,
        policy_version: str,
        *,
        to_state: AdaptivePolicyState,
        transition_id: str,
        reason_code: str,
    ) -> tuple[AdaptivePolicyLifecycleRecord, bool]:
        _require_identifier(policy_version, "adaptive transition policy")
        _require_identifier(transition_id, "adaptive transition ID")
        if _REASON_CODE.fullmatch(reason_code) is None:
            raise ValueError("adaptive transition reason must be machine-readable")
        to_state = AdaptivePolicyState(to_state)
        with self._lock(fcntl.LOCK_EX):
            payload = self._read_unlocked()
            records = payload["records"]
            transitions = payload["transitions"]
            existing_transition = transitions.get(transition_id)
            request_identity = (policy_version, to_state, reason_code)
            if existing_transition is not None:
                if existing_transition.request_identity() != request_identity:
                    raise ValueError("adaptive transition ID has conflicting content")
                return records[policy_version], False
            current = records.get(policy_version)
            if current is None:
                raise KeyError("unknown adaptive policy")
            if to_state not in _ALLOWED_TRANSITIONS[current.state]:
                raise ValueError("invalid adaptive policy state transition")
            active = payload["active_policy_version"]
            if to_state == AdaptivePolicyState.ACTIVE and active not in {
                None,
                policy_version,
            }:
                raise ValueError("another adaptive policy is already active")
            transition = AdaptivePolicyTransition(
                transition_id=transition_id,
                policy_version=policy_version,
                from_state=current.state,
                to_state=to_state,
                reason_code=reason_code,
                resulting_revision=current.revision + 1,
            )
            updated = AdaptivePolicyLifecycleRecord(
                policy_version=current.policy_version,
                artifact_id=current.artifact_id,
                artifact_digest=current.artifact_digest,
                state=to_state,
                revision=current.revision + 1,
                last_transition_id=transition_id,
                reason_code=reason_code,
            )
            records[policy_version] = updated
            transitions[transition_id] = transition
            if to_state == AdaptivePolicyState.ACTIVE:
                payload["active_policy_version"] = policy_version
            elif current.state == AdaptivePolicyState.ACTIVE:
                payload["active_policy_version"] = None
            self._write_unlocked(payload)
            return updated, True
