"""Deterministic candidate attribution over native lifecycle observations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from .native_observation import (
    NativeEpisodeObservation,
    NativeLifecycleEventType,
    ObservationStatus,
)


ATTRIBUTION_SCHEMA = "rsimem-native-failure-attribution-v1"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class FailureSurface(StrEnum):
    FORMATION_MISSING = "formation_missing"
    FORMATION_INCORRECT = "formation_incorrect"
    PERSISTENCE_FAILED = "persistence_failed"
    MAINTENANCE_STALE = "maintenance_stale"
    MAINTENANCE_CONFLICT = "maintenance_conflict"
    MAINTENANCE_POLLUTION = "maintenance_pollution"
    RETRIEVAL_MISSED = "retrieval_missed"
    RETRIEVAL_WRONG = "retrieval_wrong"
    APPLICATION_IGNORED = "application_ignored"
    NON_MEMORY_FAILURE = "non_memory_failure"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class NativeAttributionExpectation:
    contract_id: str
    contract_digest: str
    task_id: str
    required_events: tuple[NativeLifecycleEventType, ...]
    source: str = "application_contract"

    def __post_init__(self) -> None:
        if self.source not in {"application_contract", "benchmark_contract"}:
            raise ValueError("native attribution expectation source is invalid")
        if not self.contract_id or not self.task_id:
            raise ValueError("native attribution expectation identity is incomplete")
        if len(self.contract_digest) != 64:
            raise ValueError("native attribution expectation digest is invalid")
        events = tuple(NativeLifecycleEventType(value) for value in self.required_events)
        if not events or len(events) != len(set(events)):
            raise ValueError("native attribution required events must be nonempty and unique")
        object.__setattr__(self, "required_events", events)


@dataclass(frozen=True, slots=True)
class NativeAttributionCandidate:
    attribution_id: str
    observation_id: str
    case_id: str
    family_id: str
    memory_kind: str | None
    primary_failure_surface: FailureSurface
    secondary_observations: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    confidence: str
    candidate_repair_axis: str | None
    is_actionable: bool
    review_status: str
    expectation_contract_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "primary_failure_surface", FailureSurface(self.primary_failure_surface)
        )
        if self.confidence not in {"low", "medium", "high"}:
            raise ValueError("native attribution confidence is invalid")
        if self.review_status != "candidate_pending_review":
            raise ValueError("deterministic attribution cannot finalize review")
        if self.is_actionable != (self.candidate_repair_axis is not None):
            raise ValueError("native attribution actionability and repair axis disagree")
        if self.primary_failure_surface is FailureSurface.UNRESOLVED and self.is_actionable:
            raise ValueError("unresolved native attribution cannot be actionable")
        if self.attribution_id != "native-attribution." + _digest(self.identity_payload())[:40]:
            raise ValueError("native attribution ID mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": ATTRIBUTION_SCHEMA,
            "observation_id": self.observation_id,
            "case_id": self.case_id,
            "family_id": self.family_id,
            "memory_kind": self.memory_kind,
            "primary_failure_surface": self.primary_failure_surface.value,
            "secondary_observations": list(self.secondary_observations),
            "evidence_refs": list(self.evidence_refs),
            "confidence": self.confidence,
            "candidate_repair_axis": self.candidate_repair_axis,
            "is_actionable": self.is_actionable,
            "review_status": self.review_status,
            "expectation_contract_id": self.expectation_contract_id,
        }

    def payload(self) -> dict[str, object]:
        return {"attribution_id": self.attribution_id, **self.identity_payload()}


_FAILURE_FOR_EVENT = {
    NativeLifecycleEventType.FORMATION: (FailureSurface.FORMATION_MISSING, "formation"),
    NativeLifecycleEventType.COMMIT: (FailureSurface.PERSISTENCE_FAILED, "persistence"),
    NativeLifecycleEventType.RETRIEVAL: (FailureSurface.RETRIEVAL_MISSED, "retrieval"),
}


def expectation_from_benchmark_contract(
    episode: Mapping[str, object],
) -> NativeAttributionExpectation | None:
    """Project only pre-registered lifecycle requirements from a PAST result.

    Scores, grader output, final text, and task outcome are deliberately not
    inspected. This object remains in the offline benchmark-audit plane.
    """

    task_id = episode.get("task_id")
    family_id = episode.get("family_id")
    if not isinstance(task_id, str) or not isinstance(family_id, str):
        raise ValueError("benchmark attribution contract identity is incomplete")
    required: list[NativeLifecycleEventType] = []
    if (
        episode.get("persistence_allowed") is True
        and episode.get("stage") in {"learn", "learn_a", "learn_b"}
    ):
        signal = episode.get("expected_persistence_signal")
        if signal in {"memory", "session", "skill"}:
            required.extend((
                NativeLifecycleEventType.FORMATION,
                NativeLifecycleEventType.COMMIT,
            ))
    if episode.get("evaluation_requires_retrieval") is True:
        required.append(NativeLifecycleEventType.RETRIEVAL)
    if not required:
        return None
    projection = {
        "task_id": task_id,
        "family_id": family_id,
        "bucket": episode.get("bucket"),
        "stage": episode.get("stage"),
        "expected_persistence_signal": episode.get("expected_persistence_signal"),
        "persistence_allowed": episode.get("persistence_allowed"),
        "evaluation_requires_retrieval": episode.get("evaluation_requires_retrieval"),
        "required_events": [value.value for value in required],
    }
    digest = _digest(projection)
    return NativeAttributionExpectation(
        contract_id="benchmark-expectation." + digest[:40],
        contract_digest=digest,
        task_id=task_id,
        required_events=tuple(required),
        source="benchmark_contract",
    )


def attribute_native_observation(
    observation: NativeEpisodeObservation,
    expectation: NativeAttributionExpectation | None,
) -> NativeAttributionCandidate:
    """Generate a conservative candidate, never a final attribution label."""

    by_type = {value.event_type: value for value in observation.events}
    secondary = tuple(
        f"{value.event_type.value}:{value.status.value}" for value in observation.events
    )
    candidates: list[tuple[FailureSurface, str, tuple[str, ...]]] = []
    unresolved_reasons: list[str] = []
    if expectation is None:
        unresolved_reasons.append("application_expectation_not_registered")
    elif expectation.task_id != observation.task_id:
        raise ValueError("native attribution expectation task does not match observation")
    else:
        for event_type in expectation.required_events:
            event = by_type[event_type]
            if event.status is ObservationStatus.OBSERVED:
                continue
            failure = _FAILURE_FOR_EVENT.get(event_type)
            if failure is None:
                unresolved_reasons.append(
                    f"required_{event_type.value}_not_observed_without_safe_mapping"
                )
                continue
            surface, axis = failure
            parent_refs = {
                NativeLifecycleEventType.FORMATION: by_type[NativeLifecycleEventType.SOURCE].evidence_refs,
                NativeLifecycleEventType.COMMIT: by_type[NativeLifecycleEventType.FORMATION].evidence_refs,
                NativeLifecycleEventType.RETRIEVAL: tuple(
                    by_type[NativeLifecycleEventType.COMMIT].output_artifact_ids
                ),
            }[event_type]
            if not parent_refs:
                unresolved_reasons.append(
                    f"required_{event_type.value}_missing_parent_evidence"
                )
                continue
            candidates.append((surface, axis, tuple(parent_refs)))
        if not candidates and not unresolved_reasons:
            unresolved_reasons.append("required_events_observed_no_failure_evidence")
    if len(candidates) == 1 and not unresolved_reasons:
        primary, axis, refs = candidates[0]
        confidence = "high"
        actionable = True
    else:
        primary = FailureSurface.UNRESOLVED
        axis = None
        refs = tuple(sorted(set(unresolved_reasons)))
        confidence = "low"
        actionable = False
    values = {
        "schema": ATTRIBUTION_SCHEMA,
        "observation_id": observation.observation_id,
        "case_id": observation.task_id,
        "family_id": observation.family_id,
        "memory_kind": observation.memory_kind.value if observation.memory_kind else None,
        "primary_failure_surface": primary.value,
        "secondary_observations": list(secondary),
        "evidence_refs": list(refs),
        "confidence": confidence,
        "candidate_repair_axis": axis,
        "is_actionable": actionable,
        "review_status": "candidate_pending_review",
        "expectation_contract_id": expectation.contract_id if expectation else None,
    }
    return NativeAttributionCandidate(
        attribution_id="native-attribution." + _digest(values)[:40],
        observation_id=observation.observation_id,
        case_id=observation.task_id,
        family_id=observation.family_id,
        memory_kind=observation.memory_kind.value if observation.memory_kind else None,
        primary_failure_surface=primary,
        secondary_observations=secondary,
        evidence_refs=refs,
        confidence=confidence,
        candidate_repair_axis=axis,
        is_actionable=actionable,
        review_status="candidate_pending_review",
        expectation_contract_id=expectation.contract_id if expectation else None,
    )


__all__ = [
    "ATTRIBUTION_SCHEMA", "FailureSurface", "NativeAttributionCandidate",
    "NativeAttributionExpectation", "attribute_native_observation",
    "expectation_from_benchmark_contract",
]
