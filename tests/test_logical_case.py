from __future__ import annotations

import json

import pytest

from rsimem.memory.logical_case import (
    LogicalCaseIdentity,
    LogicalCaseResolutionStatus,
    PhysicalObservation,
    resolve_logical_case,
)


def _identity() -> LogicalCaseIdentity:
    return LogicalCaseIdentity.create(
        frozen_policy_digest="a" * 64,
        source_task_template_id="template.source.v1",
        source_extraction_set_id="extraction-set.source.v1",
        future_task_template_id="template.future.v1",
        observation_window="window.2026-08-30.v1",
    )


def _observation(identity: LogicalCaseIdentity, replicate: str, label: str) -> PhysicalObservation:
    return PhysicalObservation.create(
        logical_case_id=identity.logical_case_id,
        replicate_id=replicate,
        result_label=label,
        observation_digest=(replicate + label).encode().hex().ljust(64, "0")[:64],
    )


def test_replicates_share_one_logical_case_and_replay() -> None:
    identity = _identity()
    observations = (
        _observation(identity, "replicate.1", "unresolved"),
        _observation(identity, "replicate.2", "unresolved"),
        _observation(identity, "replicate.3", "unresolved"),
    )
    resolution = resolve_logical_case(identity, observations)
    assert resolution.status == LogicalCaseResolutionStatus.CONSISTENT
    assert resolution.observation_count == 3
    assert resolution.conflict_rate == 0.0
    assert PhysicalObservation.from_payload(
        json.loads(json.dumps(observations[0].payload()))
    ) == observations[0]
    payload = identity.payload()
    assert LogicalCaseIdentity.from_payload(json.loads(json.dumps(payload))) == identity


def test_conflicting_replicates_are_ambiguous_not_majority_voted() -> None:
    identity = _identity()
    result = resolve_logical_case(identity, (
        _observation(identity, "replicate.1", "useful"),
        _observation(identity, "replicate.2", "unresolved"),
        _observation(identity, "replicate.3", "useful"),
    ))
    assert result.status == LogicalCaseResolutionStatus.AMBIGUOUS
    assert result.labels == ("unresolved", "useful")
    assert result.conflict_rate > 0.0


def test_replicate_not_in_identity_and_cross_case_is_ambiguous() -> None:
    first = _identity()
    second = LogicalCaseIdentity.create(
        frozen_policy_digest="b" * 64,
        source_task_template_id="template.source.v1",
        source_extraction_set_id="extraction-set.source.v1",
        future_task_template_id="template.future.v1",
        observation_window="window.2026-08-30.v1",
    )
    assert first.logical_case_id != second.logical_case_id
    result = resolve_logical_case(first, (_observation(second, "replicate.1", "useful"),))
    assert result.status == LogicalCaseResolutionStatus.AMBIGUOUS


def test_duplicate_replicates_rejected() -> None:
    identity = _identity()
    with pytest.raises(ValueError, match="replicate IDs must be unique"):
        resolve_logical_case(identity, (
            _observation(identity, "replicate.1", "useful"),
            _observation(identity, "replicate.1", "useful"),
        ))
