from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rsimem.memory.use_attribution import (
    MemoryUseEvidence,
    MemoryUseResolutionStatus,
    OutcomeEvidenceKind,
    resolve_memory_use,
)


def _evidence(**overrides: object) -> MemoryUseEvidence:
    values: dict[str, object] = {
        "artifact_ids": ("artifact.preference.v1",),
        "retrieval_operation_id": "op.retrieve.v1",
        "retrieved_artifact_ids": ("artifact.preference.v1",),
        "injection_operation_id": "op.inject.v1",
        "injected_artifact_ids": ("artifact.preference.v1",),
        "downstream_operation_id": "op.downstream.v1",
        "used_artifact_ids": ("artifact.preference.v1",),
        "outcome_operation_id": "op.outcome.v1",
        "outcome_kind": OutcomeEvidenceKind.STATE_TRANSITION,
        "outcome_success": True,
        "observation_cutoff": "2026-08-30T01:02:03Z",
        "provenance_id": "provenance.case.v1",
    }
    values.update(overrides)
    return MemoryUseEvidence.create(**values)


def test_exact_join_resolves_attributable_use_and_replays() -> None:
    evidence = _evidence()
    resolution = resolve_memory_use(evidence)
    assert resolution.status == MemoryUseResolutionStatus.ATTRIBUTABLE_USE
    assert resolution.attributable_use is True
    assert MemoryUseEvidence.from_payload(json.loads(json.dumps(evidence.payload()))) == evidence


def test_exposure_and_behavioral_consistency_do_not_become_use() -> None:
    exposed = _evidence(downstream_operation_id=None, used_artifact_ids=())
    result = resolve_memory_use(exposed)
    assert result.status == MemoryUseResolutionStatus.EXPOSURE_ONLY
    assert result.attributable_use is False

    weak = _evidence(
        outcome_kind=OutcomeEvidenceKind.WEAK_STRING_MATCH,
        behavioral_consistency=True,
    )
    result = resolve_memory_use(weak)
    assert result.status == MemoryUseResolutionStatus.BEHAVIORAL_CONSISTENCY
    assert result.attributable_use is False


@pytest.mark.parametrize(
    "field",
    (
        "retrieval_operation_id",
        "injection_operation_id",
        "downstream_operation_id",
        "outcome_operation_id",
    ),
)
def test_missing_exact_join_stays_unresolved(field: str) -> None:
    values = {field: None}
    if field == "retrieval_operation_id":
        values["retrieved_artifact_ids"] = ()
    if field == "injection_operation_id":
        values["injected_artifact_ids"] = ()
    if field == "downstream_operation_id":
        values["used_artifact_ids"] = ()
    if field == "outcome_operation_id":
        values["outcome_kind"] = None
        values["outcome_success"] = None
    result = resolve_memory_use(_evidence(**values))
    assert result.status in {
        MemoryUseResolutionStatus.UNRESOLVED,
        MemoryUseResolutionStatus.EXPOSURE_ONLY,
        MemoryUseResolutionStatus.BEHAVIORAL_CONSISTENCY,
    }
    assert result.attributable_use is False


def test_partial_retrieval_or_injection_cannot_claim_set_use() -> None:
    result = resolve_memory_use(_evidence(
        artifact_ids=("artifact.a.v1", "artifact.b.v1"),
        retrieved_artifact_ids=("artifact.a.v1",),
        injected_artifact_ids=("artifact.a.v1",),
        used_artifact_ids=("artifact.a.v1",),
    ))
    assert result.status == MemoryUseResolutionStatus.UNRESOLVED
    assert result.reason_code == "retrieval_artifact_set_incomplete"


def test_incomplete_observation_is_censored_and_flags_are_strict() -> None:
    result = resolve_memory_use(_evidence(observation_complete=False))
    assert result.status == MemoryUseResolutionStatus.CENSORED
    with pytest.raises(TypeError, match="observation completeness"):
        _evidence(observation_complete="false")
    with pytest.raises(TypeError, match="outcome success"):
        _evidence(outcome_success="true")


def test_artifact_set_binding_is_supported_without_fact_multiplication() -> None:
    evidence = _evidence(
        artifact_ids=(),
        artifact_set_id="artifact-set.preference.v1",
        retrieved_artifact_ids=("artifact.a.v1", "artifact.b.v1"),
        injected_artifact_ids=("artifact.a.v1", "artifact.b.v1"),
        used_artifact_ids=("artifact.a.v1", "artifact.b.v1"),
    )
    assert resolve_memory_use(evidence).status == MemoryUseResolutionStatus.ATTRIBUTABLE_USE
