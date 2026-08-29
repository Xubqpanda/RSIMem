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
from rsimem.memory.artifact_set import ArtifactSetSemanticBinding


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

    completion_only = _evidence(
        downstream_operation_id=None,
        used_artifact_ids=(),
        outcome_kind=OutcomeEvidenceKind.TASK_COMPLETION,
    )
    result = resolve_memory_use(completion_only)
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


def test_retrieval_injection_and_tool_failures_are_separate_diagnostics() -> None:
    retrieval = resolve_memory_use(_evidence(retrieval_failure=True))
    assert retrieval.status == MemoryUseResolutionStatus.UNRESOLVED
    assert retrieval.reason_code == "retrieval_failure"
    injection = resolve_memory_use(_evidence(injection_failure=True))
    assert injection.status == MemoryUseResolutionStatus.UNRESOLVED
    assert injection.reason_code == "injection_failure"
    tool = resolve_memory_use(_evidence(
        outcome_kind=OutcomeEvidenceKind.TOOL_FAILURE,
        outcome_success=False,
    ))
    assert tool.status == MemoryUseResolutionStatus.UNRESOLVED
    assert tool.reason_code == "tool_failure_not_attributable"


def test_artifact_set_binding_is_supported_without_fact_multiplication() -> None:
    binding = ArtifactSetSemanticBinding.create(
        semantic_unit_id="semantic.preference.rule.v1",
        member_artifact_ids=("artifact.a.v1", "artifact.b.v1"),
        member_fact_ids=("fact.a.v1", "fact.b.v1"),
        complete=True,
        source_digest="a" * 64,
        provenance_id="provenance.extraction.v1",
    )
    evidence = _evidence(
        artifact_ids=(),
        artifact_set_id=binding.binding_id,
        retrieved_artifact_ids=binding.member_artifact_ids,
        injected_artifact_ids=binding.member_artifact_ids,
        used_artifact_ids=binding.member_artifact_ids,
    )
    assert resolve_memory_use(
        evidence, artifact_set_binding=binding
    ).status == MemoryUseResolutionStatus.ATTRIBUTABLE_USE


def test_opaque_or_mismatched_artifact_set_reference_fails_closed() -> None:
    evidence = _evidence(
        artifact_ids=(),
        artifact_set_id="artifact-set.forged.v1",
        retrieved_artifact_ids=("artifact.a.v1",),
        injected_artifact_ids=("artifact.a.v1",),
        used_artifact_ids=("artifact.a.v1",),
    )
    missing = resolve_memory_use(evidence)
    assert missing.status == MemoryUseResolutionStatus.UNRESOLVED
    assert missing.reason_code == "artifact_set_binding_missing"

    binding = ArtifactSetSemanticBinding.create(
        semantic_unit_id="semantic.preference.other.v1",
        member_artifact_ids=("artifact.a.v1",),
        member_fact_ids=("fact.a.v1",),
        complete=True,
        source_digest="b" * 64,
        provenance_id="provenance.other.v1",
    )
    mismatch = resolve_memory_use(
        evidence,
        artifact_set_binding=binding,
    )
    assert mismatch.status == MemoryUseResolutionStatus.UNRESOLVED
    assert mismatch.reason_code == "artifact_set_binding_mismatch"
