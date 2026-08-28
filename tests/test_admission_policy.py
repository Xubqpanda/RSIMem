from __future__ import annotations

import pytest

from rsimem.memory.contracts import MemoryAccessMode, MemoryBackendDescriptor, MemoryKind, MemoryKindCapability
from rsimem.memory.admission_policy import DeterministicAdmissionPolicy
from rsimem.memory.policy_contracts import ExtractionDecision, MutationKind


def _extraction(facts: tuple[str, ...]) -> ExtractionDecision:
    return ExtractionDecision.create(
        policy_version="fixed.extraction.parent.v1",
        source_revision="snapshot.rev.1",
        input_payload={"source": "source.digest"},
        output_payload={"facts": list(facts)},
        action="RUN",
        execution_status="executed",
        reason_codes=("extracted",),
        lineage_id="lineage.fixture",
        trigger_event_id="event.fixture",
        execution_receipt_id="receipt.extraction",
        candidate_fact_ids=facts,
        source_digest="a" * 64,
    )


def _backend(*, updatable: bool = True) -> MemoryBackendDescriptor:
    return MemoryBackendDescriptor(
        "backend.semantic.fixture",
        (MemoryKindCapability(MemoryKind.SEMANTIC, MemoryAccessMode.EAGER, updatable=updatable),),
    )


def test_empty_extraction_and_filtered_candidates_are_distinct() -> None:
    policy = DeterministicAdmissionPolicy()
    empty = policy.decide(_extraction(()), backend=_backend(), backend_revision="backend.rev.1")
    duplicate = policy.decide(
        _extraction(("fact.1",)),
        backend=_backend(),
        backend_revision="backend.rev.1",
        existing_artifact_ids=("fact.1",),
    )
    assert empty.mutation_kind is MutationKind.NONE
    assert empty.reason_codes == ("empty_extraction",)
    assert duplicate.mutation_kind is MutationKind.NONE
    assert duplicate.reason_codes == ("duplicate_candidates",)
    assert empty.output_digest != duplicate.output_digest


def test_add_accepts_new_candidates_and_update_requires_capability_and_target() -> None:
    policy = DeterministicAdmissionPolicy()
    add = policy.decide(
        _extraction(("fact.1",)), backend=_backend(), backend_revision="backend.rev.1"
    )
    assert add.mutation_kind is MutationKind.ADD
    update = policy.decide(
        _extraction(("fact.1",)),
        backend=_backend(),
        backend_revision="backend.rev.2",
        update=True,
        target_artifact_ids=("artifact.1",),
    )
    assert update.mutation_kind is MutationKind.UPDATE
    with pytest.raises(ValueError, match="does not support update"):
        policy.decide(
            _extraction(("fact.1",)),
            backend=_backend(updatable=False),
            backend_revision="backend.rev.1",
            update=True,
            target_artifact_ids=("artifact.1",),
        )
    with pytest.raises(ValueError, match="requires target"):
        policy.decide(
            _extraction(("fact.1",)), backend=_backend(), backend_revision="backend.rev.1", update=True
        )


def test_add_target_and_missing_revision_fail_closed() -> None:
    policy = DeterministicAdmissionPolicy()
    with pytest.raises(ValueError, match="cannot carry target"):
        policy.decide(
            _extraction(("fact.1",)),
            backend=_backend(),
            backend_revision="backend.rev.1",
            target_artifact_ids=("artifact.1",),
        )
    with pytest.raises(ValueError, match="current backend revision"):
        policy.decide(_extraction(("fact.1",)), backend=_backend(), backend_revision="")
