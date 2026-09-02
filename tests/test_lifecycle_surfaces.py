from __future__ import annotations

import pytest

from rsimem.memory import MemoryKind
from rsimem.memory.evidence_planes import EvidencePlane, EvidenceSourceKind
from rsimem.memory.lifecycle_surfaces import (
    LifecycleEvent,
    LifecycleOwnership,
    MemoryLifecycleSurface,
    MethodLifecycleDescriptor,
    classify_event_ownership,
    surface_for_policy_layer,
)
from rsimem.memory.policy_contracts import PolicyLayer


def _event(**overrides: object) -> LifecycleEvent:
    values: dict[str, object] = {
        "event_type": "memory.constructed.v1",
        "producer": "host.hermes.v1",
        "owner_method": "method.semantic.v1",
        "memory_kind": MemoryKind.SEMANTIC,
        "surface": MemoryLifecycleSurface.CONSTRUCTION,
        "input_ids": ("segment.learn.v1",),
        "output_ids": ("artifact.fact.v1",),
        "revision": "revision.1",
        "observation_cutoff": "2026-09-02T00:00:00Z",
        "evidence_plane": EvidencePlane.PURE_PROCESS,
        "evidence_source": EvidenceSourceKind.RUNTIME_OBSERVATION,
    }
    values.update(overrides)
    return LifecycleEvent.create(**values)


def test_event_identity_is_logical_and_round_trips() -> None:
    event = _event()
    payload = event.payload()
    assert event.event_id.startswith("lifecycle-event.")
    assert LifecycleEvent.from_payload(payload) == event
    assert payload["surface"] == "construction"
    assert payload["evidence_plane"] == "pure_process"


def test_event_rejects_invalid_plane_source_and_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="plane and source"):
        _event(evidence_source=EvidenceSourceKind.BENCHMARK_CONTRACT)
    with pytest.raises(ValueError, match="must not contain duplicates"):
        _event(input_ids=("segment.learn.v1", "segment.learn.v1"))
    tampered = _event().payload()
    tampered["event_id"] = "lifecycle-event.tampered"
    with pytest.raises(ValueError, match="canonical"):
        LifecycleEvent.from_payload(tampered)


def test_legacy_policy_layer_maps_extraction_only_to_construction() -> None:
    assert surface_for_policy_layer(PolicyLayer.EXTRACTION) is MemoryLifecycleSurface.CONSTRUCTION
    assert surface_for_policy_layer(PolicyLayer.ADMISSION) is MemoryLifecycleSurface.ADMISSION_MAINTENANCE
    assert surface_for_policy_layer(PolicyLayer.COMMIT) is MemoryLifecycleSurface.COMMIT_VERSIONING
    assert surface_for_policy_layer(PolicyLayer.EXPOSURE) is MemoryLifecycleSurface.RETRIEVAL_EXPOSURE


def test_method_ownership_prevents_false_credit_for_foreign_or_unowned_surface() -> None:
    method = MethodLifecycleDescriptor(
        method_id="method.semantic.v1",
        primary_kind=MemoryKind.SEMANTIC,
        owned_surfaces=(MemoryLifecycleSurface.CONSTRUCTION,),
        read_surfaces=(MemoryLifecycleSurface.RETRIEVAL_EXPOSURE,),
        observe_surfaces=(MemoryLifecycleSurface.COMMIT_VERSIONING,),
    )
    assert classify_event_ownership(_event(), method).eligible_for_method_credit

    foreign = classify_event_ownership(
        _event(owner_method="method.other.v1"),
        method,
    )
    assert foreign.ownership is LifecycleOwnership.FOREIGN_OWNER
    assert not foreign.eligible_for_method_credit

    unowned = classify_event_ownership(
        _event(surface=MemoryLifecycleSurface.ADMISSION_MAINTENANCE),
        method,
    )
    assert unowned.ownership is LifecycleOwnership.SURFACE_NOT_OWNED
    assert not unowned.eligible_for_method_credit


def test_method_kind_mismatch_cannot_receive_credit() -> None:
    method = MethodLifecycleDescriptor(
        method_id="method.semantic.v1",
        primary_kind=MemoryKind.SEMANTIC,
        owned_surfaces=(MemoryLifecycleSurface.CONSTRUCTION,),
    )
    attribution = classify_event_ownership(
        _event(memory_kind=MemoryKind.EPISODIC),
        method,
    )
    assert attribution.ownership is LifecycleOwnership.KIND_MISMATCH
    assert attribution.eligible_for_method_credit is False
