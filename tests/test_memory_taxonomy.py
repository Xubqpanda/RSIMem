from __future__ import annotations

import hashlib
import json

import pytest

from rsimem.memory.taxonomy import (
    MEMORY_TAXONOMY_SCHEMA,
    MemoryControlDescriptor,
    MemoryControlKind,
    MemoryKind,
    MemoryTransform,
    MemoryUnitDescriptor,
)


def _unit(**overrides: object) -> MemoryUnitDescriptor:
    values: dict[str, object] = {
        "unit_id": "unit.preference.v1",
        "kind": MemoryKind.SEMANTIC,
        "content_schema": "semantic.preference.v1",
        "scope": "user",
        "source_provenance": ("run.demo.v1", "task.learn.v1"),
        "temporal_identity": "validity.current.v1",
        "applicability": ("status_updates",),
        "version": "v1",
        "owner_method": "method.native.v1",
    }
    values.update(overrides)
    return MemoryUnitDescriptor(**values)


def test_memory_unit_round_trips_with_digest_and_primary_kind() -> None:
    descriptor = _unit()
    payload = descriptor.payload()

    assert descriptor.primary_kind is MemoryKind.SEMANTIC
    assert payload["schema"] == MEMORY_TAXONOMY_SCHEMA
    assert len(payload["descriptor_digest"]) == 64
    assert MemoryUnitDescriptor.from_payload(payload) == descriptor


def test_hybrid_unit_requires_distinct_secondary_kind_and_transform() -> None:
    descriptor = _unit(
        unit_id="unit.episode.consolidated.v1",
        kind=MemoryKind.SEMANTIC,
        secondary_kind=MemoryKind.EPISODIC,
        transform=MemoryTransform.CONSOLIDATION,
    )
    assert descriptor.primary_kind is MemoryKind.SEMANTIC
    assert descriptor.secondary_kind is MemoryKind.EPISODIC
    assert descriptor.transform is MemoryTransform.CONSOLIDATION

    with pytest.raises(ValueError, match="declared together"):
        _unit(secondary_kind=MemoryKind.EPISODIC)
    with pytest.raises(ValueError, match="declared together"):
        _unit(transform=MemoryTransform.PROJECTION)
    with pytest.raises(ValueError, match="differ"):
        _unit(secondary_kind=MemoryKind.SEMANTIC, transform=MemoryTransform.DERIVATION)


def test_memory_unit_rejects_duplicate_or_empty_identity_fields() -> None:
    with pytest.raises(ValueError, match="source provenance"):
        _unit(source_provenance=("run.demo.v1", "run.demo.v1"))
    with pytest.raises(ValueError, match="applicability"):
        _unit(applicability=())
    with pytest.raises(ValueError, match="schema"):
        _unit(schema="memory-taxonomy-v2")


def test_memory_unit_payload_digest_is_content_addressed() -> None:
    first = _unit()
    second = _unit(applicability=("status_updates", "release_notes"))
    assert first.descriptor_digest != second.descriptor_digest

    tampered = dict(first.payload())
    tampered["scope"] = "session"
    with pytest.raises(ValueError, match="digest mismatch"):
        MemoryUnitDescriptor.from_payload(tampered)


def test_control_descriptor_keeps_feedback_and_policy_state_out_of_memory_unit() -> None:
    digest = hashlib.sha256(json.dumps({"q": 0.5}).encode()).hexdigest()
    control = MemoryControlDescriptor(
        control_id="control.qvalue.v1",
        control_kind=MemoryControlKind.Q_VALUE,
        content_schema="qvalue.v1",
        version="v1",
        owner_method="method.native.v1",
        source_provenance=("run.demo.v1",),
        value_digest=digest,
    )
    assert control.control_kind is MemoryControlKind.Q_VALUE
    assert "memory_kind" not in control.payload()
    assert "content" not in control.payload()
    assert control.payload()["value_digest"] == digest
