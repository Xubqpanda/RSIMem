from __future__ import annotations

import json

import pytest

from rsimem.memory.evidence_planes import EvidencePlane, EvidenceSourceKind
from rsimem.memory.revocation import JsonRevocationRegistry, RevocationEntry


def _entry() -> RevocationEntry:
    return RevocationEntry.create(
        artifact_id="artifact.legacy.v1",
        artifact_schema_version=1,
        artifact_digest="a" * 64,
        evidence_plane=EvidencePlane.BENCHMARK_AUDIT,
        evidence_source=EvidenceSourceKind.BENCHMARK_CONTRACT,
        revoked_at="2026-08-30T01:02:03Z",
        reason_code="stale_schema",
    )


def test_missing_registry_fails_closed_and_initialize_is_explicit(tmp_path) -> None:
    registry = JsonRevocationRegistry(tmp_path / "revocations.jsonl")
    with pytest.raises(ValueError, match="registry is missing"):
        registry.assert_active(
            artifact_id="artifact.legacy.v1",
            artifact_schema_version=1,
            artifact_digest="a" * 64,
            evidence_plane=EvidencePlane.BENCHMARK_AUDIT,
            evidence_source=EvidenceSourceKind.BENCHMARK_CONTRACT,
        )
    registry.initialize()
    registry.assert_active(
        artifact_id="artifact.legacy.v1",
        artifact_schema_version=1,
        artifact_digest="a" * 64,
        evidence_plane=EvidencePlane.BENCHMARK_AUDIT,
        evidence_source=EvidenceSourceKind.BENCHMARK_CONTRACT,
    )


def test_revoked_artifact_is_rejected_and_append_is_idempotent(tmp_path) -> None:
    registry = JsonRevocationRegistry(tmp_path / "revocations.jsonl")
    registry.initialize()
    entry = _entry()
    assert registry.append(entry) is True
    assert registry.append(entry) is False
    with pytest.raises(ValueError, match="artifact is revoked"):
        registry.assert_active(
            artifact_id=entry.artifact_id,
            artifact_schema_version=entry.artifact_schema_version,
            artifact_digest=entry.artifact_digest,
            evidence_plane=entry.evidence_plane,
            evidence_source=entry.evidence_source,
        )
    assert RevocationEntry.from_payload(json.loads(json.dumps(entry.payload()))) == entry


def test_corrupt_or_conflicting_registry_fails_closed(tmp_path) -> None:
    path = tmp_path / "revocations.jsonl"
    registry = JsonRevocationRegistry(path)
    registry.initialize()
    entry = _entry()
    path.write_text(json.dumps(entry.payload()) + "\nnot-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed revocation registry"):
        registry.assert_active(
            artifact_id="artifact.other.v1",
            artifact_schema_version=1,
            artifact_digest="b" * 64,
            evidence_plane=EvidencePlane.BENCHMARK_AUDIT,
            evidence_source=EvidenceSourceKind.BENCHMARK_CONTRACT,
        )


def test_same_artifact_with_different_digest_is_a_registry_conflict(tmp_path) -> None:
    registry = JsonRevocationRegistry(tmp_path / "revocations.jsonl")
    registry.initialize()
    first = _entry()
    conflicting = RevocationEntry.create(
        artifact_id=first.artifact_id,
        artifact_schema_version=first.artifact_schema_version,
        artifact_digest="b" * 64,
        evidence_plane=first.evidence_plane,
        evidence_source=first.evidence_source,
        revoked_at="2026-08-30T01:02:04Z",
        reason_code="digest_mismatch",
    )
    assert registry.append(first) is True
    with pytest.raises(ValueError, match="conflicting revocation identity"):
        registry.append(conflicting)
    with pytest.raises(ValueError, match="conflicting revocation identity"):
        registry.assert_active(
            artifact_id=first.artifact_id,
            artifact_schema_version=first.artifact_schema_version,
            artifact_digest=conflicting.artifact_digest,
            evidence_plane=conflicting.evidence_plane,
            evidence_source=conflicting.evidence_source,
        )
