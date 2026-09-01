from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsimem.memory.evidence_planes import EvidencePlane, EvidenceSourceKind
from rsimem.memory.revocation import JsonRevocationRegistry, RevocationEntry, RevocationScope


ROOT = Path(__file__).resolve().parents[1]


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


def test_blank_revocation_record_fails_closed(tmp_path) -> None:
    path = tmp_path / "revocations.jsonl"
    registry = JsonRevocationRegistry(path)
    registry.initialize()
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed revocation registry"):
        registry.assert_active(
            artifact_id="artifact.other.v1",
            artifact_schema_version=1,
            artifact_digest="b" * 64,
            evidence_plane=EvidencePlane.BENCHMARK_AUDIT,
            evidence_source=EvidenceSourceKind.BENCHMARK_CONTRACT,
        )


def test_symlinked_registry_fails_closed(tmp_path) -> None:
    target = tmp_path / "target.jsonl"
    target.write_text("", encoding="utf-8")
    path = tmp_path / "revocations.jsonl"
    path.symlink_to(target)
    registry = JsonRevocationRegistry(path)
    with pytest.raises(ValueError, match="cannot be a symlink"):
        registry.assert_active(
            artifact_id="artifact.other.v1",
            artifact_schema_version=1,
            artifact_digest="b" * 64,
            evidence_plane=EvidencePlane.BENCHMARK_AUDIT,
            evidence_source=EvidenceSourceKind.BENCHMARK_CONTRACT,
        )


def test_symlinked_registry_lock_fails_closed(tmp_path) -> None:
    path = tmp_path / "revocations.jsonl"
    lock_target = tmp_path / "lock-target"
    lock_target.write_text("", encoding="utf-8")
    path.with_name(path.name + ".lock").symlink_to(lock_target)
    registry = JsonRevocationRegistry(path)
    with pytest.raises(ValueError, match="lock.*symlink"):
        registry.initialize()


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


def test_checked_in_historical_denylist_rejects_revoked_identities(tmp_path) -> None:
    path = ROOT / "configs" / "revocations.jsonl"
    allowed_fields = {
        "schema",
        "revocation_id",
        "schema_version",
        "artifact_id",
        "artifact_schema_version",
        "artifact_digest",
        "evidence_plane",
        "evidence_source",
        "revoked_at",
        "reason_code",
        "scope",
    }
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 5
    assert all(set(record) == allowed_fields for record in records)
    assert all(record["reason_code"] == "historical_attribution_revoked" for record in records)
    assert all(record["scope"] == "legacy_untyped" for record in records)
    assert all(record["evidence_plane"] is None and record["evidence_source"] is None for record in records)

    registry_path = tmp_path / "revocations.jsonl"
    registry_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    registry = JsonRevocationRegistry(registry_path)

    for record in records:
        with pytest.raises(ValueError, match="artifact is revoked"):
            registry.assert_active(
                artifact_id=record["artifact_id"],
                artifact_schema_version=record["artifact_schema_version"],
                artifact_digest=record["artifact_digest"],
                evidence_plane=EvidencePlane.PURE_PROCESS,
                evidence_source=EvidenceSourceKind.RUNTIME_OBSERVATION,
            )


def test_legacy_scope_does_not_invent_provenance_and_matches_typed_lookup(tmp_path) -> None:
    registry = JsonRevocationRegistry(tmp_path / "revocations.jsonl")
    registry.initialize()
    entry = RevocationEntry.create(
        artifact_id="artifact.legacy-untyped.v1",
        artifact_schema_version=1,
        artifact_digest="c" * 64,
        evidence_plane=None,
        evidence_source=None,
        revoked_at="2026-09-01T01:02:03Z",
        reason_code="historical_attribution_revoked",
        scope=RevocationScope.LEGACY_UNTYPED,
    )
    assert entry.evidence_plane is None
    assert entry.evidence_source is None
    assert registry.append(entry) is True
    with pytest.raises(ValueError, match="artifact is revoked"):
        registry.assert_active(
            artifact_id=entry.artifact_id,
            artifact_schema_version=entry.artifact_schema_version,
            artifact_digest=entry.artifact_digest,
            evidence_plane=EvidencePlane.PURE_PROCESS,
            evidence_source=EvidenceSourceKind.RUNTIME_OBSERVATION,
        )


def test_revocation_scope_rejects_incomplete_or_misleading_provenance() -> None:
    with pytest.raises(ValueError, match="typed revocation entries require"):
        RevocationEntry.create(
            artifact_id="artifact.typed-missing.v1",
            artifact_schema_version=1,
            artifact_digest="d" * 64,
            evidence_plane=None,
            evidence_source=None,
            revoked_at="2026-09-01T01:02:03Z",
            reason_code="stale_schema",
            scope=RevocationScope.TYPED,
        )
    with pytest.raises(ValueError, match="legacy revocation entries cannot"):
        RevocationEntry.create(
            artifact_id="artifact.legacy-claimed.v1",
            artifact_schema_version=1,
            artifact_digest="e" * 64,
            evidence_plane=EvidencePlane.PURE_PROCESS,
            evidence_source=EvidenceSourceKind.RUNTIME_OBSERVATION,
            revoked_at="2026-09-01T01:02:03Z",
            reason_code="historical_attribution_revoked",
            scope=RevocationScope.LEGACY_UNTYPED,
        )
