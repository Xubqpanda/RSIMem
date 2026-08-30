from __future__ import annotations

import json

import pytest

from rsimem.memory.artifact_set import (
    ArtifactSetResolutionStatus,
    ArtifactSetSemanticBinding,
    JsonArtifactSetBindingLog,
    resolve_artifact_set,
)


def _binding(*, complete: bool = True, suffix: str = "") -> ArtifactSetSemanticBinding:
    return ArtifactSetSemanticBinding.create(
        semantic_unit_id=f"semantic.preference.rule.v1{suffix}",
        semantic_key="preference.status.concise",
        member_artifact_ids=("artifact.fact.a.v1", "artifact.fact.b.v1"),
        member_fact_ids=("fact.a.v1", "fact.b.v1"),
        complete=complete,
        source_digest="a" * 64,
        provenance_id="provenance.extraction.v1",
        matcher_version="matcher.deterministic.v1",
        equivalence_digest="f" * 64,
    )


def test_complete_set_has_one_primary_unit_and_replays() -> None:
    binding = _binding()
    assert binding.primary_unit_id.startswith("semantic-unit.")
    assert ArtifactSetSemanticBinding.from_payload(
        json.loads(json.dumps(binding.payload()))
    ) == binding
    result = resolve_artifact_set(
        binding,
        retrieved_member_artifact_ids=binding.member_artifact_ids,
        exposed_member_artifact_ids=binding.member_artifact_ids,
    )
    assert result.status == ArtifactSetResolutionStatus.COMPLETE
    assert result.primary is True


def test_artifact_set_log_is_restart_safe(tmp_path) -> None:
    path = tmp_path / "bindings.jsonl"
    binding = _binding()
    log = JsonArtifactSetBindingLog(path)
    assert log.append(binding) is True
    assert log.append(binding) is False
    assert JsonArtifactSetBindingLog(path).records() == (binding,)


def test_artifact_set_log_replay_order_is_stable_across_write_order(tmp_path) -> None:
    path = tmp_path / "bindings.jsonl"
    first = _binding(suffix=".first")
    second = _binding(suffix=".second")
    log = JsonArtifactSetBindingLog(path)
    assert log.append(second) is True
    assert log.append(first) is True
    expected = tuple(sorted((first, second), key=lambda item: item.binding_id))
    assert JsonArtifactSetBindingLog(path).records() == expected


def test_one_artifact_can_carry_multiple_fact_members() -> None:
    binding = ArtifactSetSemanticBinding.create(
        semantic_unit_id="semantic.preference.compound.v1",
        member_artifact_ids=("artifact.compound.v1",),
        member_fact_ids=("fact.a.v1", "fact.b.v1"),
        complete=True,
        source_digest="c" * 64,
        provenance_id="provenance.extraction.compound.v1",
    )
    result = resolve_artifact_set(
        binding,
        retrieved_member_artifact_ids=binding.member_artifact_ids,
        exposed_member_artifact_ids=binding.member_artifact_ids,
    )
    assert result.primary is True


def test_matcher_requires_auditable_equivalence_digest() -> None:
    with pytest.raises(ValueError, match="equivalence digest"):
        ArtifactSetSemanticBinding.create(
            semantic_unit_id="semantic.matcher.v1",
            member_artifact_ids=("artifact.a.v1",),
            member_fact_ids=("fact.a.v1",),
            complete=True,
            source_digest="d" * 64,
            provenance_id="provenance.matcher.v1",
            matcher_version="matcher.llm.v1",
        )


@pytest.mark.parametrize(
    ("retrieved", "exposed", "reason"),
    (
        (("artifact.fact.a.v1",), ("artifact.fact.a.v1",), "partial_retrieval"),
        (("artifact.fact.a.v1", "artifact.fact.b.v1"), ("artifact.fact.a.v1",), "partial_exposure"),
        (("artifact.fact.other.v1",), (), "retrieval_member_mismatch"),
    ),
)
def test_partial_or_mismatched_members_never_resolve(
    retrieved: tuple[str, ...],
    exposed: tuple[str, ...],
    reason: str,
) -> None:
    result = resolve_artifact_set(
        _binding(),
        retrieved_member_artifact_ids=retrieved,
        exposed_member_artifact_ids=exposed,
    )
    assert result.status in {
        ArtifactSetResolutionStatus.UNRESOLVED,
        ArtifactSetResolutionStatus.AMBIGUOUS,
    }
    assert result.reason_code == reason
    assert result.primary is False


def test_member_from_other_source_is_ambiguous() -> None:
    result = resolve_artifact_set(
        _binding(),
        retrieved_member_artifact_ids=("artifact.fact.a.v1", "artifact.fact.b.v1"),
        exposed_member_artifact_ids=("artifact.fact.a.v1", "artifact.fact.b.v1"),
        observed_source_digest="e" * 64,
    )
    assert result.status == ArtifactSetResolutionStatus.AMBIGUOUS
    assert result.reason_code == "member_source_mismatch"


def test_incomplete_binding_and_duplicate_members_fail_closed() -> None:
    result = resolve_artifact_set(
        _binding(complete=False),
        retrieved_member_artifact_ids=("artifact.fact.a.v1", "artifact.fact.b.v1"),
        exposed_member_artifact_ids=("artifact.fact.a.v1", "artifact.fact.b.v1"),
    )
    assert result.status == ArtifactSetResolutionStatus.UNRESOLVED
    with pytest.raises(ValueError, match="must be unique"):
        ArtifactSetSemanticBinding.create(
            semantic_unit_id="semantic.bad.v1",
            member_artifact_ids=("artifact.a.v1", "artifact.a.v1"),
            member_fact_ids=("fact.a.v1", "fact.b.v1"),
            complete=True,
            source_digest="b" * 64,
            provenance_id="provenance.bad.v1",
        )


def test_artifact_set_log_rejects_symlinked_paths(tmp_path) -> None:
    target = tmp_path / "target"
    target.write_text("", encoding="utf-8")
    path = tmp_path / "bindings.jsonl"
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        JsonArtifactSetBindingLog(path).records()


def test_artifact_set_log_rejects_symlinked_lock(tmp_path) -> None:
    path = tmp_path / "bindings.jsonl"
    lock_target = tmp_path / "lock-target"
    lock_target.write_text("", encoding="utf-8")
    path.with_name(path.name + ".lock").symlink_to(lock_target)
    with pytest.raises(ValueError, match="lock.*symlink"):
        JsonArtifactSetBindingLog(path).records()
