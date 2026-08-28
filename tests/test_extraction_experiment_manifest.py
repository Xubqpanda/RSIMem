from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from rsimem.extraction_experiment_manifest import (
    EXTRACTION_FEEDBACK_METHOD_VARIANTS,
    EXTRACTION_METHOD_VARIANTS,
    CleanRepositoryRevision,
    extraction_execution_order,
    extraction_execution_profile,
    initialize_extraction_batch_manifest,
    load_extraction_manifest,
    next_extraction_attempt_name,
    record_extraction_attempt,
    resolve_clean_repository,
    validate_extraction_manifest,
)
from rsimem.memory.extraction_feedback import default_feedback_contract_registry
from rsimem.memory.extraction_prompt_validation import ExtractionAcceptanceCriteria
from rsimem.memory.prompt_components import (
    MatchedSemanticPolicyManifest,
    SemanticPolicyManifest,
)


def _policy(extraction_id: str, extraction_digest: str) -> SemanticPolicyManifest:
    return SemanticPolicyManifest.create(
        route="hermes-native-semantic",
        boundary="task-completed-v1",
        backend="hermes-native-semantic",
        framework_version="mem0-flat-framework-v1",
        model_profile="fixture-model-v1",
        extraction_component_id=extraction_id,
        extraction_component_digest=extraction_digest,
        update_component_id="prompt-component.update.root-v1",
        update_component_digest="1" * 64,
        retrieval_component_id="retrieval-config.root-v1",
        retrieval_component_digest="2" * 64,
    )


def _criteria() -> ExtractionAcceptanceCriteria:
    return ExtractionAcceptanceCriteria(
        minimum_matched_pairs=2,
        minimum_resolved_examples=2,
        minimum_useful_rate_delta=0.05,
        maximum_harmful_rate_delta=0.0,
        minimum_coverage_ratio=0.8,
        maximum_empty_rate=0.5,
        maximum_missed_rate_delta=0.0,
        required_metrics=("harmful_rate", "missed_rate"),
        proposal_budget_id="proposal-budget.fixture-v1",
        maximum_proposal_generations=1,
        maximum_candidate_selections=1,
    )


def _inputs(tmp_path: Path, *, phase: str = "validation") -> dict[str, object]:
    parent = _policy("prompt-component.extraction.parent", "a" * 64)
    active = (
        parent
        if phase == "feedback"
        else _policy("prompt-component.extraction.active", "b" * 64)
    )
    return {
        "path": tmp_path / f"{phase}.json",
        "registry_path": tmp_path / "batch-registry.json",
        "batch_id": f"batch.{phase}.fixture-v1",
        "phase": phase,
        "replicates": 2,
        "family_id": "SM01_preference_adoption",
        "task_template_group_id": f"sm01.{phase}.templates-v1",
        "task_manifest_digest": {"feedback": "3", "validation": "4", "final": "5"}[phase] * 64,
        "parent_policy": parent,
        "active_policy": active,
        "matched_policy": (
            None if phase == "feedback" else MatchedSemanticPolicyManifest.create(parent, active)
        ),
        "feedback_contract": default_feedback_contract_registry().resolver(
            "SM01_preference_adoption"
        ).contract,
        "acceptance_criteria": _criteria(),
        "model_profile_id": "fixture-model-v1",
        "model_profile_digest": "6" * 64,
        "request_budget_id": "past-task-budget.fixture-v1",
        "request_budget_digest": "7" * 64,
        "persistence_profile_id": "per-attempt-isolation-v1",
        "persistence_profile_digest": "8" * 64,
        "rsimem_revision": CleanRepositoryRevision("rsimem-commit", "rsimem-tree"),
        "past_bench_revision": CleanRepositoryRevision("past-commit", "past-tree"),
    }


def test_formal_method_profiles_freeze_plain_static_writeback() -> None:
    static = extraction_execution_profile("static-extraction-rsimem")
    adaptive = extraction_execution_profile("adaptive-extraction-rsimem")
    differing = {key for key in static if static[key] != adaptive[key]}

    assert differing == {"extractionArtifactRole"}
    assert static["rsimemMode"] == "native+ledger"
    assert static["semanticWritebackMode"] == "static"
    assert static["utilityGate"] == "disabled"
    assert extraction_execution_order(1) == EXTRACTION_METHOD_VARIANTS
    assert extraction_execution_order(2) == tuple(reversed(EXTRACTION_METHOD_VARIANTS))
    with pytest.raises(ValueError, match="unknown formal extraction"):
        extraction_execution_profile("adaptive-rsimem")
    with pytest.raises(ValueError, match="unknown formal extraction"):
        extraction_execution_profile("static_utility")


def test_feedback_manifest_freezes_plain_parent_and_append_only_attempts(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path, phase="feedback")
    experiment_id = initialize_extraction_batch_manifest(**inputs)
    manifest = load_extraction_manifest(inputs["path"])

    assert manifest["experimentId"] == experiment_id
    assert tuple(manifest["methods"]) == EXTRACTION_FEEDBACK_METHOD_VARIANTS
    assert manifest["split"]["role"] == "train"
    assert manifest["semanticPolicy"]["matched"] is None
    assert manifest["semanticPolicy"]["parent"] == manifest["semanticPolicy"]["active"]
    assert manifest["feedbackContract"]["contractDigest"] == (
        inputs["feedback_contract"].contract_digest
    )
    assert manifest["acceptanceCriteria"]["maximumProposalGenerations"] == 1

    method = EXTRACTION_FEEDBACK_METHOD_VARIANTS[0]
    assert next_extraction_attempt_name(
        inputs["path"], replicate=1, ordinal=1, method=method, base_run_name="r01_static"
    ) == "r01_static"
    record_extraction_attempt(
        inputs["path"], replicate=1, ordinal=1, method=method,
        run_name="r01_static", status="running",
    )
    record_extraction_attempt(
        inputs["path"], replicate=1, ordinal=1, method=method,
        run_name="r01_static", status="failed", failure_stage="provider",
    )
    assert next_extraction_attempt_name(
        inputs["path"], replicate=1, ordinal=1, method=method, base_run_name="r01_static"
    ) == "r01_static_attempt02"
    record_extraction_attempt(
        inputs["path"], replicate=1, ordinal=1, method=method,
        run_name="r01_static_attempt02", status="running",
    )
    record_extraction_attempt(
        inputs["path"], replicate=1, ordinal=1, method=method,
        run_name="r01_static_attempt02", status="completed",
    )

    history = load_extraction_manifest(inputs["path"])["attemptHistory"]
    assert [event["status"] for event in history] == [
        "running", "failed", "running", "completed",
    ]
    assert [event["attemptNumber"] for event in history] == [1, 1, 2, 2]
    assert next_extraction_attempt_name(
        inputs["path"], replicate=1, ordinal=1, method=method, base_run_name="r01_static"
    ) is None


def test_validation_manifest_records_matched_extraction_only_identity(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    initialize_extraction_batch_manifest(**inputs)
    manifest = load_extraction_manifest(inputs["path"])

    assert manifest["split"]["role"] == "validation"
    assert tuple(manifest["methods"]) == EXTRACTION_METHOD_VARIANTS
    assert manifest["semanticPolicy"]["matched"]["intervention_component"] == "extraction"
    active = manifest["semanticPolicy"]["activeArtifactByMethod"]
    assert active["static-extraction-rsimem"]["artifactDigest"] == "a" * 64
    assert active["adaptive-extraction-rsimem"]["artifactDigest"] == "b" * 64
    assert manifest["objective"] == {
        "schema": "resolved-observed-useful-rate-v1",
        "primaryUnit": "completed-source-extraction-set-future-opportunity",
        "resolvedDenominator": "useful_set_count+harmful_set_count",
    }


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda value: value["methods"].__setitem__(0, "static-rsimem"),
            "method mapping",
        ),
        (
            lambda value: value["executionProfiles"]["static-extraction-rsimem"].__setitem__(
                "semanticWritebackMode", "static_utility"
            ),
            "profile drifted",
        ),
        (
            lambda value: value["sourceProjection"].__setitem__("schemaVersion", 0),
            "stale extraction source",
        ),
        (
            lambda value: value["semanticPolicy"]["active"].__setitem__(
                "extraction_component_digest", "c" * 64
            ),
            "semantic policy composite digest",
        ),
        (
            lambda value: value["semanticPolicy"]["matched"]["candidate"].__setitem__(
                "update_component_digest", "d" * 64
            ),
            "semantic policy composite digest",
        ),
        (
            lambda value: value["modelProfile"].__setitem__("profileId", "other-model-v1"),
            "model profile disagree",
        ),
        (
            lambda value: value["feedbackContract"].__setitem__(
                "contractDigest", "e" * 64
            ),
            "feedback contract digest mismatch",
        ),
        (
            lambda value: value["acceptanceCriteria"].__setitem__(
                "maximumCandidateSelections", 2
            ),
            "criteria digest mismatch",
        ),
    ],
)
def test_manifest_rejects_stale_or_drifted_formal_identity(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    inputs = _inputs(tmp_path)
    initialize_extraction_batch_manifest(**inputs)
    value = json.loads(inputs["path"].read_text(encoding="utf-8"))
    mutate(value)
    with pytest.raises(ValueError, match=message):
        validate_extraction_manifest(value)


def test_batch_registry_rejects_reuse_and_cross_split_task_manifest(
    tmp_path: Path,
) -> None:
    feedback = _inputs(tmp_path, phase="feedback")
    initialize_extraction_batch_manifest(**feedback)
    duplicate = dict(feedback)
    duplicate["path"] = tmp_path / "duplicate.json"
    with pytest.raises(ValueError, match="batch ID cannot be reused"):
        initialize_extraction_batch_manifest(**duplicate)

    validation = _inputs(tmp_path, phase="validation")
    validation["task_manifest_digest"] = feedback["task_manifest_digest"]
    with pytest.raises(ValueError, match="cannot cross extraction splits"):
        initialize_extraction_batch_manifest(**validation)


def test_dirty_repository_and_dirty_revision_fail_before_manifest_creation(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="clean repository"):
        CleanRepositoryRevision("commit", "tree", dirty=True)

    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(("git", "init", "-q", str(repository)), check=True)
    subprocess.run(("git", "-C", str(repository), "config", "user.email", "test@example.invalid"), check=True)
    subprocess.run(("git", "-C", str(repository), "config", "user.name", "Test"), check=True)
    tracked = repository / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(repository), "add", "tracked.txt"), check=True)
    subprocess.run(("git", "-C", str(repository), "commit", "-qm", "initial"), check=True)
    clean = resolve_clean_repository(repository)
    assert clean.dirty is False

    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean repository"):
        resolve_clean_repository(repository)


def test_vendored_repository_revision_is_scoped_to_subtree(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    vendor = repository / "vendor"
    vendor.mkdir(parents=True)
    subprocess.run(("git", "init", "-q", str(repository)), check=True)
    subprocess.run(("git", "-C", str(repository), "config", "user.email", "test@example.invalid"), check=True)
    subprocess.run(("git", "-C", str(repository), "config", "user.name", "Test"), check=True)
    (vendor / "source.txt").write_text("vendor\n", encoding="utf-8")
    (repository / "root.txt").write_text("root\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(repository), "add", "."), check=True)
    subprocess.run(("git", "-C", str(repository), "commit", "-qm", "initial"), check=True)

    revision = resolve_clean_repository(vendor)
    expected_tree = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD:vendor"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert revision.tree == expected_tree

    (repository / "root.txt").write_text("root dirty\n", encoding="utf-8")
    assert resolve_clean_repository(vendor).dirty is False
    (vendor / "source.txt").write_text("vendor dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean repository"):
        resolve_clean_repository(vendor)


def test_matched_manifest_rejects_non_extraction_identity_drift() -> None:
    parent = _policy("prompt-component.extraction.parent", "a" * 64)
    candidate = _policy("prompt-component.extraction.active", "b" * 64)
    drifted = SemanticPolicyManifest.create(
        **{
            key: value
            for key, value in candidate.identity_payload().items()
            if key not in {"schema_version", "manifest_schema", "route"}
        },
        route="different-route",
    )
    with pytest.raises(ValueError, match="outside extraction"):
        MatchedSemanticPolicyManifest.create(parent, drifted)
