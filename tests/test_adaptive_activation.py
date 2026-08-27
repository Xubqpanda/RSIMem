from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rsimem.adaptive_activation import (
    MATCHED_OBSERVATION_BATCH_SCHEMA_VERSION,
    _OBSERVATION_BATCH_IDENTITY_FIELDS,
    _digest,
    activate_adaptive_policy,
    load_matched_observation_batch,
)
from rsimem.adaptive_preparation import (
    load_offline_adaptive_preparation,
    prepare_adaptive_policy,
)
from rsimem.experiment_manifest import resolved_adaptive_policy_profile
from rsimem.memory.adaptive_policy import AdaptivePolicyState
from rsimem.memory.adaptive_policy_store import JsonAdaptivePolicyStore
from test_adaptive_matched_validation import _observations
from test_adaptive_preparation import _prepared_feedback


def _observation_batch(
    path: Path,
    preparation,
    *,
    proposal_positive: bool = True,
    proposal_stability_failure: bool = False,
    official_evaluation: bool = False,
    audit_ok: bool = True,
) -> Path:
    raw_observations = _observations(
        preparation.artifact,
        preparation.split,
        proposal_positive=proposal_positive,
        proposal_stability_failure=proposal_stability_failure,
    )
    observations = []
    source_runs = []
    for observation in raw_observations:
        ordinal = 1 if observation.variant.value == "static" else 2
        source = {
            "observationId": "pending",
            "evidenceId": "pending",
            "variant": observation.variant.value,
            "runId": f"held-out-r01-{observation.variant.value}",
            "replicate": 1,
            "ordinal": ordinal,
            "status": "completed",
            "auditOk": audit_ok,
            "auditDigest": str(ordinal) * 64,
            "manifestDigest": "3" * 64,
            "modelProfileDigest": "4" * 64,
            "taskInputDigest": observation.task_input_digest,
            "budgetId": observation.budget_id,
            "evidenceCutoff": observation.evidence_cutoff,
            "derivedDatasetId": f"derived-dataset.{ordinal}",
            "derivedExampleId": f"derived-example.{ordinal}",
            "derivedEpisodeId": f"derived-episode.{ordinal}",
            "operationGraphDigest": "5" * 64,
            "resourceUsageDigest": "6" * 64,
            "feedbackContract": "sm01_tsv_v1",
        }
        evidence_identity = {
            key: source[key]
            for key in sorted(set(source) - {"observationId", "evidenceId"})
        }
        evidence_id = f"matched-evidence.{_digest(evidence_identity)[:40]}"
        observation = replace(
            observation,
            observation_id="matched-observation.pending",
            evidence_id=evidence_id,
        )
        observation_identity = observation.payload()
        observation_identity.pop("observation_id")
        observation = replace(
            observation,
            observation_id=(
                f"matched-observation.{_digest(observation_identity)[:40]}"
            ),
        )
        source["observationId"] = observation.observation_id
        source["evidenceId"] = evidence_id
        observations.append(observation)
        source_runs.append(source)
    identity = {
        "schemaVersion": MATCHED_OBSERVATION_BATCH_SCHEMA_VERSION,
        "sourceAdaptivePreparationId": preparation.manifest[
            "adaptivePreparationId"
        ],
        "benchmark": "PAST-Bench",
        "host": "Hermes",
        "familyId": "SM01_preference_adoption",
        "officialEvaluation": official_evaluation,
        "splitId": preparation.split.split_id,
        "artifactId": preparation.artifact.artifact_id,
        "policyVersions": [
            preparation.artifact.parent_policy_version,
            preparation.artifact.policy_version,
        ],
        "observations": [value.payload() for value in observations],
        "sourceRuns": source_runs,
    }
    assert tuple(identity) == _OBSERVATION_BATCH_IDENTITY_FIELDS
    payload = {
        **identity,
        "observationBatchId": f"matched-observations.{_digest(identity)[:40]}",
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _offline(tmp_path: Path):
    prepared = _prepared_feedback(tmp_path)
    offline_root = tmp_path / "offline"
    prepare_adaptive_policy(prepared, output_root=offline_root)
    return offline_root, load_offline_adaptive_preparation(offline_root)


def test_matched_observations_activate_copy_and_generate_strict_config(
    tmp_path: Path,
) -> None:
    offline_root, preparation = _offline(tmp_path)
    batch_path = _observation_batch(
        tmp_path / "matched-observations.json",
        preparation,
    )
    output = tmp_path / "active"

    report = activate_adaptive_policy(
        offline_root,
        batch_path,
        output_root=output,
    )
    replay = activate_adaptive_policy(
        offline_root,
        batch_path,
        output_root=output,
    )

    assert replay == report
    assert report["matchedDecisionAccepted"] is True
    assert report["resultingState"] == AdaptivePolicyState.ACTIVE.value
    assert report["activePolicyVersion"] == preparation.artifact.policy_version
    assert report["adaptiveConfigFile"] == "adaptive-config.json"
    source_snapshot = JsonAdaptivePolicyStore(
        offline_root / "adaptive-policy-store.json",
        trusted_root_policy_versions=(preparation.artifact.parent_policy_version,),
    ).snapshot()
    assert source_snapshot.active is None
    assert source_snapshot.records[0].state == AdaptivePolicyState.VALIDATED
    active_snapshot = JsonAdaptivePolicyStore(
        output / "adaptive-policy-store.json",
        trusted_root_policy_versions=(preparation.artifact.parent_policy_version,),
    ).snapshot()
    assert active_snapshot.active == preparation.artifact
    profile = resolved_adaptive_policy_profile(output / "adaptive-config.json")
    assert profile["activePolicyVersion"] == preparation.artifact.policy_version
    config = json.loads((output / "adaptive-config.json").read_text(encoding="utf-8"))
    assert [item["name"] for item in config["adaptive_parameters"]] == [
        "retrieval_accept_threshold"
    ]
    serialized = json.dumps(report, sort_keys=True)
    for forbidden in ('"score"', '"grader"', '"answer"', '"expectation"'):
        assert forbidden not in serialized


def test_matched_rejection_does_not_generate_deployment_config(
    tmp_path: Path,
) -> None:
    offline_root, preparation = _offline(tmp_path)
    batch_path = _observation_batch(
        tmp_path / "rejected-observations.json",
        preparation,
        proposal_positive=False,
        proposal_stability_failure=True,
    )
    output = tmp_path / "rejected"

    report = activate_adaptive_policy(
        offline_root,
        batch_path,
        output_root=output,
    )

    assert report["matchedDecisionAccepted"] is False
    assert report["resultingState"] == AdaptivePolicyState.REJECTED.value
    assert report["activePolicyVersion"] is None
    assert report["adaptiveConfigFile"] is None
    assert not (output / "adaptive-config.json").exists()


def test_matched_observation_batch_rejects_official_or_unaudited_runs(
    tmp_path: Path,
) -> None:
    _, preparation = _offline(tmp_path)
    official = _observation_batch(
        tmp_path / "official.json",
        preparation,
        official_evaluation=True,
    )
    with pytest.raises(ValueError, match="scope mismatch"):
        load_matched_observation_batch(official, preparation)

    unaudited = _observation_batch(
        tmp_path / "unaudited.json",
        preparation,
        audit_ok=False,
    )
    with pytest.raises(ValueError, match="source run mismatch"):
        load_matched_observation_batch(unaudited, preparation)

    malformed = json.loads(unaudited.read_text(encoding="utf-8"))
    for source in malformed["sourceRuns"]:
        source["auditOk"] = True
    malformed["sourceRuns"][0]["modelProfileDigest"] = []
    identity = {
        field: malformed[field]
        for field in _OBSERVATION_BATCH_IDENTITY_FIELDS
    }
    malformed["observationBatchId"] = (
        f"matched-observations.{_digest(identity)[:40]}"
    )
    unaudited.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="source digest is invalid"):
        load_matched_observation_batch(unaudited, preparation)
