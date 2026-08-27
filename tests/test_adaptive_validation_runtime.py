from __future__ import annotations

import json

import pytest

from rsimem.adaptive_preparation import (
    load_offline_adaptive_preparation,
    prepare_adaptive_policy,
)
from rsimem.adaptive_validation_runtime import (
    prepare_matched_validation_runtime,
    resolved_matched_validation_trial_profile,
)
from rsimem.experiment_manifest import (
    ADAPTIVE_VALIDATION_METHOD_VARIANTS,
    initialize_batch_manifest,
    load_manifest,
    resolved_adaptive_policy_profile,
)
from rsimem.memory.adaptive_policy import AdaptivePolicyState
from rsimem.memory.adaptive_policy_store import JsonAdaptivePolicyStore
from test_adaptive_preparation import _prepared_feedback
from test_experiment_manifest import _manifest_kwargs


def test_validation_trial_activates_only_isolated_copy_and_binds_manifest(
    tmp_path,
) -> None:
    feedback = _prepared_feedback(tmp_path)
    offline_root = tmp_path / "offline"
    prepare_adaptive_policy(feedback, output_root=offline_root)
    preparation = load_offline_adaptive_preparation(offline_root)
    trial_root = tmp_path / "trial"

    report = prepare_matched_validation_runtime(
        offline_root,
        output_root=trial_root,
    )
    replay = prepare_matched_validation_runtime(
        offline_root,
        output_root=trial_root,
    )

    assert replay == report
    assert report["officialEvaluation"] is False
    assert report["deploymentScope"] == "matched_validation_only"
    source = JsonAdaptivePolicyStore(
        offline_root / "adaptive-policy-store.json",
        trusted_root_policy_versions=(preparation.artifact.parent_policy_version,),
    ).snapshot()
    assert source.active is None
    assert source.records[0].state == AdaptivePolicyState.VALIDATED
    trial = JsonAdaptivePolicyStore(
        trial_root / "adaptive-policy-store.json",
        trusted_root_policy_versions=(preparation.artifact.parent_policy_version,),
    ).snapshot()
    assert trial.active == preparation.artifact
    profile = resolved_matched_validation_trial_profile(
        trial_root / "adaptive-config.json"
    )
    assert profile["activePolicyVersion"] == preparation.artifact.policy_version
    assert profile["preparation"] == "matched_validation_trial_store"
    with pytest.raises(ValueError, match="activation manifest cannot be read"):
        resolved_adaptive_policy_profile(trial_root / "adaptive-config.json")

    manifest_path = tmp_path / "batch.json"
    initialize_batch_manifest(
        manifest_path,
        **_manifest_kwargs(),
        execution_modes=ADAPTIVE_VALIDATION_METHOD_VARIANTS,
        adaptive_policy=profile,
        semantic_feedback_contract="sm01_tsv_v1",
    )
    assert load_manifest(manifest_path)["configuration"]["adaptivePolicy"] == profile


def test_validation_trial_manifest_tampering_fails_closed(tmp_path) -> None:
    feedback = _prepared_feedback(tmp_path)
    offline_root = tmp_path / "offline"
    prepare_adaptive_policy(feedback, output_root=offline_root)
    trial_root = tmp_path / "trial"
    prepare_matched_validation_runtime(offline_root, output_root=trial_root)
    manifest_path = trial_root / "matched-validation-trial-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["officialEvaluation"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="trial identity mismatch"):
        resolved_matched_validation_trial_profile(
            trial_root / "adaptive-config.json"
        )
