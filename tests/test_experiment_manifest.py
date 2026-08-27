from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsimem.adaptive_activation import activate_adaptive_policy
from rsimem.memory.adaptive_policy_store import (
    ADAPTIVE_POLICY_STORE_SCHEMA_VERSION,
)
from rsimem.experiment_manifest import (
    ADAPTIVE_METHOD_VARIANTS,
    ADAPTIVE_VALIDATION_METHOD_VARIANTS,
    FEEDBACK_METHOD_VARIANTS,
    STATIC_METHOD_VARIANTS,
    STATIC_UTILITY_METHOD_VARIANTS,
    adaptive_method_execution_profile,
    adaptive_validation_method_execution_profile,
    execution_order,
    initialize_batch_manifest,
    load_manifest,
    next_attempt_name,
    record_attempt,
    resolved_adaptive_policy_profile,
    resolved_environment_profile,
    resolved_family_budget,
    resolved_model_profile,
    resolved_run_profile,
    validate_manifest,
)
from test_adaptive_activation import _observation_batch, _offline


def _manifest_kwargs() -> dict[str, object]:
    return {
        "replicates": 2,
        "task_family": "memory_ability/SM01_preference_adoption",
        "agent": "hermes-luna",
        "runtime": "local",
        "model": {
            "profile": "hermes-luna/default_model",
            "modelId": "gpt-test",
            "providerBaseUrl": "https://provider.invalid/v1",
            "temperature": 0.0,
        },
        "judge": {"enabled": False, "profile": "disabled", "modelId": None},
        "budget": {
            "source": "task_manifest",
            "taskTimeoutOverrideSeconds": None,
            "tasks": [{
                "episode": "episode-1",
                "taskId": "task-1",
                "maxTurns": 20,
                "timeoutSeconds": 300,
            }],
            "taskManifestDigest": "task-digest",
        },
        "persistence_isolation": {
            "strategy": "per_attempt_trace_directory",
            "compareNoPersistence": True,
        },
        "adapter_projection_verification": True,
        "environment": {
            "pythonVersion": "3.11.0",
            "distributions": {
                "rsimem": "0.1.0",
                "past-bench": "1.0.0",
                "hermes-agent": "0.4.0",
            },
        },
        "rsimem_commit": "rsimem-head",
        "rsimem_working_tree_dirty": False,
        "past_bench_commit": "past-last-change",
        "past_bench_tree": "past-tree",
        "past_bench_dirty": False,
    }


def _adaptive_profile() -> dict[str, object]:
    return {
        "configSchemaVersion": 1,
        "configDigest": "a" * 64,
        "storeSchemaVersion": 1,
        "storeDigest": "b" * 64,
        "activePolicyVersion": "adaptive.policy-v2",
        "activeArtifactId": "adaptive-policy.artifact-v2",
        "activeArtifactDigest": "c" * 64,
        "preparation": "external_audited_active_store",
    }


def test_execution_order_rotates_modes_across_replicates() -> None:
    assert execution_order(1) == (
        "native",
        "native+ledger",
        "native+adapter+ledger",
    )
    assert execution_order(2) == (
        "native+ledger",
        "native+adapter+ledger",
        "native",
    )
    assert execution_order(3) == (
        "native+adapter+ledger",
        "native",
        "native+ledger",
    )
    assert execution_order(4) == execution_order(1)
    assert execution_order(1, STATIC_METHOD_VARIANTS) == (
        "no-persistence",
        "native-hermes",
        "static-rsimem",
    )
    assert execution_order(2, STATIC_METHOD_VARIANTS) == (
        "native-hermes",
        "static-rsimem",
        "no-persistence",
    )
    assert execution_order(1, STATIC_UTILITY_METHOD_VARIANTS) == (
        "static-rsimem",
        "static-utility-rsimem",
    )
    assert execution_order(2, STATIC_UTILITY_METHOD_VARIANTS) == (
        "static-utility-rsimem",
        "static-rsimem",
    )
    assert execution_order(3, FEEDBACK_METHOD_VARIANTS) == (
        "static-utility-rsimem",
    )


def test_adaptive_method_execution_profiles_freeze_only_intended_differences() -> None:
    profiles = {
        method: adaptive_method_execution_profile(method)
        for method in ADAPTIVE_METHOD_VARIANTS
    }
    assert profiles["no-persistence"] == {
        "persistenceVariant": "without_persistence",
        "rsimemMode": "native",
        "lifecycleEvaluatorMode": "disabled",
        "semanticWritebackMode": "disabled",
        "semanticFeedbackContract": "disabled",
        "adaptiveConfigRequired": False,
    }
    assert profiles["native-hermes"]["rsimemMode"] == "native"
    assert profiles["native-ledger"]["rsimemMode"] == "native+ledger"
    assert profiles["static-rsimem"]["semanticWritebackMode"] == (
        "static_utility"
    )
    assert profiles["static-rsimem"]["semanticFeedbackContract"] == (
        "sm01_tsv_v1"
    )
    assert profiles["adaptive-rsimem"]["semanticWritebackMode"] == (
        "adaptive_utility"
    )
    assert profiles["adaptive-rsimem"]["adaptiveConfigRequired"] is True
    assert sum(
        profile["adaptiveConfigRequired"] is True
        for profile in profiles.values()
    ) == 1
    with pytest.raises(ValueError, match="unknown adaptive"):
        adaptive_method_execution_profile("static-utility-rsimem")

    validation = {
        method: adaptive_validation_method_execution_profile(method)
        for method in ADAPTIVE_VALIDATION_METHOD_VARIANTS
    }
    assert validation["static-rsimem"]["adaptiveConfigRequired"] is False
    assert validation["proposal-rsimem"]["adaptiveConfigRequired"] is True
    assert validation["proposal-rsimem"]["validationOnly"] is True
    assert execution_order(2, ADAPTIVE_VALIDATION_METHOD_VARIANTS) == (
        "proposal-rsimem",
        "static-rsimem",
    )


def test_proposal_validation_manifest_is_distinct_from_official_adaptive(
    tmp_path: Path,
) -> None:
    profile = _adaptive_profile()
    profile["preparation"] = "matched_validation_trial_store"
    path = tmp_path / "proposal-validation.json"
    initialize_batch_manifest(
        path,
        **_manifest_kwargs(),
        execution_modes=ADAPTIVE_VALIDATION_METHOD_VARIANTS,
        adaptive_policy=profile,
        semantic_feedback_contract="sm01_tsv_v1",
    )
    manifest = load_manifest(path)
    assert manifest["configuration"]["executionModes"] == list(
        ADAPTIVE_VALIDATION_METHOD_VARIANTS
    )
    assert manifest["configuration"]["adaptivePolicy"]["preparation"] == (
        "matched_validation_trial_store"
    )

    official = _adaptive_profile()
    with pytest.raises(ValueError, match="adaptive policy schema"):
        initialize_batch_manifest(
            tmp_path / "wrong-profile.json",
            **_manifest_kwargs(),
            execution_modes=ADAPTIVE_VALIDATION_METHOD_VARIANTS,
            adaptive_policy=official,
            semantic_feedback_contract="sm01_tsv_v1",
        )


def test_manifest_records_effective_configuration_and_attempt_order(tmp_path: Path) -> None:
    path = tmp_path / "batch_manifest.json"
    experiment_id = initialize_batch_manifest(path, **_manifest_kwargs())
    record_attempt(
        path,
        replicate=2,
        ordinal=1,
        mode="native+ledger",
        run_name="r02_native_ledger",
        status="running",
    )
    record_attempt(
        path,
        replicate=2,
        ordinal=1,
        mode="native+ledger",
        run_name="r02_native_ledger",
        status="completed",
    )

    value = load_manifest(path)
    assert value["experimentId"] == experiment_id
    assert value["configuration"]["model"]["modelId"] == "gpt-test"
    assert value["configuration"]["judge"]["enabled"] is False
    assert value["configuration"]["budget"]["tasks"][0]["maxTurns"] == 20
    assert value["configuration"]["persistenceIsolation"]["strategy"] == "per_attempt_trace_directory"
    assert value["revisions"]["pastBenchTree"] == "past-tree"
    assert value["executionOrderByReplicate"]["2"][0] == "native+ledger"
    assert value["attempts"] == [{
        "actualOrdinal": 1,
        "attemptNumber": 1,
        "failureStage": None,
        "mode": "native+ledger",
        "ordinal": 1,
        "outputDirectory": "r02_native_ledger",
        "replicate": 2,
        "runName": "r02_native_ledger",
        "status": "completed",
    }]


def test_manifest_schedules_static_method_variants(tmp_path: Path) -> None:
    path = tmp_path / "static-methods.json"
    initialize_batch_manifest(
        path,
        **_manifest_kwargs(),
        execution_modes=STATIC_METHOD_VARIANTS,
    )
    value = load_manifest(path)
    assert value["configuration"]["executionModes"] == list(
        STATIC_METHOD_VARIANTS
    )
    assert value["executionOrderByReplicate"]["2"] == [
        "native-hermes",
        "static-rsimem",
        "no-persistence",
    ]
    record_attempt(
        path,
        replicate=1,
        ordinal=1,
        mode="no-persistence",
        run_name="r01_no_persistence",
        status="running",
    )

    utility_path = tmp_path / "static-utility-methods.json"
    initialize_batch_manifest(
        utility_path,
        **_manifest_kwargs(),
        execution_modes=STATIC_UTILITY_METHOD_VARIANTS,
    )
    utility = load_manifest(utility_path)
    assert utility["executionOrderByReplicate"]["2"] == [
        "static-utility-rsimem",
        "static-rsimem",
    ]

    adaptive_path = tmp_path / "adaptive-methods.json"
    initialize_batch_manifest(
        adaptive_path,
        **_manifest_kwargs(),
        execution_modes=ADAPTIVE_METHOD_VARIANTS,
        adaptive_policy=_adaptive_profile(),
        semantic_feedback_contract="sm01_tsv_v1",
    )
    adaptive = load_manifest(adaptive_path)
    assert adaptive["configuration"]["executionModes"] == list(
        ADAPTIVE_METHOD_VARIANTS
    )
    assert adaptive["configuration"]["adaptivePolicy"] == _adaptive_profile()
    assert adaptive["configuration"]["semanticFeedbackContract"] == (
        "sm01_tsv_v1"
    )
    assert adaptive["executionOrderByReplicate"]["2"] == [
        "native-hermes",
        "native-ledger",
        "static-rsimem",
        "adaptive-rsimem",
        "no-persistence",
    ]

    with pytest.raises(ValueError, match="adaptive policy profile"):
        initialize_batch_manifest(
            tmp_path / "adaptive-missing-policy.json",
            **_manifest_kwargs(),
            execution_modes=ADAPTIVE_METHOD_VARIANTS,
            semantic_feedback_contract="sm01_tsv_v1",
        )


def test_resolved_adaptive_policy_profile_binds_real_active_store(
    tmp_path: Path,
) -> None:
    offline_root, preparation = _offline(tmp_path)
    observations = _observation_batch(
        tmp_path / "manifest-observations.json",
        preparation,
    )
    active_root = tmp_path / "active"
    activate_adaptive_policy(
        offline_root,
        observations,
        output_root=active_root,
    )
    config_path = active_root / "adaptive-config.json"

    profile = resolved_adaptive_policy_profile(config_path)

    assert profile["activePolicyVersion"] == preparation.artifact.policy_version
    assert profile["activeArtifactId"] == preparation.artifact.artifact_id
    assert profile["activeArtifactDigest"] == preparation.artifact.content_digest
    assert len(profile["configDigest"]) == 64
    assert len(profile["storeDigest"]) == 64
    assert profile["preparation"] == "external_audited_active_store"

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    empty_config = empty_root / "adaptive-config.json"
    empty_store = empty_root / "adaptive-policy-store.json"
    empty_config.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (empty_root / "adaptive-activation-manifest.json").write_text(
        (active_root / "adaptive-activation-manifest.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    empty_store.write_text(json.dumps({
        "schema_version": ADAPTIVE_POLICY_STORE_SCHEMA_VERSION,
        "trusted_root_policy_versions": [
            preparation.artifact.parent_policy_version
        ],
        "artifacts": {},
        "records": {},
        "transitions": {},
        "active_policy_version": None,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="no ACTIVE policy"):
        resolved_adaptive_policy_profile(empty_config)


def test_manifest_fails_closed_for_missing_field_unknown_mode_and_dirty_benchmark(
    tmp_path: Path,
) -> None:
    path = tmp_path / "batch_manifest.json"
    kwargs = _manifest_kwargs()
    kwargs["past_bench_dirty"] = True
    with pytest.raises(ValueError, match="dirty PAST-Bench"):
        initialize_batch_manifest(path, **kwargs)

    kwargs["past_bench_dirty"] = False
    initialize_batch_manifest(path, **kwargs)
    value = json.loads(path.read_text(encoding="utf-8"))
    del value["configuration"]["budget"]
    with pytest.raises(ValueError, match="incomplete or unknown"):
        validate_manifest(value)

    value = json.loads(path.read_text(encoding="utf-8"))
    value["configuration"]["executionModes"][0] = "unknown"
    with pytest.raises(ValueError, match="unknown execution mode"):
        validate_manifest(value)

    invalid_kwargs = _manifest_kwargs()
    invalid_kwargs["model"] = {"profile": "incomplete"}
    with pytest.raises(ValueError, match="model fields"):
        initialize_batch_manifest(tmp_path / "invalid.json", **invalid_kwargs)


def test_restart_preserves_identity_and_failed_attempt_evidence(tmp_path: Path) -> None:
    path = tmp_path / "batch_manifest.json"
    kwargs = _manifest_kwargs()
    identity = initialize_batch_manifest(path, **kwargs)
    record_attempt(
        path,
        replicate=1,
        ordinal=1,
        mode="native",
        run_name="r01_native",
        status="running",
    )
    record_attempt(
        path,
        replicate=1,
        ordinal=1,
        mode="native",
        run_name="r01_native",
        status="failed",
        failure_stage="provider",
    )

    assert initialize_batch_manifest(path, **kwargs) == identity
    retry_name = next_attempt_name(
        path,
        replicate=1,
        ordinal=1,
        mode="native",
        base_run_name="r01_native",
    )
    assert retry_name == "r01_native_attempt02"
    record_attempt(
        path,
        replicate=1,
        ordinal=1,
        mode="native",
        run_name=retry_name,
        status="running",
    )
    value = load_manifest(path)
    assert [attempt["status"] for attempt in value["attempts"]] == ["failed", "running"]
    assert value["attempts"][0]["failureStage"] == "provider"

    changed = dict(kwargs)
    changed["rsimem_commit"] = "different-head"
    with pytest.raises(ValueError, match="different experiment"):
        initialize_batch_manifest(path, **changed)


def test_completed_slot_is_skipped_and_schedule_is_enforced(tmp_path: Path) -> None:
    path = tmp_path / "batch_manifest.json"
    initialize_batch_manifest(path, **_manifest_kwargs())
    record_attempt(
        path,
        replicate=1,
        ordinal=1,
        mode="native",
        run_name="r01_native",
        status="running",
    )
    record_attempt(
        path,
        replicate=1,
        ordinal=1,
        mode="native",
        run_name="r01_native",
        status="completed",
    )
    assert next_attempt_name(
        path,
        replicate=1,
        ordinal=1,
        mode="native",
        base_run_name="r01_native",
    ) is None
    with pytest.raises(ValueError, match="scheduled order"):
        next_attempt_name(
            path,
            replicate=2,
            ordinal=2,
            mode="native",
            base_run_name="wrong-order",
        )


def test_effective_profiles_and_family_budget_are_loaded_from_sources(tmp_path: Path) -> None:
    registry = tmp_path / "agents.yaml"
    registry.write_text(
        "agents:\n"
        "  hermes-luna:\n"
        "    default_model:\n"
        "      model_id: actual-model\n"
        "      api_key_env: SECRET_KEY\n"
        "      base_url: https://provider.invalid/v1\n",
        encoding="utf-8",
    )
    family_root = tmp_path / "family"
    (family_root / "episode-a").mkdir(parents=True)
    (family_root / "family.yaml").write_text("episode_order: [episode-a]\n", encoding="utf-8")
    (family_root / "episode-a/task.yaml").write_text(
        "task_id: task-a\nenvironment:\n  timeout_seconds: 45\n  max_turns: 7\n",
        encoding="utf-8",
    )

    run_config = tmp_path / "run.yaml"
    run_config.write_text(
        "judge:\n  enabled: false\nruntime:\n  mode: local\n  temperature: 0.25\n",
        encoding="utf-8",
    )
    run_profile = resolved_run_profile(run_config)
    assert run_profile == {
        "runtime": "local",
        "temperature": 0.25,
        "judge": {"enabled": False, "profile": "disabled", "modelId": None},
    }
    assert resolved_model_profile(
        registry,
        "hermes-luna",
        temperature=run_profile["temperature"],
    ) == {
        "profile": "hermes-luna/default_model",
        "modelId": "actual-model",
        "providerBaseUrl": "https://provider.invalid/v1",
        "temperature": 0.25,
    }
    budget = resolved_family_budget(family_root)
    assert budget["tasks"] == [{
        "episode": "episode-a",
        "taskId": "task-a",
        "maxTurns": 7,
        "timeoutSeconds": 45,
    }]
    assert "SECRET_KEY" not in json.dumps(budget)


def test_environment_profile_records_versions_without_install_paths() -> None:
    profile = resolved_environment_profile()
    assert profile["pythonVersion"].startswith("3.11.")
    assert {"rsimem", "past-bench", "hermes-agent"} <= set(profile["distributions"])
    serialized = json.dumps(profile)
    assert "/mnt/" not in serialized
    assert "site-packages" not in serialized
