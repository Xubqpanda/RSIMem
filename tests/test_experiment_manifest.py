from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsimem.experiment_manifest import (
    execution_order,
    initialize_batch_manifest,
    load_manifest,
    next_attempt_name,
    record_attempt,
    resolved_environment_profile,
    resolved_family_budget,
    resolved_model_profile,
    resolved_run_profile,
    validate_manifest,
)


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
