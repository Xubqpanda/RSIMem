from __future__ import annotations

import json
from pathlib import Path

import pytest

import rsimem.extraction_experiment_preflight as preflight
from rsimem.extraction_experiment_manifest import (
    CleanRepositoryRevision,
    load_extraction_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def _family(tmp_path: Path) -> Path:
    root = tmp_path / "SM01_preference_adoption"
    episode = root / "learn"
    episode.mkdir(parents=True)
    (root / "family.yaml").write_text(
        "episode_order:\n  - learn\n",
        encoding="utf-8",
    )
    (episode / "task.yaml").write_text(
        "task_id: sm01-learn\nenvironment:\n  max_turns: 20\n  timeout_seconds: 300\n",
        encoding="utf-8",
    )
    return root


def _agent_files(tmp_path: Path) -> tuple[Path, Path]:
    registry = tmp_path / "agents.yaml"
    registry.write_text(
        "agents:\n"
        "  hermes-luna:\n"
        "    default_model:\n"
        "      model_id: gpt-fixture\n"
        "      api_key_env: GPT_LUNA_API_KEY\n"
        "      base_url: https://provider.invalid/v1\n",
        encoding="utf-8",
    )
    run = tmp_path / "run.yaml"
    run.write_text(
        "runtime:\n  mode: local\n  temperature: 0.0\n"
        "judge:\n  enabled: false\n",
        encoding="utf-8",
    )
    return registry, run


def test_checked_in_feedback_config_and_plain_parent_are_valid() -> None:
    config = preflight.load_extraction_preflight_config(
        ROOT / "configs/extraction_feedback_sm01.json"
    )
    parent = preflight.resolved_plain_parent_policy()

    assert config["replicates"] == 3
    assert config["acceptanceCriteria"]["maximumProposalGenerations"] == 1
    assert parent.boundary == "task-completed-v1"
    assert parent.backend == "hermes-native-semantic"
    assert parent.model_profile == "semantic-ingestion-default-v1"


def test_task_template_digest_covers_all_family_files(tmp_path: Path) -> None:
    family = _family(tmp_path)
    first = preflight.resolved_task_template_profile(family)
    assert first["tasks"] == [{
        "episode": "learn",
        "taskId": "sm01-learn",
        "maxTurns": 20,
        "timeoutSeconds": 300,
    }]

    (family / "learn" / "prompt.md").write_text("new prompt\n", encoding="utf-8")
    second = preflight.resolved_task_template_profile(family)
    assert second["fileCount"] == first["fileCount"] + 1
    assert second["taskManifestDigest"] != first["taskManifestDigest"]


def test_model_profile_omits_credentials_and_rejects_judge(tmp_path: Path) -> None:
    registry, run = _agent_files(tmp_path)
    parent = preflight.resolved_plain_parent_policy()
    model = preflight.resolved_model_profile(
        registry,
        run,
        agent="hermes-luna",
        semantic_policy=parent,
        semantic_writeback={"timeoutSeconds": 30.0, "maxOutputTokens": 4096},
    )
    assert model["modelId"] == "gpt-fixture"
    assert model["semanticIngestionProfileId"] == parent.model_profile
    assert "api_key_env" not in json.dumps(model)

    run.write_text(
        "runtime:\n  mode: local\n  temperature: 0.0\n"
        "judge:\n  enabled: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="disabled judge"):
        preflight.resolved_model_profile(
            registry,
            run,
            agent="hermes-luna",
            semantic_policy=parent,
            semantic_writeback={"timeoutSeconds": 30.0, "maxOutputTokens": 4096},
        )


def test_formal_feedback_preflight_builds_manifest_before_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family = _family(tmp_path)
    registry, run = _agent_files(tmp_path)
    monkeypatch.setattr(
        preflight,
        "resolve_clean_repository",
        lambda path: (
            CleanRepositoryRevision("past-commit", "past-tree")
            if Path(path).name == "past"
            else CleanRepositoryRevision("rsimem-commit", "rsimem-tree")
        ),
    )
    manifest_path = tmp_path / "output" / "batch_manifest.json"

    experiment_id = preflight.initialize_formal_feedback_batch(
        manifest_path=manifest_path,
        batch_registry_path=tmp_path / "batch-registry.json",
        batch_id="formal-feedback.fixture-v1",
        rsimem_root=tmp_path,
        past_bench_root=tmp_path / "past",
        family_root=family,
        agent_registry_path=registry,
        run_config_path=run,
        experiment_config_path=ROOT / "configs/extraction_feedback_sm01.json",
    )
    manifest = load_extraction_manifest(manifest_path)

    assert manifest["experimentId"] == experiment_id
    assert manifest["phase"] == "feedback"
    assert manifest["executionProfiles"]["static-extraction-rsimem"] == {
        "persistenceVariant": "with_persistence",
        "rsimemMode": "native+ledger",
        "lifecycleEvaluatorMode": "disabled",
        "semanticWritebackMode": "static",
        "utilityGate": "disabled",
        "nativeMemoryWriter": "disabled",
        "backgroundMemoryReview": "disabled",
        "extractionArtifactRole": "parent",
    }
    assert manifest["requestBudget"]["resolved"]["tasks"][0]["maxTurns"] == 20
    assert manifest["modelProfile"]["resolved"]["providerBaseUrl"] == (
        "https://provider.invalid/v1"
    )

    resumed = preflight.initialize_formal_feedback_batch(
        manifest_path=manifest_path,
        batch_registry_path=tmp_path / "batch-registry.json",
        batch_id="formal-feedback.fixture-v1",
        rsimem_root=tmp_path,
        past_bench_root=tmp_path / "past",
        family_root=family,
        agent_registry_path=registry,
        run_config_path=run,
        experiment_config_path=ROOT / "configs/extraction_feedback_sm01.json",
    )
    assert resumed == experiment_id

    run.write_text(
        "runtime:\n  mode: local\n  temperature: 0.2\n"
        "judge:\n  enabled: false\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity drifted"):
        preflight.initialize_formal_feedback_batch(
            manifest_path=manifest_path,
            batch_registry_path=tmp_path / "batch-registry.json",
            batch_id="formal-feedback.fixture-v1",
            rsimem_root=tmp_path,
            past_bench_root=tmp_path / "past",
            family_root=family,
            agent_registry_path=registry,
            run_config_path=run,
            experiment_config_path=ROOT / "configs/extraction_feedback_sm01.json",
        )


def test_preflight_rejects_wrong_family_before_manifest_creation(
    tmp_path: Path,
) -> None:
    family = _family(tmp_path)
    wrong = tmp_path / "wrong-family"
    family.rename(wrong)
    registry, run = _agent_files(tmp_path)
    manifest = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="family.*disagree"):
        preflight.initialize_formal_feedback_batch(
            manifest_path=manifest,
            batch_registry_path=tmp_path / "registry.json",
            batch_id="wrong-family-v1",
            rsimem_root=tmp_path,
            past_bench_root=tmp_path,
            family_root=wrong,
            agent_registry_path=registry,
            run_config_path=run,
            experiment_config_path=ROOT / "configs/extraction_feedback_sm01.json",
        )
    assert not manifest.exists()
