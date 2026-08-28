from __future__ import annotations

from pathlib import Path

from rsimem.extraction_experiment_manifest import CleanRepositoryRevision
from rsimem.extraction_matched_preflight import (
    initialize_formal_matched_validation_batch,
)
from rsimem.extraction_validation_runtime import (
    prepare_extraction_matched_trial_runtime,
)
from test_extraction_matched_activation import _offline_decision
from test_extraction_offline_validation import _candidate, _parent


def test_matched_preflight_builds_validation_manifest_with_only_extraction_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    offline = _offline_decision(parent, candidate)
    trial_root = tmp_path / "trial"
    prepare_extraction_matched_trial_runtime(
        parent=parent,
        candidate=candidate,
        offline_decision=offline,
        output_root=trial_root,
    )
    family = tmp_path / "SM01_preference_adoption"
    family.mkdir()
    (family / "family.yaml").write_text("episode_order: []\n", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(
        "rsimem.extraction_matched_preflight.resolve_clean_repository",
        lambda path: CleanRepositoryRevision("a" * 40, "b" * 40),
    )
    monkeypatch.setattr(
        "rsimem.extraction_matched_preflight.resolved_task_template_profile",
        lambda path: {"taskManifestDigest": "c" * 64, "tasks": []},
    )
    monkeypatch.setattr(
        "rsimem.extraction_matched_preflight.resolved_model_profile",
        lambda *args, **kwargs: {
            "agentProfile": "hermes-luna",
            "modelId": "fixture",
            "providerBaseUrl": "https://provider.invalid/v1",
            "temperature": 0.0,
            "runtime": "local",
            "judge": "disabled",
            "semanticIngestionProfileId": kwargs["semantic_policy"].model_profile,
            "semanticWritebackTimeoutSeconds": 30.0,
            "semanticWritebackMaxOutputTokens": 4096,
        },
    )
    monkeypatch.setattr(
        "rsimem.extraction_matched_preflight.initialize_extraction_batch_manifest",
        lambda path, **kwargs: captured.update(kwargs) or "experiment.fixture-v1",
    )

    result = initialize_formal_matched_validation_batch(
        manifest_path=tmp_path / "manifest.json",
        batch_registry_path=tmp_path / "registry.json",
        batch_id="batch.validation-v1",
        rsimem_root=tmp_path,
        past_bench_root=tmp_path,
        family_root=family,
        agent_registry_path=Path("configs/agents.yaml"),
        run_config_path=Path("configs/past_bench_luna_smoke.yaml"),
        experiment_config_path=Path("configs/extraction_feedback_sm01.json"),
        trial_config_path=trial_root / "extraction-matched-trial.json",
    )

    assert result == "experiment.fixture-v1"
    assert captured["phase"] == "validation"
    assert captured["parent_policy"].extraction_component_digest == parent.body_digest
    assert captured["active_policy"].extraction_component_digest == candidate.body_digest
    assert captured["matched_policy"].parent == captured["parent_policy"]
    assert captured["matched_policy"].candidate == captured["active_policy"]
