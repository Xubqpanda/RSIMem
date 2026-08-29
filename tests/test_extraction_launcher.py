from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_formal_feedback_launcher_uses_plain_extraction_profile() -> None:
    script = (
        ROOT / "scripts/run_luna_extraction_feedback_sm01.sh"
    ).read_text(encoding="utf-8")

    assert "static-extraction-rsimem" in script
    assert "--rsimem-mode native+ledger" in script
    assert "--rsimem-lifecycle-evaluator-mode disabled" in script
    assert "--rsimem-semantic-writeback-mode static" in script
    assert "--background-review-wait-s 0" in script
    assert "--rsimem-semantic-feedback-contract sm01_tsv_v1" in script
    assert "rsimem.extraction_experiment_analysis" in script
    assert "classify_extraction_audit_failure" in script
    assert "RSIMEM_EXTRACTION_EXPERIMENT_CONFIG" in script
    assert 'rglob("extraction_sources.jsonl")' in script
    assert "audit_process_events" in script
    assert "formal process feedback audit failed" in script
    assert "static_utility" not in script
    assert "adaptive_utility" not in script
    assert "adaptive-rsimem" not in script
    assert "GPT_LUNA_API_KEY" in script
    assert "s" + "k-" not in script


def test_formal_launchers_use_one_overridable_agent_registry() -> None:
    for name in (
        "run_luna_extraction_feedback_sm01.sh",
        "run_luna_extraction_matched.sh",
    ):
        script = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert 'AGENT_REGISTRY="${RSIMEM_AGENT_REGISTRY:-' in script
        assert '--agent-registry "${AGENT_REGISTRY}"' in script
        assert '--registry "${AGENT_REGISTRY}"' in script


def test_backup_agent_registry_contains_no_credential() -> None:
    path = ROOT / "configs/agents_backup.yaml"
    raw = path.read_text(encoding="utf-8")
    registry = yaml.safe_load(raw)
    model = registry["agents"]["hermes-luna"]["default_model"]

    assert model["model_id"] == "gpt-5.6-luna"
    assert model["api_key_env"] == "GPT_LUNA_API_KEY"
    assert model["base_url"] == "http://47.88.93.22:10001/v1"
    assert "s" + "k-" not in raw
