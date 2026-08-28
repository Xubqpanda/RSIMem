from pathlib import Path


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
    assert "static_utility" not in script
    assert "adaptive_utility" not in script
    assert "adaptive-rsimem" not in script
    assert "GPT_LUNA_API_KEY" in script
    assert "s" + "k-" not in script
