from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
V2_ROOT = REPO_ROOT / "self-evolve-tasks-v2" / "proactive_information_gathering"
CFG_ROOT = REPO_ROOT / "configs" / "self_evolve_v2"


def _load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_pg05_reflection_forbids_exact_packet_details() -> None:
    expectations = _load_yaml(
        V2_ROOT
        / "PG05_change_freeze_followup"
        / "reflection_gap_review"
        / "expectations.json"
    )
    forbidden = set(expectations["forbidden_output_keywords"])
    assert "CAB-FRZ-441" in forbidden
    assert "2026-05-03T01:00Z" in forbidden


def test_pg05_control_boundary_targets_latest_packet() -> None:
    task = _load_yaml(
        V2_ROOT
        / "PG05_change_freeze_followup"
        / "control_boundary_latest_packet"
        / "task.yaml"
    )
    assert "latest matching packet" in task["prompt"]["text"]


def test_pg06_control_scope_patch_matches_current_bench_chain() -> None:
    cfg = _load_yaml(CFG_ROOT / "hermes_self_evolve_v2_pg06_kappa_integration_review_only.yaml")
    control = next(
        ep
        for ep in cfg["episodes"]
        if ep["label"] == "pg06_kappa_integration_review_control_boundary_scope_patch_mismatch"
    )
    assert control["history_mode"] == "continue"
    assert control["preseed_artifacts_dir"] == ""


def test_pg06_update_policy_prompt_patches_scope_and_preserves_workflow() -> None:
    task = _load_yaml(
        V2_ROOT
        / "PG06_kappa_integration_review"
        / "update_or_stabilize_policy_note"
        / "task.yaml"
    )
    text = task["prompt"]["text"]
    assert "payment or identity" in text
    assert "workflow steps are unchanged" in text
