"""Phase B test: V2 Layer 1 metrics + 4-level reporting.

Covers formula correctness for BaselineHeadroom / OutcomeDelta /
AblationLift / AutonomousReuseRate / TransferRobustness / ShortcutResistance /
MechanismConfidence / FamilyEvolveScore, plus ability-level aggregation and
benchmark-level capability vector.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from past_bench.metrics.v2_layer1 import (
    V2FamilyMetrics,
    aggregate_by_ability,
    build_benchmark_report,
    compute_family_metrics,
    emit_reports,
    ingest_full_reports,
    ingest_sequence_comparison,
)
from past_bench.metrics.v2_layer2 import compute_layer2
from past_bench.metrics.v2_layer3 import compute_layer3
from past_bench.models.self_evolve import V2FamilyMetadata


def _family_meta(
    fam_id: str = "PC01_sop_bootstrap",
    ability: str = "procedural_ability",
    trigger: str = "repeated_cases_induction",
    substrate: str = "skill",
) -> V2FamilyMetadata:
    return V2FamilyMetadata(
        family_id=fam_id,
        title="test",
        ability_dir=ability,
        primary_ability=ability,
        primary_trigger=trigger,
        expected_substrate=substrate,
        transfer_distance="near_far",
        family_length_tier="tier2",
        instances_per_bucket={"cold": 1, "learn": 2, "eval_near": 1, "eval_far": 1},
        total_episodes=5,
        control_set=["no_persistence", "shortcut", "wrong_mechanism"],
        status="skeleton",
    )


def _wp_block(baseline: float, evaluation: float, *, near: float | None = None, far: float | None = None, retrieval: float = 0.8):
    buckets: dict = {
        "baseline": {"avg_task_score": baseline, "retrieval_before_success_rate": 0.0},
        "learn": {"avg_task_score": baseline + 0.05, "skill_create_count": 1, "memory_write_count": 0, "retrieval_before_success_rate": 0.0},
        "evaluation": {"avg_task_score": evaluation, "retrieval_before_success_rate": retrieval},
    }
    if near is not None:
        buckets["eval_near"] = {"avg_task_score": near, "retrieval_before_success_rate": retrieval}
    if far is not None:
        buckets["eval_far"] = {"avg_task_score": far, "retrieval_before_success_rate": retrieval}
    return {"bucket_summary": buckets, "metrics": {"mechanism_score": 0.8}}


def test_baseline_headroom_measured():
    wp = _wp_block(0.30, 0.70)
    m = compute_family_metrics("F", _family_meta(), wp, None)
    assert m.baseline_headroom == pytest.approx(0.70)
    assert m.basis["baseline_headroom"] == "measured"


def test_outcome_delta_within_run():
    wp = _wp_block(0.30, 0.70)
    m = compute_family_metrics("F", _family_meta(), wp, None)
    assert m.outcome_delta == pytest.approx(0.40)


def test_ablation_lift_paired_runs():
    wp = _wp_block(0.30, 0.70)  # delta 0.40
    wo = _wp_block(0.30, 0.50)  # delta 0.20
    m = compute_family_metrics("F", _family_meta(), wp, wo)
    assert m.ablation_lift == pytest.approx(0.20)
    assert m.basis["ablation_lift"] == "measured"


def test_ablation_lift_unmeasured_without_control():
    wp = _wp_block(0.30, 0.70)
    m = compute_family_metrics("F", _family_meta(), wp, None)
    assert m.ablation_lift is None
    assert m.basis["ablation_lift"] == "unmeasured"


def test_autonomous_reuse_rate_from_retrieval_proxy():
    wp = _wp_block(0.30, 0.70, retrieval=0.75)
    m = compute_family_metrics("F", _family_meta(), wp, None)
    assert m.autonomous_reuse_rate == pytest.approx(0.75)
    assert m.basis["autonomous_reuse_rate"].startswith("proxy")


def test_transfer_robustness_near_far_ratio():
    wp = _wp_block(0.30, 0.70, near=0.80, far=0.60)
    m = compute_family_metrics("F", _family_meta(), wp, None)
    assert m.transfer_robustness == pytest.approx(0.75)


def test_transfer_robustness_defaults_when_split_absent():
    wp = _wp_block(0.30, 0.70)  # no eval_near / eval_far
    m = compute_family_metrics("F", _family_meta(), wp, None)
    assert m.transfer_robustness == 1.0
    assert "no_near_far_split" in m.basis["transfer_robustness"]


def test_shortcut_resistance_when_shortcut_control_given():
    wp = _wp_block(0.30, 0.80)
    m = compute_family_metrics("F", _family_meta(), wp, None, shortcut_eval_score=0.20)
    # 1 - 0.20/0.80 = 0.75
    assert m.shortcut_resistance == pytest.approx(0.75)
    assert m.basis["shortcut_resistance"] == "measured"


def test_shortcut_resistance_defaults_without_control():
    wp = _wp_block(0.30, 0.80)
    m = compute_family_metrics("F", _family_meta(), wp, None)
    assert m.shortcut_resistance == 1.0


def test_mechanism_confidence_passthrough_from_runner():
    wp = _wp_block(0.30, 0.70)  # mechanism_score=0.8 in metrics block
    m = compute_family_metrics("F", _family_meta(), wp, None)
    assert m.mechanism_confidence == pytest.approx(0.8)
    assert m.basis["mechanism_confidence"] == "runner_emitted"


def test_mechanism_confidence_inferred_when_runner_silent():
    wp = {"bucket_summary": {
        "baseline": {"avg_task_score": 0.3},
        "learn": {"avg_task_score": 0.4, "skill_create_count": 1},
        "evaluation": {"avg_task_score": 0.7, "retrieval_before_success_rate": 0.5},
    }}
    m = compute_family_metrics("F", _family_meta(), wp, None)
    assert m.mechanism_confidence == 1.0
    assert m.basis["mechanism_confidence"] == "inferred_from_bucket_signals"


def test_family_evolve_score_full_formula():
    wp = _wp_block(0.30, 0.70, near=0.80, far=0.60, retrieval=0.75)
    wo = _wp_block(0.30, 0.50)
    m = compute_family_metrics("F", _family_meta(), wp, wo, shortcut_eval_score=0.20)
    # lift = 0.20 ; mc = 0.8 ; reuse = 0.75 ; transfer = 0.75
    # score = 0.20 * 0.8 * 0.75 * 0.75 = 0.09
    assert m.family_evolve_score == pytest.approx(0.09)


def test_family_evolve_score_clips_negative_lift():
    wp = _wp_block(0.60, 0.50, near=0.50, far=0.50, retrieval=0.5)
    wo = _wp_block(0.60, 0.70, near=0.70, far=0.70, retrieval=0.5)
    m = compute_family_metrics("F", _family_meta(), wp, wo)
    # AblationLift = -0.10 → clipped to 0 → FamilyEvolveScore = 0
    assert m.ablation_lift is not None and m.ablation_lift < 0
    assert m.family_evolve_score == 0.0


def test_ability_aggregation_groups_by_primary_ability():
    fms = [
        V2FamilyMetrics(
            family_id="PC01", primary_ability="procedural_ability",
            expected_substrate="skill", primary_trigger="repeated_cases_induction",
            ablation_lift=0.2, outcome_delta=0.3, mechanism_confidence=0.8,
            family_evolve_score=0.1,
        ),
        V2FamilyMetrics(
            family_id="PC02", primary_ability="procedural_ability",
            expected_substrate="skill", primary_trigger="conflict_update",
            ablation_lift=0.1, outcome_delta=0.2, mechanism_confidence=0.6,
            family_evolve_score=0.05,
        ),
        V2FamilyMetrics(
            family_id="SM01", primary_ability="memory_ability",
            expected_substrate="memory", primary_trigger="explicit_instruction",
            ablation_lift=-0.05, outcome_delta=0.0, mechanism_confidence=0.0,
            family_evolve_score=0.0,
        ),
    ]
    reports = {r.ability: r for r in aggregate_by_ability(fms)}
    pc = reports["procedural_ability"]
    assert pc.family_count == 2
    assert pc.families_with_positive_lift == 2
    assert pc.avg_ablation_lift == pytest.approx(0.15)
    assert pc.avg_family_evolve_score == pytest.approx(0.075)

    sm = reports["memory_ability"]
    assert sm.family_count == 1
    assert sm.families_with_positive_lift == 0

    ig = reports["proactive_information_gathering"]
    assert ig.family_count == 0
    assert ig.avg_family_evolve_score is None


def test_benchmark_report_capability_vector_has_all_abilities():
    fms = [
        V2FamilyMetrics(
            family_id="PC01", primary_ability="procedural_ability",
            expected_substrate="skill", primary_trigger="repeated_cases_induction",
            family_evolve_score=0.2,
        ),
    ]
    ability_reports = aggregate_by_ability(fms)
    report = build_benchmark_report(fms, ability_reports, total_families=20)
    assert report.total_families == 20
    assert report.families_with_metrics == 1
    assert set(report.capability_vector.keys()) == {
        "memory_ability",
        "procedural_ability",
        "proactive_information_gathering",
        "update_ability",
    }
    assert report.capability_vector["procedural_ability"] == pytest.approx(0.2)
    assert report.capability_vector["memory_ability"] is None


def test_emit_reports_writes_three_files(tmp_path: Path):
    fms = [V2FamilyMetrics(
        family_id="PC01", primary_ability="procedural_ability",
        expected_substrate="skill", primary_trigger="repeated_cases_induction",
        family_evolve_score=0.1,
    )]
    paths = emit_reports(fms, tmp_path, v2_root=None)
    for key in ("family_report", "ability_report", "benchmark_report"):
        assert paths[key].exists()
    benchmark = json.loads(paths["benchmark_report"].read_text())
    assert benchmark["families_with_metrics"] == 1


def test_layer2_memory_runner_passthrough():
    with_block = {"benchmark_signal": {
        "avg_write_precision": 0.7,
        "avg_recall_accuracy": 0.6,
        "avg_update_correctness": 0.5,
        "avg_retention_horizon": 0.9,
        "avg_memory_pollution_rate": 0.1,
    }}
    m = compute_layer2("SM01", "memory_ability", {"bucket_summary": {}}, with_block)
    assert m.write_precision == pytest.approx(0.7)
    assert m.memory_pollution_rate == pytest.approx(0.1)
    assert m.retention_horizon == pytest.approx(0.9)
    assert m.basis["write_precision"] == "runner_emitted"


def test_layer2_procedural_skill_creation_rate():
    wp_fam = {"bucket_summary": {
        "learn": {"skill_create_count": 3, "episode_count": 2},
        "eval_near": {"avg_task_score": 0.8},
        "eval_far": {"avg_task_score": 0.6},
        "evaluation": {"retrieval_before_success_rate": 0.75},
    }}
    m = compute_layer2("PC01", "procedural_ability", wp_fam, {})
    assert m.skill_creation_rate == pytest.approx(1.5)
    assert m.transfer_gain == pytest.approx(-0.2)
    assert m.skill_reuse_rate == pytest.approx(0.75)


def test_layer2_information_gathering_stubs_basis():
    m = compute_layer2("IG01", "proactive_information_gathering", {"bucket_summary": {}}, {})
    assert m.retrieval_precision is None
    assert m.basis["retrieval_precision"] == "needs_runner_collection"


def test_layer3_conflict_update_overwrite_completeness():
    wp_fam = {"bucket_summary": {}}
    with_block = {"benchmark_signal": {"avg_update_correctness": 0.42}}
    m = compute_layer3("SM03", "conflict_update", wp_fam, with_block)
    assert m.overwrite_completeness == pytest.approx(0.42)
    assert "avg_update_correctness" in m.basis["overwrite_completeness"]


def test_layer3_repeated_cases_induction_transfer_proxy():
    wp_fam = {"bucket_summary": {
        "eval_near": {"avg_task_score": 0.8},
        "eval_far": {"avg_task_score": 0.6},
    }}
    m = compute_layer3("PC03", "repeated_cases_induction", wp_fam, {})
    assert m.transfer_without_keyword_overlap == pytest.approx(0.75)


def test_ingest_full_reports_emits_all_layers(tmp_path: Path):
    p = Path("traces/self_evolve_v2_minimax_calib_f01/sequence_comparison.json")
    if not p.exists():
        pytest.skip("no live calibration trace")
    l1, l2, l3, tasks = ingest_full_reports(p, v2_root="self-evolve-tasks-v2")
    assert len(l1) >= 1 and len(l2) == len(l1) and len(l3) == len(l1)
    assert len(tasks) >= 1
    paths = emit_reports(
        l1, tmp_path, v2_root="self-evolve-tasks-v2",
        layer2=l2, layer3=l3, task_rows=tasks,
    )
    for key in ("task_report", "family_report", "ability_report", "benchmark_report"):
        assert paths[key].exists()
    # family_report should include layer2 and layer3 blocks
    fam_json = json.loads(paths["family_report"].read_text())
    assert "layer2" in fam_json[0]
    assert "layer3" in fam_json[0]


def test_ingest_real_sequence_comparison():
    """Regression against the live minimax F01 calibration run."""
    p = Path("traces/self_evolve_v2_minimax_calib_f01/sequence_comparison.json")
    if not p.exists():
        pytest.skip("no live calibration trace available")
    fms = ingest_sequence_comparison(p, v2_root="self-evolve-tasks-v2")
    assert len(fms) >= 1
    fm = fms[0]
    assert fm.baseline_headroom is not None
    # Ablation lift should be populated because the run used --compare-no-persistence
    assert fm.ablation_lift is not None
    assert math.isfinite(fm.family_evolve_score or 0)
