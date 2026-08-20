from tests.self_evolve_synthetic_support import synthesize_agent_summary


def test_acceptance_thresholds_hold_for_synthetic_agents():
    no_persistence = synthesize_agent_summary("NoPersistenceAgent")
    rule_learner = synthesize_agent_summary("RuleLearnerAgent")
    stale_follower = synthesize_agent_summary("StaleFollowerAgent")

    no_persistence_delta = no_persistence["benchmark_signal"]["avg_family_task_score_delta"]
    assert -0.05 <= no_persistence_delta <= 0.05

    rule_family_deltas = [
        family["improvement"]["task_score_delta_eval_minus_baseline"]
        for family in rule_learner["family_summary"].values()
    ]
    assert sum(1 for delta in rule_family_deltas if delta >= 0.15) >= 5
    assert rule_learner["benchmark_signal"]["strong_attribution_families"] >= 4

    f02_gap = (
        rule_learner["family_summary"]["F02_skill_patch"]["bucket_summary"]["evaluation"]["avg_task_score"]
        - stale_follower["family_summary"]["F02_skill_patch"]["bucket_summary"]["evaluation"]["avg_task_score"]
    )
    f06_gap = (
        rule_learner["family_summary"]["F06_stale_conflict_update"]["bucket_summary"]["evaluation"]["avg_task_score"]
        - stale_follower["family_summary"]["F06_stale_conflict_update"]["bucket_summary"]["evaluation"]["avg_task_score"]
    )
    assert f02_gap >= 0.20
    assert f06_gap >= 0.20

    assert all(
        family["bucket_summary"]["evaluation"]["avg_task_score"] < 0.95
        for family in no_persistence["family_summary"].values()
    )
