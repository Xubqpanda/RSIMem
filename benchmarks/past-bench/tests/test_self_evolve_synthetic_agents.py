from tests.self_evolve_synthetic_support import synthesize_agent_summary


def test_synthetic_agent_ranking_matches_expected_order():
    no_persistence = synthesize_agent_summary("NoPersistenceAgent")
    write_only = synthesize_agent_summary("WriteOnlyAgent")
    read_only = synthesize_agent_summary("ReadOnlyCorrectPreseedAgent")
    rule_learner = synthesize_agent_summary("RuleLearnerAgent")

    assert (
        rule_learner["benchmark_signal"]["avg_evolve_index"]
        > read_only["benchmark_signal"]["avg_evolve_index"]
        > write_only["benchmark_signal"]["avg_evolve_index"]
        > no_persistence["benchmark_signal"]["avg_evolve_index"]
    )


def test_reflection_learner_dominates_failure_to_rule_family():
    reflection_learner = synthesize_agent_summary("ReflectionLearnerAgent")
    rule_learner = synthesize_agent_summary("RuleLearnerAgent")
    stale_follower = synthesize_agent_summary("StaleFollowerAgent")

    f07_reflection = reflection_learner["family_summary"]["F07_failure_to_rule"]
    f07_rule = rule_learner["family_summary"]["F07_failure_to_rule"]

    assert (
        f07_reflection["bucket_summary"]["evaluation"]["avg_task_score"]
        > f07_rule["bucket_summary"]["evaluation"]["avg_task_score"]
    )
    assert stale_follower["family_summary"]["F02_skill_patch"]["bucket_summary"]["evaluation"]["avg_task_score"] < 0.4
