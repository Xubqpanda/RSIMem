from past_bench.runner.self_evolve import summarize_sequence


def _episode(family_id: str, bucket: str, task_score: float, confidence: float, used_signal: bool) -> dict:
    return {
        "family_id": family_id,
        "bucket": bucket,
        "mechanism": "skill",
        "expected_persistence_signal": "skill",
        "task_score": task_score,
        "passed": task_score >= 0.7,
        "tool_dispatch_count": 4,
        "token_usage": {"total_tokens": 100},
        "timing": {"wall_time_s": 1.0},
        "artifacts": {"memory_file_exists": False, "user_file_exists": False, "skill_count": 1 if confidence else 0},
        "internal_tools": {
            "memory_calls": 0,
            "skill_manage_calls": 1 if confidence else 0,
            "session_search_calls": 0,
            "memory_write_count": 0,
            "skill_create_count": 1 if confidence else 0,
            "skill_update_count": 1 if confidence else 0,
        },
        "retrieval_signals": {
            "used_expected_signal": used_signal,
            "retrieval_before_first_update": used_signal,
            "memory_read_count": 0,
            "session_search_count": 0,
            "skill_read_count": 1 if used_signal else 0,
        },
        "mechanism_scores": {
            "mechanism_confidence": confidence,
            "artifact_quality_score": confidence,
            "transfer_quality": 1.0 if used_signal else 0.0,
            "shortcut_resistance": 1.0 if confidence else 0.5,
        },
    }


def test_family_attribution_labels_are_stable_for_strong_and_none():
    episodes = [
        _episode("F01_skill_bootstrap", "baseline", 0.40, 0.0, False),
        _episode("F01_skill_bootstrap", "learn", 0.62, 0.95, True),
        _episode("F01_skill_bootstrap", "learn", 0.64, 0.95, True),
        _episode("F01_skill_bootstrap", "evaluation", 0.74, 0.95, True),
        _episode("F01_skill_bootstrap", "evaluation", 0.72, 0.95, True),
        _episode("F02_skill_patch", "baseline", 0.45, 0.0, False),
        _episode("F02_skill_patch", "learn", 0.43, 0.0, False),
        _episode("F02_skill_patch", "learn", 0.44, 0.0, False),
        _episode("F02_skill_patch", "evaluation", 0.42, 0.0, False),
        _episode("F02_skill_patch", "evaluation", 0.41, 0.0, False),
    ]

    summary = summarize_sequence(sequence_name="test", variant="synthetic", episodes=episodes)

    assert summary["family_summary"]["F01_skill_bootstrap"]["attribution"]["label"] == "strong"
    assert summary["family_summary"]["F02_skill_patch"]["attribution"]["label"] == "none"


def test_positive_delta_without_sufficient_mechanism_evidence_stays_weak():
    episodes = [
        _episode("F05_hidden_correlation", "baseline", 0.35, 0.0, False),
        _episode("F05_hidden_correlation", "learn", 0.40, 0.2, False),
        _episode("F05_hidden_correlation", "learn", 0.42, 0.2, False),
        _episode("F05_hidden_correlation", "evaluation", 0.72, 0.49, False),
        _episode("F05_hidden_correlation", "evaluation", 0.70, 0.49, False),
    ]
    for episode in episodes:
        if episode["bucket"] == "evaluation":
            episode["mechanism_scores"]["artifact_quality_score"] = 0.0

    summary = summarize_sequence(sequence_name="test", variant="synthetic", episodes=episodes)

    assert summary["family_summary"]["F05_hidden_correlation"]["attribution"]["label"] == "weak"


def test_memory_injection_only_does_not_raise_evaluation_artifact_reuse_rate():
    episodes = [
        {
            **_episode("F06_stale_conflict_update", "baseline", 0.30, 0.0, False),
            "mechanism": "memory",
            "expected_persistence_signal": "memory",
        },
        {
            **_episode("F06_stale_conflict_update", "learn", 0.60, 0.85, False),
            "mechanism": "memory",
            "expected_persistence_signal": "memory",
        },
        {
            **_episode("F06_stale_conflict_update", "learn", 0.62, 0.85, False),
            "mechanism": "memory",
            "expected_persistence_signal": "memory",
        },
        {
            **_episode("F06_stale_conflict_update", "evaluation", 0.78, 0.85, False),
            "mechanism": "memory",
            "expected_persistence_signal": "memory",
        },
        {
            **_episode("F06_stale_conflict_update", "evaluation", 0.80, 0.85, False),
            "mechanism": "memory",
            "expected_persistence_signal": "memory",
        },
    ]

    for episode in episodes:
        episode["artifacts"] = {"memory_file_exists": True, "user_file_exists": False, "skill_count": 0}
        episode["internal_tools"] = {
            "memory_calls": 0,
            "skill_manage_calls": 0,
            "session_search_calls": 0,
            "memory_write_count": 0,
            "skill_create_count": 0,
            "skill_update_count": 0,
        }
        episode["retrieval_signals"] = {
            "used_expected_signal": False,
            "retrieval_before_first_update": False,
            "memory_read_count": 0,
            "memory_injection_count": 1,
            "session_search_count": 0,
            "skill_read_count": 0,
            "retrieval_signal_count": 1,
        }
        episode["mechanism_scores"] = {
            "mechanism_confidence": 0.85,
            "artifact_quality_score": 0.85,
            "transfer_quality": 0.0,
            "shortcut_resistance": 1.0,
        }

    summary = summarize_sequence(sequence_name="test", variant="synthetic", episodes=episodes)
    family = summary["family_summary"]["F06_stale_conflict_update"]

    assert family["bucket_summary"]["evaluation"]["artifact_reuse_rate"] == 0.0
    assert family["attribution"]["label"] != "strong"


def test_session_search_family_can_reach_strong_attribution_without_artifact_score():
    episodes = [
        {
            **_episode("F04_session_recall", "baseline", 0.30, 0.0, False),
            "mechanism": "session_search",
            "expected_persistence_signal": "session_search",
        },
        {
            **_episode("F04_session_recall", "learn", 0.40, 0.9, True),
            "mechanism": "session_search",
            "expected_persistence_signal": "session_search",
        },
        {
            **_episode("F04_session_recall", "learn", 0.42, 0.9, True),
            "mechanism": "session_search",
            "expected_persistence_signal": "session_search",
        },
        {
            **_episode("F04_session_recall", "evaluation", 0.78, 0.9, True),
            "mechanism": "session_search",
            "expected_persistence_signal": "session_search",
        },
        {
            **_episode("F04_session_recall", "evaluation", 0.80, 0.9, True),
            "mechanism": "session_search",
            "expected_persistence_signal": "session_search",
        },
    ]

    for episode in episodes:
        episode["mechanism_scores"]["artifact_quality_score"] = 0.0
        episode["retrieval_signals"] = {
            "used_expected_signal": episode["bucket"] != "baseline",
            "retrieval_before_first_update": episode["bucket"] != "baseline",
            "memory_read_count": 0,
            "memory_injection_count": 0,
            "session_search_count": 1 if episode["bucket"] != "baseline" else 0,
            "skill_read_count": 0,
            "retrieval_signal_count": 1 if episode["bucket"] != "baseline" else 0,
        }

    summary = summarize_sequence(sequence_name="test", variant="synthetic", episodes=episodes)

    assert summary["family_summary"]["F04_session_recall"]["attribution"]["label"] == "strong"


def test_infra_blocked_evaluation_episodes_do_not_drag_family_delta():
    episodes = [
        _episode("F04_session_recall", "baseline", 0.30, 0.0, False),
        _episode("F04_session_recall", "learn", 0.45, 0.8, True),
        _episode("F04_session_recall", "learn", 0.47, 0.8, True),
        {
            **_episode("F04_session_recall", "evaluation", 0.0, 0.0, False),
            "mechanism": "session_search",
            "expected_persistence_signal": "session_search",
            "infra_blocked": True,
        },
        {
            **_episode("F04_session_recall", "evaluation", 0.0, 0.0, False),
            "mechanism": "session_search",
            "expected_persistence_signal": "session_search",
            "infra_blocked": True,
        },
    ]

    summary = summarize_sequence(sequence_name="test", variant="synthetic", episodes=episodes)
    family = summary["family_summary"]["F04_session_recall"]

    assert family["bucket_summary"]["evaluation"]["episode_count"] == 0
    assert family["bucket_summary"]["evaluation"]["blocked_episode_count"] == 2
    assert family["improvement"]["infra_blocked_episodes"] == 2
    assert summary["benchmark_signal"]["families_with_infra_blocks"] == 1
