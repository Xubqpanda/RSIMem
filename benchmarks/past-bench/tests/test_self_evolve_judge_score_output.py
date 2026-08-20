from pathlib import Path


def test_grade_episode_writes_judge_score_field() -> None:
    source = Path("src/past_bench/runner/self_evolve.py").read_text(encoding="utf-8")
    assert '"judge_score": getattr(scores, "communication", 0.0),' in source
