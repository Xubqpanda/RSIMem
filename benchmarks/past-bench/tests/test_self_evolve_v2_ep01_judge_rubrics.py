from pathlib import Path

import yaml


def test_ep01_tasks_define_judge_rubrics() -> None:
    base = Path("self-evolve-tasks-v2/memory_ability/EP01_prior_case_recall")
    task_files = sorted(base.glob("EP01_*/task.yaml"))
    assert task_files

    for task_file in task_files:
        data = yaml.safe_load(task_file.read_text(encoding="utf-8"))
        rubric = (data.get("judge_rubric") or "").strip()
        assert rubric, f"{task_file} is missing judge_rubric"
