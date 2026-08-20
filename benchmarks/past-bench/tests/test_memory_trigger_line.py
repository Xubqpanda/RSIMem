import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
def test_memory_trigger_tasks_use_normalized_task_ids_and_baseline_bucket_role():
    task_root = ROOT / "self-evolve-tasks" / "memory_trigger"
    for task_yaml in sorted(task_root.glob("*/task.yaml")):
        task = yaml.safe_load(task_yaml.read_text(encoding="utf-8"))
        expectations = json.loads((task_yaml.parent / "expectations.json").read_text(encoding="utf-8"))

        assert task["task_id"].startswith("SE_")
        assert expectations["bucket_role"] != "cold"


def test_memory_trigger_tasks_use_explicit_final_answer_and_source_note_contract():
    task_root = ROOT / "self-evolve-tasks" / "memory_trigger"
    for task_yaml in sorted(task_root.glob("*/task.yaml")):
        task = yaml.safe_load(task_yaml.read_text(encoding="utf-8"))
        prompt = task["prompt"]["text"].lower()

        assert "final answer" in prompt
        assert "share tool" in prompt
        assert "source note" in prompt


def test_f06_evaluation_fixtures_do_not_leak_canonical_rule_tokens():
    fixture_paths = [
        ROOT / "self-evolve-tasks" / "_shared" / "fixtures" / "f06_notes" / "conflict_eval_near.json",
        ROOT / "self-evolve-tasks" / "_shared" / "fixtures" / "f06_notes" / "conflict_eval_far.json",
    ]

    banned_snippets = [
        "risk high",
        "risk medium",
        "risk low",
        "not part of internal execution",
        "share only with internal attendees",
    ]

    for fixture_path in fixture_paths:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        content = "\n".join(item["content"].lower() for item in payload)
        for snippet in banned_snippets:
            assert snippet not in content, f"{fixture_path.name} leaked {snippet!r}"
