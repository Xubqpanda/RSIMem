import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import yaml

from past_bench.cli import cmd_evolve


def _write_task(task_dir: Path, task_id: str, task_name: str) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.yaml").write_text(
        yaml.safe_dump(
            {
                "task_id": task_id,
                "task_name": task_name,
                "prompt": {"text": f"Run {task_name}"},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def _write_sequence(manifest_path: Path, *, reflection_enabled: bool) -> None:
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "name": f"f07_reflection_{'on' if reflection_enabled else 'off'}",
                "description": "Reflection control regression test.",
                "hermes": {
                    "reflection_enabled": reflection_enabled,
                    "skills_enabled": True,
                    "memory_enabled": False,
                    "session_search_enabled": False,
                },
                "episodes": [
                    {
                        "task": "tasks/F07_baseline",
                        "label": "F07 baseline",
                        "family_id": "F07_failure_to_rule",
                        "mechanism": "skill",
                        "bucket": "baseline",
                        "latent_rule_id": "f07_reflection_rule_v1",
                    },
                    {
                        "task": "tasks/F07_learn",
                        "label": "F07 learn",
                        "family_id": "F07_failure_to_rule",
                        "mechanism": "skill",
                        "bucket": "learn",
                        "latent_rule_id": "f07_reflection_rule_v1",
                        "reflection_required": True,
                    },
                    {
                        "task": "tasks/F07_eval",
                        "label": "F07 eval",
                        "family_id": "F07_failure_to_rule",
                        "mechanism": "skill",
                        "bucket": "evaluation",
                        "latent_rule_id": "f07_reflection_rule_v1",
                        "evaluation_requires_retrieval": True,
                    },
                ],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def _artifact_summary(*, has_skill: bool, reflection_write: bool) -> dict:
    skill_docs = {}
    if has_skill:
        skill_docs["f07-reflection-skill"] = {
            "sha1": "skill-sha-1",
            "content": "when_to_use: failure_to_rule\nsteps:\n- extract reusable rule\n",
        }

    return {
        "artifacts_dir": "unused",
        "memory_file_exists": False,
        "user_file_exists": False,
        "memory_chars": 0,
        "user_chars": 0,
        "memory_entries": [],
        "user_entries": [],
        "skill_count": 1 if has_skill else 0,
        "skill_names": ["f07-reflection-skill"] if has_skill else [],
        "skill_docs": skill_docs,
        "internal_tools": {
            "session_file": None,
            "tool_call_counts": {},
            "memory_calls": 0,
            "memory_action_counts": {},
            "memory_write_count": 0,
            "memory_read_count": 0,
            "skill_manage_calls": 1 if reflection_write else 0,
            "skill_manage_action_counts": {"create": 1} if reflection_write else {},
            "skill_create_count": 1 if reflection_write else 0,
            "skill_update_count": 0,
            "session_search_calls": 0,
            "skill_view_calls": 0,
            "skills_list_calls": 0,
            "skill_read_count": 0,
            "calls": [],
        },
    }


def test_cmd_evolve_reflection_ablation_changes_real_sequence_outcome(tmp_path: Path, monkeypatch):
    root = tmp_path / "self_evolve_cli"
    tasks_root = root / "tasks"
    _write_task(tasks_root / "F07_baseline", "F07_baseline", "F07 Baseline")
    _write_task(tasks_root / "F07_learn", "F07_learn", "F07 Learn")
    _write_task(tasks_root / "F07_eval", "F07_eval", "F07 Eval")

    manifest_on = root / "reflection_on.yaml"
    manifest_off = root / "reflection_off.yaml"
    _write_sequence(manifest_on, reflection_enabled=True)
    _write_sequence(manifest_off, reflection_enabled=False)

    state = {"has_skill": False, "last_kind": "attempt"}

    monkeypatch.setattr(
        "past_bench.config.load_config",
        lambda _path: SimpleNamespace(
            defaults=SimpleNamespace(trace_dir=str(root / "traces")),
            sandbox=SimpleNamespace(enabled=False),
        ),
    )
    monkeypatch.setattr("past_bench.cli._make_judge", lambda cfg, args: None)
    monkeypatch.setattr("past_bench.cli._resolve_runtime_mode", lambda args, cfg: "mock")
    monkeypatch.setattr("past_bench.cli._resolve_runtime_temperature", lambda args, cfg: None)
    monkeypatch.setattr("past_bench.cli._resolve_registry_path", lambda args, cfg: None)
    monkeypatch.setattr("past_bench.cli._check_agent_requirements", lambda *args, **kwargs: None)
    monkeypatch.setattr("past_bench.runner.services.ServiceManager", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr("past_bench.graders.registry.get_grader", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "past_bench.runner.self_evolve.build_reflection_prompt",
        lambda **kwargs: "Write a reusable skill.",
    )

    def fake_snapshot_hermes_home(_path: Path, *, include_contents: bool = True) -> dict:
        return _artifact_summary(has_skill=state["has_skill"], reflection_write=False)

    def fake_snapshot_hermes_artifacts(_path: Path) -> dict:
        return _artifact_summary(
            has_skill=state["has_skill"],
            reflection_write=state["last_kind"] == "reflection",
        )

    def fake_execute_trial(**kwargs):
        task = kwargs["task"]
        trace_dir = Path(kwargs["trace_dir"])
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / "trace.jsonl"
        trace_path.write_text("{}", encoding="utf-8")
        if task.task_id.endswith("_REFLECT"):
            state["has_skill"] = True
            state["last_kind"] = "reflection"
        else:
            state["last_kind"] = "attempt"
        return trace_path, None

    def fake_grade_episode(*, trace_path, task, artifact_before, artifact_summary, **kwargs):
        if task.task_id == "F07_baseline":
            score = 0.25
        elif task.task_id == "F07_learn":
            score = 0.35
        else:
            score = 0.85 if artifact_before["skill_count"] > 0 else 0.25

        used_skill = artifact_before["skill_count"] > 0
        return {
            "trace": str(trace_path),
            "trace_id": "trace-mock",
            "task_id": task.task_id,
            "task_name": task.task_name,
            "final_response_text": "",
            "scores": {},
            "task_score": score,
            "passed": score >= 0.7,
            "total_turns": 1,
            "tool_dispatch_count": 0,
            "token_usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "timing": {
                "wall_time_s": 0.0,
                "model_time_s": 0.0,
                "tool_time_s": 0.0,
                "other_time_s": 0.0,
            },
            "artifacts": artifact_summary,
            "artifact_diff": {},
            "retrieval_signals": {
                "expected_signal": "skill",
                "memory_read_count": 0,
                "memory_injection_count": 0,
                "skill_read_count": 1 if used_skill else 0,
                "session_search_count": 0,
                "retrieval_signal_count": 1 if used_skill else 0,
                "used_expected_signal": used_skill,
                "retrieval_before_first_update": used_skill,
                "retrieval_before_final_response": used_skill,
                "first_retrieval_at": "session_start" if used_skill else None,
                "first_write_at": None,
                "first_write_endpoint": None,
            },
            "mechanism_scores": {
                "mechanism_confidence": 0.9 if used_skill else 0.0,
                "artifact_quality_score": 0.9 if used_skill else 0.0,
                "transfer_quality": 1.0 if used_skill else 0.0,
                "shortcut_resistance": 1.0,
            },
            "internal_tools": artifact_summary["internal_tools"],
            "env_snapshot_present": False,
        }

    def fake_summarize_reflection_episode(*, trace_path, artifact_summary, task_id, label):
        return {
            "trace": str(trace_path),
            "trace_id": "trace-reflection",
            "task_id": task_id,
            "task_name": label,
            "final_response_text": "Saved reusable skill.",
            "episode_kind": "reflection",
            "bucket": "reflection",
            "task_score": 0.0,
            "passed": False,
            "total_turns": 1,
            "tool_dispatch_count": 0,
            "token_usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "timing": {
                "wall_time_s": 0.0,
                "model_time_s": 0.0,
                "tool_time_s": 0.0,
                "other_time_s": 0.0,
            },
            "artifacts": artifact_summary,
            "artifact_diff": {},
            "retrieval_signals": {},
            "mechanism_scores": {},
            "internal_tools": artifact_summary["internal_tools"],
            "env_snapshot_present": False,
        }

    monkeypatch.setattr("past_bench.runner.self_evolve.snapshot_hermes_home", fake_snapshot_hermes_home)
    monkeypatch.setattr("past_bench.runner.self_evolve.snapshot_hermes_artifacts", fake_snapshot_hermes_artifacts)
    monkeypatch.setattr("past_bench.cli._execute_trial", fake_execute_trial)
    monkeypatch.setattr("past_bench.runner.self_evolve.grade_episode", fake_grade_episode)
    monkeypatch.setattr(
        "past_bench.runner.self_evolve.summarize_reflection_episode",
        fake_summarize_reflection_episode,
    )

    def run_sequence(manifest_path: Path, trace_dir: Path) -> tuple[dict, dict]:
        state["has_skill"] = False
        state["last_kind"] = "attempt"
        args = argparse.Namespace(
            config=str(root / "dummy_config.yaml"),
            sequence=str(manifest_path),
            trace_dir=str(trace_dir),
            compare_no_persistence=False,
            agent="hermes",
            agent_profile=None,
            model="mock-model",
            api_key=None,
            base_url=None,
            sandbox=False,
            sandbox_image=None,
            sandbox_tools=False,
            runtime_image=None,
            background_review_wait_s=None,
        )
        cmd_evolve(args)
        return (
            json.loads((trace_dir / "sequence_summary.json").read_text(encoding="utf-8")),
            json.loads((trace_dir / "sequence_results.json").read_text(encoding="utf-8")),
        )

    summary_on, results_on = run_sequence(manifest_on, root / "traces_on")
    summary_off, results_off = run_sequence(manifest_off, root / "traces_off")

    assert summary_on["bucket_summary"]["reflection"]["episode_count"] == 1
    assert summary_on["bucket_summary"]["evaluation"]["avg_task_score"] == 0.85
    assert any(ep.get("bucket") == "reflection" for ep in results_on["episodes"])

    assert "reflection" not in summary_off["bucket_summary"]
    assert summary_off["bucket_summary"]["evaluation"]["avg_task_score"] == 0.25
    assert all(ep.get("bucket") != "reflection" for ep in results_off["episodes"])
