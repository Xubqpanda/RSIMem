import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
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


def _write_sequence(manifest_path: Path) -> None:
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "name": "self_evolve_cli_regression",
                "description": "CLI regression test manifest.",
                "episodes": [
                    {
                        "task": "tasks/F01_baseline",
                        "label": "F01 baseline",
                        "family_id": "F01_skill_bootstrap",
                        "mechanism": "skill",
                        "bucket": "baseline",
                        "latent_rule_id": "f01_rule_v1",
                        "expected_persistence_signal": "skill",
                    }
                ],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def _write_two_family_sequence(manifest_path: Path) -> None:
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "name": "self_evolve_family_reset",
                "description": "Ensure family baselines reset Hermes home.",
                "episodes": [
                    {
                        "task": "tasks/F01_baseline",
                        "label": "F01 baseline",
                        "family_id": "F01_skill_bootstrap",
                        "mechanism": "skill",
                        "bucket": "baseline",
                        "latent_rule_id": "f01_rule_v1",
                        "expected_persistence_signal": "skill",
                        "history_mode": "fresh",
                    },
                    {
                        "task": "tasks/F02_baseline",
                        "label": "F02 baseline",
                        "family_id": "F02_skill_patch",
                        "mechanism": "skill",
                        "bucket": "baseline",
                        "latent_rule_id": "f02_rule_v1",
                        "expected_persistence_signal": "skill",
                        "history_mode": "fresh",
                    },
                ],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def _write_history_sequence(manifest_path: Path, episodes: list[dict], *, name: str = "history_branching") -> None:
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "name": name,
                "description": "History branching regression manifest.",
                "episodes": episodes,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def _base_args(root: Path, manifest_path: Path, trace_dir: Path, *, agent: str, runtime: str) -> argparse.Namespace:
    return argparse.Namespace(
        config=str(root / "dummy_config.yaml"),
        sequence=str(manifest_path),
        trace_dir=str(trace_dir),
        compare_no_persistence=False,
        agent=agent,
        agent_profile=None,
        model="mock-model",
        api_key=None,
        base_url=None,
        sandbox=False,
        sandbox_image=None,
        sandbox_tools=False,
        runtime=runtime,
        runtime_image=None,
        background_review_wait_s=None,
        temperature=None,
        registry=None,
    )


def test_cmd_evolve_rejects_hermes_container_episode_mode(tmp_path: Path, monkeypatch):
    root = tmp_path / "self_evolve_guard"
    tasks_root = root / "tasks"
    _write_task(tasks_root / "F01_baseline", "F01_baseline", "F01 Baseline")
    manifest = root / "sequence.yaml"
    _write_sequence(manifest)

    monkeypatch.setattr(
        "past_bench.config.load_config",
        lambda _path: SimpleNamespace(
            defaults=SimpleNamespace(trace_dir=str(root / "traces"), agent_registry=None),
            sandbox=SimpleNamespace(enabled=False),
            runtime=SimpleNamespace(mode="local", registry_path=None, temperature=0.0),
        ),
    )
    monkeypatch.setattr("past_bench.cli._make_judge", lambda cfg, args: None)
    monkeypatch.setattr("past_bench.cli._check_agent_requirements", lambda *args, **kwargs: None)

    args = _base_args(root, manifest, root / "traces", agent="hermes", runtime="container")

    with pytest.raises(SystemExit, match="runtime container"):
        cmd_evolve(args)


def test_cmd_evolve_backfills_trace_end_with_graded_score(tmp_path: Path, monkeypatch):
    root = tmp_path / "self_evolve_trace_sync"
    tasks_root = root / "tasks"
    _write_task(tasks_root / "F01_baseline", "F01_baseline", "F01 Baseline")
    manifest = root / "sequence.yaml"
    _write_sequence(manifest)

    monkeypatch.setattr(
        "past_bench.config.load_config",
        lambda _path: SimpleNamespace(
            defaults=SimpleNamespace(trace_dir=str(root / "traces"), agent_registry=None),
            sandbox=SimpleNamespace(enabled=False),
            runtime=SimpleNamespace(mode="local", registry_path=None, temperature=0.0),
        ),
    )
    monkeypatch.setattr("past_bench.cli._make_judge", lambda cfg, args: None)
    monkeypatch.setattr("past_bench.cli._check_agent_requirements", lambda *args, **kwargs: None)
    monkeypatch.setattr("past_bench.runner.services.ServiceManager", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr("past_bench.graders.registry.get_grader", lambda *args, **kwargs: object())

    def fake_execute_trial(**kwargs):
        trace_dir = Path(kwargs["trace_dir"])
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / "trace.jsonl"
        events = [
            {
                "type": "trace_start",
                "trace_id": "trace-sync",
                "task_id": "F01_baseline",
                "model": "mock-model",
                "timestamp": "2026-04-06T00:00:00Z",
            },
            {
                "type": "trace_end",
                "trace_id": "trace-sync",
                "total_turns": 1,
                "model_input_tokens": 0,
                "model_output_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "model_time_s": 0.0,
                "tool_time_s": 0.0,
                "other_time_s": 0.0,
                "wall_time_s": 0.0,
                "scores": {
                    "completion": 0.0,
                    "robustness": 0.0,
                    "communication": 0.0,
                    "safety": 1.0,
                    "efficiency_turns": 0,
                    "efficiency_tokens": 0,
                    "efficiency_wall_time_s": 0.0,
                },
                "task_score": 0.0,
                "passed": False,
                "failure_modes": [],
                "timestamp": "2026-04-06T00:00:01Z",
            },
        ]
        trace_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
        return trace_path, None

    def fake_grade_episode(**kwargs):
        return {
            "trace": str(kwargs["trace_path"]),
            "trace_id": "trace-sync",
            "task_id": "F01_baseline",
            "task_name": "F01 Baseline",
            "final_response_text": "",
            "scores": {
                "completion": 0.8,
                "robustness": 0.9,
                "communication": 0.0,
                "safety": 1.0,
                "efficiency_turns": 1,
                "efficiency_tokens": 0,
                "efficiency_wall_time_s": 0.0,
            },
            "task_score": 0.82,
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
            "artifacts": {
                "memory_file_exists": False,
                "user_file_exists": False,
                "memory_chars": 0,
                "user_chars": 0,
                "memory_entries": [],
                "user_entries": [],
                "skill_count": 0,
                "skill_names": [],
                "skill_docs": {},
                "internal_tools": {
                    "memory_calls": 0,
                    "skill_manage_calls": 0,
                    "session_search_calls": 0,
                    "memory_write_count": 0,
                    "memory_read_count": 0,
                    "skill_create_count": 0,
                    "skill_update_count": 0,
                    "skill_read_count": 0,
                    "calls": [],
                },
            },
            "artifact_diff": {},
            "retrieval_signals": {},
            "mechanism_scores": {},
            "internal_tools": {
                "memory_calls": 0,
                "skill_manage_calls": 0,
                "session_search_calls": 0,
                "memory_write_count": 0,
                "memory_read_count": 0,
                "skill_create_count": 0,
                "skill_update_count": 0,
                "skill_read_count": 0,
                "calls": [],
            },
            "env_snapshot_present": False,
        }

    monkeypatch.setattr("past_bench.cli._execute_trial", fake_execute_trial)
    monkeypatch.setattr("past_bench.runner.self_evolve.grade_episode", fake_grade_episode)

    trace_dir = root / "traces"
    args = _base_args(root, manifest, trace_dir, agent="mock-agent", runtime="local")
    cmd_evolve(args)

    trace_path = trace_dir / "01_f01_baseline" / "trace.jsonl"
    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trace_end = next(event for event in events if event["type"] == "trace_end")

    assert trace_end["task_score"] == 0.82
    assert trace_end["scores"]["completion"] == 0.8
    assert trace_end["passed"] is False
    assert events[-1]["type"] == "grading_result"
    assert events[-1]["task_score"] == 0.82


def test_cmd_evolve_summary_prints_task_breakdown(tmp_path: Path, monkeypatch, capsys):
    root = tmp_path / "self_evolve_task_breakdown"
    tasks_root = root / "tasks"
    _write_task(tasks_root / "F01_baseline", "F01_baseline", "F01 Baseline")
    manifest = root / "sequence.yaml"
    _write_sequence(manifest)

    monkeypatch.setattr(
        "past_bench.config.load_config",
        lambda _path: SimpleNamespace(
            defaults=SimpleNamespace(trace_dir=str(root / "traces"), agent_registry=None),
            sandbox=SimpleNamespace(enabled=False),
            runtime=SimpleNamespace(mode="local", registry_path=None, temperature=0.0),
        ),
    )
    monkeypatch.setattr("past_bench.cli._make_judge", lambda cfg, args: None)
    monkeypatch.setattr("past_bench.cli._check_agent_requirements", lambda *args, **kwargs: None)
    monkeypatch.setattr("past_bench.runner.services.ServiceManager", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr("past_bench.graders.registry.get_grader", lambda *args, **kwargs: object())

    def fake_execute_trial(**kwargs):
        trace_dir = Path(kwargs["trace_dir"])
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / "trace.jsonl"
        trace_path.write_text("{}", encoding="utf-8")
        return trace_path, None

    def fake_grade_episode(**kwargs):
        return {
            "trace": str(kwargs["trace_path"]),
            "trace_id": "trace-breakdown",
            "task_id": "F01_baseline",
            "task_name": "F01 Baseline",
            "final_response_text": "",
            "scores": {
                "completion": 0.8,
                "robustness": 0.9,
                "communication": 0.0,
                "safety": 1.0,
                "efficiency_turns": 1,
                "efficiency_tokens": 0,
                "efficiency_wall_time_s": 0.0,
            },
            "task_score": 0.82,
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
            "artifacts": {
                "memory_file_exists": False,
                "user_file_exists": False,
                "memory_chars": 0,
                "user_chars": 0,
                "memory_entries": [],
                "user_entries": [],
                "skill_count": 0,
                "skill_names": [],
                "skill_docs": {},
                "internal_tools": {
                    "memory_calls": 0,
                    "skill_manage_calls": 0,
                    "session_search_calls": 0,
                    "memory_write_count": 0,
                    "memory_read_count": 0,
                    "skill_create_count": 0,
                    "skill_update_count": 0,
                    "skill_read_count": 0,
                    "calls": [],
                },
            },
            "artifact_diff": {},
            "retrieval_signals": {},
            "mechanism_scores": {},
            "internal_tools": {
                "memory_calls": 0,
                "skill_manage_calls": 0,
                "session_search_calls": 0,
                "memory_write_count": 0,
                "memory_read_count": 0,
                "skill_create_count": 0,
                "skill_update_count": 0,
                "skill_read_count": 0,
                "calls": [],
            },
            "env_snapshot_present": False,
        }

    monkeypatch.setattr("past_bench.cli._execute_trial", fake_execute_trial)
    monkeypatch.setattr("past_bench.runner.self_evolve.grade_episode", fake_grade_episode)

    trace_dir = root / "traces"
    args = _base_args(root, manifest, trace_dir, agent="mock-agent", runtime="local")
    cmd_evolve(args)

    out = capsys.readouterr().out
    assert "Summary [with_persistence]" in out
    assert "task_breakdown:" in out
    assert "task_id=F01_baseline" in out
    assert "score=0.820" in out


def test_cmd_evolve_resets_hermes_home_for_each_family_baseline(tmp_path: Path, monkeypatch):
    root = tmp_path / "self_evolve_family_reset"
    tasks_root = root / "tasks"
    _write_task(tasks_root / "F01_baseline", "F01_baseline", "F01 Baseline")
    _write_task(tasks_root / "F02_baseline", "F02_baseline", "F02 Baseline")
    manifest = root / "sequence.yaml"
    _write_two_family_sequence(manifest)

    monkeypatch.setattr(
        "past_bench.config.load_config",
        lambda _path: SimpleNamespace(
            defaults=SimpleNamespace(trace_dir=str(root / "traces"), agent_registry=None),
            sandbox=SimpleNamespace(enabled=False),
            runtime=SimpleNamespace(mode="local", registry_path=None, temperature=0.0),
        ),
    )
    monkeypatch.setattr("past_bench.cli._make_judge", lambda cfg, args: None)
    monkeypatch.setattr("past_bench.cli._check_agent_requirements", lambda *args, **kwargs: None)
    monkeypatch.setattr("past_bench.runner.services.ServiceManager", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr("past_bench.graders.registry.get_grader", lambda *args, **kwargs: object())

    reset_calls: list[str] = []

    def fake_reset_runtime_dir(path: Path):
        reset_calls.append(str(path))
        path.mkdir(parents=True, exist_ok=True)

    def fake_snapshot_hermes_home(_path: Path, *, include_contents: bool = True) -> dict:
        return {
            "artifacts_dir": "unused",
            "memory_file_exists": False,
            "user_file_exists": False,
            "memory_chars": 0,
            "user_chars": 0,
            "memory_entries": [],
            "user_entries": [],
            "skill_count": 0,
            "skill_names": [],
            "skill_docs": {},
            "internal_tools": {
                "session_file": None,
                "tool_call_counts": {},
                "memory_calls": 0,
                "memory_action_counts": {},
                "memory_write_count": 0,
                "memory_read_count": 0,
                "skill_manage_calls": 0,
                "skill_manage_action_counts": {},
                "skill_create_count": 0,
                "skill_update_count": 0,
                "session_search_calls": 0,
                "skill_view_calls": 0,
                "skills_list_calls": 0,
                "skill_read_count": 0,
                "calls": [],
            },
        }

    def fake_snapshot_hermes_artifacts(_path: Path) -> dict:
        return fake_snapshot_hermes_home(_path)

    def fake_execute_trial(**kwargs):
        trace_dir = Path(kwargs["trace_dir"])
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / "trace.jsonl"
        trace_path.write_text("{}", encoding="utf-8")
        return trace_path, None

    def fake_grade_episode(**kwargs):
        task = kwargs["task"]
        return {
            "trace": str(kwargs["trace_path"]),
            "trace_id": f"{task.task_id}-trace",
            "task_id": task.task_id,
            "task_name": task.task_name,
            "final_response_text": "",
            "scores": {
                "completion": 0.0,
                "robustness": 0.0,
                "communication": 0.0,
                "safety": 1.0,
                "efficiency_turns": 0,
                "efficiency_tokens": 0,
                "efficiency_wall_time_s": 0.0,
            },
            "task_score": 0.0,
            "passed": False,
            "total_turns": 1,
            "tool_dispatch_count": 0,
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "timing": {"wall_time_s": 0.0, "model_time_s": 0.0, "tool_time_s": 0.0, "other_time_s": 0.0},
            "artifacts": fake_snapshot_hermes_artifacts(Path(".")),
            "artifact_diff": {},
            "retrieval_signals": {},
            "mechanism_scores": {},
            "internal_tools": fake_snapshot_hermes_artifacts(Path("."))["internal_tools"],
            "env_snapshot_present": False,
        }

    monkeypatch.setattr("past_bench.cli._reset_runtime_dir", fake_reset_runtime_dir)
    monkeypatch.setattr("past_bench.runner.self_evolve.snapshot_hermes_home", fake_snapshot_hermes_home)
    monkeypatch.setattr("past_bench.runner.self_evolve.snapshot_hermes_artifacts", fake_snapshot_hermes_artifacts)
    monkeypatch.setattr("past_bench.cli._execute_trial", fake_execute_trial)
    monkeypatch.setattr("past_bench.runner.self_evolve.grade_episode", fake_grade_episode)

    trace_dir = root / "traces"
    args = _base_args(root, manifest, trace_dir, agent="hermes", runtime="local")
    cmd_evolve(args)

    assert str(trace_dir / "family_homes") in reset_calls
    assert str(trace_dir / "family_homes" / "F01_skill_bootstrap" / "hermes_home") in reset_calls
    assert str(trace_dir / "family_homes" / "F02_skill_patch" / "hermes_home") in reset_calls


def test_cmd_evolve_branches_history_from_anchor_without_leakage(tmp_path: Path, monkeypatch):
    root = tmp_path / "self_evolve_history_branching"
    tasks_root = root / "tasks"
    for task_id in ("I01", "I02", "I03", "I04", "I05"):
        _write_task(tasks_root / task_id, task_id, task_id)

    manifest = root / "sequence.yaml"
    _write_history_sequence(
        manifest,
        [
            {
                "task": "tasks/I01",
                "label": "I01",
                "family_id": "EP01_prior_case_recall",
                "mechanism": "session_search",
                "bucket": "baseline",
                "latent_rule_id": "ep01_rule_v1",
                "expected_persistence_signal": "session_search",
                "requires_fresh_session": False,
                "history_mode": "continue",
                "reflection_required": False,
            },
            {
                "task": "tasks/I02",
                "label": "I02",
                "family_id": "EP01_prior_case_recall",
                "mechanism": "session_search",
                "bucket": "learn",
                "latent_rule_id": "ep01_rule_v1",
                "expected_persistence_signal": "session_search",
                "requires_fresh_session": False,
                "history_mode": "continue",
                "reflection_required": False,
            },
            {
                "task": "tasks/I03",
                "label": "I03",
                "family_id": "EP01_prior_case_recall",
                "mechanism": "session_search",
                "bucket": "learn",
                "latent_rule_id": "ep01_rule_v1",
                "expected_persistence_signal": "session_search",
                "requires_fresh_session": False,
                "history_mode": "continue",
                "history_save_anchor": "ep01_post_learn",
                "reflection_required": False,
            },
            {
                "task": "tasks/I04",
                "label": "I04",
                "family_id": "EP01_prior_case_recall",
                "mechanism": "session_search",
                "bucket": "evaluation",
                "latent_rule_id": "ep01_rule_v1",
                "expected_persistence_signal": "session_search",
                "requires_fresh_session": True,
                "history_mode": "from_anchor",
                "history_load_anchor": "ep01_post_learn",
            },
            {
                "task": "tasks/I05",
                "label": "I05",
                "family_id": "EP01_prior_case_recall",
                "mechanism": "session_search",
                "bucket": "evaluation",
                "latent_rule_id": "ep01_rule_v1",
                "expected_persistence_signal": "session_search",
                "requires_fresh_session": True,
                "history_mode": "from_anchor",
                "history_load_anchor": "ep01_post_learn",
            },
        ],
    )

    monkeypatch.setattr(
        "past_bench.config.load_config",
        lambda _path: SimpleNamespace(
            defaults=SimpleNamespace(trace_dir=str(root / "traces"), agent_registry=None),
            sandbox=SimpleNamespace(enabled=False),
            runtime=SimpleNamespace(mode="local", registry_path=None, temperature=0.0),
        ),
    )
    monkeypatch.setattr("past_bench.cli._make_judge", lambda cfg, args: None)
    monkeypatch.setattr("past_bench.cli._check_agent_requirements", lambda *args, **kwargs: None)
    monkeypatch.setattr("past_bench.runner.services.ServiceManager", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr("past_bench.graders.registry.get_grader", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "past_bench.runner.self_evolve.build_hermes_extra_body",
        lambda **kwargs: {"home_dir": str(kwargs["home_dir"])},
    )

    leakage_checks: list[bool] = []

    def fake_snapshot_hermes_home(_path: Path, *, include_contents: bool = True) -> dict:
        return {
            "artifacts_dir": "unused",
            "memory_file_exists": False,
            "user_file_exists": False,
            "memory_chars": 0,
            "user_chars": 0,
            "memory_entries": [],
            "user_entries": [],
            "skill_count": 0,
            "skill_names": [],
            "skill_docs": {},
            "internal_tools": {
                "session_file": None,
                "tool_call_counts": {},
                "memory_calls": 0,
                "memory_action_counts": {},
                "memory_write_count": 0,
                "memory_read_count": 0,
                "skill_manage_calls": 0,
                "skill_manage_action_counts": {},
                "skill_create_count": 0,
                "skill_update_count": 0,
                "session_search_calls": 0,
                "skill_view_calls": 0,
                "skills_list_calls": 0,
                "skill_read_count": 0,
                "calls": [],
            },
        }

    def fake_execute_trial(**kwargs):
        trace_dir = Path(kwargs["trace_dir"])
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / "trace.jsonl"
        trace_path.write_text("{}", encoding="utf-8")

        home_dir = Path(kwargs["model_extra_body_override"]["home_dir"])
        home_dir.mkdir(parents=True, exist_ok=True)
        task_id = kwargs["task"].task_id
        if task_id == "I04":
            (home_dir / "branch_only.txt").write_text("i04", encoding="utf-8")
        if task_id == "I05":
            leakage_checks.append((home_dir / "branch_only.txt").exists())
        return trace_path, None

    def fake_grade_episode(**kwargs):
        task = kwargs["task"]
        return {
            "trace": str(kwargs["trace_path"]),
            "trace_id": f"{task.task_id}-trace",
            "task_id": task.task_id,
            "task_name": task.task_name,
            "final_response_text": "",
            "scores": {
                "completion": 0.0,
                "robustness": 0.0,
                "communication": 0.0,
                "safety": 1.0,
                "efficiency_turns": 0,
                "efficiency_tokens": 0,
                "efficiency_wall_time_s": 0.0,
            },
            "task_score": 0.0,
            "passed": False,
            "total_turns": 1,
            "tool_dispatch_count": 0,
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "timing": {"wall_time_s": 0.0, "model_time_s": 0.0, "tool_time_s": 0.0, "other_time_s": 0.0},
            "artifacts": fake_snapshot_hermes_home(Path(".")),
            "artifact_diff": {},
            "retrieval_signals": {},
            "mechanism_scores": {},
            "internal_tools": fake_snapshot_hermes_home(Path("."))["internal_tools"],
            "env_snapshot_present": False,
        }

    monkeypatch.setattr("past_bench.runner.self_evolve.snapshot_hermes_home", fake_snapshot_hermes_home)
    monkeypatch.setattr("past_bench.runner.self_evolve.snapshot_hermes_artifacts", fake_snapshot_hermes_home)
    monkeypatch.setattr("past_bench.cli._execute_trial", fake_execute_trial)
    monkeypatch.setattr("past_bench.runner.self_evolve.grade_episode", fake_grade_episode)

    args = _base_args(root, manifest, root / "traces", agent="hermes", runtime="local")
    cmd_evolve(args)

    assert leakage_checks == [False]


def test_cmd_evolve_errors_when_history_anchor_is_missing(tmp_path: Path, monkeypatch):
    root = tmp_path / "self_evolve_missing_anchor"
    tasks_root = root / "tasks"
    _write_task(tasks_root / "I04", "I04", "I04")

    manifest = root / "sequence.yaml"
    _write_history_sequence(
        manifest,
        [
            {
                "task": "tasks/I04",
                "label": "I04",
                "family_id": "EP01_prior_case_recall",
                "mechanism": "session_search",
                "bucket": "evaluation",
                "latent_rule_id": "ep01_rule_v1",
                "expected_persistence_signal": "session_search",
                "requires_fresh_session": True,
                "history_mode": "from_anchor",
                "history_load_anchor": "missing_anchor",
            }
        ],
        name="missing_anchor",
    )

    monkeypatch.setattr(
        "past_bench.config.load_config",
        lambda _path: SimpleNamespace(
            defaults=SimpleNamespace(trace_dir=str(root / "traces"), agent_registry=None),
            sandbox=SimpleNamespace(enabled=False),
            runtime=SimpleNamespace(mode="local", registry_path=None, temperature=0.0),
        ),
    )
    monkeypatch.setattr("past_bench.cli._make_judge", lambda cfg, args: None)
    monkeypatch.setattr("past_bench.cli._check_agent_requirements", lambda *args, **kwargs: None)

    args = _base_args(root, manifest, root / "traces", agent="hermes", runtime="local")
    with pytest.raises(RuntimeError, match="missing history anchor 'missing_anchor'"):
        cmd_evolve(args)


def test_cmd_evolve_keeps_history_anchors_variant_local(tmp_path: Path, monkeypatch):
    root = tmp_path / "self_evolve_variant_local_anchor"
    tasks_root = root / "tasks"
    for task_id in ("I03", "I04"):
        _write_task(tasks_root / task_id, task_id, task_id)

    manifest = root / "sequence.yaml"
    _write_history_sequence(
        manifest,
        [
            {
                "task": "tasks/I03",
                "label": "I03",
                "family_id": "EP01_prior_case_recall",
                "mechanism": "session_search",
                "bucket": "learn",
                "latent_rule_id": "ep01_rule_v1",
                "expected_persistence_signal": "session_search",
                "requires_fresh_session": False,
                "history_mode": "continue",
                "history_save_anchor": "ep01_post_learn",
                "reflection_required": False,
            },
            {
                "task": "tasks/I04",
                "label": "I04",
                "family_id": "EP01_prior_case_recall",
                "mechanism": "session_search",
                "bucket": "evaluation",
                "latent_rule_id": "ep01_rule_v1",
                "expected_persistence_signal": "session_search",
                "requires_fresh_session": True,
                "history_mode": "from_anchor",
                "history_load_anchor": "ep01_post_learn",
            },
        ],
        name="variant_local_anchor",
    )

    monkeypatch.setattr(
        "past_bench.config.load_config",
        lambda _path: SimpleNamespace(
            defaults=SimpleNamespace(trace_dir=str(root / "traces"), agent_registry=None),
            sandbox=SimpleNamespace(enabled=False),
            runtime=SimpleNamespace(mode="local", registry_path=None, temperature=0.0),
        ),
    )
    monkeypatch.setattr("past_bench.cli._make_judge", lambda cfg, args: None)
    monkeypatch.setattr("past_bench.cli._check_agent_requirements", lambda *args, **kwargs: None)
    monkeypatch.setattr("past_bench.runner.services.ServiceManager", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr("past_bench.graders.registry.get_grader", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "past_bench.runner.self_evolve.build_hermes_extra_body",
        lambda **kwargs: {"home_dir": str(kwargs["home_dir"])},
    )

    observed_markers: list[tuple[str, str]] = []

    def fake_snapshot_hermes_home(_path: Path, *, include_contents: bool = True) -> dict:
        return {
            "artifacts_dir": "unused",
            "memory_file_exists": False,
            "user_file_exists": False,
            "memory_chars": 0,
            "user_chars": 0,
            "memory_entries": [],
            "user_entries": [],
            "skill_count": 0,
            "skill_names": [],
            "skill_docs": {},
            "internal_tools": {
                "session_file": None,
                "tool_call_counts": {},
                "memory_calls": 0,
                "memory_action_counts": {},
                "memory_write_count": 0,
                "memory_read_count": 0,
                "skill_manage_calls": 0,
                "skill_manage_action_counts": {},
                "skill_create_count": 0,
                "skill_update_count": 0,
                "session_search_calls": 0,
                "skill_view_calls": 0,
                "skills_list_calls": 0,
                "skill_read_count": 0,
                "calls": [],
            },
        }

    def fake_execute_trial(**kwargs):
        trace_dir = Path(kwargs["trace_dir"])
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / "trace.jsonl"
        trace_path.write_text("{}", encoding="utf-8")

        home_dir = Path(kwargs["model_extra_body_override"]["home_dir"])
        home_dir.mkdir(parents=True, exist_ok=True)
        task_id = kwargs["task"].task_id
        variant_label = home_dir.parents[2].name
        if task_id == "I03":
            marker = "with" if variant_label == "with_persistence" else "without"
            (home_dir / "variant_marker.txt").write_text(marker, encoding="utf-8")
        if task_id == "I04":
            observed_markers.append(
                (variant_label, (home_dir / "variant_marker.txt").read_text(encoding="utf-8"))
            )
        return trace_path, None

    def fake_grade_episode(**kwargs):
        task = kwargs["task"]
        return {
            "trace": str(kwargs["trace_path"]),
            "trace_id": f"{task.task_id}-trace",
            "task_id": task.task_id,
            "task_name": task.task_name,
            "final_response_text": "",
            "scores": {
                "completion": 0.0,
                "robustness": 0.0,
                "communication": 0.0,
                "safety": 1.0,
                "efficiency_turns": 0,
                "efficiency_tokens": 0,
                "efficiency_wall_time_s": 0.0,
            },
            "task_score": 0.0,
            "passed": False,
            "total_turns": 1,
            "tool_dispatch_count": 0,
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "timing": {"wall_time_s": 0.0, "model_time_s": 0.0, "tool_time_s": 0.0, "other_time_s": 0.0},
            "artifacts": fake_snapshot_hermes_home(Path(".")),
            "artifact_diff": {},
            "retrieval_signals": {},
            "mechanism_scores": {},
            "internal_tools": fake_snapshot_hermes_home(Path("."))["internal_tools"],
            "env_snapshot_present": False,
        }

    monkeypatch.setattr("past_bench.runner.self_evolve.snapshot_hermes_home", fake_snapshot_hermes_home)
    monkeypatch.setattr("past_bench.runner.self_evolve.snapshot_hermes_artifacts", fake_snapshot_hermes_home)
    monkeypatch.setattr("past_bench.cli._execute_trial", fake_execute_trial)
    monkeypatch.setattr("past_bench.runner.self_evolve.grade_episode", fake_grade_episode)

    args = _base_args(root, manifest, root / "traces", agent="hermes", runtime="local")
    args.compare_no_persistence = True
    cmd_evolve(args)

    assert observed_markers == [
        ("with_persistence", "with"),
        ("without_persistence", "without"),
    ]


def test_cmd_evolve_separates_compare_no_persistence_variants_for_nanobot(tmp_path: Path, monkeypatch):
    root = tmp_path / "self_evolve_nanobot_compare"
    tasks_root = root / "tasks"
    for task_id in ("I03", "I04"):
        _write_task(tasks_root / task_id, task_id, task_id)

    manifest = root / "sequence.yaml"
    _write_history_sequence(
        manifest,
        [
            {
                "task": "tasks/I03",
                "label": "I03",
                "family_id": "SM01_preference_adoption",
                "mechanism": "memory",
                "bucket": "learn",
                "latent_rule_id": "sm01_rule_v1",
                "expected_persistence_signal": "memory",
                "requires_fresh_session": False,
                "history_mode": "continue",
                "history_save_anchor": "sm01_post_learn",
                "reflection_required": False,
            },
            {
                "task": "tasks/I04",
                "label": "I04",
                "family_id": "SM01_preference_adoption",
                "mechanism": "memory",
                "bucket": "evaluation",
                "latent_rule_id": "sm01_rule_v1",
                "expected_persistence_signal": "memory",
                "requires_fresh_session": True,
                "history_mode": "from_anchor",
                "history_load_anchor": "sm01_post_learn",
            },
        ],
        name="nanobot_variant_local_anchor",
    )

    monkeypatch.setattr(
        "past_bench.config.load_config",
        lambda _path: SimpleNamespace(
            defaults=SimpleNamespace(trace_dir=str(root / "traces"), agent_registry=None),
            sandbox=SimpleNamespace(enabled=False),
            runtime=SimpleNamespace(mode="local", registry_path=None, temperature=0.0),
        ),
    )
    monkeypatch.setattr("past_bench.cli._make_judge", lambda cfg, args: None)
    monkeypatch.setattr("past_bench.cli._check_agent_requirements", lambda *args, **kwargs: None)
    monkeypatch.setattr("past_bench.runner.services.ServiceManager", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr("past_bench.graders.registry.get_grader", lambda *args, **kwargs: object())

    observed_markers: list[tuple[str, str]] = []

    def fake_snapshot_nanobot(_path: Path, *, include_contents: bool = True) -> dict:
        return {
            "artifacts_dir": "unused",
            "memory_file_exists": False,
            "user_file_exists": False,
            "memory_chars": 0,
            "user_chars": 0,
            "memory_entries": [],
            "user_entries": [],
            "skill_count": 0,
            "skill_names": [],
            "skill_docs": {},
            "internal_tools": {
                "session_file": None,
                "tool_call_counts": {},
                "memory_calls": 0,
                "memory_action_counts": {},
                "memory_write_count": 0,
                "memory_read_count": 0,
                "skill_manage_calls": 0,
                "skill_manage_action_counts": {},
                "skill_create_count": 0,
                "skill_update_count": 0,
                "session_search_calls": 0,
                "skill_view_calls": 0,
                "skills_list_calls": 0,
                "skill_read_count": 0,
                "calls": [],
            },
        }

    def fake_execute_trial(**kwargs):
        trace_dir = Path(kwargs["trace_dir"])
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / "trace.jsonl"
        trace_path.write_text("{}", encoding="utf-8")

        workspace_dir = Path(kwargs["model_extra_body_override"]["nanobot"]["workspace_dir"])
        workspace_dir.mkdir(parents=True, exist_ok=True)
        task_id = kwargs["task"].task_id
        variant_label = workspace_dir.parents[3].name
        if task_id == "I03":
            marker = "with" if variant_label == "with_persistence" else "without"
            (workspace_dir / "variant_marker.txt").write_text(marker, encoding="utf-8")
        if task_id == "I04":
            observed_markers.append(
                (variant_label, (workspace_dir / "variant_marker.txt").read_text(encoding="utf-8"))
            )
        return trace_path, None

    def fake_grade_episode(**kwargs):
        task = kwargs["task"]
        return {
            "trace": str(kwargs["trace_path"]),
            "trace_id": f"{task.task_id}-trace",
            "task_id": task.task_id,
            "task_name": task.task_name,
            "final_response_text": "",
            "scores": {
                "completion": 0.0,
                "robustness": 0.0,
                "communication": 0.0,
                "safety": 1.0,
                "efficiency_turns": 0,
                "efficiency_tokens": 0,
                "efficiency_wall_time_s": 0.0,
            },
            "task_score": 0.0,
            "passed": False,
            "total_turns": 1,
            "tool_dispatch_count": 0,
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "timing": {"wall_time_s": 0.0, "model_time_s": 0.0, "tool_time_s": 0.0, "other_time_s": 0.0},
            "artifacts": fake_snapshot_nanobot(Path(".")),
            "artifact_diff": {},
            "retrieval_signals": {},
            "mechanism_scores": {},
            "internal_tools": fake_snapshot_nanobot(Path("."))["internal_tools"],
            "env_snapshot_present": False,
        }

    monkeypatch.setattr("past_bench.runner.self_evolve.snapshot_nanobot_workspace", fake_snapshot_nanobot)
    monkeypatch.setattr("past_bench.runner.self_evolve.snapshot_nanobot_artifacts", fake_snapshot_nanobot)
    monkeypatch.setattr("past_bench.cli._execute_trial", fake_execute_trial)
    monkeypatch.setattr("past_bench.runner.self_evolve.grade_episode", fake_grade_episode)

    args = _base_args(root, manifest, root / "traces", agent="nanobot", runtime="local")
    args.compare_no_persistence = True
    cmd_evolve(args)

    assert observed_markers == [
        ("with_persistence", "with"),
        ("without_persistence", "without"),
    ]


def test_cmd_evolve_preserves_history_within_family_home(tmp_path: Path, monkeypatch):
    root = tmp_path / "self_evolve_same_family_history"
    tasks_root = root / "tasks"
    for task_id in ("A01", "A02"):
        _write_task(tasks_root / task_id, task_id, task_id)

    manifest = root / "sequence.yaml"
    _write_history_sequence(
        manifest,
        [
            {
                "task": "tasks/A01",
                "label": "A01",
                "family_id": "F_same_family",
                "mechanism": "session_search",
                "bucket": "baseline",
                "latent_rule_id": "same_family_rule",
                "expected_persistence_signal": "session_search",
                "history_mode": "continue",
                "reflection_required": False,
            },
            {
                "task": "tasks/A02",
                "label": "A02",
                "family_id": "F_same_family",
                "mechanism": "session_search",
                "bucket": "evaluation",
                "latent_rule_id": "same_family_rule",
                "expected_persistence_signal": "session_search",
                "history_mode": "continue",
                "reflection_required": False,
            },
        ],
        name="same_family_history",
    )

    monkeypatch.setattr(
        "past_bench.config.load_config",
        lambda _path: SimpleNamespace(
            defaults=SimpleNamespace(trace_dir=str(root / "traces"), agent_registry=None),
            sandbox=SimpleNamespace(enabled=False),
            runtime=SimpleNamespace(mode="local", registry_path=None, temperature=0.0),
        ),
    )
    monkeypatch.setattr("past_bench.cli._make_judge", lambda cfg, args: None)
    monkeypatch.setattr("past_bench.cli._check_agent_requirements", lambda *args, **kwargs: None)
    monkeypatch.setattr("past_bench.runner.services.ServiceManager", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr("past_bench.graders.registry.get_grader", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "past_bench.runner.self_evolve.build_hermes_extra_body",
        lambda **kwargs: {"home_dir": str(kwargs["home_dir"])},
    )

    preserved = []

    def fake_snapshot_hermes_home(_path: Path, *, include_contents: bool = True) -> dict:
        return {
            "artifacts_dir": "unused",
            "memory_file_exists": False,
            "user_file_exists": False,
            "memory_chars": 0,
            "user_chars": 0,
            "memory_entries": [],
            "user_entries": [],
            "skill_count": 0,
            "skill_names": [],
            "skill_docs": {},
            "internal_tools": {
                "session_file": None,
                "tool_call_counts": {},
                "memory_calls": 0,
                "memory_action_counts": {},
                "memory_write_count": 0,
                "memory_read_count": 0,
                "skill_manage_calls": 0,
                "skill_manage_action_counts": {},
                "skill_create_count": 0,
                "skill_update_count": 0,
                "session_search_calls": 0,
                "skill_view_calls": 0,
                "skills_list_calls": 0,
                "skill_read_count": 0,
                "calls": [],
            },
        }

    def fake_execute_trial(**kwargs):
        trace_dir = Path(kwargs["trace_dir"])
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / "trace.jsonl"
        trace_path.write_text("{}", encoding="utf-8")

        home_dir = Path(kwargs["model_extra_body_override"]["home_dir"])
        home_dir.mkdir(parents=True, exist_ok=True)
        task_id = kwargs["task"].task_id
        marker = home_dir / "family_marker.txt"
        if task_id == "A01":
            marker.write_text("family state", encoding="utf-8")
        if task_id == "A02":
            preserved.append(marker.exists())
        return trace_path, None

    def fake_grade_episode(**kwargs):
        task = kwargs["task"]
        return {
            "trace": str(kwargs["trace_path"]),
            "trace_id": f"{task.task_id}-trace",
            "task_id": task.task_id,
            "task_name": task.task_name,
            "final_response_text": "",
            "scores": {
                "completion": 0.0,
                "robustness": 0.0,
                "communication": 0.0,
                "safety": 1.0,
                "efficiency_turns": 0,
                "efficiency_tokens": 0,
                "efficiency_wall_time_s": 0.0,
            },
            "task_score": 0.0,
            "passed": False,
            "total_turns": 1,
            "tool_dispatch_count": 0,
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "timing": {"wall_time_s": 0.0, "model_time_s": 0.0, "tool_time_s": 0.0, "other_time_s": 0.0},
            "artifacts": fake_snapshot_hermes_home(Path(".")),
            "artifact_diff": {},
            "retrieval_signals": {},
            "mechanism_scores": {},
            "internal_tools": fake_snapshot_hermes_home(Path("."))["internal_tools"],
            "env_snapshot_present": False,
        }

    monkeypatch.setattr("past_bench.runner.self_evolve.snapshot_hermes_home", fake_snapshot_hermes_home)
    monkeypatch.setattr("past_bench.runner.self_evolve.snapshot_hermes_artifacts", fake_snapshot_hermes_home)
    monkeypatch.setattr("past_bench.cli._execute_trial", fake_execute_trial)
    monkeypatch.setattr("past_bench.runner.self_evolve.grade_episode", fake_grade_episode)

    args = _base_args(root, manifest, root / "traces", agent="hermes", runtime="local")
    cmd_evolve(args)

    assert preserved == [True]


def test_cmd_evolve_isolates_family_homes_from_each_other(tmp_path: Path, monkeypatch):
    root = tmp_path / "self_evolve_cross_family_isolation"
    tasks_root = root / "tasks"
    for task_id in ("A01", "B01"):
        _write_task(tasks_root / task_id, task_id, task_id)

    manifest = root / "sequence.yaml"
    _write_history_sequence(
        manifest,
        [
            {
                "task": "tasks/A01",
                "label": "A01",
                "family_id": "F_alpha",
                "mechanism": "session_search",
                "bucket": "baseline",
                "latent_rule_id": "alpha_rule",
                "expected_persistence_signal": "session_search",
                "history_mode": "continue",
                "reflection_required": False,
            },
            {
                "task": "tasks/B01",
                "label": "B01",
                "family_id": "F_beta",
                "mechanism": "session_search",
                "bucket": "evaluation",
                "latent_rule_id": "beta_rule",
                "expected_persistence_signal": "session_search",
                "history_mode": "continue",
                "reflection_required": False,
            },
        ],
        name="cross_family_isolation",
    )

    monkeypatch.setattr(
        "past_bench.config.load_config",
        lambda _path: SimpleNamespace(
            defaults=SimpleNamespace(trace_dir=str(root / "traces"), agent_registry=None),
            sandbox=SimpleNamespace(enabled=False),
            runtime=SimpleNamespace(mode="local", registry_path=None, temperature=0.0),
        ),
    )
    monkeypatch.setattr("past_bench.cli._make_judge", lambda cfg, args: None)
    monkeypatch.setattr("past_bench.cli._check_agent_requirements", lambda *args, **kwargs: None)
    monkeypatch.setattr("past_bench.runner.services.ServiceManager", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr("past_bench.graders.registry.get_grader", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "past_bench.runner.self_evolve.build_hermes_extra_body",
        lambda **kwargs: {"home_dir": str(kwargs["home_dir"])},
    )

    isolated = []

    def fake_snapshot_hermes_home(_path: Path, *, include_contents: bool = True) -> dict:
        return {
            "artifacts_dir": "unused",
            "memory_file_exists": False,
            "user_file_exists": False,
            "memory_chars": 0,
            "user_chars": 0,
            "memory_entries": [],
            "user_entries": [],
            "skill_count": 0,
            "skill_names": [],
            "skill_docs": {},
            "internal_tools": {
                "session_file": None,
                "tool_call_counts": {},
                "memory_calls": 0,
                "memory_action_counts": {},
                "memory_write_count": 0,
                "memory_read_count": 0,
                "skill_manage_calls": 0,
                "skill_manage_action_counts": {},
                "skill_create_count": 0,
                "skill_update_count": 0,
                "session_search_calls": 0,
                "skill_view_calls": 0,
                "skills_list_calls": 0,
                "skill_read_count": 0,
                "calls": [],
            },
        }

    def fake_execute_trial(**kwargs):
        trace_dir = Path(kwargs["trace_dir"])
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / "trace.jsonl"
        trace_path.write_text("{}", encoding="utf-8")

        home_dir = Path(kwargs["model_extra_body_override"]["home_dir"])
        home_dir.mkdir(parents=True, exist_ok=True)
        task_id = kwargs["task"].task_id
        marker = home_dir / "family_only.txt"
        if task_id == "A01":
            marker.write_text("alpha", encoding="utf-8")
        if task_id == "B01":
            isolated.append(not marker.exists())
        return trace_path, None

    def fake_grade_episode(**kwargs):
        task = kwargs["task"]
        return {
            "trace": str(kwargs["trace_path"]),
            "trace_id": f"{task.task_id}-trace",
            "task_id": task.task_id,
            "task_name": task.task_name,
            "final_response_text": "",
            "scores": {
                "completion": 0.0,
                "robustness": 0.0,
                "communication": 0.0,
                "safety": 1.0,
                "efficiency_turns": 0,
                "efficiency_tokens": 0,
                "efficiency_wall_time_s": 0.0,
            },
            "task_score": 0.0,
            "passed": False,
            "total_turns": 1,
            "tool_dispatch_count": 0,
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "timing": {"wall_time_s": 0.0, "model_time_s": 0.0, "tool_time_s": 0.0, "other_time_s": 0.0},
            "artifacts": fake_snapshot_hermes_home(Path(".")),
            "artifact_diff": {},
            "retrieval_signals": {},
            "mechanism_scores": {},
            "internal_tools": fake_snapshot_hermes_home(Path("."))["internal_tools"],
            "env_snapshot_present": False,
        }

    monkeypatch.setattr("past_bench.runner.self_evolve.snapshot_hermes_home", fake_snapshot_hermes_home)
    monkeypatch.setattr("past_bench.runner.self_evolve.snapshot_hermes_artifacts", fake_snapshot_hermes_home)
    monkeypatch.setattr("past_bench.cli._execute_trial", fake_execute_trial)
    monkeypatch.setattr("past_bench.runner.self_evolve.grade_episode", fake_grade_episode)

    args = _base_args(root, manifest, root / "traces", agent="hermes", runtime="local")
    cmd_evolve(args)

    assert isolated == [True]


def test_cmd_evolve_resets_family_runtime_between_independent_runs(tmp_path: Path, monkeypatch):
    root = tmp_path / "self_evolve_cross_run_isolation"
    tasks_root = root / "tasks"
    _write_task(tasks_root / "A01", "A01", "A01")

    manifest = root / "sequence.yaml"
    _write_history_sequence(
        manifest,
        [
            {
                "task": "tasks/A01",
                "label": "A01",
                "family_id": "F_single",
                "mechanism": "session_search",
                "bucket": "baseline",
                "latent_rule_id": "single_rule",
                "expected_persistence_signal": "session_search",
                "history_mode": "continue",
                "reflection_required": False,
            }
        ],
        name="cross_run_isolation",
    )

    monkeypatch.setattr(
        "past_bench.config.load_config",
        lambda _path: SimpleNamespace(
            defaults=SimpleNamespace(trace_dir=str(root / "traces"), agent_registry=None),
            sandbox=SimpleNamespace(enabled=False),
            runtime=SimpleNamespace(mode="local", registry_path=None, temperature=0.0),
        ),
    )
    monkeypatch.setattr("past_bench.cli._make_judge", lambda cfg, args: None)
    monkeypatch.setattr("past_bench.cli._check_agent_requirements", lambda *args, **kwargs: None)
    monkeypatch.setattr("past_bench.runner.services.ServiceManager", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr("past_bench.graders.registry.get_grader", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "past_bench.runner.self_evolve.build_hermes_extra_body",
        lambda **kwargs: {"home_dir": str(kwargs["home_dir"])},
    )

    run_counter = 0
    second_run_saw_clean_home = []

    def fake_snapshot_hermes_home(_path: Path, *, include_contents: bool = True) -> dict:
        return {
            "artifacts_dir": "unused",
            "memory_file_exists": False,
            "user_file_exists": False,
            "memory_chars": 0,
            "user_chars": 0,
            "memory_entries": [],
            "user_entries": [],
            "skill_count": 0,
            "skill_names": [],
            "skill_docs": {},
            "internal_tools": {
                "session_file": None,
                "tool_call_counts": {},
                "memory_calls": 0,
                "memory_action_counts": {},
                "memory_write_count": 0,
                "memory_read_count": 0,
                "skill_manage_calls": 0,
                "skill_manage_action_counts": {},
                "skill_create_count": 0,
                "skill_update_count": 0,
                "session_search_calls": 0,
                "skill_view_calls": 0,
                "skills_list_calls": 0,
                "skill_read_count": 0,
                "calls": [],
            },
        }

    def fake_execute_trial(**kwargs):
        nonlocal run_counter
        trace_dir = Path(kwargs["trace_dir"])
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / "trace.jsonl"
        trace_path.write_text("{}", encoding="utf-8")

        run_counter += 1
        home_dir = Path(kwargs["model_extra_body_override"]["home_dir"])
        home_dir.mkdir(parents=True, exist_ok=True)
        marker = home_dir / "persisted.txt"
        if run_counter == 1:
            marker.write_text("first run", encoding="utf-8")
        else:
            second_run_saw_clean_home.append(not marker.exists())
        return trace_path, None

    def fake_grade_episode(**kwargs):
        task = kwargs["task"]
        return {
            "trace": str(kwargs["trace_path"]),
            "trace_id": f"{task.task_id}-trace",
            "task_id": task.task_id,
            "task_name": task.task_name,
            "final_response_text": "",
            "scores": {
                "completion": 0.0,
                "robustness": 0.0,
                "communication": 0.0,
                "safety": 1.0,
                "efficiency_turns": 0,
                "efficiency_tokens": 0,
                "efficiency_wall_time_s": 0.0,
            },
            "task_score": 0.0,
            "passed": False,
            "total_turns": 1,
            "tool_dispatch_count": 0,
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "timing": {"wall_time_s": 0.0, "model_time_s": 0.0, "tool_time_s": 0.0, "other_time_s": 0.0},
            "artifacts": fake_snapshot_hermes_home(Path(".")),
            "artifact_diff": {},
            "retrieval_signals": {},
            "mechanism_scores": {},
            "internal_tools": fake_snapshot_hermes_home(Path("."))["internal_tools"],
            "env_snapshot_present": False,
        }

    monkeypatch.setattr("past_bench.runner.self_evolve.snapshot_hermes_home", fake_snapshot_hermes_home)
    monkeypatch.setattr("past_bench.runner.self_evolve.snapshot_hermes_artifacts", fake_snapshot_hermes_home)
    monkeypatch.setattr("past_bench.cli._execute_trial", fake_execute_trial)
    monkeypatch.setattr("past_bench.runner.self_evolve.grade_episode", fake_grade_episode)

    args = _base_args(root, manifest, root / "traces", agent="hermes", runtime="local")
    cmd_evolve(args)
    cmd_evolve(args)

    assert second_run_saw_clean_home == [True]
