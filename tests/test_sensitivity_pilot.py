from __future__ import annotations

import json
from pathlib import Path

from rsimem.provider_probe import ProviderProbeResult
from rsimem.sensitivity import SensitivityPanel
from rsimem.sensitivity_pilot import run_sensitivity_pilot
from rsimem.sensitivity_pilot_audit import _usage


def _kwargs(tmp_path: Path) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    past_root = root / "benchmarks" / "past-bench"
    return {
        "panel": SensitivityPanel.SEMANTIC,
        "family_id": "SM01_preference_adoption",
        "replicate": 2,
        "batch_id": "stage3-sm01-pilot-fixture",
        "rsimem_root": root,
        "past_bench_root": past_root,
        "registry_path": root / "configs/sensitivity/oracle_seed_registry_sm01.json",
        "trusted_seed_root": past_root / "self-evolve-tasks-v2" / "_rsimem_oracles",
        "output_root": tmp_path,
        "config_path": root / "configs/past_bench_luna_smoke.yaml",
        "agent_registry_path": root / "configs/agents.yaml",
        "api_key": "test-key",
    }


def test_dry_run_writes_rotated_registered_plan_without_provider(tmp_path: Path) -> None:
    def unexpected_probe(*_args):
        raise AssertionError("dry-run must not probe provider")

    plan = run_sensitivity_pilot(
        **_kwargs(tmp_path), execute=False, provider_probe=unexpected_probe
    )
    assert [item.value for item in plan.condition_order] == [
        "native_static", "type_matched_oracle", "shortcut_current_input",
        "wrong_mechanism", "no_persistence",
    ]
    document = json.loads((tmp_path / "sensitivity_pilot_plan.json").read_text(encoding="utf-8"))
    assert document == plan.payload()
    events = [json.loads(line) for line in (tmp_path / "sensitivity_pilot_events.jsonl").read_text().splitlines()]
    assert [item["status"] for item in events] == ["planned"] * 5
    assert all("SM01_preference_adoption" not in item["run_id"] for item in events)


def test_execute_runs_only_registered_commands_after_probe(tmp_path: Path) -> None:
    calls: list[tuple[tuple[str, ...], Path]] = []

    def provider(base_url: str, api_key: str, model: str) -> ProviderProbeResult:
        assert base_url == "https://coding.tu-zi.com/v1"
        assert api_key == "test-key"
        assert model == "gpt-5.6-luna"
        return ProviderProbeResult(base_url, model, 200, True, True)

    def runner(command: tuple[str, ...], cwd: Path) -> int:
        calls.append((command, cwd))
        return 0

    plan = run_sensitivity_pilot(
        **_kwargs(tmp_path), execute=True, provider_probe=provider, runner=runner
    )
    assert len(calls) == 5
    assert all("--rsimem-method-task-id" in command for command, _ in calls)
    assert all("--no-judge" in command for command, _ in calls)
    assert all("--rsimem-mode" in command for command, _ in calls)
    assert all(cwd.name == "past-bench" for _, cwd in calls)
    events = [json.loads(line) for line in (tmp_path / "sensitivity_pilot_events.jsonl").read_text().splitlines()]
    completed = [item for item in events if item["status"] == "completed"]
    assert {item["run_id"] for item in completed} == set(plan.run_ids)
    probe = json.loads((tmp_path / "provider_probe.json").read_text(encoding="utf-8"))
    assert probe["ok"] is True


def test_usage_audit_matches_mixed_trace_end_field_names() -> None:
    events = [
        {"type": "model_call_usage", "usage": {
            "input_tokens": 10, "output_tokens": 3, "cache_read_tokens": 2,
            "cache_write_tokens": 0, "reasoning_tokens": 1,
        }, "attempt": 1},
        {"type": "trace_end", "trace_id": "trace-1",
         "model_input_tokens": 10, "model_output_tokens": 3,
         "cache_read_tokens": 2, "cache_write_tokens": 0,
         "reasoning_tokens": 1, "model_request_count": 1,
         "model_retry_count": 0, "model_usage_complete": True},
    ]
    totals, issues = _usage(events, "trace-1")
    assert issues == []
    assert totals == {
        "input_tokens": 10, "output_tokens": 3, "cache_read_tokens": 2,
        "cache_write_tokens": 0, "reasoning_tokens": 1, "requests": 1,
        "retries": 0,
    }
