from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsimem.base_memory_experiment import BaseMemoryComparisonManifest, BaseMemoryCondition, compare_run_manifests
from rsimem.base_memory_launcher import (
    _backend_descriptor, _materialize_sequence, _past_command, prepare_comparison,
    run_comparison,
)


def _digest(char: str) -> str:
    return char * 64


def test_manifest_requires_all_isolated_backends() -> None:
    manifest = BaseMemoryComparisonManifest.create(
        family_id="SM01", source_sequence_digest=_digest("a"), fixture_digest=_digest("b"),
        config_digest=_digest("c"), registry_digest=_digest("d"), token_budget=4096,
    )
    assert {run.condition for run in manifest.runs} == set(BaseMemoryCondition)
    assert len({run.hermes_home_directory for run in manifest.runs}) == 3
    assert manifest.base_model == "gpt-5.6-luna"


def test_no_memory_preserves_persistence_and_only_removes_semantic_memory() -> None:
    source = {"name": "fixture", "episodes": [{"task": "task.yaml", "shared_cold_run": True}], "hermes": {"skills_enabled": True, "session_search_enabled": True}}
    no_memory = _materialize_sequence(source, BaseMemoryCondition.NO_MEMORY)
    assert no_memory["hermes"]["memory_enabled"] is False
    assert no_memory["hermes"]["reasoning_effort"] == "none"
    assert no_memory["hermes"]["skills_enabled"] is True
    assert no_memory["episodes"][0]["shared_cold_run"] is False
    assert _backend_descriptor(BaseMemoryCondition.NO_MEMORY)["persistence_variant"] == "with_persistence"
    static = _materialize_sequence(source, BaseMemoryCondition.MEM0_STATIC)
    assert static["hermes"]["rsimem_semantic_writeback_mode"] == "static"


def test_compare_rejects_model_or_fixture_drift() -> None:
    common = {"run_id": "one", "condition": "NoMemory", "backend": {}, "port_offset": 1,
              "state_directory": "state", "trace_directory": "trace", "artifact_directory": "artifacts", "hermes_home_directory": "home", "base_model": "gpt-5.6-luna"}
    right = {**common, "run_id": "two", "condition": "Mem0Static", "base_model": "gpt-5.4"}
    with pytest.raises(ValueError, match="identity drift"):
        compare_run_manifests(common, right)


def test_prepare_writes_three_backend_specific_immutable_manifests(tmp_path: Path) -> None:
    sequence = tmp_path / "source.yaml"
    sequence.write_text("name: fixture\nepisodes:\n  - task: task.yaml\n", encoding="utf-8")
    config = tmp_path / "config.yaml"; config.write_text("x: 1\n", encoding="utf-8")
    registry = tmp_path / "registry.yaml"; registry.write_text("x: 1\n", encoding="utf-8")
    manifest = prepare_comparison(source_sequence=sequence, output_root=tmp_path / "out", family_id="SM01", config=config, registry=registry, token_budget=32)
    assert (tmp_path / "out" / "base_memory_manifest.json").is_file()
    for run in manifest.runs:
        payload = json.loads((tmp_path / "out" / run.run_id / "run_manifest.json").read_text())
        assert payload["condition"] == run.condition.value
        assert payload["backend"]["persistence_variant"] == "with_persistence"
        rendered = (tmp_path / "out" / run.run_id / "sequence.yaml").read_text(encoding="utf-8")
        assert str((tmp_path / "task.yaml").resolve()) in rendered
        command = _past_command(past_bin=Path("past-bench"), sequence=Path("sequence.yaml"), trace_dir=Path("trace"), config=config, registry=registry, run=run, base_url="https://example.test/v1")
        assert command[command.index("--agent") + 1] == "hermes"
        assert command[command.index("--persistence-variant") + 1] == "with_persistence"
        assert command[command.index("--rsimem-state-dir") + 1].endswith(run.state_directory)
        assert command[command.index("--rsimem-hermes-home-dir") + 1].endswith(run.hermes_home_directory)


def test_dry_run_materializes_isolated_command_receipts(tmp_path: Path) -> None:
    sequence = tmp_path / "source.yaml"
    sequence.write_text("name: fixture\nepisodes:\n  - task: task.yaml\n", encoding="utf-8")
    config = tmp_path / "config.yaml"; config.write_text("x: 1\n", encoding="utf-8")
    registry = tmp_path / "registry.yaml"; registry.write_text("x: 1\n", encoding="utf-8")
    manifest = run_comparison(
        source_sequence=sequence, output_root=tmp_path / "out", family_id="SM01",
        past_bin=tmp_path / "past-bench", past_root=tmp_path, config=config,
        registry=registry, base_url="https://example.test/v1", token_budget=32,
        dry_run=True,
    )
    for run in manifest.runs:
        launch = json.loads((tmp_path / "out" / run.run_id / "launch.json").read_text())
        assert launch["dry_run"] is True
        command = launch["command"]
        assert command[command.index("--rsimem-artifact-dir") + 1].endswith(run.artifact_directory)


def test_single_condition_execution_leaves_other_conditions_unlaunched(tmp_path: Path) -> None:
    sequence = tmp_path / "source.yaml"
    sequence.write_text("name: fixture\nepisodes:\n  - task: task.yaml\n", encoding="utf-8")
    config = tmp_path / "config.yaml"; config.write_text("x: 1\n", encoding="utf-8")
    registry = tmp_path / "registry.yaml"; registry.write_text("x: 1\n", encoding="utf-8")
    manifest = run_comparison(
        source_sequence=sequence, output_root=tmp_path / "out", family_id="SM01",
        past_bin=tmp_path / "past-bench", past_root=tmp_path, config=config,
        registry=registry, base_url="https://example.test/v1", token_budget=32,
        condition=BaseMemoryCondition.NO_MEMORY, dry_run=True,
    )
    for run in manifest.runs:
        launch = tmp_path / "out" / run.run_id / "launch.json"
        assert launch.exists() is (run.condition is BaseMemoryCondition.NO_MEMORY)


def test_launcher_does_not_put_api_key_in_receipts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sequence = tmp_path / "source.yaml"
    sequence.write_text("name: fixture\nepisodes:\n  - task: task.yaml\n", encoding="utf-8")
    config = tmp_path / "config.yaml"; config.write_text("x: 1\n", encoding="utf-8")
    registry = tmp_path / "registry.yaml"; registry.write_text("x: 1\n", encoding="utf-8")
    monkeypatch.setenv("GPT_LUNA_API_KEY", "sk-secret-test")
    run_comparison(
        source_sequence=sequence, output_root=tmp_path / "out", family_id="SM01",
        past_bin=tmp_path / "past-bench", past_root=tmp_path, config=config,
        registry=registry, base_url="https://example.test/v1", token_budget=32,
        condition=BaseMemoryCondition.NO_MEMORY, dry_run=True,
    )
    assert "sk-secret-test" not in (tmp_path / "out" / "base-memory-nomemory" / "launch.json").read_text()
