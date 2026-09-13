from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.base_memory.base_memory_experiment import (
    HISTORICAL_BASE_MEMORY_CONDITIONS,
    BaseMemoryComparisonManifest,
    BaseMemoryCondition,
    compare_run_manifests,
)
from experiments.base_memory.base_memory_launcher import (
    _backend_descriptor, _materialize_sequence, _past_command, _past_environment,
    prepare_comparison,
    run_comparison,
)


def _digest(char: str) -> str:
    return char * 64


def test_manifest_requires_all_isolated_backends() -> None:
    manifest = BaseMemoryComparisonManifest.create(
        family_id="SM01", source_sequence_digest=_digest("a"), fixture_digest=_digest("b"),
        config_digest=_digest("c"), registry_digest=_digest("d"), token_budget=4096,
    )
    assert {run.condition for run in manifest.runs} == set(HISTORICAL_BASE_MEMORY_CONDITIONS)
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


def test_all_memory_off_disables_every_memory_surface_without_changing_topology() -> None:
    source = {
        "name": "fixture",
        "episodes": [
            {"task": "semantic.yaml", "mechanism": "memory"},
            {"task": "episodic.yaml", "mechanism": "session_search"},
            {"task": "procedural.yaml", "mechanism": "skill"},
            {"task": "mixed.yaml", "mechanism": "mixed"},
        ],
        "hermes": {"memory_enabled": True, "user_profile_enabled": True, "skills_enabled": True, "session_search_enabled": True},
    }
    materialized = _materialize_sequence(source, BaseMemoryCondition.ALL_MEMORY_OFF)
    assert materialized["hermes"] == {
        "memory_enabled": False,
        "user_profile_enabled": False,
        "skills_enabled": False,
        "session_search_enabled": False,
        "all_memory_off": True,
        "rsimem_mode": "native+ledger",
        "rsimem_semantic_writeback_mode": "disabled",
        "reasoning_effort": "none",
    }
    assert all(episode["shared_cold_run"] is False for episode in materialized["episodes"])
    descriptor = _backend_descriptor(BaseMemoryCondition.ALL_MEMORY_OFF)
    assert descriptor["persistence_variant"] == "with_persistence"
    assert descriptor["semantic_memory_enabled"] is False
    assert descriptor["episodic_memory_enabled"] is False
    assert descriptor["procedural_memory_enabled"] is False


def test_all_memory_off_materializes_only_its_explicit_run(tmp_path: Path) -> None:
    sequence = tmp_path / "source.yaml"
    sequence.write_text("name: fixture\nepisodes:\n  - task: task.yaml\n", encoding="utf-8")
    config = tmp_path / "config.yaml"; config.write_text("x: 1\n", encoding="utf-8")
    registry = tmp_path / "registry.yaml"; registry.write_text("x: 1\n", encoding="utf-8")
    manifest = run_comparison(
        source_sequence=sequence, output_root=tmp_path / "out", family_id="SM01",
        past_bin=tmp_path / "past-bench", past_root=tmp_path, config=config,
        registry=registry, base_url="https://example.test/v1", token_budget=32,
        condition=BaseMemoryCondition.ALL_MEMORY_OFF, dry_run=True,
    )
    assert [run.condition for run in manifest.runs] == [BaseMemoryCondition.ALL_MEMORY_OFF]
    payload = json.loads((tmp_path / "out" / "base-memory-allmemoryoff" / "run_manifest.json").read_text())
    assert payload["backend"]["memory_surface_policy"] == "all_memory_off"


def test_compare_rejects_model_or_fixture_drift() -> None:
    common = {"manifest_id": "left", "runs": [], "sequence_digest": "a", "run_id": "one", "condition": "NoMemory", "backend": {}, "port_offset": 1,
              "state_directory": "state", "trace_directory": "trace", "artifact_directory": "artifacts", "hermes_home_directory": "home", "base_model": "gpt-5.6-luna"}
    right = {**common, "run_id": "two", "condition": "Mem0Static", "base_model": "gpt-5.4"}
    with pytest.raises(ValueError, match="identity drift"):
        compare_run_manifests(common, right)


def test_compare_accepts_fresh_assembly_and_backend_sequence_difference() -> None:
    left = {"manifest_id": "one", "runs": ["one"], "sequence_digest": "a", "run_id": "one", "condition": "NoMemory", "backend": {"id": "none"}, "port_offset": 1,
            "state_directory": "state-one", "trace_directory": "trace-one", "artifact_directory": "artifacts-one", "hermes_home_directory": "home-one", "base_model": "gpt-5.6-luna"}
    right = {**left, "manifest_id": "two", "runs": ["two"], "sequence_digest": "b", "run_id": "two", "condition": "Mem0Static", "backend": {"id": "mem0"}, "port_offset": 2,
             "state_directory": "state-two", "trace_directory": "trace-two", "artifact_directory": "artifacts-two", "hermes_home_directory": "home-two"}
    assert set(compare_run_manifests(left, right)) == {"manifest_id", "runs", "sequence_digest", "run_id", "condition", "backend", "port_offset", "state_directory", "trace_directory", "artifact_directory", "hermes_home_directory"}


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


def test_auxiliary_session_search_uses_same_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = _past_environment(base_url="https://provider.example/v1", api_key="run-key")
    assert env["AUXILIARY_SESSION_SEARCH_BASE_URL"] == "https://provider.example/v1"
    assert env["AUXILIARY_SESSION_SEARCH_MODEL"] == "gpt-5.6-luna"
    assert env["AUXILIARY_SESSION_SEARCH_API_KEY"] == "run-key"
    assert env["ANTHROPIC_API_KEY"] == "run-key"
