from __future__ import annotations

from pathlib import Path
import shutil

import pytest
import yaml

from experiments.legacy.native.native_attribution_launcher import prepare_native_attribution_launch
from experiments.legacy.native.native_attribution_protocol import NativeAttributionRepairProtocol
from experiments.legacy.native.native_attribution_run import build_native_attribution_manifest


ROOT = Path(__file__).resolve().parents[1]
PAST = ROOT / "benchmarks" / "past-bench"


def _run():
    manifest = build_native_attribution_manifest(
        batch_id="launcher-fixture", protocol=NativeAttributionRepairProtocol.create(),
        past_bench_root=PAST, rsimem_commit="commit.rsimem.fixture",
        past_bench_commit="commit.past.fixture", replicate_count=1, port_base_offset=200,
    )
    return manifest.runs[0]


def test_launcher_prepares_native_episodes_and_isolation_arguments(tmp_path) -> None:
    run = _run()
    prepared = prepare_native_attribution_launch(run=run, past_bench_root=PAST, output_root=tmp_path)
    document = yaml.safe_load(prepared.sequence_path.read_text(encoding="utf-8"))
    assert document["episodes"]
    assert all(episode["bucket"] != "control" for episode in document["episodes"])
    assert all(not episode.get("shared_cold_run") for episode in document["episodes"])
    assert prepared.command[prepared.command.index("--rsimem-method-task-id") + 1] == run.method_case_id
    assert run.family_id not in prepared.command[prepared.command.index("--rsimem-method-task-id") + 1]
    assert prepared.command[prepared.command.index("--port-offset") + 1] == str(run.port_offset)
    assert prepared.command[prepared.command.index("--model") + 1] == run.model_id
    assert prepared.command[prepared.command.index("--runtime") + 1] == "local"
    assert "--no-judge" in prepared.command
    assert prepared.command[prepared.command.index("--base-url") + 1] == f"https://{run.provider_id}"
    assert prepared.command[prepared.command.index("--rsimem-artifact-dir") + 1] == str(
        (tmp_path / run.artifact_directory).resolve()
    )


def test_launcher_rejects_fixture_and_port_drift(tmp_path) -> None:
    run = _run()
    copied = tmp_path / "past"
    shutil.copytree(PAST, copied, symlinks=False)
    fixture = next(
        path for path in (copied / "self-evolve-tasks-v2").rglob("*.json")
        if run.family_id.split("_", 1)[0].lower() in path.as_posix().lower()
    )
    fixture.write_text(fixture.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fixture identity drift"):
        prepare_native_attribution_launch(
            run=run, past_bench_root=copied, output_root=tmp_path / "output"
        )
