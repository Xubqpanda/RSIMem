from __future__ import annotations

import json
from pathlib import Path

import rsimem.adamem_screening_launcher as launcher


def _manifest(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({
        "schema": "rsimem-adamem-full-suite-screening-manifest-v1",
        "manifest_digest": "manifest", "records": records,
    }), encoding="utf-8")
    return path


def test_screening_marks_sm01_precovered_without_provider_call(tmp_path: Path, monkeypatch) -> None:
    manifest = _manifest(tmp_path / "manifest.json", [{
        "family_id": "SM01_preference_adoption", "config": str(tmp_path / "SM01.yaml"),
        "cutover_label": "learn",
    }])
    monkeypatch.setattr(launcher, "run_trajectory", lambda **_: (_ for _ in ()).throw(AssertionError()))
    report = launcher.run_screening(
        config_dir=tmp_path, output_root=tmp_path / "out", past_bin=tmp_path / "past",
        past_root=tmp_path, config=tmp_path / "config", registry=tmp_path / "registry",
        base_url="https://example.test/v1", api_key=None, manifest_path=manifest,
    )
    assert report["results"] == [{
        "family_id": "SM01_preference_adoption", "status": "precovered",
        "evidence": "docs/sm01_adamem_trajectory_results_20260908.md",
        "reason": "existing_accepted_formal_batch",
    }]


def test_screening_retries_legacy_split_receipt_with_full_sequence(tmp_path: Path, monkeypatch) -> None:
    sequence = tmp_path / "EP01.yaml"
    sequence.write_text("name: EP01\nepisodes: []\n", encoding="utf-8")
    manifest = _manifest(tmp_path / "manifest.json", [{
        "family_id": "EP01", "config": str(sequence), "cutover_label": "learn",
    }])
    progress = tmp_path / "out" / "screening" / "screening_progress.json"
    progress.parent.mkdir(parents=True)
    progress.write_text(json.dumps({"results": [{"family_id": "EP01", "status": "accepted"}]}), encoding="utf-8")
    calls: list[object] = []
    monkeypatch.setattr(launcher, "run_trajectory", lambda **kwargs: calls.append(kwargs) or type("R", (), {"payload": lambda _: {}})())
    report = launcher.run_screening(
        config_dir=tmp_path, output_root=tmp_path / "out", past_bin=tmp_path / "past",
        past_root=tmp_path, config=tmp_path / "config", registry=tmp_path / "registry",
        base_url="https://example.test/v1", api_key=None, manifest_path=manifest,
    )
    assert len(calls) == 1
    assert report["results"][-1]["screening_topology"] == "single_full_sequence"
