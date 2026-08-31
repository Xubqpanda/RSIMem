from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rsimem.memory.final_evaluation import (
    FinalEvaluationRecord,
    FinalEvaluationReporter,
    JsonFinalEvaluationStore,
    main,
)


def _record(**overrides: object) -> FinalEvaluationRecord:
    values: dict[str, object] = {
        "candidate_artifact_id": "extraction-prompt.candidate.v1",
        "run_id": "run.final.v1",
        "candidate_frozen_at": "2026-08-30T01:00:00Z",
        "run_completed_at": "2026-08-30T02:00:00Z",
        "score_read_at": "2026-08-30T03:00:00Z",
        "score_digest": "a" * 64,
        "metric_name": "task.pass_rate",
        "metric_value": 0.75,
    }
    values.update(overrides)
    return FinalEvaluationRecord.create(**values)


def test_final_reporter_is_separate_and_time_ordered() -> None:
    record = _record()
    assert record.evidence_plane.value == "final_evaluation"
    assert record.evidence_source.value == "final_reporter"
    assert FinalEvaluationRecord.from_payload(
        json.loads(json.dumps(record.payload()))
    ) == record


@pytest.mark.parametrize("field", ("candidate_frozen_at", "run_completed_at"))
def test_score_read_before_freeze_or_completion_is_rejected(field: str) -> None:
    with pytest.raises(ValueError, match="after"):
        _record(**{field: "2026-08-30T04:00:00Z"})


def test_run_completion_before_candidate_freeze_is_rejected() -> None:
    with pytest.raises(ValueError, match="complete after candidate freeze"):
        _record(
            candidate_frozen_at="2026-08-30T02:00:00Z",
            run_completed_at="2026-08-30T01:00:00Z",
            score_read_at="2026-08-30T03:00:00Z",
        )


def test_final_evaluation_rejects_wrong_plane_and_non_numeric_score() -> None:
    with pytest.raises(ValueError, match="plane and source identity"):
        replace(_record(), evidence_plane="pure_process")
    with pytest.raises(TypeError, match="metric value"):
        _record(metric_value=True)


def test_final_evaluation_store_is_isolated_restart_safe_and_idempotent(tmp_path) -> None:
    path = tmp_path / "final-evaluation.jsonl"
    record = _record()
    store = JsonFinalEvaluationStore(path)
    assert store.append(record) is True
    assert store.append(record) is False
    assert JsonFinalEvaluationStore(path).records() == (record,)

    conflicting = record.payload()
    conflicting["metric_value"] = 0.5
    path.write_text(
        path.read_text(encoding="utf-8")
        + json.dumps(conflicting, ensure_ascii=True, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="malformed final evaluation record"):
        JsonFinalEvaluationStore(path)


def test_final_evaluation_store_rejects_non_reporter_values(tmp_path) -> None:
    with pytest.raises(TypeError, match="FinalEvaluationRecord"):
        JsonFinalEvaluationStore(tmp_path / "final.jsonl").append({})


def test_final_evaluation_store_rejects_symlinked_paths(tmp_path) -> None:
    target = tmp_path / "target"
    target.write_text("", encoding="utf-8")
    path = tmp_path / "final.jsonl"
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        JsonFinalEvaluationStore(path).records()


def test_final_evaluation_store_rejects_symlinked_lock(tmp_path) -> None:
    path = tmp_path / "final.jsonl"
    lock_target = tmp_path / "lock-target"
    lock_target.write_text("", encoding="utf-8")
    path.with_name(path.name + ".lock").symlink_to(lock_target)
    with pytest.raises(ValueError, match="lock.*symlink"):
        JsonFinalEvaluationStore(path).records()


def test_final_evaluation_store_failed_commit_preserves_previous_record(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "final-atomic.jsonl"
    store = JsonFinalEvaluationStore(path)
    first = _record()
    assert store.append(first) is True
    second = _record(
        candidate_artifact_id="candidate.reporter.atomic.second",
        run_id="run.reporter.atomic.second",
    )

    def fail_replace(*args, **kwargs):
        raise OSError("simulated final evaluation interruption")

    monkeypatch.setattr("rsimem.memory._atomic_jsonl.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated final evaluation interruption"):
        store.append(second)
    assert JsonFinalEvaluationStore(path).records() == (first,)
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_final_reporter_reads_score_only_after_completed_run(tmp_path) -> None:
    store = JsonFinalEvaluationStore(tmp_path / "final.jsonl")
    reporter = FinalEvaluationReporter(store)
    calls: list[str] = []

    record = reporter.read_after_completion(
        candidate_artifact_id="candidate.reporter.v1",
        run_id="run.reporter.v1",
        candidate_frozen_at="2026-08-30T01:00:00Z",
        run_completed_at="2026-08-30T02:00:00Z",
        score_read_at="2026-08-30T03:00:00Z",
        metric_name="task.pass_rate",
        score_reader=lambda: calls.append("read") or 0.75,
    )
    assert calls == ["read"]
    assert record.metric_value == 0.75
    assert JsonFinalEvaluationStore(tmp_path / "final.jsonl").records() == (record,)


@pytest.mark.parametrize(
    ("run_completed_at", "score_read_at", "message"),
    (
        ("2026-08-30T00:00:00Z", "2026-08-30T03:00:00Z", "complete after candidate freeze"),
        ("2026-08-30T02:00:00Z", "2026-08-30T02:00:00Z", "read after run completion"),
    ),
)
def test_final_reporter_does_not_read_score_before_chronology(
    tmp_path,
    run_completed_at: str,
    score_read_at: str,
    message: str,
) -> None:
    reporter = FinalEvaluationReporter(JsonFinalEvaluationStore(tmp_path / "final.jsonl"))
    calls: list[str] = []
    with pytest.raises(ValueError, match=message):
        reporter.read_after_completion(
            candidate_artifact_id="candidate.reporter.invalid",
            run_id="run.reporter.invalid",
            candidate_frozen_at="2026-08-30T01:00:00Z",
            run_completed_at=run_completed_at,
            score_read_at=score_read_at,
            metric_name="task.pass_rate",
            score_reader=lambda: calls.append("read") or 0.5,
        )
    assert calls == []


def test_final_reporter_accepts_canonical_score_digest_payload(tmp_path) -> None:
    reporter = FinalEvaluationReporter(JsonFinalEvaluationStore(tmp_path / "final.jsonl"))
    digest = "a" * 64
    record = reporter.read_after_completion(
        candidate_artifact_id="candidate.reporter.payload",
        run_id="run.reporter.payload",
        candidate_frozen_at="2026-08-30T01:00:00Z",
        run_completed_at="2026-08-30T02:00:00Z",
        score_read_at="2026-08-30T03:00:00Z",
        metric_name="task.pass_rate",
        score_reader=lambda: {"metric_value": 0.5, "score_digest": digest},
    )
    assert record.score_digest == digest


def test_final_reporter_cli_uses_isolated_store(tmp_path, capsys) -> None:
    score_file = tmp_path / "official-score.json"
    score_file.write_text(json.dumps(0.625), encoding="utf-8")
    store_path = tmp_path / "final-evaluation.jsonl"
    assert main([
        "--store", str(store_path),
        "--score-file", str(score_file),
        "--candidate-artifact-id", "candidate.reporter.cli",
        "--run-id", "run.reporter.cli",
        "--candidate-frozen-at", "2026-08-30T01:00:00Z",
        "--run-completed-at", "2026-08-30T02:00:00Z",
        "--score-read-at", "2026-08-30T03:00:00Z",
        "--metric-name", "task.pass_rate",
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["evidence_plane"] == "final_evaluation"
    assert JsonFinalEvaluationStore(store_path).records()[0].metric_value == 0.625
