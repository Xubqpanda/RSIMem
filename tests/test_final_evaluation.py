from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rsimem.memory.final_evaluation import FinalEvaluationRecord, JsonFinalEvaluationStore


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
