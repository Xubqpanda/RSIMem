from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rsimem.memory.final_evaluation import FinalEvaluationRecord


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


def test_final_evaluation_rejects_wrong_plane_and_non_numeric_score() -> None:
    with pytest.raises(ValueError, match="plane and source identity"):
        replace(_record(), evidence_plane="pure_process")
    with pytest.raises(TypeError, match="metric value"):
        _record(metric_value=True)
