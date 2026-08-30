from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsimem.lifecycle import SegmentKind
from rsimem.memory.extraction_feedback import (
    DeploymentObservation,
    ObservableToolEvent,
)
from rsimem.memory.extraction_optimizer_builder import ExtractionFactContent
from rsimem.memory.extraction_optimizer_capture import (
    ExtractionOptimizerFeedbackCapture,
    ExtractionOptimizerSourceCapture,
    JsonExtractionOptimizerCaptureLog,
)
from rsimem.memory.extraction_source import (
    ExtractionSourceMessage,
    ExtractionSourceProjection,
)
from rsimem.memory.prompt_components import content_digest, text_digest


def _projection() -> ExtractionSourceProjection:
    message = ExtractionSourceMessage(
        "segment.capture-v1",
        "message.capture-v1",
        "user",
        "Remember my durable preference.",
        SegmentKind.MESSAGE,
    )
    identity = {
        "schema_version": 1,
        "schema": "completed-task-extraction-source-v1",
        "snapshot_id": "snapshot.capture-v1",
        "task_id": "task.capture-v1",
        "context_revision": "revision.capture-v1",
        "messages": [message.prompt_payload()],
        "source_message_ids": [message.source_message_id],
        "source_segment_ids": [message.segment_id],
        "omitted_segment_ids": [],
        "truncated_segment_ids": [],
        "max_content_chars": 1_000,
        "projected_content_chars": len(message.content),
    }
    digest = content_digest(identity)
    return ExtractionSourceProjection(
        f"extraction-source.{digest[:40]}",
        "snapshot.capture-v1",
        "task.capture-v1",
        "revision.capture-v1",
        (message,),
        (message.source_message_id,),
        (message.segment_id,),
        (),
        (),
        1_000,
        len(message.content),
        digest,
    )


def _observation(current_input: str = "Apply the saved preference.") -> DeploymentObservation:
    return DeploymentObservation(
        "observation.capture-v1",
        "SM01_preference_adoption",
        "eval_near",
        "task.future-v1",
        text_digest(current_input),
        (),
        ("preference.summary.tsv",),
        "owner\tpriority\ttask\tdue_date",
        (ObservableToolEvent(
            "tool-event.capture-v1",
            "sandbox_file_write",
            True,
            ("report.tsv",),
        ),),
        True,
    )


def test_optimizer_capture_round_trips_private_source_and_feedback(
    tmp_path: Path,
) -> None:
    projection = _projection()
    assert ExtractionSourceProjection.from_payload(projection.payload()) == projection
    observation = _observation()
    assert DeploymentObservation.from_payload(observation.payload()) == observation
    source = ExtractionOptimizerSourceCapture.create(
        captured_at="2026-08-28T01:00:00Z",
        source_record_id="source-record.capture-v1",
        source_record_digest="a" * 64,
        projection=projection,
        fact_contents=(ExtractionFactContent(
            "fact.capture-v1",
            "The user has a durable reporting preference.",
            True,
            None,
        ),),
    )
    feedback = ExtractionOptimizerFeedbackCapture.create(
        captured_at="2026-08-28T02:00:00Z",
        feedback_record_id="feedback-record.capture-v1",
        source_record_id=source.source_record_id,
        observation=observation,
        current_input="Apply the saved preference.",
    )
    path = tmp_path / ".rsimem" / "extraction_optimizer_capture.jsonl"
    log = JsonExtractionOptimizerCaptureLog(path)

    assert log.append(source) is True
    assert log.append(feedback) is True
    assert log.append(source) is False
    replay = ExtractionOptimizerSourceCapture.create(
        captured_at="2026-08-28T01:00:01Z",
        source_record_id=source.source_record_id,
        source_record_digest=source.source_record_digest,
        projection=source.projection,
        fact_contents=source.fact_contents,
    )
    assert log.append(replay) is False
    assert log.records() == (source, feedback)
    assert path.stat().st_mode & 0o777 == 0o600
    assert log.lock_path.stat().st_mode & 0o777 == 0o600
    raw = path.read_text(encoding="utf-8")
    assert "Remember my durable preference." in raw
    assert "Apply the saved preference." in raw


def test_optimizer_capture_fails_closed_on_conflict_and_malformed_entry(
    tmp_path: Path,
) -> None:
    source = ExtractionOptimizerSourceCapture.create(
        captured_at="2026-08-28T01:00:00Z",
        source_record_id="source-record.capture-v1",
        source_record_digest="a" * 64,
        projection=_projection(),
        fact_contents=(),
    )
    path = tmp_path / "capture.jsonl"
    log = JsonExtractionOptimizerCaptureLog(path)
    log.append(source)
    conflict = ExtractionOptimizerSourceCapture.create(
        captured_at="2026-08-28T01:00:01Z",
        source_record_id=source.source_record_id,
        source_record_digest=source.source_record_digest,
        projection=source.projection,
        fact_contents=(ExtractionFactContent(
            "fact.conflict-v1",
            "Different content.",
            True,
            None,
        ),),
    )
    with pytest.raises(ValueError, match="identity conflict"):
        log.append(conflict)

    path.write_text(json.dumps({"capture_kind": "source"}) + "\n", encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(ValueError, match="malformed optimizer capture log"):
        log.records()


def test_optimizer_capture_rejects_broad_permissions(tmp_path: Path) -> None:
    path = tmp_path / "capture.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    path.chmod(0o644)
    with pytest.raises(PermissionError, match="permissions are too broad"):
        JsonExtractionOptimizerCaptureLog(path).records()


def test_optimizer_capture_rejects_symlinked_paths(tmp_path: Path) -> None:
    target = tmp_path / "target.jsonl"
    target.write_text("sentinel\n", encoding="utf-8")
    path = tmp_path / "capture.jsonl"
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        JsonExtractionOptimizerCaptureLog(path).records()
    assert target.read_text(encoding="utf-8") == "sentinel\n"


def test_optimizer_capture_rejects_symlinked_lock(tmp_path: Path) -> None:
    path = tmp_path / "capture.jsonl"
    lock_target = tmp_path / "lock-target"
    lock_target.write_text("", encoding="utf-8")
    path.with_suffix(path.suffix + ".lock").symlink_to(lock_target)
    with pytest.raises(ValueError, match="lock.*symlink"):
        JsonExtractionOptimizerCaptureLog(path).records()
