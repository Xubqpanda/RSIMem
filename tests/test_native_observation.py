from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from experiments.legacy.native.native_observation import (
    NativeLifecycleEventType,
    ObservationStatus,
    extract_native_observations,
)
from test_native_execution_audit import _fixture


def test_native_observation_covers_all_events_without_evaluation_content(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    values = extract_native_observations(run=run, output_root=tmp_path)
    assert tuple(value.task_id for value in values) == run.native_task_ids
    for observation in values:
        assert tuple(value.event_type for value in observation.events) == tuple(
            NativeLifecycleEventType
        )
        status = {value.event_type: value.status for value in observation.events}
        assert status[NativeLifecycleEventType.SOURCE] is ObservationStatus.OBSERVED
        assert status[NativeLifecycleEventType.ADMISSION] is ObservationStatus.NOT_OBSERVED
        assert status[NativeLifecycleEventType.APPLICATION] is ObservationStatus.NOT_OBSERVED
        assert status[NativeLifecycleEventType.RETRIEVAL] is ObservationStatus.OBSERVED
        assert status[NativeLifecycleEventType.TOOL] is ObservationStatus.OBSERVED
        assert status[NativeLifecycleEventType.OUTCOME] is ObservationStatus.OBSERVED
        assert all(value.owner == "hermes-native" for value in observation.events)
        assert observation.events[0].parent_event_ids == ()
        assert all(
            value.parent_event_ids == (observation.events[index - 1].event_id,)
            for index, value in enumerate(observation.events[1:], start=1)
        )
        assert all(value.observation_cutoff == observation.trace_id for value in observation.events)
    serialized = json.dumps([value.payload() for value in values])
    for forbidden in ("task_score", "official_score", "grader", "answer", "final_response_text"):
        assert forbidden not in serialized


def test_native_observation_requires_episode_state_identity(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    sidecar = next((tmp_path / run.trace_directory).glob("**/native_episode_identity.json"))
    sidecar.unlink()
    with pytest.raises(ValueError, match="native_episode_identity.json"):
        extract_native_observations(run=run, output_root=tmp_path)


def test_native_observation_projects_only_final_output_digest(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    result_path = tmp_path / run.trace_directory / "sequence_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["episodes"][0]["final_response_text"] = "private final response"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    observation = extract_native_observations(run=run, output_root=tmp_path)[0]
    assert observation.final_output_digest == hashlib.sha256(
        b"private final response"
    ).hexdigest()
    serialized = json.dumps(observation.payload())
    assert "private final response" not in serialized


def test_native_observation_rejects_sidecar_trace_drift(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    sidecar = next((tmp_path / run.trace_directory).glob("**/native_episode_identity.json"))
    value = json.loads(sidecar.read_text())
    value["trace_id"] = "different-trace"
    sidecar.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="episode state identity is malformed"):
        extract_native_observations(run=run, output_root=tmp_path)
