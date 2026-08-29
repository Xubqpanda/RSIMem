from __future__ import annotations

import json

import pytest

from rsimem.memory.signal_protocol import (
    JsonProcessSignalAnalysisProtocolStore,
    ProcessSignalAnalysisProtocol,
)


def _protocol() -> ProcessSignalAnalysisProtocol:
    return ProcessSignalAnalysisProtocol.create(
        training_family_ids=("SM02_constraint_retention",),
        task_template_group_ids=("sm02-process-pilot-train-v1",),
        provider_model="coding.tu-zi.com/gpt-5.6-luna",
        replicate_count=3,
        observation_window="window.pre_registered.v1",
        case_dedup_rule="logical_case_v1",
        no_signal_case_id="case.sm01.no_signal.v1",
    )


def test_protocol_is_canonical_and_result_independent() -> None:
    protocol = _protocol()
    assert ProcessSignalAnalysisProtocol.from_payload(
        json.loads(json.dumps(protocol.payload()))
    ) == protocol
    assert "grader" not in json.dumps(protocol.payload())
    assert "score" not in json.dumps(protocol.payload())


def test_protocol_rejects_duplicate_or_unfrozen_configuration() -> None:
    with pytest.raises(ValueError, match="nonempty and unique"):
        ProcessSignalAnalysisProtocol.create(
            training_family_ids=("SM02_constraint_retention", "SM02_constraint_retention"),
            task_template_group_ids=("sm02-process-pilot-train-v1",),
            provider_model="coding.tu-zi.com/gpt-5.6-luna",
            replicate_count=3,
            observation_window="window.pre_registered.v1",
            case_dedup_rule="logical_case_v1",
            no_signal_case_id="case.sm01.no_signal.v1",
        )
    payload = _protocol().payload()
    payload["replicate_count"] = 4
    with pytest.raises(ValueError, match="ID mismatch|non-canonical|malformed"):
        ProcessSignalAnalysisProtocol.from_payload(payload)


def test_protocol_store_freezes_once_and_replays_after_restart(tmp_path) -> None:
    path = tmp_path / "signal-protocol.json"
    protocol = _protocol()
    store = JsonProcessSignalAnalysisProtocolStore(path)
    assert store.get() is None
    assert store.freeze(protocol) is True
    assert store.freeze(protocol) is False
    restarted = JsonProcessSignalAnalysisProtocolStore(path)
    assert restarted.get() == protocol

    changed = ProcessSignalAnalysisProtocol.create(
        training_family_ids=protocol.training_family_ids,
        task_template_group_ids=protocol.task_template_group_ids,
        provider_model=protocol.provider_model,
        replicate_count=protocol.replicate_count + 1,
        observation_window=protocol.observation_window,
        case_dedup_rule=protocol.case_dedup_rule,
        no_signal_case_id=protocol.no_signal_case_id,
    )
    with pytest.raises(ValueError, match="already frozen"):
        restarted.freeze(changed)


def test_protocol_store_fails_closed_on_corruption(tmp_path) -> None:
    path = tmp_path / "signal-protocol.json"
    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed process signal protocol store"):
        JsonProcessSignalAnalysisProtocolStore(path).get()
