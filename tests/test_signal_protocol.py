from __future__ import annotations

import json

import pytest

from rsimem.memory.signal_protocol import ProcessSignalAnalysisProtocol


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
