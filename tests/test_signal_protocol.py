from __future__ import annotations

import json

import pytest

from rsimem.memory.signal_protocol import (
    JsonProcessSignalAnalysisProtocolStore,
    ProcessSignalAnalysisProtocol,
    protocol_for_extraction_manifest,
    validate_protocol_for_extraction_manifest,
)


def _protocol() -> ProcessSignalAnalysisProtocol:
    return ProcessSignalAnalysisProtocol.create(
        training_family_ids=("SM02_constraint_retention",),
        task_template_group_ids=("sm02-process-pilot-train-v1",),
        task_manifest_digest="a" * 64,
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
            task_manifest_digest="a" * 64,
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
        task_manifest_digest=protocol.task_manifest_digest,
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


def test_protocol_store_fails_closed_on_symlink(tmp_path) -> None:
    target = tmp_path / "target.json"
    target.write_text(json.dumps(_protocol().payload()), encoding="utf-8")
    path = tmp_path / "signal-protocol.json"
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink|malformed process signal protocol store"):
        JsonProcessSignalAnalysisProtocolStore(path).get()


def test_protocol_manifest_binding_is_deterministic_and_result_independent() -> None:
    manifest = {
        "split": {
            "familyId": "SM02_constraint_retention",
            "taskTemplateGroupId": "sm02-process-pilot-train-v1",
            "taskManifestDigest": "a" * 64,
        },
        "replicates": 3,
        "modelProfile": {
            "resolved": {
                "providerBaseUrl": "https://coding.tu-zi.com/v1",
                "modelId": "gpt-5.6-luna",
            },
        },
    }
    protocol = protocol_for_extraction_manifest(manifest)
    assert validate_protocol_for_extraction_manifest(protocol, manifest) == protocol
    assert protocol.training_family_ids == ("SM02_constraint_retention",)
    assert protocol.task_template_group_ids == ("sm02-process-pilot-train-v1",)
    assert protocol.task_manifest_digest == "a" * 64
    assert protocol.provider_model == "https://coding.tu-zi.com/v1/gpt-5.6-luna"

    changed = dict(manifest)
    changed["replicates"] = 4
    with pytest.raises(ValueError, match="does not match extraction manifest"):
        validate_protocol_for_extraction_manifest(protocol, changed)

    drifted_manifest = dict(manifest)
    drifted_manifest["split"] = dict(manifest["split"])
    drifted_manifest["split"]["taskManifestDigest"] = "b" * 64
    with pytest.raises(ValueError, match="does not match extraction manifest"):
        validate_protocol_for_extraction_manifest(protocol, drifted_manifest)
