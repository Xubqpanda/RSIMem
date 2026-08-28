from __future__ import annotations

import json

import pytest

from rsimem.extraction_validation_runtime import (
    EXTRACTION_MATCHED_TRIAL_SCOPE,
    EXTRACTION_PRODUCTION_SCOPE,
    EXTRACTION_TRIAL_CONFIG_FILE,
    EXTRACTION_TRIAL_OFFLINE_DECISION_FILE,
    EXTRACTION_TRIAL_POLICY_STORE_FILE,
    load_extraction_matched_trial_profile,
    load_extraction_runtime_profile,
    prepare_extraction_matched_trial_runtime,
)
from rsimem.memory.extraction_offline_validation import (
    ExtractionOfflineValidationDecision,
)
from rsimem.memory.extraction_policy_store import (
    ExtractionPolicyState,
    JsonExtractionPolicyStore,
)
from rsimem.memory_systems.mem0_flat import MEM0_FLAT_EXTRACTION_SLOT
from test_extraction_matched_activation import _offline_decision
from test_extraction_offline_validation import _candidate, _parent


def test_trial_runtime_isolated_active_replay_and_scope_rejection(tmp_path) -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    offline = _offline_decision(parent, candidate)
    output = tmp_path / "trial"

    first = prepare_extraction_matched_trial_runtime(
        parent=parent,
        candidate=candidate,
        offline_decision=offline,
        output_root=output,
    )
    replay = prepare_extraction_matched_trial_runtime(
        parent=parent,
        candidate=candidate,
        offline_decision=offline,
        output_root=output,
    )
    assert replay == first
    assert first["deploymentScope"] == EXTRACTION_MATCHED_TRIAL_SCOPE
    assert first["officialEvaluation"] is False
    assert first["validationOnly"] is True
    assert first["productionActivationAllowed"] is False

    resolved = load_extraction_matched_trial_profile(
        output / EXTRACTION_TRIAL_CONFIG_FILE
    )
    assert resolved.parent == parent
    assert resolved.candidate == candidate
    assert resolved.offline_decision == offline
    profile = resolved.profile()
    assert profile["preparation"] == "extraction_matched_trial_store"
    assert profile["candidateArtifactId"] == candidate.artifact_id
    assert profile["productionActivationAllowed"] is False
    with pytest.raises(ValueError, match="cannot be used for this scope"):
        load_extraction_runtime_profile(
            output / EXTRACTION_TRIAL_CONFIG_FILE,
            required_scope=EXTRACTION_PRODUCTION_SCOPE,
        )

    snapshot = JsonExtractionPolicyStore(
        output / EXTRACTION_TRIAL_POLICY_STORE_FILE,
        trusted_root=parent,
        slot=MEM0_FLAT_EXTRACTION_SLOT,
    ).snapshot()
    assert snapshot.active == candidate
    record = next(
        value
        for value in snapshot.records
        if value.artifact_id == candidate.artifact_id
    )
    assert record.state == ExtractionPolicyState.ACTIVE
    assert record.reason_code == "matched_validation_trial"


def test_trial_runtime_config_is_content_free_and_offline_decision_replays(
    tmp_path,
) -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    offline = _offline_decision(parent, candidate)
    output = tmp_path / "trial"
    prepare_extraction_matched_trial_runtime(
        parent=parent,
        candidate=candidate,
        offline_decision=offline,
        output_root=output,
    )
    config_text = (output / EXTRACTION_TRIAL_CONFIG_FILE).read_text(
        encoding="utf-8"
    )
    for forbidden in (
        candidate.compiled_body,
        "task_score",
        "official_score",
        "grader",
        "answer_key",
        "lifecycle_cost",
        "token_cost",
    ):
        assert forbidden not in config_text
    offline_payload = json.loads(
        (output / EXTRACTION_TRIAL_OFFLINE_DECISION_FILE).read_text(
            encoding="utf-8"
        )
    )
    assert ExtractionOfflineValidationDecision.from_payload(offline_payload) == offline


def test_trial_runtime_rejects_unaccepted_or_mismatched_offline_candidate(
    tmp_path,
) -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    rejected = _offline_decision(parent, candidate, accepted=False)
    with pytest.raises(ValueError, match="requires offline acceptance"):
        prepare_extraction_matched_trial_runtime(
            parent=parent,
            candidate=candidate,
            offline_decision=rejected,
            output_root=tmp_path / "rejected",
        )
    other = _candidate(
        parent=parent,
        text="Keep stable preferences that can support future work.",
    )
    with pytest.raises(ValueError, match="offline join differs"):
        prepare_extraction_matched_trial_runtime(
            parent=parent,
            candidate=other,
            offline_decision=_offline_decision(parent, candidate),
            output_root=tmp_path / "mismatch",
        )
    other_parent = type(parent).create_root(
        slot=MEM0_FLAT_EXTRACTION_SLOT,
        policy_version="other-root-v1",
        spec=parent.spec,
        max_body_chars=parent.max_body_chars,
        source_provenance="other-trusted-source",
    )
    with pytest.raises(ValueError, match="parent is not trusted root"):
        prepare_extraction_matched_trial_runtime(
            parent=other_parent,
            candidate=candidate,
            offline_decision=_offline_decision(parent, candidate),
            output_root=tmp_path / "wrong-root",
        )


@pytest.mark.parametrize(
    ("target", "mutate", "message"),
    (
        (
            EXTRACTION_TRIAL_CONFIG_FILE,
            lambda value: value.__setitem__("officialEvaluation", True),
            "config identity mismatch",
        ),
        (
            EXTRACTION_TRIAL_OFFLINE_DECISION_FILE,
            lambda value: value.__setitem__("reason_codes", ["tampered"]),
            "file digest mismatch",
        ),
        (
            EXTRACTION_TRIAL_POLICY_STORE_FILE,
            lambda value: value.__setitem__("active_artifact_id", None),
            "file digest mismatch",
        ),
    ),
)
def test_trial_runtime_tampering_fails_closed(tmp_path, target, mutate, message) -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    output = tmp_path / target.replace(".json", "")
    prepare_extraction_matched_trial_runtime(
        parent=parent,
        candidate=candidate,
        offline_decision=_offline_decision(parent, candidate),
        output_root=output,
    )
    path = output / target
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_extraction_matched_trial_profile(output / EXTRACTION_TRIAL_CONFIG_FILE)


def test_trial_runtime_rejects_unknown_loader_scope(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown extraction runtime scope"):
        load_extraction_runtime_profile(
            tmp_path / "missing.json",
            required_scope="official_final",
        )
