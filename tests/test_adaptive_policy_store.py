from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rsimem.memory.adaptive_policy import (
    AdaptivePolicyState,
    DeterministicAdaptivePolicyLearner,
)
from rsimem.memory.adaptive_policy_store import JsonAdaptivePolicyStore
from test_adaptive_policy import _accepted, _config


def _artifact(*, seed: int = 17):
    _, dataset, gate = _accepted("used")
    config = replace(_config(dataset), seed=seed)
    artifact = DeterministicAdaptivePolicyLearner().learn(
        dataset,
        gate,
        config,
    )
    return dataset, artifact


def test_store_registration_transitions_restart_and_unique_active(tmp_path) -> None:
    dataset, first = _artifact(seed=17)
    _, second = _artifact(seed=18)
    path = tmp_path / "adaptive-policies.json"
    store = JsonAdaptivePolicyStore(
        path,
        trusted_root_policy_versions=(dataset.config.policy_version,),
    )

    registered, created = store.register(first)
    replay, replay_created = store.register(first)
    assert created is True
    assert replay_created is False
    assert registered == replay
    assert registered.state == AdaptivePolicyState.PROPOSAL

    validated, changed = store.transition(
        first.policy_version,
        to_state=AdaptivePolicyState.VALIDATED,
        transition_id="policy-transition.validate-first",
        reason_code="held_out_accepted",
    )
    assert changed is True
    assert validated.state == AdaptivePolicyState.VALIDATED
    duplicate, duplicate_changed = store.transition(
        first.policy_version,
        to_state=AdaptivePolicyState.VALIDATED,
        transition_id="policy-transition.validate-first",
        reason_code="held_out_accepted",
    )
    assert duplicate == validated
    assert duplicate_changed is False

    active, activated = store.transition(
        first.policy_version,
        to_state=AdaptivePolicyState.ACTIVE,
        transition_id="policy-transition.activate-first",
        reason_code="validation_passed",
    )
    assert activated is True
    assert active.state == AdaptivePolicyState.ACTIVE

    restarted = JsonAdaptivePolicyStore(
        path,
        trusted_root_policy_versions=(dataset.config.policy_version,),
    )
    snapshot = restarted.snapshot()
    assert snapshot.active_policy_version == first.policy_version
    assert snapshot.active == first
    assert len([
        record
        for record in snapshot.records
        if record.state == AdaptivePolicyState.ACTIVE
    ]) == 1

    restarted.register(second)
    restarted.transition(
        second.policy_version,
        to_state=AdaptivePolicyState.VALIDATED,
        transition_id="policy-transition.validate-second",
        reason_code="held_out_accepted",
    )
    with pytest.raises(ValueError, match="already active"):
        restarted.transition(
            second.policy_version,
            to_state=AdaptivePolicyState.ACTIVE,
            transition_id="policy-transition.activate-second-blocked",
            reason_code="validation_passed",
        )

    rolled_back, rolled = restarted.transition(
        first.policy_version,
        to_state=AdaptivePolicyState.ROLLED_BACK,
        transition_id="policy-transition.rollback-first",
        reason_code="operator_rollback",
    )
    assert rolled is True
    assert rolled_back.state == AdaptivePolicyState.ROLLED_BACK
    second_active, second_changed = restarted.transition(
        second.policy_version,
        to_state=AdaptivePolicyState.ACTIVE,
        transition_id="policy-transition.activate-second",
        reason_code="validation_passed",
    )
    assert second_changed is True
    assert second_active.state == AdaptivePolicyState.ACTIVE
    assert restarted.snapshot().active_policy_version == second.policy_version


def test_store_rejection_unknown_parent_and_transition_conflicts(tmp_path) -> None:
    dataset, proposal = _artifact(seed=20)
    store = JsonAdaptivePolicyStore(
        tmp_path / "policies.json",
        trusted_root_policy_versions=(dataset.config.policy_version,),
    )
    store.register(proposal)
    rejected, changed = store.transition(
        proposal.policy_version,
        to_state=AdaptivePolicyState.REJECTED,
        transition_id="policy-transition.reject",
        reason_code="validation_failed",
    )
    assert changed is True
    assert rejected.state == AdaptivePolicyState.REJECTED
    duplicate, duplicate_changed = store.transition(
        proposal.policy_version,
        to_state=AdaptivePolicyState.REJECTED,
        transition_id="policy-transition.reject",
        reason_code="validation_failed",
    )
    assert duplicate == rejected
    assert duplicate_changed is False
    with pytest.raises(ValueError, match="conflicting content"):
        store.transition(
            proposal.policy_version,
            to_state=AdaptivePolicyState.REJECTED,
            transition_id="policy-transition.reject",
            reason_code="operator_rejected",
        )
    with pytest.raises(ValueError, match="invalid.*transition"):
        store.transition(
            proposal.policy_version,
            to_state=AdaptivePolicyState.ACTIVE,
            transition_id="policy-transition.activate-rejected",
            reason_code="validation_passed",
        )

    untrusted = JsonAdaptivePolicyStore(
        tmp_path / "untrusted.json",
        trusted_root_policy_versions=("different-static-root-v1",),
    )
    with pytest.raises(ValueError, match="parent is unknown"):
        untrusted.register(proposal)


def test_store_fails_closed_on_artifact_and_active_pointer_corruption(tmp_path) -> None:
    dataset, first = _artifact(seed=30)
    _, second = _artifact(seed=31)
    path = tmp_path / "policies.json"
    store = JsonAdaptivePolicyStore(
        path,
        trusted_root_policy_versions=(dataset.config.policy_version,),
    )
    store.register(first)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["artifacts"][first.policy_version]["regularization"] = 99.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed adaptive policy artifact"):
        store.snapshot()

    path.unlink()
    store.register(first)
    store.register(second)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for version in (first.policy_version, second.policy_version):
        record = payload["records"][version]
        record["state"] = AdaptivePolicyState.ACTIVE.value
        record["revision"] = 2
        validate_id = f"policy-transition.corrupt-validate-{version[-6:]}"
        activate_id = f"policy-transition.corrupt-activate-{version[-6:]}"
        record["last_transition_id"] = activate_id
        record["reason_code"] = "validation_passed"
        payload["transitions"][validate_id] = {
            "transition_id": validate_id,
            "policy_version": version,
            "from_state": AdaptivePolicyState.PROPOSAL.value,
            "to_state": AdaptivePolicyState.VALIDATED.value,
            "reason_code": "held_out_accepted",
            "resulting_revision": 1,
        }
        payload["transitions"][activate_id] = {
            "transition_id": activate_id,
            "policy_version": version,
            "from_state": AdaptivePolicyState.VALIDATED.value,
            "to_state": AdaptivePolicyState.ACTIVE.value,
            "reason_code": "validation_passed",
            "resulting_revision": 2,
        }
    payload["active_policy_version"] = first.policy_version
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="active pointer"):
        store.snapshot()


def test_store_rejects_root_change_after_restart(tmp_path) -> None:
    dataset, artifact = _artifact(seed=40)
    path = tmp_path / "policies.json"
    JsonAdaptivePolicyStore(
        path,
        trusted_root_policy_versions=(dataset.config.policy_version,),
    ).register(artifact)
    changed_roots = JsonAdaptivePolicyStore(
        path,
        trusted_root_policy_versions=("different-static-root-v1",),
    )
    with pytest.raises(ValueError, match="trusted roots changed"):
        changed_roots.snapshot()


def test_store_rejects_missing_or_reordered_transition_history(tmp_path) -> None:
    dataset, artifact = _artifact(seed=50)
    path = tmp_path / "policies.json"
    store = JsonAdaptivePolicyStore(
        path,
        trusted_root_policy_versions=(dataset.config.policy_version,),
    )
    store.register(artifact)
    store.transition(
        artifact.policy_version,
        to_state=AdaptivePolicyState.VALIDATED,
        transition_id="policy-transition.validate-history",
        reason_code="held_out_accepted",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["transitions"] = {}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="history is incomplete"):
        store.snapshot()
