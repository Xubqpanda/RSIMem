from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace

import pytest

from rsimem.memory.adaptive_matched_validation import (
    AdaptiveRollbackEvidence,
    JsonMatchedValidationDecisionStore,
    MatchedAcceptanceCriteria,
    MatchedAdaptivePolicyActivationCoordinator,
    MatchedAdaptivePolicyValidator,
    MatchedPolicyObservation,
    MatchedPolicyVariant,
)
from rsimem.memory.adaptive_policy import AdaptivePolicyState
from rsimem.memory.adaptive_policy_store import JsonAdaptivePolicyStore
from rsimem.memory.adaptive_policy_validation import (
    AdaptiveAcceptanceCriteria,
    AdaptivePolicyLifecycleCoordinator,
    AdaptivePolicyValidator,
    JsonAdaptiveValidationDecisionStore,
    TimeOrderedAdaptiveSplitter,
)
from rsimem.memory.feedback_dataset import FeedbackLabel
from test_adaptive_policy_validation import _multi_dataset, _proposal
from test_feedback_dataset import POLICY_VERSION


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _observations(
    artifact,
    split,
    *,
    proposal_positive: bool = True,
    proposal_cost: float = 0.8,
):
    values = []
    for index, (example_id, episode_id) in enumerate(zip(
        split.validation_example_ids,
        split.validation_episode_ids,
    ), 1):
        common = {
            "split_id": split.split_id,
            "example_id": example_id,
            "episode_id": episode_id,
            "stability_failure": False,
            "uncertainty": 0.1,
            "evidence_cutoff": 20 + index,
            "task_input_digest": _digest(f"task-input-{index}"),
            "budget_id": "budget.held-out-v1",
        }
        values.extend((
            MatchedPolicyObservation(
                observation_id=f"matched.static.{index}",
                variant=MatchedPolicyVariant.STATIC,
                policy_version=artifact.parent_policy_version,
                label=FeedbackLabel.NEGATIVE,
                lifecycle_cost=1.0,
                evidence_id=f"evidence.static.{index}",
                **common,
            ),
            MatchedPolicyObservation(
                observation_id=f"matched.proposal.{index}",
                variant=MatchedPolicyVariant.PROPOSAL,
                policy_version=artifact.policy_version,
                label=(
                    FeedbackLabel.POSITIVE
                    if proposal_positive
                    else FeedbackLabel.NEGATIVE
                ),
                lifecycle_cost=proposal_cost,
                evidence_id=f"evidence.proposal.{index}",
                **common,
            ),
        ))
    return tuple(values)


def _offline_validated(
    tmp_path,
    *,
    suffix: str = "first",
    policy_version: str = POLICY_VERSION,
):
    dataset, gate = _multi_dataset(
        training_negative=True,
        policy_version=policy_version,
    )
    split = TimeOrderedAdaptiveSplitter().split(dataset, gate)
    artifact = _proposal(dataset, gate, split)
    offline = AdaptivePolicyValidator().evaluate(
        artifact,
        dataset,
        gate,
        split,
        AdaptiveAcceptanceCriteria(),
    )
    store = JsonAdaptivePolicyStore(
        tmp_path / f"policies-{suffix}.json",
        trusted_root_policy_versions=(dataset.config.policy_version,),
    )
    offline_coordinator = AdaptivePolicyLifecycleCoordinator(
        store,
        JsonAdaptiveValidationDecisionStore(tmp_path / f"offline-{suffix}"),
    )
    assert offline_coordinator.apply(artifact, offline).state == (
        AdaptivePolicyState.VALIDATED
    )
    return dataset, split, artifact, store, offline_coordinator


def test_matched_validation_replay_activation_restart_and_runtime_binding(
    tmp_path,
) -> None:
    dataset, split, artifact, store, offline = _offline_validated(tmp_path)
    observations = _observations(artifact, split)
    criteria = MatchedAcceptanceCriteria()
    validator = MatchedAdaptivePolicyValidator()
    decision = validator.evaluate(artifact, split, observations, criteria)
    replay = validator.evaluate(artifact, split, observations, criteria)

    assert decision == replay
    assert decision.accepted is True
    assert decision.quality_delta == 1.0
    assert decision.resolved_example_count == decision.matched_example_count
    assert decision.cost_ratio == pytest.approx(0.8)
    assert decision.used_policy_versions == (
        artifact.parent_policy_version,
        artifact.policy_version,
    )
    decisions = JsonMatchedValidationDecisionStore(tmp_path / "matched")
    coordinator = MatchedAdaptivePolicyActivationCoordinator(store, decisions)
    assert coordinator.apply(
        artifact,
        decision,
        split=split,
        observations=observations,
        criteria=criteria,
    ) == AdaptivePolicyState.ACTIVE
    assert coordinator.apply(
        artifact,
        decision,
        split=split,
        observations=observations,
        criteria=criteria,
    ) == AdaptivePolicyState.ACTIVE
    restarted = JsonAdaptivePolicyStore(
        store.path,
        trusted_root_policy_versions=(dataset.config.policy_version,),
    )
    assert restarted.snapshot().active == artifact
    binding = offline.bind_runtime_decision(
        "runtime-decision.matched",
        fallback_policy_version=dataset.config.policy_version,
    )
    assert binding.actual_policy_version == artifact.policy_version
    assert binding.adaptive is True
    decision_path = tmp_path / "matched" / f"{decision.decision_id}.json"
    assert json.loads(decision_path.read_text(encoding="utf-8")) == decision.payload()
    assert JsonMatchedValidationDecisionStore(tmp_path / "matched").get(
        decision.decision_id
    ) == decision


def test_matched_rejection_and_pair_drift_fail_closed(tmp_path) -> None:
    _, split, artifact, store, _ = _offline_validated(tmp_path, suffix="reject")
    observations = _observations(
        artifact,
        split,
        proposal_positive=False,
        proposal_cost=1.2,
    )
    criteria = MatchedAcceptanceCriteria(minimum_quality_delta=0.1)
    validator = MatchedAdaptivePolicyValidator()
    decision = validator.evaluate(artifact, split, observations, criteria)
    assert decision.accepted is False
    assert {"quality_criterion_failed", "cost_criterion_failed"} <= set(
        decision.reason_codes
    )
    coordinator = MatchedAdaptivePolicyActivationCoordinator(
        store,
        JsonMatchedValidationDecisionStore(tmp_path / "matched-rejected"),
    )
    assert coordinator.apply(
        artifact,
        decision,
        split=split,
        observations=observations,
        criteria=criteria,
    ) == AdaptivePolicyState.REJECTED
    assert store.snapshot().active is None

    duplicate_variant = (
        observations[0],
        replace(
            observations[1],
            observation_id="matched.static.duplicate",
            variant=MatchedPolicyVariant.STATIC,
            policy_version=artifact.parent_policy_version,
        ),
        *observations[2:],
    )
    with pytest.raises(ValueError, match="variant is duplicated"):
        validator.evaluate(artifact, split, duplicate_variant, criteria)
    drifted_budget = replace(observations[1], budget_id="budget.drift-v1")
    with pytest.raises(ValueError, match="pair identity differs"):
        validator.evaluate(
            artifact,
            split,
            (observations[0], drifted_budget, *observations[2:]),
            criteria,
        )
    wrong_episode = replace(observations[0], episode_id="learn-1")
    with pytest.raises(ValueError, match="outside validation split"):
        validator.evaluate(
            artifact,
            split,
            (wrong_episode, *observations[1:]),
            criteria,
        )
    with pytest.raises(ValueError, match="decision ID mismatch"):
        replace(
            decision,
            decision_id="matched-validation.tampered",
        )


def test_matched_activation_requires_offline_validation(tmp_path) -> None:
    dataset, gate = _multi_dataset(training_negative=True)
    split = TimeOrderedAdaptiveSplitter().split(dataset, gate)
    artifact = _proposal(dataset, gate, split)
    observations = _observations(artifact, split)
    criteria = MatchedAcceptanceCriteria()
    decision = MatchedAdaptivePolicyValidator().evaluate(
        artifact,
        split,
        observations,
        criteria,
    )
    store = JsonAdaptivePolicyStore(
        tmp_path / "unvalidated.json",
        trusted_root_policy_versions=(dataset.config.policy_version,),
    )
    coordinator = MatchedAdaptivePolicyActivationCoordinator(
        store,
        JsonMatchedValidationDecisionStore(tmp_path / "unvalidated-decisions"),
    )
    with pytest.raises(ValueError, match="requires offline validated"):
        coordinator.apply(
            artifact,
            decision,
            split=split,
            observations=observations,
            criteria=criteria,
        )
    assert store.snapshot().active is None


def test_matched_activation_crash_is_safe_and_retryable(tmp_path, monkeypatch) -> None:
    _, split, artifact, store, _ = _offline_validated(tmp_path, suffix="crash")
    observations = _observations(artifact, split)
    criteria = MatchedAcceptanceCriteria()
    decision = MatchedAdaptivePolicyValidator().evaluate(
        artifact,
        split,
        observations,
        criteria,
    )
    original_write = store._write_unlocked

    def fail_write(payload):
        raise RuntimeError("simulated matched activation crash")

    monkeypatch.setattr(store, "_write_unlocked", fail_write)
    coordinator = MatchedAdaptivePolicyActivationCoordinator(
        store,
        JsonMatchedValidationDecisionStore(tmp_path / "crash-decisions"),
    )
    with pytest.raises(RuntimeError, match="matched activation crash"):
        coordinator.apply(
            artifact,
            decision,
            split=split,
            observations=observations,
            criteria=criteria,
        )
    crashed = JsonAdaptivePolicyStore(
        store.path,
        trusted_root_policy_versions=(artifact.parent_policy_version,),
    ).snapshot()
    assert crashed.active is None
    assert crashed.records[0].state == AdaptivePolicyState.VALIDATED

    monkeypatch.setattr(store, "_write_unlocked", original_write)
    assert coordinator.apply(
        artifact,
        decision,
        split=split,
        observations=observations,
        criteria=criteria,
    ) == AdaptivePolicyState.ACTIVE


@pytest.mark.parametrize("automatic", (False, True))
def test_matched_operator_and_automatic_rollback_are_idempotent(
    tmp_path,
    automatic,
) -> None:
    _, split, artifact, store, _ = _offline_validated(
        tmp_path,
        suffix=f"rollback-{automatic}",
    )
    observations = _observations(artifact, split)
    criteria = MatchedAcceptanceCriteria()
    decision = MatchedAdaptivePolicyValidator().evaluate(
        artifact,
        split,
        observations,
        criteria,
    )
    coordinator = MatchedAdaptivePolicyActivationCoordinator(
        store,
        JsonMatchedValidationDecisionStore(tmp_path / f"matched-{automatic}"),
    )
    coordinator.apply(
        artifact,
        decision,
        split=split,
        observations=observations,
        criteria=criteria,
    )
    evidence = AdaptiveRollbackEvidence.create(
        policy_version=artifact.policy_version,
        automatic=automatic,
        reason_codes=(
            ("stability_regression",) if automatic else ("operator_requested",)
        ),
        evidence_cutoff=30,
    )
    first = coordinator.rollback(artifact, evidence)
    replay = coordinator.rollback(artifact, evidence)
    assert first == replay == AdaptivePolicyState.ROLLED_BACK
    assert store.snapshot().active is None


def test_matched_validation_surface_has_no_official_score_input() -> None:
    assert set(inspect.signature(
        MatchedAdaptivePolicyValidator.evaluate
    ).parameters) == {"self", "artifact", "split", "observations", "criteria"}
    dataset, gate = _multi_dataset(training_negative=True)
    split = TimeOrderedAdaptiveSplitter().split(dataset, gate)
    artifact = _proposal(dataset, gate, split)
    serialized = json.dumps(
        [item.payload() for item in _observations(artifact, split)],
        sort_keys=True,
    )
    for forbidden in ('"score"', '"grader"', '"answer"', '"expectation"'):
        assert forbidden not in serialized


def test_unresolved_matched_pairs_cannot_satisfy_quality_gate(tmp_path) -> None:
    _, split, artifact, _, _ = _offline_validated(
        tmp_path,
        suffix="unresolved",
    )
    observations = tuple(
        replace(item, label=FeedbackLabel.UNRESOLVED)
        for item in _observations(artifact, split)
    )
    decision = MatchedAdaptivePolicyValidator().evaluate(
        artifact,
        split,
        observations,
        MatchedAcceptanceCriteria(),
    )
    assert decision.accepted is False
    assert decision.resolved_example_count == 0
    assert "insufficient_resolved_examples" in decision.reason_codes
