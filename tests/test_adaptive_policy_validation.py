from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace

import pytest

from rsimem.memory.adaptive_policy import (
    AdaptiveParameterName,
    AdaptiveParameterSpec,
    AdaptivePolicyState,
    AdaptiveTrainingConfig,
    DeterministicAdaptivePolicyLearner,
)
from rsimem.memory.adaptive_policy_store import JsonAdaptivePolicyStore
from rsimem.memory.adaptive_policy_validation import (
    AdaptiveAcceptanceCriteria,
    AdaptiveRollbackReceipt,
    AdaptivePolicyLifecycleCoordinator,
    AdaptivePolicyValidator,
    AdaptiveSplitConfig,
    JsonAdaptiveValidationDecisionStore,
    MatchedActivationReceipt,
    TimeOrderedAdaptiveSplitter,
)
from rsimem.memory.attribution import DeterministicFirstAttributor
from rsimem.memory.feedback_dataset import evaluate_feedback_dataset_stage_gate
from test_feedback_dataset import _dataset, _graph


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, value: object) -> str:
    digest = hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()
    return f"{prefix}.{digest[:40]}"


def _payload_digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _rebind_example(example, ordinal: int):
    rebound = replace(
        example,
        example_id="feedback-example.placeholder",
        source_episode_id=f"learn-{ordinal}",
        observation_episode_ids=(f"learn-{ordinal}", f"future-{ordinal}"),
        session_ids=(f"session-learn-{ordinal}", f"session-future-{ordinal}"),
        task_ids=(f"task-learn-{ordinal}", f"task-future-{ordinal}"),
    )
    payload = rebound.payload()
    payload.pop("example_id")
    return replace(rebound, example_id=_stable_id("feedback-example", payload))


def _multi_dataset(*, training_negative: bool):
    negative_graph = _graph("injected_not_used")
    negative_report = DeterministicFirstAttributor().attribute(negative_graph)
    negative = _dataset(negative_graph, reports=(negative_report,))
    positive_graph = _graph("used")
    positive = _dataset(positive_graph)
    training_source = negative if training_negative else positive
    examples = (
        _rebind_example(training_source.examples[0], 1),
        _rebind_example(negative.examples[0], 2),
    )
    dataset = replace(
        training_source,
        dataset_id="feedback-dataset.placeholder",
        examples=examples,
    )
    payload = dataset.payload()
    payload.pop("dataset_id")
    dataset = replace(dataset, dataset_id=_stable_id("feedback-dataset", payload))
    source_gate = evaluate_feedback_dataset_stage_gate(
        training_source,
        training_source is negative and negative_graph or positive_graph,
        expected_config=training_source.config,
        attribution_reports=((negative_report,) if training_source is negative else ()),
    )
    dataset_payload_digest = _payload_digest(dataset.payload())
    gate = replace(
        source_gate,
        dataset_id=dataset.dataset_id,
        dataset_payload_digest=dataset_payload_digest,
        replay_dataset_id=dataset.dataset_id,
    )
    return dataset, gate


def _proposal(dataset, gate, split):
    config = AdaptiveTrainingConfig(
        parent_policy_version=dataset.config.policy_version,
        seed=23,
        parameters=(AdaptiveParameterSpec(
            parameter_id="parameter.fact",
            name=AdaptiveParameterName.RETRIEVAL_ACCEPT_THRESHOLD,
            baseline_value=0.35,
            prompt_ref="mem0-flat.retrieval",
        ),),
        training_example_ids=split.training_example_ids,
        minimum_resolved_examples=1,
        maximum_missing_propensity_rate=1.0,
    )
    return DeterministicAdaptivePolicyLearner().learn(dataset, gate, config)


def test_time_ordered_split_is_deterministic_complete_and_auditable() -> None:
    dataset, gate = _multi_dataset(training_negative=True)
    splitter = TimeOrderedAdaptiveSplitter()
    first = splitter.split(dataset, gate)
    replay = splitter.split(dataset, gate)

    assert first == replay
    assert first.payload() == replay.payload()
    assert first.training_example_ids == (dataset.examples[0].example_id,)
    assert first.validation_example_ids == (dataset.examples[1].example_id,)
    assert first.training_episode_ids == ("learn-1",)
    assert first.validation_episode_ids == ("learn-2",)
    assert first.training_cutoff_example_id == dataset.examples[0].example_id
    assert set(first.training_example_ids) | set(first.validation_example_ids) == {
        example.example_id for example in dataset.examples
    }

    with pytest.raises(ValueError, match="examples overlap"):
        replace(first, validation_example_ids=first.training_example_ids)
    single_graph = _graph("used")
    single = _dataset(single_graph)
    single_gate = evaluate_feedback_dataset_stage_gate(
        single,
        single_graph,
        expected_config=single.config,
    )
    with pytest.raises(ValueError, match="insufficient chronological groups"):
        splitter.split(single, single_gate, AdaptiveSplitConfig())


def test_validation_acceptance_is_replayable_and_uses_same_heldout_evidence() -> None:
    dataset, gate = _multi_dataset(training_negative=True)
    split = TimeOrderedAdaptiveSplitter().split(dataset, gate)
    artifact = _proposal(dataset, gate, split)
    criteria = AdaptiveAcceptanceCriteria()
    validator = AdaptivePolicyValidator()
    decision = validator.evaluate(artifact, dataset, gate, split, criteria)
    replay = validator.evaluate(artifact, dataset, gate, split, criteria)

    assert decision == replay
    assert decision.accepted is True
    assert decision.resulting_state == AdaptivePolicyState.VALIDATED
    assert decision.used_policy_versions == (
        artifact.parent_policy_version,
        artifact.policy_version,
    )
    assert decision.training_example_ids == artifact.training_example_ids
    assert decision.validation_example_ids == split.validation_example_ids
    assert decision.metrics.quality_delta is not None
    assert decision.metrics.quality_delta > 0
    assert decision.metrics.cost_delta <= criteria.maximum_cost_delta
    assert decision.metrics.stability_delta <= criteria.maximum_stability_delta
    assert decision.metrics.uncertainty <= criteria.maximum_uncertainty
    assert validator.replay_matches(
        decision,
        artifact,
        dataset,
        gate,
        split,
        criteria,
    )
    serialized = json.dumps(decision.payload(), sort_keys=True)
    for forbidden in ('"score"', '"grader"', '"expectation"', '"answer"'):
        assert forbidden not in serialized
    assert set(inspect.signature(validator.evaluate).parameters) == {
        "artifact",
        "dataset",
        "gate",
        "split",
        "criteria",
    }


def test_validation_decision_store_restarts_and_rejects_tampering(tmp_path) -> None:
    dataset, gate = _multi_dataset(training_negative=True)
    split = TimeOrderedAdaptiveSplitter().split(dataset, gate)
    artifact = _proposal(dataset, gate, split)
    decision = AdaptivePolicyValidator().evaluate(
        artifact,
        dataset,
        gate,
        split,
        AdaptiveAcceptanceCriteria(),
    )
    store = JsonAdaptiveValidationDecisionStore(tmp_path / "decisions")
    path, created = store.put(decision)
    replay_path, replay_created = store.put(decision)
    assert created is True
    assert replay_created is False
    assert replay_path == path
    restarted = JsonAdaptiveValidationDecisionStore(tmp_path / "decisions")
    assert restarted.get(decision.decision_id) == decision
    assert restarted.all() == (decision,)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["accepted"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed adaptive validation decision"):
        restarted.get(decision.decision_id)


def test_quality_regression_is_rejected_and_remains_inactive(tmp_path) -> None:
    dataset, gate = _multi_dataset(training_negative=False)
    split = TimeOrderedAdaptiveSplitter().split(dataset, gate)
    artifact = _proposal(dataset, gate, split)
    decision = AdaptivePolicyValidator().evaluate(
        artifact,
        dataset,
        gate,
        split,
        AdaptiveAcceptanceCriteria(),
    )
    assert decision.accepted is False
    assert decision.resulting_state == AdaptivePolicyState.REJECTED
    assert "quality_regression" in decision.reason_codes

    policy_store = JsonAdaptivePolicyStore(
        tmp_path / "policies.json",
        trusted_root_policy_versions=(dataset.config.policy_version,),
    )
    coordinator = AdaptivePolicyLifecycleCoordinator(
        policy_store,
        JsonAdaptiveValidationDecisionStore(tmp_path / "decisions"),
    )
    rejected = coordinator.apply(artifact, decision)
    replay = coordinator.apply(artifact, decision)
    assert rejected == replay
    assert rejected.state == AdaptivePolicyState.REJECTED
    assert policy_store.snapshot().active is None


def test_matched_receipt_activates_then_safety_receipt_rolls_back(tmp_path) -> None:
    dataset, gate = _multi_dataset(training_negative=True)
    split = TimeOrderedAdaptiveSplitter().split(dataset, gate)
    artifact = _proposal(dataset, gate, split)
    decision = AdaptivePolicyValidator().evaluate(
        artifact,
        dataset,
        gate,
        split,
        AdaptiveAcceptanceCriteria(),
    )
    policy_path = tmp_path / "policies.json"
    policy_store = JsonAdaptivePolicyStore(
        policy_path,
        trusted_root_policy_versions=(dataset.config.policy_version,),
    )
    decision_store = JsonAdaptiveValidationDecisionStore(tmp_path / "decisions")
    coordinator = AdaptivePolicyLifecycleCoordinator(policy_store, decision_store)

    active = coordinator.apply(artifact, decision)
    duplicate = coordinator.apply(artifact, decision)
    assert active == duplicate
    assert active.state == AdaptivePolicyState.VALIDATED
    restarted = JsonAdaptivePolicyStore(
        policy_path,
        trusted_root_policy_versions=(dataset.config.policy_version,),
    )
    assert restarted.snapshot().active is None
    binding = coordinator.bind_runtime_decision(
        "runtime-decision.first",
        fallback_policy_version=dataset.config.policy_version,
    )
    assert binding.actual_policy_version == dataset.config.policy_version
    assert binding.adaptive is False

    receipt = MatchedActivationReceipt.create(
        validation_decision_id=decision.decision_id,
        artifact_id=artifact.artifact_id,
        parent_policy_version=artifact.parent_policy_version,
        proposal_policy_version=artifact.policy_version,
        task_group_ids=split.validation_task_ids,
        static_trace_ids=tuple(
            f"trace.static.{index}"
            for index, _ in enumerate(split.validation_task_ids, 1)
        ),
        proposal_trace_ids=tuple(
            f"trace.proposal.{index}"
            for index, _ in enumerate(split.validation_task_ids, 1)
        ),
        budget_id="budget.held-out-v1",
        matched_audit_id="audit.matched-held-out-v1",
        accepted=True,
        reason_codes=("matched_execution_passed",),
    )
    activated = coordinator.activate(artifact, decision, receipt)
    duplicate_activation = coordinator.activate(artifact, decision, receipt)
    assert activated == duplicate_activation
    assert activated.state == AdaptivePolicyState.ACTIVE
    active_binding = coordinator.bind_runtime_decision(
        "runtime-decision.adaptive",
        fallback_policy_version=dataset.config.policy_version,
    )
    assert active_binding.actual_policy_version == artifact.policy_version
    assert active_binding.adaptive is True

    rollback = AdaptiveRollbackReceipt.create(
        policy_version=artifact.policy_version,
        automatic=True,
        evidence_ids=("evidence.stability-regression",),
        reason_code="safety_regression",
    )
    rolled = coordinator.rollback(rollback)
    replay_rollback = coordinator.rollback(rollback)
    assert rolled == replay_rollback
    assert rolled.state == AdaptivePolicyState.ROLLED_BACK
    assert policy_store.snapshot().active is None
    with pytest.raises(ValueError, match="receipt identity"):
        replace(receipt, budget_id="budget.drift")


def test_offline_validation_crash_leaves_proposal_and_retry_is_stable(
    tmp_path,
    monkeypatch,
) -> None:
    dataset, gate = _multi_dataset(training_negative=True)
    split = TimeOrderedAdaptiveSplitter().split(dataset, gate)
    artifact = _proposal(dataset, gate, split)
    decision = AdaptivePolicyValidator().evaluate(
        artifact,
        dataset,
        gate,
        split,
        AdaptiveAcceptanceCriteria(),
    )
    path = tmp_path / "policies.json"
    store = JsonAdaptivePolicyStore(
        path,
        trusted_root_policy_versions=(dataset.config.policy_version,),
    )
    store.register(artifact)
    coordinator = AdaptivePolicyLifecycleCoordinator(
        store,
        JsonAdaptiveValidationDecisionStore(tmp_path / "decisions"),
    )
    original_write = store._write_unlocked
    calls = 0

    def fail_activation(payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated validation crash")
        return original_write(payload)

    monkeypatch.setattr(store, "_write_unlocked", fail_activation)
    with pytest.raises(RuntimeError, match="validation crash"):
        coordinator.apply(artifact, decision)
    crashed = JsonAdaptivePolicyStore(
        path,
        trusted_root_policy_versions=(dataset.config.policy_version,),
    ).snapshot()
    assert crashed.active is None
    assert crashed.records[0].state == AdaptivePolicyState.PROPOSAL

    monkeypatch.setattr(store, "_write_unlocked", original_write)
    recovered = coordinator.apply(artifact, decision)
    assert recovered.state == AdaptivePolicyState.VALIDATED
    assert store.snapshot().active_policy_version is None


def test_matched_activation_crash_leaves_validated_and_retry_is_stable(
    tmp_path,
    monkeypatch,
) -> None:
    dataset, gate = _multi_dataset(training_negative=True)
    split = TimeOrderedAdaptiveSplitter().split(dataset, gate)
    artifact = _proposal(dataset, gate, split)
    decision = AdaptivePolicyValidator().evaluate(
        artifact,
        dataset,
        gate,
        split,
        AdaptiveAcceptanceCriteria(),
    )
    path = tmp_path / "policies.json"
    store = JsonAdaptivePolicyStore(
        path,
        trusted_root_policy_versions=(dataset.config.policy_version,),
    )
    coordinator = AdaptivePolicyLifecycleCoordinator(
        store,
        JsonAdaptiveValidationDecisionStore(tmp_path / "decisions"),
    )
    coordinator.apply(artifact, decision)
    receipt = MatchedActivationReceipt.create(
        validation_decision_id=decision.decision_id,
        artifact_id=artifact.artifact_id,
        parent_policy_version=artifact.parent_policy_version,
        proposal_policy_version=artifact.policy_version,
        task_group_ids=split.validation_task_ids,
        static_trace_ids=tuple(
            f"trace.static.{index}"
            for index, _ in enumerate(split.validation_task_ids, 1)
        ),
        proposal_trace_ids=tuple(
            f"trace.proposal.{index}"
            for index, _ in enumerate(split.validation_task_ids, 1)
        ),
        budget_id="budget.held-out-v1",
        matched_audit_id="audit.matched-held-out-v1",
        accepted=True,
        reason_codes=("matched_execution_passed",),
    )
    original_write = store._write_unlocked

    def fail_activation(payload):
        raise RuntimeError("simulated matched activation crash")

    monkeypatch.setattr(store, "_write_unlocked", fail_activation)
    with pytest.raises(RuntimeError, match="activation crash"):
        coordinator.activate(artifact, decision, receipt)
    crashed = JsonAdaptivePolicyStore(
        path,
        trusted_root_policy_versions=(dataset.config.policy_version,),
    ).snapshot()
    assert crashed.active is None
    assert crashed.records[0].state == AdaptivePolicyState.VALIDATED

    monkeypatch.setattr(store, "_write_unlocked", original_write)
    recovered = coordinator.activate(artifact, decision, receipt)
    assert recovered.state == AdaptivePolicyState.ACTIVE
    assert store.snapshot().active_policy_version == artifact.policy_version
