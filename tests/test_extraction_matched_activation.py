from __future__ import annotations

import json
import inspect

import pytest

from rsimem.memory.extraction_feedback import (
    ExtractionFeedbackLabel,
    ExtractionSetStatus,
)
from rsimem.memory.extraction_matched_activation import (
    ExtractionMatchedActivationCoordinator,
    ExtractionMatchedConstraint,
    ExtractionMatchedTrialDecision,
    ExtractionMatchedTrialEvaluator,
    ExtractionMatchedTrialStatus,
    ExtractionRollbackEvidence,
    ExtractionRollbackTrigger,
    JsonExtractionMatchedTrialDecisionStore,
    JsonExtractionRollbackEvidenceStore,
)
from rsimem.memory.extraction_offline_validation import (
    ExtractionPromptOfflineValidator,
)
from rsimem.memory.extraction_policy_store import (
    ExtractionPolicyState,
    JsonExtractionPolicyStore,
)
from rsimem.memory.extraction_prompt_validation import (
    ExtractionPromptValidationSplit,
    ExtractionSplitAssignment,
    ExtractionValidationObservation,
    ExtractionValidationSafetyEvidence,
    ExtractionValidationSplitRole,
    ExtractionValidationVariant,
)
from rsimem.memory.prompt_components import text_digest
from rsimem.memory_systems.mem0_flat import MEM0_FLAT_EXTRACTION_SLOT
from test_extraction_offline_validation import (
    _candidate,
    _criteria,
    _pairs,
    _parent,
    _safety_and_suite,
    _split,
)


def _offline_decision(parent, candidate, *, accepted: bool = True):
    safety, suite = _safety_and_suite(parent, candidate)
    parent_labels = (
        ExtractionFeedbackLabel.USEFUL,
        ExtractionFeedbackLabel.HARMFUL,
        ExtractionFeedbackLabel.USEFUL,
    )
    candidate_labels = (
        (ExtractionFeedbackLabel.USEFUL,) * 3
        if accepted
        else parent_labels
    )
    return ExtractionPromptOfflineValidator().evaluate(
        parent=parent,
        candidate=candidate,
        split=_split(),
        observations=_pairs(
            parent,
            candidate,
            parent_labels,
            candidate_labels,
            changed=accepted,
        ),
        criteria=_criteria(),
        static_safety=safety,
        deterministic_suite=suite,
    )


def _matched_split() -> ExtractionPromptValidationSplit:
    return ExtractionPromptValidationSplit(
        "split.matched-v1",
        (
            ExtractionSplitAssignment(
                ExtractionValidationSplitRole.TRAIN,
                "family.matched-train-v1",
                "template.matched-train-v1",
                text_digest("matched train manifest"),
            ),
            ExtractionSplitAssignment(
                ExtractionValidationSplitRole.VALIDATION,
                "family.matched-v1",
                "template.matched-v1",
                text_digest("matched validation manifest"),
            ),
            ExtractionSplitAssignment(
                ExtractionValidationSplitRole.FINAL,
                "family.matched-final-v1",
                "template.matched-final-v1",
                text_digest("matched final manifest"),
            ),
        ),
    )


def _matched_observation(
    parent,
    candidate,
    variant,
    replicate,
    label,
    *,
    changed=True,
    artifact_id=None,
    artifact_digest=None,
    failure_counts=(0, 0, 0, 0),
):
    artifact = (
        parent if variant == ExtractionValidationVariant.PARENT else candidate
    )
    return ExtractionValidationObservation.create(
        variant=variant,
        replicate=replicate,
        family_id="family.matched-v1",
        task_template_group_id="template.matched-v1",
        task_id=f"task.matched-{replicate}",
        run_id=f"run.matched-{variant.value}-{replicate}",
        episode_id=f"episode.matched-{variant.value}-{replicate}",
        extraction_set_id=f"extraction-set.matched-{variant.value}-{replicate}",
        task_manifest_digest=text_digest("matched validation manifest"),
        model_profile_digest=text_digest("matched model profile"),
        budget_id="budget.matched-v1",
        persistence_state_digest=text_digest(f"matched-state-{replicate}"),
        extraction_artifact_id=artifact_id or artifact.artifact_id,
        extraction_artifact_digest=artifact_digest or artifact.body_digest,
        extraction_output_digest=text_digest(
            f"matched-output-{replicate}-"
            f"{'candidate' if variant == ExtractionValidationVariant.PROPOSAL and changed else 'parent'}"
        ),
        label=label,
        extraction_status=ExtractionSetStatus.NONEMPTY,
        missed_assessable=None,
        failure_counts=failure_counts,
    )


def _matched_pairs(parent, candidate, *, accepted: bool = True):
    parent_labels = (
        ExtractionFeedbackLabel.USEFUL,
        ExtractionFeedbackLabel.HARMFUL,
        ExtractionFeedbackLabel.USEFUL,
    )
    candidate_labels = (
        (ExtractionFeedbackLabel.USEFUL,) * 3
        if accepted
        else parent_labels
    )
    values = []
    for replicate, (parent_label, candidate_label) in enumerate(
        zip(parent_labels, candidate_labels),
        start=1,
    ):
        values.extend((
            _matched_observation(
                parent,
                candidate,
                ExtractionValidationVariant.PARENT,
                replicate,
                parent_label,
            ),
            _matched_observation(
                parent,
                candidate,
                ExtractionValidationVariant.PROPOSAL,
                replicate,
                candidate_label,
                changed=accepted,
            ),
        ))
    return tuple(values)


def _matched_decision(parent, candidate, *, accepted: bool = True):
    offline = _offline_decision(parent, candidate)
    split = _matched_split()
    observations = _matched_pairs(parent, candidate, accepted=accepted)
    decision = ExtractionMatchedTrialEvaluator().evaluate(
        parent=parent,
        candidate=candidate,
        offline_decision=offline,
        split=split,
        observations=observations,
        criteria=_criteria(),
    )
    return offline, split, observations, decision


def _coordinator(tmp_path, parent):
    policy_store = JsonExtractionPolicyStore(
        tmp_path / "extraction-policies.json",
        trusted_root=parent,
        slot=MEM0_FLAT_EXTRACTION_SLOT,
    )
    policy_store.initialize()
    decision_store = JsonExtractionMatchedTrialDecisionStore(
        tmp_path / "matched-decisions"
    )
    rollback_store = JsonExtractionRollbackEvidenceStore(
        tmp_path / "rollback-evidence"
    )
    return (
        policy_store,
        decision_store,
        rollback_store,
        ExtractionMatchedActivationCoordinator(
            policy_store,
            decision_store,
            rollback_store,
        ),
    )


def test_matched_trial_activation_restart_and_operator_rollback_are_idempotent(
    tmp_path,
) -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    offline, split, observations, decision = _matched_decision(parent, candidate)
    assert decision.status == ExtractionMatchedTrialStatus.ACCEPTED_FOR_ACTIVATION
    assert decision.reason_codes == ("matched_trial_passed",)
    assert all(value.passed for value in decision.constraint_results)
    assert {value.constraint for value in decision.constraint_results} == set(
        ExtractionMatchedConstraint
    )
    assert ExtractionMatchedTrialDecision.from_payload(decision.payload()) == decision

    policy_store, decision_store, rollback_store, coordinator = _coordinator(
        tmp_path,
        parent,
    )
    arguments = {
        "parent": parent,
        "candidate": candidate,
        "offline_decision": offline,
        "decision": decision,
        "split": split,
        "observations": observations,
        "criteria": _criteria(),
    }
    assert coordinator.apply(**arguments) == ExtractionPolicyState.ACTIVE
    assert coordinator.apply(**arguments) == ExtractionPolicyState.ACTIVE
    assert decision_store.get(decision.decision_id) == decision

    restarted = JsonExtractionPolicyStore(
        policy_store.path,
        trusted_root=parent,
        slot=MEM0_FLAT_EXTRACTION_SLOT,
    )
    assert restarted.snapshot().active == candidate
    restarted_coordinator = ExtractionMatchedActivationCoordinator(
        restarted,
        JsonExtractionMatchedTrialDecisionStore(decision_store.root),
        JsonExtractionRollbackEvidenceStore(rollback_store.root),
    )
    rollback = ExtractionRollbackEvidence.operator_requested(
        candidate=candidate,
        matched_decision_id=decision.decision_id,
        request_id="operator-request.rollback-v1",
        request_digest=text_digest("operator approved rollback"),
    )
    assert rollback.trigger == ExtractionRollbackTrigger.OPERATOR
    assert restarted_coordinator.rollback(
        candidate=candidate,
        evidence=rollback,
    ) == ExtractionPolicyState.ROLLED_BACK
    assert rollback_store.get(rollback.evidence_id) == rollback
    assert restarted_coordinator.rollback(
        candidate=candidate,
        evidence=rollback,
    ) == ExtractionPolicyState.ROLLED_BACK
    assert restarted.active_or_root() == parent


def test_matched_rejection_is_persisted_without_active_pointer(tmp_path) -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    offline, split, observations, decision = _matched_decision(
        parent,
        candidate,
        accepted=False,
    )
    assert decision.status == ExtractionMatchedTrialStatus.REJECTED
    assert "useful_rate_not_improved" in decision.reason_codes
    assert "no_extraction_intervention" in decision.reason_codes
    failed = {
        value.constraint
        for value in decision.constraint_results
        if not value.passed
    }
    assert failed == {
        ExtractionMatchedConstraint.USEFUL_RATE,
        ExtractionMatchedConstraint.EXTRACTION_INTERVENTION,
    }
    policy_store, _, _, coordinator = _coordinator(tmp_path, parent)
    arguments = {
        "parent": parent,
        "candidate": candidate,
        "offline_decision": offline,
        "decision": decision,
        "split": split,
        "observations": observations,
        "criteria": _criteria(),
    }
    assert coordinator.apply(**arguments) == ExtractionPolicyState.REJECTED
    assert coordinator.apply(**arguments) == ExtractionPolicyState.REJECTED
    assert policy_store.snapshot().active is None
    assert policy_store.active_or_root() == parent


def test_activation_crash_leaves_proposal_and_retry_activates_once(
    tmp_path,
    monkeypatch,
) -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    offline, split, observations, decision = _matched_decision(parent, candidate)
    policy_store, decision_store, _, coordinator = _coordinator(tmp_path, parent)
    policy_store.register(candidate)
    original_write = policy_store._write_unlocked

    def fail_write(payload):
        raise RuntimeError("simulated extraction activation crash")

    monkeypatch.setattr(policy_store, "_write_unlocked", fail_write)
    with pytest.raises(RuntimeError, match="activation crash"):
        coordinator.apply(
            parent=parent,
            candidate=candidate,
            offline_decision=offline,
            decision=decision,
            split=split,
            observations=observations,
            criteria=_criteria(),
        )
    crashed = JsonExtractionPolicyStore(
        policy_store.path,
        trusted_root=parent,
        slot=MEM0_FLAT_EXTRACTION_SLOT,
    ).snapshot()
    record = next(
        value for value in crashed.records if value.artifact_id == candidate.artifact_id
    )
    assert record.state == ExtractionPolicyState.PROPOSAL
    assert crashed.active is None
    assert decision_store.get(decision.decision_id) == decision

    monkeypatch.setattr(policy_store, "_write_unlocked", original_write)
    assert coordinator.apply(
        parent=parent,
        candidate=candidate,
        offline_decision=offline,
        decision=decision,
        split=split,
        observations=observations,
        criteria=_criteria(),
    ) == ExtractionPolicyState.ACTIVE
    assert len(tuple(
        value
        for value in policy_store.snapshot().records
        if value.state == ExtractionPolicyState.ACTIVE
    )) == 1


def test_second_candidate_cannot_create_two_active_artifacts(tmp_path) -> None:
    parent = _parent()
    first = _candidate(parent=parent)
    second = _candidate(
        parent=parent,
        text="Keep durable user facts that can support later work.",
    )
    first_inputs = _matched_decision(parent, first)
    second_inputs = _matched_decision(parent, second)
    policy_store, _, _, coordinator = _coordinator(tmp_path, parent)
    policy_store.register(first)
    policy_store.register(second)
    for candidate, inputs in ((first, first_inputs),):
        offline, split, observations, decision = inputs
        assert coordinator.apply(
            parent=parent,
            candidate=candidate,
            offline_decision=offline,
            decision=decision,
            split=split,
            observations=observations,
            criteria=_criteria(),
        ) == ExtractionPolicyState.ACTIVE

    offline, split, observations, decision = second_inputs
    with pytest.raises(ValueError, match="already active"):
        coordinator.apply(
            parent=parent,
            candidate=second,
            offline_decision=offline,
            decision=decision,
            split=split,
            observations=observations,
            criteria=_criteria(),
        )
    assert policy_store.snapshot().active == first


def test_automatic_rollback_requires_observed_safety_failure(tmp_path) -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    offline, split, observations, decision = _matched_decision(parent, candidate)
    _, _, rollback_store, coordinator = _coordinator(tmp_path, parent)
    coordinator.apply(
        parent=parent,
        candidate=candidate,
        offline_decision=offline,
        decision=decision,
        split=split,
        observations=observations,
        criteria=_criteria(),
    )
    safe = ExtractionValidationSafetyEvidence.create(
        live_feedback_record_id="live-feedback.rollback-v1",
        source_record_id="source.rollback-v1",
        audit_id="audit.rollback-v1",
        audit_digest=text_digest("safe audit"),
        evidence_cutoff_operation_id="operation.rollback-cutoff-v1",
        complete=True,
        schema_failure_count=0,
        safety_failure_count=0,
        prompt_leakage_failure_count=0,
        native_writer_failure_count=0,
    )
    with pytest.raises(ValueError, match="observed safety failure"):
        ExtractionRollbackEvidence.safety_violation(
            candidate=candidate,
            matched_decision_id=decision.decision_id,
            safety=safe,
            observation=_matched_observation(
                parent,
                candidate,
                ExtractionValidationVariant.PROPOSAL,
                10,
                ExtractionFeedbackLabel.UNRESOLVED,
            ),
        )
    unsafe = ExtractionValidationSafetyEvidence.create(
        live_feedback_record_id="live-feedback.rollback-v1",
        source_record_id="source.rollback-v1",
        audit_id="audit.rollback-v1",
        audit_digest=text_digest("unsafe audit"),
        evidence_cutoff_operation_id="operation.rollback-cutoff-v1",
        complete=True,
        schema_failure_count=0,
        safety_failure_count=0,
        prompt_leakage_failure_count=1,
        native_writer_failure_count=0,
    )
    rollback = ExtractionRollbackEvidence.safety_violation(
        candidate=candidate,
        matched_decision_id=decision.decision_id,
        safety=unsafe,
        observation=_matched_observation(
            parent,
            candidate,
            ExtractionValidationVariant.PROPOSAL,
            10,
            ExtractionFeedbackLabel.UNRESOLVED,
            failure_counts=(0, 0, 1, 0),
        ),
    )
    assert rollback.violation_codes == ("prompt_leakage",)
    assert coordinator.rollback(
        candidate=candidate,
        evidence=rollback,
    ) == ExtractionPolicyState.ROLLED_BACK
    assert ExtractionRollbackEvidence.from_payload(rollback.payload()) == rollback
    assert rollback_store.get(rollback.evidence_id) == rollback


def test_matched_trial_rejects_offline_reuse_digest_drift_and_rejected_gate() -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    accepted_offline = _offline_decision(parent, candidate)
    rejected_offline = _offline_decision(parent, candidate, accepted=False)
    split = _matched_split()
    observations = list(_matched_pairs(parent, candidate))
    evaluator = ExtractionMatchedTrialEvaluator()

    with pytest.raises(ValueError, match="offline accepted"):
        evaluator.evaluate(
            parent=parent,
            candidate=candidate,
            offline_decision=rejected_offline,
            split=split,
            observations=tuple(observations),
            criteria=_criteria(),
        )

    reused = _pairs(
        parent,
        candidate,
        (
            ExtractionFeedbackLabel.USEFUL,
            ExtractionFeedbackLabel.HARMFUL,
            ExtractionFeedbackLabel.USEFUL,
        ),
        (ExtractionFeedbackLabel.USEFUL,) * 3,
    )
    reused_split = ExtractionPromptValidationSplit(
        "split.renamed-offline-v1",
        _split().assignments,
    )
    with pytest.raises(ValueError, match="reuses offline validation evidence"):
        evaluator.evaluate(
            parent=parent,
            candidate=candidate,
            offline_decision=accepted_offline,
            split=reused_split,
            observations=reused,
            criteria=_criteria(),
        )

    observations[1] = _matched_observation(
        parent,
        candidate,
        ExtractionValidationVariant.PROPOSAL,
        1,
        ExtractionFeedbackLabel.USEFUL,
        artifact_digest=text_digest("wrong matched candidate body"),
    )
    with pytest.raises(ValueError, match="observation body digest mismatch"):
        evaluator.evaluate(
            parent=parent,
            candidate=candidate,
            offline_decision=accepted_offline,
            split=split,
            observations=tuple(observations),
            criteria=_criteria(),
        )


def test_matched_decision_store_fails_closed_on_corruption(tmp_path) -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    _, _, _, decision = _matched_decision(parent, candidate)
    store = JsonExtractionMatchedTrialDecisionStore(tmp_path / "decisions")
    path, created = store.put(decision)
    assert created is True
    assert store.put(decision) == (path, False)
    path.write_text(json.dumps({"decision_id": decision.decision_id}), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed extraction matched"):
        store.get(decision.decision_id)
    with pytest.raises(ValueError, match="conflicts with its ID"):
        store.put(decision)


def test_matched_activation_surface_has_no_score_or_cost_input() -> None:
    names = set(inspect.signature(ExtractionMatchedTrialEvaluator.evaluate).parameters)
    names.update(
        inspect.signature(ExtractionMatchedActivationCoordinator.apply).parameters
    )
    assert not names & {
        "score",
        "task_score",
        "grader",
        "answer_key",
        "cost",
        "usage",
        "latency",
        "storage_bytes",
    }


def test_matched_trial_separates_policy_and_runtime_component_identity() -> None:
    parent = _parent()
    candidate = _candidate(parent=parent)
    offline = _offline_decision(parent, candidate)
    parent_runtime_id = parent.to_prompt_component(
        MEM0_FLAT_EXTRACTION_SLOT
    ).artifact_id
    candidate_runtime_id = candidate.to_prompt_component(
        MEM0_FLAT_EXTRACTION_SLOT
    ).artifact_id
    observations = []
    parent_labels = (
        ExtractionFeedbackLabel.USEFUL,
        ExtractionFeedbackLabel.HARMFUL,
        ExtractionFeedbackLabel.USEFUL,
    )
    for replicate, parent_label in enumerate(parent_labels, start=1):
        observations.extend((
            _matched_observation(
                parent,
                candidate,
                ExtractionValidationVariant.PARENT,
                replicate,
                parent_label,
                artifact_id=parent_runtime_id,
            ),
            _matched_observation(
                parent,
                candidate,
                ExtractionValidationVariant.PROPOSAL,
                replicate,
                ExtractionFeedbackLabel.USEFUL,
                artifact_id=candidate_runtime_id,
            ),
        ))
    decision = ExtractionMatchedTrialEvaluator().evaluate(
        parent=parent,
        candidate=candidate,
        offline_decision=offline,
        split=_matched_split(),
        observations=tuple(observations),
        criteria=_criteria(),
        parent_runtime_artifact_id=parent_runtime_id,
        candidate_runtime_artifact_id=candidate_runtime_id,
    )
    assert decision.parent_artifact_id == parent.artifact_id
    assert decision.candidate_artifact_id == candidate.artifact_id
    assert decision.parent_runtime_artifact_id == parent_runtime_id
    assert decision.candidate_runtime_artifact_id == candidate_runtime_id
    assert decision.quality_decision.parent_artifact_id == parent_runtime_id
    assert decision.quality_decision.proposal_artifact_id == candidate_runtime_id
