from __future__ import annotations

from rsimem.memory.extraction_offline_validation import (
    ExtractionCandidateStaticValidator,
    ExtractionOfflineDecisionStatus,
    ExtractionPromptOfflineValidator,
)
from rsimem.memory.extraction_policy_store import (
    ExtractionPolicyState,
    JsonExtractionPolicyStore,
)
from rsimem.memory.extraction_prompt_optimizer import (
    CapturedExtractionOptimizerClient,
    ExtractionPromptOptimizer,
)
from rsimem.memory.policy_feasibility import project_optimizer_result
from rsimem.memory.policy_feasibility import LayerIntervention
from rsimem.memory.policy_contracts import PolicyArtifactIdentity, PolicyArtifactKind, PolicyLayer
from rsimem.memory.policy_feasibility_fixture import build_extraction_feedback_fixture
from rsimem.memory.policy_intervention_path import (
    FeasibilityInterventionPath,
    JsonFeasibilityInterventionPathStore,
)
from rsimem.memory_systems.mem0_flat import (
    MEM0_FLAT_EXTRACTION_SLOT,
    MEM0_FLAT_EXTRACTION_SLOT_ID,
    Mem0FlatPromptAdapter,
)

from test_extraction_optimizer_contracts import (
    _corpus,
    _multi_corpus,
    _parent,
    _proposal_output,
)
from test_policy_feasibility import _replays
from test_extraction_offline_validation import (
    _criteria,
    _pairs,
    _safety_and_suite,
    _split,
)
from rsimem.memory.extraction_feedback import ExtractionFeedbackLabel
from rsimem.memory.policy_feasibility import OptimizerHypothesisDecision


def test_optimizer_to_offline_gate_to_restart_safe_future_candidate(tmp_path) -> None:
    parent = _parent()
    corpus = _multi_corpus((ExtractionFeedbackLabel.USEFUL,) * 3)
    result = ExtractionPromptOptimizer(
        CapturedExtractionOptimizerClient(_proposal_output)
    ).propose(parent, corpus)
    assert result.candidate is not None
    projection = project_optimizer_result(
        result,
        corpus,
        parent_artifact_id=parent.artifact_id,
    )
    assert projection.candidate_artifact_id == result.candidate.artifact_id

    candidate = result.candidate
    safety, suite = _safety_and_suite(parent, candidate)
    offline = ExtractionPromptOfflineValidator().evaluate(
        parent=parent,
        candidate=candidate,
        split=_split(),
        observations=_pairs(
            parent,
            candidate,
            (ExtractionFeedbackLabel.HARMFUL,) * 3,
            (ExtractionFeedbackLabel.USEFUL,) * 3,
        ),
        criteria=_criteria(),
        static_safety=safety,
        deterministic_suite=suite,
    )
    assert safety.passed is True
    assert suite.passed is True
    assert offline.status is ExtractionOfflineDecisionStatus.ACCEPTED_FOR_MATCHED_TRIAL

    store = JsonExtractionPolicyStore(
        tmp_path / "extraction-policy.json",
        trusted_root=parent,
        slot=MEM0_FLAT_EXTRACTION_SLOT,
    )
    store.initialize()
    record, created = store.register(candidate)
    assert created is True
    assert record.state is ExtractionPolicyState.PROPOSAL
    restarted = JsonExtractionPolicyStore(
        tmp_path / "extraction-policy.json",
        trusted_root=Mem0FlatPromptAdapter().export_root_policy_artifact(
            MEM0_FLAT_EXTRACTION_SLOT_ID
        ),
        slot=MEM0_FLAT_EXTRACTION_SLOT,
    )
    snapshot = restarted.snapshot()
    assert snapshot.active_artifact_id is None
    loaded = next(
        artifact for artifact in snapshot.artifacts
        if artifact.artifact_id == projection.candidate_artifact_id
    )
    assert loaded == candidate
    assert loaded.parent_artifact_id == projection.parent_artifact_id
    assert loaded.to_prompt_component(MEM0_FLAT_EXTRACTION_SLOT).slot_id == (
        MEM0_FLAT_EXTRACTION_SLOT_ID
    )

    past_parent, past_candidate = _replays()
    feedback_fixture = build_extraction_feedback_fixture(outcome=ExtractionFeedbackLabel.USEFUL)
    primary = next(example for example in feedback_fixture.dataset.examples if example.primary)
    parent_identity = PolicyArtifactIdentity(
        projection.parent_artifact_id,
        "fixed.extraction.parent.v1",
        PolicyArtifactKind.FIXED,
        (PolicyLayer.EXTRACTION,),
    )
    candidate_identity = PolicyArtifactIdentity(
        projection.candidate_artifact_id,
        "adaptive.extraction.candidate.v1",
        PolicyArtifactKind.SINGLE_LAYER_ADAPTIVE,
        (PolicyLayer.EXTRACTION,),
    )
    intervention = LayerIntervention.from_extraction_feedback(
        case_id="case.future.fixture.intervention",
        parent=past_parent,
        candidate=past_candidate,
        parent_artifact=parent_identity,
        candidate_artifact=candidate_identity,
        example=primary,
    )
    path = FeasibilityInterventionPath.from_projection_case(projection, intervention)
    path_store = JsonFeasibilityInterventionPathStore(tmp_path / "interventions.jsonl")
    _, created = path_store.put(path)
    assert created is True
    restarted_paths = JsonFeasibilityInterventionPathStore(
        tmp_path / "interventions.jsonl"
    )
    assert restarted_paths.records == (path,)
    _, replayed = restarted_paths.put(path)
    assert replayed is False


def test_nplus1_rejection_no_proposal_and_rollback_paths_are_restart_safe(tmp_path) -> None:
    parent = _parent()
    no_signal = ExtractionPromptOptimizer(
        CapturedExtractionOptimizerClient(_proposal_output)
    ).propose(parent, _corpus())
    projection = project_optimizer_result(
        no_signal,
        _corpus(),
        parent_artifact_id=parent.artifact_id,
    )
    assert projection.decision is OptimizerHypothesisDecision.NO_PROPOSAL
    assert projection.candidate_artifact_id is None

    proposal = ExtractionPromptOptimizer(
        CapturedExtractionOptimizerClient(_proposal_output)
    ).propose(parent, _multi_corpus((ExtractionFeedbackLabel.USEFUL,) * 3))
    assert proposal.candidate is not None

    rejected_path = tmp_path / "rejected-policy.json"
    rejected_store = JsonExtractionPolicyStore(
        rejected_path,
        trusted_root=parent,
        slot=MEM0_FLAT_EXTRACTION_SLOT,
    )
    rejected_store.initialize()
    rejected_store.register(proposal.candidate)
    rejected_store.transition(
        proposal.candidate.artifact_id,
        to_state=ExtractionPolicyState.REJECTED,
        transition_id="transition.feasibility.rejected",
        reason_code="offline_gate_rejected",
    )
    restarted_rejected = JsonExtractionPolicyStore(
        rejected_path,
        trusted_root=_parent(),
        slot=MEM0_FLAT_EXTRACTION_SLOT,
    )
    rejected_snapshot = restarted_rejected.snapshot()
    rejected_record = next(
        record for record in rejected_snapshot.records
        if record.artifact_id == proposal.candidate.artifact_id
    )
    assert rejected_record.state is ExtractionPolicyState.REJECTED
    try:
        restarted_rejected.transition(
            proposal.candidate.artifact_id,
            to_state=ExtractionPolicyState.ACTIVE,
            transition_id="transition.feasibility.rejected-active",
            reason_code="must_fail",
        )
    except ValueError:
        pass
    else:  # pragma: no cover - defensive assertion for the state machine
        raise AssertionError("rejected candidate became active")

    rollback_path = tmp_path / "rollback-policy.json"
    rollback_store = JsonExtractionPolicyStore(
        rollback_path,
        trusted_root=_parent(),
        slot=MEM0_FLAT_EXTRACTION_SLOT,
    )
    rollback_store.initialize()
    rollback_store.register(proposal.candidate)
    rollback_store.transition(
        proposal.candidate.artifact_id,
        to_state=ExtractionPolicyState.ACTIVE,
        transition_id="transition.feasibility.active",
        reason_code="matched_trial_passed",
    )
    rollback_store.transition(
        proposal.candidate.artifact_id,
        to_state=ExtractionPolicyState.ROLLED_BACK,
        transition_id="transition.feasibility.rollback",
        reason_code="operator_rollback",
    )
    restarted_rollback = JsonExtractionPolicyStore(
        rollback_path,
        trusted_root=_parent(),
        slot=MEM0_FLAT_EXTRACTION_SLOT,
    )
    rollback_snapshot = restarted_rollback.snapshot()
    assert rollback_snapshot.active_artifact_id is None
    rolled_back = next(
        record for record in rollback_snapshot.records
        if record.artifact_id == proposal.candidate.artifact_id
    )
    assert rolled_back.state is ExtractionPolicyState.ROLLED_BACK
