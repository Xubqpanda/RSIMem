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
from rsimem.memory_systems.mem0_flat import (
    MEM0_FLAT_EXTRACTION_SLOT,
    MEM0_FLAT_EXTRACTION_SLOT_ID,
    Mem0FlatPromptAdapter,
)

from test_extraction_optimizer_contracts import (
    _multi_corpus,
    _parent,
    _proposal_output,
)
from test_extraction_offline_validation import (
    _criteria,
    _pairs,
    _safety_and_suite,
    _split,
)
from rsimem.memory.extraction_feedback import ExtractionFeedbackLabel


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
