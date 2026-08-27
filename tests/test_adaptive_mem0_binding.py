from __future__ import annotations

from types import SimpleNamespace

from rsimem.memory.adaptive_matched_validation import (
    JsonMatchedValidationDecisionStore,
    MatchedAcceptanceCriteria,
    MatchedAdaptivePolicyActivationCoordinator,
    MatchedAdaptivePolicyValidator,
)
from rsimem.memory.adaptive_mem0_binding import (
    ActiveAdaptiveMem0Binder,
    TrustedAdaptiveMem0Parameter,
    audit_adaptive_mem0_loop,
)
from rsimem.memory.adaptive_policy import AdaptiveParameterName
from rsimem.memory.adaptive_policy_store import JsonAdaptivePolicyStore
from rsimem.memory.ingestion import InternalMemoryAction
from rsimem.memory.utility import UtilityTarget
from rsimem.memory_systems.mem0_flat import FrozenMem0UtilityGate
from test_adaptive_matched_validation import _observations, _offline_validated
from test_mem0_flat_policy import _operation_response, _setup


TRUSTED_RETRIEVAL = TrustedAdaptiveMem0Parameter(
    parameter_id="parameter.fact",
    name=AdaptiveParameterName.RETRIEVAL_ACCEPT_THRESHOLD,
    prompt_ref="mem0-flat.retrieval",
    baseline_value=0.35,
)


def _bind(store, parent_policy_version, base=None):
    return ActiveAdaptiveMem0Binder((TRUSTED_RETRIEVAL,)).bind(
        store,
        base or FrozenMem0UtilityGate(),
        expected_parent_policy_version=parent_policy_version,
    )


def _active(tmp_path):
    dataset, split, artifact, store, offline = _offline_validated(
        tmp_path,
        suffix="binding",
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
        JsonMatchedValidationDecisionStore(tmp_path / "matched-binding"),
    )
    coordinator.apply(
        artifact,
        decision,
        split=split,
        observations=observations,
        criteria=criteria,
    )
    offline_decision = offline.decision_store.all()[0]
    return dataset, artifact, store, offline_decision, decision


def test_binder_preserves_static_default_without_active_policy(tmp_path) -> None:
    dataset, _, active_store, _, _ = _active(tmp_path)
    inactive = JsonAdaptivePolicyStore(
        tmp_path / "inactive.json",
        trusted_root_policy_versions=(dataset.config.policy_version,),
    )
    base = FrozenMem0UtilityGate()
    binding = _bind(inactive, dataset.config.policy_version, base)
    assert binding.adaptive is False
    assert binding.gate is base
    assert binding.gate.digest == FrozenMem0UtilityGate().digest
    assert binding.artifact_id is None
    assert active_store.snapshot().active is not None


def test_active_artifact_changes_retrieval_and_records_actual_version(tmp_path) -> None:
    _, artifact, store, offline_decision, matched_decision = _active(tmp_path)
    base = FrozenMem0UtilityGate()
    binding = _bind(store, artifact.parent_policy_version, base)
    assert binding.adaptive is True
    assert binding.actual_policy_version == artifact.policy_version
    assert binding.artifact_id == artifact.artifact_id
    assert binding.parameter_names == (
        AdaptiveParameterName.RETRIEVAL_ACCEPT_THRESHOLD,
    )
    adaptive_retrieval = binding.gate.policy_for(UtilityTarget.RETRIEVAL)
    assert adaptive_retrieval.policy_version == artifact.policy_version
    assert adaptive_retrieval.accept_threshold == artifact.parameters[0].proposed_value

    setup = _setup(tmp_path / "request", utility_gate=base)
    request = setup[5]
    changed = None
    for index in range(101):
        score = index / 100
        view = SimpleNamespace(
            score=score,
            candidate=SimpleNamespace(candidate_id=f"candidate.{index:040d}"),
        )
        static_kept = bool(base.rank_related(request, (view,)))
        static_decision = base.decisions(request.idempotency_key)[-1]
        adaptive_kept = bool(binding.gate.rank_related(request, (view,)))
        adaptive_decision = binding.gate.decisions(request.idempotency_key)[-1]
        if static_kept != adaptive_kept:
            changed = (
                score,
                static_kept,
                adaptive_kept,
                static_decision,
                adaptive_decision,
            )
            break
    assert changed is not None
    assert changed[1:3] == (True, False)
    static_decision, adaptive_decision = changed[3:]
    assert adaptive_decision.policy_version == artifact.policy_version
    audit = audit_adaptive_mem0_loop(
        store=store,
        binding=binding,
        offline_decision=offline_decision,
        matched_decision=matched_decision,
        parent_future_decision=static_decision,
        adaptive_future_decision=adaptive_decision,
    )
    assert audit.ok is True
    assert audit.issues == ()
    assert audit.dataset_id == artifact.dataset_id
    assert audit.parent_disposition == "accept"
    assert audit.adaptive_disposition == "defer"


def test_active_gate_preserves_prompt_cadence_and_versions_all_decisions(tmp_path) -> None:
    _, artifact, store, _, _ = _active(tmp_path)
    binding = _bind(store, artifact.parent_policy_version)
    setup = _setup(
        tmp_path / "adaptive-ingest",
        operation=_operation_response(
            InternalMemoryAction.ADD,
            use_candidate=False,
        ),
        utility_gate=binding.gate,
        policy_version=artifact.policy_version,
    )
    setup[-1].ingest(setup[5], setup[6])
    assert len(setup[3].calls) == 2
    decisions = binding.gate.decisions(setup[5].idempotency_key)
    assert decisions
    assert all(item.policy_version == artifact.policy_version for item in decisions)
    evidence = binding.observer_evidence()
    assert evidence["actual_policy_version"] == artifact.policy_version
    assert evidence["adaptive"] is True
