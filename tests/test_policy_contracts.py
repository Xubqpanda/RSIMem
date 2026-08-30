from __future__ import annotations

import pytest

from rsimem.memory.policy_contracts import (
    AdmissionDecision,
    CommitDecision,
    DecisionAction,
    ExecutionStatus,
    ExposureDecision,
    ExposureMode,
    ExtractionDecision,
    FORMATION_LAYERS,
    MutationKind,
    PolicyArtifactIdentity,
    PolicyArtifactKind,
    PolicyLayer,
    PolicyLineage,
    PolicyDecisionContract,
    ProjectionMode,
    SafetyBoundary,
    SourceSelectionDecision,
    TriggerDecision,
    TriggerEvent,
    validate_policy_episode,
    validate_policy_lineage,
    decision_contract_for_layer,
)


def _base_kwargs(lineage: str = "lineage.fixture") -> dict[str, object]:
    return {
        "policy_version": "fixed.parent.v1",
        "source_revision": "snapshot.rev.1",
        "input_payload": {"source": ["segment.done"]},
        "output_payload": {"decision": "run"},
        "action": DecisionAction.RUN,
        "execution_status": ExecutionStatus.EXECUTED,
        "reason_codes": ("fixture",),
        "lineage_id": lineage,
        "trigger_event_id": "event.fixture",
        "execution_receipt_id": "receipt.decision",
    }


def test_trigger_event_and_decision_replay_are_stable() -> None:
    first = TriggerEvent.create(
        event_type="task_completed",
        source_revision="snapshot.rev.1",
        input_payload={"task": "SM01", "turn": 2},
        session_id="session.fixture",
        task_id="task.fixture",
    )
    replay = TriggerEvent.create(
        event_type="task_completed",
        source_revision="snapshot.rev.1",
        input_payload={"task": "SM01", "turn": 2},
        session_id="session.fixture",
        task_id="task.fixture",
    )
    assert first == replay

    first_decision = TriggerDecision.create(**_base_kwargs())
    replay_decision = TriggerDecision.create(**_base_kwargs())
    assert first_decision == replay_decision
    assert first_decision.is_canonical


def test_each_policy_layer_exposes_a_typed_decision_contract() -> None:
    for layer in PolicyLayer:
        contract = decision_contract_for_layer(layer)
        assert isinstance(contract, PolicyDecisionContract)
        assert contract.layer is layer
        assert contract.decision_type.endswith("Decision")
        assert set(contract.allowed_actions) == set(DecisionAction)
        assert {"decision_id", "action", "input_digest", "output_digest"}.issubset(
            contract.required_fields
        )
        assert PolicyDecisionContract.from_payload(contract.payload()) == contract


def test_decision_contract_rejects_unknown_layer_or_tampered_identity() -> None:
    with pytest.raises(ValueError, match="unknown policy decision layer"):
        decision_contract_for_layer("not-a-layer")
    contract = decision_contract_for_layer(PolicyLayer.EXTRACTION)
    payload = contract.payload()
    payload["contract_id"] = "policy-decision-contract.trigger.v1"
    with pytest.raises(ValueError, match="ID mismatch|malformed"):
        PolicyDecisionContract.from_payload(payload)


def test_decision_identity_includes_layer_specific_output() -> None:
    first = SourceSelectionDecision.create(
        **_base_kwargs(),
        projection_mode=ProjectionMode.SELECTED_COMPLETED_SEGMENTS,
        selected_segment_ids=("segment.a",),
        source_payload={"segment.a": "one"},
    )
    changed = SourceSelectionDecision.create(
        **_base_kwargs(),
        projection_mode=ProjectionMode.SELECTED_COMPLETED_SEGMENTS,
        selected_segment_ids=("segment.b",),
        source_payload={"segment.b": "two"},
    )
    assert first.decision_id != changed.decision_id


def test_skip_and_defer_have_closed_execution_semantics() -> None:
    with pytest.raises(ValueError, match="skipped status"):
        TriggerDecision.create(
            **{**_base_kwargs(), "action": DecisionAction.SKIP, "execution_status": ExecutionStatus.EXECUTED}
        )
    with pytest.raises(ValueError, match="next eligible boundary"):
        TriggerDecision.create(
            **{**_base_kwargs(), "action": DecisionAction.DEFER, "execution_status": ExecutionStatus.DEFERRED}
        )
    deferred = TriggerDecision.create(
        **{**_base_kwargs(), "action": DecisionAction.DEFER, "execution_status": ExecutionStatus.DEFERRED},
        next_eligible_boundary="task_completed.next",
    )
    assert deferred.action is DecisionAction.DEFER
    with pytest.raises(ValueError, match="non-executing status"):
        TriggerDecision.create(
            **{**_base_kwargs(), "action": DecisionAction.RUN, "execution_status": ExecutionStatus.SKIPPED}
        )
    receiptful_skip = TriggerDecision.create(
        **{**_base_kwargs(), "action": DecisionAction.SKIP, "execution_status": ExecutionStatus.SKIPPED}
    )
    report = validate_policy_episode((receiptful_skip,), require_all_layers=False)
    assert not report.ok
    assert any("non-executing decision" in error for error in report.errors)


def test_source_selection_cannot_bypass_active_current_or_tool_closure() -> None:
    safety = SafetyBoundary(
        active_segment_ids=("segment.active",),
        current_turn_id="turn.current",
        tool_closures=(("tool.call", "tool.result"),),
    )
    for selected in (("segment.active",), ("turn.current",), ("tool.call",)):
        with pytest.raises(ValueError):
            SourceSelectionDecision.create(
                **_base_kwargs(),
                selected_segment_ids=selected,
                source_payload={"selected": list(selected)},
                safety=safety,
            )
    allowed = SourceSelectionDecision.create(
        **_base_kwargs(),
        selected_segment_ids=("tool.call", "tool.result"),
        source_payload={"selected": ["tool.call", "tool.result"]},
        safety=safety,
    )
    assert allowed.selected_segment_ids == ("tool.call", "tool.result")


def test_admission_and_commit_reject_unsafe_or_unsupported_actions() -> None:
    with pytest.raises(ValueError, match="does not support update"):
        AdmissionDecision.create(
            **_base_kwargs(),
            mutation_kind=MutationKind.UPDATE,
            candidate_fact_ids=("fact.1",),
            accepted_fact_ids=("fact.1",),
            target_artifact_ids=("artifact.1",),
            backend_revision="backend.rev.1",
            update_supported=False,
        )
    unsafe = SafetyBoundary(cas_valid=False)
    with pytest.raises(ValueError, match="safety boundary"):
        CommitDecision.create(
            **_base_kwargs(),
            mutation_ids=("mutation.1",),
            expected_revision="backend.rev.1",
            safety=unsafe,
        )


def test_exposure_requires_exact_ordering_and_mode() -> None:
    with pytest.raises(ValueError, match="exposure mode"):
        ExposureDecision.create(
            **_base_kwargs(),
            selected_artifact_ids=("artifact.1",),
            ordering=("artifact.1",),
        )
    decision = ExposureDecision.create(
        **_base_kwargs(),
        exposure_mode=ExposureMode.SELECTIVE_RETRIEVAL,
        selected_artifact_ids=("artifact.1", "artifact.2"),
        ordering=("artifact.2", "artifact.1"),
        injection_position="system.memory",
        injection_receipt_id="receipt.injection",
    )
    assert decision.ordering == ("artifact.2", "artifact.1")


def test_artifact_identity_kinds_are_distinct_and_canonical() -> None:
    fixed = PolicyArtifactIdentity.create(
        policy_version="fixed.parent.v1", kind=PolicyArtifactKind.FIXED, layers=(PolicyLayer.EXTRACTION,)
    )
    single = PolicyArtifactIdentity.create(
        policy_version="adaptive.extraction.v1", kind=PolicyArtifactKind.SINGLE_LAYER_ADAPTIVE, layers=(PolicyLayer.EXTRACTION,)
    )
    joint = PolicyArtifactIdentity.create(
        policy_version="joint.v1", kind=PolicyArtifactKind.JOINT, layers=(PolicyLayer.COMMIT, PolicyLayer.TRIGGER)
    )
    assert fixed.is_canonical and single.is_canonical and joint.is_canonical
    assert len({fixed.artifact_id, single.artifact_id, joint.artifact_id}) == 3
    assert joint.layers == (PolicyLayer.COMMIT, PolicyLayer.TRIGGER)


def test_audit_rejects_missing_layer_and_execution_receipt() -> None:
    trigger = TriggerDecision.create(**{**_base_kwargs(), "execution_receipt_id": None})
    report = validate_policy_episode((trigger,))
    assert not report.ok
    assert any("missing decisions" in error for error in report.errors)
    assert any("missing execution receipt" in error for error in report.errors)
    assert set(FORMATION_LAYERS) == {
        PolicyLayer.TRIGGER,
        PolicyLayer.SOURCE_SELECTION,
        PolicyLayer.EXTRACTION,
        PolicyLayer.ADMISSION,
        PolicyLayer.COMMIT,
    }


def test_lineage_join_rejects_mismatched_downstream_evidence() -> None:
    trigger = TriggerDecision.create(**_base_kwargs())
    lineage = PolicyLineage.from_decisions(
        (trigger,), mutation_receipt_ids=("receipt.mutation",),
    )
    ok = validate_policy_lineage(
        lineage,
        (trigger,),
        mutation_receipt_ids=("receipt.mutation",),
        require_all_layers=False,
    )
    assert ok.ok
    mismatch = validate_policy_lineage(
        lineage,
        (trigger,),
        mutation_receipt_ids=("receipt.other",),
        require_all_layers=False,
    )
    assert not mismatch.ok
    assert any("mutation receipt" in error for error in mismatch.errors)
