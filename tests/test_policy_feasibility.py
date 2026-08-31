from __future__ import annotations

import pytest
from dataclasses import replace

from rsimem.lifecycle.snapshot import (
    ContextSnapshot,
    ProvenanceRef,
    SnapshotSegment,
    TaskLifecycleState,
)
from rsimem.memory.contracts import (
    MemoryAccessMode,
    MemoryBackendDescriptor,
    MemoryKind,
    MemoryKindCapability,
)
from rsimem.memory.policy_contracts import (
    PolicyArtifactIdentity,
    PolicyArtifactKind,
    PolicyLayer,
)
from rsimem.memory.policy_feasibility import (
    FeasibilityEvidenceRecord,
    FeedbackChain,
    FeasibilityOutcome,
    FeasibilityStatus,
    LayerBenefitExplanation,
    LayerIntervention,
    LayerFeasibilityCensus,
    JsonFeasibilityEvidenceLedger,
    PolicyHypothesis,
    OptimizerHypothesisDecision,
    OptimizerHypothesisProjection,
    feedback_chain_from_extraction_example,
    project_optimizer_result,
    build_feasibility_report,
    build_extraction_feedback_interventions,
    build_optimizer_corpus_interventions,
    validate_feasibility_case,
)
from rsimem.memory.policy_replay import DeterministicPolicyReplay
from rsimem.memory.policy_feasibility_fixture import (
    build_default_feasibility_cases,
    build_extraction_feedback_fixture,
    run_default_feasibility_census,
)
from rsimem.memory.policy_audit import PolicyAuditReport
from rsimem.memory.source_selection_policy import (
    DeterministicSourceSelectionPolicy,
    SourceSelectionConfig,
)
from rsimem.memory.trigger_policy import (
    DeterministicTriggerPolicy,
    TriggerPolicyConfig,
)
from rsimem.memory.trigger_policy import HostTriggerAdapter


def _snapshot() -> ContextSnapshot:
    snapshot_id = "snapshot.feasibility"
    segments = (
        SnapshotSegment(
            "segment.durable",
            "message.durable",
            "user",
            "The user prefers TSV output.",
            "turn.1",
            6,
            completed=True,
        ),
        SnapshotSegment(
            "segment.temporary",
            "message.temporary",
            "assistant",
            "A temporary formatting request.",
            "turn.2",
            5,
            completed=True,
        ),
    )
    return ContextSnapshot(
        "run.feasibility",
        "episode.feasibility",
        "session.feasibility",
        "task.feasibility",
        snapshot_id,
        "revision.feasibility",
        segments,
        (),
        None,
        TaskLifecycleState.COMPLETED,
        "task_completed",
        (),
        11,
        ProvenanceRef(
            "run.feasibility",
            "episode.feasibility",
            "session.feasibility",
            "task.feasibility",
            snapshot_id,
            "fixture.sm01",
        ),
    )


def _backend() -> MemoryBackendDescriptor:
    return MemoryBackendDescriptor(
        "backend.feasibility",
        (MemoryKindCapability(MemoryKind.SEMANTIC, MemoryAccessMode.EAGER),),
    )


def _replays():
    snapshot = _snapshot()
    event = HostTriggerAdapter().event(
        "task_completed",
        source_revision=snapshot.context_revision,
        payload={"snapshot_id": snapshot.snapshot_id, "fixture": "sm01"},
        session_id=snapshot.session_id,
        task_id=snapshot.task_id,
        turn_index=2,
    )
    parent = DeterministicPolicyReplay().run(
        snapshot,
        event,
        backend=_backend(),
        candidate_fact_ids=("fact.tsv_preference",),
        artifact_ids=("artifact.tsv_preference",),
        mutation_ids=("mutation.parent",),
    )
    candidate = DeterministicPolicyReplay().run(
        snapshot,
        event,
        backend=_backend(),
        candidate_fact_ids=(),
        artifact_ids=("artifact.tsv_preference",),
        mutation_ids=("mutation.candidate",),
    )
    return parent, candidate


def _artifact(version: str, kind: PolicyArtifactKind) -> PolicyArtifactIdentity:
    return PolicyArtifactIdentity.create(
        policy_version=version,
        kind=kind,
        layers=(PolicyLayer.EXTRACTION,),
    )


def _case(
    case_id: str,
    outcome: FeasibilityOutcome,
    feedback: FeedbackChain,
) -> LayerIntervention:
    parent, candidate = _replays()
    return LayerIntervention(
        case_id=case_id,
        target_layer=PolicyLayer.EXTRACTION,
        parent=parent,
        candidate=candidate,
        parent_artifact=_artifact("fixed.extraction.parent.v1", PolicyArtifactKind.FIXED),
        candidate_artifact=_artifact("adaptive.extraction.candidate.v1", PolicyArtifactKind.SINGLE_LAYER_ADAPTIVE),
        process_signal=True,
        outcome=outcome,
        feedback=feedback,
        reason_codes=("fixture_sm01",),
    )


def test_complete_useful_and_missed_chains_make_extraction_optimization_ready() -> None:
    useful = _case(
        "case.useful",
        FeasibilityOutcome.USEFUL,
        FeedbackChain("opportunity.1", "use.1", "outcome.1"),
    )
    missed = _case(
        "case.missed",
        FeasibilityOutcome.MISSED,
        FeedbackChain(
            source_id="source.1",
            demand_id="demand.1",
            absence_id="absence.1",
            outcome_id="outcome.2",
        ),
    )

    validate_feasibility_case(useful)
    validate_feasibility_case(missed)
    report = build_feasibility_report((useful, missed))
    census = next(item for item in report.census if item.layer == PolicyLayer.EXTRACTION)

    assert census.status is FeasibilityStatus.OPTIMIZATION_READY
    assert census.signal_coverage == 1.0
    assert census.action_variation == 1.0
    assert census.outcome_variation == 1.0
    assert census.outcome_counts == {"useful": 1, "missed": 1}
    assert census.complete_feedback_count == 2
    assert census.unknown_count == 0
    assert census.unresolved_count == 0
    assert census.censored_count == 0
    assert census.unresolved_ratio == 0.0
    assert census.censored_ratio == 0.0
    assert census.useful_count == 1
    assert census.harmful_count == 0
    assert census.missed_count == 1
    assert census.resolved_useful_rate == 1.0
    assert useful.replay_payload["parent_audit_ok"] is True
    assert useful.replay_payload["candidate_audit_ok"] is True
    assert useful.process_feedback is not None
    assert useful.process_feedback.event_id == useful.parent.event.event_id
    assert useful.process_feedback.target_layer is PolicyLayer.EXTRACTION
    assert useful.process_feedback.parent_decision_id == useful.parent_decision.decision_id
    assert useful.process_feedback == type(useful.process_feedback).from_payload(
        useful.process_feedback.payload()
    )
    assert useful.hypothesis is not None
    assert useful.hypothesis.target_layer is PolicyLayer.EXTRACTION
    assert useful.hypothesis.candidate_artifact_id == useful.candidate_artifact.artifact_id
    assert useful.hypothesis == type(useful.hypothesis).from_payload(
        useful.hypothesis.payload()
    )
    assert useful.replay_payload["intervention_fingerprint"] == useful.intervention_fingerprint
    report_case = next(item for item in report.payload()["cases"] if item["caseId"] == useful.case_id)
    assert report_case["processFeedback"]["feedback_id"] == useful.process_feedback.feedback_id
    assert report_case["hypothesis"]["hypothesis_id"] == useful.hypothesis.hypothesis_id
    assert report.digest == build_feasibility_report((useful, missed)).digest
    # The report is intentionally not globally ready until all six layers have
    # cases; unrepresented layers remain diagnostic-only.
    assert not report.ok


def test_outcome_variation_excludes_unresolved_and_censored_cases() -> None:
    useful = _case(
        "case.variation.useful",
        FeasibilityOutcome.USEFUL,
        FeedbackChain("opportunity.variation", "use.variation", "outcome.variation"),
    )
    missed = _case(
        "case.variation.missed",
        FeasibilityOutcome.MISSED,
        FeedbackChain(
            source_id="source.variation",
            demand_id="demand.variation",
            absence_id="absence.variation",
            outcome_id="outcome.variation.missed",
        ),
    )
    unresolved = _case(
        "case.variation.unresolved",
        FeasibilityOutcome.UNRESOLVED,
        FeedbackChain(),
    )
    census = next(
        item
        for item in build_feasibility_report((useful, missed, unresolved)).census
        if item.layer is PolicyLayer.EXTRACTION
    )
    assert census.outcome_counts == {
        "useful": 1,
        "missed": 1,
        "unresolved": 1,
    }
    assert census.outcome_variation_count == 2
    assert census.outcome_variation == 2 / 3
    assert census.unknown_count == 1


def test_missing_feedback_node_degrades_to_unresolved() -> None:
    case = _case(
        "case.incomplete",
        FeasibilityOutcome.USEFUL,
        FeedbackChain(opportunity_id="opportunity.1", use_id="use.1"),
    )

    assert case.outcome is FeasibilityOutcome.UNRESOLVED
    assert "incomplete_useful_feedback" in case.reason_codes
    report = build_feasibility_report((case,))
    census = next(item for item in report.census if item.layer == PolicyLayer.EXTRACTION)
    assert census.unknown_count == 1
    assert census.complete_feedback_count == 0
    assert census.status is FeasibilityStatus.VALIDATION_ONLY


def test_missing_missed_node_also_degrades_without_reward() -> None:
    case = _case(
        "case.incomplete_missed",
        FeasibilityOutcome.MISSED,
        FeedbackChain(source_id="source.1", demand_id="demand.1", outcome_id="outcome.1"),
    )
    assert case.outcome is FeasibilityOutcome.UNRESOLVED
    assert "incomplete_missed_feedback" in case.reason_codes


def test_incomplete_harmful_node_also_degrades_without_negative_reward() -> None:
    parent, candidate = _replays()
    case = LayerIntervention(
        case_id="case.incomplete_harmful",
        target_layer=PolicyLayer.EXTRACTION,
        parent=parent,
        candidate=candidate,
        parent_artifact=_artifact("fixed.extraction.parent.v1", PolicyArtifactKind.FIXED),
        candidate_artifact=_artifact(
            "adaptive.extraction.candidate.v1",
            PolicyArtifactKind.SINGLE_LAYER_ADAPTIVE,
        ),
        process_signal=True,
        outcome=FeasibilityOutcome.HARMFUL,
        reason_codes=("incomplete_harm",),
    )
    assert case.outcome is FeasibilityOutcome.UNRESOLVED
    assert "incomplete_harmful_feedback" in case.reason_codes
    assert case.benefit_explanation.outcome_status == "unresolved"


def test_candidate_must_change_target_layer_and_artifact_scope() -> None:
    parent, candidate = _replays()
    parent_artifact = _artifact("fixed.extraction.parent.v1", PolicyArtifactKind.FIXED)
    candidate_artifact = _artifact("adaptive.extraction.candidate.v1", PolicyArtifactKind.SINGLE_LAYER_ADAPTIVE)

    with pytest.raises(ValueError, match="must change the target-layer"):
        LayerIntervention(
            case_id="case.no_change",
            target_layer=PolicyLayer.EXTRACTION,
            parent=parent,
            candidate=parent,
            parent_artifact=parent_artifact,
            candidate_artifact=candidate_artifact,
            process_signal=True,
            outcome=FeasibilityOutcome.UNRESOLVED,
            reason_codes=("no_change",),
        )

    with pytest.raises(ValueError, match="artifacts must be distinct"):
        LayerIntervention(
            case_id="case.same_artifact",
            target_layer=PolicyLayer.EXTRACTION,
            parent=parent,
            candidate=candidate,
            parent_artifact=parent_artifact,
            candidate_artifact=parent_artifact,
            process_signal=True,
            outcome=FeasibilityOutcome.UNRESOLVED,
            reason_codes=("same_artifact",),
        )

    wrong_layer = PolicyArtifactIdentity.create(
        policy_version="adaptive.trigger.candidate.v1",
        kind=PolicyArtifactKind.SINGLE_LAYER_ADAPTIVE,
        layers=(PolicyLayer.TRIGGER,),
    )
    with pytest.raises(ValueError, match="target layer"):
        LayerIntervention(
            case_id="case.wrong_layer",
            target_layer=PolicyLayer.EXTRACTION,
            parent=parent,
            candidate=candidate,
            parent_artifact=parent_artifact,
            candidate_artifact=wrong_layer,
            process_signal=True,
            outcome=FeasibilityOutcome.UNRESOLVED,
            reason_codes=("wrong_layer",),
        )


def test_single_layer_intervention_rejects_upstream_decision_drift() -> None:
    snapshot, event = _event()
    common = {
        "backend": _backend(),
        "candidate_fact_ids": ("fact.tsv_preference",),
        "artifact_ids": ("artifact.tsv_preference",),
        "mutation_ids": ("mutation.parent",),
    }
    parent = DeterministicPolicyReplay().run(snapshot, event, **common)
    candidate = DeterministicPolicyReplay().run(
        snapshot, event, **{**common, "candidate_fact_ids": ()}
    )
    shadow_trigger = DeterministicPolicyReplay(
        trigger=DeterministicTriggerPolicy(
            config=TriggerPolicyConfig(task_completed_enabled=False)
        )
    ).run(snapshot, event, **common)
    altered = replace(
        candidate,
        decisions=(shadow_trigger.decisions[0], *candidate.decisions[1:]),
    )
    with pytest.raises(ValueError, match="upstream decision"):
        LayerIntervention(
            case_id="case.upstream-drift",
            target_layer=PolicyLayer.EXTRACTION,
            parent=parent,
            candidate=altered,
            parent_artifact=_artifact(
                "fixed.extraction.parent.v1", PolicyArtifactKind.FIXED
            ),
            candidate_artifact=_artifact(
                "adaptive.extraction.candidate.v1",
                PolicyArtifactKind.SINGLE_LAYER_ADAPTIVE,
            ),
            process_signal=True,
            outcome=FeasibilityOutcome.UNRESOLVED,
            reason_codes=("upstream_drift",),
        )

def test_contradictory_complete_chain_cannot_be_marked_unresolved() -> None:
    with pytest.raises(ValueError, match="cannot carry a complete reward chain"):
        _case(
            "case.contradictory",
            FeasibilityOutcome.UNRESOLVED,
            FeedbackChain("opportunity.1", "use.1", "outcome.1"),
        )


def test_feedback_ids_must_be_non_empty_strings() -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        FeedbackChain(opportunity_id=" ")


def test_hypothesis_must_bind_feedback_and_artifacts() -> None:
    parent, candidate = _replays()
    parent_artifact = _artifact("fixed.extraction.parent.v1", PolicyArtifactKind.FIXED)
    candidate_artifact = _artifact("adaptive.extraction.candidate.v1", PolicyArtifactKind.SINGLE_LAYER_ADAPTIVE)
    hypothesis = PolicyHypothesis.create(
        parent_artifact_id=parent_artifact.artifact_id,
        candidate_artifact_id=candidate_artifact.artifact_id,
        target_layer=PolicyLayer.EXTRACTION,
        feedback_ids=("foreign.feedback",),
    )
    with pytest.raises(ValueError, match="not bound to intervention"):
        LayerIntervention(
            case_id="case.foreign_hypothesis",
            target_layer=PolicyLayer.EXTRACTION,
            parent=parent,
            candidate=candidate,
            parent_artifact=parent_artifact,
            candidate_artifact=candidate_artifact,
            process_signal=True,
            outcome=FeasibilityOutcome.UNRESOLVED,
            reason_codes=("foreign_hypothesis",),
            hypothesis=hypothesis,
        )


def test_missing_process_signal_is_diagnostic_only_without_hypothesis() -> None:
    parent, candidate = _replays()
    case = LayerIntervention(
        case_id="case.no_process_signal",
        target_layer=PolicyLayer.EXTRACTION,
        parent=parent,
        candidate=candidate,
        parent_artifact=_artifact("fixed.extraction.parent.v1", PolicyArtifactKind.FIXED),
        candidate_artifact=_artifact("adaptive.extraction.candidate.v1", PolicyArtifactKind.SINGLE_LAYER_ADAPTIVE),
        process_signal=False,
        outcome=FeasibilityOutcome.UNRESOLVED,
        reason_codes=("process_signal_missing",),
    )
    assert case.process_feedback is None
    assert case.hypothesis is None
    census = next(
        item for item in build_feasibility_report((case,)).census
        if item.layer is PolicyLayer.EXTRACTION
    )
    assert census.status is FeasibilityStatus.DIAGNOSTIC_ONLY
    assert "no_process_signal" in census.reason_codes


def test_process_feedback_allows_empty_execution_receipts_but_requires_reasons() -> None:
    case = _case("case.receipts", FeasibilityOutcome.UNRESOLVED, FeedbackChain())
    assert case.process_feedback is not None
    feedback = replace(
        case.process_feedback,
        parent_execution_receipt_ids=(),
        candidate_execution_receipt_ids=(),
    )
    assert feedback.parent_execution_receipt_ids == ()
    assert feedback.candidate_execution_receipt_ids == ()

    with pytest.raises(ValueError, match="process feedback reason codes"):
        replace(feedback, reason_codes=())


def test_string_normalization_rejects_unhashable_values() -> None:
    case = _case("case.unhashable", FeasibilityOutcome.UNRESOLVED, FeedbackChain())
    assert case.process_feedback is not None
    with pytest.raises(ValueError, match="candidate execution receipts"):
        replace(case.process_feedback, candidate_execution_receipt_ids=({"id": "x"},))


def test_census_reports_unresolved_and_censored_separately() -> None:
    unresolved = _case("case.unresolved_ratio", FeasibilityOutcome.UNRESOLVED, FeedbackChain())
    censored = _case("case.censored_ratio", FeasibilityOutcome.CENSORED, FeedbackChain())
    census = next(
        item for item in build_feasibility_report((unresolved, censored)).census
        if item.layer is PolicyLayer.EXTRACTION
    )
    assert census.unknown_count == 2
    assert census.unresolved_count == 1
    assert census.censored_count == 1
    assert census.unresolved_ratio == 0.5
    assert census.censored_ratio == 0.5
    payload = census.payload()
    assert payload["unresolvedCount"] == 1
    assert payload["censoredCount"] == 1
    assert payload["resolvedUsefulRate"] is None


def test_census_reports_ambiguous_reason_separately() -> None:
    parent, candidate = _replays()
    case = LayerIntervention(
        case_id="case.ambiguous",
        target_layer=PolicyLayer.EXTRACTION,
        parent=parent,
        candidate=candidate,
        parent_artifact=_artifact("fixed.extraction.parent.v1", PolicyArtifactKind.FIXED),
        candidate_artifact=_artifact("adaptive.extraction.candidate.v1", PolicyArtifactKind.SINGLE_LAYER_ADAPTIVE),
        process_signal=True,
        outcome=FeasibilityOutcome.UNRESOLVED,
        reason_codes=("ambiguous_multi_artifact",),
    )
    census = next(
        item for item in build_feasibility_report((case,)).census
        if item.layer is PolicyLayer.EXTRACTION
    )
    assert census.ambiguous_count == 1
    assert census.ambiguous_ratio == 1.0


def test_failed_replay_audit_rejects_intervention() -> None:
    parent, candidate = _replays()
    failed_parent = replace(
        parent,
        audit=PolicyAuditReport(False, 0, 0, (), ("fixture_audit_failure",)),
    )
    with pytest.raises(ValueError, match="replay audit must pass"):
        LayerIntervention(
            case_id="case.failed_audit",
            target_layer=PolicyLayer.EXTRACTION,
            parent=failed_parent,
            candidate=candidate,
            parent_artifact=_artifact("fixed.extraction.parent.v1", PolicyArtifactKind.FIXED),
            candidate_artifact=_artifact("adaptive.extraction.candidate.v1", PolicyArtifactKind.SINGLE_LAYER_ADAPTIVE),
            process_signal=True,
            outcome=FeasibilityOutcome.UNRESOLVED,
            reason_codes=("audit_failure",),
        )


def test_report_rejects_duplicate_case_identity() -> None:
    case = _case(
        "case.duplicate",
        FeasibilityOutcome.UNRESOLVED,
        FeedbackChain(),
    )
    with pytest.raises(ValueError, match="case IDs must be unique"):
        build_feasibility_report((case, case))


def test_layer_census_requires_complete_bounded_outcome_counts() -> None:
    with pytest.raises(ValueError, match="outcome counts"):
        LayerFeasibilityCensus(
            layer=PolicyLayer.EXTRACTION,
            case_count=1,
            signal_count=1,
            action_variation_count=1,
            outcome_variation_count=0,
            outcome_counts={},
            unknown_count=0,
            complete_feedback_count=0,
            ambiguous_count=0,
            status=FeasibilityStatus.VALIDATION_ONLY,
            reason_codes=("no_outcome_variation",),
        )

    with pytest.raises(ValueError, match="outcome counts"):
        LayerFeasibilityCensus(
            layer=PolicyLayer.EXTRACTION,
            case_count=1,
            signal_count=1,
            action_variation_count=1,
            outcome_variation_count=0,
            outcome_counts={"not-a-feasibility-outcome": 1},
            unknown_count=0,
            complete_feedback_count=0,
            ambiguous_count=0,
            status=FeasibilityStatus.VALIDATION_ONLY,
            reason_codes=("no_outcome_variation",),
        )


def test_feasibility_evidence_ledger_is_idempotent_across_restart(tmp_path) -> None:
    case = _case(
        "case.durable",
        FeasibilityOutcome.USEFUL,
        FeedbackChain("opportunity.1", "use.1", "outcome.1"),
    )
    path = tmp_path / "policy-feasibility.jsonl"
    first = JsonFeasibilityEvidenceLedger(path)
    record = first.record_case(case)
    first.record(record)
    assert FeasibilityEvidenceRecord.from_payload(record.payload()) == record

    restarted = JsonFeasibilityEvidenceLedger(path)
    assert restarted.records == (record,)
    assert restarted.records[0].replay_payload == case.replay_payload
    assert restarted.verify_case(case) == record
    assert "The user prefers TSV output" not in path.read_text(encoding="utf-8")


def test_feasibility_evidence_ledger_rejects_corruption_and_conflict(tmp_path) -> None:
    case = _case(
        "case.corruption",
        FeasibilityOutcome.UNRESOLVED,
        FeedbackChain(),
    )
    path = tmp_path / "policy-feasibility.jsonl"
    ledger = JsonFeasibilityEvidenceLedger(path)
    record = ledger.record_case(case)
    payload = record.payload()
    payload["replayPayload"] = dict(payload["replayPayload"])
    payload["replayPayload"]["outcome"] = "useful"
    with pytest.raises(ValueError, match="record ID mismatch|conflicting|malformed"):
        FeasibilityEvidenceRecord.from_payload(payload)
    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed feasibility evidence"):
        JsonFeasibilityEvidenceLedger(path)


def test_feasibility_evidence_ledger_rejects_symlinked_paths(tmp_path) -> None:
    target = tmp_path / "target.jsonl"
    target.write_text("sentinel\n", encoding="utf-8")
    path = tmp_path / "policy-feasibility.jsonl"
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        JsonFeasibilityEvidenceLedger(path)
    assert target.read_text(encoding="utf-8") == "sentinel\n"


def test_feasibility_evidence_ledger_rejects_symlinked_lock(tmp_path) -> None:
    path = tmp_path / "policy-feasibility.jsonl"
    lock_target = tmp_path / "lock-target"
    lock_target.write_text("", encoding="utf-8")
    path.with_name(path.name + ".lock").symlink_to(lock_target)
    with pytest.raises(ValueError, match="lock.*symlink"):
        JsonFeasibilityEvidenceLedger(path)


def test_feasibility_evidence_schema_bump_rejects_old_payload() -> None:
    case = _case(
        "case.schema_bump",
        FeasibilityOutcome.UNRESOLVED,
        FeedbackChain(),
    )
    payload = FeasibilityEvidenceRecord.from_case(case).payload()
    payload["schemaVersion"] = 1
    with pytest.raises(ValueError, match="unsupported feasibility evidence schema|malformed"):
        FeasibilityEvidenceRecord.from_payload(payload)


def test_feasibility_ledger_missing_case_fails_closed(tmp_path) -> None:
    case = _case(
        "case.missing_receipt",
        FeasibilityOutcome.UNRESOLVED,
        FeedbackChain(),
    )
    ledger = JsonFeasibilityEvidenceLedger(tmp_path / "missing.jsonl")
    with pytest.raises(ValueError, match="record is missing"):
        ledger.verify_case(case)


def test_feasibility_ledger_does_not_retain_deleted_file_cache(tmp_path) -> None:
    case = _case(
        "case.deleted_file",
        FeasibilityOutcome.UNRESOLVED,
        FeedbackChain(),
    )
    path = tmp_path / "deleted.jsonl"
    ledger = JsonFeasibilityEvidenceLedger(path)
    ledger.record_case(case)
    path.unlink()
    with pytest.raises(ValueError, match="record is missing"):
        ledger.verify_case(case)


def test_process_feedback_tampering_is_rejected() -> None:
    case = _case(
        "case.feedback_tamper",
        FeasibilityOutcome.USEFUL,
        FeedbackChain("opportunity.1", "use.1", "outcome.1"),
    )
    payload = case.process_feedback.payload()
    payload["observed_after_digest"] = "0" * 64
    with pytest.raises(ValueError, match="ID mismatch|malformed"):
        type(case.process_feedback).from_payload(payload)


def test_hypothesis_tampering_is_rejected() -> None:
    case = _case(
        "case.hypothesis_tamper",
        FeasibilityOutcome.USEFUL,
        FeedbackChain("opportunity.1", "use.1", "outcome.1"),
    )
    payload = case.hypothesis.payload()
    payload["feedback_ids"] = ["foreign.feedback"]
    with pytest.raises(ValueError, match="ID mismatch|malformed"):
        type(case.hypothesis).from_payload(payload)


def test_feedback_and_hypothesis_collections_require_json_lists() -> None:
    case = _case(
        "case.collection_types",
        FeasibilityOutcome.USEFUL,
        FeedbackChain("opportunity.1", "use.1", "outcome.1"),
    )
    feedback = case.process_feedback.payload()
    feedback["reason_codes"] = "decision_observed"
    with pytest.raises(ValueError, match="malformed process feedback"):
        type(case.process_feedback).from_payload(feedback)
    hypothesis = case.hypothesis.payload()
    hypothesis["feedback_ids"] = "foreign.feedback"
    with pytest.raises(ValueError, match="malformed policy hypothesis"):
        type(case.hypothesis).from_payload(hypothesis)


def test_feasibility_replay_rejects_unhashable_id_values() -> None:
    case = _case(
        "case.unhashable_replay",
        FeasibilityOutcome.UNRESOLVED,
        FeedbackChain(),
    )
    payload = case.replay_payload
    payload["reason_codes"] = [{"reason": "not-a-string"}]
    with pytest.raises(ValueError, match="malformed feasibility evidence record"):
        FeasibilityEvidenceRecord.from_payload({
            "schemaVersion": 3,
            "recordId": "feasibility-record.invalid",
            "replayPayload": payload,
        })


def test_feasibility_replay_rejects_mismatched_benefit_explanation() -> None:
    case = _case(
        "case.benefit_tamper",
        FeasibilityOutcome.USEFUL,
        FeedbackChain("opportunity.1", "use.1", "outcome.1"),
    )
    payload = dict(case.replay_payload)
    payload["benefit_explanation"] = LayerBenefitExplanation.create(
        target_layer=PolicyLayer.SOURCE_SELECTION,
        outcome=FeasibilityOutcome.USEFUL,
    ).payload()
    with pytest.raises(ValueError, match="benefit explanation|malformed feasibility"):
        FeasibilityEvidenceRecord.from_payload({
            "schemaVersion": 3,
            "recordId": "feasibility-record.invalid",
            "replayPayload": payload,
        })


def test_executable_feasibility_census_replays_and_persists(tmp_path) -> None:
    evidence = tmp_path / "fixture-evidence.jsonl"
    first = run_default_feasibility_census(evidence_path=evidence)
    second = run_default_feasibility_census(evidence_path=evidence)
    assert first.digest == second.digest
    ledger = JsonFeasibilityEvidenceLedger(evidence)
    cases = build_default_feasibility_cases()
    assert len(ledger.records) == len(cases) == 7
    for case in cases:
        ledger.verify_case(case)
    assert first.payload()["caseCount"] == 7


def test_each_fixture_layer_has_explicit_benefit_explanation() -> None:
    cases = build_default_feasibility_cases()
    assert {case.target_layer for case in cases} == set(PolicyLayer)
    for case in cases:
        explanation = case.benefit_explanation
        assert isinstance(explanation, LayerBenefitExplanation)
        assert explanation.target_layer is case.target_layer
        assert explanation.payload()["mechanism_code"]
        assert explanation.summary
        assert explanation.outcome is case.outcome
        assert case.replay_payload["benefit_explanation"] == explanation.payload()
    report = run_default_feasibility_census().payload()
    assert all("benefitExplanation" in item for item in report["cases"])


def test_feasibility_report_exposes_decision_contract_for_every_layer() -> None:
    report = run_default_feasibility_census().payload()
    contracts = {
        item["layer"]: item["decisionContract"]
        for item in report["layers"]
    }
    assert set(contracts) == {layer.value for layer in PolicyLayer}
    for layer, contract in contracts.items():
        assert contract["layer"] == layer
        assert contract["contract_id"].startswith(
            "policy-decision-contract."
        )
        assert contract["required_fields"]
        assert set(contract["allowed_actions"]) == {"RUN", "SKIP", "DEFER"}


def test_every_layer_case_has_matched_process_intervention_identity() -> None:
    cases = build_default_feasibility_cases()
    assert {case.target_layer for case in cases} == set(PolicyLayer)
    for case in cases:
        validate_feasibility_case(case)
        feedback = case.process_feedback
        assert feedback is not None
        assert feedback.event_id == case.parent.event.event_id == case.candidate.event.event_id
        assert feedback.source_revision == case.parent.event.source_revision == case.candidate.event.source_revision
        assert feedback.parent_decision_id == case.parent_decision.decision_id
        assert feedback.candidate_decision_id == case.candidate_decision.decision_id
        assert feedback.observed_before_digest == case.parent_decision.output_digest
        assert feedback.observed_after_digest == case.candidate_decision.output_digest
        assert feedback.observed_before_digest != feedback.observed_after_digest
        assert case.action_changed is True
        target_events = tuple(
            event
            for event in case.candidate.process_events
            if event.policy_layer is case.target_layer
        )
        assert len(target_events) == 1
        assert target_events[0].policy_decision_id == case.candidate_decision.decision_id
        assert target_events[0].input_digest == case.candidate_decision.input_digest
        assert target_events[0].output_digest == case.candidate_decision.output_digest
        assert target_events[0].event_id == type(target_events[0]).from_payload(
            target_events[0].payload()
        ).event_id


def test_optimizer_corpus_primary_examples_feed_extraction_census() -> None:
    from test_extraction_optimizer_contracts import _multi_corpus
    from rsimem.memory.extraction_feedback import ExtractionFeedbackLabel

    corpus = _multi_corpus((ExtractionFeedbackLabel.USEFUL, ExtractionFeedbackLabel.MISSED))
    parent, candidate = _replays()
    cases = build_optimizer_corpus_interventions(
        corpus.examples,
        parent=parent,
        candidate=candidate,
        parent_artifact=_artifact("fixed.extraction.parent.v1", PolicyArtifactKind.FIXED),
        candidate_artifact=_artifact(
            "adaptive.extraction.candidate.v1",
            PolicyArtifactKind.SINGLE_LAYER_ADAPTIVE,
        ),
    )
    assert len(cases) == 2
    assert {case.outcome for case in cases} == {
        FeasibilityOutcome.USEFUL,
        FeasibilityOutcome.MISSED,
    }


def test_optimizer_result_projection_preserves_no_proposal_and_candidate_identity() -> None:
    from test_extraction_optimizer_contracts import (
        _corpus,
        _multi_corpus,
        _parent,
        _proposal_output,
    )
    from rsimem.memory.extraction_feedback import ExtractionFeedbackLabel
    from rsimem.memory.extraction_prompt_optimizer import (
        CapturedExtractionOptimizerClient,
        ExtractionPromptOptimizer,
    )

    parent = _parent()
    no_signal_corpus = _corpus()
    no_signal = ExtractionPromptOptimizer(
        CapturedExtractionOptimizerClient(_proposal_output)
    ).propose(parent, no_signal_corpus)
    projected_no_signal = project_optimizer_result(
        no_signal,
        no_signal_corpus,
        parent_artifact_id=parent.artifact_id,
    )
    assert projected_no_signal.decision is OptimizerHypothesisDecision.NO_PROPOSAL
    assert projected_no_signal.candidate_artifact_id is None
    assert OptimizerHypothesisProjection.from_payload(
        projected_no_signal.payload()
    ) == projected_no_signal

    corpus = _multi_corpus((ExtractionFeedbackLabel.USEFUL, ExtractionFeedbackLabel.USEFUL))
    proposal = ExtractionPromptOptimizer(
        CapturedExtractionOptimizerClient(_proposal_output)
    ).propose(parent, corpus)
    projected = project_optimizer_result(
        proposal,
        corpus,
        parent_artifact_id=parent.artifact_id,
    )
    assert projected.decision is OptimizerHypothesisDecision.PROPOSE
    assert projected.candidate_artifact_id == proposal.candidate.artifact_id
    assert set(projected.evidence_example_ids) == set(proposal.request.primary_example_ids)


def test_optimizer_result_projection_rejects_corpus_or_parent_drift() -> None:
    from test_extraction_optimizer_contracts import _corpus, _multi_corpus, _parent, _proposal_output
    from rsimem.memory.extraction_feedback import ExtractionFeedbackLabel
    from rsimem.memory.extraction_prompt_optimizer import CapturedExtractionOptimizerClient, ExtractionPromptOptimizer

    parent = _parent()
    corpus = _multi_corpus((ExtractionFeedbackLabel.USEFUL, ExtractionFeedbackLabel.USEFUL))
    result = ExtractionPromptOptimizer(CapturedExtractionOptimizerClient(_proposal_output)).propose(parent, corpus)
    with pytest.raises(ValueError, match="parent artifact"):
        project_optimizer_result(result, corpus, parent_artifact_id="extraction-prompt.foreign")
    with pytest.raises(ValueError, match="corpus identity"):
        project_optimizer_result(result, _corpus(), parent_artifact_id=parent.artifact_id)


def test_real_extraction_feedback_examples_project_only_resolved_primary_chain() -> None:
    from rsimem.memory.extraction_feedback import (
        ExtractionFeedbackBuilder,
        ExtractionFeedbackLabel,
        ExtractionFeedbackLevel,
        ExtractionSourceEvidence,
        ExtractedFactEvidence,
        FactDisposition,
        FutureMemoryEvidence,
        ArtifactSemanticBinding,
        DeploymentObservation,
        ExposureMode,
        ObservableToolEvent,
        default_feedback_contract_registry,
    )
    import hashlib

    digest = hashlib.sha256(b"source").hexdigest()
    source = ExtractionSourceEvidence(
        "source.real",
        digest,
        "extraction-set.real",
        "nonempty",
        ("preference.summary.tsv",),
        (ExtractedFactEvidence(
            "fact.real", ("preference.summary.tsv",), FactDisposition.PERSISTED,
            artifact_id="artifact.real",
        ),),
    )
    future = FutureMemoryEvidence(
        "opportunity.real",
        ExposureMode.EAGER_SYSTEM_PROMPT,
        (ArtifactSemanticBinding("artifact.real", "preference.summary.tsv"),),
        "operation.opportunity",
        "operation.injection",
    )
    observation = DeploymentObservation(
        "observation.real",
        "SM01_preference_adoption",
        "eval_near",
        "task.real",
        hashlib.sha256(b"current").hexdigest(),
        (),
        ("preference.summary.tsv",),
        "owner\tpriority\ttask\tdue_date\nA\thigh\tShip\t2026/09/01",
        (ObservableToolEvent("tool.real", "notes_share", True, recipient_ids=("owner",)),),
        True,
    )
    dataset = ExtractionFeedbackBuilder(default_feedback_contract_registry()).build(
        source, observation, future,
    )
    primary = next(item for item in dataset.examples if item.primary)
    chain = feedback_chain_from_extraction_example(primary)
    assert primary.label is ExtractionFeedbackLabel.USEFUL
    assert primary.level is ExtractionFeedbackLevel.EXTRACTION_SET
    assert chain.complete_useful
    assert chain.opportunity_id == future.future_opportunity_id
    parent, candidate = _replays()
    projected = LayerIntervention.from_extraction_feedback(
        case_id="case.real_feedback_projection",
        parent=parent,
        candidate=candidate,
        parent_artifact=_artifact("fixed.extraction.parent.v1", PolicyArtifactKind.FIXED),
        candidate_artifact=_artifact(
            "adaptive.extraction.candidate.v1",
            PolicyArtifactKind.SINGLE_LAYER_ADAPTIVE,
        ),
        example=primary,
    )
    assert projected.outcome is FeasibilityOutcome.USEFUL
    assert projected.feedback.complete_useful
    projected_batch = build_extraction_feedback_interventions(
        dataset.examples,
        parent=parent,
        candidate=candidate,
        parent_artifact=_artifact("fixed.extraction.parent.v1", PolicyArtifactKind.FIXED),
        candidate_artifact=_artifact(
            "adaptive.extraction.candidate.v1",
            PolicyArtifactKind.SINGLE_LAYER_ADAPTIVE,
        ),
    )
    assert len(projected_batch) == 1
    assert projected_batch[0].outcome is FeasibilityOutcome.USEFUL
    fact = next(item for item in dataset.examples if item.level is ExtractionFeedbackLevel.FACT)
    with pytest.raises(ValueError, match="primary extraction-set"):
        LayerIntervention.from_extraction_feedback(
            case_id="case.fact_feedback_rejected",
            parent=parent,
            candidate=candidate,
            parent_artifact=_artifact("fixed.extraction.parent.v1", PolicyArtifactKind.FIXED),
            candidate_artifact=_artifact(
                "adaptive.extraction.candidate.v1",
                PolicyArtifactKind.SINGLE_LAYER_ADAPTIVE,
            ),
            example=fact,
        )

    unresolved = next(
        item for item in ExtractionFeedbackBuilder(default_feedback_contract_registry()).build(
            source,
            observation.__class__(
                observation.observation_id, observation.family_id, observation.stage,
                observation.task_id, observation.current_input_projection_digest,
                observation.current_input_semantic_keys, observation.task_semantic_keys,
                "ordinary prose", observation.tool_events, observation.completed,
            ),
            future,
        ).examples if item.primary
    )
    assert unresolved.label is ExtractionFeedbackLabel.UNRESOLVED
    assert feedback_chain_from_extraction_example(unresolved).ids == ()


def test_deterministic_past_future_fixture_replays_useful_and_missed_feedback() -> None:
    parent, candidate = _replays()
    projected = {}
    for label in ("useful", "missed"):
        fixture = build_extraction_feedback_fixture(outcome=label)
        primary = next(example for example in fixture.dataset.examples if example.primary)
        chain = feedback_chain_from_extraction_example(primary)
        assert fixture.past_snapshot.snapshot_id == "snapshot.feasibility.default"
        assert fixture.past_snapshot.segments[0].segment_id == "segment.durable"
        assert fixture.past_snapshot.segments[1].segment_id == "segment.temporary"
        assert primary.label.value == label
        assert chain.complete_useful is (label == "useful")
        assert chain.complete_missed is (label == "missed")
        cases = build_extraction_feedback_interventions(
            fixture.dataset.examples,
            parent=parent,
            candidate=candidate,
            parent_artifact=_artifact(
                "fixed.extraction.parent.v1", PolicyArtifactKind.FIXED
            ),
            candidate_artifact=_artifact(
                "adaptive.extraction.candidate.v1",
                PolicyArtifactKind.SINGLE_LAYER_ADAPTIVE,
            ),
            case_id_prefix=f"case.fixture.{label}",
        )
        assert len(cases) == 1
        projected[label] = cases[0]
        assert "TSV output" not in str(cases[0].replay_payload)
    assert projected["useful"].outcome is FeasibilityOutcome.USEFUL
    assert projected["missed"].outcome is FeasibilityOutcome.MISSED


def _layer_artifact(layer: PolicyLayer, version: str, kind: PolicyArtifactKind) -> PolicyArtifactIdentity:
    return PolicyArtifactIdentity.create(policy_version=version, kind=kind, layers=(layer,))


def _event() -> tuple[ContextSnapshot, object]:
    snapshot = _snapshot()
    event = HostTriggerAdapter().event(
        "task_completed",
        source_revision=snapshot.context_revision,
        payload={"snapshot_id": snapshot.snapshot_id, "fixture": "all_layers"},
        session_id=snapshot.session_id,
        task_id=snapshot.task_id,
        turn_index=2,
    )
    return snapshot, event


def _run_pair(layer: PolicyLayer):
    snapshot, event = _event()
    common = {
        "backend": _backend(),
        "candidate_fact_ids": ("fact.tsv_preference",),
        "artifact_ids": ("artifact.tsv_preference", "artifact.secondary"),
        "mutation_ids": ("mutation.parent",),
    }
    parent = DeterministicPolicyReplay().run(snapshot, event, **common)
    if layer is PolicyLayer.TRIGGER:
        candidate = DeterministicPolicyReplay(
            trigger=DeterministicTriggerPolicy(
                TriggerPolicyConfig(task_completed_enabled=False)
            )
        ).run(snapshot, event, **common)
    elif layer is PolicyLayer.SOURCE_SELECTION:
        candidate = DeterministicPolicyReplay(
            source=DeterministicSourceSelectionPolicy(
                SourceSelectionConfig(
                    projection_mode="selected_completed_segments",
                    selected_segment_ids=("segment.durable",),
                )
            )
        ).run(snapshot, event, **common)
    elif layer is PolicyLayer.EXTRACTION:
        candidate = DeterministicPolicyReplay().run(
            snapshot, event, **{**common, "candidate_fact_ids": ()}
        )
    elif layer is PolicyLayer.ADMISSION:
        candidate = DeterministicPolicyReplay().run(
            snapshot,
            event,
            **{
                **common,
                "existing_artifact_ids": ("fact.tsv_preference",),
                "admission_update": True,
                "target_artifact_ids": ("artifact.existing",),
            },
        )
    elif layer is PolicyLayer.COMMIT:
        candidate = DeterministicPolicyReplay().run(
            snapshot, event, **{**common, "mutation_ids": ("mutation.candidate",)}
        )
    elif layer is PolicyLayer.EXPOSURE:
        candidate = DeterministicPolicyReplay().run(
            snapshot, event, **{**common, "artifact_ids": ("artifact.tsv_preference",)}
        )
    else:  # pragma: no cover - protects this fixture if PolicyLayer grows.
        raise AssertionError(layer)
    return parent, candidate


def _layer_case(
    layer: PolicyLayer,
    outcome: FeasibilityOutcome,
    suffix: str | None = None,
) -> LayerIntervention:
    parent, candidate = _run_pair(layer)
    feedback = (
        FeedbackChain("opportunity.layer", "use.layer", "outcome.layer")
        if outcome is FeasibilityOutcome.USEFUL
        else FeedbackChain(
            "opportunity.layer", "use.layer", "outcome.layer"
        )
        if outcome is FeasibilityOutcome.HARMFUL
        else FeedbackChain(
            source_id="source.layer",
            demand_id="demand.layer",
            absence_id="absence.layer",
            outcome_id="outcome.layer",
        )
        if outcome is FeasibilityOutcome.MISSED
        else FeedbackChain()
    )
    return LayerIntervention(
        case_id=f"case.{layer.value}{f'.{suffix}' if suffix else ''}",
        target_layer=layer,
        parent=parent,
        candidate=candidate,
        parent_artifact=_layer_artifact(layer, f"fixed.{layer.value}.v1", PolicyArtifactKind.FIXED),
        candidate_artifact=_layer_artifact(
            layer,
            f"adaptive.{layer.value}.candidate.v1",
            PolicyArtifactKind.SINGLE_LAYER_ADAPTIVE,
        ),
        process_signal=True,
        outcome=outcome,
        feedback=feedback,
        reason_codes=(f"fixture_{layer.value}",),
    )


def test_deterministic_fixture_covers_all_six_layers() -> None:
    cases = (
        _layer_case(PolicyLayer.TRIGGER, FeasibilityOutcome.UNRESOLVED),
        _layer_case(PolicyLayer.SOURCE_SELECTION, FeasibilityOutcome.MISSED),
        _layer_case(PolicyLayer.EXTRACTION, FeasibilityOutcome.USEFUL, "useful"),
        _layer_case(PolicyLayer.EXTRACTION, FeasibilityOutcome.MISSED, "missed"),
        _layer_case(PolicyLayer.ADMISSION, FeasibilityOutcome.HARMFUL),
        _layer_case(PolicyLayer.COMMIT, FeasibilityOutcome.USEFUL),
        _layer_case(PolicyLayer.EXPOSURE, FeasibilityOutcome.USEFUL),
    )
    for case in cases:
        validate_feasibility_case(case)
        assert case.action_changed
        assert case.parent.event == case.candidate.event
        assert case.parent.lineage.trigger_event_id == case.candidate.lineage.trigger_event_id

    report = build_feasibility_report(cases)
    assert {item.layer for item in report.census if item.case_count} == set(PolicyLayer)
    extraction = next(item for item in report.census if item.layer is PolicyLayer.EXTRACTION)
    assert extraction.status is FeasibilityStatus.OPTIMIZATION_READY
    assert extraction.benefit_explanation_codes == ("candidate_fact_set",)
    assert all(item.benefit_explanation_codes for item in report.census if item.case_count)
    assert all(
        item.status in {FeasibilityStatus.OPTIMIZATION_READY, FeasibilityStatus.VALIDATION_ONLY}
        for item in report.census
        if item.case_count
    )
    assert report.ok
    assert report.optimization_ready_layers == (PolicyLayer.EXTRACTION,)
    assert report.effect_experiment_ready is False
    payload = report.payload()
    assert payload["optimizationReadyLayers"] == ["extraction"]
    assert payload["effectExperimentReady"] is False
