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
    LayerIntervention,
    JsonFeasibilityEvidenceLedger,
    PolicyHypothesis,
    feedback_chain_from_extraction_example,
    build_feasibility_report,
    build_extraction_feedback_interventions,
    validate_feasibility_case,
)
from rsimem.memory.policy_replay import DeterministicPolicyReplay
from rsimem.memory.policy_feasibility_fixture import (
    build_default_feasibility_cases,
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


def test_feasibility_ledger_missing_case_fails_closed(tmp_path) -> None:
    case = _case(
        "case.missing_receipt",
        FeasibilityOutcome.UNRESOLVED,
        FeedbackChain(),
    )
    ledger = JsonFeasibilityEvidenceLedger(tmp_path / "missing.jsonl")
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
    assert all(
        item.status in {FeasibilityStatus.OPTIMIZATION_READY, FeasibilityStatus.VALIDATION_ONLY}
        for item in report.census
        if item.case_count
    )
    assert report.ok
