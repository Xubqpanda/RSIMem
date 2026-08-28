from __future__ import annotations

import pytest

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
    FeedbackChain,
    FeasibilityOutcome,
    FeasibilityStatus,
    LayerIntervention,
    build_feasibility_report,
    validate_feasibility_case,
)
from rsimem.memory.policy_replay import DeterministicPolicyReplay
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
    assert useful.replay_payload["intervention_fingerprint"] == useful.intervention_fingerprint
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


def test_report_rejects_duplicate_case_identity() -> None:
    case = _case(
        "case.duplicate",
        FeasibilityOutcome.UNRESOLVED,
        FeedbackChain(),
    )
    with pytest.raises(ValueError, match="case IDs must be unique"):
        build_feasibility_report((case, case))


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
