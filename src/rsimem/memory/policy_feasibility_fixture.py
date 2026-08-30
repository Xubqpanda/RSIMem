"""Executable deterministic six-layer feasibility fixture.

The fixture is intentionally provider-free and content-free at its report
boundary.  It is suitable for replay/census smoke tests; its outcome labels
are fixture-local demonstrations, not deployment rewards.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..lifecycle.snapshot import (
    ContextSnapshot,
    ProvenanceRef,
    SnapshotSegment,
    TaskLifecycleState,
)
from .contracts import MemoryAccessMode, MemoryBackendDescriptor, MemoryKind, MemoryKindCapability
from .policy_contracts import (
    PolicyArtifactIdentity,
    PolicyArtifactKind,
    PolicyLayer,
    content_digest,
)
from .policy_feasibility import (
    FeasibilityOutcome,
    FeedbackChain,
    JsonFeasibilityEvidenceLedger,
    LayerIntervention,
    PolicyFeasibilityReport,
    build_feasibility_report,
)
from .policy_replay import DeterministicPolicyReplay
from .source_selection_policy import DeterministicSourceSelectionPolicy, SourceSelectionConfig
from .trigger_policy import DeterministicTriggerPolicy, HostTriggerAdapter, TriggerPolicyConfig
from .extraction_feedback import (
    ArtifactSemanticBinding,
    DeploymentObservation,
    ExtractedFactEvidence,
    ExtractionFeedbackBuilder,
    ExtractionFeedbackDataset,
    ExtractionFeedbackLabel,
    ExtractionSetStatus,
    ExtractionSourceEvidence,
    FactDisposition,
    FeedbackOperationJoin,
    FutureMemoryEvidence,
    MissedExtractionEvidence,
    ObservableToolEvent,
    ExposureMode,
    default_feedback_contract_registry,
)


@dataclass(frozen=True, slots=True)
class DeterministicExtractionFeedbackFixture:
    """A replayable past-context/late-outcome extraction fixture.

    The source and delayed observation are content-bearing only inside the
    owner-controlled fixture.  Any feasibility report built from this object
    should project stable IDs and digests, never the message or memory text.
    """

    past_snapshot: ContextSnapshot
    source: ExtractionSourceEvidence
    observation: DeploymentObservation
    future: FutureMemoryEvidence
    operation_join: FeedbackOperationJoin
    missed: tuple[MissedExtractionEvidence, ...]
    dataset: ExtractionFeedbackDataset


_TSV_KEY = "preference.summary.tsv"


def build_extraction_feedback_fixture(
    *,
    outcome: ExtractionFeedbackLabel = ExtractionFeedbackLabel.MISSED,
) -> DeterministicExtractionFeedbackFixture:
    """Build deterministic SM01 useful or missed delayed feedback.

    The past snapshot includes one durable preference and one temporary
    request.  The future task requires only the durable preference.  ``MISSED``
    models policy N omitting that fact; ``USEFUL`` models the same fact being
    exposed and explicitly used.  No grader, answer, benchmark score, or
    resource quantity is reachable from this contract.
    """

    outcome = ExtractionFeedbackLabel(outcome)
    if outcome not in {ExtractionFeedbackLabel.MISSED, ExtractionFeedbackLabel.USEFUL}:
        raise ValueError("deterministic extraction fixture supports useful or missed")
    source_digest = content_digest({
        "fixture": "sm01-preference",
        "snapshot_id": build_fixture_snapshot().snapshot_id,
        "segments": [segment.segment_id for segment in build_fixture_snapshot().segments],
    })
    if outcome is ExtractionFeedbackLabel.USEFUL:
        facts = (ExtractedFactEvidence(
            "fact.feasibility.tsv",
            (_TSV_KEY,),
            FactDisposition.PERSISTED,
            artifact_id="artifact.feasibility.tsv",
        ),)
        source_status = ExtractionSetStatus.NONEMPTY
        future = FutureMemoryEvidence(
            "opportunity.feasibility.useful",
            ExposureMode.EAGER_SYSTEM_PROMPT,
            (ArtifactSemanticBinding("artifact.feasibility.tsv", (_TSV_KEY,)),),
            "op.opportunity.feasibility",
            "op.injection.feasibility",
        )
        observation = DeploymentObservation(
            "observation.feasibility.useful",
            "SM01_preference_adoption",
            "eval_near",
            "task.feasibility.future",
            content_digest("Prepare the task report."),
            (),
            (_TSV_KEY,),
            "owner\tpriority\ttask\tdue_date\nA\thigh\tShip\t2026/09/01",
            (ObservableToolEvent(
                "tool.feasibility.share",
                "notes_share",
                True,
                recipient_ids=("owner",),
            ),),
            True,
        )
        missed = ()
    else:
        facts = ()
        source_status = ExtractionSetStatus.EMPTY
        future = FutureMemoryEvidence(
            "opportunity.feasibility.missed",
            ExposureMode.NOT_EXPOSED,
            (),
            "op.opportunity.feasibility",
            None,
        )
        observation = DeploymentObservation(
            "observation.feasibility.missed",
            "SM01_preference_adoption",
            "eval_near",
            "task.feasibility.future",
            content_digest("Prepare the task report."),
            (),
            (_TSV_KEY,),
            "I will prepare the report.",
            (),
            True,
        )
        missed = ()
    source = ExtractionSourceEvidence(
        "source.feasibility.sm01",
        source_digest,
        "extraction-set.feasibility.sm01",
        source_status,
        (_TSV_KEY,),
        facts,
    )
    operation_join = FeedbackOperationJoin(
        "op.opportunity.feasibility",
        "op.use.feasibility",
        "op.outcome.feasibility",
    )
    if outcome is ExtractionFeedbackLabel.MISSED:
        missed = ExtractionFeedbackBuilder(
            default_feedback_contract_registry()
        ).derive_missed(
            source,
            observation,
            future,
            operation_join=operation_join,
        )
    dataset = ExtractionFeedbackBuilder(
        default_feedback_contract_registry()
    ).build(
        source,
        observation,
        future,
        missed=missed,
        operation_join=operation_join,
    )
    if dataset.examples[0].label is not outcome:
        raise AssertionError("deterministic extraction fixture produced wrong label")
    return DeterministicExtractionFeedbackFixture(
        build_fixture_snapshot(),
        source,
        observation,
        future,
        operation_join,
        tuple(missed),
        dataset,
    )


def build_fixture_snapshot() -> ContextSnapshot:
    snapshot_id = "snapshot.feasibility.default"
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
        "run.feasibility.default",
        "episode.feasibility.default",
        "session.feasibility.default",
        "task.feasibility.default",
        snapshot_id,
        "revision.feasibility.default",
        segments,
        (),
        None,
        TaskLifecycleState.COMPLETED,
        "task_completed",
        (),
        11,
        ProvenanceRef(
            "run.feasibility.default",
            "episode.feasibility.default",
            "session.feasibility.default",
            "task.feasibility.default",
            snapshot_id,
            "fixture.sm01",
        ),
    )


def build_fixture_backend() -> MemoryBackendDescriptor:
    return MemoryBackendDescriptor(
        "backend.feasibility.default",
        (MemoryKindCapability(MemoryKind.SEMANTIC, MemoryAccessMode.EAGER),),
    )


def _artifact(layer: PolicyLayer, version: str, kind: PolicyArtifactKind) -> PolicyArtifactIdentity:
    return PolicyArtifactIdentity.create(policy_version=version, kind=kind, layers=(layer,))


def _event(snapshot: ContextSnapshot):
    return HostTriggerAdapter().event(
        "task_completed",
        source_revision=snapshot.context_revision,
        payload={"snapshot_id": snapshot.snapshot_id, "fixture": "six_layer"},
        session_id=snapshot.session_id,
        task_id=snapshot.task_id,
        turn_index=2,
    )


def _pair(snapshot: ContextSnapshot, layer: PolicyLayer):
    event = _event(snapshot)
    common = {
        "backend": build_fixture_backend(),
        "candidate_fact_ids": ("fact.tsv_preference",),
        "artifact_ids": ("artifact.tsv_preference", "artifact.secondary"),
        "mutation_ids": ("mutation.parent",),
    }
    parent = DeterministicPolicyReplay().run(snapshot, event, **common)
    if layer is PolicyLayer.TRIGGER:
        candidate = DeterministicPolicyReplay(
            trigger=DeterministicTriggerPolicy(TriggerPolicyConfig(task_completed_enabled=False))
        ).run(snapshot, event, **common)
    elif layer is PolicyLayer.SOURCE_SELECTION:
        candidate = DeterministicPolicyReplay(
            source=DeterministicSourceSelectionPolicy(SourceSelectionConfig(
                projection_mode="selected_completed_segments",
                selected_segment_ids=("segment.durable",),
            ))
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
    else:  # pragma: no cover
        raise ValueError(f"unsupported fixture layer: {layer}")
    return parent, candidate


def build_default_feasibility_cases() -> tuple[LayerIntervention, ...]:
    snapshot = build_fixture_snapshot()
    cases: list[LayerIntervention] = []
    definitions = (
        (PolicyLayer.TRIGGER, FeasibilityOutcome.UNRESOLVED, FeedbackChain(), "trigger"),
        (PolicyLayer.SOURCE_SELECTION, FeasibilityOutcome.MISSED,
         FeedbackChain(source_id="source.layer", demand_id="demand.layer", absence_id="absence.layer", outcome_id="outcome.layer"), "source"),
        (PolicyLayer.EXTRACTION, FeasibilityOutcome.USEFUL,
         FeedbackChain(opportunity_id="opportunity.useful", use_id="use.useful", outcome_id="outcome.useful"), "extraction-useful"),
        (PolicyLayer.EXTRACTION, FeasibilityOutcome.MISSED,
         FeedbackChain(source_id="source.missed", demand_id="demand.missed", absence_id="absence.missed", outcome_id="outcome.missed"), "extraction-missed"),
        (PolicyLayer.ADMISSION, FeasibilityOutcome.HARMFUL,
         FeedbackChain(
             opportunity_id="opportunity.admission",
             use_id="use.admission",
             outcome_id="outcome.admission",
         ), "admission"),
        (PolicyLayer.COMMIT, FeasibilityOutcome.USEFUL,
         FeedbackChain(opportunity_id="opportunity.commit", use_id="use.commit", outcome_id="outcome.commit"), "commit"),
        (PolicyLayer.EXPOSURE, FeasibilityOutcome.USEFUL,
         FeedbackChain(opportunity_id="opportunity.exposure", use_id="use.exposure", outcome_id="outcome.exposure"), "exposure"),
    )
    for layer, outcome, feedback, suffix in definitions:
        parent, candidate = _pair(snapshot, layer)
        cases.append(LayerIntervention(
            case_id=f"case.{suffix}",
            target_layer=layer,
            parent=parent,
            candidate=candidate,
            parent_artifact=_artifact(layer, f"fixed.{layer.value}.parent.v1", PolicyArtifactKind.FIXED),
            candidate_artifact=_artifact(layer, f"adaptive.{layer.value}.{suffix}.v1", PolicyArtifactKind.SINGLE_LAYER_ADAPTIVE),
            process_signal=True,
            outcome=outcome,
            feedback=feedback,
            reason_codes=(f"fixture_{layer.value}",),
        ))
    return tuple(cases)


def run_default_feasibility_census(
    *,
    evidence_path: Path | None = None,
) -> PolicyFeasibilityReport:
    cases = build_default_feasibility_cases()
    report = build_feasibility_report(cases)
    if evidence_path is not None:
        ledger = JsonFeasibilityEvidenceLedger(evidence_path)
        for case in cases:
            ledger.record_case(case)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_default_feasibility_census(evidence_path=args.evidence)
    serialized = json.dumps(report.payload(), ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    return 0


__all__ = [
    "DeterministicExtractionFeedbackFixture",
    "build_fixture_snapshot",
    "build_fixture_backend",
    "build_extraction_feedback_fixture",
    "build_default_feasibility_cases",
    "run_default_feasibility_census",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
