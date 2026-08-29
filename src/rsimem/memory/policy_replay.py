"""Host-neutral six-layer deterministic replay harness.

This is a feasibility fixture, not a live mutation runner.  It composes the
current fixed parent policies while keeping backend writes and model calls
outside the replay boundary.  The result is therefore safe to use for
decision/action/lineage coverage before opening any adaptive layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..lifecycle.snapshot import ContextSnapshot
from .admission_policy import DeterministicAdmissionPolicy
from .contracts import MemoryBackendDescriptor
from .policy_audit import PolicyAuditReport, audit_policy_evidence
from .policy_contracts import (
    AdmissionDecision,
    CommitDecision,
    DecisionAction,
    ExecutionStatus,
    ExposureDecision,
    ExtractionDecision,
    FORMATION_LAYERS,
    MutationKind,
    PolicyLineage,
    PolicyDecision,
    SafetyBoundary,
    SourceSelectionDecision,
    TriggerEvent,
    content_digest,
)
from .source_selection_policy import DeterministicSourceSelectionPolicy
from .trigger_policy import DeterministicTriggerPolicy
from .exposure_policy import DeterministicExposurePolicy
from .process_feedback import ProcessEvent


@dataclass(frozen=True, slots=True)
class PolicyReplayResult:
    event: TriggerEvent
    decisions: tuple[PolicyDecision, ...]
    lineage: PolicyLineage
    audit: PolicyAuditReport
    process_events: tuple[ProcessEvent, ...] = ()


class DeterministicPolicyReplay:
    """Compose fixed policies into a stable, no-side-effect replay."""

    def __init__(
        self,
        *,
        trigger: DeterministicTriggerPolicy | None = None,
        source: DeterministicSourceSelectionPolicy | None = None,
        admission: DeterministicAdmissionPolicy | None = None,
        exposure: DeterministicExposurePolicy | None = None,
    ) -> None:
        self.trigger = trigger or DeterministicTriggerPolicy()
        self.source = source or DeterministicSourceSelectionPolicy()
        self.admission = admission or DeterministicAdmissionPolicy()
        self.exposure = exposure or DeterministicExposurePolicy()

    def run(
        self,
        snapshot: ContextSnapshot,
        event: TriggerEvent,
        *,
        backend: MemoryBackendDescriptor,
        candidate_fact_ids: Sequence[str] = (),
        artifact_ids: Sequence[str] = (),
        existing_artifact_ids: Sequence[str] = (),
        admission_update: bool = False,
        target_artifact_ids: Sequence[str] = (),
        mutation_ids: Sequence[str] = (),
        backend_revision: str = "backend.revision.1",
    ) -> PolicyReplayResult:
        if event.source_revision != snapshot.context_revision:
            raise ValueError("replay event and snapshot revisions differ")
        trigger_observation = self.trigger.decide(event)
        trigger = trigger_observation.decision
        decisions: list[PolicyDecision] = [trigger]
        source = self.source.select(snapshot, event)
        decisions.append(source)

        # A skipped/deferred trigger never executes downstream policy layers.
        # We return the partial chain so the caller can audit the explicit
        # reason without manufacturing extraction or mutation records.
        if trigger.action != DecisionAction.RUN:
            lineage = PolicyLineage.from_decisions(decisions)
            report = _audit_decisions(decisions)
            return PolicyReplayResult(
                event, tuple(decisions), lineage, report,
                _process_events(snapshot, event, decisions),
            )

        extraction = ExtractionDecision.create(
            policy_version="fixed.extraction.parent.v1",
            source_revision=snapshot.context_revision,
            input_payload={"source_decision_id": source.decision_id, "source_digest": source.source_digest},
            output_payload={"candidate_fact_ids": list(candidate_fact_ids)},
            action=DecisionAction.RUN if candidate_fact_ids else DecisionAction.SKIP,
            execution_status=ExecutionStatus.PENDING if candidate_fact_ids else ExecutionStatus.SKIPPED,
            reason_codes=("candidate_extracted" if candidate_fact_ids else "empty_extraction",),
            lineage_id=trigger.lineage_id,
            trigger_event_id=event.event_id,
            execution_receipt_id=(f"receipt.extraction.{source.decision_id[-12:]}" if candidate_fact_ids else None),
            candidate_fact_ids=tuple(candidate_fact_ids),
            source_digest=source.source_digest,
            request_id=f"request.{source.decision_id[-24:]}",
        )
        decisions.append(extraction)
        if extraction.action != DecisionAction.RUN:
            lineage = PolicyLineage.from_decisions(decisions)
            report = _audit_decisions(decisions)
            return PolicyReplayResult(
                event, tuple(decisions), lineage, report,
                _process_events(snapshot, event, decisions),
            )

        safety = SafetyBoundary(
            active_segment_ids=snapshot.active_segment_ids,
            current_turn_id=snapshot.current_turn_id,
            current_turn_segment_ids=tuple(
                segment.segment_id
                for segment in snapshot.segments
                if snapshot.current_turn_id is not None and segment.turn_id == snapshot.current_turn_id
            ),
            tool_closures=tuple(closure.segment_ids for closure in snapshot.tool_closures),
        )
        admission = self.admission.decide(
            extraction,
            backend=backend,
            backend_revision=backend_revision,
            existing_artifact_ids=existing_artifact_ids,
            update=admission_update,
            target_artifact_ids=target_artifact_ids,
            safety=safety,
        )
        decisions.append(admission)

        if admission.mutation_kind != MutationKind.NONE:
            resolved_mutation_ids = tuple(mutation_ids) or (
                f"mutation.{content_digest({'admission': admission.decision_id})[:24]}",
            )
            commit = CommitDecision.create(
                policy_version="fixed.commit.parent.v1",
                source_revision=snapshot.context_revision,
                input_payload={"admission_decision_id": admission.decision_id},
                output_payload={"mutation_ids": list(resolved_mutation_ids)},
                action=DecisionAction.RUN,
                execution_status=ExecutionStatus.PENDING,
                reason_codes=("commit_scheduled",),
                lineage_id=trigger.lineage_id,
                trigger_event_id=event.event_id,
                execution_receipt_id=f"receipt.commit.{admission.decision_id[-12:]}",
                mutation_ids=resolved_mutation_ids,
                expected_revision=backend_revision,
                execution_boundary="task_completed",
                safety=safety,
            )
            decisions.append(commit)

        exposure = self.exposure.decide(event, tuple(artifact_ids))
        decisions.append(exposure)
        injection_ids = (
            (f"receipt.injection.{exposure.decision_id[-12:]}",)
            if exposure.action == DecisionAction.RUN
            else ()
        )
        lineage = PolicyLineage.from_decisions(
            decisions,
            mutation_receipt_ids=tuple(
                item.execution_receipt_id
                for item in decisions
                if item.layer.value == "commit" and item.execution_receipt_id
            ),
            injection_receipt_ids=injection_ids,
        )
        report = _audit_decisions(
            decisions,
            mutation_receipt_ids=lineage.mutation_receipt_ids,
            injection_receipt_ids=lineage.injection_receipt_ids,
            require_all_layers=False,
        )
        return PolicyReplayResult(
            event, tuple(decisions), lineage, report,
            _process_events(
                snapshot,
                event,
                decisions,
                mutation_receipt_ids=lineage.mutation_receipt_ids,
                injection_receipt_ids=lineage.injection_receipt_ids,
            ),
        )


def _audit_decisions(
    decisions: Sequence[PolicyDecision],
    *,
    mutation_receipt_ids: Sequence[str] = (),
    injection_receipt_ids: Sequence[str] = (),
    require_all_layers: bool = False,
) -> PolicyAuditReport:
    from .policy_contracts import validate_policy_episode

    return validate_policy_episode(
        decisions,
        mutation_receipt_ids=mutation_receipt_ids,
        injection_receipt_ids=injection_receipt_ids,
        require_all_layers=require_all_layers,
    )


def _process_events(
    snapshot: ContextSnapshot,
    event: TriggerEvent,
    decisions: Sequence[PolicyDecision],
    *,
    mutation_receipt_ids: Sequence[str] = (),
    injection_receipt_ids: Sequence[str] = (),
) -> tuple[ProcessEvent, ...]:
    """Project every replayed decision into content-free process evidence."""

    result: list[ProcessEvent] = []
    for decision in decisions:
        receipts = tuple(mutation_receipt_ids) if decision.layer.value == "commit" else (
            tuple(injection_receipt_ids) if decision.layer.value == "exposure" else ()
        )
        result.append(ProcessEvent.from_policy_decision(
            decision,
            run_id=snapshot.run_id,
            variant="deterministic",
            trace_id=snapshot.episode_id,
            episode_id=snapshot.episode_id,
            session_id=snapshot.session_id,
            task_id=snapshot.task_id,
            host_event_id=event.event_id,
            family_id=None,
            stage=None,
            execution_receipt_ids=receipts,
        ))
    return tuple(result)


__all__ = ["PolicyReplayResult", "DeterministicPolicyReplay"]
