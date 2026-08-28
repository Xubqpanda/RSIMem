"""Deterministic admission envelope for extracted memory candidates (2D.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .contracts import MemoryBackendDescriptor, MemoryKind
from .policy_contracts import (
    AdmissionDecision,
    DecisionAction,
    ExecutionStatus,
    MutationKind,
    PolicyArtifactIdentity,
    PolicyArtifactKind,
    PolicyLayer,
    SafetyBoundary,
    ExtractionDecision,
)


@dataclass(frozen=True, slots=True)
class AdmissionPolicyConfig:
    policy_version: str = "fixed.admission.parent.v1"
    memory_kind: MemoryKind = MemoryKind.SEMANTIC

    def __post_init__(self) -> None:
        if not self.policy_version.strip():
            raise ValueError("admission policy version must not be empty")
        object.__setattr__(self, "memory_kind", MemoryKind(self.memory_kind))


class DeterministicAdmissionPolicy:
    """Backend-aware parent admission policy.

    Candidate extraction is kept separate from admission: an empty extraction,
    filtered candidates, and a backend-rejected mutation produce distinct
    reason codes and digests.
    """

    layer = PolicyLayer.ADMISSION

    def __init__(self, config: AdmissionPolicyConfig | None = None) -> None:
        self.config = config or AdmissionPolicyConfig()

    @property
    def artifact_identity(self) -> PolicyArtifactIdentity:
        return PolicyArtifactIdentity.create(
            policy_version=self.config.policy_version,
            kind=PolicyArtifactKind.FIXED,
            layers=(PolicyLayer.ADMISSION,),
        )

    def decide(
        self,
        extraction: ExtractionDecision,
        *,
        backend: MemoryBackendDescriptor,
        backend_revision: str,
        existing_artifact_ids: Sequence[str] = (),
        update: bool = False,
        target_artifact_ids: Sequence[str] = (),
        safety: SafetyBoundary | None = None,
    ) -> AdmissionDecision:
        if not backend_revision.strip():
            raise ValueError("admission requires current backend revision")
        capability = backend.capability_for(self.config.memory_kind)
        if capability is None or not capability.writable:
            raise ValueError("backend does not support admission for memory kind")
        if update and (capability.updatable is not True):
            raise ValueError("backend does not support update")
        if type(update) is not bool:
            raise ValueError("update must be bool")
        candidate_ids = extraction.candidate_fact_ids
        existing = set(existing_artifact_ids)
        if len(existing) != len(tuple(existing_artifact_ids)):
            raise ValueError("existing artifact IDs must be unique")
        if update and not target_artifact_ids:
            raise ValueError("update admission requires target artifacts")
        if not update and target_artifact_ids:
            raise ValueError("add admission cannot carry target artifacts")
        filtered = tuple(item for item in candidate_ids if item in existing)
        accepted = tuple(item for item in candidate_ids if item not in existing)
        if update:
            accepted = candidate_ids
            filtered = ()
        if not candidate_ids:
            reason = "empty_extraction"
            mutation = MutationKind.NONE
        elif not accepted and not update:
            reason = "duplicate_candidates"
            mutation = MutationKind.NONE
        elif update:
            reason = "update_candidate"
            mutation = MutationKind.UPDATE
        else:
            reason = "add_candidate"
            mutation = MutationKind.ADD
        boundary = safety or SafetyBoundary()
        boundary.require_safe()
        return AdmissionDecision.create(
            policy_version=self.config.policy_version,
            source_revision=extraction.source_revision,
            input_payload={
                "extraction_decision_id": extraction.decision_id,
                "candidate_fact_ids": list(candidate_ids),
                "backend": backend.name,
                "backend_revision": backend_revision,
            },
            output_payload={
                "accepted_fact_ids": list(accepted),
                "filtered_fact_ids": list(filtered),
                "mutation_kind": mutation.value,
                "reason": reason,
                "target_artifact_ids": list(target_artifact_ids),
            },
            action=DecisionAction.RUN,
            execution_status=ExecutionStatus.PENDING,
            reason_codes=(reason,),
            lineage_id=extraction.lineage_id,
            trigger_event_id=extraction.trigger_event_id,
            candidate_fact_ids=candidate_ids,
            accepted_fact_ids=accepted,
            filtered_fact_ids=filtered,
            mutation_kind=mutation,
            backend_revision=backend_revision,
            target_artifact_ids=tuple(target_artifact_ids),
            update_supported=capability.updatable,
            safety=boundary,
        )

    def decide_admission(self, extraction: ExtractionDecision, **kwargs: object) -> AdmissionDecision:
        return self.decide(extraction, **kwargs)  # type: ignore[arg-type]


__all__ = ["AdmissionPolicyConfig", "DeterministicAdmissionPolicy"]
