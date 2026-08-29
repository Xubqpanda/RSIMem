"""Deterministic admission envelope for extracted memory candidates (2D.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

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


@dataclass(frozen=True, slots=True)
class AdmissionCensus:
    """Content-free counts used by the admission anti-collapse gate.

    The census is intentionally separate from extraction quality: it reports
    whether a candidate changed admission behavior and whether that change
    looks like blanket ADD/NONE or duplicate admission.  It never treats a
    ``NONE`` caused by empty extraction as a duplicate.
    """

    decision_count: int
    nonempty_extraction_count: int
    add_count: int
    update_count: int
    delete_count: int
    none_count: int
    duplicate_add_count: int
    accepted_fact_count: int
    filtered_fact_count: int

    def __post_init__(self) -> None:
        values = (
            self.decision_count,
            self.nonempty_extraction_count,
            self.add_count,
            self.update_count,
            self.delete_count,
            self.none_count,
            self.duplicate_add_count,
            self.accepted_fact_count,
            self.filtered_fact_count,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("admission census counts must be non-negative integers")
        if any(value > self.decision_count for value in values[:7]):
            raise ValueError("admission census counts cannot exceed decision count")

    @property
    def all_add(self) -> bool:
        return bool(self.decision_count) and self.add_count == self.decision_count

    @property
    def all_none(self) -> bool:
        return bool(self.decision_count) and self.none_count == self.decision_count

    @property
    def nonempty_coverage(self) -> float | None:
        return (
            self.nonempty_extraction_count / self.decision_count
            if self.decision_count
            else None
        )

    def payload(self) -> dict[str, object]:
        return {
            "decisionCount": self.decision_count,
            "nonemptyExtractionCount": self.nonempty_extraction_count,
            "addCount": self.add_count,
            "updateCount": self.update_count,
            "deleteCount": self.delete_count,
            "noneCount": self.none_count,
            "duplicateAddCount": self.duplicate_add_count,
            "acceptedFactCount": self.accepted_fact_count,
            "filteredFactCount": self.filtered_fact_count,
            "allAdd": self.all_add,
            "allNone": self.all_none,
            "nonemptyCoverage": self.nonempty_coverage,
        }


def census_admission_decisions(
    decisions: Sequence[AdmissionDecision],
) -> AdmissionCensus:
    """Summarize admission decisions without reading fact content."""

    items = tuple(decisions)
    if any(not isinstance(item, AdmissionDecision) for item in items):
        raise TypeError("admission census requires AdmissionDecision values")
    counts = {kind: 0 for kind in MutationKind}
    nonempty = 0
    duplicate_add = 0
    accepted = 0
    filtered = 0
    for item in items:
        counts[item.mutation_kind] += 1
        if item.candidate_fact_ids:
            nonempty += 1
        accepted += len(item.accepted_fact_ids)
        filtered += len(item.filtered_fact_ids)
        if (
            item.mutation_kind is MutationKind.ADD
            and any("duplicate" in reason.casefold() for reason in item.reason_codes)
        ):
            duplicate_add += 1
    return AdmissionCensus(
        decision_count=len(items),
        nonempty_extraction_count=nonempty,
        add_count=counts[MutationKind.ADD],
        update_count=counts[MutationKind.UPDATE],
        delete_count=counts[MutationKind.DELETE],
        none_count=counts[MutationKind.NONE],
        duplicate_add_count=duplicate_add,
        accepted_fact_count=accepted,
        filtered_fact_count=filtered,
    )


def validate_admission_candidate(
    parent: Mapping[str, AdmissionDecision],
    candidate: Mapping[str, AdmissionDecision],
    *,
    minimum_coverage_ratio: float = 1.0,
) -> tuple[AdmissionCensus, AdmissionCensus]:
    """Fail closed on blanket admission strategies.

    ``parent`` and ``candidate`` are keyed by a caller-owned stable source ID;
    positional pairing is deliberately not accepted.  The candidate cannot
    manufacture a quality improvement by turning every source into ADD,
    turning every source into NONE, or emitting an ADD marked as duplicate.
    Coverage is measured on the extraction candidate set, not on accepted fact
    count, so one source cannot be amplified by splitting facts.
    """

    if type(minimum_coverage_ratio) not in {int, float} or not 0 <= minimum_coverage_ratio <= 1:
        raise ValueError("minimum coverage ratio must be between zero and one")
    if not parent or not candidate or set(parent) != set(candidate):
        raise ValueError("admission parent/candidate source identities must match")
    for source_id, decision in (*parent.items(), *candidate.items()):
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("admission source IDs must be non-empty")
        if not isinstance(decision, AdmissionDecision):
            raise TypeError("admission candidate requires AdmissionDecision values")
    for source_id in parent:
        before = parent[source_id]
        after = candidate[source_id]
        if before.source_revision != after.source_revision:
            raise ValueError("admission parent/candidate source revision differs")
    parent_census = census_admission_decisions(tuple(parent.values()))
    candidate_census = census_admission_decisions(tuple(candidate.values()))
    reasons: list[str] = []
    if candidate_census.duplicate_add_count:
        reasons.append("duplicate_add")
    if candidate_census.all_none and not parent_census.all_none:
        reasons.append("candidate_all_none")
    if candidate_census.all_add and (
        not parent_census.all_add
        or candidate_census.nonempty_extraction_count
        > parent_census.nonempty_extraction_count
    ):
        reasons.append("candidate_all_add")
    parent_coverage = parent_census.nonempty_coverage
    candidate_coverage = candidate_census.nonempty_coverage
    if (
        parent_coverage is None
        or candidate_coverage is None
        or candidate_coverage < parent_coverage * float(minimum_coverage_ratio)
    ):
        reasons.append("admission_coverage_collapse")
    if reasons:
        raise ValueError("admission candidate rejected: " + ",".join(dict.fromkeys(reasons)))
    return parent_census, candidate_census


__all__ = [
    "AdmissionPolicyConfig",
    "DeterministicAdmissionPolicy",
    "AdmissionCensus",
    "census_admission_decisions",
    "validate_admission_candidate",
]
