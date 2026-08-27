"""Frozen utility gate for Mem0-flat generation, operations, and retrieval."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from ...memory.ingestion import (
    InternalMemoryAction,
    InternalOperationProposal,
    SemanticIngestRequest,
)
from ...memory.utility import (
    InterpretableStaticUtilityScorer,
    LifecycleCostName,
    StaticUtilityFeatureExtractor,
    STATIC_UTILITY_FEATURE_SCHEMA,
    StaticUtilityPolicy,
    UtilityDecision,
    UtilityDisposition,
    UtilityTarget,
    known_lifecycle_costs,
)


FROZEN_MEM0_UTILITY_GATE_VERSION = "mem0-flat-static-utility-gate-v1"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenMem0UtilityConfig:
    version: str = FROZEN_MEM0_UTILITY_GATE_VERSION
    recency: float = 1.0
    add_conflict_risk: float = 0.05
    update_conflict_risk: float = 0.25
    delete_conflict_risk: float = 0.50
    add_recovery_risk: float = 0.05
    update_recovery_risk: float = 0.25
    delete_recovery_risk: float = 0.40
    generation_costs: tuple[tuple[LifecycleCostName, float], ...] = (
        (LifecycleCostName.GENERATION_INPUT_TOKENS, 5_000),
        (LifecycleCostName.GENERATION_OUTPUT_TOKENS, 500),
        (LifecycleCostName.STORAGE_BYTES, 512),
        (LifecycleCostName.RETRIEVAL_COUNT, 3),
        (LifecycleCostName.INJECTION_TOKENS, 128),
        (LifecycleCostName.RECOVERY_DURATION_MS, 0),
    )
    operation_costs: tuple[tuple[LifecycleCostName, float], ...] = (
        (LifecycleCostName.GENERATION_INPUT_TOKENS, 5_000),
        (LifecycleCostName.GENERATION_OUTPUT_TOKENS, 500),
        (LifecycleCostName.STORAGE_BYTES, 512),
        (LifecycleCostName.RETRIEVAL_COUNT, 3),
        (LifecycleCostName.INJECTION_TOKENS, 128),
        (LifecycleCostName.RECOVERY_DURATION_MS, 1_000),
    )
    retrieval_costs: tuple[tuple[LifecycleCostName, float], ...] = (
        (LifecycleCostName.GENERATION_INPUT_TOKENS, 0),
        (LifecycleCostName.GENERATION_OUTPUT_TOKENS, 0),
        (LifecycleCostName.STORAGE_BYTES, 0),
        (LifecycleCostName.RETRIEVAL_COUNT, 1),
        (LifecycleCostName.INJECTION_TOKENS, 128),
        (LifecycleCostName.RECOVERY_DURATION_MS, 0),
    )

    def __post_init__(self) -> None:
        for value in (
            self.recency,
            self.add_conflict_risk,
            self.update_conflict_risk,
            self.delete_conflict_risk,
            self.add_recovery_risk,
            self.update_recovery_risk,
            self.delete_recovery_risk,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("frozen Mem0 utility risk features must be in [0,1]")
        for costs in (self.generation_costs, self.operation_costs, self.retrieval_costs):
            if len(costs) != len(set(name for name, _ in costs)) or set(
                name for name, _ in costs
            ) != set(LifecycleCostName):
                raise ValueError("frozen Mem0 utility costs must cover every bucket")
            if any(value < 0 for _, value in costs):
                raise ValueError("frozen Mem0 utility costs must not be negative")

    @property
    def digest(self) -> str:
        return _digest({
            "version": self.version,
            "recency": self.recency,
            "conflict": [
                self.add_conflict_risk,
                self.update_conflict_risk,
                self.delete_conflict_risk,
            ],
            "recovery": [
                self.add_recovery_risk,
                self.update_recovery_risk,
                self.delete_recovery_risk,
            ],
            "generation_costs": [(name.value, value) for name, value in self.generation_costs],
            "operation_costs": [(name.value, value) for name, value in self.operation_costs],
            "retrieval_costs": [(name.value, value) for name, value in self.retrieval_costs],
        })

@dataclass(frozen=True, slots=True)
class FrozenMem0UtilityGate:
    """Apply a frozen scorer without changing Mem0 prompt invocation cadence."""

    config: FrozenMem0UtilityConfig = FrozenMem0UtilityConfig()
    policy: StaticUtilityPolicy = StaticUtilityPolicy()
    target_policies: tuple[tuple[UtilityTarget, StaticUtilityPolicy], ...] = ()
    update_policy: StaticUtilityPolicy | None = None
    _decisions: dict[str, list[UtilityDecision]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        normalized = tuple(
            (UtilityTarget(target), policy)
            for target, policy in self.target_policies
        )
        object.__setattr__(self, "target_policies", normalized)
        targets = tuple(target for target, _ in normalized)
        if len(targets) != len(set(targets)) or any(
            target not in {
                UtilityTarget.GENERATION,
                UtilityTarget.INTERNAL_OPERATION,
                UtilityTarget.RETRIEVAL,
            }
            for target in targets
        ):
            raise ValueError("Mem0 utility target policy overrides are invalid")

    @property
    def digest(self) -> str:
        identity = {
            "gate": self.config.digest,
            "policy": self.policy.digest,
        }
        if self.target_policies:
            identity["target_policies"] = [
                (target.value, policy.digest)
                for target, policy in self.target_policies
            ]
        if self.update_policy is not None:
            identity["update_policy"] = self.update_policy.digest
        return _digest(identity)

    @property
    def feature_schema(self) -> str:
        return STATIC_UTILITY_FEATURE_SCHEMA

    def decisions(self, request_id: str) -> tuple[UtilityDecision, ...]:
        return tuple(self._decisions.get(request_id, ()))

    def begin_request(self, request_id: str) -> None:
        if not request_id.strip():
            raise ValueError("utility request identity must not be empty")
        self._decisions[request_id] = []

    def observer_evidence(self, request_id: str) -> dict[str, object]:
        return {
            "schema_version": 1,
            "gate_version": self.config.version,
            "gate_digest": self.digest,
            "feature_schema": self.feature_schema,
            "bound_policy_versions": sorted({
                self.policy.policy_version,
                *(policy.policy_version for _, policy in self.target_policies),
                *(
                    ()
                    if self.update_policy is None
                    else (self.update_policy.policy_version,)
                ),
            }),
            "request_id": request_id,
            "decisions": [
                decision.observer_evidence()
                for decision in self.decisions(request_id)
            ],
        }

    def policy_for(self, target: UtilityTarget) -> StaticUtilityPolicy:
        target = UtilityTarget(target)
        return next(
            (
                policy
                for candidate, policy in self.target_policies
                if candidate == target
            ),
            self.policy,
        )

    def _score(
        self,
        request: SemanticIngestRequest,
        *,
        target: UtilityTarget,
        reuse_likelihood: float,
        conflict_risk: float,
        recovery_risk: float,
        predicted_benefit: float | None,
        costs: tuple[tuple[LifecycleCostName, float], ...],
        policy_override: StaticUtilityPolicy | None = None,
    ) -> UtilityDecision:
        features = StaticUtilityFeatureExtractor().extract(
            request.exit_evidence,
            available_at=0,
            recency=self.config.recency,
            reuse_likelihood=reuse_likelihood,
            conflict_risk=conflict_risk,
            recovery_risk=recovery_risk,
            predicted_benefit=predicted_benefit,
        )
        decision = InterpretableStaticUtilityScorer(
            policy_override or self.policy_for(target)
        ).score(
            features,
            known_lifecycle_costs(available_at=0, values=dict(costs)),
            target=target,
            cutoff=0,
        )
        self._decisions.setdefault(request.idempotency_key, []).append(decision)
        return decision

    def generation_decisions(
        self,
        request: SemanticIngestRequest,
        fact_count: int,
    ) -> tuple[UtilityDecision, ...]:
        reuse = 1.0 if request.exit_evidence.reusable_facts else request.exit_evidence.utility_estimate
        return tuple(
            self._score(
                request,
                target=UtilityTarget.GENERATION,
                reuse_likelihood=reuse,
                conflict_risk=self.config.add_conflict_risk,
                recovery_risk=self.config.add_recovery_risk,
                predicted_benefit=None,
                costs=self.config.generation_costs,
            )
            for _ in range(fact_count)
        )

    def rank_related(
        self,
        request: SemanticIngestRequest,
        views: Sequence[Any],
    ) -> tuple[Any, ...]:
        ranked = []
        for view in views:
            similarity = float(view.score)
            decision = self._score(
                request,
                target=UtilityTarget.RETRIEVAL,
                reuse_likelihood=similarity,
                conflict_risk=0.0,
                recovery_risk=0.0,
                predicted_benefit=similarity,
                costs=self.config.retrieval_costs,
            )
            if decision.disposition == UtilityDisposition.ACCEPT:
                ranked.append((decision.score, view.candidate.candidate_id, view))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return tuple(item[2] for item in ranked)

    def apply_operations(
        self,
        request: SemanticIngestRequest,
        proposals: Sequence[InternalOperationProposal],
        generation: Sequence[UtilityDecision],
    ) -> tuple[InternalOperationProposal, ...]:
        if len(proposals) != len(generation):
            raise ValueError("utility generation decisions must align with proposals")
        resolved = []
        for proposal, generation_decision in zip(proposals, generation):
            if proposal.action == InternalMemoryAction.NONE:
                resolved.append(proposal)
                continue
            conflict = {
                InternalMemoryAction.ADD: self.config.add_conflict_risk,
                InternalMemoryAction.UPDATE: self.config.update_conflict_risk,
                InternalMemoryAction.DELETE: self.config.delete_conflict_risk,
            }[proposal.action]
            recovery = {
                InternalMemoryAction.ADD: self.config.add_recovery_risk,
                InternalMemoryAction.UPDATE: self.config.update_recovery_risk,
                InternalMemoryAction.DELETE: self.config.delete_recovery_risk,
            }[proposal.action]
            operation = self._score(
                request,
                target=UtilityTarget.INTERNAL_OPERATION,
                reuse_likelihood=(
                    1.0
                    if request.exit_evidence.reusable_facts
                    else request.exit_evidence.utility_estimate
                ),
                conflict_risk=conflict,
                recovery_risk=recovery,
                predicted_benefit=None,
                costs=self.config.operation_costs,
                policy_override=(
                    self.update_policy
                    if proposal.action == InternalMemoryAction.UPDATE
                    and self.update_policy is not None
                    else None
                ),
            )
            if (
                generation_decision.disposition == UtilityDisposition.ACCEPT
                and operation.disposition == UtilityDisposition.ACCEPT
            ):
                resolved.append(proposal)
            else:
                resolved.append(InternalOperationProposal(
                    InternalMemoryAction.NONE,
                    (
                        "utility_rejected"
                        if UtilityDisposition.REJECT in {
                            generation_decision.disposition,
                            operation.disposition,
                        }
                        else "utility_deferred"
                    ),
                ))
        return tuple(resolved)
