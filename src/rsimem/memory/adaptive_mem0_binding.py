"""Bind one active adaptive artifact to the fixed Mem0-flat utility gate."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..memory_systems.mem0_flat.utility_gate import FrozenMem0UtilityGate
from .adaptive_policy import (
    FIXED_INVOCATION_BOUNDARY,
    FIXED_SEMANTIC_ROUTE,
    AdaptiveParameterName,
)
from .adaptive_policy_store import JsonAdaptivePolicyStore
from .adaptive_policy_validation import AdaptiveValidationDecision
from .adaptive_matched_validation import MatchedValidationDecision
from .utility import (
    STATIC_UTILITY_FEATURE_SCHEMA,
    STATIC_UTILITY_POLICY_VERSION,
    StaticUtilityPolicy,
    UtilityDecision,
    UtilityTarget,
)


@dataclass(frozen=True, slots=True)
class AdaptiveMem0Binding:
    gate: FrozenMem0UtilityGate
    adaptive: bool
    actual_policy_version: str
    artifact_id: str | None
    parameter_names: tuple[AdaptiveParameterName, ...]

    def observer_evidence(self) -> dict[str, object]:
        return {
            "adaptive": self.adaptive,
            "actual_policy_version": self.actual_policy_version,
            "artifact_id": self.artifact_id,
            "parameter_names": [value.value for value in self.parameter_names],
            "gate_digest": self.gate.digest,
            "feature_schema": self.gate.feature_schema,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveMem0EndToEndAudit:
    ok: bool
    issues: tuple[str, ...]
    dataset_id: str
    artifact_id: str
    parent_policy_version: str
    parent_future_policy_version: str
    adaptive_policy_version: str
    offline_decision_id: str
    matched_decision_id: str
    future_target: UtilityTarget
    parent_disposition: str
    adaptive_disposition: str


def audit_adaptive_mem0_loop(
    *,
    store: JsonAdaptivePolicyStore,
    binding: AdaptiveMem0Binding,
    offline_decision: AdaptiveValidationDecision,
    matched_decision: MatchedValidationDecision,
    parent_future_decision: UtilityDecision,
    adaptive_future_decision: UtilityDecision,
) -> AdaptiveMem0EndToEndAudit:
    issues = set()
    active = store.snapshot().active
    if active is None:
        raise ValueError("adaptive Mem0 audit requires one active artifact")
    if (
        not offline_decision.accepted
        or offline_decision.artifact_id != active.artifact_id
        or offline_decision.policy_version != active.policy_version
        or offline_decision.dataset_id != active.dataset_id
        or offline_decision.training_example_ids != active.training_example_ids
    ):
        issues.add("offline_validation_mismatch")
    if (
        not matched_decision.accepted
        or matched_decision.artifact_id != active.artifact_id
        or matched_decision.policy_version != active.policy_version
        or matched_decision.parent_policy_version != active.parent_policy_version
    ):
        issues.add("matched_validation_mismatch")
    if (
        not binding.adaptive
        or binding.artifact_id != active.artifact_id
        or binding.actual_policy_version != active.policy_version
    ):
        issues.add("runtime_binding_mismatch")
    if (
        parent_future_decision.target != adaptive_future_decision.target
        or parent_future_decision.feature_digest
        != adaptive_future_decision.feature_digest
        or parent_future_decision.cost_digest != adaptive_future_decision.cost_digest
    ):
        issues.add("future_input_mismatch")
    if parent_future_decision.policy_version != STATIC_UTILITY_POLICY_VERSION:
        issues.add("parent_future_version_mismatch")
    if adaptive_future_decision.policy_version != active.policy_version:
        issues.add("adaptive_future_version_mismatch")
    if parent_future_decision.disposition == adaptive_future_decision.disposition:
        issues.add("future_decision_unchanged")
    return AdaptiveMem0EndToEndAudit(
        ok=not issues,
        issues=tuple(sorted(issues)),
        dataset_id=active.dataset_id,
        artifact_id=active.artifact_id,
        parent_policy_version=active.parent_policy_version,
        parent_future_policy_version=parent_future_decision.policy_version,
        adaptive_policy_version=active.policy_version,
        offline_decision_id=offline_decision.decision_id,
        matched_decision_id=matched_decision.decision_id,
        future_target=adaptive_future_decision.target,
        parent_disposition=parent_future_decision.disposition.value,
        adaptive_disposition=adaptive_future_decision.disposition.value,
    )


class ActiveAdaptiveMem0Binder:
    """Apply only allowlisted threshold values from the unique active artifact."""

    def bind(
        self,
        store: JsonAdaptivePolicyStore,
        base_gate: FrozenMem0UtilityGate = FrozenMem0UtilityGate(),
    ) -> AdaptiveMem0Binding:
        active = store.snapshot().active
        if active is None:
            return AdaptiveMem0Binding(
                gate=base_gate,
                adaptive=False,
                actual_policy_version=base_gate.policy.policy_version,
                artifact_id=None,
                parameter_names=(),
            )
        if (
            active.route_backend != FIXED_SEMANTIC_ROUTE
            or active.invocation_boundary != FIXED_INVOCATION_BOUNDARY
            or active.feature_schema != STATIC_UTILITY_FEATURE_SCHEMA
        ):
            raise ValueError("active adaptive artifact changes the fixed semantic route")

        runtime_policy = replace(
            base_gate.policy,
            policy_version=active.policy_version,
        )
        target_policies = {
            target: replace(policy, policy_version=active.policy_version)
            for target, policy in base_gate.target_policies
        }
        update_policy = (
            None
            if base_gate.update_policy is None
            else replace(
                base_gate.update_policy,
                policy_version=active.policy_version,
            )
        )
        names = []
        for update in active.parameters:
            names.append(update.name)
            target = {
                AdaptiveParameterName.EXTRACTION_ACCEPT_THRESHOLD: (
                    UtilityTarget.GENERATION
                ),
                AdaptiveParameterName.INTERNAL_OPERATION_ACCEPT_THRESHOLD: (
                    UtilityTarget.INTERNAL_OPERATION
                ),
                AdaptiveParameterName.RETRIEVAL_ACCEPT_THRESHOLD: (
                    UtilityTarget.RETRIEVAL
                ),
            }.get(update.name)
            base_policy = (
                update_policy
                if update.name == AdaptiveParameterName.CONSOLIDATION_UPDATE_THRESHOLD
                and update_policy is not None
                else target_policies.get(target, runtime_policy)
            )
            policy = replace(
                base_policy,
                policy_version=active.policy_version,
                accept_threshold=update.proposed_value,
            )
            if update.name == AdaptiveParameterName.EXTRACTION_ACCEPT_THRESHOLD:
                target_policies[UtilityTarget.GENERATION] = policy
            elif update.name == (
                AdaptiveParameterName.INTERNAL_OPERATION_ACCEPT_THRESHOLD
            ):
                target_policies[UtilityTarget.INTERNAL_OPERATION] = policy
            elif update.name == AdaptiveParameterName.RETRIEVAL_ACCEPT_THRESHOLD:
                target_policies[UtilityTarget.RETRIEVAL] = policy
            elif update.name == AdaptiveParameterName.CONSOLIDATION_UPDATE_THRESHOLD:
                update_policy = policy
            else:  # pragma: no cover - enum normalization owns this boundary.
                raise ValueError("unsupported adaptive Mem0 parameter")
        gate = FrozenMem0UtilityGate(
            config=base_gate.config,
            policy=runtime_policy,
            target_policies=tuple(sorted(
                target_policies.items(),
                key=lambda item: item[0].value,
            )),
            update_policy=update_policy,
        )
        return AdaptiveMem0Binding(
            gate=gate,
            adaptive=True,
            actual_policy_version=active.policy_version,
            artifact_id=active.artifact_id,
            parameter_names=tuple(names),
        )
