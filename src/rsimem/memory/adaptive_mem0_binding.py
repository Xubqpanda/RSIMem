"""Bind one active adaptive artifact to the fixed Mem0-flat utility gate."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from ..memory_systems.mem0_flat.utility_gate import FrozenMem0UtilityGate
from .adaptive_policy import (
    ADAPTIVE_POLICY_ARTIFACT_SCHEMA,
    ADAPTIVE_POLICY_OBJECTIVE,
    ADAPTIVE_POLICY_SCHEMA_VERSION,
    FIXED_INVOCATION_BOUNDARY,
    FIXED_SEMANTIC_ROUTE,
    AdaptiveParameterName,
)
from .adaptive_policy_store import JsonAdaptivePolicyStore
from .adaptive_policy_validation import AdaptiveValidationDecision
from .adaptive_matched_validation import MatchedValidationDecision
from .feedback_dataset import (
    DELAYED_FEEDBACK_DATASET_VERSION,
    DELAYED_FEEDBACK_LABEL_SCHEMA,
    DELAYED_FEEDBACK_WINDOW_VERSION,
)
from .utility import (
    STATIC_UTILITY_FEATURE_SCHEMA,
    STATIC_UTILITY_POLICY_VERSION,
    StaticUtilityPolicy,
    UtilityDecision,
    UtilityTarget,
)


@dataclass(frozen=True, slots=True)
class TrustedAdaptiveMem0Parameter:
    """Runtime-owned identity for one threshold the learner may update."""

    parameter_id: str
    name: AdaptiveParameterName
    prompt_ref: str
    baseline_value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", AdaptiveParameterName(self.name))
        if not self.parameter_id.strip() or not self.prompt_ref.strip():
            raise ValueError("trusted adaptive Mem0 parameter identity is incomplete")
        if (
            not isinstance(self.baseline_value, (int, float))
            or isinstance(self.baseline_value, bool)
            or not math.isfinite(float(self.baseline_value))
            or not 0.0 <= float(self.baseline_value) <= 1.0
        ):
            raise ValueError("trusted adaptive Mem0 baseline must be in [0,1]")
        object.__setattr__(self, "baseline_value", float(self.baseline_value))


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

    def __init__(
        self,
        trusted_parameters: tuple[TrustedAdaptiveMem0Parameter, ...],
    ) -> None:
        self.trusted_parameters = tuple(trusted_parameters)
        parameter_ids = tuple(item.parameter_id for item in self.trusted_parameters)
        parameter_names = tuple(item.name for item in self.trusted_parameters)
        if (
            not self.trusted_parameters
            or len(parameter_ids) != len(set(parameter_ids))
            or len(parameter_names) != len(set(parameter_names))
        ):
            raise ValueError("trusted adaptive Mem0 parameters must be unique")

    def bind(
        self,
        store: JsonAdaptivePolicyStore,
        base_gate: FrozenMem0UtilityGate = FrozenMem0UtilityGate(),
        *,
        expected_parent_policy_version: str,
    ) -> AdaptiveMem0Binding:
        if not expected_parent_policy_version.strip():
            raise ValueError("adaptive Mem0 parent policy identity is required")
        active = store.snapshot().active
        if active is None:
            return AdaptiveMem0Binding(
                gate=base_gate,
                adaptive=False,
                actual_policy_version=expected_parent_policy_version,
                artifact_id=None,
                parameter_names=(),
            )
        if (
            active.schema_version != ADAPTIVE_POLICY_SCHEMA_VERSION
            or active.artifact_schema != ADAPTIVE_POLICY_ARTIFACT_SCHEMA
            or active.parent_policy_version != expected_parent_policy_version
            or active.route_backend != FIXED_SEMANTIC_ROUTE
            or active.invocation_boundary != FIXED_INVOCATION_BOUNDARY
            or active.feature_schema != STATIC_UTILITY_FEATURE_SCHEMA
            or active.label_schema != DELAYED_FEEDBACK_LABEL_SCHEMA
            or active.dataset_version != DELAYED_FEEDBACK_DATASET_VERSION
            or active.window_version != DELAYED_FEEDBACK_WINDOW_VERSION
            or active.objective != ADAPTIVE_POLICY_OBJECTIVE
        ):
            raise ValueError("active adaptive artifact changes the frozen runtime contract")

        trusted = {
            item.parameter_id: item
            for item in self.trusted_parameters
        }
        names = tuple(update.name for update in active.parameters)
        if len(names) != len(set(names)):
            raise ValueError("active adaptive Mem0 parameter names must be unique")
        for update, prompt_ref in zip(active.parameters, active.prompt_refs):
            owner = trusted.get(update.parameter_id)
            if (
                owner is None
                or owner.name != update.name
                or owner.prompt_ref != prompt_ref
                or owner.baseline_value != update.baseline_value
            ):
                raise ValueError("active adaptive Mem0 parameter is not runtime-owned")

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
            if base_policy.accept_threshold != update.baseline_value:
                raise ValueError("active adaptive Mem0 baseline differs from runtime")
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
