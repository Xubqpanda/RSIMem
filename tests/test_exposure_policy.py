from __future__ import annotations

import pytest

from rsimem.memory.exposure_policy import (
    DeterministicExposurePolicy,
    ExposurePolicyConfig,
    InjectionReceiptStatus,
)
from rsimem.memory.policy_contracts import DecisionAction, ExposureMode, SafetyBoundary
from rsimem.memory.trigger_policy import HostTriggerAdapter


def _event():
    return HostTriggerAdapter().event(
        "task_completed",
        source_revision="context.rev.1",
        payload={"turn": 2},
        session_id="session.fixture",
        task_id="task.fixture",
    )


def test_empty_memory_is_skip_and_does_not_create_injection_receipt() -> None:
    decision = DeterministicExposurePolicy().decide(_event(), ())
    assert decision.action.value == "SKIP"
    assert decision.exposure_mode is ExposureMode.NOT_EXPOSED
    with pytest.raises(ValueError, match="only run"):
        DeterministicExposurePolicy.bind_injection(decision, context_revision="context.rev.1", render_fingerprint="render.1")


def test_eager_exposure_selects_in_order_and_receipt_exactly_joins() -> None:
    policy = DeterministicExposurePolicy(ExposurePolicyConfig(max_tokens=5))
    decision = policy.decide(_event(), ("artifact.1", "artifact.2", "artifact.3"), artifact_token_counts=(2, 3, 2))
    assert decision.action.value == "RUN"
    assert decision.exposure_mode is ExposureMode.EAGER_SYSTEM_PROMPT
    assert decision.selected_artifact_ids == ("artifact.1", "artifact.2")
    receipt = policy.bind_injection(decision, context_revision="context.rev.1", render_fingerprint="render.abc")
    assert receipt.artifact_ids == decision.selected_artifact_ids
    assert receipt.status is InjectionReceiptStatus.COMMITTED


def test_selective_mode_and_budget_zero_are_deterministic() -> None:
    policy = DeterministicExposurePolicy(ExposurePolicyConfig(mode=ExposureMode.SELECTIVE_RETRIEVAL))
    first = policy.decide(_event(), ("artifact.1", "artifact.2"), artifact_token_counts=(1, 1), budget_tokens=0)
    replay = policy.decide(_event(), ("artifact.1", "artifact.2"), artifact_token_counts=(1, 1), budget_tokens=0)
    assert first == replay
    assert first.action.value == "SKIP"


def test_invalid_safety_boundary_fails_closed_without_injection() -> None:
    decision = DeterministicExposurePolicy().decide(
        _event(),
        ("artifact.1",),
        safety=SafetyBoundary(schema_valid=False),
    )
    assert decision.action is DecisionAction.SKIP
    assert decision.execution_status.value == "skipped"
    assert decision.selected_artifact_ids == ()
    assert decision.reason_codes == ("safety_boundary_invalid",)


def test_receipt_rejects_revision_mismatch() -> None:
    decision = DeterministicExposurePolicy().decide(_event(), ("artifact.1",))
    with pytest.raises(ValueError, match="revision"):
        DeterministicExposurePolicy.bind_injection(decision, context_revision="context.rev.2", render_fingerprint="render.1")
