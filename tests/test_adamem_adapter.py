from __future__ import annotations

import pytest

from rsimem.adamem_adapter import (
    AdaMemFeedbackView,
    AdaMemPolicy,
    apply_policy_patch,
    bind_to_mem0_flat,
    build_feedback_request,
    render_extraction_instructions,
    update_policy,
)


def test_patch_updates_policy_and_render_boundary() -> None:
    root = AdaMemPolicy.root()
    result = update_policy(
        parent=root,
        feedback_view=AdaMemFeedbackView.TERMINAL,
        feedback={"outcome": "not_resolved", "user_feedback": "Please retain decision details."},
        reflect=lambda _: '{"set":{"Ava":"Capture Ava\'s decisions and commitments."}}',
    )
    assert result.outcome == "updated"
    assert result.candidate_policy.parent_version == root.version
    rendered = render_extraction_instructions(result.candidate_policy)
    assert "Ava" in rendered
    assert result.candidate_policy.general_policy not in rendered


@pytest.mark.parametrize("raw,reason", [
    ("not json", "reflect_parse_failed"),
    ("[]", "patch_shape_invalid"),
    ("{}", "empty_patch"),
])
def test_patch_failure_or_noop_retains_parent(raw: str, reason: str) -> None:
    root = AdaMemPolicy.root()
    result = update_policy(
        parent=root,
        feedback_view=AdaMemFeedbackView.TERMINAL,
        feedback={"outcome": "visible_failure"},
        reflect=lambda _: raw,
    )
    assert result.outcome == "no_update"
    assert result.reason_code == reason
    assert result.candidate_policy == root


def test_feedback_views_enforce_content_boundary() -> None:
    root = AdaMemPolicy.root()
    with pytest.raises(ValueError, match="disallowed"):
        build_feedback_request(
            policy=root,
            feedback_view=AdaMemFeedbackView.TERMINAL,
            feedback={"messages": []},
        )
    with pytest.raises(ValueError, match="forbidden"):
        build_feedback_request(
            policy=root,
            feedback_view=AdaMemFeedbackView.FULL_TRAJECTORY,
            feedback={"outcome": "x", "grader": "must-not-leak"},
        )


def test_malformed_patch_is_rejected() -> None:
    assert apply_policy_patch(AdaMemPolicy.root(), {"unknown": "field"}) is None


def test_mem0_binding_changes_only_extraction_component() -> None:
    root = AdaMemPolicy.root()
    binding = bind_to_mem0_flat(root)
    assert binding.policy_version == root.version
    assert binding.extraction_component.slot_id == "mem0-flat.semantic.extraction"
    assert binding.extraction_component.source_provenance == root.digest
    assert binding.binding.artifact_id == binding.extraction_component.artifact_id
    assert binding.extraction_policy_artifact.policy_version == root.version
    assert binding.extraction_policy_artifact.compiled_body == binding.extraction_component.policy_body


def test_policy_payload_round_trip_is_strict() -> None:
    root = AdaMemPolicy.root()
    assert AdaMemPolicy.from_payload(root.payload()) == root
    with pytest.raises(ValueError, match="malformed"):
        AdaMemPolicy.from_payload({"version": root.version})
