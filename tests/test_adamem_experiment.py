from __future__ import annotations

import pytest

from rsimem.adamem_adapter import AdaMemFeedbackView
from rsimem.adamem_experiment import (
    AdaMemComparisonManifest,
    AdaMemCondition,
    feedback_view_for_condition,
)


def _manifest(replicates: int = 1) -> AdaMemComparisonManifest:
    return AdaMemComparisonManifest.create(
        family_id="SM01_preference_adoption", train_sequence_id="train.v1",
        n_plus_one_sequence_id="n-plus-one.v1", fixture_digest="a" * 64,
        base_model="gpt-5.6-luna", meta_agent_model="gpt-5.6-luna",
        token_budget=4096, update_budget=1, replicate_count=replicates,
    )


def test_manifest_fixes_identity_and_isolates_every_condition() -> None:
    manifest = _manifest(2)
    assert len(manifest.runs) == 6
    assert len({run.mem0_collection for run in manifest.runs}) == 6
    assert manifest.policy_update_space == "versioned_semantic_extraction_policy_only"
    assert manifest.identity_payload()["base_model"] == manifest.identity_payload()["meta_agent_model"]


def test_conditions_differ_only_in_feedback_view() -> None:
    assert feedback_view_for_condition(AdaMemCondition.MEM0_STATIC) is None
    assert feedback_view_for_condition(AdaMemCondition.ADAMEM_TERMINAL) is AdaMemFeedbackView.TERMINAL
    assert feedback_view_for_condition(AdaMemCondition.ADAMEM_FULL_TRAJECTORY) is AdaMemFeedbackView.FULL_TRAJECTORY


def test_manifest_rejects_non_policy_update_space() -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match="update space"):
        AdaMemComparisonManifest(
            manifest_id=manifest.manifest_id, family_id=manifest.family_id,
            train_sequence_id=manifest.train_sequence_id,
            n_plus_one_sequence_id=manifest.n_plus_one_sequence_id,
            fixture_digest=manifest.fixture_digest, base_model=manifest.base_model,
            meta_agent_model=manifest.meta_agent_model, temperature=manifest.temperature,
            token_budget=manifest.token_budget, update_budget=manifest.update_budget,
            mem0_backend=manifest.mem0_backend, policy_update_space="retrieval_and_policy",
            upstream_commit=manifest.upstream_commit, runs=manifest.runs,
        )
