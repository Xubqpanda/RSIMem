from __future__ import annotations

from dataclasses import replace

import pytest

from rsimem.lifecycle import (
    CompletionStatus,
    ExitEvidence,
    MemoryScope,
    TemporalValidity,
)
from rsimem.memory.utility import (
    CostObservation,
    FeatureObservation,
    FeatureSource,
    InterpretableStaticUtilityScorer,
    LifecycleCostName,
    LifecycleCostProfile,
    MissingReason,
    StaticUtilityFeatureExtractor,
    UtilityDisposition,
    UtilityFeatureName,
    UtilityTarget,
    known_lifecycle_costs,
)


def _exit_evidence() -> ExitEvidence:
    return ExitEvidence(
        CompletionStatus.COMPLETED,
        ("host_task_completed",),
        True,
        None,
        MemoryScope.USER,
        TemporalValidity.DURABLE,
        ("snapshot:fixture",),
        ("Use TSV output.",),
        (),
        (),
    )


def _features(**overrides):
    values = {
        "available_at": 10,
        "recency": 0.8,
        "reuse_likelihood": 0.9,
        "conflict_risk": 0.1,
        "recovery_risk": 0.1,
        "predicted_benefit": 0.9,
        "confidence": 0.9,
    }
    values.update(overrides)
    return StaticUtilityFeatureExtractor().extract(_exit_evidence(), **values)


def _costs(**overrides):
    values = {
        LifecycleCostName.GENERATION_INPUT_TOKENS: 1000,
        LifecycleCostName.GENERATION_OUTPUT_TOKENS: 100,
        LifecycleCostName.STORAGE_BYTES: 100,
        LifecycleCostName.RETRIEVAL_COUNT: 1,
        LifecycleCostName.INJECTION_TOKENS: 100,
        LifecycleCostName.RECOVERY_DURATION_MS: 10,
    }
    values.update(overrides)
    return known_lifecycle_costs(available_at=10, values=values)


def test_deterministic_feature_extraction_and_shared_target_semantics() -> None:
    first = _features()
    second = _features()
    assert first == second
    assert first.digest == second.digest
    assert {item.source for item in first.observations} == {
        FeatureSource.HOST_OBSERVED,
        FeatureSource.MODEL_PREDICTED,
    }

    scorer = InterpretableStaticUtilityScorer()
    decisions = [
        scorer.score(first, _costs(), target=target, cutoff=10)
        for target in UtilityTarget
    ]
    assert {item.score for item in decisions}.__len__() == 1
    assert all(item.disposition == UtilityDisposition.ACCEPT for item in decisions)
    assert all(item.feature_digest == first.digest for item in decisions)
    assert all(item.contributions for item in decisions)


def test_cost_and_benefit_monotonicity() -> None:
    scorer = InterpretableStaticUtilityScorer()
    baseline = scorer.score(
        _features(predicted_benefit=0.7),
        _costs(),
        target=UtilityTarget.GENERATION,
        cutoff=10,
    )
    costly = scorer.score(
        _features(predicted_benefit=0.7),
        _costs(**{LifecycleCostName.STORAGE_BYTES: 4096}),
        target=UtilityTarget.GENERATION,
        cutoff=10,
    )
    useful = scorer.score(
        _features(predicted_benefit=1.0),
        _costs(),
        target=UtilityTarget.GENERATION,
        cutoff=10,
    )
    assert costly.score <= baseline.score
    assert useful.score >= baseline.score


def test_missing_cost_no_history_low_confidence_and_conflict_are_conservative() -> None:
    scorer = InterpretableStaticUtilityScorer()
    unknown_costs = replace(
        _costs(),
        observations=tuple(
            CostObservation(
                item.name,
                item.source,
                item.available_at,
                None,
                MissingReason.UNKNOWN,
            )
            if item.name == LifecycleCostName.STORAGE_BYTES
            else item
            for item in _costs().observations
        ),
    )
    unknown = scorer.score(
        _features(), unknown_costs, target=UtilityTarget.GENERATION, cutoff=10
    )
    assert unknown.disposition == UtilityDisposition.DEFER
    assert unknown.reason_codes == ("unknown_lifecycle_cost",)

    no_history = scorer.score(
        _features(reuse_likelihood=None, no_history=True),
        _costs(),
        target=UtilityTarget.RETRIEVAL,
        cutoff=10,
    )
    assert no_history.disposition == UtilityDisposition.DEFER
    assert no_history.reason_codes == ("no_history",)

    low_confidence = scorer.score(
        _features(confidence=0.2),
        _costs(),
        target=UtilityTarget.INTERNAL_OPERATION,
        cutoff=10,
    )
    assert low_confidence.disposition == UtilityDisposition.DEFER
    assert low_confidence.reason_codes == ("low_confidence",)

    conflict = scorer.score(
        _features(conflict_risk=0.9),
        _costs(),
        target=UtilityTarget.GENERATION,
        cutoff=10,
    )
    assert conflict.disposition == UtilityDisposition.REJECT
    assert conflict.reason_codes == ("high_conflict_risk",)


def test_unknown_out_of_range_and_future_or_delayed_evidence_fail_closed() -> None:
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        _features(recency=1.1)
    with pytest.raises(ValueError, match="non-negative"):
        _costs(**{LifecycleCostName.STORAGE_BYTES: -1})
    with pytest.raises(ValueError, match="requires host evidence"):
        replace(
            _features().get(UtilityFeatureName.COMPLETION_STATUS),
            source=FeatureSource.MODEL_PREDICTED,
        )

    future = replace(
        _features(),
        observations=tuple(
            replace(item, available_at=11)
            if item.name == UtilityFeatureName.RECENCY
            else item
            for item in _features().observations
        ),
    )
    with pytest.raises(ValueError, match="future-dated"):
        InterpretableStaticUtilityScorer().score(
            future, _costs(), target=UtilityTarget.GENERATION, cutoff=10
        )

    delayed = replace(
        _features(),
        observations=tuple(
            replace(item, source=FeatureSource.DELAYED)
            if item.name == UtilityFeatureName.REUSE_LIKELIHOOD
            else item
            for item in _features().observations
        ),
    )
    with pytest.raises(ValueError, match="delayed evidence"):
        InterpretableStaticUtilityScorer().score(
            delayed, _costs(), target=UtilityTarget.RETRIEVAL, cutoff=10
        )


def test_incomplete_or_expired_source_is_rejected_before_utility_threshold() -> None:
    incomplete = StaticUtilityFeatureExtractor().extract(
        replace(
            _exit_evidence(),
            completion_status=CompletionStatus.IN_PROGRESS,
            unresolved_state="open_task",
        ),
        available_at=10,
        recency=1.0,
        reuse_likelihood=1.0,
        conflict_risk=0.0,
        recovery_risk=0.0,
        predicted_benefit=1.0,
        confidence=1.0,
    )
    decision = InterpretableStaticUtilityScorer().score(
        incomplete, _costs(), target=UtilityTarget.GENERATION, cutoff=10
    )
    assert decision.disposition == UtilityDisposition.REJECT
    assert decision.reason_codes == ("unsafe_incomplete_source",)

    expired = StaticUtilityFeatureExtractor().extract(
        replace(_exit_evidence(), temporal_validity=TemporalValidity.EXPIRED),
        available_at=10,
        recency=1.0,
        reuse_likelihood=1.0,
        conflict_risk=0.0,
        recovery_risk=0.0,
        predicted_benefit=1.0,
        confidence=1.0,
    )
    decision = InterpretableStaticUtilityScorer().score(
        expired, _costs(), target=UtilityTarget.GENERATION, cutoff=10
    )
    assert decision.disposition == UtilityDisposition.REJECT
    assert decision.reason_codes == ("non_durable_or_unknown_validity",)
