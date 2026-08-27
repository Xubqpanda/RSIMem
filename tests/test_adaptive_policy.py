from __future__ import annotations

import inspect
import json
from dataclasses import replace

import pytest

from rsimem.memory.adaptive_policy import (
    ADAPTIVE_POLICY_FEATURES,
    ADAPTIVE_POLICY_OBJECTIVE,
    AdaptiveFallbackReason,
    AdaptiveParameterName,
    AdaptiveParameterSpec,
    AdaptivePolicyState,
    AdaptiveTrainingConfig,
    DeterministicAdaptivePolicyLearner,
)
from rsimem.memory.adaptation_contract import (
    AdaptiveArtifactKind,
    require_extraction_prompt_artifact,
)
from rsimem.memory.attribution import DeterministicFirstAttributor
from rsimem.memory.feedback_dataset import evaluate_feedback_dataset_stage_gate
from test_feedback_dataset import _dataset, _graph


def _accepted(exposure: str, *, attributed: bool = False):
    graph = _graph(exposure)
    reports = (
        (DeterministicFirstAttributor().attribute(graph),)
        if attributed
        else ()
    )
    dataset = _dataset(graph, reports=reports)
    gate = evaluate_feedback_dataset_stage_gate(
        dataset,
        graph,
        expected_config=dataset.config,
        attribution_reports=reports,
    )
    assert gate.ok
    assert dataset.examples[0].policy_parameter_ids == ("parameter.fact",)
    return graph, dataset, gate


def _config(dataset, **changes) -> AdaptiveTrainingConfig:
    values = {
        "parent_policy_version": dataset.config.policy_version,
        "seed": 17,
        "parameters": (AdaptiveParameterSpec(
            parameter_id="parameter.fact",
            name=AdaptiveParameterName.RETRIEVAL_ACCEPT_THRESHOLD,
            baseline_value=0.35,
            prompt_ref="mem0-flat.retrieval",
        ),),
        "minimum_resolved_examples": 1,
        "maximum_missing_propensity_rate": 1.0,
    }
    values.update(changes)
    return AdaptiveTrainingConfig(**values)


def test_positive_feedback_produces_deterministic_bounded_artifact() -> None:
    _, dataset, gate = _accepted("used")
    config = _config(dataset)
    learner = DeterministicAdaptivePolicyLearner()
    first = learner.learn(dataset, gate, config)
    replay = learner.learn(dataset, gate, config)

    assert first == replay
    assert first.payload() == replay.payload()
    assert first.state == AdaptivePolicyState.PROPOSAL
    assert first.parent_policy_version == dataset.config.policy_version
    assert first.dataset_id == dataset.dataset_id
    assert first.dataset_payload_digest == gate.dataset_payload_digest
    assert first.dataset_version == dataset.config.dataset_version
    assert first.feature_schema == dataset.config.feature_schema
    assert first.label_schema == dataset.config.label_schema
    assert first.training_seed == config.seed
    assert first.objective == ADAPTIVE_POLICY_OBJECTIVE
    assert first.artifact_kind == AdaptiveArtifactKind.LEGACY_THRESHOLD_EXPERIMENT
    assert "raw_resource_usage" not in ADAPTIVE_POLICY_FEATURES
    assert first.regularization == config.l2_regularization
    assert first.training_config_digest == config.digest
    assert first.prompt_refs == ("mem0-flat.retrieval",)
    assert first.training_example_ids == (dataset.examples[0].example_id,)
    update = first.parameters[0]
    assert update.positive_count == 1
    assert update.negative_count == 0
    assert update.fallback_reason == AdaptiveFallbackReason.NONE
    assert update.proposed_value < update.baseline_value
    assert update.attributed_example_ids == (dataset.examples[0].example_id,)
    assert update.failure_subgraph_operation_ids == ()
    assert learner.audit(first, dataset, gate, config).ok

    different_seed = learner.learn(dataset, gate, replace(config, seed=18))
    assert different_seed.artifact_id != first.artifact_id


def test_legacy_threshold_contract_cannot_bind_extraction_runtime() -> None:
    _, dataset, gate = _accepted("used")
    config = _config(dataset)
    artifact = DeterministicAdaptivePolicyLearner().learn(dataset, gate, config)

    for payload in (config.payload(), artifact.payload()):
        serialized = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "raw_resource_usage",
            "cost_weight",
            "maximum_cost_ratio",
            "lifecycleCostUnits",
        ):
            assert forbidden not in serialized
    with pytest.raises(ValueError, match="extraction prompt artifact"):
        require_extraction_prompt_artifact(artifact)
    require_extraction_prompt_artifact({
        "artifact_kind": AdaptiveArtifactKind.EXTRACTION_PROMPT.value,
    })


def test_negative_feedback_only_uses_matching_failure_subgraph() -> None:
    _, dataset, gate = _accepted("injected_not_used", attributed=True)
    learner = DeterministicAdaptivePolicyLearner()
    retrieval = learner.learn(dataset, gate, _config(dataset)).parameters[0]

    assert retrieval.positive_count == 0
    assert retrieval.negative_count == 1
    assert retrieval.proposed_value > retrieval.baseline_value
    assert retrieval.failure_subgraph_operation_ids == (
        "op.retrieval",
        "op.use",
    )

    extraction_config = _config(
        dataset,
        parameters=(AdaptiveParameterSpec(
            parameter_id="parameter.fact",
            name=AdaptiveParameterName.EXTRACTION_ACCEPT_THRESHOLD,
            baseline_value=0.35,
            prompt_ref="mem0-flat.fact-extraction",
        ),),
    )
    extraction = learner.learn(dataset, gate, extraction_config).parameters[0]
    assert extraction.negative_count == 0
    assert extraction.delta == 0.0
    assert extraction.fallback_reason == AdaptiveFallbackReason.MISSING_FEATURE


def test_training_contract_rejects_scope_gate_and_content_tampering() -> None:
    _, dataset, gate = _accepted("used")
    learner = DeterministicAdaptivePolicyLearner()
    config = _config(dataset)
    artifact = learner.learn(dataset, gate, config)

    with pytest.raises(ValueError, match="training identity"):
        replace(config, feature_schema="feature-drift-v1")
    with pytest.raises(ValueError):
        AdaptiveParameterSpec(
            parameter_id="parameter.fact",
            name="route_selector",
            baseline_value=0.5,
            prompt_ref="route.selector",
        )
    with pytest.raises(ValueError, match="dataset evidence mismatch"):
        learner.learn(
            dataset,
            replace(gate, dataset_payload_digest="0" * 64),
            config,
        )
    with pytest.raises(ValueError, match="training membership"):
        learner.learn(
            dataset,
            gate,
            replace(config, training_example_ids=("feedback-example.unknown",)),
        )
    with pytest.raises(ValueError, match="parent policy is unknown"):
        learner.learn(
            dataset,
            gate,
            replace(config, parent_policy_version="unknown-parent-v1"),
        )
    unknown_parameter = replace(
        config,
        parameters=(AdaptiveParameterSpec(
            parameter_id="parameter.unknown",
            name=AdaptiveParameterName.RETRIEVAL_ACCEPT_THRESHOLD,
            baseline_value=0.35,
            prompt_ref="mem0-flat.retrieval",
        ),),
    )
    with pytest.raises(ValueError, match="unknown policy parameter"):
        learner.learn(dataset, gate, unknown_parameter)
    with pytest.raises(ValueError, match="content digest mismatch"):
        replace(artifact, regularization=2.0)
    with pytest.raises(ValueError, match="provenance"):
        replace(artifact.parameters[0], attributed_example_ids=())


def test_low_sample_missing_feature_and_distribution_shift_are_conservative() -> None:
    _, dataset, gate = _accepted("injected_not_used", attributed=True)
    learner = DeterministicAdaptivePolicyLearner()

    low_sample = learner.learn(
        dataset,
        gate,
        _config(dataset, minimum_resolved_examples=2),
    ).parameters[0]
    assert low_sample.fallback_reason == AdaptiveFallbackReason.LOW_SAMPLE
    assert low_sample.delta == 0.0

    missing = learner.learn(
        dataset,
        gate,
        _config(
            dataset,
            parameters=(AdaptiveParameterSpec(
                parameter_id="parameter.fact",
                name=AdaptiveParameterName.EXTRACTION_ACCEPT_THRESHOLD,
                baseline_value=0.35,
                prompt_ref="mem0-flat.fact-extraction",
            ),),
        ),
    ).parameters[0]
    assert missing.fallback_reason == AdaptiveFallbackReason.MISSING_FEATURE
    assert missing.delta == 0.0

    _, unexposed, unexposed_gate = _accepted("not_retrieved")
    shifted = learner.learn(
        unexposed,
        unexposed_gate,
        _config(unexposed, maximum_missing_propensity_rate=0.0),
    ).parameters[0]
    assert shifted.fallback_reason == AdaptiveFallbackReason.DISTRIBUTION_SHIFT
    assert shifted.delta == 0.0


def test_learner_has_no_hidden_score_or_schedule_input_surface() -> None:
    parameters = set(inspect.signature(
        DeterministicAdaptivePolicyLearner.learn
    ).parameters)
    assert parameters == {"self", "dataset", "gate", "config"}
    _, dataset, _ = _accepted("used")
    config = _config(dataset)
    assert config.feature_names == ADAPTIVE_POLICY_FEATURES
    serialized = json.dumps(config.payload(), sort_keys=True)
    for forbidden in (
        '"score"',
        '"grader"',
        '"expectation"',
        '"answer"',
        '"route_selector"',
        '"invocation_schedule"',
    ):
        assert forbidden not in serialized
