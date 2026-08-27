from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from rsimem.adaptive_preparation import prepare_adaptive_policy
from rsimem.feedback_preparation import (
    FEEDBACK_PREPARATION_SCHEMA_VERSION,
    _digest,
    _stage_gate_payload,
)
from rsimem.memory.adaptive_policy import AdaptivePolicyState
from rsimem.memory.adaptive_policy_store import JsonAdaptivePolicyStore
from rsimem.memory.feedback_dataset import (
    FeedbackLabel,
    JsonDelayedFeedbackDatasetStore,
    build_feedback_dataset_report,
)
from rsimem.memory.operation_graph import OperationKind
from rsimem.memory.utility import MEM0_UTILITY_PARAMETER_IDS, UtilityTarget
from rsimem.memory_systems.mem0_flat import FrozenMem0UtilityGate
from rsimem.memory_systems.mem0_flat.policy import Mem0FlatSemanticPolicy
from test_adaptive_policy_validation import (
    _multi_dataset,
    _payload_digest,
    _rebind_example,
    _stable_id,
)


def _prepared_feedback(tmp_path: Path) -> Path:
    parent_policy = Mem0FlatSemanticPolicy(
        object(),
        utility_gate=FrozenMem0UtilityGate(),
    ).descriptor.policy_version
    source, source_gate = _multi_dataset(
        training_negative=True,
        policy_version=parent_policy,
    )
    owner = MEM0_UTILITY_PARAMETER_IDS[UtilityTarget.RETRIEVAL]
    examples = []
    for ordinal, original in enumerate((
        source.examples[0],
        source.examples[1],
        source.examples[1],
    ), start=1):
        example = _rebind_example(original, ordinal)
        example = replace(
            example,
            example_id="feedback-example.placeholder",
            policy_parameter_ids=(owner,),
        )
        payload = example.payload()
        payload.pop("example_id")
        examples.append(replace(
            example,
            example_id=_stable_id("feedback-example", payload),
        ))
    dataset = replace(
        source,
        dataset_id="feedback-dataset.placeholder",
        examples=tuple(examples),
    )
    payload = dataset.payload()
    payload.pop("dataset_id")
    dataset = replace(
        dataset,
        dataset_id=_stable_id("feedback-dataset", payload),
    )
    counts = Counter(example.label for example in dataset.examples)
    gate = replace(
        source_gate,
        dataset_id=dataset.dataset_id,
        dataset_payload_digest=_payload_digest(dataset.payload()),
        replay_dataset_id=dataset.dataset_id,
        audit=replace(
            source_gate.audit,
            example_count=len(dataset.examples),
            label_counts=tuple(
                (label, counts[label]) for label in FeedbackLabel
            ),
        ),
        report=build_feedback_dataset_report(dataset),
    )
    prepared = tmp_path / "prepared-feedback"
    dataset_path, _ = JsonDelayedFeedbackDatasetStore(
        prepared / "datasets"
    ).put(dataset)
    operation_counts = {kind.value: 0 for kind in OperationKind}
    operation_counts[OperationKind.FUTURE_QUERY.value] = (
        dataset.source_operation_count
    )
    identity = {
        "schemaVersion": FEEDBACK_PREPARATION_SCHEMA_VERSION,
        "sourceExperimentId": "experiment.feedback-fixture",
        "sourceRevisions": {
            "rsimemCommit": "1" * 40,
            "pastBenchCommit": "2" * 40,
        },
        "feedbackContract": "sm01_tsv_v1",
        "parentPolicyVersion": parent_policy,
        "operationGraphDigest": "3" * 64,
        "datasetId": dataset.dataset_id,
        "datasetPayloadDigest": gate.dataset_payload_digest,
        "datasetConfigDigest": dataset.config.digest,
        "observationCutoffOperationId": dataset.window.cutoff_operation_id,
        "sourceLogs": [{
            "replicate": 1,
            "runName": "feedback-fixture-r01",
            "relativePath": "feedback-fixture-r01/operations.jsonl",
            "sha256": "4" * 64,
            "eventCount": 1,
        }],
        "exampleIds": [example.example_id for example in dataset.examples],
        "runtimeOwnedParameterIds": [owner],
        "stageGate": _stage_gate_payload(gate),
    }
    report = {
        **identity,
        "preparationId": f"feedback-preparation.{_digest(identity)[:40]}",
        "attemptCount": 1,
        "operationCount": dataset.source_operation_count,
        "artifactCount": 1,
        "mutationCount": 1,
        "operationKindCounts": operation_counts,
        "labelCounts": {
            label.value: counts[label] for label in FeedbackLabel
        },
        "resolvedExampleCount": (
            counts[FeedbackLabel.POSITIVE] + counts[FeedbackLabel.NEGATIVE]
        ),
        "censoredExampleCount": counts[FeedbackLabel.CENSORED],
        "datasetPath": dataset_path.relative_to(prepared).as_posix(),
    }
    (prepared / "preparation_manifest.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return prepared


def test_adaptive_preparation_persists_validated_retrieval_only_policy(
    tmp_path: Path,
) -> None:
    prepared = _prepared_feedback(tmp_path)
    output = tmp_path / "adaptive"

    report = prepare_adaptive_policy(prepared, output_root=output)
    replay = prepare_adaptive_policy(prepared, output_root=output)

    assert replay == report
    assert report["offlineValidationAccepted"] is True
    assert report["resultingState"] == AdaptivePolicyState.VALIDATED.value
    assert report["activePolicyVersion"] is None
    assert report["trainingExampleCount"] == 2
    assert report["validationExampleCount"] == 1
    assert report["runtimeOwnedParameterIds"] == [
        MEM0_UTILITY_PARAMETER_IDS[UtilityTarget.RETRIEVAL]
    ]
    assert [value["name"] for value in report["parameterUpdates"]] == [
        "retrieval_accept_threshold"
    ]
    store = JsonAdaptivePolicyStore(
        output / "adaptive-policy-store.json",
        trusted_root_policy_versions=(report["parentPolicyVersion"],),
    )
    snapshot = store.snapshot()
    assert snapshot.active is None
    assert snapshot.active_policy_version is None
    assert len(snapshot.artifacts) == 1
    assert len(snapshot.records) == 1
    assert snapshot.records[0].state == AdaptivePolicyState.VALIDATED
    assert snapshot.artifacts[0].parameters[0].parameter_id == (
        MEM0_UTILITY_PARAMETER_IDS[UtilityTarget.RETRIEVAL]
    )
    serialized = json.dumps(report, sort_keys=True)
    for forbidden in ('"score"', '"grader"', '"answer"', '"expectation"'):
        assert forbidden not in serialized
    assert not (output / "adaptive-config.json").exists()


def test_adaptive_preparation_rejects_active_or_ambiguous_output(
    tmp_path: Path,
) -> None:
    prepared = _prepared_feedback(tmp_path)
    output = tmp_path / "adaptive"
    report = prepare_adaptive_policy(prepared, output_root=output)
    store = JsonAdaptivePolicyStore(
        output / "adaptive-policy-store.json",
        trusted_root_policy_versions=(report["parentPolicyVersion"],),
    )
    store.transition(
        report["policyVersion"],
        to_state=AdaptivePolicyState.ACTIVE,
        transition_id="policy-transition.test-active",
        reason_code="test_activation",
    )
    with pytest.raises(ValueError, match="cannot modify an ACTIVE"):
        prepare_adaptive_policy(prepared, output_root=output)
