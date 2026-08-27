from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rsimem.experiment_manifest import (
    FEEDBACK_METHOD_VARIANTS,
    initialize_batch_manifest,
    record_attempt,
)
from rsimem.feedback_preparation import assemble_feedback_batch
from rsimem.memory.feedback_dataset import JsonDelayedFeedbackDatasetStore
from rsimem.memory.operation_graph import (
    AppendOnlyOperationEvidenceLog,
    AtomicOperationRecorder,
    OperationGraph,
)
from rsimem.memory.utility import MEM0_UTILITY_PARAMETER_IDS, UtilityTarget
from rsimem.memory_systems.mem0_flat import FrozenMem0UtilityGate
from rsimem.memory_systems.mem0_flat.policy import Mem0FlatSemanticPolicy
from test_experiment_manifest import _manifest_kwargs
from test_feedback_dataset import _graph


def _batch(tmp_path: Path, exposure: str = "used") -> Path:
    root = tmp_path / "batch"
    manifest = root / "batch_manifest.json"
    kwargs = _manifest_kwargs()
    kwargs["replicates"] = 1
    initialize_batch_manifest(
        manifest,
        **kwargs,
        execution_modes=FEEDBACK_METHOD_VARIANTS,
        semantic_feedback_contract="sm01_tsv_v1",
    )
    run_name = "feedback-r01"
    record_attempt(
        manifest,
        replicate=1,
        ordinal=1,
        mode=FEEDBACK_METHOD_VARIANTS[0],
        run_name=run_name,
        status="running",
    )
    record_attempt(
        manifest,
        replicate=1,
        ordinal=1,
        mode=FEEDBACK_METHOD_VARIANTS[0],
        run_name=run_name,
        status="completed",
    )
    base_policy = Mem0FlatSemanticPolicy(
        object(),
        utility_gate=FrozenMem0UtilityGate(),
    ).descriptor.policy_version
    graph = _graph(exposure, policy_version=base_policy)
    source_parameter = "parameter.fact"
    retrieval_parameter = MEM0_UTILITY_PARAMETER_IDS[UtilityTarget.RETRIEVAL]
    graph = OperationGraph(
        tuple(
            replace(artifact, artifact_id=retrieval_parameter)
            if artifact.artifact_id == source_parameter
            else artifact
            for artifact in graph.artifacts
        ),
        tuple(
            replace(
                operation,
                input_artifact_ids=tuple(
                    retrieval_parameter if value == source_parameter else value
                    for value in operation.input_artifact_ids
                ),
            )
            for operation in graph.operations
        ),
        graph.mutations,
    )
    sink = AppendOnlyOperationEvidenceLog()
    recorder = AtomicOperationRecorder(sink)
    for artifact in graph.artifacts:
        recorder.record_artifact(artifact)
    for operation in graph.operations:
        recorder.record_operation(operation)
    for mutation in graph.mutations:
        recorder.record_mutation(mutation)
    path = root / run_name / "01_fixture" / "artifacts" / (
        "rsimem_semantic_operations.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in sink.events),
        encoding="utf-8",
    )
    return root


def test_feedback_preparation_assembles_content_free_replayable_dataset(
    tmp_path: Path,
) -> None:
    batch = _batch(tmp_path)
    output = tmp_path / "prepared"

    report = assemble_feedback_batch(batch, output_root=output)

    assert report["attemptCount"] == 1
    assert report["resolvedExampleCount"] == 1
    assert report["labelCounts"]["positive"] == 1
    assert report["stageGate"] == {
        "ok": True,
        "issues": [],
        "replayDatasetId": report["datasetId"],
    }
    dataset = JsonDelayedFeedbackDatasetStore(output / "datasets").get(
        report["datasetId"]
    )
    assert dataset is not None
    assert [example.example_id for example in dataset.examples] == report["exampleIds"]
    serialized = (output / "preparation_manifest.json").read_text(encoding="utf-8")
    for forbidden in ('"content"', '"prompt"', '"response"', '"score"', '"grader"'):
        assert forbidden not in serialized


def test_feedback_preparation_rejects_unresolved_or_incomplete_batches(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="no resolved example"):
        assemble_feedback_batch(
            _batch(tmp_path / "unresolved", "censored"),
            output_root=tmp_path / "unresolved-output",
        )

    batch = _batch(tmp_path / "incomplete")
    manifest_path = batch / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["attempts"][-1]["status"] = "failed"
    manifest["attempts"][-1]["failureStage"] = "provider"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete or ambiguous"):
        assemble_feedback_batch(
            batch,
            output_root=tmp_path / "incomplete-output",
        )
