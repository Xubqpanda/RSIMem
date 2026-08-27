from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rsimem.adaptive_activation import (
    activate_adaptive_policy,
    load_matched_observation_batch,
)
from rsimem.adaptive_preparation import (
    load_offline_adaptive_preparation,
    prepare_adaptive_policy,
)
from rsimem.adaptive_validation_evidence import (
    assemble_matched_validation_observations,
)
from rsimem.adaptive_validation_runtime import (
    prepare_matched_validation_runtime,
    resolved_matched_validation_trial_profile,
)
from rsimem.experiment_manifest import (
    ADAPTIVE_VALIDATION_METHOD_VARIANTS,
    initialize_batch_manifest,
    record_attempt,
)
from rsimem.memory.operation_graph import (
    AppendOnlyOperationEvidenceLog,
    AtomicOperationRecorder,
    OperationGraph,
)
from rsimem.memory.utility import MEM0_UTILITY_PARAMETER_IDS, UtilityTarget
from test_adaptive_preparation import _prepared_feedback
from test_experiment_manifest import _manifest_kwargs
from test_feedback_dataset import _graph


def _runtime_graph(exposure: str, *, run_name: str, policy_version: str):
    graph = _graph(exposure, policy_version=policy_version)
    old_owner = "parameter.fact"
    owner = MEM0_UTILITY_PARAMETER_IDS[UtilityTarget.RETRIEVAL]
    return OperationGraph(
        tuple(
            replace(artifact, artifact_id=owner)
            if artifact.artifact_id == old_owner
            else artifact
            for artifact in graph.artifacts
        ),
        tuple(
            replace(
                operation,
                context=replace(
                    operation.context,
                    run_id=run_name,
                    policy_version=policy_version,
                ),
                input_artifact_ids=tuple(
                    owner if value == old_owner else value
                    for value in operation.input_artifact_ids
                ),
            )
            for operation in graph.operations
        ),
        graph.mutations,
    )


def _write_run(
    run_dir: Path,
    *,
    run_name: str,
    policy_version: str,
    exposure: str,
) -> None:
    graph = _runtime_graph(
        exposure,
        run_name=run_name,
        policy_version=policy_version,
    )
    sink = AppendOnlyOperationEvidenceLog()
    recorder = AtomicOperationRecorder(sink)
    for artifact in graph.artifacts:
        recorder.record_artifact(artifact)
    for operation in graph.operations:
        recorder.record_operation(operation)
    for mutation in graph.mutations:
        recorder.record_mutation(mutation)
    evidence = run_dir / "01_fixture" / "artifacts" / (
        "rsimem_semantic_operations.jsonl"
    )
    evidence.parent.mkdir(parents=True)
    evidence.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in sink.events),
        encoding="utf-8",
    )
    audit = {
        "ok": True,
        "issues": [],
        "runId": run_name,
        "privacy": {
            "absoluteSourcePaths": 0,
            "memoryTextLeaks": 0,
            "credentialPatternHits": {
                "aws": 0,
                "bearer": 0,
                "github": 0,
                "openai_style": 0,
            },
        },
        "uniquePhysicalUsage": {
            "inputTokens": 100,
            "outputTokens": 10,
            "cacheReadTokens": 0,
            "cacheWriteTokens": 0,
            "reasoningTokens": 0,
            "requests": 2,
            "retries": 0,
        },
    }
    (run_dir / "audit.json").write_text(
        json.dumps(audit, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _matched_batch(tmp_path: Path):
    feedback = _prepared_feedback(tmp_path)
    offline_root = tmp_path / "offline"
    prepare_adaptive_policy(feedback, output_root=offline_root)
    preparation = load_offline_adaptive_preparation(offline_root)
    trial_root = tmp_path / "trial"
    prepare_matched_validation_runtime(
        offline_root,
        output_root=trial_root,
    )
    profile = resolved_matched_validation_trial_profile(
        trial_root / "adaptive-config.json"
    )
    batch = tmp_path / "batch"
    manifest_path = batch / "batch_manifest.json"
    kwargs = _manifest_kwargs()
    kwargs["replicates"] = 1
    kwargs["budget"] = {
        **kwargs["budget"],
        "taskManifestDigest": "a" * 64,
    }
    initialize_batch_manifest(
        manifest_path,
        **kwargs,
        execution_modes=ADAPTIVE_VALIDATION_METHOD_VARIANTS,
        adaptive_policy=profile,
        semantic_feedback_contract="sm01_tsv_v1",
    )
    for ordinal, method in enumerate(ADAPTIVE_VALIDATION_METHOD_VARIANTS, 1):
        run_name = f"validation-r01-{method}"
        record_attempt(
            manifest_path,
            replicate=1,
            ordinal=ordinal,
            mode=method,
            run_name=run_name,
            status="running",
        )
        _write_run(
            batch / run_name,
            run_name=run_name,
            policy_version=(
                preparation.artifact.parent_policy_version
                if method == "static-rsimem"
                else preparation.artifact.policy_version
            ),
            exposure=(
                "injected_not_used"
                if method == "static-rsimem"
                else "used"
            ),
        )
        record_attempt(
            manifest_path,
            replicate=1,
            ordinal=ordinal,
            mode=method,
            run_name=run_name,
            status="completed",
        )
    return batch, offline_root, preparation


def test_assembler_derives_matched_observations_and_activation_input(
    tmp_path: Path,
) -> None:
    batch, offline_root, preparation = _matched_batch(tmp_path)
    output = tmp_path / "matched-observations.json"

    report = assemble_matched_validation_observations(
        batch,
        offline_root,
        output_path=output,
    )
    replay = assemble_matched_validation_observations(
        batch,
        offline_root,
        output_path=output,
    )

    assert replay == report
    assert report["officialEvaluation"] is False
    assert [item["variant"] for item in report["observations"]] == [
        "static",
        "proposal",
    ]
    assert [item["label"] for item in report["observations"]] == [
        "negative",
        "positive",
    ]
    assert all(source["auditOk"] for source in report["sourceRuns"])
    loaded = load_matched_observation_batch(output, preparation)
    assert len(loaded.observations) == 2
    activation = activate_adaptive_policy(
        offline_root,
        output,
        output_root=tmp_path / "active",
    )
    assert activation["resultingState"] == "active"


def test_assembler_rejects_incomplete_resource_accounting(tmp_path: Path) -> None:
    batch, offline_root, _ = _matched_batch(tmp_path)
    audit_path = batch / "validation-r01-static-rsimem" / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    del audit["uniquePhysicalUsage"]["reasoningTokens"]
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(ValueError, match="resource usage is incomplete"):
        assemble_matched_validation_observations(
            batch,
            offline_root,
            output_path=tmp_path / "invalid.json",
        )
