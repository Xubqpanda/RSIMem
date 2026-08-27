"""Assemble audited delayed feedback from predeclared live preparation runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .experiment_manifest import FEEDBACK_METHOD_VARIANTS, load_manifest
from .memory.attribution import DeterministicFirstAttributor
from .memory.feedback_dataset import (
    DelayedFeedbackConfig,
    DelayedFeedbackDatasetBuilder,
    FeedbackLabel,
    FeedbackDatasetStageGate,
    FeedbackObservationWindow,
    JsonDelayedFeedbackDatasetStore,
    evaluate_feedback_dataset_stage_gate,
)
from .memory.operation_graph import (
    AppendOnlyOperationEvidenceLog,
    OperationKind,
    materialize_operation_graph,
)
from .memory.utility import (
    MEM0_UTILITY_PARAMETER_IDS,
    STATIC_UTILITY_FEATURE_SCHEMA,
    UtilityTarget,
)
from .memory_systems.mem0_flat import FrozenMem0UtilityGate
from .memory_systems.mem0_flat.policy import Mem0FlatSemanticPolicy


FEEDBACK_PREPARATION_SCHEMA_VERSION = 2
_REQUIRED_FUTURE_KINDS = {
    OperationKind.FUTURE_QUERY,
    OperationKind.RETRIEVAL,
    OperationKind.INJECTION,
    OperationKind.USE,
    OperationKind.DOWNSTREAM_OUTCOME,
}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _stage_gate_payload(gate: FeedbackDatasetStageGate) -> dict[str, Any]:
    return {
        "ok": gate.ok,
        "issues": list(gate.issues),
        "datasetId": gate.dataset_id,
        "datasetPayloadDigest": gate.dataset_payload_digest,
        "replayDatasetId": gate.replay_dataset_id,
        "expectedConfigDigest": gate.expected_config_digest,
        "actualConfigDigest": gate.actual_config_digest,
        "audit": {
            "ok": gate.audit.ok,
            "issues": list(gate.audit.issues),
            "exampleCount": gate.audit.example_count,
            "labelCounts": {
                label.value: count for label, count in gate.audit.label_counts
            },
        },
        "report": {
            "observationCount": gate.report.observation_count,
            "opportunityCount": gate.report.opportunity_count,
            "candidateCount": gate.report.candidate_count,
            "filteredCount": gate.report.filtered_count,
            "missingPropensityCount": gate.report.missing_propensity_count,
            "censoredCount": gate.report.censored_count,
            "censoringRate": gate.report.censoring_rate,
            "labelCounts": {
                label.value: count for label, count in gate.report.label_counts
            },
            "exposureCounts": {
                exposure.value: count
                for exposure, count in gate.report.exposure_counts
            },
        },
    }


def _completed_feedback_attempts(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    if tuple(manifest["configuration"]["executionModes"]) != FEEDBACK_METHOD_VARIANTS:
        raise ValueError("feedback preparation requires the frozen method set")
    if manifest["configuration"]["semanticFeedbackContract"] != "sm01_tsv_v1":
        raise ValueError("feedback preparation requires the frozen signal contract")
    selected = []
    for replicate in range(1, manifest["replicates"] + 1):
        completed = [
            attempt
            for attempt in manifest["attempts"]
            if attempt["replicate"] == replicate
            and attempt["mode"] == FEEDBACK_METHOD_VARIANTS[0]
            and attempt["status"] == "completed"
        ]
        if len(completed) != 1:
            raise ValueError("feedback preparation replicate is incomplete or ambiguous")
        selected.append(completed[0])
    return tuple(selected)


def assemble_feedback_batch(
    batch_root: Path,
    *,
    output_root: Path,
) -> dict[str, Any]:
    batch_root = batch_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    manifest = load_manifest(batch_root / "batch_manifest.json")
    attempts = _completed_feedback_attempts(manifest)
    merged = AppendOnlyOperationEvidenceLog()
    source_logs = []
    for attempt in attempts:
        run_dir = (batch_root / attempt["outputDirectory"]).resolve()
        if not run_dir.is_relative_to(batch_root):
            raise ValueError("feedback attempt directory escapes its batch")
        paths = tuple(sorted(
            run_dir.glob("[0-9][0-9]_*/*/rsimem_semantic_operations.jsonl")
        ))
        if not paths:
            raise ValueError("feedback attempt has no semantic operation evidence")
        for path in paths:
            serialized = path.read_bytes()
            log = AppendOnlyOperationEvidenceLog(path)
            for event in log.events:
                merged.append(event)
            source_logs.append({
                "replicate": attempt["replicate"],
                "runName": attempt["runName"],
                "relativePath": path.relative_to(batch_root).as_posix(),
                "sha256": hashlib.sha256(serialized).hexdigest(),
                "eventCount": len(log.events),
            })

    graph = materialize_operation_graph(merged.events)
    policy_versions = sorted({
        operation.context.policy_version for operation in graph.operations
    })
    expected_parent = Mem0FlatSemanticPolicy(
        object(),
        utility_gate=FrozenMem0UtilityGate(),
    ).descriptor.policy_version
    if policy_versions != [expected_parent]:
        raise ValueError("feedback evidence does not use the frozen parent policy")
    kind_counts = Counter(operation.kind for operation in graph.operations)
    if any(kind_counts[kind] < 1 for kind in _REQUIRED_FUTURE_KINDS):
        raise ValueError("feedback evidence is missing future lifecycle operations")

    attribution = DeterministicFirstAttributor().attribute(graph)
    config = DelayedFeedbackConfig(
        expected_parent,
        STATIC_UTILITY_FEATURE_SCHEMA,
    )
    window = FeedbackObservationWindow.create(graph, complete=True)
    dataset = DelayedFeedbackDatasetBuilder(config).build(
        graph,
        window,
        attribution_reports=(attribution,),
    )
    gate = evaluate_feedback_dataset_stage_gate(
        dataset,
        graph,
        expected_config=config,
        attribution_reports=(attribution,),
    )
    if not gate.ok:
        raise ValueError(
            "feedback dataset gate failed: " + ",".join(gate.issues)
        )
    labels = Counter(example.label for example in dataset.examples)
    resolved_count = labels[FeedbackLabel.POSITIVE] + labels[FeedbackLabel.NEGATIVE]
    if resolved_count < 1:
        raise ValueError("feedback preparation produced no resolved example")
    retrieval_parameter = MEM0_UTILITY_PARAMETER_IDS[UtilityTarget.RETRIEVAL]
    if any(
        retrieval_parameter not in example.policy_parameter_ids
        for example in dataset.examples
    ):
        raise ValueError("feedback evidence lacks trusted retrieval ownership")

    dataset_path, _ = JsonDelayedFeedbackDatasetStore(
        output_root / "datasets"
    ).put(dataset)
    identity = {
        "schemaVersion": FEEDBACK_PREPARATION_SCHEMA_VERSION,
        "sourceExperimentId": manifest["experimentId"],
        "sourceRevisions": manifest["revisions"],
        "feedbackContract": manifest["configuration"]["semanticFeedbackContract"],
        "parentPolicyVersion": expected_parent,
        "operationGraphDigest": _digest(merged.events),
        "datasetId": dataset.dataset_id,
        "datasetPayloadDigest": gate.dataset_payload_digest,
        "datasetConfigDigest": dataset.config.digest,
        "observationCutoffOperationId": dataset.window.cutoff_operation_id,
        "sourceLogs": source_logs,
        "exampleIds": [example.example_id for example in dataset.examples],
        "runtimeOwnedParameterIds": [retrieval_parameter],
        "stageGate": _stage_gate_payload(gate),
    }
    report = {
        **identity,
        "preparationId": f"feedback-preparation.{_digest(identity)[:40]}",
        "attemptCount": len(attempts),
        "operationCount": len(graph.operations),
        "artifactCount": len(graph.artifacts),
        "mutationCount": len(graph.mutations),
        "operationKindCounts": {
            kind.value: kind_counts[kind] for kind in OperationKind
        },
        "labelCounts": {
            label.value: labels[label] for label in FeedbackLabel
        },
        "resolvedExampleCount": resolved_count,
        "censoredExampleCount": labels[FeedbackLabel.CENSORED],
        "datasetPath": dataset_path.relative_to(output_root).as_posix(),
    }
    _write_json(output_root / "preparation_manifest.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = assemble_feedback_batch(args.batch_root, output_root=args.output)
    print(json.dumps({
        "preparationId": report["preparationId"],
        "datasetId": report["datasetId"],
        "resolvedExampleCount": report["resolvedExampleCount"],
    }, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
