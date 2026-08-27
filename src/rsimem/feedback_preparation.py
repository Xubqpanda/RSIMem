"""Assemble audited delayed feedback from predeclared live preparation runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .experiment_manifest import FEEDBACK_METHOD_VARIANTS, load_manifest
from .memory.attribution import DeterministicFirstAttributor
from .memory.feedback_dataset import (
    DelayedFeedbackConfig,
    DelayedFeedbackDataset,
    DelayedFeedbackDatasetBuilder,
    FeedbackDatasetAudit,
    FeedbackLabel,
    FeedbackDatasetStageGate,
    FeedbackObservationWindow,
    JsonDelayedFeedbackDatasetStore,
    build_feedback_dataset_report,
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
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_PREPARATION_IDENTITY_FIELDS = (
    "schemaVersion",
    "sourceExperimentId",
    "sourceRevisions",
    "feedbackContract",
    "parentPolicyVersion",
    "operationGraphDigest",
    "datasetId",
    "datasetPayloadDigest",
    "datasetConfigDigest",
    "observationCutoffOperationId",
    "sourceLogs",
    "exampleIds",
    "runtimeOwnedParameterIds",
    "stageGate",
)
_PREPARATION_SUMMARY_FIELDS = (
    "preparationId",
    "attemptCount",
    "operationCount",
    "artifactCount",
    "mutationCount",
    "operationKindCounts",
    "labelCounts",
    "resolvedExampleCount",
    "censoredExampleCount",
    "datasetPath",
)
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


def _strict_mapping(
    value: object,
    fields: set[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"malformed {name}")
    return value


def load_prepared_feedback_dataset(
    prepared_root: Path,
) -> tuple[DelayedFeedbackDataset, FeedbackDatasetStageGate, dict[str, Any]]:
    """Load a content-addressed dataset from an accepted preparation manifest."""

    prepared_root = prepared_root.expanduser().resolve()
    manifest_path = prepared_root / "preparation_manifest.json"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("feedback preparation manifest cannot be read") from exc
    expected_fields = set((*_PREPARATION_IDENTITY_FIELDS, *_PREPARATION_SUMMARY_FIELDS))
    manifest = dict(_strict_mapping(raw, expected_fields, "feedback preparation manifest"))
    if manifest["schemaVersion"] != FEEDBACK_PREPARATION_SCHEMA_VERSION:
        raise ValueError("unsupported feedback preparation schema")
    dataset_id = manifest["datasetId"]
    dataset_path_value = manifest["datasetPath"]
    if (
        not isinstance(dataset_id, str)
        or not isinstance(dataset_path_value, str)
        or dataset_path_value != f"datasets/{dataset_id}.json"
    ):
        raise ValueError("feedback preparation dataset path is invalid")
    dataset_path = (prepared_root / dataset_path_value).resolve()
    if not dataset_path.is_relative_to(prepared_root):
        raise ValueError("feedback preparation dataset path escapes its root")
    try:
        serialized = dataset_path.read_text(encoding="utf-8")
        dataset = DelayedFeedbackDataset.from_payload(json.loads(serialized))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("prepared feedback dataset is malformed") from exc
    if serialized != _canonical(dataset.payload()) + "\n":
        raise ValueError("prepared feedback dataset is not canonical")
    dataset_digest = _digest(dataset.payload())
    if (
        dataset.dataset_id != dataset_id
        or manifest["datasetPayloadDigest"] != dataset_digest
        or manifest["datasetConfigDigest"] != dataset.config.digest
        or manifest["parentPolicyVersion"] != dataset.config.policy_version
        or manifest["observationCutoffOperationId"]
        != dataset.window.cutoff_operation_id
        or manifest["exampleIds"]
        != [example.example_id for example in dataset.examples]
    ):
        raise ValueError("feedback preparation dataset identity mismatch")

    retrieval_parameter = MEM0_UTILITY_PARAMETER_IDS[UtilityTarget.RETRIEVAL]
    if manifest["runtimeOwnedParameterIds"] != [retrieval_parameter] or any(
        retrieval_parameter not in example.policy_parameter_ids
        for example in dataset.examples
    ):
        raise ValueError("feedback preparation runtime owner mismatch")
    if manifest["feedbackContract"] != "sm01_tsv_v1":
        raise ValueError("feedback preparation signal contract is not frozen")

    label_counts = Counter(example.label for example in dataset.examples)
    audit = FeedbackDatasetAudit(
        ok=True,
        issues=(),
        example_count=len(dataset.examples),
        label_counts=tuple(
            (label, label_counts[label]) for label in FeedbackLabel
        ),
    )
    gate = FeedbackDatasetStageGate(
        ok=True,
        issues=(),
        dataset_id=dataset.dataset_id,
        dataset_payload_digest=dataset_digest,
        replay_dataset_id=dataset.dataset_id,
        expected_config_digest=dataset.config.digest,
        actual_config_digest=dataset.config.digest,
        audit=audit,
        report=build_feedback_dataset_report(dataset),
    )
    if manifest["stageGate"] != _stage_gate_payload(gate):
        raise ValueError("feedback preparation stage gate mismatch")
    expected_labels = {
        label.value: label_counts[label] for label in FeedbackLabel
    }
    if (
        manifest["labelCounts"] != expected_labels
        or manifest["resolvedExampleCount"]
        != label_counts[FeedbackLabel.POSITIVE] + label_counts[FeedbackLabel.NEGATIVE]
        or manifest["censoredExampleCount"] != label_counts[FeedbackLabel.CENSORED]
        or manifest["operationCount"] != dataset.source_operation_count
    ):
        raise ValueError("feedback preparation summary mismatch")
    operation_counts = _strict_mapping(
        manifest["operationKindCounts"],
        {kind.value for kind in OperationKind},
        "feedback operation counts",
    )
    if any(type(value) is not int or value < 0 for value in operation_counts.values()) or (
        sum(operation_counts.values()) != manifest["operationCount"]
    ):
        raise ValueError("feedback preparation operation counts are invalid")
    if any(
        type(manifest[field]) is not int or manifest[field] < 0
        for field in ("attemptCount", "artifactCount", "mutationCount")
    ) or manifest["attemptCount"] < 1:
        raise ValueError("feedback preparation counts are invalid")
    for field in ("operationGraphDigest", "datasetPayloadDigest"):
        if not isinstance(manifest[field], str) or _DIGEST.fullmatch(manifest[field]) is None:
            raise ValueError("feedback preparation digest is invalid")
    if not isinstance(manifest["sourceRevisions"], Mapping):
        raise ValueError("feedback preparation revisions are invalid")
    source_logs = manifest["sourceLogs"]
    if not isinstance(source_logs, list) or not source_logs:
        raise ValueError("feedback preparation source logs are invalid")
    for value in source_logs:
        source = _strict_mapping(
            value,
            {"replicate", "runName", "relativePath", "sha256", "eventCount"},
            "feedback preparation source log",
        )
        relative = Path(str(source["relativePath"]))
        if (
            type(source["replicate"]) is not int
            or source["replicate"] < 1
            or type(source["eventCount"]) is not int
            or source["eventCount"] < 1
            or relative.is_absolute()
            or ".." in relative.parts
            or not isinstance(source["sha256"], str)
            or _DIGEST.fullmatch(source["sha256"]) is None
        ):
            raise ValueError("feedback preparation source log is invalid")

    identity = {field: manifest[field] for field in _PREPARATION_IDENTITY_FIELDS}
    expected_id = f"feedback-preparation.{_digest(identity)[:40]}"
    if manifest["preparationId"] != expected_id:
        raise ValueError("feedback preparation identity mismatch")
    return dataset, gate, manifest


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
