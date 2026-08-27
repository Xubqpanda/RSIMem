"""Assemble matched static/proposal observations from audited PAST runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .adaptive_activation import (
    MATCHED_OBSERVATION_BATCH_SCHEMA_VERSION,
    _OBSERVATION_BATCH_IDENTITY_FIELDS,
    _SOURCE_RUN_FIELDS,
)
from .adaptive_preparation import load_offline_adaptive_preparation
from .experiment_manifest import (
    ADAPTIVE_VALIDATION_METHOD_VARIANTS,
    load_manifest,
)
from .memory.attribution import DeterministicFirstAttributor
from .memory.feedback_dataset import (
    DelayedFeedbackConfig,
    DelayedFeedbackDatasetBuilder,
    FeedbackObservationWindow,
    evaluate_feedback_dataset_stage_gate,
)
from .memory.operation_graph import (
    AppendOnlyOperationEvidenceLog,
    materialize_operation_graph,
)
from .memory.utility import (
    MEM0_UTILITY_PARAMETER_IDS,
    STATIC_UTILITY_FEATURE_SCHEMA,
    UtilityTarget,
)
from .memory.adaptive_matched_validation import (
    MatchedPolicyObservation,
    MatchedPolicyVariant,
)


MATCHED_RESOURCE_COST_SCHEMA = "matched-token-request-storage-v1"
_METHOD_VARIANTS = {
    "static-rsimem": MatchedPolicyVariant.STATIC,
    "proposal-rsimem": MatchedPolicyVariant.PROPOSAL,
}
_USAGE_FIELDS = (
    "inputTokens",
    "outputTokens",
    "cacheReadTokens",
    "cacheWriteTokens",
    "reasoningTokens",
    "requests",
    "retries",
)


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _completed_attempts(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    if tuple(manifest["configuration"]["executionModes"]) != (
        ADAPTIVE_VALIDATION_METHOD_VARIANTS
    ):
        raise ValueError("matched evidence requires the frozen validation methods")
    selected = []
    for replicate in range(1, manifest["replicates"] + 1):
        for method in ADAPTIVE_VALIDATION_METHOD_VARIANTS:
            attempts = [
                attempt
                for attempt in manifest["attempts"]
                if attempt["replicate"] == replicate
                and attempt["mode"] == method
                and attempt["status"] == "completed"
            ]
            if len(attempts) != 1:
                raise ValueError("matched validation attempt is incomplete or ambiguous")
            selected.append(attempts[0])
    return tuple(selected)


def _run_dataset(
    run_dir: Path,
    *,
    expected_policy_version: str,
):
    paths = tuple(sorted(
        run_dir.glob("[0-9][0-9]_*/*/rsimem_semantic_operations.jsonl")
    ))
    if not paths:
        raise ValueError("matched validation run has no operation evidence")
    merged = AppendOnlyOperationEvidenceLog()
    source_digests = []
    for path in paths:
        source_digests.append({
            "relativePath": path.relative_to(run_dir).as_posix(),
            "sha256": _file_digest(path),
        })
        for event in AppendOnlyOperationEvidenceLog(path).events:
            merged.append(event)
    graph = materialize_operation_graph(merged.events)
    policy_versions = {
        operation.context.policy_version for operation in graph.operations
    }
    if policy_versions != {expected_policy_version}:
        raise ValueError("matched validation run used another policy version")
    attribution = DeterministicFirstAttributor().attribute(graph)
    config = DelayedFeedbackConfig(
        expected_policy_version,
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
        raise ValueError("matched validation feedback dataset gate failed")
    owner = MEM0_UTILITY_PARAMETER_IDS[UtilityTarget.RETRIEVAL]
    candidates = tuple(
        example
        for example in dataset.examples
        if owner in example.policy_parameter_ids
    )
    if len(candidates) != 1:
        raise ValueError("matched validation run has ambiguous retrieval feedback")
    return dataset, candidates[0], _digest(merged.events), source_digests


def assemble_matched_validation_observations(
    batch_root: Path,
    offline_preparation_root: Path,
    *,
    output_path: Path,
) -> dict[str, Any]:
    batch_root = batch_root.expanduser().resolve()
    preparation = load_offline_adaptive_preparation(offline_preparation_root)
    manifest_path = batch_root / "batch_manifest.json"
    manifest = load_manifest(manifest_path)
    attempts = _completed_attempts(manifest)
    if (
        manifest["configuration"]["semanticFeedbackContract"] != "sm01_tsv_v1"
        or manifest["configuration"]["taskFamily"]
        != "memory_ability/SM01_preference_adoption"
        or manifest["replicates"] != len(preparation.split.validation_example_ids)
    ):
        raise ValueError("matched validation batch scope differs from held-out split")
    policy_profile = manifest["configuration"]["adaptivePolicy"]
    if (
        policy_profile["preparation"] != "matched_validation_trial_store"
        or policy_profile["activePolicyVersion"]
        != preparation.artifact.policy_version
        or policy_profile["activeArtifactId"] != preparation.artifact.artifact_id
        or policy_profile["activeArtifactDigest"]
        != preparation.artifact.content_digest
    ):
        raise ValueError("matched validation batch binds another proposal")
    task_digest = manifest["configuration"]["budget"]["taskManifestDigest"]
    if (
        not isinstance(task_digest, str)
        or len(task_digest) != 64
        or any(character not in "0123456789abcdef" for character in task_digest)
    ):
        raise ValueError("matched validation task input digest is invalid")
    cost_identity = {
        "schema": MATCHED_RESOURCE_COST_SCHEMA,
        "budget": manifest["configuration"]["budget"],
    }
    budget_id = f"budget.{_digest(cost_identity)[:40]}"
    model_digest = _digest(manifest["configuration"]["model"])
    batch_manifest_digest = _file_digest(manifest_path)
    membership = dict(preparation.split.validation_membership)
    validation_ids = preparation.split.validation_example_ids
    observations = []
    sources = []
    for attempt in attempts:
        replicate = attempt["replicate"]
        example_id = validation_ids[replicate - 1]
        episode_id = membership[example_id]
        variant = _METHOD_VARIANTS[attempt["mode"]]
        expected_policy = (
            preparation.artifact.parent_policy_version
            if variant == MatchedPolicyVariant.STATIC
            else preparation.artifact.policy_version
        )
        run_dir = (batch_root / attempt["outputDirectory"]).resolve()
        if not run_dir.is_relative_to(batch_root):
            raise ValueError("matched validation run directory escapes batch")
        audit_path = run_dir / "audit.json"
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("matched validation audit cannot be read") from exc
        privacy = audit.get("privacy")
        if (
            audit.get("ok") is not True
            or audit.get("issues") != []
            or not isinstance(privacy, dict)
            or privacy.get("absoluteSourcePaths") != 0
            or privacy.get("memoryTextLeaks") != 0
            or any((privacy.get("credentialPatternHits") or {}).values())
            or audit.get("runId") != attempt["runName"]
        ):
            raise ValueError("matched validation run audit failed")
        usage = audit.get("uniquePhysicalUsage")
        if not isinstance(usage, dict) or set(usage) != set(_USAGE_FIELDS) or any(
            type(usage[field]) is not int or usage[field] < 0
            for field in _USAGE_FIELDS
        ):
            raise ValueError("matched validation resource usage is incomplete")
        dataset, derived, graph_digest, source_logs = _run_dataset(
            run_dir,
            expected_policy_version=expected_policy,
        )
        resource_identity = {
            "schema": MATCHED_RESOURCE_COST_SCHEMA,
            "usage": usage,
            "storageBytes": derived.resources.storage_bytes,
            "sourceLogs": source_logs,
        }
        lifecycle_cost = float(
            1
            + sum(usage[field] for field in _USAGE_FIELDS)
            + derived.resources.storage_bytes
        )
        evidence_cutoff = len(manifest["configuration"]["budget"]["tasks"])
        source = {
            "observationId": "pending",
            "evidenceId": "pending",
            "variant": variant.value,
            "runId": attempt["runName"],
            "replicate": replicate,
            "ordinal": attempt["ordinal"],
            "status": attempt["status"],
            "auditOk": audit["ok"],
            "auditDigest": _file_digest(audit_path),
            "manifestDigest": batch_manifest_digest,
            "modelProfileDigest": model_digest,
            "taskInputDigest": task_digest,
            "budgetId": budget_id,
            "evidenceCutoff": evidence_cutoff,
            "derivedDatasetId": dataset.dataset_id,
            "derivedExampleId": derived.example_id,
            "derivedEpisodeId": derived.source_episode_id,
            "operationGraphDigest": graph_digest,
            "resourceUsageDigest": _digest(resource_identity),
            "feedbackContract": "sm01_tsv_v1",
        }
        if set(source) != _SOURCE_RUN_FIELDS:
            raise AssertionError("matched source run contract drifted")
        evidence_identity = {
            key: source[key]
            for key in sorted(_SOURCE_RUN_FIELDS - {
                "observationId",
                "evidenceId",
            })
        }
        evidence_id = f"matched-evidence.{_digest(evidence_identity)[:40]}"
        observation = MatchedPolicyObservation(
            observation_id="matched-observation.pending",
            split_id=preparation.split.split_id,
            example_id=example_id,
            episode_id=episode_id,
            variant=variant,
            policy_version=expected_policy,
            label=derived.label,
            lifecycle_cost=lifecycle_cost,
            stability_failure=False,
            uncertainty=0.0,
            evidence_id=evidence_id,
            evidence_cutoff=evidence_cutoff,
            task_input_digest=task_digest,
            budget_id=budget_id,
        )
        observation_identity = observation.payload()
        observation_identity.pop("observation_id")
        observation = MatchedPolicyObservation(
            **{
                **observation_identity,
                "observation_id": (
                    f"matched-observation.{_digest(observation_identity)[:40]}"
                ),
            }
        )
        source["observationId"] = observation.observation_id
        source["evidenceId"] = evidence_id
        observations.append(observation)
        sources.append(source)
    identity = {
        "schemaVersion": MATCHED_OBSERVATION_BATCH_SCHEMA_VERSION,
        "sourceAdaptivePreparationId": preparation.manifest[
            "adaptivePreparationId"
        ],
        "benchmark": "PAST-Bench",
        "host": "Hermes",
        "familyId": "SM01_preference_adoption",
        "officialEvaluation": False,
        "splitId": preparation.split.split_id,
        "artifactId": preparation.artifact.artifact_id,
        "policyVersions": [
            preparation.artifact.parent_policy_version,
            preparation.artifact.policy_version,
        ],
        "observations": [item.payload() for item in observations],
        "sourceRuns": sources,
    }
    if tuple(identity) != _OBSERVATION_BATCH_IDENTITY_FIELDS:
        raise AssertionError("matched observation batch contract drifted")
    report = {
        **identity,
        "observationBatchId": f"matched-observations.{_digest(identity)[:40]}",
    }
    _write_json(output_path.expanduser().resolve(), report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("offline_preparation_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = assemble_matched_validation_observations(
        args.batch_root,
        args.offline_preparation_root,
        output_path=args.output,
    )
    print(json.dumps({
        "observationBatchId": report["observationBatchId"],
        "observationCount": len(report["observations"]),
    }, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
