"""Prepare one offline-screened adaptive retrieval policy from live feedback."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .feedback_preparation import load_prepared_feedback_dataset
from .memory.adaptive_policy import (
    AdaptiveParameterName,
    AdaptiveParameterSpec,
    AdaptivePolicyState,
    AdaptiveTrainingConfig,
    DeterministicAdaptivePolicyLearner,
)
from .memory.adaptive_policy_store import JsonAdaptivePolicyStore
from .memory.adaptive_policy_validation import (
    AdaptiveAcceptanceCriteria,
    AdaptivePolicyLifecycleCoordinator,
    AdaptivePolicyValidator,
    AdaptiveSplitConfig,
    JsonAdaptiveValidationDecisionStore,
    TimeOrderedAdaptiveSplitter,
)
from .memory.utility import (
    MEM0_UTILITY_PARAMETER_IDS,
    StaticUtilityPolicy,
    UtilityTarget,
)


ADAPTIVE_PREPARATION_SCHEMA_VERSION = 1
ADAPTIVE_TRAINING_SEED = 20260828
ADAPTIVE_POLICY_STORE_FILE = "adaptive-policy-store.json"
ADAPTIVE_DECISION_DIRECTORY = "validation-decisions"
ADAPTIVE_SPLIT_FILE = "validation-split.json"
ADAPTIVE_TRAINING_CONFIG_FILE = "training-config.json"
ADAPTIVE_CRITERIA_FILE = "acceptance-criteria.json"
ADAPTIVE_PREPARATION_MANIFEST_FILE = "adaptive-preparation-manifest.json"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def prepare_adaptive_policy(
    prepared_feedback_root: Path,
    *,
    output_root: Path,
) -> dict[str, Any]:
    """Train and offline-screen one retrieval-threshold proposal without activation."""

    prepared_feedback_root = prepared_feedback_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    dataset, gate, source_manifest = load_prepared_feedback_dataset(
        prepared_feedback_root
    )
    split_config = AdaptiveSplitConfig(validation_group_count=1)
    split = TimeOrderedAdaptiveSplitter().split(dataset, gate, split_config)
    retrieval_parameter = MEM0_UTILITY_PARAMETER_IDS[UtilityTarget.RETRIEVAL]
    static_policy = StaticUtilityPolicy()
    training_config = AdaptiveTrainingConfig(
        parent_policy_version=dataset.config.policy_version,
        seed=ADAPTIVE_TRAINING_SEED,
        parameters=(AdaptiveParameterSpec(
            parameter_id=retrieval_parameter,
            name=AdaptiveParameterName.RETRIEVAL_ACCEPT_THRESHOLD,
            baseline_value=static_policy.accept_threshold,
            prompt_ref="mem0-flat.retrieval",
        ),),
        training_example_ids=split.training_example_ids,
    )
    learner = DeterministicAdaptivePolicyLearner()
    artifact = learner.learn(dataset, gate, training_config)
    learner_audit = learner.audit(artifact, dataset, gate, training_config)
    if not learner_audit.ok or learner_audit.replay_artifact_id != artifact.artifact_id:
        raise ValueError("adaptive proposal replay audit failed")
    if tuple(update.parameter_id for update in artifact.parameters) != (
        retrieval_parameter,
    ):
        raise ValueError("adaptive proposal changed a non-retrieval parameter")

    criteria = AdaptiveAcceptanceCriteria()
    validator = AdaptivePolicyValidator()
    decision = validator.evaluate(artifact, dataset, gate, split, criteria)
    if not validator.replay_matches(
        decision,
        artifact,
        dataset,
        gate,
        split,
        criteria,
    ):
        raise ValueError("adaptive offline validation replay failed")

    policy_path = output_root / ADAPTIVE_POLICY_STORE_FILE
    policy_store = JsonAdaptivePolicyStore(
        policy_path,
        trusted_root_policy_versions=(dataset.config.policy_version,),
    )
    existing = policy_store.snapshot()
    if existing.active_policy_version is not None:
        raise ValueError("offline preparation cannot modify an ACTIVE policy store")
    if existing.artifacts and any(
        candidate.artifact_id != artifact.artifact_id
        for candidate in existing.artifacts
    ):
        raise ValueError("adaptive preparation output contains another proposal")
    decision_store = JsonAdaptiveValidationDecisionStore(
        output_root / ADAPTIVE_DECISION_DIRECTORY
    )
    record = AdaptivePolicyLifecycleCoordinator(
        policy_store,
        decision_store,
    ).apply(artifact, decision)
    snapshot = policy_store.snapshot()
    if snapshot.active_policy_version is not None or snapshot.active is not None:
        raise ValueError("offline preparation unexpectedly activated a policy")
    if record.state != decision.resulting_state or record.state not in {
        AdaptivePolicyState.VALIDATED,
        AdaptivePolicyState.REJECTED,
    }:
        raise ValueError("adaptive preparation lifecycle state mismatch")
    stored_decision = decision_store.get(decision.decision_id)
    if stored_decision != decision:
        raise ValueError("adaptive validation decision did not persist")

    split_path = output_root / ADAPTIVE_SPLIT_FILE
    training_path = output_root / ADAPTIVE_TRAINING_CONFIG_FILE
    criteria_path = output_root / ADAPTIVE_CRITERIA_FILE
    _write_json(split_path, split.payload())
    _write_json(training_path, training_config.payload())
    _write_json(criteria_path, criteria.payload())
    decision_path = (
        output_root
        / ADAPTIVE_DECISION_DIRECTORY
        / f"{decision.decision_id}.json"
    )
    identity = {
        "schemaVersion": ADAPTIVE_PREPARATION_SCHEMA_VERSION,
        "sourceFeedbackPreparationId": source_manifest["preparationId"],
        "datasetId": dataset.dataset_id,
        "datasetPayloadDigest": gate.dataset_payload_digest,
        "parentPolicyVersion": dataset.config.policy_version,
        "runtimeOwnedParameterIds": [retrieval_parameter],
        "splitId": split.split_id,
        "splitDigest": split.split_digest,
        "trainingConfigDigest": training_config.digest,
        "criteriaDigest": criteria.digest,
        "artifactId": artifact.artifact_id,
        "artifactDigest": artifact.content_digest,
        "policyVersion": artifact.policy_version,
        "decisionId": decision.decision_id,
        "decisionDigest": decision.content_digest,
        "resultingState": record.state.value,
        "activePolicyVersion": None,
        "files": {
            ADAPTIVE_POLICY_STORE_FILE: _file_digest(policy_path),
            ADAPTIVE_SPLIT_FILE: _file_digest(split_path),
            ADAPTIVE_TRAINING_CONFIG_FILE: _file_digest(training_path),
            ADAPTIVE_CRITERIA_FILE: _file_digest(criteria_path),
            decision_path.relative_to(output_root).as_posix(): _file_digest(
                decision_path
            ),
        },
    }
    report = {
        **identity,
        "adaptivePreparationId": (
            f"adaptive-preparation.{_digest(identity)[:40]}"
        ),
        "trainingExampleCount": len(split.training_example_ids),
        "validationExampleCount": len(split.validation_example_ids),
        "proposalReplayAuditOk": learner_audit.ok,
        "offlineValidationAccepted": decision.accepted,
        "reasonCodes": list(decision.reason_codes),
        "parameterUpdates": [{
            "parameterId": update.parameter_id,
            "name": update.name.value,
            "baselineValue": update.baseline_value,
            "proposedValue": update.proposed_value,
            "delta": update.delta,
            "fallbackReason": update.fallback_reason.value,
            "positiveCount": update.positive_count,
            "negativeCount": update.negative_count,
        } for update in artifact.parameters],
    }
    _write_json(output_root / ADAPTIVE_PREPARATION_MANIFEST_FILE, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_feedback_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = prepare_adaptive_policy(
        args.prepared_feedback_root,
        output_root=args.output,
    )
    print(json.dumps({
        "adaptivePreparationId": report["adaptivePreparationId"],
        "policyVersion": report["policyVersion"],
        "resultingState": report["resultingState"],
        "offlineValidationAccepted": report["offlineValidationAccepted"],
    }, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
