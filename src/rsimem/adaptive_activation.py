"""Activate an offline proposal only from matched held-out run evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .adaptive_preparation import (
    ADAPTIVE_POLICY_STORE_FILE,
    LoadedAdaptivePreparation,
    load_offline_adaptive_preparation,
)
from .memory.adaptive_matched_validation import (
    JsonMatchedValidationDecisionStore,
    MatchedAcceptanceCriteria,
    MatchedAdaptivePolicyActivationCoordinator,
    MatchedAdaptivePolicyValidator,
    MatchedPolicyObservation,
    MatchedPolicyVariant,
)
from .memory.adaptive_policy import AdaptivePolicyState
from .memory.adaptive_policy_store import JsonAdaptivePolicyStore
from .memory.utility import MEM0_UTILITY_PARAMETER_IDS, UtilityTarget


MATCHED_OBSERVATION_BATCH_SCHEMA_VERSION = 1
ADAPTIVE_ACTIVATION_SCHEMA_VERSION = 1
MATCHED_DECISION_DIRECTORY = "matched-validation-decisions"
ADAPTIVE_CONFIG_FILE = "adaptive-config.json"
ADAPTIVE_ACTIVATION_MANIFEST_FILE = "adaptive-activation-manifest.json"
_OBSERVATION_BATCH_IDENTITY_FIELDS = (
    "schemaVersion",
    "sourceAdaptivePreparationId",
    "benchmark",
    "host",
    "familyId",
    "officialEvaluation",
    "splitId",
    "artifactId",
    "policyVersions",
    "observations",
    "sourceRuns",
)
_SOURCE_RUN_FIELDS = {
    "observationId",
    "evidenceId",
    "variant",
    "runId",
    "replicate",
    "ordinal",
    "status",
    "auditOk",
    "auditDigest",
    "manifestDigest",
    "modelProfileDigest",
    "taskInputDigest",
    "budgetId",
    "evidenceCutoff",
}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_mapping(
    value: object,
    fields: set[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"malformed {name}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True, slots=True)
class LoadedMatchedObservationBatch:
    payload: dict[str, Any]
    observations: tuple[MatchedPolicyObservation, ...]


def load_matched_observation_batch(
    path: Path,
    preparation: LoadedAdaptivePreparation,
) -> LoadedMatchedObservationBatch:
    try:
        raw = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("matched observation batch cannot be read") from exc
    expected = set((*_OBSERVATION_BATCH_IDENTITY_FIELDS, "observationBatchId"))
    payload = dict(_strict_mapping(raw, expected, "matched observation batch"))
    identity = {
        field: payload[field] for field in _OBSERVATION_BATCH_IDENTITY_FIELDS
    }
    if payload["schemaVersion"] != MATCHED_OBSERVATION_BATCH_SCHEMA_VERSION or (
        payload["observationBatchId"]
        != f"matched-observations.{_digest(identity)[:40]}"
    ):
        raise ValueError("matched observation batch identity mismatch")
    if (
        payload["sourceAdaptivePreparationId"]
        != preparation.manifest["adaptivePreparationId"]
        or payload["benchmark"] != "PAST-Bench"
        or payload["host"] != "Hermes"
        or payload["familyId"] != "SM01_preference_adoption"
        or payload["officialEvaluation"] is not False
        or payload["splitId"] != preparation.split.split_id
        or payload["artifactId"] != preparation.artifact.artifact_id
        or payload["policyVersions"] != [
            preparation.artifact.parent_policy_version,
            preparation.artifact.policy_version,
        ]
    ):
        raise ValueError("matched observation batch scope mismatch")
    if not isinstance(payload["observations"], list):
        raise ValueError("matched observation payload is malformed")
    observations = tuple(
        MatchedPolicyObservation.from_payload(value)
        for value in payload["observations"]
    )
    if not observations or not isinstance(payload["sourceRuns"], list):
        raise ValueError("matched observation batch is empty")
    sources = {}
    for raw_source in payload["sourceRuns"]:
        source = _strict_mapping(
            raw_source,
            _SOURCE_RUN_FIELDS,
            "matched observation source run",
        )
        observation_id = source["observationId"]
        if not isinstance(observation_id, str) or observation_id in sources:
            raise ValueError("matched observation source identity is duplicated")
        sources[observation_id] = source
    if set(sources) != {item.observation_id for item in observations}:
        raise ValueError("matched observations and source runs differ")
    model_digests = set()
    manifest_digests = set()
    runs_by_example: dict[str, list[Mapping[str, Any]]] = {}
    for observation in observations:
        source = sources[observation.observation_id]
        runs_by_example.setdefault(observation.example_id, []).append(source)
        if (
            source["evidenceId"] != observation.evidence_id
            or source["variant"] != observation.variant.value
            or source["taskInputDigest"] != observation.task_input_digest
            or source["budgetId"] != observation.budget_id
            or source["evidenceCutoff"] != observation.evidence_cutoff
            or source["status"] != "completed"
            or source["auditOk"] is not True
            or type(source["replicate"]) is not int
            or source["replicate"] < 1
            or type(source["ordinal"]) is not int
            or source["ordinal"] not in {1, 2}
            or not isinstance(source["runId"], str)
            or not source["runId"].strip()
        ):
            raise ValueError("matched observation source run mismatch")
        for digest_field in (
            "auditDigest",
            "manifestDigest",
            "modelProfileDigest",
        ):
            value = source[digest_field]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError("matched observation source digest is invalid")
        model_digests.add(source["modelProfileDigest"])
        manifest_digests.add(source["manifestDigest"])
    if len(model_digests) != 1 or len(manifest_digests) != 1:
        raise ValueError("matched observation execution profile differs")
    for source_pair in runs_by_example.values():
        if (
            len(source_pair) != 2
            or {item["variant"] for item in source_pair}
            != {variant.value for variant in MatchedPolicyVariant}
            or len({item["replicate"] for item in source_pair}) != 1
            or {item["ordinal"] for item in source_pair} != {1, 2}
        ):
            raise ValueError("matched observation run pair differs")
    MatchedAdaptivePolicyValidator().evaluate(
        preparation.artifact,
        preparation.split,
        observations,
        MatchedAcceptanceCriteria(),
    )
    return LoadedMatchedObservationBatch(payload, observations)


def _copy_validated_store(
    preparation: LoadedAdaptivePreparation,
    output_root: Path,
) -> Path:
    source = preparation.root / ADAPTIVE_POLICY_STORE_FILE
    target = output_root / ADAPTIVE_POLICY_STORE_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    fd, temporary = tempfile.mkstemp(prefix=".adaptive-store.", dir=target.parent)
    os.close(fd)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return target


def activate_adaptive_policy(
    offline_preparation_root: Path,
    observation_batch_path: Path,
    *,
    output_root: Path,
) -> dict[str, Any]:
    """Apply the only production activation path from matched held-out evidence."""

    preparation = load_offline_adaptive_preparation(offline_preparation_root)
    if preparation.decision.resulting_state != AdaptivePolicyState.VALIDATED:
        raise ValueError("matched activation requires offline VALIDATED policy")
    output_root = output_root.expanduser().resolve()
    if output_root == preparation.root:
        raise ValueError("matched activation output must not mutate offline preparation")
    batch = load_matched_observation_batch(observation_batch_path, preparation)
    criteria = MatchedAcceptanceCriteria()
    decision = MatchedAdaptivePolicyValidator().evaluate(
        preparation.artifact,
        preparation.split,
        batch.observations,
        criteria,
    )
    policy_path = _copy_validated_store(preparation, output_root)
    policy_store = JsonAdaptivePolicyStore(
        policy_path,
        trusted_root_policy_versions=(
            preparation.artifact.parent_policy_version,
        ),
    )
    snapshot = policy_store.snapshot()
    if (
        len(snapshot.artifacts) != 1
        or snapshot.artifacts[0] != preparation.artifact
        or snapshot.records[0].state not in {
            AdaptivePolicyState.VALIDATED,
            AdaptivePolicyState.ACTIVE,
            AdaptivePolicyState.REJECTED,
        }
    ):
        raise ValueError("matched activation output store conflicts with proposal")
    decision_store = JsonMatchedValidationDecisionStore(
        output_root / MATCHED_DECISION_DIRECTORY
    )
    resulting_state = MatchedAdaptivePolicyActivationCoordinator(
        policy_store,
        decision_store,
    ).apply(
        preparation.artifact,
        decision,
        split=preparation.split,
        observations=batch.observations,
        criteria=criteria,
    )
    config_path = output_root / ADAPTIVE_CONFIG_FILE
    config_digest = None
    profile = None
    if resulting_state == AdaptivePolicyState.ACTIVE:
        update = preparation.artifact.parameters[0]
        retrieval_parameter = MEM0_UTILITY_PARAMETER_IDS[UtilityTarget.RETRIEVAL]
        if update.parameter_id != retrieval_parameter:
            raise ValueError("ACTIVE policy does not own the retrieval threshold")
        config = {
            "schema_version": 1,
            "prepared_policy_store_file": ADAPTIVE_POLICY_STORE_FILE,
            "adaptive_policy_store_path": ".rsimem/adaptive-policies.json",
            "adaptive_trusted_roots": [
                preparation.artifact.parent_policy_version
            ],
            "adaptive_parameters": [{
                "parameter_id": update.parameter_id,
                "name": update.name.value,
                "prompt_ref": preparation.artifact.prompt_refs[0],
                "baseline_value": update.baseline_value,
            }],
        }
        _write_json(config_path, config)
        config_digest = _digest(config)
        from .experiment_manifest import resolved_adaptive_policy_profile

        profile = resolved_adaptive_policy_profile(config_path)
        if profile["activePolicyVersion"] != preparation.artifact.policy_version:
            raise ValueError("generated adaptive config binds another policy")
    elif config_path.exists():
        raise ValueError("rejected adaptive output unexpectedly has a config")

    decision_path = (
        output_root
        / MATCHED_DECISION_DIRECTORY
        / f"{decision.decision_id}.json"
    )
    identity = {
        "schemaVersion": ADAPTIVE_ACTIVATION_SCHEMA_VERSION,
        "sourceAdaptivePreparationId": preparation.manifest[
            "adaptivePreparationId"
        ],
        "observationBatchId": batch.payload["observationBatchId"],
        "criteriaDigest": criteria.digest,
        "artifactId": preparation.artifact.artifact_id,
        "artifactDigest": preparation.artifact.content_digest,
        "policyVersion": preparation.artifact.policy_version,
        "matchedDecisionId": decision.decision_id,
        "matchedDecisionDigest": decision.content_digest,
        "matchedDecisionAccepted": decision.accepted,
        "resultingState": resulting_state.value,
        "activePolicyVersion": (
            preparation.artifact.policy_version
            if resulting_state == AdaptivePolicyState.ACTIVE
            else None
        ),
        "policyStoreDigest": _file_digest(policy_path),
        "matchedDecisionFileDigest": _file_digest(decision_path),
        "adaptiveConfigFile": (
            ADAPTIVE_CONFIG_FILE
            if resulting_state == AdaptivePolicyState.ACTIVE
            else None
        ),
        "adaptiveConfigDigest": config_digest,
    }
    report = {
        **identity,
        "activationId": f"adaptive-activation.{_digest(identity)[:40]}",
        "reasonCodes": list(decision.reason_codes),
        "matchedExampleCount": decision.matched_example_count,
        "resolvedExampleCount": decision.resolved_example_count,
        "activeProfile": profile,
    }
    _write_json(output_root / ADAPTIVE_ACTIVATION_MANIFEST_FILE, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("offline_preparation_root", type=Path)
    parser.add_argument("observation_batch", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = activate_adaptive_policy(
        args.offline_preparation_root,
        args.observation_batch,
        output_root=args.output,
    )
    print(json.dumps({
        "activationId": report["activationId"],
        "policyVersion": report["policyVersion"],
        "resultingState": report["resultingState"],
        "matchedDecisionAccepted": report["matchedDecisionAccepted"],
    }, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
