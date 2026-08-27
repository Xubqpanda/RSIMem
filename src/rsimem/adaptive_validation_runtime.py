"""Prepare an isolated ACTIVE deployment used only for matched validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .adaptive_activation import (
    ADAPTIVE_CONFIG_FILE,
    build_adaptive_runtime_config,
)
from .adaptive_preparation import (
    ADAPTIVE_POLICY_STORE_FILE,
    load_offline_adaptive_preparation,
)
from .memory.adaptive_mem0_binding import ActiveAdaptiveMem0Binder
from .memory.adaptive_policy import AdaptivePolicyState
from .memory.adaptive_policy_store import JsonAdaptivePolicyStore
from .memory.live_writeback import StaticSemanticWritebackConfig
from .memory_systems.mem0_flat import FrozenMem0UtilityGate
from .memory_systems.mem0_flat.policy import Mem0FlatSemanticPolicy


MATCHED_VALIDATION_TRIAL_SCHEMA_VERSION = 1
MATCHED_VALIDATION_TRIAL_MANIFEST_FILE = "matched-validation-trial-manifest.json"


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


def _strict_mapping(
    value: object,
    fields: set[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"malformed {name}")
    return value


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
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


def prepare_matched_validation_runtime(
    offline_preparation_root: Path,
    *,
    output_root: Path,
) -> dict[str, Any]:
    preparation = load_offline_adaptive_preparation(offline_preparation_root)
    if preparation.decision.resulting_state != AdaptivePolicyState.VALIDATED:
        raise ValueError("matched validation runtime requires VALIDATED proposal")
    output_root = output_root.expanduser().resolve()
    if output_root == preparation.root:
        raise ValueError("matched validation runtime must not mutate offline preparation")
    policy_path = output_root / ADAPTIVE_POLICY_STORE_FILE
    _copy_file(preparation.root / ADAPTIVE_POLICY_STORE_FILE, policy_path)
    store = JsonAdaptivePolicyStore(
        policy_path,
        trusted_root_policy_versions=(
            preparation.artifact.parent_policy_version,
        ),
    )
    snapshot = store.snapshot()
    if (
        len(snapshot.artifacts) != 1
        or snapshot.artifacts[0] != preparation.artifact
        or snapshot.records[0].state not in {
            AdaptivePolicyState.VALIDATED,
            AdaptivePolicyState.ACTIVE,
        }
    ):
        raise ValueError("matched validation runtime store conflicts with proposal")
    if snapshot.records[0].state == AdaptivePolicyState.VALIDATED:
        store.transition(
            preparation.artifact.policy_version,
            to_state=AdaptivePolicyState.ACTIVE,
            transition_id="policy-transition.matched-validation-trial",
            reason_code="matched_validation_trial",
        )
    active = store.snapshot().active
    if active != preparation.artifact:
        raise ValueError("matched validation runtime did not bind proposal")
    config = build_adaptive_runtime_config(preparation.artifact)
    config_path = output_root / ADAPTIVE_CONFIG_FILE
    _write_json(config_path, config)
    identity = {
        "schemaVersion": MATCHED_VALIDATION_TRIAL_SCHEMA_VERSION,
        "sourceAdaptivePreparationId": preparation.manifest[
            "adaptivePreparationId"
        ],
        "officialEvaluation": False,
        "deploymentScope": "matched_validation_only",
        "artifactId": preparation.artifact.artifact_id,
        "artifactDigest": preparation.artifact.content_digest,
        "parentPolicyVersion": preparation.artifact.parent_policy_version,
        "activePolicyVersion": preparation.artifact.policy_version,
        "policyStoreFile": ADAPTIVE_POLICY_STORE_FILE,
        "policyStoreDigest": _file_digest(policy_path),
        "adaptiveConfigFile": ADAPTIVE_CONFIG_FILE,
        "adaptiveConfigDigest": _file_digest(config_path),
    }
    report = {
        **identity,
        "trialId": f"matched-validation-trial.{_digest(identity)[:40]}",
    }
    _write_json(output_root / MATCHED_VALIDATION_TRIAL_MANIFEST_FILE, report)
    return report


def resolved_matched_validation_trial_profile(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    root = config_path.parent
    manifest_path = root / MATCHED_VALIDATION_TRIAL_MANIFEST_FILE
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("matched validation trial cannot be read") from exc
    fields = {
        "schemaVersion",
        "sourceAdaptivePreparationId",
        "officialEvaluation",
        "deploymentScope",
        "artifactId",
        "artifactDigest",
        "parentPolicyVersion",
        "activePolicyVersion",
        "policyStoreFile",
        "policyStoreDigest",
        "adaptiveConfigFile",
        "adaptiveConfigDigest",
        "trialId",
    }
    manifest = dict(_strict_mapping(
        raw_manifest,
        fields,
        "matched validation trial manifest",
    ))
    identity = {key: manifest[key] for key in fields if key != "trialId"}
    if (
        manifest["schemaVersion"] != MATCHED_VALIDATION_TRIAL_SCHEMA_VERSION
        or manifest["officialEvaluation"] is not False
        or manifest["deploymentScope"] != "matched_validation_only"
        or manifest["trialId"]
        != f"matched-validation-trial.{_digest(identity)[:40]}"
        or manifest["policyStoreFile"] != ADAPTIVE_POLICY_STORE_FILE
        or manifest["adaptiveConfigFile"] != ADAPTIVE_CONFIG_FILE
        or manifest["adaptiveConfigDigest"] != _file_digest(config_path)
    ):
        raise ValueError("matched validation trial identity mismatch")
    expected_config = {
        "schema_version",
        "prepared_policy_store_file",
        "adaptive_policy_store_path",
        "adaptive_trusted_roots",
        "adaptive_parameters",
    }
    config = dict(_strict_mapping(config, expected_config, "adaptive trial config"))
    if config["prepared_policy_store_file"] != manifest["policyStoreFile"]:
        raise ValueError("matched validation trial store binding mismatch")
    policy_path = root / manifest["policyStoreFile"]
    if manifest["policyStoreDigest"] != _file_digest(policy_path):
        raise ValueError("matched validation trial store digest mismatch")
    runtime = StaticSemanticWritebackConfig.from_mapping({
        "mode": "adaptive_utility",
        "adaptive_policy_store_path": config["adaptive_policy_store_path"],
        "adaptive_trusted_roots": config["adaptive_trusted_roots"],
        "adaptive_parameters": config["adaptive_parameters"],
    })
    store = JsonAdaptivePolicyStore(
        policy_path,
        trusted_root_policy_versions=runtime.adaptive_trusted_roots,
    )
    snapshot = store.snapshot()
    if (
        snapshot.active is None
        or snapshot.active.policy_version != manifest["activePolicyVersion"]
        or snapshot.active.artifact_id != manifest["artifactId"]
        or snapshot.active.content_digest != manifest["artifactDigest"]
        or snapshot.active.parent_policy_version != manifest["parentPolicyVersion"]
    ):
        raise ValueError("matched validation trial ACTIVE identity mismatch")
    base_gate = FrozenMem0UtilityGate()
    base_policy = Mem0FlatSemanticPolicy(object(), utility_gate=base_gate)
    binding = ActiveAdaptiveMem0Binder(runtime.adaptive_parameters).bind(
        store,
        base_gate,
        expected_parent_policy_version=base_policy.descriptor.policy_version,
    )
    if not binding.adaptive or binding.artifact_id != snapshot.active.artifact_id:
        raise ValueError("matched validation trial does not bind runtime")
    runtime_identity = {
        key: value
        for key, value in config.items()
        if key != "prepared_policy_store_file"
    }
    return {
        "configSchemaVersion": config["schema_version"],
        "configDigest": _digest(runtime_identity),
        "storeSchemaVersion": 1,
        "storeDigest": _digest(json.loads(policy_path.read_text(encoding="utf-8"))),
        "activePolicyVersion": snapshot.active.policy_version,
        "activeArtifactId": snapshot.active.artifact_id,
        "activeArtifactDigest": snapshot.active.content_digest,
        "preparation": "matched_validation_trial_store",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("offline_preparation_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = prepare_matched_validation_runtime(
        args.offline_preparation_root,
        output_root=args.output,
    )
    print(json.dumps({
        "trialId": report["trialId"],
        "activePolicyVersion": report["activePolicyVersion"],
        "officialEvaluation": report["officialEvaluation"],
    }, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
