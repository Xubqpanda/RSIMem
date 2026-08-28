"""Prepare an isolated extraction-candidate runtime for matched validation."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .memory.extraction_offline_validation import (
    ExtractionOfflineDecisionStatus,
    ExtractionOfflineValidationDecision,
)
from .memory.extraction_policy_artifact import ExtractionPromptPolicyArtifact
from .memory.extraction_policy_store import (
    ExtractionPolicyState,
    JsonExtractionPolicyStore,
)
from .memory.prompt_components import canonical_json, content_digest
from .memory_systems.mem0_flat import (
    MEM0_FLAT_EXTRACTION_SLOT,
    MEM0_FLAT_EXTRACTION_SLOT_ID,
    Mem0FlatPromptAdapter,
)


EXTRACTION_MATCHED_TRIAL_RUNTIME_SCHEMA_VERSION = 1
EXTRACTION_MATCHED_TRIAL_RUNTIME_SCHEMA = "extraction-matched-trial-runtime-v1"
EXTRACTION_MATCHED_TRIAL_SCOPE = "matched_validation_only"
EXTRACTION_PRODUCTION_SCOPE = "production"
EXTRACTION_TRIAL_CONFIG_FILE = "extraction-matched-trial.json"
EXTRACTION_TRIAL_POLICY_STORE_FILE = "extraction-trial-policies.json"
EXTRACTION_TRIAL_OFFLINE_DECISION_FILE = "offline-validation-decision.json"
_ALLOWED_RUNTIME_SCOPES = {
    EXTRACTION_MATCHED_TRIAL_SCOPE,
    EXTRACTION_PRODUCTION_SCOPE,
}


def _file_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"{path.name} cannot be read") from exc


def _strict_mapping(
    value: object,
    fields: set[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"malformed {name}")
    return value


def _read_json(path: Path, name: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} cannot be read") from exc


def _write_immutable_json(path: Path, value: object) -> None:
    serialized = canonical_json(value) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise ValueError(f"{path.name} conflicts with existing content")
        return
    file_descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_offline_join(
    parent: ExtractionPromptPolicyArtifact,
    candidate: ExtractionPromptPolicyArtifact,
    offline_decision: ExtractionOfflineValidationDecision,
) -> None:
    expected_root = Mem0FlatPromptAdapter().export_root_policy_artifact(
        MEM0_FLAT_EXTRACTION_SLOT_ID
    )
    if parent != expected_root:
        raise ValueError("extraction matched trial parent is not trusted root")
    if parent.parent_artifact_id is not None:
        raise ValueError("extraction matched trial parent must be a root artifact")
    if candidate.parent_artifact_id != parent.artifact_id:
        raise ValueError("extraction matched trial candidate parent differs")
    if (
        offline_decision.status
        != ExtractionOfflineDecisionStatus.ACCEPTED_FOR_MATCHED_TRIAL
        or offline_decision.eligible_next_stage != "matched_trial"
    ):
        raise ValueError("extraction matched trial requires offline acceptance")
    if (
        offline_decision.parent_artifact_id != parent.artifact_id
        or offline_decision.parent_artifact_digest != parent.artifact_digest
        or offline_decision.candidate_artifact_id != candidate.artifact_id
        or offline_decision.candidate_artifact_digest != candidate.artifact_digest
    ):
        raise ValueError("extraction matched trial offline join differs")
    parent.to_prompt_component(MEM0_FLAT_EXTRACTION_SLOT)
    candidate.to_prompt_component(MEM0_FLAT_EXTRACTION_SLOT)


@dataclass(frozen=True, slots=True)
class ResolvedExtractionMatchedTrialRuntime:
    config_path: Path
    policy_store_path: Path
    offline_decision_path: Path
    trial_id: str
    parent: ExtractionPromptPolicyArtifact
    candidate: ExtractionPromptPolicyArtifact
    offline_decision: ExtractionOfflineValidationDecision

    def profile(self) -> dict[str, object]:
        return {
            "schemaVersion": EXTRACTION_MATCHED_TRIAL_RUNTIME_SCHEMA_VERSION,
            "preparation": "extraction_matched_trial_store",
            "deploymentScope": EXTRACTION_MATCHED_TRIAL_SCOPE,
            "officialEvaluation": False,
            "validationOnly": True,
            "productionActivationAllowed": False,
            "trialId": self.trial_id,
            "slotId": MEM0_FLAT_EXTRACTION_SLOT_ID,
            "parentArtifactId": self.parent.artifact_id,
            "parentArtifactDigest": self.parent.artifact_digest,
            "candidateArtifactId": self.candidate.artifact_id,
            "candidateArtifactDigest": self.candidate.artifact_digest,
            "offlineDecisionId": self.offline_decision.decision_id,
            "configDigest": _file_digest(self.config_path),
            "policyStoreDigest": _file_digest(self.policy_store_path),
            "offlineDecisionDigest": _file_digest(self.offline_decision_path),
        }


def prepare_extraction_matched_trial_runtime(
    *,
    parent: ExtractionPromptPolicyArtifact,
    candidate: ExtractionPromptPolicyArtifact,
    offline_decision: ExtractionOfflineValidationDecision,
    output_root: Path,
) -> dict[str, object]:
    """Activate one candidate only inside a validation-scoped policy store."""

    _validate_offline_join(parent, candidate, offline_decision)
    output = output_root.expanduser().resolve()
    config_path = output / EXTRACTION_TRIAL_CONFIG_FILE
    store_path = output / EXTRACTION_TRIAL_POLICY_STORE_FILE
    offline_path = output / EXTRACTION_TRIAL_OFFLINE_DECISION_FILE
    with _exclusive_lock(output / ".extraction-trial.lock"):
        store = JsonExtractionPolicyStore(
            store_path,
            trusted_root=parent,
            slot=MEM0_FLAT_EXTRACTION_SLOT,
        )
        store.initialize()
        record, _ = store.register(candidate)
        transition_id = (
            "extraction-transition.matched-validation-trial."
            f"{content_digest(offline_decision.decision_id)[:24]}"
        )
        if record.state == ExtractionPolicyState.PROPOSAL:
            record, _ = store.transition(
                candidate.artifact_id,
                to_state=ExtractionPolicyState.ACTIVE,
                transition_id=transition_id,
                reason_code="matched_validation_trial",
            )
        if (
            record.state != ExtractionPolicyState.ACTIVE
            or record.last_transition_id != transition_id
            or record.reason_code != "matched_validation_trial"
            or store.snapshot().active != candidate
        ):
            raise ValueError("extraction matched trial store conflicts with candidate")
        _write_immutable_json(offline_path, offline_decision.payload())
        identity = {
            "schemaVersion": EXTRACTION_MATCHED_TRIAL_RUNTIME_SCHEMA_VERSION,
            "configSchema": EXTRACTION_MATCHED_TRIAL_RUNTIME_SCHEMA,
            "deploymentScope": EXTRACTION_MATCHED_TRIAL_SCOPE,
            "officialEvaluation": False,
            "validationOnly": True,
            "productionActivationAllowed": False,
            "slotId": MEM0_FLAT_EXTRACTION_SLOT_ID,
            "slotContractDigest": MEM0_FLAT_EXTRACTION_SLOT.contract_digest,
            "frozenWrapperDigest": MEM0_FLAT_EXTRACTION_SLOT.frozen_wrapper_digest,
            "parentArtifactId": parent.artifact_id,
            "parentArtifactDigest": parent.artifact_digest,
            "candidateArtifactId": candidate.artifact_id,
            "candidateArtifactDigest": candidate.artifact_digest,
            "offlineDecisionId": offline_decision.decision_id,
            "policyStoreFile": EXTRACTION_TRIAL_POLICY_STORE_FILE,
            "policyStoreDigest": _file_digest(store_path),
            "offlineDecisionFile": EXTRACTION_TRIAL_OFFLINE_DECISION_FILE,
            "offlineDecisionDigest": _file_digest(offline_path),
        }
        config = {
            **identity,
            "trialId": f"extraction-trial.{content_digest(identity)[:40]}",
        }
        _write_immutable_json(config_path, config)
    return config


def load_extraction_runtime_profile(
    config_path: Path,
    *,
    required_scope: str,
) -> ResolvedExtractionMatchedTrialRuntime:
    """Load a trial config only for its declared scope."""

    if required_scope not in _ALLOWED_RUNTIME_SCOPES:
        raise ValueError("unknown extraction runtime scope")
    path = config_path.expanduser().resolve()
    fields = {
        "schemaVersion",
        "configSchema",
        "deploymentScope",
        "officialEvaluation",
        "validationOnly",
        "productionActivationAllowed",
        "slotId",
        "slotContractDigest",
        "frozenWrapperDigest",
        "parentArtifactId",
        "parentArtifactDigest",
        "candidateArtifactId",
        "candidateArtifactDigest",
        "offlineDecisionId",
        "policyStoreFile",
        "policyStoreDigest",
        "offlineDecisionFile",
        "offlineDecisionDigest",
        "trialId",
    }
    config = dict(_strict_mapping(
        _read_json(path, "extraction runtime config"),
        fields,
        "extraction runtime config",
    ))
    identity = {key: config[key] for key in fields if key != "trialId"}
    if (
        config["schemaVersion"]
        != EXTRACTION_MATCHED_TRIAL_RUNTIME_SCHEMA_VERSION
        or config["configSchema"] != EXTRACTION_MATCHED_TRIAL_RUNTIME_SCHEMA
        or config["deploymentScope"] != EXTRACTION_MATCHED_TRIAL_SCOPE
        or config["officialEvaluation"] is not False
        or config["validationOnly"] is not True
        or config["productionActivationAllowed"] is not False
        or config["slotId"] != MEM0_FLAT_EXTRACTION_SLOT_ID
        or config["slotContractDigest"]
        != MEM0_FLAT_EXTRACTION_SLOT.contract_digest
        or config["frozenWrapperDigest"]
        != MEM0_FLAT_EXTRACTION_SLOT.frozen_wrapper_digest
        or config["policyStoreFile"] != EXTRACTION_TRIAL_POLICY_STORE_FILE
        or config["offlineDecisionFile"]
        != EXTRACTION_TRIAL_OFFLINE_DECISION_FILE
        or config["trialId"]
        != f"extraction-trial.{content_digest(identity)[:40]}"
    ):
        raise ValueError("extraction matched trial config identity mismatch")
    if config["deploymentScope"] != required_scope:
        raise ValueError("extraction runtime config cannot be used for this scope")
    root = path.parent
    store_path = root / config["policyStoreFile"]
    offline_path = root / config["offlineDecisionFile"]
    if (
        config["policyStoreDigest"] != _file_digest(store_path)
        or config["offlineDecisionDigest"] != _file_digest(offline_path)
    ):
        raise ValueError("extraction matched trial file digest mismatch")
    parent = Mem0FlatPromptAdapter().export_root_policy_artifact(
        MEM0_FLAT_EXTRACTION_SLOT_ID
    )
    store = JsonExtractionPolicyStore(
        store_path,
        trusted_root=parent,
        slot=MEM0_FLAT_EXTRACTION_SLOT,
    )
    snapshot = store.snapshot()
    candidate = snapshot.active
    if (
        candidate is None
        or snapshot.root != parent
        or config["parentArtifactId"] != parent.artifact_id
        or config["parentArtifactDigest"] != parent.artifact_digest
        or config["candidateArtifactId"] != candidate.artifact_id
        or config["candidateArtifactDigest"] != candidate.artifact_digest
    ):
        raise ValueError("extraction matched trial ACTIVE identity mismatch")
    offline = ExtractionOfflineValidationDecision.from_payload(
        _read_json(offline_path, "offline extraction decision")
    )
    if config["offlineDecisionId"] != offline.decision_id:
        raise ValueError("extraction matched trial offline decision mismatch")
    _validate_offline_join(parent, candidate, offline)
    return ResolvedExtractionMatchedTrialRuntime(
        path,
        store_path,
        offline_path,
        config["trialId"],
        parent,
        candidate,
        offline,
    )


def load_extraction_matched_trial_profile(
    config_path: Path,
) -> ResolvedExtractionMatchedTrialRuntime:
    return load_extraction_runtime_profile(
        config_path,
        required_scope=EXTRACTION_MATCHED_TRIAL_SCOPE,
    )


def _load_candidate(path: Path) -> ExtractionPromptPolicyArtifact:
    try:
        return ExtractionPromptPolicyArtifact.from_payload(
            _read_json(path.expanduser().resolve(), "extraction candidate artifact")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("extraction candidate artifact cannot be loaded") from exc


def _load_offline_decision(path: Path) -> ExtractionOfflineValidationDecision:
    try:
        return ExtractionOfflineValidationDecision.from_payload(
            _read_json(path.expanduser().resolve(), "offline extraction decision")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("offline extraction decision cannot be loaded") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_artifact", type=Path)
    parser.add_argument("offline_decision", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    parent = Mem0FlatPromptAdapter().export_root_policy_artifact(
        MEM0_FLAT_EXTRACTION_SLOT_ID
    )
    config = prepare_extraction_matched_trial_runtime(
        parent=parent,
        candidate=_load_candidate(args.candidate_artifact),
        offline_decision=_load_offline_decision(args.offline_decision),
        output_root=args.output,
    )
    print(canonical_json({
        "trialId": config["trialId"],
        "candidateArtifactId": config["candidateArtifactId"],
        "officialEvaluation": config["officialEvaluation"],
        "deploymentScope": config["deploymentScope"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
