"""Fail-closed manifests for formal extraction-prompt experiments."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .memory.extraction_feedback import (
    FamilyFeedbackContract,
    OpportunityContract,
    OutcomeContract,
    UseContract,
)
from .memory.extraction_prompt_validation import ExtractionAcceptanceCriteria
from .memory.extraction_source import (
    EXTRACTION_SOURCE_SCHEMA,
    EXTRACTION_SOURCE_SCHEMA_VERSION,
)
from .memory.prompt_components import (
    MatchedSemanticPolicyManifest,
    PromptPolicyStage,
    SemanticPolicyManifest,
)


EXTRACTION_EXPERIMENT_SCHEMA_VERSION = 1
EXTRACTION_EXPERIMENT_SCHEMA = "extraction-prompt-experiment-manifest-v1"
EXTRACTION_BATCH_REGISTRY_SCHEMA_VERSION = 1
EXTRACTION_METHOD_VARIANTS = (
    "static-extraction-rsimem",
    "adaptive-extraction-rsimem",
)
EXTRACTION_FEEDBACK_METHOD_VARIANTS = ("static-extraction-rsimem",)
EXTRACTION_PHASES = ("feedback", "validation", "final")
EXTRACTION_SPLIT_ROLES = ("train", "validation", "final_test")
EXTRACTION_PRIMARY_OBJECTIVE_SCHEMA = "resolved-observed-useful-rate-v1"
EXTRACTION_PRIMARY_UNIT = "completed-source-extraction-set-future-opportunity"
EXTRACTION_RESOLVED_DENOMINATOR = "useful_set_count+harmful_set_count"

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_ATTEMPT_STATUSES = {"running", "completed", "failed"}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _require_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")
    return value


def _require_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be sha256")
    return value


def _require_exact(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields are incomplete or unknown")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True, slots=True)
class CleanRepositoryRevision:
    commit: str
    tree: str
    dirty: bool = False

    def __post_init__(self) -> None:
        _require_identifier(self.commit, "repository commit")
        _require_identifier(self.tree, "repository tree")
        if type(self.dirty) is not bool:
            raise TypeError("repository dirty state must be bool")
        if self.dirty:
            raise ValueError("formal extraction experiments require a clean repository")

    def payload(self) -> dict[str, object]:
        return {"commit": self.commit, "tree": self.tree, "dirty": self.dirty}


def resolve_clean_repository(path: Path) -> CleanRepositoryRevision:
    """Resolve a Git commit/tree and reject tracked or untracked changes."""

    root = path.expanduser().resolve()

    def git(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ("git", "-C", str(root), *arguments),
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(f"repository identity cannot be resolved: {root.name}") from exc
        return result.stdout.strip()

    dirty = bool(git("status", "--porcelain", "--untracked-files=normal"))
    return CleanRepositoryRevision(
        commit=git("rev-parse", "HEAD"),
        tree=git("rev-parse", "HEAD^{tree}"),
        dirty=dirty,
    )


def extraction_execution_profile(method: str) -> dict[str, object]:
    """Return the formal runtime profile; only extraction identity may differ."""

    if method not in EXTRACTION_METHOD_VARIANTS:
        raise ValueError("unknown formal extraction method")
    return {
        "persistenceVariant": "with_persistence",
        "rsimemMode": "native+ledger",
        "lifecycleEvaluatorMode": "disabled",
        "semanticWritebackMode": "static",
        "utilityGate": "disabled",
        "nativeMemoryWriter": "disabled",
        "backgroundMemoryReview": "disabled",
        "extractionArtifactRole": (
            "parent" if method == EXTRACTION_METHOD_VARIANTS[0] else "active"
        ),
    }


def extraction_execution_order(
    replicate: int,
    methods: tuple[str, ...] = EXTRACTION_METHOD_VARIANTS,
) -> tuple[str, ...]:
    if type(replicate) is not int or replicate < 1:
        raise ValueError("replicate must be positive")
    if (
        not methods
        or len(methods) != len(set(methods))
        or any(method not in EXTRACTION_METHOD_VARIANTS for method in methods)
    ):
        raise ValueError("formal extraction methods must be unique and known")
    offset = (replicate - 1) % len(methods)
    return methods[offset:] + methods[:offset]


def _criteria_payload(criteria: ExtractionAcceptanceCriteria) -> dict[str, object]:
    return {
        "minimumMatchedPairs": criteria.minimum_matched_pairs,
        "minimumResolvedExamples": criteria.minimum_resolved_examples,
        "minimumUsefulRateDelta": criteria.minimum_useful_rate_delta,
        "maximumHarmfulRateDelta": criteria.maximum_harmful_rate_delta,
        "minimumCoverageRatio": criteria.minimum_coverage_ratio,
        "maximumEmptyRate": criteria.maximum_empty_rate,
        "maximumMissedRateDelta": criteria.maximum_missed_rate_delta,
        "requiredMetrics": list(criteria.required_metrics),
        "proposalBudgetId": criteria.proposal_budget_id,
        "maximumProposalGenerations": criteria.maximum_proposal_generations,
        "maximumCandidateSelections": criteria.maximum_candidate_selections,
        "criteriaDigest": criteria.digest,
    }


def _semantic_from_payload(value: object) -> SemanticPolicyManifest:
    payload = _require_exact(value, {
        "schema_version",
        "manifest_schema",
        "route",
        "boundary",
        "backend",
        "framework_version",
        "model_profile",
        "extraction_component_id",
        "extraction_component_digest",
        "update_component_id",
        "update_component_digest",
        "retrieval_component_id",
        "retrieval_component_digest",
        "composite_digest",
        "composite_policy_version",
    }, "semantic policy manifest")
    return SemanticPolicyManifest(**payload)


def _matched_from_payload(value: object) -> MatchedSemanticPolicyManifest:
    payload = _require_exact(value, {
        "schema_version",
        "manifest_schema",
        "intervention_component",
        "parent",
        "candidate",
        "matched_digest",
    }, "matched semantic policy manifest")
    return MatchedSemanticPolicyManifest(
        parent=_semantic_from_payload(payload["parent"]),
        candidate=_semantic_from_payload(payload["candidate"]),
        intervention_component=PromptPolicyStage(payload["intervention_component"]),
        matched_digest=payload["matched_digest"],
        manifest_schema=payload["manifest_schema"],
        schema_version=payload["schema_version"],
    )


def _validate_feedback_contract(value: object, family_id: str) -> dict[str, Any]:
    wrapper = _require_exact(value, {"identity", "contractDigest"}, "feedback contract")
    identity = _require_exact(wrapper["identity"], {
        "schema_version",
        "contract_schema",
        "opportunity",
        "use",
        "outcome",
    }, "feedback contract identity")
    digest = _require_digest(wrapper["contractDigest"], "feedback contract digest")
    if digest != _digest(identity):
        raise ValueError("feedback contract digest mismatch")
    opportunity_payload = _require_exact(identity["opportunity"], {
        "schema_version", "contract_id", "family_id", "eligible_stages",
        "memory_scope_keys", "allowed_surfaces", "ambiguity_semantics",
    }, "opportunity contract")
    use_payload = _require_exact(identity["use"], {
        "schema_version", "contract_id", "family_id", "parser_id",
        "allowed_surfaces",
    }, "use contract")
    outcome_payload = _require_exact(identity["outcome"], {
        "schema_version", "contract_id", "family_id", "parser_id",
        "allowed_surfaces",
    }, "outcome contract")
    contract = FamilyFeedbackContract(
        opportunity=OpportunityContract(
            contract_id=opportunity_payload["contract_id"],
            family_id=opportunity_payload["family_id"],
            eligible_stages=tuple(opportunity_payload["eligible_stages"]),
            memory_scope_keys=tuple(opportunity_payload["memory_scope_keys"]),
            allowed_surfaces=tuple(opportunity_payload["allowed_surfaces"]),
            ambiguity_semantics=opportunity_payload["ambiguity_semantics"],
            schema_version=opportunity_payload["schema_version"],
        ),
        use=UseContract(
            contract_id=use_payload["contract_id"],
            family_id=use_payload["family_id"],
            parser_id=use_payload["parser_id"],
            allowed_surfaces=tuple(use_payload["allowed_surfaces"]),
            schema_version=use_payload["schema_version"],
        ),
        outcome=OutcomeContract(
            contract_id=outcome_payload["contract_id"],
            family_id=outcome_payload["family_id"],
            parser_id=outcome_payload["parser_id"],
            allowed_surfaces=tuple(outcome_payload["allowed_surfaces"]),
            schema_version=outcome_payload["schema_version"],
        ),
        contract_digest=digest,
        contract_schema=identity["contract_schema"],
        schema_version=identity["schema_version"],
    )
    if contract.family_id != family_id:
        raise ValueError("feedback contract family mismatch")
    return wrapper


def _validate_criteria(value: object) -> dict[str, Any]:
    payload = _require_exact(value, {
        "minimumMatchedPairs",
        "minimumResolvedExamples",
        "minimumUsefulRateDelta",
        "maximumHarmfulRateDelta",
        "minimumCoverageRatio",
        "maximumEmptyRate",
        "maximumMissedRateDelta",
        "requiredMetrics",
        "proposalBudgetId",
        "maximumProposalGenerations",
        "maximumCandidateSelections",
        "criteriaDigest",
    }, "extraction acceptance criteria")
    criteria = ExtractionAcceptanceCriteria(
        minimum_matched_pairs=payload["minimumMatchedPairs"],
        minimum_resolved_examples=payload["minimumResolvedExamples"],
        minimum_useful_rate_delta=payload["minimumUsefulRateDelta"],
        maximum_harmful_rate_delta=payload["maximumHarmfulRateDelta"],
        minimum_coverage_ratio=payload["minimumCoverageRatio"],
        maximum_empty_rate=payload["maximumEmptyRate"],
        maximum_missed_rate_delta=payload["maximumMissedRateDelta"],
        required_metrics=tuple(payload["requiredMetrics"]),
        proposal_budget_id=payload["proposalBudgetId"],
        maximum_proposal_generations=payload["maximumProposalGenerations"],
        maximum_candidate_selections=payload["maximumCandidateSelections"],
    )
    if payload["criteriaDigest"] != criteria.digest:
        raise ValueError("extraction acceptance criteria digest mismatch")
    return payload


def _validate_attempt_history(value: dict[str, Any]) -> None:
    history = value.get("attemptHistory")
    if not isinstance(history, list):
        raise ValueError("extraction attempt history must be a list")
    fields = {
        "eventNumber", "attemptNumber", "replicate", "ordinal", "method",
        "runName", "outputDirectory", "status", "failureStage",
    }
    run_states: dict[str, str] = {}
    slot_attempts: dict[tuple[int, int], int] = {}
    completed_slots: set[tuple[int, int]] = set()
    for index, event in enumerate(history, start=1):
        event = _require_exact(event, fields, "extraction attempt event")
        if event["eventNumber"] != index:
            raise ValueError("extraction attempt event order is invalid")
        replicate = event["replicate"]
        ordinal = event["ordinal"]
        method = event["method"]
        _validate_slot(value, replicate, ordinal, method)
        slot = (replicate, ordinal)
        status = event["status"]
        run_name = event["runName"]
        if (
            status not in _ATTEMPT_STATUSES
            or not isinstance(run_name, str)
            or not run_name
            or event["outputDirectory"] != run_name
        ):
            raise ValueError("extraction attempt event identity is invalid")
        previous = run_states.get(run_name)
        if previous is None:
            expected_attempt = slot_attempts.get(slot, 0) + 1
            if status != "running" or slot in completed_slots:
                raise ValueError("new extraction attempt must start in running state")
            if event["attemptNumber"] != expected_attempt:
                raise ValueError("extraction attempt number is invalid")
            slot_attempts[slot] = expected_attempt
        elif previous != "running" or status == "running":
            raise ValueError("extraction attempt transition is invalid")
        if previous is not None and event["attemptNumber"] != slot_attempts[slot]:
            raise ValueError("extraction attempt terminal identity is invalid")
        failure_stage = event["failureStage"]
        if status == "failed":
            if not isinstance(failure_stage, str) or not failure_stage:
                raise ValueError("failed extraction attempt requires a failure stage")
        elif failure_stage is not None:
            raise ValueError("non-failed extraction attempt has a failure stage")
        run_states[run_name] = status
        if status == "completed":
            completed_slots.add(slot)


def _validate_slot(
    value: dict[str, Any],
    replicate: object,
    ordinal: object,
    method: object,
) -> None:
    if type(replicate) is not int or type(ordinal) is not int:
        raise ValueError("extraction attempt slot is invalid")
    schedule = value["executionOrderByReplicate"].get(str(replicate))
    if (
        not isinstance(schedule, list)
        or ordinal < 1
        or ordinal > len(schedule)
        or schedule[ordinal - 1] != method
    ):
        raise ValueError("extraction attempt does not match its scheduled slot")


def validate_extraction_manifest(value: object) -> dict[str, Any]:
    manifest = _require_exact(value, {
        "schemaVersion",
        "manifestSchema",
        "batchId",
        "experimentId",
        "phase",
        "replicates",
        "split",
        "methods",
        "executionProfiles",
        "executionOrderByReplicate",
        "sourceProjection",
        "semanticPolicy",
        "feedbackContract",
        "objective",
        "acceptanceCriteria",
        "modelProfile",
        "requestBudget",
        "persistenceIsolation",
        "revisions",
        "attemptHistory",
    }, "extraction experiment manifest")
    if (
        manifest["schemaVersion"] != EXTRACTION_EXPERIMENT_SCHEMA_VERSION
        or manifest["manifestSchema"] != EXTRACTION_EXPERIMENT_SCHEMA
    ):
        raise ValueError("unsupported extraction experiment manifest schema")
    _require_identifier(manifest["batchId"], "extraction batch ID")
    phase = manifest["phase"]
    if phase not in EXTRACTION_PHASES:
        raise ValueError("unknown extraction experiment phase")
    if type(manifest["replicates"]) is not int or manifest["replicates"] < 1:
        raise ValueError("extraction replicate count must be positive")

    split = _require_exact(manifest["split"], {
        "role", "familyId", "taskTemplateGroupId", "taskManifestDigest",
    }, "extraction split")
    if split["role"] not in EXTRACTION_SPLIT_ROLES:
        raise ValueError("unknown extraction split role")
    for field in ("familyId", "taskTemplateGroupId"):
        _require_identifier(split[field], f"extraction split {field}")
    _require_digest(split["taskManifestDigest"], "task manifest digest")
    expected_split = {"feedback": "train", "validation": "validation", "final": "final_test"}
    if split["role"] != expected_split[phase]:
        raise ValueError("extraction phase and split role disagree")

    methods = manifest["methods"]
    expected_methods = (
        EXTRACTION_FEEDBACK_METHOD_VARIANTS
        if phase == "feedback"
        else EXTRACTION_METHOD_VARIANTS
    )
    if not isinstance(methods, list) or tuple(methods) != expected_methods:
        raise ValueError("formal extraction method mapping is invalid")
    profiles = _require_exact(
        manifest["executionProfiles"], set(methods), "extraction execution profiles"
    )
    for method in methods:
        if profiles[method] != extraction_execution_profile(method):
            raise ValueError("formal extraction execution profile drifted")

    source = _require_exact(
        manifest["sourceProjection"], {"schema", "schemaVersion"},
        "extraction source projection",
    )
    if source != {
        "schema": EXTRACTION_SOURCE_SCHEMA,
        "schemaVersion": EXTRACTION_SOURCE_SCHEMA_VERSION,
    }:
        raise ValueError("stale extraction source projection schema")

    policy = _require_exact(manifest["semanticPolicy"], {
        "parent", "active", "matched", "activeArtifactByMethod",
    }, "extraction semantic policy")
    parent = _semantic_from_payload(policy["parent"])
    active = _semantic_from_payload(policy["active"])
    if phase == "feedback":
        if policy["matched"] is not None or active != parent:
            raise ValueError("feedback phase must use the plain static parent")
    else:
        matched = _matched_from_payload(policy["matched"])
        if matched.parent != parent or matched.candidate != active:
            raise ValueError("matched extraction policy identity mismatch")
    active_by_method = _require_exact(
        policy["activeArtifactByMethod"], set(methods),
        "active extraction artifact mapping",
    )
    expected_artifacts = {
        method: {
            "artifactId": (
                parent.extraction_component_id
                if method == EXTRACTION_METHOD_VARIANTS[0]
                else active.extraction_component_id
            ),
            "artifactDigest": (
                parent.extraction_component_digest
                if method == EXTRACTION_METHOD_VARIANTS[0]
                else active.extraction_component_digest
            ),
        }
        for method in methods
    }
    if active_by_method != expected_artifacts:
        raise ValueError("active extraction artifact mapping mismatch")

    _validate_feedback_contract(manifest["feedbackContract"], split["familyId"])
    criteria = _validate_criteria(manifest["acceptanceCriteria"])
    objective = _require_exact(manifest["objective"], {
        "schema", "primaryUnit", "resolvedDenominator",
    }, "extraction objective")
    if objective != {
        "schema": EXTRACTION_PRIMARY_OBJECTIVE_SCHEMA,
        "primaryUnit": EXTRACTION_PRIMARY_UNIT,
        "resolvedDenominator": EXTRACTION_RESOLVED_DENOMINATOR,
    }:
        raise ValueError("formal extraction objective is invalid")

    model = _require_exact(manifest["modelProfile"], {"profileId", "profileDigest"}, "model profile")
    _require_identifier(model["profileId"], "model profile ID")
    _require_digest(model["profileDigest"], "model profile digest")
    if parent.model_profile != model["profileId"] or active.model_profile != model["profileId"]:
        raise ValueError("semantic policy and model profile disagree")
    budget = _require_exact(manifest["requestBudget"], {"budgetId", "budgetDigest"}, "request budget")
    _require_identifier(budget["budgetId"], "request budget ID")
    _require_digest(budget["budgetDigest"], "request budget digest")
    isolation = _require_exact(
        manifest["persistenceIsolation"],
        {"profileId", "profileDigest", "strategy"},
        "persistence isolation",
    )
    _require_identifier(isolation["profileId"], "persistence profile ID")
    _require_digest(isolation["profileDigest"], "persistence profile digest")
    if isolation["strategy"] != "per_attempt_trace_directory":
        raise ValueError("formal extraction persistence is not isolated")

    revisions = _require_exact(manifest["revisions"], {"rsimem", "pastBench"}, "repository revisions")
    for name in ("rsimem", "pastBench"):
        revision = _require_exact(revisions[name], {"commit", "tree", "dirty"}, f"{name} revision")
        CleanRepositoryRevision(**revision)

    schedule = manifest["executionOrderByReplicate"]
    if not isinstance(schedule, dict) or len(schedule) != manifest["replicates"]:
        raise ValueError("extraction execution schedule is incomplete")
    for replicate in range(1, manifest["replicates"] + 1):
        if schedule.get(str(replicate)) != list(
            extraction_execution_order(replicate, tuple(methods))
        ):
            raise ValueError("extraction execution schedule is invalid")
    _validate_attempt_history(manifest)

    identity = {
        key: manifest[key]
        for key in manifest
        if key not in {"experimentId", "attemptHistory"}
    }
    if manifest["experimentId"] != _digest(identity):
        raise ValueError("extraction experiment identity mismatch")
    return manifest


def load_extraction_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("extraction experiment manifest cannot be read") from exc
    return validate_extraction_manifest(value)


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schemaVersion": EXTRACTION_BATCH_REGISTRY_SCHEMA_VERSION, "entries": []}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("extraction batch registry cannot be read") from exc
    registry = _require_exact(value, {"schemaVersion", "entries"}, "extraction batch registry")
    if registry["schemaVersion"] != EXTRACTION_BATCH_REGISTRY_SCHEMA_VERSION or not isinstance(registry["entries"], list):
        raise ValueError("unsupported extraction batch registry schema")
    fields = {
        "ordinal", "batchId", "experimentId", "phase", "splitRole",
        "familyId", "taskTemplateGroupId", "taskManifestDigest",
        "rsimemCommit", "rsimemTree", "pastBenchCommit", "pastBenchTree",
    }
    batch_ids: set[str] = set()
    experiment_ids: set[str] = set()
    split_by_task: dict[str, str] = {}
    for ordinal, entry in enumerate(registry["entries"], start=1):
        entry = _require_exact(entry, fields, "extraction batch registry entry")
        if entry["ordinal"] != ordinal:
            raise ValueError("extraction batch registry order is invalid")
        batch_id = _require_identifier(entry["batchId"], "registered batch ID")
        experiment_id = _require_digest(entry["experimentId"], "registered experiment ID")
        if batch_id in batch_ids or experiment_id in experiment_ids:
            raise ValueError("extraction batch registry contains a duplicate identity")
        batch_ids.add(batch_id)
        experiment_ids.add(experiment_id)
        digest = _require_digest(entry["taskManifestDigest"], "registered task manifest digest")
        previous = split_by_task.setdefault(digest, entry["splitRole"])
        if previous != entry["splitRole"]:
            raise ValueError("task manifest digest is registered across splits")
    return registry


def initialize_extraction_batch_manifest(
    path: Path,
    *,
    registry_path: Path,
    batch_id: str,
    phase: str,
    replicates: int,
    family_id: str,
    task_template_group_id: str,
    task_manifest_digest: str,
    parent_policy: SemanticPolicyManifest,
    active_policy: SemanticPolicyManifest,
    matched_policy: MatchedSemanticPolicyManifest | None,
    feedback_contract: FamilyFeedbackContract,
    acceptance_criteria: ExtractionAcceptanceCriteria,
    model_profile_id: str,
    model_profile_digest: str,
    request_budget_id: str,
    request_budget_digest: str,
    persistence_profile_id: str,
    persistence_profile_digest: str,
    rsimem_revision: CleanRepositoryRevision,
    past_bench_revision: CleanRepositoryRevision,
) -> str:
    _require_identifier(batch_id, "extraction batch ID")
    if phase not in EXTRACTION_PHASES:
        raise ValueError("unknown extraction experiment phase")
    if type(replicates) is not int or replicates < 1:
        raise ValueError("replicates must be positive")
    methods = (
        EXTRACTION_FEEDBACK_METHOD_VARIANTS
        if phase == "feedback"
        else EXTRACTION_METHOD_VARIANTS
    )
    split_role = {"feedback": "train", "validation": "validation", "final": "final_test"}[phase]
    semantic_policy = {
        "parent": parent_policy.payload(),
        "active": active_policy.payload(),
        "matched": matched_policy.payload() if matched_policy is not None else None,
        "activeArtifactByMethod": {
            method: {
                "artifactId": (
                    parent_policy.extraction_component_id
                    if method == EXTRACTION_METHOD_VARIANTS[0]
                    else active_policy.extraction_component_id
                ),
                "artifactDigest": (
                    parent_policy.extraction_component_digest
                    if method == EXTRACTION_METHOD_VARIANTS[0]
                    else active_policy.extraction_component_digest
                ),
            }
            for method in methods
        },
    }
    value: dict[str, Any] = {
        "schemaVersion": EXTRACTION_EXPERIMENT_SCHEMA_VERSION,
        "manifestSchema": EXTRACTION_EXPERIMENT_SCHEMA,
        "batchId": batch_id,
        "phase": phase,
        "replicates": replicates,
        "split": {
            "role": split_role,
            "familyId": family_id,
            "taskTemplateGroupId": task_template_group_id,
            "taskManifestDigest": task_manifest_digest,
        },
        "methods": list(methods),
        "executionProfiles": {
            method: extraction_execution_profile(method) for method in methods
        },
        "executionOrderByReplicate": {
            str(replicate): list(extraction_execution_order(replicate, methods))
            for replicate in range(1, replicates + 1)
        },
        "sourceProjection": {
            "schema": EXTRACTION_SOURCE_SCHEMA,
            "schemaVersion": EXTRACTION_SOURCE_SCHEMA_VERSION,
        },
        "semanticPolicy": semantic_policy,
        "feedbackContract": {
            "identity": feedback_contract.identity_payload(),
            "contractDigest": feedback_contract.contract_digest,
        },
        "objective": {
            "schema": EXTRACTION_PRIMARY_OBJECTIVE_SCHEMA,
            "primaryUnit": EXTRACTION_PRIMARY_UNIT,
            "resolvedDenominator": EXTRACTION_RESOLVED_DENOMINATOR,
        },
        "acceptanceCriteria": _criteria_payload(acceptance_criteria),
        "modelProfile": {
            "profileId": model_profile_id,
            "profileDigest": model_profile_digest,
        },
        "requestBudget": {
            "budgetId": request_budget_id,
            "budgetDigest": request_budget_digest,
        },
        "persistenceIsolation": {
            "profileId": persistence_profile_id,
            "profileDigest": persistence_profile_digest,
            "strategy": "per_attempt_trace_directory",
        },
        "revisions": {
            "rsimem": rsimem_revision.payload(),
            "pastBench": past_bench_revision.payload(),
        },
        "attemptHistory": [],
    }
    identity = dict(value)
    identity.pop("attemptHistory")
    value["experimentId"] = _digest(identity)
    validate_extraction_manifest(value)

    manifest_path = path.expanduser().resolve()
    registry = registry_path.expanduser().resolve()
    lock_path = registry.with_suffix(registry.suffix + ".lock")
    with _exclusive_lock(lock_path):
        current = _load_registry(registry)
        if manifest_path.exists():
            raise ValueError("formal extraction batch manifest cannot be reused")
        if any(entry["batchId"] == batch_id for entry in current["entries"]):
            raise ValueError("formal extraction batch ID cannot be reused")
        if any(entry["experimentId"] == value["experimentId"] for entry in current["entries"]):
            raise ValueError("formal extraction experiment identity cannot be reused")
        for entry in current["entries"]:
            if (
                entry["taskManifestDigest"] == task_manifest_digest
                and entry["splitRole"] != split_role
            ):
                raise ValueError("task manifest digest cannot cross extraction splits")
        _write_json(manifest_path, value)
        current["entries"].append({
            "ordinal": len(current["entries"]) + 1,
            "batchId": batch_id,
            "experimentId": value["experimentId"],
            "phase": phase,
            "splitRole": split_role,
            "familyId": family_id,
            "taskTemplateGroupId": task_template_group_id,
            "taskManifestDigest": task_manifest_digest,
            "rsimemCommit": rsimem_revision.commit,
            "rsimemTree": rsimem_revision.tree,
            "pastBenchCommit": past_bench_revision.commit,
            "pastBenchTree": past_bench_revision.tree,
        })
        _write_json(registry, current)
        _load_registry(registry)
    return value["experimentId"]


def next_extraction_attempt_name(
    path: Path,
    *,
    replicate: int,
    ordinal: int,
    method: str,
    base_run_name: str,
) -> str | None:
    value = load_extraction_manifest(path)
    _validate_slot(value, replicate, ordinal, method)
    events = [
        event for event in value["attemptHistory"]
        if event["replicate"] == replicate and event["ordinal"] == ordinal
    ]
    if any(event["status"] == "completed" for event in events):
        return None
    running = [event for event in events if event["status"] == "running"]
    terminal_names = {
        event["runName"] for event in events if event["status"] != "running"
    }
    if any(event["runName"] not in terminal_names for event in running):
        raise ValueError("scheduled extraction slot already has a running attempt")
    attempt = 1 + sum(event["status"] == "running" for event in events)
    return base_run_name if attempt == 1 else f"{base_run_name}_attempt{attempt:02d}"


def record_extraction_attempt(
    path: Path,
    *,
    replicate: int,
    ordinal: int,
    method: str,
    run_name: str,
    status: str,
    failure_stage: str | None = None,
) -> None:
    if status not in _ATTEMPT_STATUSES:
        raise ValueError("invalid extraction attempt status")
    if status != "failed" and failure_stage is not None:
        raise ValueError("only failed extraction attempts have a failure stage")
    if (
        not isinstance(run_name, str)
        or not run_name
        or "/" in run_name
        or run_name in {".", ".."}
    ):
        raise ValueError("extraction run name is invalid")
    resolved = path.expanduser().resolve()
    with _exclusive_lock(resolved.with_suffix(resolved.suffix + ".lock")):
        value = load_extraction_manifest(resolved)
        _validate_slot(value, replicate, ordinal, method)
        events = [event for event in value["attemptHistory"] if event["runName"] == run_name]
        slot_events = [
            event for event in value["attemptHistory"]
            if event["replicate"] == replicate and event["ordinal"] == ordinal
        ]
        if not events:
            if status != "running":
                raise ValueError("new extraction attempt must start in running state")
            terminal_names = {
                event["runName"] for event in slot_events if event["status"] != "running"
            }
            if any(
                event["status"] == "completed"
                or (event["status"] == "running" and event["runName"] not in terminal_names)
                for event in slot_events
            ):
                raise ValueError("scheduled extraction slot does not accept another attempt")
            attempt_number = 1 + sum(
                event["status"] == "running" for event in slot_events
            )
        else:
            latest = events[-1]
            if (
                latest["status"] != "running"
                or status == "running"
                or latest["replicate"] != replicate
                or latest["ordinal"] != ordinal
                or latest["method"] != method
            ):
                raise ValueError("invalid extraction attempt status transition")
            attempt_number = latest["attemptNumber"]
        value["attemptHistory"].append({
            "eventNumber": len(value["attemptHistory"]) + 1,
            "attemptNumber": attempt_number,
            "replicate": replicate,
            "ordinal": ordinal,
            "method": method,
            "runName": run_name,
            "outputDirectory": run_name,
            "status": status,
            "failureStage": failure_stage if status == "failed" else None,
        })
        validate_extraction_manifest(value)
        _write_json(resolved, value)
