"""Resolve and register a formal plain-parent extraction feedback batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from .extraction_experiment_manifest import (
    CleanRepositoryRevision,
    build_extraction_batch_manifest,
    initialize_extraction_batch_manifest,
    resolve_clean_repository,
    resume_extraction_batch_manifest,
)
from .memory.extraction_feedback import default_feedback_contract_registry
from .memory.extraction_prompt_validation import ExtractionAcceptanceCriteria
from .memory.prompt_components import PromptAdapterRegistry, SemanticPolicyManifest
from .memory_systems.mem0_flat import (
    MEM0_FLAT_EXTRACTION_SLOT_ID,
    Mem0FlatPromptAdapter,
    Mem0FlatSemanticPolicy,
)


EXTRACTION_PREFLIGHT_CONFIG_SCHEMA_VERSION = 1


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("extraction experiment config cannot be read") from exc
    if not isinstance(value, dict):
        raise ValueError("extraction experiment config must be an object")
    return value


def _read_yaml(path: Path, name: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"{name} cannot be read") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _exact(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields are incomplete or unknown")
    return value


def load_extraction_preflight_config(path: Path) -> dict[str, Any]:
    value = _exact(_read_json(path), {
        "schemaVersion",
        "familyId",
        "taskTemplateGroupId",
        "replicates",
        "requestBudgetId",
        "persistenceProfileId",
        "semanticWriteback",
        "acceptanceCriteria",
    }, "extraction experiment config")
    if value["schemaVersion"] != EXTRACTION_PREFLIGHT_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported extraction experiment config schema")
    for field in (
        "familyId",
        "taskTemplateGroupId",
        "requestBudgetId",
        "persistenceProfileId",
    ):
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError(f"extraction experiment {field} is invalid")
    if type(value["replicates"]) is not int or value["replicates"] < 1:
        raise ValueError("extraction experiment replicates must be positive")
    writeback = _exact(
        value["semanticWriteback"], {"timeoutSeconds", "maxOutputTokens"},
        "semantic writeback config",
    )
    if (
        not isinstance(writeback["timeoutSeconds"], (int, float))
        or isinstance(writeback["timeoutSeconds"], bool)
        or writeback["timeoutSeconds"] <= 0
        or type(writeback["maxOutputTokens"]) is not int
        or writeback["maxOutputTokens"] < 1
    ):
        raise ValueError("semantic writeback budget is invalid")
    acceptance = _exact(value["acceptanceCriteria"], {
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
    }, "extraction acceptance criteria")
    ExtractionAcceptanceCriteria(
        minimum_matched_pairs=acceptance["minimumMatchedPairs"],
        minimum_resolved_examples=acceptance["minimumResolvedExamples"],
        minimum_useful_rate_delta=acceptance["minimumUsefulRateDelta"],
        maximum_harmful_rate_delta=acceptance["maximumHarmfulRateDelta"],
        minimum_coverage_ratio=acceptance["minimumCoverageRatio"],
        maximum_empty_rate=acceptance["maximumEmptyRate"],
        maximum_missed_rate_delta=acceptance["maximumMissedRateDelta"],
        required_metrics=tuple(acceptance["requiredMetrics"]),
        proposal_budget_id=acceptance["proposalBudgetId"],
        maximum_proposal_generations=acceptance["maximumProposalGenerations"],
        maximum_candidate_selections=acceptance["maximumCandidateSelections"],
    )
    return value


def _acceptance_criteria(config: dict[str, Any]) -> ExtractionAcceptanceCriteria:
    value = config["acceptanceCriteria"]
    return ExtractionAcceptanceCriteria(
        minimum_matched_pairs=value["minimumMatchedPairs"],
        minimum_resolved_examples=value["minimumResolvedExamples"],
        minimum_useful_rate_delta=value["minimumUsefulRateDelta"],
        maximum_harmful_rate_delta=value["maximumHarmfulRateDelta"],
        minimum_coverage_ratio=value["minimumCoverageRatio"],
        maximum_empty_rate=value["maximumEmptyRate"],
        maximum_missed_rate_delta=value["maximumMissedRateDelta"],
        required_metrics=tuple(value["requiredMetrics"]),
        proposal_budget_id=value["proposalBudgetId"],
        maximum_proposal_generations=value["maximumProposalGenerations"],
        maximum_candidate_selections=value["maximumCandidateSelections"],
    )


def resolved_plain_parent_policy() -> SemanticPolicyManifest:
    adapter = Mem0FlatPromptAdapter()
    registry = PromptAdapterRegistry()
    registry.register(adapter)
    artifact = registry.root_artifact(MEM0_FLAT_EXTRACTION_SLOT_ID)
    binding = registry.bind(MEM0_FLAT_EXTRACTION_SLOT_ID, artifact)
    policy = Mem0FlatSemanticPolicy(
        object(),
        fact_prompt=adapter.bound_template(binding),
        extraction_binding=binding,
    )
    return policy.semantic_manifest


def resolved_task_template_profile(family_root: Path) -> dict[str, Any]:
    root = family_root.expanduser().resolve()
    if not (root / "family.yaml").is_file():
        raise ValueError("PAST family root has no family.yaml")
    files = tuple(sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
    ))
    if not files:
        raise ValueError("PAST family has no task template files")
    file_identities = [{
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    } for path in files]
    family = _read_yaml(root / "family.yaml", "PAST family config")
    episode_order = family.get("episode_order")
    if not isinstance(episode_order, list) or not episode_order:
        raise ValueError("PAST family episode order is missing")
    task_budgets = []
    for episode in episode_order:
        if not isinstance(episode, str) or not episode:
            raise ValueError("PAST episode identity is invalid")
        task = _read_yaml(root / episode / "task.yaml", "PAST task config")
        environment = task.get("environment")
        task_id = task.get("task_id")
        max_turns = environment.get("max_turns") if isinstance(environment, dict) else None
        timeout = environment.get("timeout_seconds") if isinstance(environment, dict) else None
        if (
            not isinstance(task_id, str)
            or not task_id
            or type(max_turns) is not int
            or max_turns < 1
            or type(timeout) is not int
            or timeout < 1
        ):
            raise ValueError("PAST task budget is incomplete")
        task_budgets.append({
            "episode": episode,
            "taskId": task_id,
            "maxTurns": max_turns,
            "timeoutSeconds": timeout,
        })
    return {
        "taskManifestDigest": _digest(file_identities),
        "fileCount": len(file_identities),
        "tasks": task_budgets,
    }


def resolved_model_profile(
    agent_registry_path: Path,
    run_config_path: Path,
    *,
    agent: str,
    semantic_policy: SemanticPolicyManifest,
    semantic_writeback: dict[str, Any],
) -> dict[str, Any]:
    registry = _read_yaml(agent_registry_path, "agent registry")
    agents = registry.get("agents")
    profile = agents.get(agent) if isinstance(agents, dict) else None
    model = profile.get("default_model") if isinstance(profile, dict) else None
    if not isinstance(model, dict):
        raise ValueError("agent default model profile is missing")
    run = _read_yaml(run_config_path, "PAST run config")
    runtime = run.get("runtime")
    judge = run.get("judge")
    if not isinstance(runtime, dict) or not isinstance(judge, dict):
        raise ValueError("PAST runtime or judge config is missing")
    model_id = model.get("model_id")
    base_url = model.get("base_url")
    temperature = runtime.get("temperature")
    if (
        not isinstance(model_id, str)
        or not model_id
        or not isinstance(base_url, str)
        or not base_url
        or not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
    ):
        raise ValueError("resolved model profile is incomplete")
    if runtime.get("mode") != "local" or judge.get("enabled") is not False:
        raise ValueError("formal extraction feedback requires local runtime and disabled judge")
    return {
        "agentProfile": agent,
        "modelId": model_id,
        "providerBaseUrl": base_url,
        "temperature": float(temperature),
        "runtime": "local",
        "judge": "disabled",
        "semanticIngestionProfileId": semantic_policy.model_profile,
        "semanticWritebackTimeoutSeconds": float(semantic_writeback["timeoutSeconds"]),
        "semanticWritebackMaxOutputTokens": semantic_writeback["maxOutputTokens"],
    }


def initialize_formal_feedback_batch(
    *,
    manifest_path: Path,
    batch_registry_path: Path,
    batch_id: str,
    rsimem_root: Path,
    past_bench_root: Path,
    family_root: Path,
    agent_registry_path: Path,
    run_config_path: Path,
    experiment_config_path: Path,
    agent: str = "hermes-luna",
) -> str:
    config = load_extraction_preflight_config(experiment_config_path)
    if family_root.expanduser().resolve().name != config["familyId"]:
        raise ValueError("PAST family and extraction experiment config disagree")
    task_profile = resolved_task_template_profile(family_root)
    parent = resolved_plain_parent_policy()
    model_profile = resolved_model_profile(
        agent_registry_path,
        run_config_path,
        agent=agent,
        semantic_policy=parent,
        semantic_writeback=config["semanticWriteback"],
    )
    request_budget = {
        "taskManifestDigest": task_profile["taskManifestDigest"],
        "tasks": task_profile["tasks"],
        "semanticWriteback": config["semanticWriteback"],
    }
    persistence = {
        "strategy": "per_attempt_trace_directory",
        "homeSeedState": "empty",
        "crossAttemptReuse": False,
    }
    contract = default_feedback_contract_registry().resolver(
        config["familyId"]
    ).contract
    rsimem_revision = resolve_clean_repository(rsimem_root)
    past_bench_revision = resolve_clean_repository(past_bench_root)
    manifest_arguments = {
        "registry_path": batch_registry_path,
        "batch_id": batch_id,
        "phase": "feedback",
        "replicates": config["replicates"],
        "family_id": config["familyId"],
        "task_template_group_id": config["taskTemplateGroupId"],
        "task_manifest_digest": task_profile["taskManifestDigest"],
        "parent_policy": parent,
        "active_policy": parent,
        "matched_policy": None,
        "feedback_contract": contract,
        "acceptance_criteria": _acceptance_criteria(config),
        "model_profile_id": parent.model_profile,
        "resolved_model_profile": model_profile,
        "request_budget_id": config["requestBudgetId"],
        "resolved_request_budget": request_budget,
        "persistence_profile_id": config["persistenceProfileId"],
        "persistence_profile_digest": _digest(persistence),
        "rsimem_revision": rsimem_revision,
        "past_bench_revision": past_bench_revision,
    }
    build_arguments = {
        key: value
        for key, value in manifest_arguments.items()
        if key != "registry_path"
    }
    if manifest_path.expanduser().resolve().exists():
        return resume_extraction_batch_manifest(
            manifest_path,
            registry_path=batch_registry_path,
            expected=build_extraction_batch_manifest(**build_arguments),
        )
    return initialize_extraction_batch_manifest(
        manifest_path,
        **manifest_arguments,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register a clean formal extraction feedback batch",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-registry", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--rsimem-root", type=Path, required=True)
    parser.add_argument("--past-bench-root", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--agent-registry", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--agent", default="hermes-luna")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment_id = initialize_formal_feedback_batch(
        manifest_path=args.manifest,
        batch_registry_path=args.batch_registry,
        batch_id=args.batch_id,
        rsimem_root=args.rsimem_root,
        past_bench_root=args.past_bench_root,
        family_root=args.family_root,
        agent_registry_path=args.agent_registry,
        run_config_path=args.run_config,
        experiment_config_path=args.experiment_config,
        agent=args.agent,
    )
    print(json.dumps({
        "batchId": args.batch_id,
        "experimentId": experiment_id,
        "manifest": str(args.manifest.expanduser().resolve()),
    }, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
