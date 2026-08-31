"""Register a clean parent/candidate extraction matched-validation batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .extraction_experiment_manifest import (
    initialize_extraction_batch_manifest,
    resolve_clean_repository,
)
from .extraction_experiment_preflight import (
    _acceptance_criteria,
    _digest,
    load_extraction_preflight_config,
    resolved_model_profile,
    resolved_task_template_profile,
)
from .extraction_validation_runtime import load_extraction_matched_trial_profile
from .extraction_split_plan import ExtractionSplitRole, load_extraction_split_plan
from .memory.extraction_feedback import default_feedback_contract_registry
from .memory.revocation import JsonRevocationRegistry
from .memory.prompt_components import MatchedSemanticPolicyManifest
from .memory_systems.mem0_flat import (
    MEM0_FLAT_EXTRACTION_SLOT_ID,
    Mem0FlatPromptAdapter,
    Mem0FlatSemanticPolicy,
)


def _semantic_policy(artifact):
    adapter = Mem0FlatPromptAdapter()
    binding = adapter.bind_policy_artifact(
        MEM0_FLAT_EXTRACTION_SLOT_ID,
        artifact,
    )
    return Mem0FlatSemanticPolicy(
        object(),
        fact_prompt=adapter.bound_template(binding),
        extraction_binding=binding,
    ).semantic_manifest


def initialize_formal_matched_validation_batch(
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
    trial_config_path: Path,
    split_plan_path: Path,
    revocation_registry_path: Path | None = None,
    agent: str = "hermes-luna",
) -> str:
    config = load_extraction_preflight_config(experiment_config_path)
    if family_root.expanduser().resolve().name != config["familyId"]:
        raise ValueError("PAST family and extraction experiment config disagree")
    if revocation_registry_path is None:
        raise ValueError("formal matched validation requires a revocation registry")
    revocation_registry = (
        JsonRevocationRegistry(revocation_registry_path)
    )
    trial = load_extraction_matched_trial_profile(
        trial_config_path,
        revocation_registry=revocation_registry,
        require_revocation_registry=True,
    )
    parent_policy = _semantic_policy(trial.parent)
    candidate_policy = _semantic_policy(trial.candidate)
    matched = MatchedSemanticPolicyManifest.create(parent_policy, candidate_policy)
    task_profile = resolved_task_template_profile(family_root)
    load_extraction_split_plan(split_plan_path).assignment_for(
        role=ExtractionSplitRole.VALIDATION,
        family_id=config["familyId"],
        task_template_group_id=config["taskTemplateGroupId"],
        task_manifest_digest=task_profile["taskManifestDigest"],
    )
    model_profile = resolved_model_profile(
        agent_registry_path,
        run_config_path,
        agent=agent,
        semantic_policy=parent_policy,
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
    feedback_contract = default_feedback_contract_registry().resolver(
        config["familyId"]
    ).contract
    return initialize_extraction_batch_manifest(
        manifest_path,
        registry_path=batch_registry_path,
        batch_id=batch_id,
        phase="validation",
        replicates=config["replicates"],
        family_id=config["familyId"],
        task_template_group_id=config["taskTemplateGroupId"],
        task_manifest_digest=task_profile["taskManifestDigest"],
        parent_policy=parent_policy,
        active_policy=candidate_policy,
        matched_policy=matched,
        feedback_contract=feedback_contract,
        acceptance_criteria=_acceptance_criteria(config),
        model_profile_id=parent_policy.model_profile,
        resolved_model_profile=model_profile,
        request_budget_id=config["requestBudgetId"],
        resolved_request_budget=request_budget,
        persistence_profile_id=config["persistenceProfileId"],
        persistence_profile_digest=_digest(persistence),
        rsimem_revision=resolve_clean_repository(rsimem_root),
        past_bench_revision=resolve_clean_repository(past_bench_root),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-registry", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--rsimem-root", type=Path, required=True)
    parser.add_argument("--past-bench-root", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--agent-registry", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--trial-config", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument(
        "--revocation-registry",
        type=Path,
        required=True,
        help="owner-controlled revocation registry",
    )
    parser.add_argument("--agent", default="hermes-luna")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment_id = initialize_formal_matched_validation_batch(
        manifest_path=args.manifest,
        batch_registry_path=args.batch_registry,
        batch_id=args.batch_id,
        rsimem_root=args.rsimem_root,
        past_bench_root=args.past_bench_root,
        family_root=args.family_root,
        agent_registry_path=args.agent_registry,
        run_config_path=args.run_config,
        experiment_config_path=args.experiment_config,
        trial_config_path=args.trial_config,
        split_plan_path=args.split_plan,
        revocation_registry_path=args.revocation_registry,
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
