"""Run one isolated B0/B1/B2 AdaMem trajectory through vendored PAST-Bench."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Mapping

import yaml

from .adamem_adapter import AdaMemPolicy, bind_to_mem0_flat, update_policy
from .adamem_experiment import AdaMemCondition, AdaMemRunSpec, feedback_view_for_condition
from .adamem_runtime import (
    AdaMemPolicyReceipt,
    build_pure_process_feedback,
    materialize_phase_manifest,
    split_family_manifest,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _read_yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("AdaMem source sequence must be a mapping")
    return value


def _trace_paths(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*.jsonl")
        if path.parent.name != "artifacts" and path.name != "ledger.jsonl"
    )


def _operation_paths(root: Path) -> list[Path]:
    return sorted(root.rglob("rsimem_semantic_operations.jsonl"))


def _reflect_via_openai(
    request: Mapping[str, object], *, api_key: str, base_url: str, model: str,
    max_tokens: int, temperature: float,
) -> str:
    body = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You update an AdaMem policy. Return only one JSON object using "
                    "optional fields general_policy, set, and remove. Make a reusable "
                    "policy change only when the deployment-visible feedback supports it; "
                    "otherwise return {}. Do not add task answers or benchmark-specific facts."
                ),
            },
            {"role": "user", "content": json.dumps(request, ensure_ascii=True, sort_keys=True)},
        ],
    }
    endpoint = base_url.rstrip("/") + "/chat/completions"
    transport = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=True).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(transport, timeout=90) as response:
        payload = json.loads(response.read().decode("utf-8"))
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("AdaMem reflection response has no unique choice")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ValueError("AdaMem reflection response has no text content")
    return content


def _past_command(
    *, past_bin: Path, past_root: Path, sequence: Path, trace_dir: Path,
    config: Path, registry: Path, model: str, base_url: str, policy_path: Path | None,
) -> list[str]:
    command = [
        str(past_bin), "evolve", "--sequence", str(sequence), "--agent", "hermes-luna",
        "--runtime", "local", "--sandbox", "--sandbox-tools", "--no-judge",
        "--persistence-variant", "with_persistence", "--config", str(config),
        "--registry", str(registry), "--trace-dir", str(trace_dir), "--model", model,
        "--base-url", base_url, "--rsimem-mode", "native+ledger",
        "--rsimem-semantic-writeback-mode", "static",
    ]
    if policy_path is not None:
        command.extend(["--rsimem-adamem-policy", str(policy_path)])
    return command


def run_trajectory(
    *, source_sequence: Path, run: AdaMemRunSpec, condition: AdaMemCondition,
    cutover_label: str, output_root: Path, past_bin: Path, past_root: Path,
    config: Path, registry: Path, base_model: str, meta_agent_model: str,
    base_url: str, api_key: str | None, update_budget: int, temperature: float,
    dry_run: bool = False,
) -> AdaMemPolicyReceipt:
    """Execute prefix/update/suffix with a policy state transition at the cutover."""

    if update_budget != 1:
        raise ValueError("AdaMem baseline currently permits exactly one update")
    source = _read_yaml(source_sequence)
    episodes = source.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("AdaMem source sequence requires episodes")
    # Temporary phase manifests live under the run artifact root, so retain
    # the original sequence directory as the authority for relative task refs.
    for episode in episodes:
        if not isinstance(episode, dict) or not isinstance(episode.get("task"), str):
            raise ValueError("AdaMem source episode task is invalid")
        task = Path(episode["task"])
        if not task.is_absolute():
            episode["task"] = str((source_sequence.parent / task).resolve())
    split = split_family_manifest(source, cutover_label=cutover_label)
    if split.family_id not in source_sequence.name:
        # The source filename is audit metadata, never a replacement for YAML identity.
        pass
    run_root = output_root / run.run_id
    prefix_root = run_root / "prefix"
    suffix_root = run_root / "suffix"
    policy_root = AdaMemPolicy.root()
    root_file = run_root / "policies" / "policy_root.json"
    _write_json(root_file, policy_root.payload())
    prefix_manifest = materialize_phase_manifest(source, split=split, phase="prefix")
    prefix_manifest_file = run_root / "manifests" / "prefix.yaml"
    prefix_manifest_file.parent.mkdir(parents=True, exist_ok=True)
    prefix_manifest_file.write_text(yaml.safe_dump(prefix_manifest, sort_keys=False), encoding="utf-8")
    _write_json(run_root / "trajectory_split.json", split.payload())

    if not dry_run:
        subprocess.run(
            _past_command(
                past_bin=past_bin, past_root=past_root, sequence=prefix_manifest_file,
                trace_dir=prefix_root, config=config, registry=registry, model=base_model,
                base_url=base_url, policy_path=None,
            ), cwd=past_root, check=True,
        )

    feedback_view = feedback_view_for_condition(condition)
    policy = policy_root
    if feedback_view is None:
        receipt = AdaMemPolicyReceipt.static(split=split, policy=policy_root)
    else:
        if dry_run:
            feedback = {"outcome": "deterministic_dry_run"}
        else:
            feedback = build_pure_process_feedback(
                feedback_view=feedback_view, trace_paths=_trace_paths(prefix_root),
                operation_paths=_operation_paths(prefix_root),
            )
        if api_key is None and not dry_run:
            raise ValueError("AdaMem update requires an API key")
        result = update_policy(
            parent=policy_root, feedback_view=feedback_view, feedback=feedback,
            reflect=lambda request: _reflect_via_openai(
                request, api_key=api_key, base_url=base_url, model=meta_agent_model,
                max_tokens=1024, temperature=temperature,
            ) if not dry_run else "{}",
        )
        policy = result.candidate_policy
        receipt = AdaMemPolicyReceipt.from_update(split=split, result=result)
        _write_json(run_root / "feedback_request.json", {
            "feedback_view": feedback_view.value,
            "feedback_digest": result.feedback_digest,
            "request_digest": result.request_digest,
        })
    policy_file = run_root / "policies" / "policy_active.json"
    _write_json(policy_file, policy.payload())
    binding = bind_to_mem0_flat(policy)
    _write_json(run_root / "policy_binding.json", {
        "policy_version": binding.policy_version,
        "policy_digest": binding.policy_digest,
        "artifact_id": binding.extraction_policy_artifact.artifact_id,
        "artifact_digest": binding.extraction_policy_artifact.artifact_digest,
        "binding_id": binding.binding.binding_id,
    })
    _write_json(run_root / "policy_receipt.json", receipt.payload())

    prefix_home = prefix_root / "family_homes" / split.family_id / "hermes_home"
    suffix_manifest = materialize_phase_manifest(
        source, split=split, phase="suffix", initial_home_fixture_dir=str(prefix_home),
    )
    suffix_manifest_file = run_root / "manifests" / "suffix.yaml"
    suffix_manifest_file.write_text(yaml.safe_dump(suffix_manifest, sort_keys=False), encoding="utf-8")
    if not dry_run:
        subprocess.run(
            _past_command(
                past_bin=past_bin, past_root=past_root, sequence=suffix_manifest_file,
                trace_dir=suffix_root, config=config, registry=registry, model=base_model,
                base_url=base_url,
                policy_path=(None if condition == AdaMemCondition.MEM0_STATIC else policy_file),
            ), cwd=past_root, check=True,
        )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--cutover-label", required=True)
    parser.add_argument("--condition", choices=[item.value for item in AdaMemCondition], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--past-bin", type=Path, required=True)
    parser.add_argument("--past-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--meta-agent-model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="GPT_LUNA_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    condition = AdaMemCondition(args.condition)
    run = AdaMemRunSpec(
        run_id=args.run_id, condition=condition, replicate=1,
        state_directory="runtime-state", trace_directory="runtime-trace",
        artifact_directory="runtime-artifacts", mem0_collection="runtime-mem0",
    )
    receipt = run_trajectory(
        source_sequence=args.sequence.resolve(), run=run, condition=condition,
        cutover_label=args.cutover_label, output_root=args.output_root.resolve(),
        past_bin=args.past_bin.resolve(), past_root=args.past_root.resolve(),
        config=args.config.resolve(), registry=args.registry.resolve(),
        base_model=args.base_model, meta_agent_model=args.meta_agent_model,
        base_url=args.base_url, api_key=os.environ.get(args.api_key_env),
        update_budget=1, temperature=args.temperature, dry_run=args.dry_run,
    )
    print(json.dumps(receipt.payload(), ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
