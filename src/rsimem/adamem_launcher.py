"""Run one isolated B0/B1/B2 AdaMem trajectory through vendored PAST-Bench."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Mapping

import yaml

from .adamem_adapter import AdaMemFeedbackView, AdaMemPolicy, bind_to_mem0_flat, update_policy
from .adamem_experiment import AdaMemCondition, AdaMemRunSpec, feedback_view_for_condition
from .adamem_runtime import (
    AdaMemPolicyReceipt,
    build_pure_process_feedback,
    materialize_phase_manifest,
    materialize_static_screening_manifest,
    split_family_manifest,
)


FROZEN_MODEL_ID = "gpt-5.6-luna"


def _past_environment(*, base_url: str, api_key: str | None) -> dict[str, str]:
    """Build the isolated PAST environment with audited auxiliary routing.

    Hermes' session_search may make a secondary model request.  Keep that
    request on the same OpenAI-compatible endpoint/model as the primary agent
    so its usage is attributable to the trajectory and cannot silently fall
    through to an unrelated provider.
    """
    environment = os.environ.copy()
    if not environment.get("ANTHROPIC_API_KEY"):
        compatibility_key = api_key or environment.get("GPT_LUNA_API_KEY")
        if compatibility_key:
            environment["ANTHROPIC_API_KEY"] = compatibility_key
    environment["AUXILIARY_SESSION_SEARCH_BASE_URL"] = base_url
    environment["AUXILIARY_SESSION_SEARCH_MODEL"] = FROZEN_MODEL_ID
    if api_key:
        environment["AUXILIARY_SESSION_SEARCH_API_KEY"] = api_key
    return environment


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _require_accepted_phase(root: Path) -> None:
    results_path = root / "sequence_results.json"
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        episodes = payload["episodes"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("AdaMem phase has no readable sequence results") from exc
    if not isinstance(episodes, list) or not episodes:
        raise RuntimeError("AdaMem phase has no episodes")
    rejected = []
    for episode in episodes:
        usage = episode.get("token_usage") if isinstance(episode, Mapping) else None
        if not isinstance(usage, Mapping) or usage.get("model_usage_complete") is not True:
            rejected.append(str(episode.get("task_id", "unknown")) if isinstance(episode, Mapping) else "unknown")
    if rejected:
        raise RuntimeError(
            "AdaMem infrastructure failure: incomplete model usage for " + ", ".join(rejected)
        )


def compare_run_manifests(left_path: Path, right_path: Path) -> dict[str, tuple[object, object]]:
    """Return only permitted B0/B1/B2 identity differences or fail closed."""
    try:
        left = json.loads(left_path.read_text(encoding="utf-8"))
        right = json.loads(right_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("AdaMem run manifest cannot be read") from exc
    if not isinstance(left, dict) or not isinstance(right, dict) or set(left) != set(right):
        raise ValueError("AdaMem run manifest shape differs")
    allowed = {"run_id", "condition", "feedback_view", "port_offset"}
    differences = {
        key: (left[key], right[key]) for key in left if left[key] != right[key]
    }
    unexpected = set(differences) - allowed
    if unexpected:
        raise ValueError(
            "AdaMem run identity drift: " + ", ".join(sorted(unexpected))
        )
    return differences


def _reflect_via_openai(
    request: Mapping[str, object], *, api_key: str, base_url: str, model: str,
    max_tokens: int, temperature: float,
) -> tuple[str, dict[str, int | None]]:
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
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, Mapping):
        raise ValueError("AdaMem reflection response lacks usage")

    def integer(*names: str, required: bool = False) -> int | None:
        for name in names:
            value = usage.get(name)
            if type(value) is int and value >= 0:
                return value
        if required:
            raise ValueError("AdaMem reflection usage is incomplete")
        return None

    normalized_usage = {
        "input_tokens": integer("prompt_tokens", "input_tokens", required=True),
        "output_tokens": integer("completion_tokens", "output_tokens", required=True),
        "total_tokens": integer("total_tokens"),
        "cache_read_tokens": integer("prompt_cache_hit_tokens", "cache_read_tokens"),
        "cache_write_tokens": integer("cache_creation_input_tokens", "cache_write_tokens"),
        "reasoning_tokens": integer("reasoning_tokens"),
        "request_count": 1,
        "usage_complete": True,
    }
    return content, normalized_usage


def _past_command(
    *, past_bin: Path, past_root: Path, sequence: Path, trace_dir: Path,
    config: Path, registry: Path, model: str, base_url: str, policy_path: Path | None,
    port_offset: int, state_dir: Path | None = None, hermes_home_dir: Path | None = None,
    artifact_dir: Path | None = None,
) -> list[str]:
    command = [
        str(past_bin), "evolve", "--sequence", str(sequence), "--agent", "hermes",
        "--runtime", "local", "--sandbox", "--sandbox-tools", "--no-judge",
        "--persistence-variant", "with_persistence", "--config", str(config),
        "--registry", str(registry), "--trace-dir", str(trace_dir), "--model", model,
        "--base-url", base_url, "--rsimem-mode", "native+ledger",
        "--rsimem-semantic-writeback-mode", "static",
        "--port-offset", str(port_offset),
    ]
    if policy_path is not None:
        command.extend(["--rsimem-adamem-policy", str(policy_path)])
    isolated = (state_dir, hermes_home_dir, artifact_dir)
    if any(value is not None for value in isolated):
        if any(value is None for value in isolated):
            raise ValueError("AdaMem phase isolation requires state, home, and artifact paths")
        command.extend([
            "--rsimem-state-dir", str(state_dir),
            "--rsimem-hermes-home-dir", str(hermes_home_dir),
            "--rsimem-artifact-dir", str(artifact_dir),
        ])
    return command


def run_trajectory(
    *, source_sequence: Path, run: AdaMemRunSpec, condition: AdaMemCondition,
    cutover_label: str, output_root: Path, past_bin: Path, past_root: Path,
    config: Path, registry: Path, base_model: str, meta_agent_model: str,
    base_url: str, api_key: str | None, update_budget: int, temperature: float,
    port_offset: int = 1000, dry_run: bool = False,
    feedback_view_override: AdaMemFeedbackView | None = None,
    condition_label: str | None = None,
) -> AdaMemPolicyReceipt:
    """Execute prefix/update/suffix with a policy state transition at the cutover."""

    if update_budget != 1:
        raise ValueError("AdaMem baseline currently permits exactly one update")
    if base_model != FROZEN_MODEL_ID or meta_agent_model != FROZEN_MODEL_ID:
        raise ValueError(
            "AdaMem trajectory baseline requires the frozen base/meta model "
            + FROZEN_MODEL_ID
        )
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
    cutover_index = next(
        (index for index, item in enumerate(episodes)
         if isinstance(item, dict) and item.get("label") == cutover_label),
        None,
    )
    post_cutover = episodes[cutover_index + 1:] if cutover_index is not None else []
    has_post_update_learning = any(item.get("bucket") == "learn" for item in post_cutover)
    split = split_family_manifest(
        source, cutover_label=cutover_label,
        require_post_update_learning=(
            condition is not AdaMemCondition.MEM0_STATIC and has_post_update_learning
        ),
    )
    if split.family_id not in source_sequence.name:
        # The source filename is audit metadata, never a replacement for YAML identity.
        pass
    run_root = output_root / run.run_id
    prefix_root = run_root / "prefix"
    suffix_root = run_root / "suffix"
    policy_root = AdaMemPolicy.root()
    root_file = run_root / "policies" / "policy_root.json"
    _write_json(root_file, policy_root.payload())
    _write_json(run_root / "trajectory_split.json", split.payload())
    _write_json(run_root / "run_manifest.json", {
        "schema": "rsimem-adamem-run-manifest-v1",
        "protocol_id": "adamem-trajectory-baseline-v1",
        "run_id": run.run_id,
        "condition": condition_label or condition.value,
        "feedback_view": (
            feedback_view_for_condition(condition).value
            if feedback_view_for_condition(condition) is not None else None
        ),
        "source_sequence_digest": _file_digest(source_sequence),
        "split_id": split.split_id,
        "base_model": base_model,
        "meta_agent_model": meta_agent_model,
        "temperature": temperature,
        "update_budget": update_budget,
        "policy_update_space": "versioned_semantic_extraction_policy_only",
        "mem0_backend": "mem0-flat-hermes-v1",
        "past_bin_digest": _file_digest(past_bin),
        "config_digest": _file_digest(config),
        "registry_digest": _file_digest(registry),
        "port_offset": port_offset,
        "replicate": run.replicate,
        "state_directory": run.state_directory,
        "trace_directory": run.trace_directory,
        "artifact_directory": run.artifact_directory,
        "mem0_collection": run.mem0_collection,
    })

    if condition is AdaMemCondition.MEM0_STATIC:
        # B0 has no update boundary.  Preserve the family's original history
        # anchors by executing its complete sequence in one PAST process.
        static_manifest_file = run_root / "manifests" / "static.yaml"
        static_manifest_file.parent.mkdir(parents=True, exist_ok=True)
        static_manifest_file.write_text(
            yaml.safe_dump(materialize_static_screening_manifest(source), sort_keys=False),
            encoding="utf-8",
        )
        if not dry_run:
            environment = _past_environment(base_url=base_url, api_key=api_key)
            subprocess.run(
                _past_command(
                    past_bin=past_bin, past_root=past_root, sequence=static_manifest_file,
                    trace_dir=run_root / "static", config=config, registry=registry,
                    model=base_model, base_url=base_url, policy_path=None,
                    port_offset=port_offset,
                    state_dir=run_root / run.state_directory / "static",
                    hermes_home_dir=run_root / "hermes_home" / "static",
                    artifact_dir=run_root / run.artifact_directory / "static",
                ), cwd=past_root, check=True, env=environment,
            )
            _require_accepted_phase(run_root / "static")
        receipt = AdaMemPolicyReceipt.static(split=split, policy=policy_root)
        _write_json(run_root / "policy_receipt.json", receipt.payload())
        return receipt

    prefix_manifest = materialize_phase_manifest(source, split=split, phase="prefix")
    prefix_manifest_file = run_root / "manifests" / "prefix.yaml"
    prefix_manifest_file.parent.mkdir(parents=True, exist_ok=True)
    prefix_manifest_file.write_text(yaml.safe_dump(prefix_manifest, sort_keys=False), encoding="utf-8")

    if not dry_run:
        environment = _past_environment(base_url=base_url, api_key=api_key)
        subprocess.run(
            _past_command(
                past_bin=past_bin, past_root=past_root, sequence=prefix_manifest_file,
                trace_dir=prefix_root, config=config, registry=registry, model=base_model,
                base_url=base_url, policy_path=None,
                port_offset=port_offset,
                state_dir=run_root / run.state_directory / "prefix",
                hermes_home_dir=run_root / "hermes_home" / "prefix",
                artifact_dir=run_root / run.artifact_directory / "prefix",
            ), cwd=past_root, check=True, env=environment,
        )
        _require_accepted_phase(prefix_root)

    feedback_view = feedback_view_override or feedback_view_for_condition(condition)
    policy = policy_root
    if feedback_view is None:
        receipt = AdaMemPolicyReceipt.static(split=split, policy=policy_root)
    else:
        if dry_run:
            feedback = (
                {"messages": []}
                if feedback_view is AdaMemFeedbackView.NATIVE
                else {"outcome": "deterministic_dry_run"}
            )
        else:
            feedback = build_pure_process_feedback(
                feedback_view=feedback_view, trace_paths=_trace_paths(prefix_root),
                operation_paths=_operation_paths(prefix_root),
            )
        if api_key is None and not dry_run:
            raise ValueError("AdaMem update requires an API key")
        updater_usage: dict[str, int | None] | None = None

        def reflect(request: Mapping[str, object]) -> str:
            nonlocal updater_usage
            if dry_run:
                return "{}"
            content, updater_usage = _reflect_via_openai(
                request, api_key=api_key, base_url=base_url, model=meta_agent_model,
                max_tokens=1024, temperature=temperature,
            )
            return content

        result = update_policy(
            parent=policy_root, feedback_view=feedback_view, feedback=feedback,
            reflect=reflect,
        )
        policy = result.candidate_policy
        receipt = AdaMemPolicyReceipt.from_update(split=split, result=result)
        _write_json(run_root / "feedback_request.json", {
            "feedback_view": feedback_view.value,
            "feedback_digest": result.feedback_digest,
            "request_digest": result.request_digest,
        })
        if not dry_run:
            if updater_usage is None:
                raise RuntimeError("AdaMem updater did not return complete usage")
            _write_json(run_root / "updater_usage.json", updater_usage)
    policy_file = run_root / "policies" / "policy_active.json"
    _write_json(policy_file, policy.payload())
    policy_updated = receipt.outcome == "updated"
    if not policy_updated:
        _write_json(run_root / "policy_binding.json", {
            "binding_source": "mem0-flat-root",
            "policy_version": "root-v1",
            "adamem_policy_applied": False,
            "reason_code": receipt.reason_code,
        })
    else:
        binding = bind_to_mem0_flat(policy)
        _write_json(run_root / "policy_binding.json", {
            "binding_source": "adamem",
            "policy_version": binding.policy_version,
            "policy_digest": binding.policy_digest,
            "artifact_id": binding.extraction_policy_artifact.artifact_id,
            "artifact_digest": binding.extraction_policy_artifact.artifact_digest,
            "binding_id": binding.binding.binding_id,
            "adamem_policy_applied": True,
        })
    _write_json(run_root / "policy_receipt.json", receipt.payload())

    prefix_home = run_root / "hermes_home" / "prefix"
    suffix_manifest = materialize_phase_manifest(
        source, split=split, phase="suffix", initial_home_fixture_dir=str(prefix_home),
    )
    suffix_manifest_file = run_root / "manifests" / "suffix.yaml"
    suffix_manifest_file.write_text(yaml.safe_dump(suffix_manifest, sort_keys=False), encoding="utf-8")
    if not dry_run:
        environment = _past_environment(base_url=base_url, api_key=api_key)
        subprocess.run(
            _past_command(
                past_bin=past_bin, past_root=past_root, sequence=suffix_manifest_file,
                trace_dir=suffix_root, config=config, registry=registry, model=base_model,
                base_url=base_url,
                policy_path=(policy_file if policy_updated else None),
                port_offset=port_offset,
                state_dir=run_root / run.state_directory / "suffix",
                hermes_home_dir=run_root / "hermes_home" / "suffix",
                artifact_dir=run_root / run.artifact_directory / "suffix",
            ), cwd=past_root, check=True, env=environment,
        )
        _require_accepted_phase(suffix_root)
    return receipt


def run_native_fidelity(
    *, source_sequence: Path, run: AdaMemRunSpec, cutover_label: str,
    output_root: Path, past_bin: Path, past_root: Path, config: Path,
    registry: Path, base_model: str, meta_agent_model: str, base_url: str,
    api_key: str | None, update_budget: int, temperature: float,
    port_offset: int = 1000, dry_run: bool = False,
) -> AdaMemPolicyReceipt:
    """Run the audited AdaMem-native feedback shape as a separate fidelity baseline."""
    return run_trajectory(
        source_sequence=source_sequence, run=run, condition=AdaMemCondition.ADAMEM_TERMINAL,
        cutover_label=cutover_label, output_root=output_root, past_bin=past_bin,
        past_root=past_root, config=config, registry=registry, base_model=base_model,
        meta_agent_model=meta_agent_model, base_url=base_url, api_key=api_key,
        update_budget=update_budget, temperature=temperature, port_offset=port_offset,
        dry_run=dry_run, feedback_view_override=AdaMemFeedbackView.NATIVE,
        condition_label="AdaMem-native",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--cutover-label", required=True)
    parser.add_argument("--condition", choices=[item.value for item in AdaMemCondition] + ["AdaMem-native"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--past-bin", type=Path, required=True)
    parser.add_argument("--past-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--base-model", default=FROZEN_MODEL_ID)
    parser.add_argument("--meta-agent-model", default=FROZEN_MODEL_ID)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="GPT_LUNA_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--port-offset", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    is_native = args.condition == "AdaMem-native"
    condition = AdaMemCondition.ADAMEM_TERMINAL if is_native else AdaMemCondition(args.condition)
    run = AdaMemRunSpec(
        run_id=args.run_id, condition=condition, replicate=1,
        state_directory="runtime-state", trace_directory="runtime-trace",
        artifact_directory="runtime-artifacts", mem0_collection="runtime-mem0",
    )
    common = dict(
        source_sequence=args.sequence.resolve(), run=run,
        cutover_label=args.cutover_label, output_root=args.output_root.resolve(),
        past_bin=args.past_bin.resolve(), past_root=args.past_root.resolve(),
        config=args.config.resolve(), registry=args.registry.resolve(),
        base_model=args.base_model, meta_agent_model=args.meta_agent_model,
        base_url=args.base_url, api_key=os.environ.get(args.api_key_env),
        update_budget=1, temperature=args.temperature, dry_run=args.dry_run,
        port_offset=args.port_offset,
    )
    receipt = (
        run_native_fidelity(**common)
        if is_native else run_trajectory(condition=condition, **common)
    )
    print(json.dumps(receipt.payload(), ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
