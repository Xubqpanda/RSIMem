"""Run one condition's three independent AdaMem replicates as a batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Mapping

import yaml

from .adamem_experiment import AdaMemComparisonManifest, AdaMemCondition, AdaMemRunSpec
from .adamem_launcher import run_trajectory


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def build_batch_manifest(
    *, source_sequence: Path, family_id: str, config: Path, registry: Path,
    base_model: str = "gpt-5.6-luna", meta_agent_model: str = "gpt-5.6-luna",
    token_budget: int = 4096, update_budget: int = 1, replicate_count: int = 3,
    temperature: float = 0.0,
) -> AdaMemComparisonManifest:
    """Construct the immutable comparison identity before provider execution."""
    if replicate_count != 3:
        raise ValueError("formal AdaMem batch requires exactly three replicates")
    source = yaml.safe_load(source_sequence.read_text(encoding="utf-8"))
    if not isinstance(source, Mapping) or source.get("family_id") not in {None, family_id}:
        # Generated PAST manifests often carry the family in episode metadata;
        # the explicit family argument remains the owner identity.
        if not isinstance(source, Mapping):
            raise ValueError("AdaMem source sequence is invalid")
    return AdaMemComparisonManifest.create(
        family_id=family_id,
        train_sequence_id=source_sequence.stem + ".train",
        n_plus_one_sequence_id=source_sequence.stem + ".n_plus_one",
        fixture_digest=_digest(source_sequence),
        base_model=base_model,
        meta_agent_model=meta_agent_model,
        token_budget=token_budget,
        update_budget=update_budget,
        replicate_count=replicate_count,
        temperature=temperature,
    )


def run_replicate_batch(
    *, source_sequence: Path, family_id: str, condition: AdaMemCondition,
    output_root: Path, past_bin: Path, past_root: Path, config: Path,
    registry: Path, base_url: str, api_key: str | None, batch_id: str,
    cutover_label: str,
    replicate_count: int = 3, temperature: float = 0.0, dry_run: bool = False,
    max_workers: int = 3,
) -> dict[str, object]:
    """Execute exactly one condition's replicate batch, bounded at three workers."""
    if max_workers != 3:
        raise ValueError("formal AdaMem batch concurrency is fixed at three")
    condition = AdaMemCondition(condition)
    manifest = build_batch_manifest(
        source_sequence=source_sequence, family_id=family_id, config=config,
        registry=registry, replicate_count=replicate_count, temperature=temperature,
    )
    batch_root = output_root / batch_id
    _write(batch_root / "batch_manifest.json", {
        "schema": "rsimem-adamem-replicate-batch-v1",
        "batch_id": batch_id,
        "condition": condition.value,
        "manifest": manifest.payload(),
        "replicate_count": replicate_count,
        "max_workers": max_workers,
        "run_ids": [run.run_id for run in manifest.runs if run.condition is condition],
    })
    runs = tuple(run for run in manifest.runs if run.condition is condition)
    # The manifest is written first; only these three same-condition runs may
    # overlap. A failed future is recorded and causes the batch to fail closed.
    def execute(run: AdaMemRunSpec):
        return run_trajectory(
            source_sequence=source_sequence,
            run=run,
            condition=condition,
            cutover_label=cutover_label,
            output_root=batch_root,
            past_bin=past_bin,
            past_root=past_root,
            config=config,
            registry=registry,
            base_model="gpt-5.6-luna",
            meta_agent_model="gpt-5.6-luna",
            base_url=base_url,
            api_key=api_key,
            update_budget=1,
            temperature=temperature,
            port_offset=20000 + (run.replicate - 1) * 100,
            dry_run=dry_run,
        )

    outcomes: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="adamem-replicate") as pool:
        futures = {pool.submit(execute, run): run for run in runs}
        for future in as_completed(futures):
            run = futures[future]
            try:
                outcomes[run.run_id] = future.result().payload()
            except Exception as exc:
                outcomes[run.run_id] = {"status": "infrastructure_failure", "error_type": type(exc).__name__}
    accepted = len(outcomes) == len(runs) and all(
        isinstance(value, Mapping) and value.get("status") != "infrastructure_failure"
        for value in outcomes.values()
    )
    report = {
        "schema": "rsimem-adamem-replicate-batch-result-v1",
        "batch_id": batch_id,
        "condition": condition.value,
        "accepted": accepted,
        "outcomes": outcomes,
    }
    _write(batch_root / "batch_result.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--cutover-label", required=True)
    parser.add_argument("--condition", choices=[item.value for item in AdaMemCondition], required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--past-bin", type=Path, required=True)
    parser.add_argument("--past-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="GPT_LUNA_API_KEY")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    environment_key = os.environ.get(args.api_key_env)
    report = run_replicate_batch(
        source_sequence=args.sequence.resolve(), family_id=args.family_id,
        condition=AdaMemCondition(args.condition), output_root=args.output_root.resolve(),
        past_bin=args.past_bin.resolve(), past_root=args.past_root.resolve(),
        config=args.config.resolve(), registry=args.registry.resolve(),
        base_url=args.base_url, api_key=environment_key, batch_id=args.batch_id,
        cutover_label=args.cutover_label,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0 if report["accepted"] else 1


__all__ = ["build_batch_manifest", "run_replicate_batch"]


if __name__ == "__main__":
    raise SystemExit(main())
