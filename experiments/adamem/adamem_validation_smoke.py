"""Provider-free routing/materialization smoke for the B0/B2 validation path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import yaml

from rsimem.memory.surface_policy import MemorySurface, RuntimeSurfacePolicy

from .adamem_batch_launcher import run_replicate_batch
from .adamem_experiment import AdaMemCondition
from .adamem_validation_manifest import load, write


SMOKE_FAMILIES = {
    "SM01_preference_adoption": {"memory"},
    "EP01_prior_case_recall": {"session_search"},
    "PC01_sop_bootstrap_01": {"skill"},
    "PG01_release_decision_followup": {"session_search", "skill"},
}


def _source_routes(path: Path) -> set[str]:
    source = yaml.safe_load(path.read_text(encoding="utf-8"))
    episodes = source.get("episodes") if isinstance(source, Mapping) else None
    if not isinstance(episodes, list):
        raise ValueError("smoke source has no episodes")
    return {str(item["mechanism"]) for item in episodes if isinstance(item, Mapping) and item.get("mechanism")}


def _assert_materialized_batch(root: Path, expected_routes: set[str], condition: AdaMemCondition) -> dict[str, object]:
    batch = json.loads((root / "batch_manifest.json").read_text(encoding="utf-8"))
    if batch.get("condition") != condition.value or batch.get("max_workers") != 3:
        raise ValueError("routing smoke batch identity drift")
    manifests = []
    for run_id in batch.get("run_ids", ()):
        run_root = root / str(run_id)
        manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
        policy = RuntimeSurfacePolicy.from_payload(manifest["surface_policy"])
        if policy.policy_id != "mem0-semantic-v1" or not policy.enabled(MemorySurface.SEMANTIC):
            raise ValueError("routing smoke surface policy drift")
        phase_files = [run_root / "manifests" / "static.yaml"] if condition is AdaMemCondition.MEM0_STATIC else [
            run_root / "manifests" / "prefix.yaml", run_root / "manifests" / "suffix.yaml",
        ]
        observed = set()
        for phase_file in phase_files:
            if not phase_file.is_file():
                raise ValueError("routing smoke phase manifest is missing")
            phase = yaml.safe_load(phase_file.read_text(encoding="utf-8"))
            for episode in phase.get("episodes", ()):
                if isinstance(episode, Mapping) and episode.get("mechanism"):
                    observed.add(str(episode["mechanism"]))
        if not expected_routes.issubset(observed):
            raise ValueError("routing smoke mechanism route was not materialized")
        manifests.append(manifest)
    identities = [tuple(item.get(key) for key in ("state_directory", "trace_directory", "artifact_directory", "port_offset")) for item in manifests]
    if len(identities) != len(set(identities)):
        raise ValueError("routing smoke replicate isolation drift")
    return {"condition": condition.value, "replicates": len(manifests), "routes": sorted(expected_routes)}


def run_routing_smoke(
    *, formal_manifest: Path, output_root: Path, past_bin: Path, past_root: Path,
    config: Path, registry: Path, base_url: str,
) -> dict[str, object]:
    formal = load(formal_manifest)
    records = {str(item["family_id"]): item for item in formal.get("families", ()) if isinstance(item, Mapping)}
    smoke_root = output_root.resolve()
    reports = []
    for index, (family_id, expected_routes) in enumerate(SMOKE_FAMILIES.items(), start=1):
        record = records.get(family_id)
        if record is None:
            raise ValueError(f"routing smoke family missing: {family_id}")
        source = Path(str(record["config"])).resolve()
        if _source_routes(source) != expected_routes:
            raise ValueError(f"routing smoke source route drift: {family_id}")
        family_report = {"family_id": family_id, "conditions": []}
        for condition in (AdaMemCondition.MEM0_STATIC, AdaMemCondition.ADAMEM_FULL_TRAJECTORY):
            root_name = f"{index:02d}-{family_id.lower()}-{condition.value.lower()}"
            run_replicate_batch(
                source_sequence=source, family_id=family_id, condition=condition,
                output_root=smoke_root, past_bin=past_bin, past_root=past_root,
                config=config, registry=registry, base_url=base_url, api_key=None,
                batch_id=root_name, cutover_label=str(record["cutover_label"]),
                dry_run=True,
            )
            family_report["conditions"].append(
                _assert_materialized_batch(smoke_root / root_name, expected_routes, condition)
            )
        reports.append(family_report)
    report = {
        "schema": "rsimem-adamem-mem0-refactor-routing-smoke-v1",
        "accepted": True,
        "provider_free": True,
        "conditions": [AdaMemCondition.MEM0_STATIC.value, AdaMemCondition.ADAMEM_FULL_TRAJECTORY.value],
        "families": reports,
    }
    write(smoke_root / "routing_smoke.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--past-bin", type=Path, required=True)
    parser.add_argument("--past-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_routing_smoke(
        formal_manifest=args.formal_manifest.resolve(), output_root=args.output_root.resolve(),
        past_bin=args.past_bin.resolve(), past_root=args.past_root.resolve(),
        config=args.config.resolve(), registry=args.registry.resolve(), base_url=args.base_url,
    ), ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SMOKE_FAMILIES", "run_routing_smoke"]
