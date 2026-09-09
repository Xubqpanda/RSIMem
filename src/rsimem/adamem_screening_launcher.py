"""Serial B0 screening across the frozen PAST-Bench family pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import yaml

from .adamem_experiment import AdaMemCondition, AdaMemRunSpec
from .adamem_launcher import run_trajectory


PRE_COVERED_FAMILIES = {
    "SM01_preference_adoption": {
        "evidence": "docs/sm01_adamem_trajectory_results_20260908.md",
        "reason": "existing_accepted_formal_batch",
    },
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def build_screening_manifest(config_dir: Path, *, output_root: Path) -> dict[str, object]:
    configs = sorted(Path(config_dir).glob("hermes_self_evolve_v2_*.yaml"))
    records: list[dict[str, object]] = []
    for config in configs:
        source = yaml.safe_load(config.read_text(encoding="utf-8"))
        episodes = source.get("episodes") if isinstance(source, Mapping) else None
        if not isinstance(episodes, list) or not episodes:
            raise ValueError(f"screening config has no episodes: {config}")
        family_ids = {item.get("family_id") for item in episodes if isinstance(item, Mapping)}
        family_ids.discard(None)
        if len(family_ids) != 1:
            raise ValueError(f"screening config has ambiguous family: {config}")
        family_id = next(iter(family_ids))
        cutovers = [item.get("label") for item in episodes if isinstance(item, Mapping) and item.get("bucket") == "learn"]
        if not cutovers or not isinstance(cutovers[0], str):
            raise ValueError(f"screening config has no learn cutover: {config}")
        prefix = str(family_id).split("_", 1)[0].upper()
        category = "semantic-targeted" if prefix.startswith("SM") else "cross-workflow"
        records.append({
            "family_id": family_id, "config": str(config.resolve()),
            "config_digest": _digest(config), "cutover_label": cutovers[0],
            "category": category, "condition": AdaMemCondition.MEM0_STATIC.value,
            "static_policy_version": "adamem-root-v1", "base_model": "gpt-5.6-luna",
        })
    if len(records) != 26 or len({item["family_id"] for item in records}) != 26:
        raise ValueError("screening candidate pool must contain exactly 26 unique families")
    payload = {
        "schema": "rsimem-adamem-full-suite-screening-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(Path(output_root).resolve()),
        "family_count": len(records), "execution_order": [item["family_id"] for item in records],
        "records": records,
    }
    payload["manifest_digest"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def run_screening(*, config_dir: Path, output_root: Path, past_bin: Path, past_root: Path,
                  config: Path, registry: Path, base_url: str, api_key: str | None,
                  manifest_path: Path, stop_on_failure: bool = True,
                  retry_index: int = 1) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "rsimem-adamem-full-suite-screening-manifest-v1":
        raise ValueError("unsupported screening manifest")
    progress_path = output_root / "screening" / "screening_progress.json"
    previous = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    results: list[dict[str, object]] = list(previous.get("results", ())) if isinstance(previous, Mapping) else []
    accepted_families = {
        item.get("family_id") for item in results
        if isinstance(item, Mapping) and item.get("status") == "accepted"
        and item.get("screening_topology") == "single_full_sequence"
    }
    for index, record in enumerate(manifest["records"], 1):
        family_id = str(record["family_id"])
        if family_id in accepted_families:
            continue
        if family_id in PRE_COVERED_FAMILIES:
            results.append({
                "family_id": family_id,
                "status": "precovered",
                **PRE_COVERED_FAMILIES[family_id],
            })
            _write(output_root / "screening" / "screening_progress.json", {
                "manifest_digest": manifest["manifest_digest"], "results": results,
            })
            continue
        family_root = output_root / "screening" / family_id / f"attempt-{retry_index:02d}"
        core = f"{manifest['manifest_digest']}:{family_id}:attempt:{retry_index}"
        run_id = "adamem-screening." + hashlib.sha256(core.encode()).hexdigest()[:40]
        run = AdaMemRunSpec(
            run_id=run_id, condition=AdaMemCondition.MEM0_STATIC, replicate=1,
            state_directory=f"screening/{family_id}/state", trace_directory=f"screening/{family_id}/trace",
            artifact_directory=f"screening/{family_id}/artifacts", mem0_collection="screening-mem0-" + hashlib.sha256(core.encode()).hexdigest()[:24],
        )
        try:
            receipt = run_trajectory(
                source_sequence=Path(record["config"]), run=run, condition=AdaMemCondition.MEM0_STATIC,
                cutover_label=str(record["cutover_label"]), output_root=family_root,
                past_bin=past_bin, past_root=past_root, config=config, registry=registry,
                base_model="gpt-5.6-luna", meta_agent_model="gpt-5.6-luna", base_url=base_url,
                api_key=api_key, update_budget=1, temperature=0.0, port_offset=30000 + index * 100,
            )
            result = {
                "family_id": family_id, "status": "accepted", "attempt_index": retry_index,
                "screening_topology": "single_full_sequence", "receipt": receipt.payload(),
            }
        except Exception as exc:
            result = {"family_id": family_id, "status": "infrastructure_failure", "attempt_index": retry_index, "error_type": type(exc).__name__}
            results.append(result)
            _write(output_root / "screening" / "screening_progress.json", {"manifest_digest": manifest["manifest_digest"], "results": results})
            if stop_on_failure:
                raise
            continue
        results.append(result)
        _write(output_root / "screening" / "screening_progress.json", {"manifest_digest": manifest["manifest_digest"], "results": results})
    report = {"schema": "rsimem-adamem-full-suite-screening-result-v1", "manifest_digest": manifest["manifest_digest"], "results": results}
    _write(output_root / "screening" / "screening_result.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--past-bin", type=Path, required=True)
    parser.add_argument("--past-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--retry-index", type=int, default=1)
    args = parser.parse_args(argv)
    if args.write_manifest:
        _write(args.manifest, build_screening_manifest(args.config_dir, output_root=args.output_root))
        return 0
    report = run_screening(
        config_dir=args.config_dir.resolve(), output_root=args.output_root.resolve(), past_bin=args.past_bin.resolve(),
        past_root=args.past_root.resolve(), config=args.config.resolve(), registry=args.registry.resolve(),
        base_url=args.base_url, api_key=__import__("os").environ.get("GPT_LUNA_API_KEY"),
        manifest_path=args.manifest.resolve(), stop_on_failure=not args.continue_on_failure,
        retry_index=args.retry_index,
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
