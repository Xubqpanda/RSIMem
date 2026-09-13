"""Run the post-refactor AdaMem + Mem0 validation matrix."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .adamem_batch_launcher import run_replicate_batch
from .adamem_experiment import AdaMemCondition
from .adamem_validation_audit import audit_matched_family
from .adamem_validation_aggregate import aggregate_validation_batches
from .adamem_validation_manifest import SCHEMA, VALIDATION_CONDITIONS, load, write


def _batch_id(family_id: str, condition: AdaMemCondition) -> str:
    slug = family_id.lower().replace("_", "-")
    return f"{slug}-{condition.value.lower()}-refactor-validation-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _accepted_batch(
    output_root: Path,
    family: Mapping[str, object],
    condition: AdaMemCondition,
    manifest_code_revision: Mapping[str, object],
) -> Path | None:
    prefix = f"{str(family['family_id']).lower().replace('_', '-')}-{condition.value.lower()}-refactor-validation-"
    candidates = sorted(output_root.glob(prefix + "*/batch_result.json"), reverse=True)
    for result_path in candidates:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            batch_root = result_path.parent
            batch = json.loads((batch_root / "batch_manifest.json").read_text(encoding="utf-8"))
            run_ids = batch.get("run_ids", ())
            if result.get("accepted") is not True or not isinstance(run_ids, list) or len(run_ids) != 3:
                continue
            manifests = [json.loads((batch_root / str(run_id) / "run_manifest.json").read_text(encoding="utf-8")) for run_id in run_ids]
            if any(item.get("source_sequence_digest") != family.get("config_digest") for item in manifests):
                continue
            if any(item.get("code_revision") != manifest_code_revision for item in manifests):
                continue
            if any(item.get("condition") != condition.value for item in manifests):
                continue
            return batch_root
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            continue
    return None


def run_validation_suite(
    *, manifest_path: Path, output_root: Path, past_bin: Path, past_root: Path,
    config: Path, registry: Path, api_key: str | None, dry_run: bool = False,
) -> dict[str, object]:
    manifest = load(manifest_path)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported refactor validation manifest")
    progress: dict[str, object] = {
        "schema": "rsimem-adamem-mem0-refactor-validation-progress-v1",
        "manifest_digest": manifest["manifest_digest"],
        "status": "running",
        "families": [],
    }
    write(output_root / "validation_progress.json", progress)
    manifest_code_revision = manifest.get("code_revision")
    if not isinstance(manifest_code_revision, Mapping):
        raise ValueError("validation manifest lacks code revision")
    family_roots: dict[str, dict[AdaMemCondition, Path]] = {}
    for family in manifest["families"]:
        family_id = str(family["family_id"])
        roots: dict[AdaMemCondition, Path] = {}
        conditions = tuple(AdaMemCondition(str(value)) for value in manifest.get("conditions", ()))
        if conditions != VALIDATION_CONDITIONS:
            raise ValueError("validation manifest must contain B0 and B2 in order")
        for condition in conditions:
            root = _accepted_batch(output_root, family, condition, manifest_code_revision) if not dry_run else None
            if root is None:
                batch_id = _batch_id(family_id, condition)
                report = run_replicate_batch(
                    source_sequence=Path(str(family["config"])), family_id=family_id,
                    condition=condition, output_root=output_root, past_bin=past_bin,
                    past_root=past_root, config=config, registry=registry,
                    base_url=str(manifest["base_url"]), api_key=api_key,
                    batch_id=batch_id, cutover_label=str(family["cutover_label"]),
                    dry_run=dry_run,
                    code_revision=manifest_code_revision,
                )
                root = output_root / batch_id
            else:
                report = json.loads((root / "batch_result.json").read_text(encoding="utf-8"))
            if not dry_run:
                from .adamem_batch_audit import audit_batch
                audit = audit_batch(root)
                write(root / "batch_audit.json", audit)
            roots[condition] = root
            if not report.get("accepted"):
                raise ValueError(f"validation batch failed: {family_id}/{condition.value}")
        if not dry_run:
            matched = audit_matched_family(family_id=family_id, batch_roots=roots)
            write(output_root / f"{family_id.lower()}-matched-audit.json", matched)
        family_roots[family_id] = roots
        progress["families"].append({
            "family_id": family_id,
            "conditions": {condition.value: str(root.resolve()) for condition, root in roots.items()},
            "accepted": True,
        })
        write(output_root / "validation_progress.json", progress)
    if not dry_run:
        aggregate = aggregate_validation_batches(family_roots=family_roots)
        write(output_root / "validation_aggregate.json", aggregate)
    progress["status"] = "completed"
    write(output_root / "validation_progress.json", progress)
    return progress


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--past-bin", type=Path, required=True)
    parser.add_argument("--past-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--api-key-env", default="GPT_LUNA_API_KEY")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    import os
    report = run_validation_suite(
        manifest_path=args.manifest.resolve(), output_root=args.output_root.resolve(),
        past_bin=args.past_bin.resolve(), past_root=args.past_root.resolve(),
        config=args.config.resolve(), registry=args.registry.resolve(),
        api_key=os.environ.get(args.api_key_env), dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_validation_suite"]
