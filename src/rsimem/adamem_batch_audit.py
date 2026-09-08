"""Fail-closed audit for a completed three-replicate AdaMem batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from .adamem_experiment import AdaMemCondition
from .adamem_launcher import _require_accepted_phase


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"unreadable AdaMem batch evidence: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("AdaMem batch evidence must be an object")
    return value


def audit_batch(batch_root: Path) -> dict[str, object]:
    batch_root = Path(batch_root).expanduser().resolve()
    batch = _load(batch_root / "batch_manifest.json")
    condition = AdaMemCondition(batch.get("condition"))
    run_ids = batch.get("run_ids")
    if not isinstance(run_ids, list) or len(run_ids) != 3 or len(set(run_ids)) != 3:
        raise ValueError("AdaMem batch must contain exactly three unique replicate IDs")
    result = _load(batch_root / "batch_result.json")
    if result.get("accepted") is not True:
        raise ValueError("AdaMem batch result is not accepted")
    outcomes = result.get("outcomes")
    if not isinstance(outcomes, Mapping) or set(outcomes) != set(run_ids):
        raise ValueError("AdaMem batch result does not cover all replicates")
    identities: list[dict[str, object]] = []
    for run_id in run_ids:
        root = batch_root / str(run_id)
        manifest = _load(root / "run_manifest.json")
        if manifest.get("condition") != condition.value:
            raise ValueError("AdaMem replicate condition differs from batch")
        if manifest.get("replicate") not in {1, 2, 3}:
            raise ValueError("AdaMem replicate number is invalid")
        _require_accepted_phase(root / "prefix")
        _require_accepted_phase(root / "suffix")
        if condition is not AdaMemCondition.MEM0_STATIC and (root / "updater_usage.json").exists():
            usage = _load(root / "updater_usage.json")
            if usage.get("usage_complete") is not True:
                raise ValueError("AdaMem updater usage is incomplete")
            for field in ("input_tokens", "output_tokens", "request_count"):
                value = usage.get(field)
                if type(value) is not int or value < 0:
                    raise ValueError("AdaMem updater usage is malformed")
        identities.append(manifest)
    for field in ("state_directory", "trace_directory", "artifact_directory", "mem0_collection"):
        values = [item.get(field) for item in identities]
        if len(set(values)) != 3:
            raise ValueError(f"AdaMem replicate {field} is not isolated")
    ports = [item.get("port_offset") for item in identities]
    if len(set(ports)) != 3:
        raise ValueError("AdaMem replicate ports are not isolated")
    return {
        "schema": "rsimem-adamem-replicate-batch-audit-v1",
        "accepted": True,
        "batch_id": batch.get("batch_id"),
        "condition": condition.value,
        "replicates": [
            {"run_id": item.get("run_id"), "replicate": item.get("replicate"), "outcome": outcomes[item.get("run_id")]}
            for item in identities
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit_batch(args.batch_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


__all__ = ["audit_batch"]


if __name__ == "__main__":
    raise SystemExit(main())
