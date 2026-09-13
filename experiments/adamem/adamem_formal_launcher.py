"""Freeze and run the serial full-suite AdaMem formal protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .adamem_batch_launcher import run_replicate_batch
from .adamem_experiment import AdaMemCondition
from .adamem_launcher import run_trajectory
from .adamem_experiment import AdaMemRunSpec


FORMAL_SCHEMA = "rsimem-adamem-full-suite-formal-manifest-v1"
CONDITION_ORDER = tuple(AdaMemCondition)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def build_formal_manifest(*, screening_manifest: Path, screening_result: Path,
                          output_root: Path, repo_root: Path) -> dict[str, object]:
    manifest = _load(screening_manifest)
    result = _load(screening_result)
    if not isinstance(manifest, Mapping) or manifest.get("schema") != "rsimem-adamem-full-suite-screening-manifest-v1":
        raise ValueError("invalid screening manifest")
    if not isinstance(result, Mapping) or result.get("manifest_digest") != manifest.get("manifest_digest"):
        raise ValueError("screening result does not bind to manifest")
    records = {str(item.get("family_id")): item for item in manifest.get("records", ()) if isinstance(item, Mapping)}
    accepted = {
        str(item.get("family_id")) for item in result.get("results", ())
        if isinstance(item, Mapping) and item.get("status") == "accepted"
        and item.get("screening_topology") == "single_full_sequence"
    }
    precovered = {
        str(item.get("family_id")) for item in result.get("results", ())
        if isinstance(item, Mapping) and item.get("status") == "precovered"
    }
    if len(accepted) != 25 or precovered != {"SM01_preference_adoption"}:
        raise ValueError("formal suite requires 25 accepted families plus precovered SM01")
    families: list[dict[str, object]] = []
    for family_id in manifest.get("execution_order", ()):
        record = records.get(str(family_id))
        if record is None:
            raise ValueError(f"manifest record missing for {family_id}")
        families.append({
            "family_id": family_id,
            "config": record["config"],
            "config_digest": record["config_digest"],
            "cutover_label": record["cutover_label"],
            "category": record["category"],
            "screening_status": "precovered" if family_id in precovered else "accepted",
        })
    payload: dict[str, object] = {
        "schema": FORMAL_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(Path(output_root).resolve()),
        "repo_root": str(Path(repo_root).resolve()),
        "screening_manifest_digest": manifest["manifest_digest"],
        "screening_result_digest": _digest(screening_result),
        "base_model": "gpt-5.6-luna",
        "meta_agent_model": "gpt-5.6-luna",
        "temperature": 0.0,
        "replicate_count": 3,
        "condition_order": [condition.value for condition in CONDITION_ORDER],
        "family_count": len(families),
        "families": families,
    }
    payload["formal_manifest_digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-manifest", type=Path, required=True)
    parser.add_argument("--screening-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--write-manifest", type=Path)
    args = parser.parse_args(argv)
    payload = build_formal_manifest(
        screening_manifest=args.screening_manifest.resolve(),
        screening_result=args.screening_result.resolve(),
        output_root=args.output_root.resolve(), repo_root=args.repo_root.resolve(),
    )
    target = (args.write_manifest or args.output_root / "formal_manifest.json").resolve()
    _write(target, payload)
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
