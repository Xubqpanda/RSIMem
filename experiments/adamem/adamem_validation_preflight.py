"""Fail-closed provider-free preflight for the full B0/B2 validation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

from rsimem.memory.surface_policy import RuntimeSurfacePolicy

from .adamem_experiment import AdaMemCondition
from .adamem_validation_manifest import SCHEMA, VALIDATION_FAMILIES, load, write


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_manifest(path: Path) -> dict[str, object]:
    manifest = load(path)
    families = manifest.get("families")
    conditions = tuple(manifest.get("conditions", ()))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported full validation manifest schema")
    if manifest.get("family_count") != 26 or not isinstance(families, list) or len(families) != 26:
        raise ValueError("full validation manifest must contain 26 families")
    if conditions != tuple(item.value for item in (AdaMemCondition.MEM0_STATIC, AdaMemCondition.ADAMEM_FULL_TRAJECTORY)):
        raise ValueError("full validation manifest must contain B0 and B2 only")
    if manifest.get("replicate_count") != 3:
        raise ValueError("full validation manifest must require three replicates")
    ids = [item.get("family_id") for item in families if isinstance(item, Mapping)]
    if len(ids) != 26 or len(set(ids)) != 26 or any(not isinstance(item, str) for item in ids):
        raise ValueError("full validation family identities are not unique")
    if set(ids) != set(VALIDATION_FAMILIES):
        raise ValueError("full validation family identities do not match the frozen 26-family set")
    for item in families:
        if not isinstance(item, Mapping):
            raise ValueError("full validation family record is malformed")
        config = Path(str(item.get("config")))
        if not config.is_file() or _digest(config) != item.get("config_digest"):
            raise ValueError(f"family config digest drift: {item.get('family_id')}")
        if not isinstance(item.get("cutover_label"), str) or not item["cutover_label"]:
            raise ValueError(f"missing cutover label: {item.get('family_id')}")
    policy = RuntimeSurfacePolicy.from_payload(manifest["surface_policy"])
    if policy.policy_id != "mem0-semantic-v1":
        raise ValueError("full validation surface policy is not Mem0 semantic")
    return {
        "schema": "rsimem-adamem-mem0-refactor-validation-preflight-v1",
        "accepted": True,
        "manifest": str(path.resolve()),
        "manifest_digest": manifest["manifest_digest"],
        "family_count": 26,
        "family_ids": ids,
        "conditions": list(conditions),
        "replicate_count": 3,
        "surface_policy": policy.payload(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit_manifest(args.manifest.resolve())
    write(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_manifest"]
