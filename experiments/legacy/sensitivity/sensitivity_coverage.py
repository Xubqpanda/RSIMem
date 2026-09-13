"""Content-free aggregation of Stage 3 sensitivity pilot audits.

The aggregator consumes only pilot plans, manifests, and the structured audit
outputs produced by :mod:`experiments.legacy.sensitivity.sensitivity_pilot_audit`. It deliberately
rejects benchmark score, grader, answer, and raw trace fields at the boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .sensitivity import SensitivityCondition, SensitivityPanel
from rsimem.memory.family_matrix import FamilyRole, PastFamilyMatrix


COVERAGE_SCHEMA = "rsimem-sensitivity-coverage-v1"
COVERAGE_SCHEMA_VERSION = 1
_FORBIDDEN = frozenset({
    "score", "scores", "official_score", "grader", "answer", "answer_key",
    "expectation", "hidden_expectation", "judge", "prompt", "response",
})


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _assert_content_free(value: object, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"coverage input has a non-string key: {path}")
            if key.lower() in _FORBIDDEN:
                raise ValueError(f"coverage input contains forbidden field: {path}.{key}")
            _assert_content_free(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_content_free(child, path=f"{path}[{index}]")
    elif isinstance(value, str) and any(token in value.lower() for token in ("grader", "answer_key", "hidden_expectation")):
        raise ValueError(f"coverage input contains forbidden marker: {path}")


def _load_audit(path: Path) -> dict[str, Any]:
    value = _read_object(path)
    _assert_content_free(value, path=str(path))
    if value.get("schema") != "rsimem-sensitivity-pilot-audit-v1":
        raise ValueError(f"unsupported sensitivity audit schema: {path}")
    if not isinstance(value.get("pilot_id"), str) or not isinstance(value.get("runs"), list):
        raise ValueError(f"malformed sensitivity audit: {path}")
    return value


def _load_plan(root: Path) -> dict[str, Any]:
    manifest_path = root / "sensitivity_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"sensitivity manifest is missing: {root}")
    manifest = _read_object(manifest_path)
    _assert_content_free(manifest, path=str(manifest_path))
    value = _read_object(root / "sensitivity_pilot_plan.json")
    _assert_content_free(value, path=str(root / "sensitivity_pilot_plan.json"))
    required = {"pilot_id", "family_id", "panel", "replicate", "condition_order", "run_ids"}
    if set(value) - required - {"schema", "schema_version", "command_digests"} or not required <= set(value):
        raise ValueError(f"malformed sensitivity pilot plan: {root}")
    if not isinstance(value["pilot_id"], str) or not isinstance(value["family_id"], str):
        raise ValueError(f"malformed sensitivity pilot identity: {root}")
    panel = SensitivityPanel(value["panel"])
    if type(value["replicate"]) is not int or value["replicate"] < 1:
        raise ValueError(f"malformed sensitivity replicate: {root}")
    conditions = tuple(SensitivityCondition(item) for item in value["condition_order"])
    if set(conditions) != set(SensitivityCondition) or len(conditions) != len(SensitivityCondition):
        raise ValueError(f"sensitivity pilot does not contain exactly five conditions: {root}")
    run_ids = value["run_ids"]
    if not isinstance(run_ids, list) or len(run_ids) != len(conditions) or not all(isinstance(item, str) for item in run_ids):
        raise ValueError(f"malformed sensitivity run IDs: {root}")
    manifest_runs = manifest.get("runs")
    if not isinstance(manifest_runs, list):
        raise ValueError(f"sensitivity manifest runs are missing: {root}")
    manifest_by_id = {
        item.get("run_id"): item
        for item in manifest_runs
        if isinstance(item, Mapping) and isinstance(item.get("run_id"), str)
    }
    if not set(run_ids).issubset(set(manifest_by_id)):
        raise ValueError(f"sensitivity plan/manifest run set mismatch: {root}")
    for condition, run_id in zip(conditions, run_ids, strict=True):
        item = manifest_by_id[run_id]
        if (
            item.get("family_id") != value["family_id"]
            or item.get("panel") != panel.value
            or item.get("replicate") != value["replicate"]
            or item.get("condition") != condition.value
        ):
            raise ValueError(f"sensitivity plan/manifest identity mismatch: {root}")
    return {**value, "panel": panel.value, "condition_order": [item.value for item in conditions]}


def _candidate_roots(output_root: Path) -> Iterable[Path]:
    for path in sorted(output_root.glob("**/sensitivity_pilot_plan.json")):
        yield path.parent


def aggregate_sensitivity_coverage(output_root: Path) -> dict[str, object]:
    """Aggregate all audit files beneath ``output_root`` without task content."""

    root = Path(output_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("sensitivity output root is invalid")
    records: list[dict[str, object]] = []
    seen_pilots: set[str] = set()
    for pilot_root in _candidate_roots(root):
        audit_path = pilot_root / "audit.json"
        if not audit_path.is_file():
            # Early pilots used the explicit schema name before the shorter
            # ``audit.json`` convention was standardized.
            audit_path = pilot_root / "sensitivity_pilot_audit.json"
        if not audit_path.is_file():
            continue
        plan = _load_plan(pilot_root)
        audit = _load_audit(audit_path)
        if audit["pilot_id"] != plan["pilot_id"]:
            raise ValueError(f"pilot/audit identity mismatch: {pilot_root}")
        if plan["pilot_id"] in seen_pilots:
            raise ValueError(f"duplicate pilot identity: {plan['pilot_id']}")
        seen_pilots.add(plan["pilot_id"])
        audit_runs = {row.get("run_id"): row for row in audit["runs"] if isinstance(row, dict)}
        if set(audit_runs) != set(plan["run_ids"]):
            raise ValueError(f"pilot/audit run set mismatch: {pilot_root}")
        condition_rows = []
        for condition, run_id in zip(plan["condition_order"], plan["run_ids"], strict=True):
            row = audit_runs[run_id]
            if row.get("condition") != condition or type(row.get("ok")) is not bool:
                raise ValueError(f"pilot/audit condition mismatch: {pilot_root}")
            condition_rows.append({
                "condition": condition,
                "run_id": run_id,
                "ok": row["ok"],
                "trace_count": row.get("trace_count", 0),
                "memory_event_count": row.get("memory_event_count", 0),
                "issues": sorted(str(item) for item in row.get("issues", [])),
            })
        records.append({
            "pilot_id": plan["pilot_id"],
            "family_id": plan["family_id"],
            "panel": plan["panel"],
            "replicate": plan["replicate"],
            "pilot_ok": audit.get("ok") is True,
            "provider_probe_ok": audit.get("provider_probe_ok") is True,
            "conditions": condition_rows,
            "source_digest": _digest({"plan": plan, "audit": audit}),
        })
    records.sort(key=lambda item: (str(item["panel"]), str(item["family_id"]), int(item["replicate"]), str(item["pilot_id"])))
    by_panel: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_panel[str(record["panel"])].append(record)
    panels: dict[str, object] = {}
    family_matrix = PastFamilyMatrix.create_default()
    for panel in SensitivityPanel:
        panel_records = by_panel[panel.value]
        accepted = [record for record in panel_records if record["pilot_ok"]]
        family_ids = sorted({str(record["family_id"]) for record in accepted})
        expected_family_ids = sorted(
            spec.family_id
            for spec in family_matrix.families
            if spec.panel.value == panel.value and spec.role is FamilyRole.TARGET
        )
        condition_coverage = {
            condition.value: sum(
                1 for record in accepted
                if any(row["condition"] == condition.value and row["ok"] for row in record["conditions"])
            )
            for condition in SensitivityCondition
        }
        panels[panel.value] = {
            "accepted_pilot_count": len(accepted),
            "accepted_family_ids": family_ids,
            "expected_family_ids": expected_family_ids,
            "missing_family_ids": sorted(set(expected_family_ids) - set(family_ids)),
            "accepted_replicates": sorted({int(record["replicate"]) for record in accepted}),
            "condition_coverage": condition_coverage,
            "excluded_pilot_count": len(panel_records) - len(accepted),
            "pilot_count": len(panel_records),
            "all_families_covered": set(expected_family_ids).issubset(family_ids),
            "expected_replicates": list(range(1, family_matrix.replicate_count + 1)),
            "replicate_coverage_complete": all(
                set(range(1, family_matrix.replicate_count + 1)).issubset(
                    {
                        int(record["replicate"])
                        for record in accepted
                        if record["family_id"] == family_id
                    }
                )
                for family_id in expected_family_ids
            ),
        }
        panels[panel.value]["ready_for_replicate_analysis"] = bool(
            panels[panel.value]["all_families_covered"]
            and panels[panel.value]["replicate_coverage_complete"]
        )
    identity = {"schema": COVERAGE_SCHEMA, "schema_version": COVERAGE_SCHEMA_VERSION, "panels": panels, "records": records}
    return {"schema": COVERAGE_SCHEMA, "schema_version": COVERAGE_SCHEMA_VERSION, "coverage_id": "sensitivity-coverage." + _digest(identity)[:40], "panels": panels, "records": records}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = aggregate_sensitivity_coverage(args.output_root)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["aggregate_sensitivity_coverage"]
