"""Frozen manifest contract for the post-refactor AdaMem/Mem0 validation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from rsimem.memory.surface_policy import RuntimeSurfacePolicy

from .adamem_experiment import AdaMemCondition


SCHEMA = "rsimem-adamem-mem0-refactor-validation-manifest-v3"
VALIDATION_FAMILIES = (
    "EP01_prior_case_recall",
    "EP02_exception_list_recall",
    "EP03_recall_then_modify",
    "PC01_sop_bootstrap_01",
    "PC01_sop_bootstrap_02",
    "PC01_sop_bootstrap_03",
    "PC01_sop_bootstrap_04",
    "PC01_sop_bootstrap_05",
    "PC01_sop_bootstrap_06",
    "PC02_sop_patch_01",
    "PC02_sop_patch_02",
    "PC03_latent_rule_induction_01",
    "PC04_failure_to_rule_01",
    "PG01_release_decision_followup",
    "PG02_ops_exception_desk",
    "PG03_oncall_handoff_lookup",
    "PG04_temporary_waiver_audit",
    "PG05_change_freeze_followup",
    "PG06_kappa_integration_review",
    "SM01_preference_adoption",
    "SM02_constraint_retention",
    "SM03_fact_correction",
    "SM04_rule_migration",
    "SM05_weak_trigger_preference_adoption",
    "SM06_temporary_exception_pollution",
    "SM07_scoped_rule_migration",
)
VALIDATION_CONDITIONS = (
    AdaMemCondition.MEM0_STATIC,
    AdaMemCondition.ADAMEM_FULL_TRAJECTORY,
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"validation manifest must be an object: {path}")
    return value


def build_validation_manifest(
    *, source_formal_manifest: Path, output_root: Path, code_revision: Mapping[str, object],
    base_url: str, family_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    source = load(source_formal_manifest)
    if source.get("schema") != "rsimem-adamem-full-suite-formal-manifest-v1":
        raise ValueError("validation source must be the frozen AdaMem full-suite manifest")
    if source.get("base_model") != "gpt-5.6-luna" or source.get("replicate_count") != 3:
        raise ValueError("validation source model or replicate identity drifted")
    records = {
        str(item["family_id"]): item
        for item in source.get("families", ())
        if isinstance(item, Mapping) and isinstance(item.get("family_id"), str)
    }
    source_family_ids = tuple(
        str(item["family_id"])
        for item in source.get("families", ())
        if isinstance(item, Mapping) and isinstance(item.get("family_id"), str)
    )
    if len(source_family_ids) != 26 or len(set(source_family_ids)) != 26:
        raise ValueError("validation source must contain 26 unique families")
    if set(source_family_ids) != set(VALIDATION_FAMILIES):
        raise ValueError("validation source must match the frozen 26-family set")
    selected_family_ids = tuple(family_ids) if family_ids is not None else source_family_ids
    if len(selected_family_ids) != 26 or len(set(selected_family_ids)) != 26:
        raise ValueError("full validation requires exactly 26 unique families")
    if set(selected_family_ids) != set(VALIDATION_FAMILIES):
        raise ValueError("validation family selection must match the frozen 26-family set")
    families = []
    for family_id in selected_family_ids:
        record = records[family_id]
        config = Path(str(record["config"])).resolve()
        if not config.is_file() or _file_digest(config) != record.get("config_digest"):
            raise ValueError(f"validation config digest drift: {family_id}")
        families.append({
            "family_id": family_id,
            "config": str(config),
            "config_digest": record["config_digest"],
            "cutover_label": record["cutover_label"],
            "category": record["category"],
        })
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_formal_manifest": str(source_formal_manifest.resolve()),
        "source_formal_manifest_digest": _file_digest(source_formal_manifest),
        "source_formal_protocol_digest": source["formal_manifest_digest"],
        "output_root": str(output_root.resolve()),
        "base_url": base_url,
        "base_model": "gpt-5.6-luna",
        "meta_agent_model": "gpt-5.6-luna",
        "temperature": 0.0,
        "token_budget": 4096,
        "update_budget": 1,
        "replicate_count": 3,
        "condition_order": [condition.value for condition in VALIDATION_CONDITIONS],
        "conditions": [condition.value for condition in VALIDATION_CONDITIONS],
        "mem0_backend": "mem0-flat-hermes-v1",
        "policy_update_space": "versioned_semantic_extraction_policy_only",
        "surface_policy": RuntimeSurfacePolicy.from_hermes_flags(
            memory_enabled=True, user_profile_enabled=True,
            skills_enabled=True, session_search_enabled=True,
            policy_id="mem0-semantic-v1",
        ).payload(),
        "code_revision": dict(code_revision),
        "family_count": len(families),
        "families": families,
    }
    payload["manifest_digest"] = _digest(payload)
    return payload


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "SCHEMA", "VALIDATION_CONDITIONS", "VALIDATION_FAMILIES",
    "build_validation_manifest", "load", "write",
]
