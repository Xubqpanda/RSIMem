# Documentation Status Inventory

Status: `current`  
Date: 2026-09-08

This inventory classifies every top-level file in `docs/` without moving or
deleting evidence. The active entrypoints are intentionally small; a file not
listed as current is not an active AdaMem experiment instruction.

## Current

| Files | Status | Role |
| --- | --- | --- |
| `README.md`, `implementation_handoff_checklist.md`, `progress.md` | `current` | Entry, active protocol, and current execution state. |
| `adamem_adapter_audit_20260908.md` | `reference` | Frozen upstream/adaptation boundary for the current protocol. |
| `document_status_inventory_20260908.md` | `current` | This classification index. |

## Reference

The following are reusable contracts rather than experiment entrypoints:

| Files | Status |
| --- | --- |
| `dataset_selection.md`, `memory_adapters.md`, `usage_accounting.md`, `lifecycle_controller.md` | `reference` |

## Historical Plans

These files retain earlier routes, phases, or design context. They must not be
read as requirements for `adamem-trajectory-baseline-v1`.

| Filename or family | Status |
| --- | --- |
| `experiment_plan.md`, `lifecycle_implementation_plan.md`, `current_checkpoint_20260901.md`, `case_analysis.md`, `provider_probe.md` | `historical` |
| `research_protocol_v1.json`, `native_attribution_repair_protocol_v1.json` | `historical` |
| `phase1_acceptance_*.md`, `phase2k_*.md`, `matched_*.md`, `policy_feasibility_*.md` | `historical` |
| `static_*.md`, `smoke_*.md`, `extraction_*.md` | `historical` |

## Generated Evidence

The following filename families are dated, immutable reports or manifests. They
remain at their existing paths and are not active protocol definitions:

| Filename family | Status |
| --- | --- |
| `baseline_manifest*.json`, `asset_inventory_*.md` | `generated evidence` |
| `sensitivity_*.md` | `generated evidence` |
| `stage0*.md`, `stage1_*.md`, `stage3_*.md` | `generated evidence` |
| `extraction_stage*.md` | `generated evidence` |

## Coverage Check

The following command lists any top-level document not covered by the rules
above. It must produce no output when this inventory is updated:

```bash
find docs -maxdepth 1 -type f -printf '%f\n' | sort | awk '
  $0 == "README.md" || $0 == "implementation_handoff_checklist.md" ||
  $0 == "progress.md" || $0 == "adamem_adapter_audit_20260908.md" ||
  $0 == "document_status_inventory_20260908.md" { next }
  $0 ~ /^(dataset_selection|memory_adapters|usage_accounting|lifecycle_controller)\.md$/ { next }
  $0 ~ /^(experiment_plan|lifecycle_implementation_plan|current_checkpoint_20260901|case_analysis|provider_probe)\.md$/ { next }
  $0 ~ /^(research_protocol_v1|native_attribution_repair_protocol_v1)\.json$/ { next }
  $0 ~ /^(phase1_acceptance|phase2k_|matched_|policy_feasibility_|static_|smoke_|extraction_)/ { next }
  $0 ~ /^(baseline_manifest|asset_inventory_|sensitivity_|stage0|stage1_|stage3_|extraction_stage)/ { next }
  { print }
'
```

The check is intentionally conservative: a new document is unclassified until
someone explicitly assigns it a role here.

## Redundancy Audit

The 2026-09-08 audit checked repository references for unreferenced top-level
documents. Unreferenced files are all dated sensitivity, extraction, native
attribution, baseline, or smoke evidence; they are retained because they carry
accepted/excluded attempts or reproducible historical measurements. No
unreferenced draft was identified that could be deleted without removing
research evidence, so this pass performs no deletion.
