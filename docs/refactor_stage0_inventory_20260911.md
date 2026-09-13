# Refactor Stage 0 Inventory

Status: `frozen before source migration`
Date: `2026-09-11`

This inventory records the repository state before the first production-module
move. It includes the existing user changes in the working tree; those changes
are not part of the structural migration and must be preserved.

## Repository State

| Item | Value |
| --- | --- |
| HEAD | `e1830e3afa33fad92b5cf9ad1921aba5961d9005` |
| HEAD subject | `refactor: establish package architecture boundaries` |
| Top-level `src/rsimem/*.py` files | `68` |
| `src/rsimem/memory/**/*.py` files | `73` |
| `src/rsimem/lifecycle/**/*.py` files | `9` |
| Test files | `129` |
| Output root | `outputs/` (`2.3G` at inventory time) |
| Provider-backed batch | none running, according to the active handoff |

Dirty paths present before this inventory:

```text
docs/README.md
docs/document_status_inventory_20260908.md
docs/implementation_handoff_checklist.md
docs/progress.md
src/rsimem/base_memory_experiment.py
src/rsimem/base_memory_launcher.py
tests/test_base_memory_experiment.py
docs/codebase_refactor_checklist.md
docs/current_experiment_results.md
docs/current_goal.md
docs/main_table.md
src/rsimem/base_memory_formal_launcher.py
tests/test_base_memory_formal_launcher.py
```

## Module Ownership Map

| Current paths | Intended owner | State | Notes |
| --- | --- | --- | --- |
| `memory/`, `memory_systems/`, `lifecycle/` | `rsimem` domain | active/reusable | Host-neutral contracts and runtime; `mem0_flat` is the current semantic policy implementation. |
| `adamem_adapter.py`, `adamem_experiment.py`, `adamem_runtime.py`, `adamem_launcher.py`, `adamem_batch_*.py`, `adamem_formal_launcher.py`, `adamem_screening_launcher.py`, `adamem_smoke_audit.py` | `experiments/adamem/` | active experiment | B0/B1/B2 policy-update protocol and runners. |
| `base_memory_experiment.py`, `base_memory_launcher.py`, `base_memory_formal_launcher.py`, `base_memory_smoke_audit.py` | `experiments/base_memory/` | active experiment | NoMemory/HermesNative/Mem0Static baseline. |
| `hermes_host_adapter.py`, `hermes_integration.py`, `hermes_past_bridge.py` | `rsimem/hosts/hermes/` | active integration | Hermes execution, storage, prompt, and lifecycle bridge. |
| `adapter_contracts.py`, `adapter_harness.py`, `past_bench_adapter.py`, `past_runtime_coordinator.py` | `rsimem/benchmarks/past/` | active integration | Host/benchmark translation contracts and terminal binding. |
| `ledger.py`, `audit.py`, `provider_probe.py`, `baseline.py`, `matched_analysis.py`, `static_utility_analysis.py` | `rsimem/evaluation/` | active evaluation | Content-free evidence, usage, audit, and reporting. |
| `preflight.py`, `secret_scan.py` | `rsimem/cli/` plus evaluation owners | active tooling | Existing commands need thin CLI wrappers; ownership must be split before moving. |
| `adaptive_*`, `extraction_*`, `native_*`, `sensitivity_*`, `research_protocol.py`, `oracle_seed_registry.py`, `feedback_preparation.py` | `experiments/legacy/` | historical | Preserve replay/test reachability before any deletion. Several `memory/` support modules are shared by these routes and require a reachability audit. |

## Public Entry Surfaces

The current `pyproject.toml` exposes 25 console scripts. They all point to
top-level `rsimem` modules today; the final layout must route them through
`rsimem.cli` wrappers while preserving command behavior. The current shell
scripts also invoke `python -m rsimem.preflight`,
`rsimem.experiment_manifest`, `rsimem.ledger`, and `rsimem.audit` directly.

Every top-level experiment module with a `main()` is also a potential
`python -m rsimem.<module>` compatibility surface. It must be inventoried
before moving: AdaMem, Base Memory, adaptive, extraction, native attribution,
sensitivity, preflight, evaluation, and protocol modules are all represented.

## Dependency Findings

- `experiments/` already exists with ownership notes, and
  `src/rsimem/{hosts,benchmarks,evaluation,cli}` already exists as package
  skeletons from HEAD. The checklist boxes describing their creation are
  stale; the actual migration is still incomplete.
- `base_memory_launcher.py` imports private `_past_environment` and
  `_require_accepted_phase` from `adamem_launcher.py`. Base Memory migration
  must remove this cross-family dependency before the old module becomes a
  shim.
- `hermes_past_bridge.py` currently imports a wide set of memory, lifecycle,
  host, adapter, and extraction modules. It requires interface splitting, not
  a filename-only move.
- `memory/` contains both reusable runtime code and historical extraction or
  adaptive support. Directory placement alone is not sufficient evidence of
  domain ownership.

## Deterministic Baseline

The following checks were run before source migration:

| Check | Result |
| --- | --- |
| `python -m compileall -q src` | passed |
| `pip check` | passed |
| `pytest -q --tb=short` | `1302 passed, 2 failed` |
| `git diff --check` | passed |

The two baseline test failures were caused by tests reading
`docs/baseline_manifest_20260901.json` after the historical document cleanup
had moved it to `docs/archive/baseline_manifest_20260901.json`. The test path
has been corrected without changing the manifest content.

Frozen formal manifest identities retained for later comparison:

| Artifact | Schema | Manifest digest | File SHA-256 |
| --- | --- | --- | --- |
| `outputs/adamem_full_suite_20260908/formal_manifest.json` | `rsimem-adamem-full-suite-formal-manifest-v1` | `153689a96e7f6245d1771ae5079da76af0fc3332d3e15e48a1e7d8b156835f76` | `c2a349320414c73d856d3ec770ad36a9f723d7970f39f194678b007c3c012001` |
| `outputs/base_memory_formal_20260909/formal_manifest.json` | `rsimem-base-memory-full-suite-formal-manifest-v1` | `34be3c17a3d066a46889c5141e70533c44ad0b7a5280cce08516ee418490f153` | `6a880f0ab819881ad906d4516841253054a18f82fb0d3a2c4dd0ca7080e6be95` |

No output, benchmark fixture, model configuration, or historical evidence was
changed during Stage 0.
