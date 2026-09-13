# Refactor Reachability Audit

Status: `completed after canonical migration`
Date: `2026-09-12`

This audit records the post-freeze migration boundary. Historical protocol
implementations remain under `experiments/legacy/`; only their temporary
top-level forwarding modules were removed.

## Legacy Source Mapping

| Protocol group | New implementation path | Former public path |
| --- | --- | --- |
| AdaMem active protocol | `experiments/adamem/` | removed `rsimem.adamem_*` shims |
| Base Memory active protocol | `experiments/base_memory/` | removed `rsimem.base_memory_*` shims |
| Adaptive historical route | `experiments/legacy/adaptive/` | removed `rsimem.adaptive_*` shims |
| Extraction historical route | `experiments/legacy/extraction/` | removed `rsimem.extraction_*` shims; reusable runtime is `rsimem.memory.extraction_validation_runtime` |
| Native-attribution historical route | `experiments/legacy/native/` | removed `rsimem.native_*` shims |
| Sensitivity/static-utility historical route | `experiments/legacy/sensitivity/` | removed the old sensitivity/static-utility shims |

All active callers now import the canonical owners directly. No root-level
experiment, benchmark, Hermes, evaluation, or historical-route shim remains;
the architecture test scans the package for accidental imports of the
repository-level `experiments` package.

## Command Reachability

All `pyproject.toml` console scripts resolve through `rsimem.cli.*` wrappers.
Historical commands are retained through their canonical
`experiments.legacy.*` modules because they have at least one of the following:

- direct RSIMem regression coverage;
- a documented historical/reproducibility reference; or
- an accepted evidence artifact whose replay path uses the command's module.

The old `python -m rsimem.<legacy_module>` paths are no longer supported or
documented. Explicit historical replay uses the canonical module path or the
named `rsimem-*` console command.

## Shared Runtime Exceptions

`rsimem.memory.extraction_validation_runtime` remains a reusable runtime
component consumed by the active Hermes host boundary. The AdaMem policy
adapter is owned by the concrete semantic system at
`rsimem.memory_systems.semantic.mem0_flat.adamem_adapter`; there is no
root-level forwarding facade for it.

## Verification

- Legacy focused tests remain under the canonical `experiments.legacy.*`
  imports.
- `tests/test_architecture_boundaries.py` now verifies canonical ownership and
  absence of root-level compatibility shims.
- No output, task fixture, grader, provider configuration, or historical
  evidence artifact was modified during the migration.
