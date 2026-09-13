# Refactor Release Notes

The canonical module paths are now the supported paths after the frozen
AllMemoryOff baseline was completed.

Active experiment modules are under `experiments.adamem` and
`experiments.base_memory`. Historical adaptive, extraction, native-attribution,
and sensitivity modules are under `experiments.legacy.*`. Reusable runtime
owners remain under `rsimem.memory`, `rsimem.hosts.hermes`,
`rsimem.benchmarks.past`, and `rsimem.evaluation`; command entry points are
under `rsimem.cli` and the named `rsimem-*` console scripts.

The former root-level `rsimem.<legacy_module>` forwarding imports and module
CLI paths are removed. They are a breaking change and are not restored for
compatibility. Historical replay must use the canonical module path or the
named console command. Historical source and output artifacts remain intact.
