# RSIMem Architecture

The package is being separated into explicit ownership layers:

```text
experiments/  ->  rsimem (memory, lifecycle, hosts, benchmarks, evaluation)
```

`experiments/` contains paper protocols and runners. `memory/` and
`lifecycle/` are host-neutral domain code. `hosts/` integrates Hermes and other
execution hosts. `benchmarks/` adapts PAST-Bench fixtures. `evaluation/`
records usage, audits, and reports without changing policy or memory state.
`cli/` only parses arguments and dispatches to application entry points.

During migration, old `rsimem.<module>` imports remain supported as deprecated
forwarding shims. No task fixture, provider configuration, manifest schema, or
historical output is changed by the structural migration.
