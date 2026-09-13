# RSIMem Architecture

The package is being separated into explicit ownership layers:

```text
experiments/  ->  rsimem (memory, lifecycle, hosts, benchmarks, evaluation)
```

`experiments/` contains paper protocols and runners. The active protocols are
`experiments/base_memory/` and `experiments/adamem/`; historical routes are
grouped under `experiments/legacy/`. `memory/` and `lifecycle/` are host-neutral
contracts, runtime primitives, and evidence code. Concrete systems live under
`memory_systems/`, organized by memory kind: `semantic/mem0_flat`,
`semantic/hermes_native`, `episodic/hermes_native`, and
`procedural/hermes_native`; `registry.py` is the capability declaration and
native-registry construction boundary. `hosts/` integrates Hermes and other
execution hosts.
`benchmarks/` adapts PAST-Bench fixtures. `evaluation/` records usage, audits,
and reports without changing policy or memory state. `cli/` only parses
arguments and dispatches to application entry points.

The active package has canonical paths only; old root-level `rsimem.<module>`
imports and module CLI commands are removed rather than maintained as runtime
forwarding shims. Current paths are
`experiments.base_memory.*`, `experiments.adamem.*`, and
`experiments.legacy.<protocol>.*`. No task fixture,
provider configuration, manifest schema, or historical output is changed by
the structural migration.
