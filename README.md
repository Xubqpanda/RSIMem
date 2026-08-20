# RSIMem

RSIMem is the experiment and evaluation repository for LightRSI's memory-mediated recursive self-improvement research.

It studies whether an agent can improve future behavior by recursively updating its memory policy while accounting for the full lifecycle cost of context, memory, controller, and downstream execution.

The initial evaluation uses [PAST-Bench](https://github.com/Gen-Verse/PAST-Bench), an interactive benchmark with ordered cross-session tasks, real tools, sandbox execution, and matched persistence controls.

See [`docs/dataset_selection.md`](docs/dataset_selection.md) for the benchmark rationale and [`docs/experiment_plan.md`](docs/experiment_plan.md) for the staged evaluation plan.
