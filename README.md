# RSIMem

RSIMem is the experiment and evaluation repository for LightRSI's memory-mediated recursive self-improvement research.

It studies whether an agent can improve future behavior by recursively updating its memory policy while accounting for the full lifecycle cost of context, memory, controller, and downstream execution.

The initial evaluation uses [PAST-Bench](https://github.com/Gen-Verse/PAST-Bench), an interactive benchmark with ordered cross-session tasks, real tools, sandbox execution, and matched persistence controls.

See [`docs/progress.md`](docs/progress.md) for current status and next tasks, [`docs/implementation_handoff_checklist.md`](docs/implementation_handoff_checklist.md) for the complete serial implementation and acceptance checklist, [`docs/dataset_selection.md`](docs/dataset_selection.md) for the benchmark rationale, [`docs/experiment_plan.md`](docs/experiment_plan.md) for the staged evaluation plan, [`docs/memory_adapters.md`](docs/memory_adapters.md) for the typed backend and compiler architecture, [`docs/lifecycle_controller.md`](docs/lifecycle_controller.md) for the context evaluation control plane, [`docs/usage_accounting.md`](docs/usage_accounting.md) for the request-level accounting contract, and [`docs/smoke_20260820.md`](docs/smoke_20260820.md) for the first end-to-end Hermes/GPT-Luna smoke report.

## Local Setup

RSIMem uses a dedicated Python 3.11 environment. Benchmark implementations live under `benchmarks/` so experiment-specific instrumentation can be versioned with the paper code while preserving each benchmark's upstream license and attribution.

```bash
uv venv --python 3.11 --seed .venv
uv pip install --python .venv/bin/python -e "./benchmarks/past-bench[mock,sandbox,dev]" -e ./benchmarks/past-bench/agents/hermes-agent -e .
```

The initial GPT-Luna smoke uses `gpt-5.6-luna` through the provider's Responses-compatible endpoint. Hermes defaults to reasoning effort `medium`; PAST-Bench keeps its runtime temperature at `0.0`.

```bash
export GPT_LUNA_API_KEY=...
./scripts/run_luna_smoke.sh
```

The smoke runs the native-persistence and no-persistence variants of `memory_ability/SM01_preference_adoption`. It disables the LLM judge because the GPT-Luna provider's Chat Completions route is not yet part of the verified setup; agent traces and deterministic grading evidence are still written under `outputs/smoke/`.

After a successful paired run, the launcher derives `ledger.jsonl` from PAST-Bench traces and Hermes artifacts. Episode-local RSIMem runtime evidence is discovered only beside comparison-owned traces and joined through strict experiment identity checks. The ledger records evidence-backed model calls, token buckets, tools, memory operations, model-visible memory injections, storage snapshots, latency, and outcomes without retaining memory text. Every physical model request exposed by the Hermes runtime receives a sanitized `model_call_usage` event; provider fields that are unavailable remain explicit `null` values rather than inferred zeros.

The launcher then runs `rsimem-audit` and writes `audit.json`. The audit fails the run when request events do not reconcile with `TraceEnd`, billing identities cannot be deduplicated, usage is incomplete, or the ledger contains memory text, absolute source paths, or credential-shaped values. Existing runs can be checked independently with `rsimem-audit outputs/smoke/<run>`.

The opt-in matched launcher runs the same SM01 family under `native`,
`native+ledger`, and `native+adapter+ledger`, preserving the paired persistence
control and raw resource evidence for every run:

```bash
export GPT_LUNA_API_KEY=...
./scripts/run_luna_rsimem_matched.sh
```

It defaults to one independent replicate. Set `RSIMEM_REPLICATES=3` only when
the corresponding model cost is intended. The current runtime does not expose
a provider seed, which is recorded explicitly in `batch_manifest.json`; these
runs must not be described as deterministic seeded executions.

The first live infrastructure replicate, its failed provider attempt, clean
retry, raw resource totals, and interpretation limits are recorded in
[`docs/matched_20260827.md`](docs/matched_20260827.md).

## Verification

Run the RSIMem checks from the repository root:

```bash
.venv/bin/python -m compileall -q src
.venv/bin/pytest -q tests
.venv/bin/pip check
```

Run the vendored PAST-Bench suite from its own directory so its top-level
`agent` package is resolved consistently:

```bash
cd benchmarks/past-bench
../../.venv/bin/pytest -q
```

Do not substitute `pytest -q benchmarks/past-bench` from the RSIMem root; that
layout can collect a conflicting `agent.evolve_controller` module.
