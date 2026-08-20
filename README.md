# RSIMem

RSIMem is the experiment and evaluation repository for LightRSI's memory-mediated recursive self-improvement research.

It studies whether an agent can improve future behavior by recursively updating its memory policy while accounting for the full lifecycle cost of context, memory, controller, and downstream execution.

The initial evaluation uses [PAST-Bench](https://github.com/Gen-Verse/PAST-Bench), an interactive benchmark with ordered cross-session tasks, real tools, sandbox execution, and matched persistence controls.

See [`docs/dataset_selection.md`](docs/dataset_selection.md) for the benchmark rationale, [`docs/experiment_plan.md`](docs/experiment_plan.md) for the staged evaluation plan, [`docs/usage_accounting.md`](docs/usage_accounting.md) for the request-level accounting contract, and [`docs/smoke_20260820.md`](docs/smoke_20260820.md) for the first end-to-end Hermes/GPT-Luna smoke report.

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

After a successful paired run, the launcher derives `ledger.jsonl` from PAST-Bench traces and Hermes artifacts. The ledger records evidence-backed model calls, token buckets, tools, memory operations, model-visible memory injections, storage snapshots, latency, and outcomes without retaining memory text. Every physical model request exposed by the Hermes runtime receives a sanitized `model_call_usage` event; provider fields that are unavailable remain explicit `null` values rather than inferred zeros.
