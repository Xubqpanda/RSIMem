# Dataset Selection

## Research requirement

MemBridge needs two different evaluation regimes:

1. A deterministic memory benchmark to validate lifecycle accounting and compare memory backends under matched questions.
2. An interactive agent benchmark to expose externalities such as tool calls, retries, failed runs, and wall-clock time.

No single existing dataset provides both regimes well.

## Phase A: MemBase-native smoke test

### LoCoMo

- 10 long conversational trajectories.
- Multiple dated sessions per trajectory.
- Question categories include single-hop, multi-hop, temporal, open-domain, and adversarial questions.
- Each question contains evidence message IDs, which gives us a useful correctness audit trail.
- MemBase already provides a loader and construction/search/evaluation runners.

LoCoMo is the first smoke test because it has the lowest integration cost and makes the accounting pipeline easy to debug. It does not, by itself, support a strong claim about full agent cost: it has no rich tool-use or retry loop.

### LongMemEval

LongMemEval is the second benchmark. It provides longer interactive histories and a standardized long-term memory QA protocol. We should use it after the LoCoMo ledger is stable, mainly to test whether cost policies transfer to longer contexts.

### RealMem

RealMem is useful for an online construction/retrieval experiment because MemBase can evaluate tasks during memory construction. It should be added after the offline pipeline, since its online environment and model dependencies make debugging harder.

## Phase B: global agent-cost validation

### PAST-Bench

PAST-Bench is the strongest current candidate for the global-cost claim:

- ordered task-family sequences;
- persistence on/off matched controls;
- real tools and sandbox execution;
- task score, mechanism evidence, tokens per episode, and wall time.

It can reveal whether a cheaper memory policy merely shifts cost into model calls, tools, retries, or failed episodes. MemBase does not currently include a PAST-Bench adapter, so this should be a separate integration layer in MemBridge.

The benchmark source is available locally at `Study/PAST-Bench` (Apache-2.0):

```text
https://github.com/Gen-Verse/PAST-Bench
arXiv:2608.04003
```

The suite contains 26 task families and 204 ordered episodes under `self-evolve-tasks-v2/`. Its runtime requires Python 3.11+, Docker, and an LLM API profile, so it is intentionally kept as an external benchmark checkout rather than vendored into this repository.

AppWorld and LifelongAgentBench are possible follow-up environments, but should not block the first implementation.

## Recommended order

```text
LoCoMo (one trajectory)
  -> LoCoMo (full set, multiple memory backends)
  -> LongMemEval
  -> RealMem online evaluation
  -> PAST-Bench global agent-cost evaluation
```

## Local data layout

Raw and processed datasets are intentionally ignored by Git:

```text
data/raw/locomo/locomo10.json
data/raw/longmemeval/longmemeval_s_cleaned.json
data/raw/realmem/...
data/processed/...
```

Download LoCoMo with:

```bash
bash scripts/download_locomo.sh
```

The script is proxy-compatible through the standard `http_proxy` and `https_proxy` environment variables.
