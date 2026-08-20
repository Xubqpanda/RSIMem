# Request-Level Usage Accounting

## Objective

Lifecycle cost must be derived from the physical model requests that produced it. Episode totals alone cannot distinguish task execution from memory estimation, compression, writeback, reflection, retries, or subagent work, and they cannot prove that shared executions were counted once.

RSIMem therefore records one sanitized `model_call_usage` event for every model request exposed by the Hermes runtime. Tool HTTP calls remain separate `tool_dispatch` evidence because they do not share model-token semantics.

## Event Contract

Each model-call event records:

- stable trace-local `call_id` and monotonic `sequence`;
- `component` and `purpose` attribution;
- provider, model, and API mode;
- logical retry `attempt` and terminal status;
- input, output, cache-read, cache-write, and reasoning token buckets;
- usage availability, duration, HTTP status, and error category;
- no prompt, response, request body, error text, Authorization header, or API key.

Unavailable provider fields are `null`. A failed request without provider usage evidence has `usage_available=false`; its token fields are not interpreted as proof of zero billing. Provider prices are not embedded in raw events and are joined later from a versioned pricing table.

## Aggregation

`TraceEnd` aggregates the request events and stores model request count, retry count, token buckets, and a completeness flag. PAST-Bench propagates these quantities into each episode's `token_usage`. RSIMem then emits both the request-level events and one episode-level `model_usage` summary in `ledger.jsonl`.

The legacy `total_tokens` field remains input plus output for benchmark compatibility. Cache and reasoning buckets are retained separately so paper accounting can define its own lifecycle total without changing PAST-Bench grading semantics.

Shared cold episodes carry the same trace-level billing execution identity in both comparison branches. Downstream cost aggregation must deduplicate that identity rather than summing both variant views.

## Hermes Coverage

The recorder covers:

- primary agent turns and provider retries;
- memory flush and context-compression calls;
- iteration-limit summary calls;
- delegated subagents;
- background memory or skill review;
- synchronous and asynchronous Hermes auxiliary-client calls;
- smart approval and mixture-of-agents calls that bypass the auxiliary helper.

Any future component that sends a model request must use the same runtime record contract or an accounting-aware request executor. Tests must reject new unaccounted direct SDK call sites.

## Live Probe

The contract was verified on 2026-08-20 with one `SM01_COLD_001` GPT-Luna episode. The trace contained four successful request events and no retries:

| Metric | Request-event sum | `TraceEnd` |
|---|---:|---:|
| Input tokens | 7,035 | 7,035 |
| Output tokens | 228 | 228 |
| Cache-read tokens | 0 | 0 |
| Cache-write tokens | 0 | 0 |
| Reasoning tokens | 46 | 46 |
| Model requests | 4 | 4 |

`model_usage_complete` was true. A credential scan found no API key or Authorization value in the trace. This probe validates accounting plumbing only and is not a benchmark result.
