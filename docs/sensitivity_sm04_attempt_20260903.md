# SM04 Semantic Pilot Attempt - 2026-09-03

This document records an infrastructure-failed attempt for
`SM04_rule_migration`, replicate 1. It is retained for provider and runner
diagnostics and is excluded from all sensitivity, mechanism, and quality
denominators. No candidate policy, optimizer input, or N+1 update was
produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: five registered semantic conditions
- Replicate: `1`
- Batch: `semantic-sm04-r01-retry3-20260902_210655`
- Judge: disabled
- Method boundary: opaque sensitivity case ID only

Two earlier batches failed during the provider probe and executed no task. The
retry-3 probe passed and all five launcher commands returned code `0`, but exit
code is not sufficient for acceptance.

## Audit Result

The content-free sensitivity audit returned `ok=false` with `usage_incomplete`.

| Condition | Traces | Memory events | Requests | Retries | Audit |
| --- | ---: | ---: | ---: | ---: | --- |
| no_persistence | 1 | 0 | 3 | 0 | pass |
| native_static | 3 | 28 | 13 | 1 | `usage_incomplete` |
| type_matched_oracle | 2 | 20 | 8 | 0 | pass |
| shortcut_current_input | 1 | 0 | 4 | 0 | pass |
| wrong_mechanism | 1 | 0 | 9 | 2 | `usage_incomplete` |

The logs contain provider connection/read timeouts and long fallback retries.
The affected traces remain available for diagnosis, but the batch is not a
valid pilot because usage accounting is incomplete.

## Disposition

This attempt must not be used for panel sensitivity, process-signal census,
paired deltas, or resource comparisons. A future retry requires a new batch
identity, fresh isolated state, and a fresh completion probe. Raw resource
fields remain audit data only and are not policy rewards.
