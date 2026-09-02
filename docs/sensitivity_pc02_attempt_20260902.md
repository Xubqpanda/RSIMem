# PC02 Procedural Pilot Attempt - 2026-09-02

This document records an infrastructure-failed attempt for
`PC02_sop_patch_01`, replicate 1. It is retained for provider and runner
diagnostics and is excluded from all sensitivity, mechanism, and quality
denominators. No candidate policy, optimizer input, or N+1 update was
produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: five registered procedural conditions
- Replicate: `1`
- Batch: `procedural-pc02-01-r01-20260902_194050`
- Judge: disabled
- Method boundary: opaque sensitivity case ID only

The initial provider completion probe passed. The launcher executed all five
registered commands and each command returned code `0`, but command exit code
alone is insufficient for acceptance.

## Audit Result

The content-free sensitivity audit returned `ok=false` with:

- `usage_incomplete`
- `usage_total_mismatch:retries`

Condition summary:

| Condition | Traces | Memory events | Requests | Retries | Audit |
| --- | ---: | ---: | ---: | ---: | --- |
| no_persistence | 1 | 0 | 7 | 0 | pass |
| native_static | 3 | 11 | 19 | 1 | `usage_incomplete` |
| type_matched_oracle | 2 | 6 | 10 | 0 | pass |
| shortcut_current_input | 1 | 0 | 3 | 3 | `usage_incomplete`, retry mismatch |
| wrong_mechanism | 1 | 0 | 3 | 3 | `usage_incomplete`, retry mismatch |

The failed conditions included provider request timeouts and exhausted
fallback retries. Their traces remain available for diagnosis, but the batch
is not a valid pilot because usage accounting is incomplete.

## Disposition

This attempt must not be used for panel sensitivity, process-signal census,
paired deltas, or resource comparisons. A retry requires a new batch identity,
fresh isolated state, and a fresh completion probe. Raw resource fields remain
audit data only and are not policy rewards.
