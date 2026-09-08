# PC03 Procedural Pilot Attempt - 2026-09-03

This document records an infrastructure-interrupted attempt for
`PC03_latent_rule_induction_01`, replicate 1. It is retained for runner and
provider diagnostics and is excluded from all sensitivity, mechanism, and
quality denominators. No candidate policy, optimizer input, or N+1 update was
produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: five registered procedural conditions
- Replicate: `1`
- Batch: `procedural-pc03-r01-20260903_102233`
- Judge: disabled
- Method boundary: opaque sensitivity case ID only

The provider probe passed and the no-persistence condition completed. The
native-static condition then made no progress for more than 20 minutes after
its last trace update, so the runner was terminated with SIGTERM. Remaining
conditions were never started.

## Audit Result

The content-free sensitivity audit returned `ok=false` with:

- `run_not_completed`
- `sequence_results_missing`

The audit observed one complete no-persistence run and one interrupted
native-static run; no usage was available for the interrupted run or the three
unstarted conditions. The batch therefore cannot be used for panel sensitivity,
process-signal census, paired deltas, or resource comparisons.

## Disposition

The partial traces remain available for diagnosis. A future retry requires a
new batch identity, fresh isolated state, and a fresh completion probe. Raw
resource fields remain audit data only and are not policy rewards.
