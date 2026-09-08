# PC01 Bootstrap-02 Procedural Pilot Attempt - 2026-09-03

This document records an infrastructure-interrupted replicate-1 attempt for
`PC01_sop_bootstrap_02`. It is retained for runner/provider diagnostics and is
excluded from all sensitivity, mechanism, quality, and resource denominators.
No candidate policy, optimizer input, or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: five registered procedural conditions
- Replicate: `1`
- Batch: `procedural-pc01-02-r01-retry-20260903_200529`
- Judge: disabled
- Method boundary: opaque sensitivity case ID only

The provider probe passed. The `no_persistence`, `native_static`,
`type_matched_oracle`, and `shortcut_current_input` conditions completed with
return code `0`. The `wrong_mechanism` condition started but did not produce a
terminal trace completion before the bounded runner timed out after 30 minutes.

## Audit Result

The content-free sensitivity audit returned `ok=false` with:

- `run_not_completed`
- `sequence_results_missing`

The four completed conditions had complete usage and trace identities. The
unfinished control had no usable sequence result, so the five-condition pilot
cannot be accepted. Its partial traces remain available for diagnosis. A
future retry requires a new batch identity and fresh isolated state.

Raw resource fields remain audit data only and are not policy rewards.
