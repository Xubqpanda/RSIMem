# S1 extraction feedback pilot — 2026-08-29

This is the first real-provider, parent-only feedback batch after the
deterministic/offline feasibility gates.  It is a signal-collection pilot, not
an adaptive effect comparison: no N+1 artifact was activated and no uplift is
claimed.

## Frozen identity

- Batch: `s1-sm01-feedback-20260829`
- Experiment identity: `85696d62392ec10bb13e856e4651ddfdaced61a99229acc3ead82cd1bac01485`
- RSIMem commit: `245ef9341d7e1d0a18ff29e4ccff84f9556cdcf2`
- PAST-Bench commit: `41696746d56383bb21dc9fb560c67cb85040217b`
- Method: `static-extraction-rsimem` (plain parent, `native+ledger`,
  `with_persistence`)
- Replicates: 3, fixed SM01 train manifest and feedback contract

The complete ignored output bundle is under
`outputs/extraction_feedback/hermes_luna/s1-sm01-feedback-20260829/` and
contains the manifest, per-replicate traces, ledger, audit, preparation,
private corpus, and proposal result.

## Result

All three attempts completed with `audit.ok=true`, zero audit issues, zero
privacy leaks, and zero retries.  The batch produced 24 primary feedback
examples:

| label | count |
| --- | ---: |
| useful | 0 |
| harmful | 0 |
| missed | 0 |
| unresolved | 24 |
| censored | 0 |

The extraction preparation audit therefore reports
`optimizerSignalReady=false` and `actionablePrimaryCount=0`.  Running
`rsimem.extraction_proposal` with this corpus performed no provider call and
returned `NO_PROPOSAL`; no candidate artifact was written.  This is a valid
strict-attribution no-signal result, not a negative quality label and not a
provider failure.

## Raw resource usage

| replicate | requests | input tokens | output tokens | reasoning tokens | cache-read tokens | retries | wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 49 | 62,294 | 5,639 | 404 | 4,608 | 0 | 253.39 |
| 2 | 47 | 66,025 | 5,340 | 326 | 4,608 | 0 | 254.84 |
| 3 | 51 | 69,457 | 6,357 | 292 | 4,608 | 0 | 282.38 |

The analyzer reports ingestion usage separately (8 model requests per
replicate), injected characters, storage bytes, and unknown buckets.  No
mixed cost scalar is computed.

## Boundary and next step

The run proves that the real Hermes/PAST-Bench path can produce auditable source
and feedback evidence under the frozen parent configuration.  It does not
provide enough resolved extraction-owned outcome variation to train a proposal.
The next eligible step is to predeclare a family/fixture with explicit
deployment-observable use and outcome evidence (or retain this as the
no-signal pilot) before any adaptive candidate or matched validation run.
