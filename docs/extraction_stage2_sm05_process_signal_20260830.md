# Stage 2 process-signal census: SM05 — 2026-08-30

This is a clean fixed-parent SM05 process-observability run.  It is not a
parent/candidate effect experiment and it is not benchmark-quality evidence.
The signal census uses only deployment-visible process evidence; grader
output, answer keys, hidden expectations, official scores, and resource cost
are excluded from policy decisions.  Resource usage below is raw accounting
only.

## Scope and provenance

The batch used the `native+ledger` Hermes path, static semantic extraction,
judge disabled, and three replicates.  The provider was the configured
OpenAI-compatible endpoint with model `gpt-5.6-luna`.  The manifest records
RSIMem revision `8b4422e132ba4a56bdaacb54372dc8d18fd3281e` and PAST-Bench
revision `75d898a9f8dc9c05a3809b746acb08da052f0b78`; these identify the clean
recorded run and do not override the current checkout.

| replicate | status | process events | model requests | input tokens | output tokens | wall seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| r01 | completed, audit clean | 175 | 60 | 88,946 | 5,719 | 266.75 |
| r02 | completed, audit clean | 165 | 56 | 71,524 | 4,701 | 242.56 |
| r03 | completed, audit clean | 155 | 52 | 59,825 | 4,472 | 219.29 |

All three audits report `ok=true`, no issues, no projection mismatches, and
no native bypasses.  The different event/request counts are retained as raw
run variation, not treated as policy outcomes.

## Pure-process evidence

The merged pure-process corpus contains 495 events:

| kind | count |
| --- | ---: |
| trigger / source selection / extraction / admission / commit | 21 each |
| retrieval | 48 |
| exposure | 72 |
| task outcome | 42 |
| tool call / tool result | 114 each |

Status counts are `executed=156`, `failed=147`, `pending=78`, `skipped=51`,
`success=51`, and `unknown=12`.  Reason counts are
`decision_observed=294`, `tool_failure=114`, `retrieval_miss=33`,
`task_completed=30`, `absence=18`, and `non_use=6`.

Tool closures are one-to-one (114 calls and 114 results).  The 114 failed tool
results and 33 retrieval misses are recorded process state; neither is
attributed to extraction.  The six `non_use` exposure observations likewise
do not establish that a persisted memory was used or harmful.

## Logical-case replay census

The persisted `process_signal_cases.jsonl` files contain 30 physical
observations and 10 logical cases, with no replicate-status conflicts:

```text
physical observations: 30
logical cases:          10
conflicts:               0
status:                 observable_only = 8, censored = 2
```

Trigger, source selection, and extraction decisions are all shadow `pending`
observations.  Admission and commit are fixed `executed` observations.
Exposure contains a fixed pending/skipped pattern with some `non_use`, while
retrieval and tool events do not provide a complete extraction-owned chain.
In particular, the corpus has no case satisfying:

```text
executed extraction
-> persisted artifact identity
-> retrieval
-> exposure/injection
-> memory-specific use
-> observable outcome
```

The two censored cases remain censored; they are not converted into missed or
harmful labels.

## Audit plane and gate

The separate benchmark-audit projection reports nine completed source tasks,
36 eligible opportunities, 12 audit-only `missed` labels, and 24 audit-only
`unresolved` labels.  It reports no resolved useful or harmful examples.  The
audit labels are retained for diagnosing the attribution contract only and do
not enter the pure-process learner.

The extraction activation funnel is empty: `eligible`, `renderedNPlus1`,
`changedExtraction`, `changedArtifact`, `futureExposure`,
`attributableUse`, and `attributableOutcome` are all zero.  The claim gate is
therefore `operationAttributedExtractionAdaptation=false` with reason
`activation_funnel_incomplete`.

The correct stage-2 decision is:

```text
STOP_NO_SIGNAL
```

Do not generate an Extraction N+1 candidate, held-out matched validation, an
adaptive effect batch, or a six-layer joint intervention from this run.

## Next permitted step

Keep the fixed parent path and instrumentation.  A future signal-ready case
must predeclare an executed extraction action, a deterministic artifact or
artifact-set identity, and a later retrieval → exposure → memory-specific
use/outcome join.  Only after that chain is replay-stable and abstractable
without SM05 values may one Extraction N+1 candidate be proposed; the other
five layers remain closed or shadow-only.

