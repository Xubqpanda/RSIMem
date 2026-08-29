# Stage 2 process-signal census: SM02 final — 2026-08-30

This is the final deterministic process-observability report for the SM02
constraint-retention parent batch. It is a stage-2 signal check, not an effect
experiment and not benchmark-quality evidence. The pure-process census does
not read grader output, answer keys, hidden expectations, official scores, or
resource cost. Raw usage below is accounting only.

## Scope and provenance

The batch ran the fixed `native+ledger` Hermes path with static semantic
extraction, judge disabled, and three replicates. The provider/model were
`https://coding.tu-zi.com/v1` and `gpt-5.6-luna`. The batch manifest records
RSIMem revision `643dce1939db7e58b4dd36f1c60ad7afbc25f935` and PAST-Bench
revision `75d898a9f8dc9c05a3809b746acb08da052f0b78`; these are the revisions of
the recorded run, not a claim about the current checkout.

Replicate 3's first attempt timed out. The manifest retains that failed
infrastructure attempt, and attempt 2 completed successfully; only the
completed attempt is used for the r03 census.

| replicate | selected run | status | process events | audit | model requests | input tokens | output tokens |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| r01 | `..._r01_static_extraction_rsimem` | completed | 125 | clean | 40 | 55,427 | 4,898 |
| r02 | `..._r02_static_extraction_rsimem` | completed | 119 | clean | 37 | 56,362 | 4,213 |
| r03 | `..._r03_static_extraction_rsimem_attempt02` | completed | 117 | clean | 35 | 54,595 | 4,172 |

All three selected audits report `ok=true`, zero issues, zero projection
mismatches, zero native bypasses, zero unresolved memory injections, and nine
traces. The failed r03 attempt is retained only in `batch_manifest.json` as an
infrastructure event.

## Aggregate process evidence

The three completed pure-process corpora contain 361 physical events. Every
event has the expected `family_id=SM02_constraint_retention`, actual stage,
task/session/trace identity, and `variant=with_persistence`.

Event counts:

| kind | count |
| --- | ---: |
| trigger | 18 |
| source selection | 18 |
| extraction | 18 |
| admission | 18 |
| commit | 24 |
| retrieval | 42 |
| exposure | 66 |
| tool call | 65 |
| tool result | 65 |
| task outcome | 27 |

Status counts are `pending=66`, `executed=107`, `skipped=54`, `failed=95`, and
`success=39`. Reason counts are `decision_observed=215`, `tool_failure=65`,
`retrieval_miss=30`, `task_completed=27`, and `absence=24`.

There are 138 policy-bound events (38.23%), 295 receipt-bound events
(81.72%), and 40 distinct source revisions. Tool closures are exact in every
replicate: 65 calls pair one-to-one with 65 results, with no orphan,
duplicate, or scope mismatch. All 65 observed tool results have
`tool_success=false`; this is recorded process state and is not attributed to
semantic extraction.

## Logical-case replay census

The identity `(task_id, stage)` yields nine logical cases and 27 physical
observations (nine cases × three completed replicates). No case has a
replicate-status conflict. For every policy layer, the same case has the same
layer/status pattern in all three replicates; the only repeated variation is
the deterministic exposure split described below.

The six policy-layer observations are:

| layer | observations | observed status/action pattern | process interpretation |
| --- | ---: | --- | --- |
| trigger | 18 | all `pending` shadow decisions | no executed trigger intervention |
| source selection | 18 | all `pending` shadow decisions | no executed source-selection intervention |
| extraction | 18 | all `pending`; extraction receipts present but no executed mutation | no persisted extraction action is attributable |
| admission | 18 | all `executed` | fixed execution, no action variation |
| commit | 24 | all `executed` | fixed execution, no action variation |
| exposure | 66 | 42 policy-bound `pending` and 24 non-policy `skipped` | some exposure observations, but no attributable memory use |

Retrieval has 12 `success` and 30 `failed` (`retrieval_miss`) observations;
exposure is not upgraded to memory use from retrieval alone. The corpus has no
complete and attributable

```text
executed extraction -> persisted artifact identity -> retrieval
-> exposure/injection -> memory-specific use -> observable outcome
```

chain. Task-completion events therefore remain process observations, not
useful/harmful credit for extraction.

## Layer classification and gate

Using process metadata only, the current classification is:

| layer | classification | reason |
| --- | --- | --- |
| Extraction | diagnostic-only / observable-only | extraction remains shadow `pending`; no attributable artifact/use/outcome chain |
| Trigger | validation-only | deterministic shadow decisions with no executed variation |
| Source selection | validation-only | deterministic shadow decisions with no executed variation |
| Admission | validation-only | uniformly executed and fixed; no variation or outcome attribution |
| Commit | validation-only | uniformly executed and fixed; no variation or outcome attribution |
| Exposure | diagnostic-only / validation-only | pending/skipped variation is visible, but injection/use is not attributable |

No layer is optimization-ready. `extraction_analysis.json` independently
records an empty activation funnel (`eligible`, `renderedNPlus1`,
`changedExtraction`, `changedArtifact`, `futureExposure`, `attributableUse`,
and `attributableOutcome` are all zero) and the claim gate
`operationAttributedExtractionAdaptation=false` with reason
`activation_funnel_incomplete`.

The correct stage-2 decision is therefore:

```text
STOP_NO_SIGNAL
```

Do not generate an Extraction N+1 candidate, held-out matched validation, an
adaptive effect batch, or a six-layer joint intervention from this batch.
Pending shadow decisions, unresolved observations, retrieval misses, tool
failures, and not-exposed events must not be relabeled as missed or harmful.

## Next permitted step

Keep this fixed parent path and its process instrumentation. Before opening
Extraction optimization, predeclare a replay-stable case in which an
extraction action executes, a deterministic artifact/mutation identity is
recorded, and a later retrieval → exposure → memory-specific use/outcome chain
is observable. Only after that signal passes the stage-2 gate may one
Extraction N+1 candidate be proposed; Trigger, Admission, Source Selection,
Commit, and Exposure remain closed or shadow-only.
