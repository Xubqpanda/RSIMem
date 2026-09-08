# SM02 process-feedback batch v5 — 2026-08-29

This is a completed real-provider plain-parent process-feedback batch for the
semantic constraint family.  It is not a parent/candidate matched effect run;
no candidate was activated and no uplift is claimed.

## Frozen identity

- Batch: `s1-sm02-feedback-20260829-v5`
- Experiment ID: `58d561b03021c966a19349130b5fcfb7d42929d22071c7f1dfdbc24e97734af5`
- RSIMem revision used by the batch: `aaa4859`
- Family: `SM02_constraint_retention`
- Method: `static-extraction-rsimem`, `native+ledger`, `with_persistence`
- Model: `gpt-5.6-luna`, primary provider endpoint
- Replicates: 3; all attempts completed with clean runtime/audit stages

Raw ignored output is retained under
`outputs/extraction_feedback/hermes_luna/s1-sm02-feedback-20260829-v5/`.

## Feedback and process result

| label | count |
| --- | ---: |
| useful | 0 |
| harmful | 0 |
| missed | 8 |
| unresolved | 16 |
| censored | 0 |

There are 24 eligible opportunities and six completed sources.  Three sources
are non-empty and three are empty (`nonemptyCoverage=0.5`,
`emptyExtractionRate=0.5`).  Strict resolved denominator remains zero because
there is no useful/harmful variation; `resolved_useful_rate` and
high-confidence missed rate are unknown.  The eight missed observations remain
valid contract evidence, while the 16 unresolved observations are not treated
as negative labels.

The canonical process corpus contains 231 content-free events, including
trigger, source-selection, extraction, admission, commit, retrieval, exposure
and task-outcome stages.  Official grader/score data is inaccessible to the
policy path.  Audits report no privacy leaks, schema/safety failures or native
writer bypass.

## Raw usage (per replicate)

| replicate | requests | input tokens | output tokens | ingestion requests | ingestion input | ingestion output | wall seconds | injected chars | peak stored bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 37 | 53,472 | 3,939 | 7 | 16,242 | 757 | 612.57 | 663 | 94,437 |
| 2 | 38 | 54,568 | 4,702 | 7 | 16,488 | 684 | 234.20 | 663 | 98,533 |
| 3 | 36 | 51,737 | 3,709 | 7 | 15,251 | 724 | 169.03 | 663 | 94,437 |

Provider-unreported buckets remain unknown; no mixed cost scalar is computed.

## Optimizer boundary

The content-bearing corpus was reconstructed successfully.  A deterministic
request projection initially exceeded the frozen 160,000-character input
budget because replicated source/evidence text was repeated.  The optimizer
contract now stores repeated source, fact and delayed-evidence content once in
a content catalog and keeps unresolved/censored units as content-free
diagnostic context.  The resulting SM02 request is bounded (about 105k
characters) without silently dropping evidence.

The first two provider-backed proposal attempts returned malformed completion
fields and were rejected by the strict output schema; no candidate artifact was
written.  This is a provider/model-output diagnostic, not a negative task
label.  A future proposal attempt may proceed only with a valid frozen-schema
completion and must remain subject to static safety and held-out validation.
