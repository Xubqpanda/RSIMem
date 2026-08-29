# SM01 feedback batch v9a — 2026-08-29

This is a completed real-provider, plain-parent feedback batch.  It is a
feedback/corpus pilot, not a parent-vs-candidate matched effect experiment.
No N+1 artifact was activated and no uplift claim is made.

## Frozen identity

- Batch: `s1-sm01-feedback-20260829-v9a`
- Experiment ID: `112cd3749990c7f2e89aa896e05aab2def03b9aa725ba8eca7b923a2d03ffc52`
- RSIMem revision: `1b24226` (clean detached worktree)
- PAST-Bench revision: recorded in `batch_manifest.json`
- Method: `static-extraction-rsimem`, `native+ledger`, `with_persistence`
- Model: `gpt-5.6-luna`, base URL `https://coding.tu-zi.com/v1`
- Replicates: 3; all attempts completed with `audit.ok=true`

The ignored raw bundle is retained under
`outputs/extraction_feedback/hermes_luna/s1-sm01-feedback-20260829-v9a/`.

## Feedback result

| label | count |
| --- | ---: |
| useful | 0 |
| harmful | 0 |
| missed | 0 |
| unresolved | 24 |
| censored | 0 |

The batch has 24 eligible opportunities and six completed sources; three
sources produced non-empty extraction and three produced empty extraction.
Strict resolved denominator is zero, so `resolved_useful_rate` and
high-confidence missed rate are `unknown`, not zero.  The extraction optimizer
corpus preparation succeeded, but `optimizerSignalReady=false` and
`actionablePrimaryCount=0`; proposal generation therefore made zero provider
calls and returned `NO_PROPOSAL`.  No candidate artifact was created.

## Raw usage (per replicate)

| replicate | requests | input tokens | output tokens | ingestion requests | ingestion input | ingestion output | wall seconds | injected chars | peak stored bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 42 | 62,568 | 4,597 | 7 | 20,855 | 610 | 215.41 | 636 | 94,428 |
| 2 | 47 | 72,768 | 4,815 | 7 | 21,893 | 690 | 331.38 | 735 | 102,653 |
| 3 | 54 | 75,793 | 5,120 | 7 | 21,159 | 687 | 410.98 | 660 | 114,916 |

Reasoning tokens, ingestion cache buckets and other provider-unreported
fields remain unknown where the analyzer reports them as unknown.  No mixed
cost scalar is computed.

## Process corpus and safety

The canonical process corpus contains 231 content-free events across the
three replicates.  It includes trigger, source selection, extraction,
admission, commit, retrieval, exposure and task-outcome events; official score
and grader fields are inaccessible (`evaluationScoreAccessible=false`).
All three audits report zero privacy leaks, zero schema/safety failures and no
native-writer bypass.  Reflection episodes are process-only after `e22af5c`,
and the formal launcher canonicalizes shared-cold duplicate events before
audit after `569e295`.

## Disposition

This batch proves that the corrected Hermes path can complete three clean
parent feedback replicates and reconstruct the content-bearing corpus without
turning unresolved evidence into labels.  It does not satisfy the N+1 matched
validation gate: the corpus has no resolved actionable signal and no candidate
intervention.  The next eligible step is a predeclared family/fixture with
explicit deployment-observable use/outcome variation, followed by an
independent matched parent/proposal batch only after a candidate passes the
offline and safety gates.
