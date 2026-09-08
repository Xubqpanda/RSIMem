# Extraction Stage 2E Feedback v10 Record

Date: 2026-08-28

## Decision

The backup-provider plain-parent batch completed, but it produced no actionable
extraction signal. The frozen optimizer returned `NO_PROPOSAL` without a model
request or candidate. Stage 2E matched parent/candidate validation therefore
remains unstarted.

This is an accepted no-signal result, not an adaptation result. The resolved
signal threshold, anti-collapse gates, and candidate-generation budget were not
changed.

## Provenance

- Batch: `feedback-sm01-20260828-v10-backup`
- Experiment: `535d9d41aa356160a7d126e027f956b0bdec9d5c98c2c33f7a37c7fd6d29c826`
- RSIMem run commit: `2bf6c36f70ce709047bb2f46f3fd6ad162d81d63`
- PAST-Bench commit: `41696746d56383bb21dc9fb560c67cb85040217b`
- Provider profile: `http://47.88.93.22:10001/v1`, model `gpt-5.6-luna`
- Post-run exact-join fix: `09f0a5c`
- Post-run deterministic no-signal gate fix: `98c177a`

The batch used one provider profile for all three replicates. All attempts
closed as `completed`; there were no failed attempts or retries.

## Raw Usage

| Replicate | Requests | Input tokens | Output tokens | Retries | Completed sources | Unresolved labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 48 | 61,599 | 5,163 | 0 | 2 | 8 |
| 2 | 47 | 65,036 | 5,155 | 0 | 2 | 8 |
| 3 | 52 | 65,757 | 5,766 | 0 | 2 | 8 |

These are raw resource quantities. No provider price or mixed cost scalar is
derived from them.

## Evidence Gate

The preparation audit
`extraction-preparation-audit.550177a81a04dd212da0051aa34de825dd3c6892`
reported:

- 6 unique source records using source schema v3;
- 21 private source captures;
- 24 public feedback records and 24 matching private feedback captures;
- 24 unresolved primary labels;
- 0 useful, harmful, missed, or censored primary labels;
- 0 actionable primary examples, below the frozen minimum of 2;
- `corpusReady=true` and `optimizerSignalReady=false`.

The private corpus was reconstructed with exact source, capture, feedback,
operation, mutation, and artifact joins. Its digest is
`9687ae4e2fb274bbe3b408a9500f2cf23a8bee8381468b91aaad1d1ce5bc274c`.
The corpus and content-bearing captures remain in ignored owner-controlled
output with `0700` directories and `0600` files.

The deterministic proposal result was:

- decision: `NO_PROPOSAL`;
- reason: `no_actionable_extraction_signal`;
- provider eligible: false;
- optimizer model requests: 0;
- candidate artifact: none.

## Exclusions And Next Gate

- v7 is excluded because its third primary-provider replicate failed audit
  after two HTTP 503 requests.
- v8 is excluded because the unversioned backup URL returned the Sub2API HTML
  frontend to Chat Completions calls.
- v9 completed provider execution but is excluded from optimizer input because
  feedback captures were lost at the PAST history-branch boundary.

SM01 remains a pipeline pilot and cannot be forced into a training result. The
next compliant step is to freeze another semantic-memory training family before
viewing future-test evidence, collect a new plain-parent batch, and retain the
same minimum actionable signal threshold. Stage 2G, matched validation, and
production activation remain blocked until a real N+1 exists.
