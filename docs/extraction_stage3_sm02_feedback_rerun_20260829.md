# SM02 train feedback rerun — 2026-08-29

This is a clean, three-replicate real-provider plain-parent batch for the
frozen `train` assignment in
`configs/extraction_split_plan_sm02_sm03_sm04.json`. It is retained as
training evidence and optimizer diagnostics, not as a matched effect result.

## Batch and audit

Batch ID: `sm02-feedback-20260829-rerun-main`  
Family: `SM02_constraint_retention`  
Method: `static-extraction-rsimem`  
Provider: the primary configured HTTPS endpoint  
Split role: `train`

All three replicates passed the structural audit and produced a durable
`process_corpus.json`. The batch contains 24 primary feedback records: 12
contract-resolved `missed` labels and 12 `unresolved` labels, with no useful or
harmful labels. The unresolved records remain outside the resolved denominator;
they are not converted into negatives.

The three attempts produced the following raw accounting vectors. These are
reported independently and are not combined into a learner cost objective.

| replicate | requests | input tokens | output tokens | process events | audit |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 41 | 55,077 | 5,134 | 77 | clean |
| 2 | 38 | 52,031 | 4,461 | 77 | clean |
| 3 | 40 | 57,324 | 4,447 | 77 | clean |

The batch totals are 119 requests, 164,432 input tokens, 14,042 output
tokens and 231 process events. Cache buckets were unavailable and remain
unknown in the raw audit.

## Optimizer attempt

The corpus audit found 12 actionable primary examples, exceeding the frozen
minimum of two. One bounded optimizer request was sent with the frozen strict
JSON-schema contract. The provider returned a schema-valid proposal, but the
candidate content gate rejected it for copying a corpus-specific value. No
candidate artifact was written and no policy was activated. The rejection is
recorded as `candidate_corpus_value`; the persisted optimizer result retains
only request/completion identities and raw usage.

The optimizer request used one model request with 44,170 input tokens, 479
output tokens and 216 reasoning tokens. This accounting is separate from the
PAST-Bench task traces. The durable result is under the ignored
`outputs/extraction_optimizer_owner/sm02-feedback-20260829-rerun-main/` tree.

This run therefore does not provide an N+1 candidate or permit matched
validation. It does provide an auditable train corpus, a strict safety
rejection, and evidence that the provider path is currently reachable. The
next valid step is to author or collect another predeclared training batch or
adjust the optimizer proposal only under a new frozen budget; the rejected
completion must not be retried indefinitely against the same validation data.
