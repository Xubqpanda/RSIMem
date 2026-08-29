# SM02 process-feedback pilot — 2026-08-29

This report records two independent real-provider attempts used to test the
process-feedback path for `SM02_constraint_retention`.  They are feasibility
and signal-census runs, not a parent/candidate effect comparison and not paper
quality evidence.  The two endpoints were never pooled into one experiment.

## Batch outcomes

`s1-sm02-process-20260829-v2` used the primary `coding.tu-zi.com` endpoint.
Replicates 1 and 2 completed with clean audits; replicate 3 encountered HTTP
503 (`资源不足`) during a learn extraction call, so its missing history anchor
was recorded as a failed `past_bench` attempt.  The batch is incomplete and is
not eligible for optimizer input or activation.

`s1-sm02-process-20260829-backup` used the backup endpoint.  Its first replicate
failed in the cold baseline after three HTTP 503 retries; the remaining slots
were not run.  This is retained as a provider-failure attempt, not as a task
failure label.

| batch / slot | status | requests | input tokens | output tokens | retries | traces | audit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| v2 / r01 | completed | 42 | 56,578 | 5,136 | 0 | 9 | clean |
| v2 / r02 | completed | 43 | 69,661 | 5,053 | 0 | 9 | clean |
| v2 / r03 | failed | provider 503 | unknown | unknown | provider retries | partial | not eligible |
| backup / r01 | failed | 27 | 0 | 0 | 18 | 9 | provider failure |

The completed v2 slots produced 8 primary feedback examples: 8 `missed` and 8
`unresolved` observations across the two replicate histories (the same logical
source is present in each replicate).  No `useful` or `harmful` label was
generated.  `missed` was tied to the registered SM02 boundary contract;
`unresolved/not_exposed` remained outside the resolved denominator.  Because the
formal batch was incomplete and has no resolved useful/harmful variation, no
optimizer proposal was generated.

## Process corpus

Each clean slot produced an owner-controlled `process_corpus.json` with 68 and
69 deduplicated content-free events respectively.  The corpus includes
retrieval, extraction, admission, commit, exposure and task-outcome stages;
shared-cold duplicate events are collapsed only when their canonical payload is
identical, while conflicting payloads fail closed.  Current per-slot reason
counts include `decision_observed`, `retrieval_miss`, `absence` and
`task_completed`.  The runtime fix in commit `ad6b916` also projects trigger and
source-selection decisions into this ledger for subsequent runs; the two
already-produced corpora predate that fix and are preserved unchanged.

Process corpus and official PAST-Bench score are separate objects.  No grader,
answer key, hidden expectation, token count or cost value is supplied to the
policy learner.  Raw request/token/retry vectors remain accounting evidence.

## Diagnosis and next gate

The primary endpoint is intermittently capacity-limited; the backup endpoint
was unavailable for this run as well.  The completed SM02 traces nevertheless
show useful process density (notes calls, memory exposure, extraction/admission
decisions and task outcomes), but strict memory-specific use/outcome evidence
is still sparse.  The next eligible action is a new clean, single-provider
SM02 process batch after provider health is confirmed, followed by a process
signal census.  No adaptive candidate, matched effect claim or uplift should be
started from these incomplete attempts.
