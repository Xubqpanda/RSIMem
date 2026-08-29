# Stage 2 process-signal census: SM02 — 2026-08-30

This report covers the clean primary-provider SM02 parent rerun after the
Hermes tool-closure projection fix in commit `643dce1`. It is a deterministic
process-observability report, not an effect experiment and not benchmark
quality evidence. Grader output, answer keys, task scores, hidden
expectations, and resource cost are not inputs to the signal classification.

## Batch status

The batch used `https://coding.tu-zi.com/v1` with `gpt-5.6-luna`,
`native+ledger`, static extraction, and judge disabled. Replicates 1 and 2
completed with clean audits. Replicate 3 exceeded the execution timeout and
is retained as an infrastructure failure; it is excluded from the census.

| replicate | status | process events | audit | model requests | input tokens | output tokens |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| r01 | completed | 125 | clean | 40 | 55,427 | 4,898 |
| r02 | completed | 119 | clean | 37 | 56,362 | 4,213 |
| r03 | failed (timeout) | — | — | — | — | — |

The resource columns are raw accounting only; they were not used by any
policy or signal decision.

## Process evidence

The two completed corpora contain 244 physical events in total. Their merged
event census is:

- 92 policy-bound events (37.70%); 200 events carry execution receipts
  (81.97%);
- 12 trigger, 12 source-selection, 12 extraction, 12 admission, and 16
  commit events across the two replicates;
- 44 exposure events, 28 retrieval events, 18 task-outcome events;
- 45 tool-call and 45 tool-result events (all 90 tool events carry the
  expected `family_id=SM02_constraint_retention` and their actual stage);
- reason counts: `decision_observed` 145, `tool_failure` 45,
  `retrieval_miss` 20, `task_completed` 18, `absence` 16.

The previous failed audit was caused by tool events losing family/stage during
projection. `643dce1` keeps the join contract host-neutral and injects scope
only into the projected `ProcessEvent`; both completed reruns now pass the
cross-ledger identity audit.

## Logical-case census

Each `(task_id, stage)` is one logical case and each replicate is one physical
observation. The 18 physical observations reduce to 9 logical cases, with no
replicate status conflicts:

```text
physical observations: 18
logical cases:          9
conflicts:              0
status:                 observable_only = 9
```

The status is intentionally conservative. Extraction and source-selection
events are shadow decisions with `pending` execution status, so they do not
count as executed extraction observations. Exposure is similarly pending or
skipped, and no memory injection is observed. The corpus does contain tool
closures, retrieval outcomes, commits, and task outcomes, but it does not
contain a complete source → extraction → persistence → retrieval → exposure →
outcome chain attributable to an extraction action.

Therefore the current gate is:

```text
Extraction: diagnostic-only / observable-only
Other five layers: validation-only or observable-only
Decision: STOP_NO_SIGNAL
```

No N+1 candidate, held-out matched validation, adaptive effect batch, or
six-layer joint intervention should be opened from this batch. In particular,
`pending` shadow events and unresolved/not-exposed observations are not
converted into negative labels.

## Next action

Keep the fixed parent path and process instrumentation. Before opening
Extraction optimization, obtain a predeclared case with an executed extraction
action, a deterministic memory mutation or artifact identity, and a later
observable use/outcome chain. Once that signal is present and replay-stable,
the next permitted step is a single Extraction N+1 candidate; Trigger,
Admission, Exposure, and the remaining layers stay closed or shadow-only.
