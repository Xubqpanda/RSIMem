# EP03 Recall-Then-Modify Sensitivity Pilot - 2026-09-04 (Replicate 3)

This report records the accepted replicate-3 execution pilot for
`EP03_recall_then_modify`. It is execution and mechanism-readiness evidence,
not a completed Stage 3 sensitivity result. No candidate policy, optimizer
input, or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five registered episodic conditions
- Replicate: `3`
- Execution order: `type_matched_oracle`, `shortcut_current_input`,
  `wrong_mechanism`, `no_persistence`, `native_static`
- Batch: `episodic-ep03-r03-20260904_160519`
- Judge: disabled
- Method boundary: opaque sensitivity case ID only
- Oracle basis: checked-in episodic seed manually authored from public
  recall/update inputs; no grader, answer, expectation, or official score was
  used to author the seed or method metadata.

## Execution Audit

The provider completion probe returned HTTP `200` with non-empty content and a
usage object. All five registered commands exited with code `0`. The dedicated
content-free sensitivity audit reported `ok=true` and no issues:

| Condition | Traces | Memory events | Input tokens | Output tokens | Cache read | Cache write | Reasoning | Requests | Retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| type_matched_oracle | 2 | 10 | 31,849 | 816 | 4,608 | 0 | 312 | 12 | 0 |
| shortcut_current_input | 3 | 0 | 44,145 | 3,262 | 43,520 | 0 | 2,235 | 21 | 0 |
| wrong_mechanism | 3 | 0 | 67,074 | 4,255 | 10,240 | 0 | 3,071 | 22 | 0 |
| no_persistence | 3 | 0 | 56,624 | 3,781 | 19,456 | 0 | 2,554 | 22 | 0 |
| native_static | 3 | 80 | 102,526 | 3,267 | 51,200 | 0 | 1,893 | 34 | 0 |

Native and oracle slices produced episodic memory events. The no-persistence
and control slices produced no memory events, as required by their deployment
contracts. Every trace had a terminal event with complete usage; the audit
found no usage mismatch, missing trace, retry, or opaque task ID mismatch.

## Boundary And Next Gate

Task outputs and task scores remain inside PAST traces as audit-plane data.
They were not read by RSIMem policy, optimizer, process-signal metadata, or
this report. This replicate supports neither a condition ranking nor a
sensitivity estimate. It confirms only that the five EP03 recall-then-modify
deployment paths can be isolated, executed, and audited with the registered
episodic oracle seed and explicit controls.

EP01, EP02, and EP03 now each have accepted replicate-1, replicate-2, and
replicate-3 pilots. The content-free coverage aggregator therefore marks the
episodic panel replicate-analysis ready. Raw resources remain reporting fields
and are not policy rewards.
