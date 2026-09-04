# EP03 Recall-Then-Modify Sensitivity Pilot - 2026-09-04 (Replicate 2)

This report records the accepted replicate-2 execution pilot for
`EP03_recall_then_modify`. It is execution and mechanism-readiness evidence,
not a completed Stage 3 sensitivity result. No candidate policy, optimizer
input, or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five registered episodic conditions
- Replicate: `2`
- Execution order: `native_static`, `type_matched_oracle`,
  `shortcut_current_input`, `wrong_mechanism`, `no_persistence`
- Batch: `episodic-ep03-r02-20260904_121925`
- Judge: disabled
- Method boundary: opaque sensitivity case ID only
- Oracle basis: checked-in episodic seed manually authored from public task
  state and recall/update inputs; no grader, answer, expectation, or official
  score was used to author the seed or method metadata.

## Execution Audit

The provider completion probe returned HTTP `200` with non-empty content and a
usage object. All five registered commands exited with code `0`. The dedicated
content-free sensitivity audit reported `ok=true` and no issues:

| Condition | Traces | Memory events | Input tokens | Output tokens | Cache read | Reasoning | Requests | Retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native_static | 3 | 66 | 61,799 | 3,118 | 95,232 | 1,858 | 34 | 0 |
| type_matched_oracle | 2 | 10 | 21,200 | 951 | 13,312 | 444 | 11 | 0 |
| shortcut_current_input | 3 | 0 | 28,042 | 2,792 | 30,208 | 1,933 | 18 | 0 |
| wrong_mechanism | 3 | 0 | 33,704 | 5,019 | 48,128 | 3,263 | 24 | 0 |
| no_persistence | 3 | 0 | 69,277 | 4,977 | 51,712 | 3,027 | 26 | 0 |

Native and oracle slices produced episodic memory events. The no-persistence
and control slices produced no memory events, as required by their deployment
contracts. Every trace had a terminal event with complete usage; the audit
found no usage mismatch, missing trace, retry, or opaque task ID mismatch.

## Boundary And Next Gate

Task outputs and task scores remain inside PAST traces as audit-plane data.
They were not read by RSIMem policy, optimizer, process-signal metadata, or
this report. This single replicate supports neither a condition ranking nor a
sensitivity estimate. It confirms only that the five EP03 recall-then-modify
deployment paths can be isolated, executed, and audited with the registered
episodic oracle seed and explicit controls.

All episodic families now have replicate-1 and replicate-2 pilots; the
predeclared replicate-3 slots remain pending. Raw resources remain reporting
fields and are not policy rewards.
