# SM02 Constraint Retention Sensitivity Pilot - 2026-09-04 (Replicate 3)

This report records the accepted replicate-3 execution pilot for
`SM02_constraint_retention`. It is execution and mechanism-readiness evidence,
not a completed Stage 3 sensitivity result. No candidate policy, optimizer
input, or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five registered semantic conditions
- Replicate: `3`
- Execution order: `type_matched_oracle`, `shortcut_current_input`,
  `wrong_mechanism`, `no_persistence`, `native_static`
- Batch: `semantic-sm02-r03-20260904_140654-retry`
- Judge: disabled
- Method boundary: opaque sensitivity case ID only
- Oracle basis: checked-in semantic seed manually authored from public
  constraint-retention inputs; no grader, answer, expectation, or official
  score was used to author the seed or method metadata.

## Execution Audit

The provider completion probe returned HTTP `200` with non-empty content and a
usage object. All five registered commands exited with code `0`. The dedicated
content-free sensitivity audit reported `ok=true` and no issues:

| Condition | Traces | Memory events | Input tokens | Output tokens | Cache read | Reasoning | Requests | Retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| type_matched_oracle | 2 | 20 | 13,668 | 732 | 3,072 | 431 | 7 | 0 |
| shortcut_current_input | 1 | 0 | 4,513 | 595 | 3,072 | 412 | 4 | 0 |
| wrong_mechanism | 1 | 0 | 3,232 | 213 | 0 | 164 | 2 | 0 |
| no_persistence | 1 | 0 | 5,086 | 359 | 0 | 272 | 3 | 0 |
| native_static | 5 | 48 | 31,809 | 2,093 | 9,216 | 1,232 | 17 | 0 |

Native and oracle slices produced semantic memory events. The no-persistence
and control slices produced no memory events, as required by their deployment
contracts. Every trace had a terminal event with complete usage; the audit
found no usage mismatch, missing trace, retry, or opaque task ID mismatch.

## Boundary And Next Gate

Task outputs and task scores remain inside PAST traces as audit-plane data.
They were not read by RSIMem policy, optimizer, process-signal metadata, or
this report. This single replicate supports neither a condition ranking nor a
sensitivity estimate. It confirms only that the five SM02 constraint-retention
deployment paths can be isolated, executed, and audited with the registered
semantic oracle seed and explicit controls.

SM02 now has replicate-1, replicate-2, and replicate-3 pilots. Other semantic
families still require replicate-3 execution. Raw resources remain reporting
fields and are not policy rewards.
