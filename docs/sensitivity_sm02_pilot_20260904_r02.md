# SM02 Constraint Retention Sensitivity Pilot - 2026-09-04 (Replicate 2)

This report records the accepted replicate-2 execution pilot for
`SM02_constraint_retention`. It is execution and mechanism-readiness evidence,
not a completed Stage 3 sensitivity result. No candidate policy, optimizer
input, or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five registered semantic conditions
- Replicate: `2`
- Execution order: `native_static`, `type_matched_oracle`,
  `shortcut_current_input`, `wrong_mechanism`, `no_persistence`
- Batch: `semantic-sm02-r02-20260904_125002`
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
| native_static | 5 | 48 | 41,793 | 2,245 | 23,552 | 1,300 | 20 | 0 |
| type_matched_oracle | 2 | 20 | 9,355 | 651 | 7,680 | 361 | 7 | 0 |
| shortcut_current_input | 1 | 0 | 4,363 | 416 | 3,072 | 215 | 4 | 0 |
| wrong_mechanism | 1 | 0 | 3,233 | 307 | 0 | 234 | 2 | 0 |
| no_persistence | 1 | 0 | 3,694 | 273 | 1,536 | 178 | 3 | 0 |

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

SM02 now has replicate-1 and replicate-2 pilots; its replicate-3 slot and
other semantic replicate slots remain pending. Raw resources remain reporting
fields and are not policy rewards.
