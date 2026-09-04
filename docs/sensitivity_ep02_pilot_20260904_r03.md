# EP02 Exception List Recall Sensitivity Pilot - 2026-09-04 (Replicate 3)

This report records the accepted replicate-3 execution pilot for
`EP02_exception_list_recall`. It is execution and mechanism-readiness evidence,
not a completed Stage 3 sensitivity result. No candidate policy, optimizer
input, or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five registered episodic conditions
- Replicate: `3`
- Execution order: `type_matched_oracle`, `shortcut_current_input`,
  `wrong_mechanism`, `no_persistence`, `native_static`
- Batch: `episodic-ep02-r03-20260904_154843`
- Judge: disabled
- Method boundary: opaque sensitivity case ID only
- Oracle basis: checked-in episodic seed manually authored from public
  exception-list inputs; no grader, answer, expectation, or official score was
  used to author the seed or method metadata.

## Execution Audit

The provider completion probe returned HTTP `200` with non-empty content and a
usage object. All five registered commands exited with code `0`. The dedicated
content-free sensitivity audit reported `ok=true` and no issues:

| Condition | Traces | Memory events | Input tokens | Output tokens | Cache read | Reasoning | Requests | Retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| type_matched_oracle | 2 | 8 | 30,033 | 1,111 | 1,536 | 360 | 10 | 0 |
| shortcut_current_input | 4 | 0 | 23,566 | 1,775 | 11,776 | 692 | 15 | 0 |
| wrong_mechanism | 4 | 0 | 20,355 | 1,436 | 7,680 | 529 | 13 | 0 |
| no_persistence | 1 | 0 | 2,200 | 125 | 1,536 | 57 | 2 | 0 |
| native_static | 4 | 8 | 31,682 | 1,363 | 12,288 | 322 | 14 | 0 |

Native and oracle slices produced episodic memory events. The no-persistence
and control slices produced no memory events, as required by their deployment
contracts. Every trace had a terminal event with complete usage; the audit
found no usage mismatch, missing trace, retry, or opaque task ID mismatch.

## Boundary And Next Gate

Task outputs and task scores remain inside PAST traces as audit-plane data.
They were not read by RSIMem policy, optimizer, process-signal metadata, or
this report. This single replicate supports neither a condition ranking nor a
sensitivity estimate. It confirms only that the five EP02 exception-list
deployment paths can be isolated, executed, and audited with the registered
episodic oracle seed and explicit controls.

EP02 now has replicate-1, replicate-2, and replicate-3 pilots. EP03 still
requires replicate-3 execution. Raw resources remain reporting fields and are
not policy rewards.
