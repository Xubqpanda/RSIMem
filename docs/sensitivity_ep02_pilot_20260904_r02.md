# EP02 Exception-List Recall Sensitivity Pilot - 2026-09-04 (Replicate 2)

This report records the accepted replicate-2 execution pilot for
`EP02_exception_list_recall`. It is execution and mechanism-readiness
evidence, not a completed Stage 3 sensitivity result. No candidate policy,
optimizer input, or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five registered episodic conditions
- Replicate: `2`
- Execution order: `native_static`, `type_matched_oracle`,
  `shortcut_current_input`, `wrong_mechanism`, `no_persistence`
- Batch: `episodic-ep02-r02-retry2-20260904_120623`
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
| native_static | 4 | 6 | 34,103 | 1,292 | 15,360 | 331 | 16 | 0 |
| type_matched_oracle | 2 | 10 | 17,588 | 917 | 17,920 | 236 | 11 | 0 |
| shortcut_current_input | 4 | 0 | 25,018 | 1,490 | 6,144 | 575 | 14 | 0 |
| wrong_mechanism | 4 | 0 | 19,479 | 1,316 | 8,704 | 490 | 13 | 0 |
| no_persistence | 1 | 0 | 2,230 | 265 | 1,536 | 197 | 2 | 0 |

Native and oracle slices produced episodic memory events. The no-persistence
and control slices produced no memory events, as required by their deployment
contracts. Every trace had a terminal event with complete usage; the audit
found no usage mismatch, missing trace, retry, or opaque task ID mismatch.

## Boundary And Next Gate

Task outputs and task scores remain inside PAST traces as audit-plane data.
They were not read by RSIMem policy, optimizer, process-signal metadata, or
this report. This single replicate supports neither a condition ranking nor a
sensitivity estimate. It confirms only that the five EP02 exception-list
recall deployment paths can be isolated, executed, and audited with the
registered episodic oracle seed and explicit controls.

EP03 replicate-2 and all predeclared replicate-3 slots remain pending. Raw
resources remain reporting fields and are not policy rewards.
