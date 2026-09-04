# PC01 Bootstrap-06 Procedural Sensitivity Pilot - 2026-09-04

This report records the accepted replicate-1 execution pilot for
`PC01_sop_bootstrap_06`. It is execution and mechanism-readiness evidence, not
a completed Stage 3 sensitivity result. No candidate policy, optimizer input,
or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five registered procedural conditions
- Replicate: `1`
- Execution order: `no_persistence`, `native_static`, `type_matched_oracle`,
  `shortcut_current_input`, `wrong_mechanism`
- Batch: `procedural-pc01-06-r01-20260904_105906`
- Judge: disabled
- Method boundary: opaque sensitivity case ID only
- Oracle basis: checked-in procedural skill seed manually authored from public
  learn/update inputs; no grader, answer, expectation, or official score was
  used to author the seed or method metadata.

## Execution Audit

The provider completion probe returned HTTP `200` with non-empty content and a
usage object. All five registered commands exited with code `0`. The dedicated
content-free sensitivity audit reported `ok=true` and no issues:

| Condition | Traces | Memory events | Input tokens | Output tokens | Cache read | Reasoning | Requests | Retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_persistence | 1 | 0 | 9,531 | 1,072 | 1,536 | 451 | 4 | 0 |
| native_static | 4 | 18 | 91,086 | 4,231 | 53,760 | 1,287 | 36 | 0 |
| type_matched_oracle | 2 | 9 | 40,024 | 2,249 | 27,136 | 856 | 14 | 0 |
| shortcut_current_input | 1 | 0 | 9,140 | 674 | 4,096 | 190 | 4 | 0 |
| wrong_mechanism | 1 | 0 | 14,779 | 869 | 2,560 | 292 | 6 | 0 |

Native and oracle slices produced procedural memory events. The
no-persistence and control slices produced no memory events, as required by
their deployment contracts. Every trace had a terminal event with complete
usage; the audit found no usage mismatch, missing trace, retry, or opaque task
ID mismatch.

## Boundary And Next Gate

Task outputs and task scores remain inside PAST traces as audit-plane data.
They were not read by RSIMem policy, optimizer, process-signal metadata, or
this report. This single replicate supports neither a condition ranking nor a
sensitivity estimate. It confirms only that the five PC01 bootstrap-06
deployment paths can be isolated, executed, and audited with the registered
procedural oracle seed and explicit controls.

The remaining procedural family is `PC04_failure_to_rule_01`; predeclared
matched replicates are still pending. Raw resources remain reporting fields
and are not policy rewards.
