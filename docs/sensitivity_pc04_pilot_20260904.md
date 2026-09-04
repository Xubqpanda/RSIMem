# PC04 Failure-to-Rule Procedural Sensitivity Pilot - 2026-09-04

This report records the accepted replicate-1 execution pilot for
`PC04_failure_to_rule_01`. It is execution and mechanism-readiness evidence,
not a completed Stage 3 sensitivity result. No candidate policy, optimizer
input, or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five registered procedural conditions
- Replicate: `1`
- Execution order: `no_persistence`, `native_static`, `type_matched_oracle`,
  `shortcut_current_input`, `wrong_mechanism`
- Batch: `procedural-pc04-r01-20260904_111427`
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
| no_persistence | 1 | 0 | 10,838 | 2,149 | 12,288 | 1,703 | 6 | 0 |
| native_static | 4 | 12 | 70,218 | 3,018 | 48,640 | 1,477 | 26 | 0 |
| type_matched_oracle | 2 | 6 | 39,852 | 3,310 | 39,424 | 2,456 | 17 | 0 |
| shortcut_current_input | 1 | 0 | 4,762 | 336 | 3,072 | 162 | 3 | 0 |
| wrong_mechanism | 1 | 0 | 28,286 | 1,722 | 6,144 | 1,161 | 10 | 0 |

Native and oracle slices produced procedural memory events. The
no-persistence and control slices produced no memory events, as required by
their deployment contracts. Every trace had a terminal event with complete
usage; the audit found no usage mismatch, missing trace, retry, or opaque task
ID mismatch.

## Boundary And Next Gate

Task outputs and task scores remain inside PAST traces as audit-plane data.
They were not read by RSIMem policy, optimizer, process-signal metadata, or
this report. This single replicate supports neither a condition ranking nor a
sensitivity estimate. It confirms only that the five PC04 failure-to-rule
deployment paths can be isolated, executed, and audited with the registered
procedural oracle seed and explicit controls.

Procedural family coverage is now complete (`10/10`), but the predeclared
matched replicate requirement remains incomplete. Raw resources remain
reporting fields and are not policy rewards.
