# PC01 SOP Bootstrap-01 Sensitivity Pilot - 2026-09-04 (Replicate 2)

This report records the accepted replicate-2 execution pilot for
`PC01_sop_bootstrap_01`. It is execution and mechanism-readiness evidence,
not a completed Stage 3 sensitivity result. No candidate policy, optimizer
input, or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five registered procedural conditions
- Replicate: `2`
- Execution order: `native_static`, `type_matched_oracle`,
  `shortcut_current_input`, `wrong_mechanism`, `no_persistence`
- Batch: `procedural-pc01-01-r02-20260904_163746`
- Judge: disabled
- Method boundary: opaque sensitivity case ID only
- Oracle basis: checked-in procedural seed manually authored from public SOP
  bootstrap inputs; no grader, answer, expectation, or official score was
  used to author the seed or method metadata.

## Execution Audit

The provider completion probe returned HTTP `200` with non-empty content and a
usage object. All five registered commands exited with code `0`. The dedicated
content-free sensitivity audit reported `ok=true` and no issues:

| Condition | Traces | Memory events | Input tokens | Output tokens | Cache read | Cache write | Reasoning | Requests | Retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native_static | 4 | 12 | 112,727 | 3,624 | 35,328 | 0 | 1,475 | 29 | 0 |
| type_matched_oracle | 2 | 9 | 63,746 | 3,638 | 29,696 | 0 | 2,486 | 19 | 0 |
| shortcut_current_input | 1 | 0 | 6,375 | 550 | 1,536 | 0 | 391 | 3 | 0 |
| wrong_mechanism | 1 | 0 | 23,424 | 1,610 | 4,608 | 0 | 1,133 | 9 | 0 |
| no_persistence | 1 | 0 | 17,164 | 1,171 | 2,560 | 0 | 722 | 6 | 0 |

Native and oracle slices produced procedural memory events. The
no-persistence and control slices produced no memory events, as required by
their deployment contracts. Every trace had a terminal event with complete
usage; the audit found no usage mismatch, missing trace, retry, or opaque task
ID mismatch.

## Boundary And Next Gate

Task outputs and task scores remain inside PAST traces as audit-plane data.
They were not read by RSIMem policy, optimizer, process-signal metadata, or
this report. This replicate supports neither a condition ranking nor a
sensitivity estimate. It confirms only that the five PC01 bootstrap-01
deployment paths can be isolated, executed, and audited with the registered
procedural oracle seed and explicit controls.

Procedural coverage remains replicate-incomplete. Raw resources remain
reporting fields and are not policy rewards.
