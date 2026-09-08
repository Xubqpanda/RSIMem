# PC01 Bootstrap-03 Procedural Sensitivity Pilot - 2026-09-04

This report records the accepted replicate-1 execution pilot for
`PC01_sop_bootstrap_03`. It is execution and mechanism-readiness evidence, not
a completed Stage 3 sensitivity result. No candidate policy, optimizer input,
or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five registered procedural conditions
- Replicate: `1`
- Execution order: `no_persistence`, `native_static`, `type_matched_oracle`,
  `shortcut_current_input`, `wrong_mechanism`
- Batch: `procedural-pc01-03-r01-retry2-20260904_095530`
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
| no_persistence | 1 | 0 | 16,405 | 1,237 | 2,560 | 391 | 6 | 0 |
| native_static | 4 | 9 | 86,941 | 3,604 | 32,768 | 624 | 25 | 0 |
| type_matched_oracle | 2 | 6 | 24,082 | 1,581 | 13,824 | 241 | 10 | 0 |
| shortcut_current_input | 1 | 0 | 8,788 | 474 | 3,072 | 50 | 4 | 0 |
| wrong_mechanism | 1 | 0 | 15,269 | 687 | 0 | 191 | 5 | 0 |

Native and oracle slices produced procedural memory events. The
no-persistence and control slices produced no memory events, as required by
their deployment contracts. Every trace had a terminal event with complete
usage; the audit found no usage mismatch, missing trace, retry, or opaque task
ID mismatch.

## Boundary And Next Gate

Task outputs and task scores remain inside PAST traces as audit-plane data.
They were not read by RSIMem policy, optimizer, process-signal metadata, or
this report. This single replicate supports neither a condition ranking nor a
sensitivity estimate. It confirms only that the five PC01 bootstrap-03
deployment paths can be isolated, executed, and audited with the registered
procedural oracle seed and explicit controls.

The earlier attempt is retained separately in
[`sensitivity_pc01_03_attempt_20260903.md`](sensitivity_pc01_03_attempt_20260903.md)
and excluded after its `wrong_mechanism` control lacked terminal completion.
Remaining procedural families and predeclared matched replicates are still
pending. Raw resources remain reporting fields and are not policy rewards.
