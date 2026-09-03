# PC01 Bootstrap-02 Procedural Sensitivity Pilot - 2026-09-03

This report records the accepted replicate-1 execution pilot for
`PC01_sop_bootstrap_02`. It is execution and mechanism-readiness evidence, not
a completed Stage 3 sensitivity result. No candidate policy, optimizer input,
or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five registered procedural conditions
- Replicate: `1`
- Execution order: `no_persistence`, `native_static`, `type_matched_oracle`,
  `shortcut_current_input`, `wrong_mechanism`
- Batch: `procedural-pc01-02-r01-retry2-20260903_211117`
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
| no_persistence | 1 | 0 | 7,297 | 604 | 3,072 | 279 | 4 | 0 |
| native_static | 4 | 9 | 42,091 | 3,305 | 49,152 | 882 | 23 | 0 |
| type_matched_oracle | 2 | 6 | 14,899 | 1,439 | 19,968 | 748 | 10 | 0 |
| shortcut_current_input | 1 | 0 | 7,083 | 504 | 4,096 | 125 | 4 | 0 |
| wrong_mechanism | 1 | 0 | 8,189 | 534 | 4,096 | 143 | 6 | 0 |

Native and oracle slices produced procedural memory events. The
no-persistence and control slices produced no memory events, as required by
their deployment contracts. Every trace had a terminal event with complete
usage; the audit found no usage mismatch, missing trace, retry, or opaque task
ID mismatch.

## Boundary And Next Gate

Task outputs and task scores remain inside PAST traces as audit-plane data.
They were not read by RSIMem policy, optimizer, process-signal metadata, or
this report. This single replicate supports neither a condition ranking nor a
sensitivity estimate. It confirms only that the five PC01 bootstrap-02
deployment paths can be isolated, executed, and audited with the registered
procedural oracle seed and explicit controls.

The earlier retry attempt is retained separately in
[`sensitivity_pc01_02_attempt_20260903.md`](sensitivity_pc01_02_attempt_20260903.md)
and excluded after its `wrong_mechanism` control timed out. Remaining
procedural families and predeclared matched replicates are still pending. Raw
resources remain reporting fields and are not policy rewards.
