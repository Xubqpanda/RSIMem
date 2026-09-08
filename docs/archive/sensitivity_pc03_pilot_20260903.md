# PC03 Procedural Sensitivity Pilot - 2026-09-03

This report records the accepted replicate-1 execution pilot for
`PC03_latent_rule_induction_01`. It is execution and mechanism-readiness
evidence, not a completed Stage 3 sensitivity result. No candidate policy,
optimizer input, or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five registered procedural conditions
- Replicate: `1`
- Execution order: `no_persistence`, `native_static`, `type_matched_oracle`,
  `shortcut_current_input`, `wrong_mechanism`
- Batch: `procedural-pc03-r01-retry2-20260903_191830`
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
| no_persistence | 1 | 0 | 21,332 | 2,501 | 23,040 | 1,949 | 11 | 0 |
| native_static | 5 | 21 | 102,274 | 5,751 | 65,024 | 2,455 | 33 | 0 |
| type_matched_oracle | 2 | 6 | 43,519 | 1,883 | 13,312 | 1,208 | 13 | 0 |
| shortcut_current_input | 1 | 0 | 7,350 | 248 | 4,096 | 49 | 4 | 0 |
| wrong_mechanism | 1 | 0 | 19,639 | 1,597 | 13,312 | 1,107 | 10 | 0 |

Native and oracle slices produced procedural memory events. The
no-persistence and control slices produced no memory events, as required by
their deployment contracts. Every trace had a terminal event with complete
usage; the audit found no usage mismatch, missing trace, retry, or opaque task
ID mismatch.

## Boundary And Next Gate

Task outputs and task scores remain inside PAST traces as audit-plane data.
They were not read by RSIMem policy, optimizer, process-signal metadata, or
this report. This single replicate supports neither a condition ranking nor a
sensitivity estimate. It confirms only that the five PC03 deployment paths can
be isolated, executed, and audited with the registered procedural oracle seed
and explicit controls.

The earlier PC03 attempt is retained separately in
[`sensitivity_pc03_attempt_20260903.md`](sensitivity_pc03_attempt_20260903.md)
and excluded after an interrupted `native_static` run. This retry uses a new
batch identity and fresh isolated state. Remaining procedural families and
predeclared matched replicates are still pending. Raw resources remain
reporting fields and are not policy rewards.
