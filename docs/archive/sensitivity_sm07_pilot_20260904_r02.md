# SM07 Scoped Rule Migration Sensitivity Pilot - 2026-09-04 (Replicate 2)

This report records the accepted replicate-2 execution pilot for
`SM07_scoped_rule_migration`. It is execution and mechanism-readiness
evidence, not a completed Stage 3 sensitivity result. No candidate policy,
optimizer input, or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five registered semantic conditions
- Replicate: `2`
- Execution order: `native_static`, `type_matched_oracle`,
  `shortcut_current_input`, `wrong_mechanism`, `no_persistence`
- Batch: `semantic-sm07-r02-20260904_134330`
- Judge: disabled
- Method boundary: opaque sensitivity case ID only
- Oracle basis: checked-in semantic seed manually authored from public scoped
  rule-migration inputs; no grader, answer, expectation, or official score was
  used to author the seed or method metadata.

## Execution Audit

The provider completion probe returned HTTP `200` with non-empty content and a
usage object. All five registered commands exited with code `0`. The dedicated
content-free sensitivity audit reported `ok=true` and no issues:

| Condition | Traces | Memory events | Input tokens | Output tokens | Cache read | Reasoning | Requests | Retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native_static | 4 | 38 | 45,476 | 4,982 | 14,848 | 3,996 | 22 | 0 |
| type_matched_oracle | 3 | 30 | 36,813 | 5,160 | 19,968 | 4,438 | 19 | 0 |
| shortcut_current_input | 1 | 0 | 15,174 | 1,929 | 5,632 | 1,598 | 8 | 0 |
| wrong_mechanism | 1 | 0 | 10,535 | 1,158 | 3,584 | 876 | 5 | 0 |
| no_persistence | 1 | 0 | 4,372 | 455 | 3,072 | 336 | 4 | 0 |

Native and oracle slices produced semantic memory events. The no-persistence
and control slices produced no memory events, as required by their deployment
contracts. Every trace had a terminal event with complete usage; the audit
found no usage mismatch, missing trace, retry, or opaque task ID mismatch.

## Boundary And Next Gate

Task outputs and task scores remain inside PAST traces as audit-plane data.
They were not read by RSIMem policy, optimizer, process-signal metadata, or
this report. This single replicate supports neither a condition ranking nor a
sensitivity estimate. It confirms only that the five SM07 scoped-rule
migration deployment paths can be isolated, executed, and audited with the
registered semantic oracle seed and explicit controls.

SM07 now has replicate-1 and replicate-2 pilots; its replicate-3 slot and
other panel replicate slots remain pending. Raw resources remain reporting
fields and are not policy rewards.
