# SM07 Scoped Rule Migration Sensitivity Pilot - 2026-09-04 (Replicate 3)

This report records the accepted replicate-3 execution pilot for
`SM07_scoped_rule_migration`. It is execution and mechanism-readiness
evidence, not a completed Stage 3 sensitivity result. No candidate policy,
optimizer input, or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five registered semantic conditions
- Replicate: `3`
- Execution order: `type_matched_oracle`, `shortcut_current_input`,
  `wrong_mechanism`, `no_persistence`, `native_static`
- Batch: `semantic-sm07-r03-20260904_145547`
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
| type_matched_oracle | 3 | 30 | 34,008 | 2,981 | 18,944 | 2,378 | 16 | 0 |
| shortcut_current_input | 1 | 0 | 21,495 | 2,129 | 32,256 | 1,398 | 13 | 0 |
| wrong_mechanism | 1 | 0 | 8,342 | 712 | 4,096 | 513 | 5 | 0 |
| no_persistence | 1 | 0 | 10,511 | 652 | 0 | 420 | 5 | 0 |
| native_static | 4 | 38 | 50,349 | 3,941 | 7,680 | 3,112 | 20 | 0 |

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

SM02 through SM07 now have three accepted replicates. SM01 still lacks its
pre-registered replicate-1 pilot, so the semantic panel is not yet
replicate-analysis ready. Raw resources remain reporting fields and are not
policy rewards.
