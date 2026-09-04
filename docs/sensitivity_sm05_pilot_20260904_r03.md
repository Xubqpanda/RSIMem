# SM05 Weak Trigger Preference Adoption Sensitivity Pilot - 2026-09-04 (Replicate 3)

This report records the accepted replicate-3 execution pilot for
`SM05_weak_trigger_preference_adoption`. It is execution and
mechanism-readiness evidence, not a completed Stage 3 sensitivity result. No
candidate policy, optimizer input, or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five registered semantic conditions
- Replicate: `3`
- Execution order: `type_matched_oracle`, `shortcut_current_input`,
  `wrong_mechanism`, `no_persistence`, `native_static`
- Batch: `semantic-sm05-r03-20260904_143550`
- Judge: disabled
- Method boundary: opaque sensitivity case ID only
- Oracle basis: checked-in semantic seed manually authored from public weak
  trigger preference-adoption inputs; no grader, answer, expectation, or
  official score was used to author the seed or method metadata.

## Execution Audit

The provider completion probe returned HTTP `200` with non-empty content and a
usage object. All five registered commands exited with code `0`. The dedicated
content-free sensitivity audit reported `ok=true` and no issues:

| Condition | Traces | Memory events | Input tokens | Output tokens | Cache read | Reasoning | Requests | Retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| type_matched_oracle | 2 | 20 | 14,828 | 685 | 4,608 | 333 | 8 | 0 |
| shortcut_current_input | 1 | 0 | 4,393 | 331 | 3,072 | 177 | 4 | 0 |
| wrong_mechanism | 1 | 0 | 4,154 | 272 | 3,072 | 100 | 4 | 0 |
| no_persistence | 1 | 0 | 7,068 | 308 | 0 | 106 | 4 | 0 |
| native_static | 6 | 66 | 48,933 | 1,948 | 7,680 | 816 | 23 | 0 |

Native and oracle slices produced semantic memory events. The no-persistence
and control slices produced no memory events, as required by their deployment
contracts. Every trace had a terminal event with complete usage; the audit
found no usage mismatch, missing trace, retry, or opaque task ID mismatch.

## Boundary And Next Gate

Task outputs and task scores remain inside PAST traces as audit-plane data.
They were not read by RSIMem policy, optimizer, process-signal metadata, or
this report. This single replicate supports neither a condition ranking nor a
sensitivity estimate. It confirms only that the five SM05 weak-trigger
preference-adoption deployment paths can be isolated, executed, and audited
with the registered semantic oracle seed and explicit controls.

SM05 now has replicate-1, replicate-2, and replicate-3 pilots. SM06 and SM07
still require replicate-3 execution. Raw resources remain reporting fields and
are not policy rewards.
