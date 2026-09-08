# SM01 Preference Adoption Sensitivity Pilot - 2026-09-04 (Replicate 1)

This report records the accepted replicate-1 execution pilot for
`SM01_preference_adoption`. It is execution and mechanism-readiness evidence,
not a completed Stage 3 sensitivity result. No candidate policy, optimizer
input, or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five registered semantic conditions
- Replicate: `1`
- Execution order: `no_persistence`, `native_static`, `type_matched_oracle`,
  `shortcut_current_input`, `wrong_mechanism`
- Batch: `semantic-sm01-r01-20260904_152345`
- Judge: disabled
- Method boundary: opaque sensitivity case ID only
- Oracle basis: checked-in semantic seed manually authored from public
  preference-adoption inputs; no grader, answer, expectation, or official
  score was used to author the seed or method metadata.

## Execution Audit

The provider completion probe returned HTTP `200` with non-empty content and a
usage object. All five registered commands exited with code `0`. The dedicated
content-free sensitivity audit reported `ok=true` and no issues:

| Condition | Traces | Memory events | Input tokens | Output tokens | Cache read | Reasoning | Requests | Retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_persistence | 1 | 0 | 3,193 | 198 | 0 | 145 | 2 | 0 |
| native_static | 5 | 48 | 26,832 | 1,996 | 16,896 | 1,152 | 18 | 0 |
| type_matched_oracle | 2 | 20 | 24,729 | 1,366 | 4,608 | 965 | 10 | 0 |
| shortcut_current_input | 1 | 0 | 5,630 | 285 | 1,536 | 133 | 4 | 0 |
| wrong_mechanism | 1 | 0 | 5,457 | 344 | 1,536 | 166 | 4 | 0 |

Native and oracle slices produced semantic memory events. The no-persistence
and control slices produced no memory events, as required by their deployment
contracts. Every trace had a terminal event with complete usage; the audit
found no usage mismatch, missing trace, retry, or opaque task ID mismatch.

## Boundary And Next Gate

Task outputs and task scores remain inside PAST traces as audit-plane data.
They were not read by RSIMem policy, optimizer, process-signal metadata, or
this report. This single replicate supports neither a condition ranking nor a
sensitivity estimate. It confirms only that the five SM01 preference-adoption
deployment paths can be isolated, executed, and audited with the registered
semantic oracle seed and explicit controls.

All seven semantic families now have three accepted replicates. The coverage
aggregator reports `replicate_coverage_complete=true` and
`ready_for_replicate_analysis=true` for semantic. This does not itself
establish a sensitivity status. Raw resources remain reporting fields and are
not policy rewards.
