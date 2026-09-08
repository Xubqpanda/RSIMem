# PC02 Procedural Sensitivity Pilot - 2026-09-02

This report records the accepted retry-2 replicate-1 execution pilot for the
procedural `PC02_sop_patch_01` family. It is execution and
mechanism-readiness evidence, not a completed Stage 3 sensitivity result. No
candidate policy, optimizer input, or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five registered procedural conditions
- Replicate: `1`
- Batch: `procedural-pc02-r01-retry2-20260902_201104`
- Execution order: `no_persistence`, `native_static`, `type_matched_oracle`,
  `shortcut_current_input`, `wrong_mechanism`
- Judge: disabled
- Method boundary: opaque sensitivity case ID only
- Oracle basis: checked-in procedural skill seed manually authored from public
  learn/update inputs; no grader, answer, expectation, or official score was
  used to author the seed or method metadata.

The earlier batch `procedural-pc02-01-r01-20260902_194050` is retained as a
failed infrastructure attempt and is documented separately. It is excluded
from this accepted pilot and all sensitivity denominators.

## Execution Audit

The retry-2 provider completion probe returned HTTP `200` with non-empty content
and a usage object. All five registered commands exited with code `0`. The
dedicated content-free sensitivity audit reported `ok=true` and no issues:

| Condition | Traces | Memory events | Input tokens | Output tokens | Cache read | Reasoning | Requests | Retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_persistence | 1 | 0 | 30,843 | 1,691 | 29,696 | 837 | 10 | 0 |
| native_static | 3 | 11 | 50,812 | 2,351 | 13,824 | 667 | 15 | 0 |
| type_matched_oracle | 2 | 6 | 38,824 | 1,676 | 12,288 | 658 | 11 | 0 |
| shortcut_current_input | 1 | 0 | 10,951 | 468 | 1,536 | 103 | 4 | 0 |
| wrong_mechanism | 1 | 0 | 10,307 | 623 | 6,656 | 132 | 5 | 0 |

Native and oracle slices produced procedural memory events. The
no-persistence and control slices produced no memory events, as required by
their deployment contracts. Every trace had a terminal event with complete
usage; the audit found no usage mismatch, missing trace, retry, or opaque task
ID mismatch.

## Boundary And Next Gate

Task outputs and task scores remain inside PAST traces as audit-plane data. They
were not read by RSIMem policy, optimizer, process-signal metadata, or this
report. This single replicate supports neither a condition ranking nor a
sensitivity estimate. It confirms only that the five PC02 deployment paths can
be isolated, executed, and audited with the registered procedural oracle seed
and explicit controls.

The procedural panel now has accepted replicate-1 pilots for PC01 and PC02;
the remaining procedural families and predeclared matched replicates are still
pending. Raw resources remain reporting fields and are not policy rewards.
