# PC02 Patch-02 Procedural Sensitivity Pilot - 2026-09-03

This report records one bounded replicate-1 execution pilot for
`PC02_sop_patch_02`. It is execution and mechanism-readiness evidence, not a
completed Stage 3 sensitivity result. No candidate policy, optimizer input, or
N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five registered procedural conditions
- Replicate: `1`
- Execution order: `no_persistence`, `native_static`, `type_matched_oracle`,
  `shortcut_current_input`, `wrong_mechanism`
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
| no_persistence | 1 | 0 | 16,155 | 1,089 | 10,752 | 557 | 8 | 0 |
| native_static | 3 | 14 | 86,796 | 3,900 | 29,696 | 1,680 | 24 | 0 |
| type_matched_oracle | 2 | 6 | 21,713 | 1,416 | 12,800 | 720 | 10 | 0 |
| shortcut_current_input | 1 | 0 | 7,135 | 494 | 3,072 | 207 | 4 | 0 |
| wrong_mechanism | 1 | 0 | 10,782 | 616 | 5,632 | 259 | 6 | 0 |

Native and oracle slices produced procedural memory events. The
no-persistence and control slices produced no memory events, as required by
their deployment contracts. Every trace had a terminal event with complete
usage; the audit found no usage mismatch, missing trace, retry, or opaque task
ID mismatch.

## Boundary And Next Gate

Task outputs and task scores remain inside PAST traces as audit-plane data. They
were not read by RSIMem policy, optimizer, process-signal metadata, or this
report. This single replicate supports neither a condition ranking nor a
sensitivity estimate. It confirms only that the five PC02 patch-02 deployment
paths can be isolated, executed, and audited with the registered procedural
oracle seed and explicit controls.

The procedural panel now has accepted replicate-1 pilots for PC01 and both
PC02 patch families. Remaining procedural families and predeclared matched
replicates are still pending. Raw resources remain reporting fields and are not
policy rewards.
