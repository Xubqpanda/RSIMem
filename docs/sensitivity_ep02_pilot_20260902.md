# EP02 Episodic Sensitivity Pilot - 2026-09-02

This report records one bounded replicate-1 execution pilot for the episodic
`EP02_exception_list_recall` family. It is execution and mechanism-readiness
evidence, not a completed Stage 3 sensitivity result. No candidate policy,
optimizer input, or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five registered episodic conditions
- Replicate: `1`
- Execution order: `no_persistence`, `native_static`, `type_matched_oracle`,
  `shortcut_current_input`, `wrong_mechanism`
- Judge: disabled
- Method boundary: opaque sensitivity case ID only
- Oracle basis: checked-in episodic seed manually authored from the public
  learn input; no grader, answer, expectation, or official score was used to
  author the seed or method metadata.

## Execution Audit

The provider completion probe returned HTTP `200` with non-empty content and a
usage object. All five registered commands exited with code `0`. The dedicated
content-free sensitivity audit reported `ok=true` and no issues:

| Condition | Traces | Memory events | Input tokens | Output tokens | Cache read | Reasoning | Requests | Retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_persistence | 1 | 0 | 2,220 | 198 | 1,536 | 130 | 2 | 0 |
| native_static | 4 | 8 | 35,464 | 1,215 | 10,752 | 323 | 15 | 0 |
| type_matched_oracle | 2 | 8 | 11,846 | 737 | 8,192 | 178 | 7 | 0 |
| shortcut_current_input | 4 | 0 | 21,835 | 1,615 | 10,240 | 664 | 14 | 0 |
| wrong_mechanism | 4 | 0 | 24,070 | 2,408 | 13,312 | 1,223 | 15 | 0 |

Native and oracle slices produced episodic memory events. The
no-persistence and control slices produced no memory events, as required by
their deployment contracts. Every trace had a terminal event with complete
usage; the audit found no usage mismatch, missing trace, retry, or opaque task
ID mismatch.

## Boundary And Next Gate

Task outputs and task scores remain inside PAST traces as audit-plane data. They
were not read by RSIMem policy, optimizer, process-signal metadata, or this
report. This single replicate supports neither a condition ranking nor a
sensitivity estimate. It confirms only that the five EP02 deployment paths can
be isolated, executed, and audited with the registered episodic oracle seed and
explicit controls.

The episodic panel still needs its predeclared matched replicates and the EP03
family pilot. Raw resources remain reporting fields and are not policy rewards.
