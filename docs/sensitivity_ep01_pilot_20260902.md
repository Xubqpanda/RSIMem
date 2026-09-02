# EP01 Episodic Sensitivity Pilot - 2026-09-02

This report records one bounded replicate-1 execution pilot for the episodic
`EP01_prior_case_recall` family. It is execution and mechanism-readiness
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
| no_persistence | 1 | 0 | 4,497 | 272 | 1,536 | 139 | 3 | 0 |
| native_static | 4 | 14 | 36,872 | 1,804 | 22,528 | 1,021 | 19 | 0 |
| type_matched_oracle | 2 | 12 | 15,775 | 1,275 | 10,752 | 777 | 9 | 0 |
| shortcut_current_input | 4 | 0 | 18,351 | 845 | 7,680 | 454 | 13 | 0 |
| wrong_mechanism | 4 | 0 | 16,047 | 1,226 | 7,680 | 809 | 12 | 0 |

Native and oracle slices produced episodic memory events. The
no-persistence and control slices produced no memory events, as required by
their deployment contracts. Every trace had a terminal event with complete
usage; the audit found no usage mismatch, missing trace, retry, or opaque task
ID mismatch.

## Boundary And Next Gate

Task outputs and task scores remain inside PAST traces as audit-plane data. They
were not read by RSIMem policy, optimizer, process-signal metadata, or this
report. This single replicate supports neither a condition ranking nor a
sensitivity estimate. It confirms only that the five EP01 deployment paths can
be isolated, executed, and audited with the registered episodic oracle seed and
explicit controls.

The episodic panel still needs its predeclared matched replicates and the EP02
and EP03 family pilots. Raw resources remain reporting fields and are not policy
rewards.
