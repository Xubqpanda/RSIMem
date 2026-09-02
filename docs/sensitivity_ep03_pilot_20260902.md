# EP03 Episodic Sensitivity Pilot - 2026-09-02

This report records one bounded replicate-1 execution pilot for the episodic
`EP03_recall_then_modify` family. It is execution and mechanism-readiness
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
| no_persistence | 3 | 0 | 39,380 | 3,137 | 16,896 | 2,187 | 18 | 0 |
| native_static | 3 | 66 | 68,762 | 2,480 | 71,680 | 1,340 | 32 | 0 |
| type_matched_oracle | 2 | 8 | 20,890 | 660 | 8,704 | 266 | 10 | 0 |
| shortcut_current_input | 3 | 0 | 45,472 | 3,744 | 23,552 | 2,835 | 19 | 0 |
| wrong_mechanism | 3 | 0 | 50,776 | 4,486 | 30,720 | 3,147 | 22 | 0 |

Native and oracle slices produced episodic memory events. The
no-persistence and control slices produced no memory events, as required by
their deployment contracts. Every trace had a terminal event with complete
usage; the audit found no usage mismatch, missing trace, retry, or opaque task
ID mismatch.

## Boundary And Next Gate

Task outputs and task scores remain inside PAST traces as audit-plane data. They
were not read by RSIMem policy, optimizer, process-signal metadata, or this
report. This single replicate supports neither a condition ranking nor a
sensitivity estimate. It confirms only that the five EP03 deployment paths can
be isolated, executed, and audited with the registered episodic oracle seed and
explicit controls.

The episodic panel now has one audited replicate for each of EP01, EP02, and
EP03. It still needs the predeclared matched replicates and family-level paired
analysis before any sensitivity status can be assigned. Raw resources remain
reporting fields and are not policy rewards.
