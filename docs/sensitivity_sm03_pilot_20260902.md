# SM03 Semantic Sensitivity Pilot - 2026-09-02

This report records one bounded replicate-1 execution pilot for the semantic
`SM03_fact_correction` family. It is execution and mechanism-readiness
evidence, not a completed Stage 3 sensitivity result. No candidate policy,
optimizer input, or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five registered semantic conditions
- Replicate: `1`
- Execution order: `no_persistence`, `native_static`, `type_matched_oracle`,
  `shortcut_current_input`, `wrong_mechanism`
- Judge: disabled
- Method boundary: opaque sensitivity case ID only
- Oracle basis: checked-in semantic seed manually authored from the public
  learn/update input; no grader, answer, expectation, or official score was
  used to author the seed or method metadata.

## Execution Audit

The provider completion probe returned HTTP `200` with non-empty content and a
usage object. All five registered commands exited with code `0`. The dedicated
content-free sensitivity audit reported `ok=true` and no issues:

| Condition | Traces | Memory events | Input tokens | Output tokens | Cache read | Reasoning | Requests | Retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_persistence | 1 | 0 | 5,506 | 429 | 1,536 | 273 | 4 | 0 |
| native_static | 3 | 28 | 22,178 | 1,841 | 15,872 | 1,058 | 14 | 0 |
| type_matched_oracle | 2 | 20 | 18,653 | 1,346 | 8,192 | 858 | 9 | 0 |
| shortcut_current_input | 1 | 0 | 13,680 | 1,339 | 7,168 | 749 | 9 | 0 |
| wrong_mechanism | 1 | 0 | 14,079 | 1,657 | 10,240 | 1,129 | 8 | 0 |

Native and oracle slices produced semantic memory events. The no-persistence
and control slices produced no memory events, as required by their deployment
contracts. Every trace had a terminal event with complete usage; the audit
found no usage mismatch, missing trace, retry, or opaque task-ID mismatch.

## Boundary And Next Gate

Task outputs and task scores remain inside PAST traces as audit-plane data. They
were not read by RSIMem policy, optimizer, process-signal metadata, or this
report. This single replicate supports neither a condition ranking nor a
sensitivity estimate. It confirms only that the five SM03 deployment paths can
be isolated, executed, and audited with the registered semantic oracle seed and
explicit controls.

The semantic panel still needs its predeclared matched replicates and the
remaining SM families' pilots. Raw resources remain reporting fields and are
not policy rewards.
