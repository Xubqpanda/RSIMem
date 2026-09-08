# SM04 Semantic Sensitivity Pilot - 2026-09-03

This report records the accepted retry-4 replicate-1 execution pilot for the
semantic `SM04_rule_migration` family. It is execution and
mechanism-readiness evidence, not a completed Stage 3 sensitivity result. No
candidate policy, optimizer input, or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five registered semantic conditions
- Replicate: `1`
- Batch: `semantic-sm04-r01-retry4-20260903_182231`
- Execution order: `no_persistence`, `native_static`, `type_matched_oracle`,
  `shortcut_current_input`, `wrong_mechanism`
- Judge: disabled
- Method boundary: opaque sensitivity case ID only
- Oracle basis: checked-in semantic seed manually authored from public
  learn/update inputs; no grader, answer, expectation, or official score was
  used to author the seed or method metadata.

Earlier SM04 batches are retained as provider/infrastructure attempts and are
excluded from this accepted pilot and all sensitivity denominators.

## Execution Audit

The retry-4 provider completion probe returned HTTP `200` with non-empty content
and a usage object. All five registered commands exited with code `0`. The
dedicated content-free sensitivity audit reported `ok=true` and no issues:

| Condition | Traces | Memory events | Input tokens | Output tokens | Cache read | Reasoning | Requests | Retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_persistence | 1 | 0 | 5,184 | 388 | 0 | 290 | 3 | 0 |
| native_static | 3 | 28 | 23,823 | 1,196 | 6,144 | 619 | 12 | 0 |
| type_matched_oracle | 2 | 20 | 16,437 | 588 | 3,072 | 304 | 8 | 0 |
| shortcut_current_input | 1 | 0 | 6,559 | 478 | 3,072 | 291 | 5 | 0 |
| wrong_mechanism | 1 | 0 | 18,474 | 760 | 3,584 | 434 | 6 | 0 |

Native and oracle slices produced semantic memory events. The no-persistence
and control slices produced no memory events, as required by their deployment
contracts. Every trace had a terminal event with complete usage; the audit
found no usage mismatch, missing trace, retry, or opaque task-ID mismatch.

## Boundary And Next Gate

Task outputs and task scores remain inside PAST traces as audit-plane data. They
were not read by RSIMem policy, optimizer, process-signal metadata, or this
report. This single replicate supports neither a condition ranking nor a
sensitivity estimate. It confirms only that the five SM04 deployment paths can
be isolated, executed, and audited with the registered semantic oracle seed and
explicit controls.

The semantic panel now has accepted replicate-1 pilots for all seven families.
It still needs the predeclared matched replicates, family-level paired analysis,
and mechanism interpretation. Raw resources remain reporting fields and are
not policy rewards.
