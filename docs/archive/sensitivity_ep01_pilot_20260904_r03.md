# EP01 Prior Case Recall Sensitivity Pilot - 2026-09-04 (Replicate 3)

This report records the accepted replicate-3 execution pilot for
`EP01_prior_case_recall`. It is execution and mechanism-readiness evidence,
not a completed Stage 3 sensitivity result. No candidate policy, optimizer
input, or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five registered episodic conditions
- Replicate: `3`
- Execution order: `type_matched_oracle`, `shortcut_current_input`,
  `wrong_mechanism`, `no_persistence`, `native_static`
- Batch: `episodic-ep01-r03-20260904_153439`
- Judge: disabled
- Method boundary: opaque sensitivity case ID only
- Oracle basis: checked-in episodic seed manually authored from public prior
  case inputs; no grader, answer, expectation, or official score was used to
  author the seed or method metadata.

## Execution Audit

The provider completion probe returned HTTP `200` with non-empty content and a
usage object. All five registered commands exited with code `0`. The dedicated
content-free sensitivity audit reported `ok=true` and no issues:

| Condition | Traces | Memory events | Input tokens | Output tokens | Cache read | Reasoning | Requests | Retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| type_matched_oracle | 2 | 14 | 17,621 | 1,319 | 11,776 | 874 | 10 | 0 |
| shortcut_current_input | 4 | 0 | 21,255 | 961 | 4,608 | 581 | 13 | 0 |
| wrong_mechanism | 4 | 0 | 12,954 | 1,178 | 10,752 | 766 | 12 | 0 |
| no_persistence | 1 | 0 | 6,033 | 300 | 0 | 160 | 3 | 0 |
| native_static | 4 | 8 | 41,384 | 1,315 | 8,192 | 713 | 17 | 0 |

Native and oracle slices produced episodic memory events. The no-persistence
and control slices produced no memory events, as required by their deployment
contracts. Every trace had a terminal event with complete usage; the audit
found no usage mismatch, missing trace, retry, or opaque task ID mismatch.

## Boundary And Next Gate

Task outputs and task scores remain inside PAST traces as audit-plane data.
They were not read by RSIMem policy, optimizer, process-signal metadata, or
this report. This single replicate supports neither a condition ranking nor a
sensitivity estimate. It confirms only that the five EP01 prior-case-recall
deployment paths can be isolated, executed, and audited with the registered
episodic oracle seed and explicit controls.

EP01 now has replicate-1, replicate-2, and replicate-3 pilots. EP02 and EP03
still require replicate-3 execution. Raw resources remain reporting fields and
are not policy rewards.
