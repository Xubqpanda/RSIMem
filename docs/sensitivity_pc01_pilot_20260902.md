# PC01 Procedural Sensitivity Pilot - 2026-09-02

This report records one bounded, replicate-1 execution pilot for the procedural
`PC01_sop_bootstrap_01` family. It is infrastructure and mechanism-readiness
evidence, not a completed Stage 3 sensitivity result. No candidate policy,
optimizer input, or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five procedural conditions
- Replicate: `1`
- Execution order: `no_persistence`, `native_static`, `type_matched_oracle`,
  `shortcut_current_input`, `wrong_mechanism`
- Judge: disabled
- Method boundary: opaque sensitivity case ID only
- Oracle basis: checked-in procedural skill seed manually authored from public
  learn/update inputs; no grader, answer, expectation, or official score was
  used to author the seed or method metadata.

## Execution Audit

The provider completion probe returned HTTP 200 with non-empty content and a
usage object. All five registered commands exited with code 0. The dedicated
content-free sensitivity audit reported `ok=true`:

| Condition | Traces | Memory events | Input tokens | Output tokens | Cache read | Reasoning | Requests | Retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_persistence | 1 | 0 | 13,630 | 1,101 | 3,072 | 654 | 5 | 0 |
| native_static | 4 | 12 | 84,573 | 3,224 | 31,232 | 724 | 25 | 0 |
| type_matched_oracle | 2 | 6 | 30,093 | 1,705 | 17,408 | 831 | 12 | 0 |
| shortcut_current_input | 1 | 0 | 9,999 | 541 | 1,536 | 122 | 4 | 0 |
| wrong_mechanism | 1 | 0 | 12,933 | 665 | 4,096 | 220 | 6 | 0 |

The native and oracle slices produced procedural skill events. The three
no-persistence/control slices produced no memory events, as required by their
deployment contract. Every trace had a terminal event with complete usage; the
audit found no usage mismatch, missing trace, or opaque task-ID mismatch.

## Boundary And Next Gate

Task outputs and task scores remain inside PAST traces as audit-plane data. They
were not read by RSIMem policy, optimizer, process-signal metadata, or this
report. This single replicate therefore supports neither a condition ranking
nor a sensitivity estimate. It confirms only that the five PC01 deployment
paths can be isolated, executed, and audited with the registered procedural
oracle seed and explicit controls.

The procedural panel still needs its predeclared matched replicates and the
remaining nine families' pilots. Episodic oracle seeds remain a separate gate;
raw resources are reporting fields and are not policy rewards.
