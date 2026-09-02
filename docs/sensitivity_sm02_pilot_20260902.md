# SM02 Sensitivity Pilot - 2026-09-02

This report records one bounded, replicate-1 execution pilot for the semantic
`SM02_constraint_retention` family. It is infrastructure and mechanism-readiness
evidence, not a completed Stage 3 sensitivity result. No candidate policy,
optimizer input, or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five semantic conditions
- Replicate: `1`
- Execution order: `no_persistence`, `native_static`, `type_matched_oracle`,
  `shortcut_current_input`, `wrong_mechanism`
- Judge: disabled
- Method boundary: opaque sensitivity case ID only
- Oracle basis: checked-in semantic seed manually authored from the public
  SM02 learn/update inputs; no grader, answer, expectation, or official score
  was used to author the seed or method metadata.

## Execution Audit

The provider completion probe returned HTTP 200 with non-empty content and a
usage object. All five registered commands exited with code 0. The dedicated
content-free sensitivity audit reported `ok=true`:

| Condition | Traces | Memory events | Input tokens | Output tokens | Cache read | Reasoning | Requests | Retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_persistence | 1 | 0 | 3,253 | 275 | 0 | 219 | 2 | 0 |
| native_static | 5 | 48 | 33,360 | 2,389 | 13,312 | 1,478 | 19 | 0 |
| type_matched_oracle | 2 | 20 | 15,140 | 884 | 4,608 | 487 | 8 | 0 |
| shortcut_current_input | 1 | 0 | 7,253 | 892 | 4,096 | 620 | 5 | 0 |
| wrong_mechanism | 1 | 0 | 3,621 | 304 | 1,536 | 208 | 3 | 0 |

The native and oracle slices produced semantic memory events. The three
no-persistence/control slices produced no memory events, as required by their
deployment contract. Every trace had a terminal event with complete usage; the
audit found no usage mismatch, missing trace, or opaque task-ID mismatch.

## Boundary And Next Gate

The task outputs and task scores remain inside PAST traces as audit-plane data.
They were not read by RSIMem policy, optimizer, process-signal metadata, or
this report. This single replicate therefore supports neither a condition
ranking nor a sensitivity estimate. It confirms only that the five SM02
deployment paths can be isolated, executed, and audited with the registered
oracle seed.

The semantic panel still needs its predeclared matched replicates. Episodic
oracle seeds and procedural control tasks remain separate gates; raw resources
are reporting fields and are not policy rewards.
