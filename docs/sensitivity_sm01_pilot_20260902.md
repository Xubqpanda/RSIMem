# SM01 Sensitivity Pilot - 2026-09-02

This report records one bounded, replicate-2 execution pilot for the semantic
`SM01_preference_adoption` family. It is infrastructure and mechanism-readiness
evidence, not a completed Stage 3 sensitivity result. No candidate policy,
optimizer input, or N+1 update was produced.

## Protocol Boundary

- Provider: `coding.tu-zi.com/v1`
- Model: `gpt-5.6-luna`
- Conditions: all five semantic conditions
- Replicate: `2`
- Execution order: `native_static`, `type_matched_oracle`,
  `shortcut_current_input`, `wrong_mechanism`, `no_persistence`
- Judge: disabled
- Method boundary: opaque sensitivity case ID only
- Oracle basis: checked-in semantic seed manually authored from the public
  SM01 learn input; no grader, answer, expectation, or official score was
  used to author the seed or method metadata.

## Execution Audit

The provider completion probe returned HTTP 200 with non-empty content and a
usage object. All five registered commands exited with code 0. The dedicated
content-free sensitivity audit reported `ok=true`:

| Condition | Traces | Memory events | Input tokens | Output tokens | Cache read | Reasoning | Requests | Retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native_static | 5 | 48 | 31,414 | 1,966 | 12,288 | 1,139 | 18 | 0 |
| type_matched_oracle | 2 | 20 | 14,523 | 978 | 7,680 | 581 | 9 | 0 |
| shortcut_current_input | 1 | 0 | 5,723 | 396 | 1,536 | 242 | 4 | 0 |
| wrong_mechanism | 1 | 0 | 5,754 | 467 | 1,536 | 280 | 4 | 0 |
| no_persistence | 1 | 0 | 5,071 | 260 | 0 | 166 | 3 | 0 |

The native and oracle slices produced semantic memory events. The three
no-persistence/control slices produced no memory events, as required by their
deployment contract. Every trace had a terminal event with complete usage;
the audit found no usage mismatch, missing trace, or opaque task-ID mismatch.

## Observed Task Signals

The pilot's benchmark task scores are retained in the PAST trace and sequence
summary as audit-plane output only. They are not entered into RSIMem policy,
optimizer, or process-signal metadata. The run therefore does not support a
quality claim, a condition ranking, or a sensitivity estimate. It only confirms
that the five SM01 deployment paths can be isolated, executed, and audited.

## Limitations And Next Gate

This is one unseeded model replicate and uses the existing Hermes native memory
path with observer ledger instrumentation. It does not establish adapter
equivalence, cross-replicate sensitivity, or an optimization-ready pure-process
signal. Before broader Stage 3 execution, add the remaining semantic oracle
seeds, then complete episodic and procedural host assets and collect the
predeclared replicate set. Raw resource quantities remain reporting fields;
they are not a policy reward.
