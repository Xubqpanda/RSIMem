# Stage 3 Coverage Audit - 2026-09-03

This report is generated from the content-free pilot audits under
`outputs/sensitivity`. It does not read task scores, prompts, grader fields,
answers, or raw model responses. The reproducible command is:

```bash
PYTHONPATH=src .venv/bin/python -m rsimem.sensitivity_coverage \
  outputs/sensitivity --output outputs/sensitivity/stage3_coverage.json
```

The generated coverage manifest is regenerated after each accepted or excluded
attempt; its current ID is recorded in `outputs/sensitivity/stage3_coverage.json`.

| Panel | Accepted pilots | Excluded attempts | Accepted families | Replicates |
| --- | ---: | ---: | --- | --- |
| Semantic | 8 | 1 | SM01, SM02, SM03, SM04, SM05, SM06, SM07 | 1, 2, 3 (SM01 only) |
| Episodic | 6 | 1 | EP01, EP02, EP03 | 1, 2 (EP01/EP02/EP03) |
| Procedural | 10 | 4 | PC01 bootstrap-01/02/03/04/05/06, PC02 patch-01/02, PC03, PC04 | 1 |

Expected-family coverage is therefore semantic `7/7`, episodic `3/3`, and
procedural `10/10`. There is no missing semantic or procedural family.

The manifest also records readiness for replicate analysis. All three panels
are currently `ready_for_replicate_analysis=false`: episodic covers all three
families and all three have replicate 2, while semantic and procedural have
incomplete replicate coverage.

Every accepted pilot contains all five registered conditions with complete
content-free audit. The excluded attempts are PC02's first attempt
(`usage_incomplete` and retry mismatch), the earlier SM04 attempt
(`usage_incomplete` after provider timeouts), the first PC03 attempt, the PC01
bootstrap-02 attempt, and the PC01 bootstrap-03 attempt
(`run_not_completed`/`sequence_results_missing`).
The accepted retry-4 SM04 pilot and PC03 retry-2 pilot replace their earlier
failed attempts and are included above. The incomplete PC01 bootstrap-02
attempt is superseded by its accepted retry-2 pilot. The incomplete PC01
bootstrap-03 attempt is likewise superseded by its accepted retry-2 pilot.
Excluded attempts remain in the manifest for
provenance but are excluded from all coverage counts and sensitivity
denominators.

The EP02 replicate-2 attempt is additionally excluded after
`usage_incomplete` control traces; it requires a fresh retry. Excluded
attempts remain in the manifest for provenance but are excluded from coverage
counts and sensitivity denominators.

This is a coverage/readiness audit, not a sensitivity status. Stage 3 still
requires predeclared replicate coverage, family-level paired deltas, variation, oracle coverage, and explicit
mechanism interpretation before `SENSITIVE`, `PARTIALLY_SENSITIVE`,
`INSENSITIVE`, or `INVALID` can be assigned.
