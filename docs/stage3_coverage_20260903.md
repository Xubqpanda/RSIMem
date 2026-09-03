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
| Semantic | 7 | 1 | SM01, SM02, SM03, SM04, SM05, SM06, SM07 | 1, 2 (SM01 only) |
| Episodic | 3 | 0 | EP01, EP02, EP03 | 1 |
| Procedural | 4 | 3 | PC01, PC02 patch-01, PC02 patch-02, PC03 | 1 |

Expected-family coverage is therefore semantic `7/7`, episodic `3/3`, and
procedural `4/10`. There is no missing semantic family; the
missing procedural families are `PC01_sop_bootstrap_02..06`,
`PC04_failure_to_rule_01`.

The manifest also records readiness for replicate analysis. All three panels
are currently `ready_for_replicate_analysis=false`: episodic covers all three
families but has only replicate 1, while semantic and procedural still have
missing families in addition to incomplete replicate coverage.

Every accepted pilot contains all five registered conditions with complete
content-free audit. The excluded attempts are PC02's first attempt
(`usage_incomplete` and retry mismatch), the earlier SM04 attempt
(`usage_incomplete` after provider timeouts), the first PC03 attempt, and the
PC01 bootstrap-02 attempt (`run_not_completed`/`sequence_results_missing`).
The accepted retry-4 SM04 pilot and PC03 retry-2 pilot replace their earlier
failed attempts and are included above. The PC01 bootstrap-02 attempt remains
incomplete and has no accepted replacement. Excluded attempts remain in the
manifest for provenance but are excluded from all coverage counts and
sensitivity denominators.

This is a coverage/readiness audit, not a sensitivity status. Stage 3 still
requires clean coverage for the excluded families, predeclared replicate
coverage, family-level paired deltas, variation, oracle coverage, and explicit
mechanism interpretation before `SENSITIVE`, `PARTIALLY_SENSITIVE`,
`INSENSITIVE`, or `INVALID` can be assigned.
