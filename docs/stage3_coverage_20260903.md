# Stage 3 Coverage Audit - 2026-09-03

This report is generated from the content-free pilot audits under
`outputs/sensitivity`. It does not read task scores, prompts, grader fields,
answers, or raw model responses. The reproducible command is:

```bash
PYTHONPATH=src .venv/bin/python -m rsimem.sensitivity_coverage \
  outputs/sensitivity --output outputs/sensitivity/stage3_coverage.json
```

The generated coverage manifest has ID
`sensitivity-coverage.43be847b4d5ea8bc3cfe36821cb5c9aa3c3bc5d0`.

| Panel | Accepted pilots | Excluded attempts | Accepted families | Replicates |
| --- | ---: | ---: | --- | --- |
| Semantic | 6 | 1 | SM01, SM02, SM03, SM05, SM06, SM07 | 1, 2 (SM01 only) |
| Episodic | 3 | 0 | EP01, EP02, EP03 | 1 |
| Procedural | 2 | 2 | PC01, PC02 | 1 |

Expected-family coverage is therefore semantic `6/7`, episodic `3/3`, and
procedural `2/10`. The missing semantic family is `SM04_rule_migration`; the
missing procedural families are `PC01_sop_bootstrap_02..06`,
`PC02_sop_patch_02`, `PC03_latent_rule_induction_01`, and
`PC04_failure_to_rule_01`.

The manifest also records readiness for replicate analysis. All three panels
are currently `ready_for_replicate_analysis=false`: episodic covers all three
families but has only replicate 1, while semantic and procedural still have
missing families in addition to incomplete replicate coverage.

Every accepted pilot contains all five registered conditions with complete
content-free audit. The excluded attempts are PC02's first attempt
(`usage_incomplete` and retry mismatch), SM04 (`usage_incomplete` after
provider timeouts), and PC03 (`run_not_completed`/`sequence_results_missing`).
They remain in the manifest for provenance but are excluded from all coverage
counts and sensitivity denominators.

This is a coverage/readiness audit, not a sensitivity status. Stage 3 still
requires clean coverage for the excluded families, predeclared replicate
coverage, family-level paired deltas, variation, oracle coverage, and explicit
mechanism interpretation before `SENSITIVE`, `PARTIALLY_SENSITIVE`,
`INSENSITIVE`, or `INVALID` can be assigned.
