# AdaMem Mem0 Full-Trajectory Validation: 26 Families

Status: `accepted and complete`
Date: `2026-09-13`
Protocol: `adamem-trajectory-baseline-v1`

## Scope

This is the frozen B0/B2 comparison required by the refactor handoff:

- B0: `Mem0Static`
- B2: `Mem0 + AdaMem full trajectory`
- 26 frozen families, three accepted replicates per family-condition
- deployment-visible trajectory only for the updater
- matched train -> Near/Far N+1 execution with isolated run roots

B1 terminal feedback was not materialized or run. No Hermes + AdaMem or
RSIMem-enhanced condition was run in this protocol.

## Formal Artifacts

The dedicated output root is
`outputs/adamem_mem0_full_trajectory_26family_20260912/`.

| Artifact | Path | Digest |
| --- | --- | --- |
| Manifest | `formal_manifest.json` | `3a558d406e19e95f06268762a5fdb0f25a99c8c6e4ec3595acefd7a57bcba40b` |
| Manifest preflight | `manifest_preflight.json` | `504f066f49897038b8138b912beef417c872d67a5d03432e1f76c2ba69d05c5c` |
| Materialization dry run | `dry_run_preflight.json` | `d249b871138d0cff1ccb8f2a8f74b903957f45de69193c2c4b1689f8716076fd` |
| Aggregate | `provider_runs/validation_aggregate.json` | `1b8067f33973b7bd2787391123bef0dd0eef9b31f0fe52e9c07826ef9a51cea5` |

The manifest fixes 26 families, three replicates, B0/B2 only, the
`mem0-semantic-v1` surface policy, and the recorded frozen code revision.
The provider-free preflight accepted the manifest. The dry run accepted 52
materialized batches and 156 run roots, with routing coverage EP 3, PC 10,
PG 6, and SM 7.

## Acceptance Accounting

| Evidence | Count |
| --- | ---: |
| Completed families | 26 |
| Accepted condition-family batches | 52 |
| Batch audits | 52 |
| Matched audits | 26 |
| Accepted B0 runs | 78 |
| Accepted B2 runs | 78 |
| Complete B2 updater usage records | 78 |

All accepted runs have the required phase results and usage evidence. Failed
attempts remain as provenance and are excluded from the aggregate.

## Aggregate Result

The pooled evaluation score across accepted evaluation episodes is:

| Condition | Accepted runs | Evaluation mean | SD |
| --- | ---: | ---: | ---: |
| B0 `Mem0Static` | 78 | 0.69204 | 0.21993 |
| B2 `Mem0 + AdaMem full trajectory` | 78 | 0.71190 | 0.21044 |

The pooled B2 minus B0 difference is `+0.01986`. This is a descriptive
comparison under the frozen protocol, not evidence that every update helps.

## Update And Resource Behavior

| Metric | B0 | B2 |
| --- | ---: | ---: |
| Update rate | n/a | 58/78 (74.36%) |
| Abstention rate | n/a | 20/78 (25.64%) |
| Rollback rate | n/a | 0/78 (0%) |
| Harmful update rate | n/a | 9/78 (11.54%) |
| Updater input tokens, mean (SD) | n/a | 11,357.7 (5,344.5) |
| Updater output tokens, mean (SD) | n/a | 229.1 (95.0) |
| Updater latency, mean (SD) | n/a | 4,833.3 ms (1,570.1) |
| Updater requests | n/a | one per B2 run |

`Harmful update` means that the matched N+1 macro score was lower than the
paired B0 result. This observed rate is a required part of the result.

## Group-Level Means

The following values are macro means over family-level replicate means. Near
and Far values are shown where the fixtures define those labels.

| Group | Families | B0 Near / Far | B2 Near / Far |
| --- | ---: | ---: | ---: |
| Semantic | 7 | 0.9625 / 0.9681 | 0.9684 / 0.9693 |
| Episodic | 3 | 0.5607 / 0.6066 | 0.5795 / 0.6063 |
| Procedural | 10 | 0.5278 / 0.5323 | 0.5645 / 0.5427 |
| Proactive retrieval | 6 | 0.6888 / 0.6983 | 0.6936 / 0.7077 |
| Overall | 26 | 0.7327 / 0.7356 | 0.7448 / 0.7411 |

Overall family-level macro means, including families without conventional
Near/Far labels, are B0 `0.6833` and B2 `0.7048`.

## Per-Family Appendix

Values are family-level means across three accepted replicates. `Update rate`
is the number of B2 replicates that activated a policy update.

| Family | Group | B0 | B2 | B2-B0 | Update rate |
| --- | --- | ---: | ---: | ---: | ---: |
| SM01_preference_adoption | Semantic | 1.000 | 1.000 | 0.000 | 2/3 |
| SM02_constraint_retention | Semantic | 0.977 | 0.977 | 0.000 | 3/3 |
| SM03_fact_correction | Semantic | 1.000 | 1.000 | 0.000 | 2/3 |
| SM04_rule_migration | Semantic | 1.000 | 0.977 | -0.023 | 3/3 |
| SM05_weak_trigger_preference_adoption | Semantic | 0.982 | 0.982 | 0.000 | 3/3 |
| SM06_temporary_exception_pollution | Semantic | 1.000 | 1.000 | 0.000 | 3/3 |
| SM07_scoped_rule_migration | Semantic | 0.840 | 0.876 | +0.036 | 3/3 |
| EP01_prior_case_recall | Episodic | 0.483 | 0.487 | +0.004 | 3/3 |
| EP02_exception_list_recall | Episodic | 0.741 | 0.765 | +0.024 | 3/3 |
| EP03_recall_then_modify | Episodic | 0.526 | 0.526 | 0.000 | 3/3 |
| PC01_sop_bootstrap_01 | Procedural | 0.742 | 0.759 | +0.017 | 3/3 |
| PC01_sop_bootstrap_02 | Procedural | 0.393 | 0.631 | +0.238 | 3/3 |
| PC01_sop_bootstrap_03 | Procedural | 0.742 | 0.725 | -0.017 | 3/3 |
| PC01_sop_bootstrap_04 | Procedural | 0.370 | 0.370 | 0.000 | 2/3 |
| PC01_sop_bootstrap_05 | Procedural | 0.880 | 0.880 | 0.000 | 3/3 |
| PC01_sop_bootstrap_06 | Procedural | 0.400 | 0.400 | 0.000 | 1/3 |
| PC02_sop_patch_01 | Procedural | 0.381 | 0.370 | -0.011 | 3/3 |
| PC02_sop_patch_02 | Procedural | 0.370 | 0.495 | +0.125 | 3/3 |
| PC03_latent_rule_induction_01 | Procedural | 0.370 | 0.499 | +0.129 | 1/3 |
| PC04_failure_to_rule_01 | Procedural | 0.415 | 0.415 | 0.000 | 2/3 |
| PG01_release_decision_followup | Proactive retrieval | 0.662 | 0.662 | 0.000 | 1/3 |
| PG02_ops_exception_desk | Proactive retrieval | 0.696 | 0.696 | 0.000 | 1/3 |
| PG03_oncall_handoff_lookup | Proactive retrieval | 0.662 | 0.662 | 0.000 | 1/3 |
| PG04_temporary_waiver_audit | Proactive retrieval | 0.696 | 0.696 | 0.000 | 1/3 |
| PG05_change_freeze_followup | Proactive retrieval | 0.662 | 0.662 | 0.000 | 1/3 |
| PG06_kappa_integration_review | Proactive retrieval | 0.772 | 0.810 | +0.038 | 1/3 |

## Interpretation Limits

- B1 terminal feedback was not run, so no B1 comparison is available.
- No RSIMem-enhanced condition was run; this report is not an RSIMem
  effectiveness result.
- The comparison supports statements about this frozen protocol and accepted
  artifacts only. It does not establish a causal mechanism for cross-family
  transfer or generalize beyond these fixtures and model/provider settings.
- Some families are saturated, limiting their usefulness for quality-uplift
  claims.
- The nonzero harmful-update rate means the B2 mean should be reported with
  update safety behavior, not as an unconditional improvement claim.
