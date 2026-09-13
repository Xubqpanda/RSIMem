# Planned Main Tables

Status: `Stage 1 freeze and full-suite B0/B2 AdaMem comparison complete; B1 and RSIMem pending`
Last updated: 2026-09-13
Protocol: `adamem-trajectory-baseline-v1`

This document is the paper-facing table. Only the completed full-suite B0/B2
comparison is populated; B1, Hermes + AdaMem, and RSIMem remain blank because
they were not run under this frozen protocol. Do not insert screening values.

## Reporting Unit

One family-condition value is the mean task score over that family's matched
Near/Far N+1 evaluation episodes for one accepted replicate. Each method is
run for three independent accepted replicates per family. Report every group as
the macro-average over its accepted family means, followed by replicate
variation in parentheses.

| Group | Families | Role |
| --- | ---: | --- |
| Semantic | 7 | Direct target of the current Semantic extraction-policy update |
| Episodic | 3 | Cross-workflow effect of the Semantic policy update |
| Procedural | 10 | Cross-workflow effect of the Semantic policy update |
| Proactive retrieval | 6 | Cross-workflow effect of the Semantic policy update |
| Overall | 26 | Macro-average over all accepted families; no cherry-picking |

If a family is excluded because of provider, service, or task-integration
failure, report its identity and reason below the table. Never replace it with
another family. If Semantic Memory does not write, retrieve, or inject during a
family, retain the result and mark the family `mechanism-inapplicable` in the
per-family appendix; do not silently omit it from the coverage report.

## Table 1: Full-Suite Quality

Each value is `mean (SD)` over the three accepted replicates. Higher is better.

| Layer | Method | Semantic (7) | Episodic (3) | Procedural (10) | Proactive retrieval (6) | Overall (26) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| L0 | AllMemoryOff | 0.3962 (0.0101) / 0.3943 (0.0151) | 0.3133 (0.0793) / 0.3083 (0.0880) | 0.3560 (0.0556) / 0.3560 (0.0556) | 0.2968 (0.0668) / 0.3060 (0.0707) | 0.3482 (0.0012) / 0.3493 (0.0013) |
| L1 | HermesNative |  |  |  |  |  |
| L1 | Mem0Static (B0) | 0.9625 / 0.9681 | 0.5607 / 0.6066 | 0.5278 / 0.5323 | 0.6888 / 0.6983 | 0.7327 / 0.7356 |
| L2 | Mem0 + AdaMem terminal (B1) |  |  |  |  |  |
| L2 | Mem0 + AdaMem full trajectory (B2) | 0.9684 / 0.9693 | 0.5795 / 0.6063 | 0.5645 / 0.5427 | 0.6936 / 0.7077 | 0.7448 / 0.7411 |
| L3 | Mem0 + AdaMem + RSIMem |  |  |  |  |  |
| L2 | HermesNative + AdaMem |  |  |  |  |  |
| L3 | HermesNative + AdaMem + RSIMem |  |  |  |  |  |

Values are shown as `Near / Far`. The `AllMemoryOff` row is frozen from the
78 accepted-run control input at
`outputs/all_memory_off_formal_20260911/all_memory_off_main_table_input.json`.

Rows not in the currently frozen experiment are intentionally blank:

- The historical semantic-disabled control is not a valid all-memory-off row
  and is not included in this table.
- The first formal comparison is the B0/B1/B2 Mem0 path.
- L3 and HermesNative + AdaMem rows are future work and must not be populated
  with proxy results.

The completed B0/B2 machine-readable aggregate is
`outputs/adamem_mem0_full_trajectory_26family_20260912/provider_runs/validation_aggregate.json`.
Values are Near / Far macro means across accepted family means. The B2 row is
not an RSIMem result, and the aggregate includes 9/78 harmful updates.

## Table 2: Update Behavior and Resource Cost

Quality and updater behavior are reported separately. `Update rate` is the
fraction of accepted replicate-family runs that activated a new policy;
`harmful update rate` is the fraction whose matched N+1 macro score is lower
than B0 for the paired replicate. Resource fields cover the meta-agent updater,
not the base Agent task calls.

| Method | Update rate | Abstention rate | Rollback rate | Harmful update rate | Updater input tokens | Updater calls | Updater latency | N+1 delta vs. B0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mem0Static (B0) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.000 |
| Mem0 + AdaMem terminal (B1) | not run | not run | not run | not run | not run | not run | not run | not run |
| Mem0 + AdaMem full trajectory (B2) | 74.36% | 25.64% | 0% | 11.54% | 11,357.7 | 1/run | 4,833.3 ms | +0.01986 |
| Mem0 + AdaMem + RSIMem |  |  |  |  |  |  |  |  |

## Table 3: Per-Family Appendix

This appendix records the accepted family-level means and update counts. It
prevents group averages from hiding saturation or harmful updates.

| Family | Group | Semantic mechanism applicable | B0 | B1 | B2 | B2-B0 | B2-B1 | B2 update rate | Notes / exclusions |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SM01_preference_adoption | Semantic | yes | 1.000 | not run | 1.000 | 0.000 | n/a | 2/3 | Saturated; mechanism/resource evidence |
| SM02_constraint_retention | Semantic | yes | 0.977 | not run | 0.977 | 0.000 | n/a | 3/3 | |
| SM03_fact_correction | Semantic | yes | 1.000 | not run | 1.000 | 0.000 | n/a | 2/3 | |
| SM04_rule_migration | Semantic | yes | 1.000 | not run | 0.977 | -0.023 | n/a | 3/3 | |
| SM05_weak_trigger_preference_adoption | Semantic | yes | 0.982 | not run | 0.982 | 0.000 | n/a | 3/3 | |
| SM06_temporary_exception_pollution | Semantic | yes | 1.000 | not run | 1.000 | 0.000 | n/a | 3/3 | |
| SM07_scoped_rule_migration | Semantic | yes | 0.840 | not run | 0.876 | +0.036 | n/a | 3/3 | |
| EP01_prior_case_recall | Episodic | cross-surface | 0.483 | not run | 0.487 | +0.004 | n/a | 3/3 | |
| EP02_exception_list_recall | Episodic | cross-surface | 0.741 | not run | 0.765 | +0.024 | n/a | 3/3 | |
| EP03_recall_then_modify | Episodic | cross-surface | 0.526 | not run | 0.526 | 0.000 | n/a | 3/3 | |
| PC01_sop_bootstrap_01 | Procedural | cross-surface | 0.742 | not run | 0.759 | +0.017 | n/a | 3/3 | |
| PC01_sop_bootstrap_02 | Procedural | cross-surface | 0.393 | not run | 0.631 | +0.238 | n/a | 3/3 | |
| PC01_sop_bootstrap_03 | Procedural | cross-surface | 0.742 | not run | 0.725 | -0.017 | n/a | 3/3 | |
| PC01_sop_bootstrap_04 | Procedural | cross-surface | 0.370 | not run | 0.370 | 0.000 | n/a | 2/3 | |
| PC01_sop_bootstrap_05 | Procedural | cross-surface | 0.880 | not run | 0.880 | 0.000 | n/a | 3/3 | |
| PC01_sop_bootstrap_06 | Procedural | cross-surface | 0.400 | not run | 0.400 | 0.000 | n/a | 1/3 | |
| PC02_sop_patch_01 | Procedural | cross-surface | 0.381 | not run | 0.370 | -0.011 | n/a | 3/3 | |
| PC02_sop_patch_02 | Procedural | cross-surface | 0.370 | not run | 0.495 | +0.125 | n/a | 3/3 | |
| PC03_latent_rule_induction_01 | Procedural | cross-surface | 0.370 | not run | 0.499 | +0.129 | n/a | 1/3 | |
| PC04_failure_to_rule_01 | Procedural | cross-surface | 0.415 | not run | 0.415 | 0.000 | n/a | 2/3 | |
| PG01_release_decision_followup | Proactive retrieval | cross-surface | 0.662 | not run | 0.662 | 0.000 | n/a | 1/3 | |
| PG02_ops_exception_desk | Proactive retrieval | cross-surface | 0.696 | not run | 0.696 | 0.000 | n/a | 1/3 | |
| PG03_oncall_handoff_lookup | Proactive retrieval | cross-surface | 0.662 | not run | 0.662 | 0.000 | n/a | 1/3 | |
| PG04_temporary_waiver_audit | Proactive retrieval | cross-surface | 0.696 | not run | 0.696 | 0.000 | n/a | 1/3 | |
| PG05_change_freeze_followup | Proactive retrieval | cross-surface | 0.662 | not run | 0.662 | 0.000 | n/a | 1/3 | |
| PG06_kappa_integration_review | Proactive retrieval | cross-surface | 0.772 | not run | 0.810 | +0.038 | n/a | 1/3 | |

## Current Formal Evidence

The full-suite B0/B2 result supersedes the earlier SM01-SM03 execution-only
headline. B1 was not materialized or run in this protocol. The detailed result,
including audit counts, group summaries, and limitations, is in
[`adamem_mem0_full_trajectory_26family_results_20260913.md`](adamem_mem0_full_trajectory_26family_results_20260913.md).

The frozen AllMemoryOff control is available as a separate machine-readable
main-table input at
`outputs/all_memory_off_formal_20260911/all_memory_off_main_table_input.json`.
The historical `NoMemory`/`SemanticDisabled` row has been removed from the
active comparison and is not used as a paper baseline.
