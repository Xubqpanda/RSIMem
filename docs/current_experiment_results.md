# Current Experiment Results

Status: `26-family B0/B2 full-trajectory validation accepted; B1 and RSIMem pending`
Last updated: 2026-09-13
Active protocol: `adamem-trajectory-baseline-v1`

## Refactor Validation: B0/B2 Full Trajectory

Status: `accepted for all 26 frozen families`
Manifest: `outputs/adamem_mem0_full_trajectory_26family_20260912/formal_manifest.json`
Aggregate: `outputs/adamem_mem0_full_trajectory_26family_20260912/provider_runs/validation_aggregate.json`
Manifest digest: `3a558d406e19e95f06268762a5fdb0f25a99c8c6e4ec3595acefd7a57bcba40b`

The accepted comparison contains three matched replicates for each B0
`Mem0Static` and B2 `Mem0 + AdaMem full trajectory` condition across all 26
frozen families. It has 52 accepted condition-family batches, 52 batch audits,
26 matched audits, and 156 accepted runs (78 B0 and 78 B2). All B2 runs have
complete updater usage and accepted prefix/suffix phase results.

B0 pooled evaluation mean is `0.69204`; B2 is `0.71190`. B2 updated 58/78
runs, abstained 20/78, rolled back 0/78, and had 9/78 harmful updates. Mean
updater use is 11,357.7 input tokens, 229.1 output tokens, and 4,833.3 ms
latency. The nonzero harmful-update rate is part of the result: this evidence
does not support wording that full-trajectory updates always improve quality.

The detailed audit, group summaries, per-family appendix, and interpretation
limits are in
[`adamem_mem0_full_trajectory_26family_results_20260913.md`](adamem_mem0_full_trajectory_26family_results_20260913.md).

This ledger separates formal comparisons from single-run screening. Only a
completed three-replicate matched comparison may support a quality claim in the
paper. Screening values are retained to document coverage, saturation, and
mechanism applicability; they must not be mixed into a main-result mean.

## Frozen Formal Control: AllMemoryOff

The new formal control is complete over the frozen 26-family suite:

| Backend | Families | Accepted replicates per family | Accepted runs |
| --- | ---: | ---: | ---: |
| AllMemoryOff | 26 | 3 | 78 |

The frozen result ledger is
[`all_memory_off_result_ledger.json`](../outputs/all_memory_off_formal_20260911/all_memory_off_result_ledger.json)
with digest `b768499ba7987249f76b71d8de376a8dc47272d9c66db22c3e14425d94164455`.
Its separate main-table input is
[`all_memory_off_main_table_input.json`](../outputs/all_memory_off_formal_20260911/all_memory_off_main_table_input.json).
It reports task quality and usage only for the 78 accepted AllMemoryOff runs.

## Formal Result: SM01

Family: `SM01_preference_adoption`
Backend: `mem0-flat-hermes-v1`
Base agent and meta-agent: `gpt-5.6-luna`
Protocol: three independent accepted train -> update/abstain -> matched Near/Far
N+1 replicates per condition.

| Condition | Policy outcome | Update rate | Abstention rate | Updater input tokens, mean (SD) | Near N+1 | Far N+1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B0 Mem0 static | updater disabled | 0/3 | n/a | n/a | 1.000 (0.000) | 1.000 (0.000) |
| B1 AdaMem terminal | `no_update` in 3/3 | 0/3 | 3/3 | 143 (0.0) | 1.000 (0.000) | 1.000 (0.000) |
| B2 AdaMem full trajectory | activated update in 3/3 | 3/3 | 0/3 | 15,819 (724.8) | 1.000 (0.000) | 1.000 (0.000) |

All paired B1-B0, B2-B0, and B2-B1 deltas are `0.000` on both Near and Far
N+1 tasks. B2 updater inputs were 15,294, 15,517, and 16,646 tokens; B1 used
143 tokens in each replicate. No rollback occurred.

Interpretation: full deployment-visible trajectory reliably changes AdaMem's
policy-update behavior, while terminal feedback reliably abstains. SM01 is
saturated at 1.0 in every condition, so it is mechanism/fidelity and
resource-overhead evidence only. It is not an effectiveness claim and should
not be rerun unless the frozen protocol changes.

Source aggregate:
[`sm01-comparison-aggregate-r3.json`](../outputs/adamem_formal_20260908/sm01-comparison-aggregate-r3.json).
The detailed per-family record remains
[`sm01_adamem_trajectory_results_20260908.md`](sm01_adamem_trajectory_results_20260908.md).

## Frozen Full-Suite Static Screening

Screening condition: `B0_mem0_static`. Each accepted family is one isolated
full train -> matched N+1 run. This is not a B0/B1/B2 comparison and does not
measure AdaMem or RSIMem effectiveness.

The current screening set is complete: `25/25` new families were accepted
with the `single_full_sequence` topology, and SM01 is retained as the
precovered formal anchor. Therefore the frozen suite coverage is `26/26`.
The screening manifest is
[`screening_manifest.json`](../outputs/adamem_full_suite_20260908/screening_manifest.json)
with digest `975bba8c6610b1ada3be3fe0800e63d2e8fae58080d9b10b8613651be799c5d9`.

| Group | Coverage | Interpretation boundary |
| --- | ---: | --- |
| Semantic | SM01-SM07 | Direct target of the current extraction-policy updater |
| Episodic | EP01-EP03 | Cross-workflow evaluation only |
| Procedural | all 10 PC families | Cross-workflow evaluation only |
| Proactive retrieval | PG01-PG06 | Cross-workflow evaluation only |

Accepted early screening shows that several families are non-saturated under
the static Mem0 backend. For example, the matched evaluation episodes in
EP01 score 0.4493/0.5400 (Near/Far), EP02 0.7100/0.7780, EP03
0.5264/0.5264, and several PC/PG families also have evaluation components well
below 1.0. These are useful candidate families for later formal comparison,
but are single-run values and must not be reported as a quality result.

Earlier infrastructure-failure attempts remain attempt-level provider/runtime
evidence. They are not family exclusions and are not part of the accepted
screening set. All attempts remain in
[`screening_progress.json`](../outputs/adamem_full_suite_20260908/screening/screening_progress.json).

## Earlier Per-Family AdaMem Evidence

`SM02_constraint_retention` and `SM03_fact_correction` each have accepted,
three-replicate B0/B1/B2 batches. In both families B1 abstained in `3/3`
replicates and B2 activated an update in `3/3`; this is reliable
trajectory-dependent updater behavior, but not yet a full-suite quality claim.

| Family | B2 vs. B0 Far N+1 mean delta | B2 vs. B0 Near N+1 mean delta | B2 updater input tokens, mean |
| --- | ---: | ---: | ---: |
| SM02 | +0.0907 | -0.0053 | 19,391.0 |
| SM03 | +0.0080 | 0.0000 | 16,789.3 |

The corresponding aggregates are
[`sm02-comparison-aggregate.json`](../outputs/adamem_full_suite_20260908/formal_batches/sm02-comparison-aggregate.json)
and
[`sm03-comparison-aggregate.json`](../outputs/adamem_full_suite_20260908/formal_batches/sm03-comparison-aggregate.json).
They remain earlier per-family evidence. The later 26-family B0/B2 aggregate
above is the current formal result and does not include B1.

## Frozen Formal Baseline: AllMemoryOff, HermesNative, Mem0Static

The obsolete semantic-disabled `NoMemory` result has been removed from the
active result set. The current formal baseline uses the independently audited
`AllMemoryOff` control, whose manifest and result ledger are recorded in
[all_memory_off_baseline_20260911.md](all_memory_off_baseline_20260911.md).

The current formal target is a complete, three-backend comparison over the
frozen 26-family suite:

| Backend | Families | Accepted replicates per family | Required accepted runs |
| --- | ---: | ---: | ---: |
| AllMemoryOff | 26 | 3 | 78 |
| HermesNative | 26 | 3 | 78 |
| Mem0Static | 26 | 3 | 78 |
| Total | 26 | 9 | 234 |

Every replicate must execute the full train -> matched Near/Far N+1 sequence
with isolated state, memory storage, Hermes home, service port, fixture copy,
trace, and artifact root. Provider failures, incomplete usage, and identity
drift are rejected and retried in a new isolated run. The single SM01
three-backend smoke is only an execution/isolation check and contributes no
number to this formal comparison.

## Completed Base-Memory Baseline

The frozen 26-family suite completed with three accepted replicates for each
condition: `26 x 3 x 3 = 234` accepted runs. The suite runner ended with
`status=completed`, `completed_families=26`, and no unresolved infrastructure
error. The failed pre-retry `SM07 / Mem0Static` attempt is excluded.

| Condition | Accepted replicates | Near N+1 score | Far N+1 score |
| --- | ---: | ---: | ---: |
| AllMemoryOff | 78 | 0.3482 +/- 0.0012 | 0.3493 +/- 0.0013 |
| HermesNative | 78 | 0.7411 +/- 0.1771 | 0.7533 +/- 0.2030 |
| Mem0Static | 78 | 0.7718 +/- 0.1658 | 0.7945 +/- 0.1871 |

The Near/Far values contain 72 and 48 scored episodes respectively, because
the frozen family fixtures do not all define both evaluation types. Relative
to AllMemoryOff, HermesNative improves the pooled Near/Far scores by `+0.3929`
/ `+0.4040`; Mem0Static improves them by `+0.4236` / `+0.4452`. These are
backend baseline comparisons, not AdaMem or RSIMem uplift claims.

By task family and evaluation distance, the pooled evaluation scores are shown
as `mean +/- standard deviation`. `n` is the number of replicate-level
evaluation scores contributing to that cell (the PC fixtures use their
`I04`/`I05` near/far labels).

| Family | Split | n | AllMemoryOff | HermesNative | Mem0Static |
| --- | --- | ---: | ---: | ---: | ---: |
| SM | Near | 27 | 0.3962 +/- 0.0101 | 0.8818 +/- 0.1690 | 0.9465 +/- 0.0920 |
| SM | Far | 21 | 0.3943 +/- 0.0151 | 0.9125 +/- 0.1489 | 0.9757 +/- 0.0535 |
| EP | Near | 9 | 0.3133 +/- 0.0793 | 0.5279 +/- 0.2023 | 0.5607 +/- 0.1479 |
| EP | Far | 9 | 0.3083 +/- 0.0880 | 0.5093 +/- 0.1908 | 0.5638 +/- 0.1553 |
| PC | Near | 15 | 0.3560 +/- 0.0556 | 0.5232 +/- 0.1547 | 0.5128 +/- 0.1760 |
| PC | Far | 15 | 0.3560 +/- 0.0556 | 0.5187 +/- 0.1495 | 0.5306 +/- 0.1837 |
| PG | Near | 36 | 0.2968 +/- 0.0668 | 0.6888 +/- 0.0461 | 0.6936 +/- 0.0546 |
| PG | Far | 18 | 0.3060 +/- 0.0707 | 0.6896 +/- 0.0490 | 0.6983 +/- 0.0615 |

Mem0Static produced semantic operation evidence in 36 of 78 replicate records;
the other records are mechanism-inapplicable or had no semantic operation.
AllMemoryOff and HermesNative are reported separately because their evidence
surfaces are not equivalent to the Mem0 operation ledger.

## What Is Not Known Yet

- B1 terminal feedback was not run in the completed 26-family protocol, so a
  B0/B1/B2 feedback-granularity comparison remains unavailable.
- The B0/B2 aggregate is protocol-specific. It does not establish an RSIMem
  effect, causal transfer mechanism, or unconditional quality gain; B2 had
  9/78 harmful updates.
- Screening is complete, but it is not a three-replicate backend baseline and
  cannot be used in the paper main table.
- No RSIMem structured, grounded, or analyst feedback view has been evaluated.

## Next Result Update

The next result-producing work must be a separately frozen protocol for B1,
Hermes + AdaMem, or RSIMem-enhanced feedback. It must not reuse the B0/B2
numbers as proxy values.
