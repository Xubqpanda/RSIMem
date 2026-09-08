# SM01 AdaMem Trajectory Results

Status: `current result`  
Date: 2026-09-08  
Protocol: `adamem-trajectory-baseline-v1`

This record preserves the final provenance-complete formal comparison for
`SM01_preference_adoption`. It reports raw resources and matched N+1 outcomes;
it does not use grader output, hidden answers, official scores, or resource
cost as updater input.

## Protocol

- Base agent and meta-agent: `gpt-5.6-luna`
- Backend: `mem0-flat-hermes-v1`
- Conditions: `B0_mem0_static`, `B1_mem0_adamem_terminal`, and
  `B2_mem0_adamem_full_trajectory`
- Each condition: three accepted isolated train -> update/abstain -> matched
  N+1 replicates
- Final batch roots: `sm01-b0-formal-20260908-r2`,
  `sm01-b1-formal-20260908-r3`, and `sm01-b2-formal-20260908-r3`

The source aggregate is
[`sm01-comparison-aggregate-r3.json`](../outputs/adamem_formal_20260908/sm01-comparison-aggregate-r3.json).

## Results

| Condition | Policy outcome | Update rate | Abstention rate | Updater input tokens, mean (SD) | N+1 Near | N+1 Far |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B0 static | updater disabled | 0/3 | n/a | n/a | 1.0 (0.0) | 1.0 (0.0) |
| B1 terminal | `no_update` in 3/3 | 0/3 | 3/3 | 143 (0.0) | 1.0 (0.0) | 1.0 (0.0) |
| B2 full trajectory | `updated` in 3/3 | 3/3 | 0/3 | 15,819 (724.8) | 1.0 (0.0) | 1.0 (0.0) |

B2 updater input-token values were 15,294, 15,517, and 16,646. B1 used 143
tokens in every replicate. No rollback occurred.

Every paired delta for B1-B0, B2-B0, and B2-B1 was `0.0` on both N+1 tasks.

## Interpretation

The full deployment-visible trajectory consistently caused AdaMem to produce
and activate a semantic extraction-policy revision, while terminal feedback
consistently led to abstention. This is a reproducible policy-update signal.

The matched SM01 tasks are saturated at 1.0 in all conditions. Therefore this
experiment does not demonstrate task-quality uplift. It demonstrates a large
resource difference between the two feedback views: B2's updater input is
about 111 times B1's mean input, while the observed N+1 score remains
unchanged.

The result should be reported as a mechanism/fidelity and raw-resource finding,
not as an RSIMem effectiveness claim. Additional families with unsaturated
matched N+1 outcomes are required before assessing quality improvement.
