# SM03 held-out extraction preflight — 2026-08-29

The split identity blocker is resolved without relabeling any completed train
trace.  The frozen plan is
`configs/extraction_split_plan_sm02_sm03_sm04.json`:

| role | family | task-template group | manifest digest (prefix) |
| --- | --- | --- | --- |
| train | `SM02_constraint_retention` | `sm02-process-pilot-train-v1` | `698882eefe98` |
| validation | `SM03_fact_correction` | `sm03-correction-heldout-validation-v1` | `cece4019357f` |
| final_test | `SM04_rule_migration` | `sm04-migration-heldout-final-v1` | `0022269fd7e8` |

`SM03_fact_correction` is an update-ability family used only as an
extraction-validation substrate.  Its update prompt and retrieval/writeback
components remain frozen.  The new `sm03_fact_correction_v1` contract exposes
only the deployment-observable corrected Phoenix freeze-date key
`fact.phoenix.release_freeze_date`; it distinguishes corrected, stale-only,
current-input-confounded, outcome-unattributable and censored observations.
It does not read grader, answer key, official score or benchmark-only labels.

This is a preflight and contract result, not a validation result.  No provider
matched batch has been started.  Before running one, the candidate must have a
trusted offline decision and trial profile, both repository trees must be
clean, the provider probe must pass, and every parent/proposal replicate must
produce a complete process corpus and audit.

Verification at this revision:

- RSIMem: `700 passed`
- vendored PAST-Bench: `397 passed, 2 skipped`
- `compileall`, `pip check`, `bash -n scripts/*.sh`, `git diff --check`: passed
- split-plan digest replay against all three vendored family roots: passed

