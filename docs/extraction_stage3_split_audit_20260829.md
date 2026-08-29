# Extraction split audit — 2026-08-29

The new `ExtractionSplitPlan` contract (commit `3373a78`) requires explicit
train, validation, and final-test assignments and rejects a task-manifest
digest appearing in more than one role. The matched preflight accepts an
optional plan and verifies the current validation family/template/digest before
any benchmark task starts.

The completed semantic pilots currently occupy the following roles:

| batch | family | role | task manifest digest (prefix) |
| --- | --- | --- | --- |
| `s1-sm01-feedback-20260829-v9a` | SM01_preference_adoption | train | `507fc684a092` |
| `s1-sm02-feedback-20260829-v5` | SM02_constraint_retention | train | `698882eefe98` |
| `s1-sm05-feedback-20260829-v1` | SM05_weak_trigger_preference_adoption | train | `6e9a01dfda44` |

The current pilots therefore cannot be relabeled as validation.  An explicit
held-out plan is now checked in at
`configs/extraction_split_plan_sm02_sm03_sm04.json`:

| role | family | task-template group |
| --- | --- | --- |
| train | `SM02_constraint_retention` | `sm02-process-pilot-train-v1` |
| validation | `SM03_fact_correction` | `sm03-correction-heldout-validation-v1` |
| final_test | `SM04_rule_migration` | `sm04-migration-heldout-final-v1` |

`SM03_fact_correction` is an update-ability family and is used only as an
extraction validation substrate with the update prompt frozen.  Its
deployment-observable contract is registered as
`sm03_fact_correction_v1`; it binds the corrected Phoenix freeze-date fact and
does not read the official grader or answer key.  The plan is a frozen
preflight artifact, not evidence that validation has run: a candidate trial
profile, clean trees, provider probe, complete process corpus and matched
parent/proposal execution are still required before activation can be
considered.  `SM04_rule_migration` is reserved for final-test bookkeeping and
must not be run until its family-specific feedback contract is registered.
