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

Therefore none of the currently authored semantic families is an uncontaminated
held-out validation family: reusing any of these exact task manifests would
cross the frozen train/validation boundary. The SM02 candidate remains a
proposal, but a formal matched validation requires a new semantic task-template
group/family (or an explicitly authored held-out split) before its provider
execution can start. No existing train trace is relabeled as validation.
