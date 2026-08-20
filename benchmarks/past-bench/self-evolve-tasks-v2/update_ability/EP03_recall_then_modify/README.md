# EP03_recall_then_modify — Recall Then Modify

**Ability**: `update_ability`
**Primary trigger**: `explicit_instruction`
**Expected substrate**: `session_search`
**Family length tier**: `tier2` (total 6 episodes)
**Transfer distance**: `near_far` · **Overwrite**: `not_required`

## Purpose
Prove that a prior artifact is retrieved through `session_search`, mapped onto the correct current shell record, and bounded-updated rather than rewritten from scratch.

## Must Demonstrate
- retrieval returns the right prior artifact
- prior history is seeded in native Hermes `state.db`, not only mirrored JSON logs
- recalled prior metadata is used to pick the right current shell
- modifications are localized and preserve unrelated state
- no re-derivation from current prompt alone

## Bucket Plan
- `cold`: 1
- `eval_near`: 1
- `eval_far`: 1
- `control`: 3

## Common Pitfalls to Avoid
- agent rewrites the playbook from scratch or updates the wrong shell record
- eval accepts a content-only patch without checking grounded prior recall
- shortcut control leaks enough local detail to bypass `session_search`
- wrong-mechanism control never injects a competing durable cache path

## Status
Canonical 6-episode slice runs from a native Hermes home fixture under
`_shared/home_fixtures/update_ability/EP03_recall_then_modify`. The legacy
`I02/I03` task files remain on disk as inactive history-generation material,
but the runnable sequence now starts eval and control episodes directly from
the synthetic `state.db` fixture.
