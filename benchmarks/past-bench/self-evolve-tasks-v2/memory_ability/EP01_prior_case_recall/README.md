# EP01_prior_case_recall — Prior Case Recall

**Ability**: `memory_ability`
**Primary trigger**: `explicit_instruction`
**Expected substrate**: `session_search`
**Family length tier**: `tier2` (total 8 episodes)
**Transfer distance**: `near_far` · **Overwrite**: `not_required`

## Purpose
Prove that vague references to prior work can trigger correct session retrieval rather than re-derivation or over-storage.

## Must Demonstrate
- relevant prior session is retrieved on vague cue
- retrieval precedes the decisive action
- episodic content is not promoted into durable memory unnecessarily

## Bucket Plan
- `cold`: 1
- `learn`: 2
- `eval_near`: 1
- `eval_far`: 1
- `control`: 3

## Common Pitfalls to Avoid
- user restates episodic detail in eval
- over-storing one-off context as durable memory

## Status
Canonical 8-episode slice present on disk. Eval fixtures no longer expose the
seed note directly, and the family includes explicit `no_persistence`,
`shortcut`, and `wrong_mechanism` controls.
