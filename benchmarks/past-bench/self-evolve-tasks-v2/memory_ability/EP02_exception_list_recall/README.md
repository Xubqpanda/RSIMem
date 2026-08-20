# EP02_exception_list_recall — Exception List Recall

**Ability**: `memory_ability`
**Primary trigger**: `explicit_instruction`
**Expected substrate**: `session_search`
**Family length tier**: `tier2` (total 8 episodes)
**Transfer distance**: `near_far` · **Overwrite**: `not_required`

## Purpose
Prove that a session-bounded restart subset is recalled later via session search rather than guessed from the current batch or over-stored as a standing rule.

## Must Demonstrate
- exception set is recoverable via session search
- exceptions are not elevated into durable memory
- application to eval uses the correct subset across multiple candidates

## Bucket Plan
- `cold`: 1
- `learn`: 2
- `eval_near`: 1
- `eval_far`: 1
- `control`: 3

## Common Pitfalls to Avoid
- turning the restart subset into durable policy or convenience-cache memory
- making the correct subset inferable from current notes without prior-session recall
- collapsing the family into single-object recall instead of selective subset reuse

## Status
Canonical 8-episode slice present on disk. `I01-I03` build the unique
prior-session exception history, `I04/I05/I07/I08` branch from that saved
anchor, `I06` disables persistence for the mechanism-level ablation, and
`I08` injects a decoy durable memory while still requiring `session_search`.
