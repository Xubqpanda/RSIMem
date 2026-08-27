# Phase 1 Infrastructure Acceptance

Date: 2026-08-27

This report freezes the Phase 1 RSIMem infrastructure boundary. It covers native read equivalence, real Hermes lifecycle dry-run wiring, persistent idempotency, request accounting, ledger join, restart/failure handling, and privacy. It does not evaluate RSIMem memory quality, execute an RSIMem compiler or backend mutation, or support a recursive-improvement claim.

## Decision

Phase 1 passes.

- Accepted lifecycle run: `outputs/acceptance/20260827_phase1d_live_attempt04`
- Accepted RSIMem runtime commit: `c7cc375e3a3d24b26ca3cffa06534904dfa09a26`
- PAST-Bench last-change commit: `389b18f2bb496354eb4b175e6d2a40a31c066abc`
- PAST-Bench subtree: `78750da27849dfd9895e93859b3aff3fc0277683`
- Contract schema: `LIFECYCLE_CONTRACT_SCHEMA_VERSION=1`
- Configuration: `native+adapter+ledger`, native-shadow verification enabled, fail closed, paired persistence control, deterministic evaluator, policy `phase1-acceptance-v1`, compiler `uncompiled-v0`.
- Audit: `ok=true`, `issues=[]`.

The accepted run used the fixed `memory_ability/SM01_preference_adoption` family and task manifests with their declared budgets. PAST-Bench created isolated Hermes homes per execution. Direct native remains the default configuration; lifecycle dry-run requires an explicit evaluator-mode override.

## Lifecycle Evidence

There were 17 unique physical traces. Each trace emitted exactly two lifecycle chains, one at task completion and one at session end:

| Joined event kind | Count |
|---|---:|
| `context_snapshot` | 34 |
| `evaluation_accepted` | 34 |
| `plan_created` | 34 |
| `plan_validated` | 34 |
| `dry_run_mutation` | 34 |

The raw artifact tree contains 19 copied lifecycle files and 38 receipts because the shared cold trace is materialized into both comparison variants. Ledger identity deduplication correctly joins 17 physical traces and 34 logical boundaries. There were no rejected, stale, duplicate, or malformed lifecycle events in the accepted run.

Every accepted plan used the frozen semantic dry-run contract. No compiler ran, no RSIMem memory mutation executed, and no context segment was physically evicted. Native Hermes memory behavior remained enabled in the with-persistence control and is accounted separately from RSIMem dry-run evidence.

## Raw Resources

Raw quantities are reported without provider prices:

| Metric | Value |
|---|---:|
| Unique physical traces | 17 |
| Physical model requests | 68 |
| Input tokens | 122,101 |
| Output tokens | 6,147 |
| Cache-read tokens | 29,184 |
| Cache-write tokens | 0 |
| Reasoning tokens | 3,097 |
| Retries | 0 |
| Native-shadow projection checks | 28 |

All 68 physical requests had `status=success`. Projection mismatches, adapter bypasses, unresolved injections, memory-text leaks, credential-pattern hits, and absolute observer source paths were all zero.

## Recovery Evidence

Deterministic acceptance fixtures cover disabled, success, evaluator timeout/exception, malformed/partial/unknown output, retry, stale snapshot, concurrent receipt reservation, corrupted receipt, and coordinator restart. Persistent receipts return duplicate rather than reapplying a dry-run mutation after restart. Pre-snapshot host failures and evaluator failures produce content-free rejection evidence and do not alter the native request path.

Schema mismatch tests reject non-v1 snapshots, evaluation requests, plans, exit evidence, and usage objects. Unknown dataclass fields fail construction, and stale revisions fail plan validation.

## Excluded Attempts

- `20260827_phase1d_live` was stopped after exposing that semantic-only runs did not create `SessionDB` when `session_search` was disabled. All observed boundaries failed closed. Commit `389b18f` fixed snapshot persistence without exposing the session-search tool.
- `20260827_phase1d_live_attempt02` completed structurally, but one provider request failed before returning usage. Audit rejected the run with `incomplete_model_usage`; it is not included in accepted accounting.
- `20260827_phase1d_live_attempt03` passed lifecycle and audit checks before explicit contract schema v1 was added. It is retained as development evidence and excluded from the final frozen acceptance.

## Frozen Inputs For Phase 2

Phase 2 may depend on the v1 `ContextSnapshot`, `ContextEvaluationRequest`, `ContextEvaluation`, `ProvenanceRef`, `ExitEvidence`, `WritebackPlan`, and `RawResourceUsage` contracts. A validated plan exposes stable source segment IDs, source provenance, base revision, route-relevant memory kind, structured exit evidence, policy/compiler identity, and persistent idempotency identity.

Observer-facing evidence paths are `artifacts/rsimem_memory_events.jsonl`, `artifacts/rsimem_lifecycle_events.jsonl`, and `artifacts/rsimem_lifecycle_receipts.json`. Raw context and rendered evaluator prompts remain owner-controlled runtime data and are not ledger fields.

Known limitations are deliberate Phase 1 boundaries: live SM01 exercises semantic reads; episodic and procedural reads are covered by deterministic fixtures; context-pressure scheduling remains disabled without trusted host totals and thresholds; the deterministic evaluator only validates infrastructure; and there is no RSIMem compiler, mutation executor, static policy, adaptive policy, or physical context rewrite yet.
