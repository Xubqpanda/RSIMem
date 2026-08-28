# Extraction-Prompt Stage 1 Acceptance

Date: 2026-08-28

## Decision

Extraction-prompt Stage 1 passes. Stage 2A implementation may begin.

- Accepted batch: `outputs/stage1_static_smoke/stage1-static-smoke-20260828-v5`
- Experiment ID: `43ab7e9fadc00c8709fe4e902e6d4325321aa588d2dfe1c225b4a210b8a94286`
- RSIMem commit: `2d0e8828dd165954c55c753367ebc8df8669e531`
- PAST-Bench commit/tree: `c6bb7fcf99265e0a01111825acf62b84f0cc09be` / `f3a537081bef9e7a56aa9452eecc82fdf4a99ad6`
- Method: `static-extraction-rsimem`
- Execution: one unseeded replicate, `native+ledger`, lifecycle evaluator disabled, static semantic writeback, utility gate disabled, native writer disabled.
- Audit: `ok=true`, `issues=[]`.
- Analysis: `qualityReady=true`, no failed attempt in the accepted batch.

This is an implementation smoke, not a quality comparison. It does not contain an adaptive extraction artifact, prompt N+1, matched static/adaptive delta, or evidence of quality superiority.

## Reconstructable Method Evidence

The accepted run produced two unique completed-source records from one canonical persisted source file:

| Source status | Count |
|---|---:|
| `nonempty` | 1 |
| `empty` | 1 |

Both eligible future evaluation stages emitted an incrementally persisted feedback record. The two primary labels were `unresolved`, which is the correct fail-closed result because the deployment evidence did not establish explicit attributable use. No useful, harmful, or missed label was fabricated.

The run recorded seven unique ingestion executions, nine ingestion-policy model requests, two planned-mutation outcomes, five no-change outcomes, nine mutation request/commit pairs, three model-visible memory injections, and zero utility decisions. Source projection, active parent artifact, receipts, operation graph, future feedback, ledger, and analyzer identities join without schema or safety failures.

## Raw Resources

Provider prices are not applied.

| Metric | Value |
|---|---:|
| Physical model requests | 50 |
| Input tokens | 83,851 |
| Output tokens | 6,091 |
| Cache-read tokens | 0 |
| Cache-write tokens | 0 |
| Reasoning tokens | 337 |
| Retries | 0 |
| Wall time | 274.28 s |
| Peak stored bytes | 127,374 |
| Injected characters | 513 |
| Ledger events | 171 |

Ingestion-only accounting reports 9 requests, 27,889 input tokens, 1,054 output tokens, 40,514 ms, and 0 retries. The provider did not expose complete ingestion cache-read, cache-write, or reasoning buckets. Those fields remain `unknown`, so top-level `usageComplete=false`; their observed zero placeholders are not used as complete values.

All 50 request events have `status=success`. Audit found zero credential-pattern hits, memory-text leaks, absolute observer paths, adapter bypasses, unresolved injections, projection mismatches, or accounting issues.

## Excluded Attempts

- `v1` is retained as a provider failure: all 27 requests returned HTTP 503 and no source or feedback evidence was produced.
- `v2` completed provider execution but exposed that ledger validation had not registered semantic `writer_identity`; it is retained with `failureStage=ledger`.
- `v3` passed ledger and audit but exposed a launcher glob that missed the final history-anchor source location; it is retained with `failureStage=extraction_audit`.
- `v4` exposed two fail-closed defects: distinct multi-fact ADD proposals were treated as duplicates, and an infra-blocked empty trace was read as a directory. It is retained with `failureStage=ledger`.

These attempts were not reclassified or overwritten. Each fix was committed before a new clean batch ID was registered.

## Verification

- RSIMem: `442 passed`.
- Vendored PAST-Bench, run from its own directory: `394 passed, 2 skipped`.
- `compileall`: passed for RSIMem and vendored PAST sources/tests.
- `pip check`: no broken requirements.
- All `scripts/*.sh`: `bash -n` passed.
- `git diff --check`: passed.
- Tracked secret scan: passed.

## Next Boundary

Stage 2A may define the host-neutral extraction policy envelope and immutable prompt artifact. Adaptive live execution remains disallowed until the artifact, optimizer corpus, proposal, held-out validation, activation, rollback, and matched-run gates in Stage 2 are implemented and tested. The accepted smoke supports only the Stage 1 pipeline claim.
