# Current Checkpoint: Provider Probe Recovered

Date: 2026-09-01

## Decision

Formal provider-backed experiments remain paused until a fresh preflight is
run immediately before a registered batch. This is an operational gate, not a
negative task result and not an update to any policy conclusion.

The latest bounded provider probe (2026-09-01) passed:

- Primary OpenAI-compatible endpoint: HTTP `200`, non-empty content, usage
  object present.
- Model: `gpt-5.6-luna`.

No benchmark was started by this probe. A fresh probe must still pass under the
exact frozen run profile after batch registration and before the first task.
Provider diagnostics remain outside benchmark accounting and must not enter a
process corpus.

## Implementation Position

The generic pure-process runtime path is wired and covered by deterministic
acceptance. At a trusted completed-task boundary, the runtime persists a pure
extraction source record. On later task activity it collects opportunity,
retrieval/exposure/use, artifact-set, and exact tool call/result evidence;
then persists feedback and constructs replayable process-signal cases.

The completed implementation components include:

- `PureExtractionSourceRecord` and durable source storage.
- `OpportunityEvidence` from frozen application-owned visible schemas.
- `ArtifactSetSemanticBinding`, memory-use attribution, and exact tool joins.
- `PureExtractionFeedbackRecord`, pure optimizer corpus, and signal gate.
- Pure proposal construction, with policy calls gated before execution.
- Logical process-signal cases, protocol binding, replay, and census.
- Semantic adapter future traces that reuse the actual rendered retrieval hits.

This establishes process observability and replayability. It does not establish
that an extraction prompt has a generalizable optimization signal, that an N+1
candidate is valid, that a policy is ACTIVE, or that task quality improved.

## Evidence Boundary

The following work is complete as deterministic/runtime infrastructure:

| Area | Status | What it demonstrates |
| --- | --- | --- |
| Context, provenance, revision, CAS, receipts, rollback | complete | fail-closed writeback control-plane behavior |
| Hermes semantic source/feedback/case dataflow | complete | host runtime reaches durable pure-process records |
| Semantic, episodic, and procedural adapter surfaces | complete at storage boundary | projection/read-through behavior, not live policy benefit |
| Six policy-layer contracts | deterministic/shadow complete | decision/action/replay feasibility only |
| Process-signal case and optimizer gate | complete | signal eligibility can be tested without evaluator data |

The following remain deliberately unavailable or deferred:

| Area | State | Release condition |
| --- | --- | --- |
| Provider-backed process-signal census | paused | healthy probe, then a fresh clean parent batch |
| Extraction N+1 proposal | locked | replicated, generalizable pure-process signal |
| Held-out validation and ACTIVE pointer | locked | independently authored candidate plus valid parent evidence |
| Matched uplift/adaptive claim | deferred | completed held-out validation and matched runs |
| Joint six-layer policy optimization | deferred | each layer first demonstrates its own signal |

Existing SM01 `unresolved` observations remain valid no-signal evidence.
Historical SM02/SM05 `missed`, candidates derived from them, and associated
offline-validation interpretations remain revoked as defined in
[`implementation_handoff_checklist.md`](implementation_handoff_checklist.md).
They may be retained for audit and regression fixtures but cannot enter a new
proposal, validation input, ACTIVE pointer, or paper result.

The checked-in denylist is `rsimem-revocation-registry-v2`. Its five historical
entries use `scope=legacy_untyped` and null evidence plane/source because their
original provenance is unavailable; they match any typed lookup only by the
artifact identity (ID, schema version, and digest). New revocations must use
`scope=typed` and carry validated evidence provenance.

## Verification Baseline

The most recent deterministic acceptance baseline is:

- RSIMem: `1101 passed`.
- Vendored PAST-Bench: `401 passed, 2 skipped` when invoked from
  `benchmarks/past-bench`.
- `compileall`, `pip check`, `bash -n scripts/*.sh`, `git diff --check`, and
  `.venv/bin/python -m rsimem.secret_scan`: passed.

`rsimem.secret_scan` intentionally scans only Git-tracked regular files. It
does not inspect untracked credential files, ignored run outputs, or drafts.

## Resume Order

The next authorized action is one fresh, pre-registered clean parent batch. It
must pass provider preflight immediately before task execution and retain the
current conservative `unresolved`/`censored` semantics. Only after its
pure-process corpus yields a valid replicated signal may the proposal gate be
opened. Cost and token data remain raw accounting fields, never policy input.

Until then, permitted work is deterministic regression, documentation,
contract review, and replay/audit maintenance.
