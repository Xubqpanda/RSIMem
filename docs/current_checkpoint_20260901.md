# Current Checkpoint: Provider Probe Recovered

Date: 2026-09-01

## Decision

The research execution plan has since been superseded by the revised
foundation checklist, which broadens the target to semantic, episodic, and
procedural memory and stops the extraction-only N+1 path.  Stage 0A baseline
freezing and 0B asset classification are complete; Stage 0C/0D cleanup and
Stage 1 protocol freezing are complete.  The cleanup pass has
removed the stopped launcher group, extraction-only proposal entry point, and
three orphaned configs.  The second call-site audit is recorded in
[`stage0_cleanup_audit_20260902.md`](stage0_cleanup_audit_20260902.md); three
configs remain only as deterministic preflight fixtures classified
`GENERALIZE`.  The pre-cleanup identity is recorded
in [`baseline_manifest_20260901.json`](baseline_manifest_20260901.json), with
the candidate inventory in [`asset_inventory_20260901.md`](asset_inventory_20260901.md).
Remaining cleanup deletion is held until the baseline gate and dependency
audit pass for each candidate; completed removals are recorded in commits
`b1c9970`, `480f77b`, `3b2cbb4`, and `e7e214e`.  Stage 1 contracts are now
frozen in [`research_protocol_v1.json`](research_protocol_v1.json), and the
next implementation boundary is the four-adapter Stage 2 split.  Typed
Benchmark/Host/Method/Feedback contracts, deterministic host/method fixtures,
the PAST public-identity adapter, and the Hermes projection-wrapper split are
now implemented.  The remaining Stage 2 work is full runner/bridge wiring and
golden-trace equivalence.

The provider gate is healthy, and the finite Stage 2 clean-parent rerun is now
complete. Formal proposal, held-out, activation, and matched-effect work
remains closed because the rerun produced no extraction-owned signal. This is
an evidence decision, not a negative task-quality result.

The latest bounded provider probe (2026-09-01) passed:

- Primary OpenAI-compatible endpoint: HTTP `200`, non-empty content, usage
  object present.
- Model: `gpt-5.6-luna`.

After this checkpoint was written, five consecutive provider-only probes against
the same endpoint/model also returned HTTP `200`, non-empty content, and a usage
object.  These probes are connectivity diagnostics only; they do not reopen the
closed SM02/SM05 process-signal census or authorize a repeated no-signal batch.

The probe itself was outside benchmark accounting. Each registered clean-parent
batch also passed the same completion probe immediately before its first task;
provider diagnostics do not enter a process corpus.

After the deterministic boundary-join fix (`a89f7d7`), a new SM02 batch was
registered as `s2-sm02-clean-parent-20260901-v2`. Its first replicate was
audit-clean; the second replicate had one provider `InternalServerError` and
failed `incomplete_model_usage`, then passed on attempt 2; the third replicate
failed `incomplete_model_usage` after a fail-closed `skip/defer extraction`
response. The batch is therefore an infrastructure/provider attempt, not a
valid process-signal census. Its partial artifacts are retained but excluded
from all counts and conclusions. SM05 was not started from this attempt.

## Implementation Position

The generic pure-process runtime path is wired and covered by deterministic
acceptance. At a trusted completed-task boundary, the runtime persists a pure
extraction source record. On later task activity it collects opportunity,
retrieval/exposure/use, artifact-set, and exact tool call/result evidence;
then persists feedback and constructs replayable process-signal cases.

Stage 1 is frozen as a result-independent protocol: versioned memory taxonomy
and control-state separation, six lifecycle surfaces with ownership gates, all
26 PAST family roles and confounders, five sensitivity conditions, isolated
split rules, and raw resource accounting.  The checked-in manifest is
metadata-only and contains no API key, grader field, answer, or official score.

Stage 3 now has a result-independent sensitivity harness.  It builds isolated
semantic (7 families), episodic (3 families), and procedural (10 families)
matrices with five conditions each and audit-only type-matched oracle digests.
An immutable Stage 3 run manifest now expands each
`family x condition x replicate` into independent state, Hermes-home, and
trace directories.  It provides only its opaque case ID through
`rsimem_method_task_id`; PAST family/task identity remains outside the method
boundary.  The three non-native condition deployment mechanisms are not
generally implemented, so the manifest correctly rejects an incomplete matrix
rather than treating five conditions as one runtime path. A case-bound
semantic SM01 type-matched oracle seed is now registered from the public learn
input only; its preparation path copies an evaluation-only seed home and passes
only the opaque case ID to PAST. The semantic catalog also makes native and the
three declared PAST control slices executable; shortcut and wrong-mechanism
run without persistence. A manifest-bound pilot executor completed one
unseeded SM01 replicate-2 across all five conditions after a passing provider
probe; a content-free audit reconciled 10 traces and all raw usage buckets.
This is execution/readiness evidence only. The full semantic panel now has
seven case-bound oracle seeds prepared from public learn/update input and
verified as evaluation-only fresh-state slices. Episodic and procedural oracle
target cases stay non-executable, and no replicated sensitivity or quality
claim has been made.
The source-level readiness catalog does not read task prompts, graders, or
answers. It reports that six semantic and all three episodic targets still lack
case-bound type-matched oracle seeds, while procedural targets additionally
lack a compatible wrong-mechanism control and PC03 lacks a no-persistence
control. A named PAST task control remains audit evidence, not an executable
host deployment, until its artifact/state and launcher configuration are
registered.
The launcher can prepare a case-specific PAST sequence slice and passes only
the opaque method case ID, alongside registered isolated state, Hermes-home,
and trace locations. It rejects non-executable deployments before forming any
provider command. This is execution plumbing, not an oracle artifact or a
model sensitivity result.

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

The formal PAST-Bench runtime currently has no trusted application-owned
memory-use attribution callback. Consequently, benchmark-family resolver
inference remains audit-only: pure-process retrieval and exposure are recorded,
but pure `USE` stays unknown unless the host explicitly supplies used artifact
IDs. This boundary is enforced by `b1f9cd1` and prevents audit labels from
silently becoming optimizer evidence.

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
| Provider-backed process-signal census | complete for SM02/SM05 v1 attempt | both fresh train batches completed; `STOP_NO_SIGNAL`; later v2 attempt excluded after incomplete model usage |
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

- RSIMem: `1151 passed` before the pilot-audit additions; the focused Stage 3
  regression suite now passes `26` tests. Stage 1 contracts, Stage 2 adapter
  contracts, the Stage 3 sensitivity harness, and isolated run registration
  remain covered.
- Vendored PAST-Bench: `401 passed, 2 skipped` when invoked from
  `benchmarks/past-bench`.
- `compileall`, `pip check`, `bash -n scripts/*.sh`, `git diff --check`, and
  `.venv/bin/python -m rsimem.secret_scan`: passed.

`rsimem.secret_scan` intentionally scans only Git-tracked regular files. It
does not inspect untracked credential files, ignored run outputs, or drafts.

## Resume Order

The registered SM02 and SM05 clean-parent v1 attempt remains the only valid
provider census and is complete with `STOP_NO_SIGNAL`. The later SM02 v2
attempt did not satisfy the replicated audit gate and must not be resumed as a
partial experiment; any future retry requires a newly registered batch and a
fresh pre-task probe. The next authorized implementation work is deterministic
contract/replay maintenance, or a separately pre-registered family only if a
new application-owned opportunity schema is available. No proposal gate opens
until a replicated pure-process corpus contains a valid extraction-owned
signal. Cost and token data remain raw accounting fields, never policy input.

Until then, permitted work is deterministic regression, documentation,
contract review, and replay/audit maintenance.
