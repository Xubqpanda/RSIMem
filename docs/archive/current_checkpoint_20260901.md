# Historical Checkpoint: Stage 3 Pilot Coverage

Date: 2026-09-04

> Superseded on 2026-09-05. The current protocol is
> [`implementation_handoff_checklist.md`](../implementation_handoff_checklist.md),
> with Stage 0C classification in
> [`stage0_native_attribution_cleanup_audit_20260905.md`](stage0_native_attribution_cleanup_audit_20260905.md).
> The five-condition matrix and extraction-first path below are historical
> evidence only. New execution is gated on Stage 0D and uses isolated
> `native_static` attribution runs.

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
run without persistence. A manifest-bound pilot executor completed SM01
replicate-2, SM02 replicate-1, SM03 replicate-1, SM04 replicate-1, SM05 replicate-1, SM06 replicate-1, SM07 replicate-1, EP01 replicate-1, EP02 replicate-1, EP03 replicate-1, PC01 replicate-1, PC02 patch-01 replicate-1, PC02 patch-02 replicate-1, and PC03 replicate-1 across
all five conditions after passing provider probes; each content-free audit
reconciled all traces and raw usage buckets. This is execution/readiness
evidence only. The full semantic, episodic, and procedural panels now have 7,
3, and 10 case-bound oracle seeds, respectively, prepared from public
learn/update input and verified as evaluation-only fresh-state slices. The
source-level readiness catalog does not read task prompts, graders, or answers.
Remaining family pilots and replicates are pending, and no panel sensitivity
claim has been made. SM03, SM04, SM05, SM06, and SM07 are recorded with the semantic pilot
reports; EP01,
EP02, and EP03 are recorded in the episodic pilot reports; both PC02 patch
pilots are recorded in the procedural pilot reports; PC02 retry-2, PC03
retry-2, and the accepted PC01 bootstrap-02 retry-2 are recorded in the
procedural pilot reports. A named PAST task
control remains audit evidence, not an
executable host deployment, until its artifact/state and launcher configuration
are registered.
The first PC02 procedural replicate-1 attempt is excluded after the
content-free audit found incomplete usage and retry-accounting mismatches; its
diagnostic report is retained in
[`sensitivity_pc02_attempt_20260902.md`](sensitivity_pc02_attempt_20260902.md).
The accepted retry-2 report is
[`sensitivity_pc02_pilot_20260902.md`](sensitivity_pc02_pilot_20260902.md);
the first attempt remains excluded from all sensitivity denominators.
The earlier SM04 retry-3 attempt is likewise excluded because provider connection/read
timeouts caused incomplete usage; its diagnostic report is
[`sensitivity_sm04_attempt_20260903.md`](sensitivity_sm04_attempt_20260903.md).
The accepted SM04 retry-4 report is
[`sensitivity_sm04_pilot_20260903.md`](sensitivity_sm04_pilot_20260903.md).
The first PC03 replicate-1 attempt is excluded after the native-static run was
interrupted for lack of progress; its audit reported
`run_not_completed`/`sequence_results_missing` and is recorded in
[`sensitivity_pc03_attempt_20260903.md`](sensitivity_pc03_attempt_20260903.md).
The fresh retry-2 batch completed all five conditions and is accepted in
[`sensitivity_pc03_pilot_20260903.md`](sensitivity_pc03_pilot_20260903.md).
The PC01 bootstrap-02 retry is excluded after its `wrong_mechanism` control
timed out before terminal completion; its audit and partial traces are
recorded in [`sensitivity_pc01_02_attempt_20260903.md`](sensitivity_pc01_02_attempt_20260903.md).
Its retry-2 pilot completed and is accepted. The first PC01 bootstrap-03 pilot
is separately excluded after the same control lacked terminal completion; see
[`sensitivity_pc01_03_attempt_20260903.md`](sensitivity_pc01_03_attempt_20260903.md).
Its fresh retry-2 completed all five conditions and is accepted in
[`sensitivity_pc01_03_pilot_20260904.md`](sensitivity_pc01_03_pilot_20260904.md).
The PC01 bootstrap-04 pilot completed all five conditions and is accepted in
[`sensitivity_pc01_04_pilot_20260904.md`](sensitivity_pc01_04_pilot_20260904.md).
The PC01 bootstrap-05 pilot completed all five conditions and is accepted in
[`sensitivity_pc01_05_pilot_20260904.md`](sensitivity_pc01_05_pilot_20260904.md).
The PC01 bootstrap-06 pilot completed all five conditions and is accepted in
[`sensitivity_pc01_06_pilot_20260904.md`](sensitivity_pc01_06_pilot_20260904.md).
The PC04 failure-to-rule pilot completed all five conditions and is accepted in
[`sensitivity_pc04_pilot_20260904.md`](sensitivity_pc04_pilot_20260904.md).
The EP01 replicate-2 pilot completed all five conditions and is accepted in
[`sensitivity_ep01_pilot_20260904_r02.md`](sensitivity_ep01_pilot_20260904_r02.md).
The first EP02 replicate-2 attempt is excluded after usage-incomplete control
traces; see [`sensitivity_ep02_attempt_20260904_r02.md`](sensitivity_ep02_attempt_20260904_r02.md).
Its fresh retry-2 completed all five conditions and is accepted in
[`sensitivity_ep02_pilot_20260904_r02.md`](sensitivity_ep02_pilot_20260904_r02.md).
The EP03 replicate-2 pilot completed all five conditions and is accepted in
[`sensitivity_ep03_pilot_20260904_r02.md`](sensitivity_ep03_pilot_20260904_r02.md).
The SM01 replicate-3 pilot completed all five conditions and is accepted in
[`sensitivity_sm01_pilot_20260904_r03.md`](sensitivity_sm01_pilot_20260904_r03.md).
The SM02 replicate-2 pilot completed all five conditions and is accepted in
[`sensitivity_sm02_pilot_20260904_r02.md`](sensitivity_sm02_pilot_20260904_r02.md).
The SM03 replicate-2 pilot completed all five conditions and is accepted in
[`sensitivity_sm03_pilot_20260904_r02.md`](sensitivity_sm03_pilot_20260904_r02.md).
The SM04 replicate-2 pilot completed all five conditions and is accepted in
[`sensitivity_sm04_pilot_20260904_r02.md`](sensitivity_sm04_pilot_20260904_r02.md).
The SM05 replicate-2 pilot completed all five conditions and is accepted in
[`sensitivity_sm05_pilot_20260904_r02.md`](sensitivity_sm05_pilot_20260904_r02.md).
The SM06 replicate-2 pilot completed all five conditions and is accepted in
[`sensitivity_sm06_pilot_20260904_r02.md`](sensitivity_sm06_pilot_20260904_r02.md).
The SM07 replicate-2 pilot completed all five conditions and is accepted in
[`sensitivity_sm07_pilot_20260904_r02.md`](sensitivity_sm07_pilot_20260904_r02.md).
The SM02 replicate-3 pilot completed all five conditions and is accepted in
[`sensitivity_sm02_pilot_20260904_r03.md`](sensitivity_sm02_pilot_20260904_r03.md).
The SM03 replicate-3 pilot completed all five conditions and is accepted in
[`sensitivity_sm03_pilot_20260904_r03.md`](sensitivity_sm03_pilot_20260904_r03.md).
The SM04 replicate-3 pilot completed all five conditions and is accepted in
[`sensitivity_sm04_pilot_20260904_r03.md`](sensitivity_sm04_pilot_20260904_r03.md).
The SM05 replicate-3 pilot completed all five conditions and is accepted in
[`sensitivity_sm05_pilot_20260904_r03.md`](sensitivity_sm05_pilot_20260904_r03.md).
The SM06 replicate-3 pilot completed all five conditions and is accepted in
[`sensitivity_sm06_pilot_20260904_r03.md`](sensitivity_sm06_pilot_20260904_r03.md).
The SM07 replicate-3 pilot completed all five conditions and is accepted in
[`sensitivity_sm07_pilot_20260904_r03.md`](sensitivity_sm07_pilot_20260904_r03.md).
The SM01 replicate-1 pilot completed all five conditions and is accepted in
[`sensitivity_sm01_pilot_20260904_r01.md`](sensitivity_sm01_pilot_20260904_r01.md).
All seven semantic families now have complete three-replicate coverage. The
coverage aggregator reports semantic as replicate-analysis ready; this is not
yet a sensitivity status.
The EP01 replicate-3 pilot completed all five conditions and is accepted in
[`sensitivity_ep01_pilot_20260904_r03.md`](sensitivity_ep01_pilot_20260904_r03.md).
The EP02 replicate-3 pilot completed all five conditions and is accepted in
[`sensitivity_ep02_pilot_20260904_r03.md`](sensitivity_ep02_pilot_20260904_r03.md).
The EP03 replicate-3 pilot completed all five conditions and is accepted in
[`sensitivity_ep03_pilot_20260904_r03.md`](sensitivity_ep03_pilot_20260904_r03.md).
All three episodic families now have complete three-replicate coverage. The
coverage aggregator reports episodic as replicate-analysis ready; this is not
yet a sensitivity status.
The PC01 bootstrap-01 replicate-2 pilot completed all five conditions and is
accepted in
[`sensitivity_pc01_01_pilot_20260904_r02.md`](sensitivity_pc01_01_pilot_20260904_r02.md).
The content-free coverage manifest reports 41 accepted family-level pilots and
six excluded attempts, with expected-family coverage semantic `7/7`,
episodic `3/3`, and procedural `10/10`; procedural remains
replicate-incomplete. Its reproducible summary is
[`stage3_coverage_20260903.md`](stage3_coverage_20260903.md). This manifest is
coverage/readiness evidence only and does not assign a sensitivity status.
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
[`implementation_handoff_checklist.md`](../implementation_handoff_checklist.md).
They may be retained for audit and regression fixtures but cannot enter a new
proposal, validation input, ACTIVE pointer, or paper result.

The checked-in denylist is `rsimem-revocation-registry-v2`. Its five historical
entries use `scope=legacy_untyped` and null evidence plane/source because their
original provenance is unavailable; they match any typed lookup only by the
artifact identity (ID, schema version, and digest). New revocations must use
`scope=typed` and carry validated evidence provenance.

## Verification Baseline

The most recent deterministic acceptance baseline is:

- RSIMem: `1174 passed` (verified 2026-09-03); the focused launcher/catalog/registry/prepare/run
  regression suite passes `33` tests. Stage 1 contracts, Stage 2 adapter
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
