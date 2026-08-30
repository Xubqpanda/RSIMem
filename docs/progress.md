# RSIMem Progress

Last updated: 2026-08-30

This document tracks implementation progress, the current experimental boundary, and the next executable milestones. Research motivation and the full staged evaluation design remain in [`experiment_plan.md`](experiment_plan.md). The detailed lifecycle implementation sequence is in [`lifecycle_implementation_plan.md`](lifecycle_implementation_plan.md), and the complete two-stage serial implementation and acceptance requirements are in [`implementation_handoff_checklist.md`](implementation_handoff_checklist.md).

## Status Legend

- [x] Completed and verified.
- [ ] Not completed.
- **Current** identifies the milestone that should receive implementation effort next.

## Current State

RSIMem can run the vendored PAST-Bench with Hermes and GPT-Luna, account for every exposed model request, derive a privacy-safe lifecycle ledger, audit run completeness, and represent Hermes semantic, episodic, and procedural memory through typed backend contracts.

The typed memory runtime is connected to the PAST-Bench Hermes execution path behind explicit opt-in modes. Direct native remains the default. Static Mem0-flat semantic writeback is available through live experiment configuration. The older adaptive utility/threshold mode remains only as replayable infrastructure; it is not the extraction-prompt method. The active implementation scope is semantic-first over Hermes native semantic storage. Episodic and procedural adapters remain verified read surfaces, but their policy implementations are deferred until methods are selected. The current research milestone is deterministic/shadow feasibility for the six host-neutral policy layers, not a live adaptive effect claim.

Raw extraction-validation observations now have a separate crash-safe,
frozen-split-bound store, so prompt-quality decisions remain reconstructible
from persisted observations rather than depending on a derived decision file.
Matched evidence assembly now reloads those durable observations before running
the evaluator, completing the raw-observation-to-decision replay boundary.
`ExtractionValidationReplay.verify_store()` now exposes the same restart-safe
recomputation as a public API and rejects missing or extra observation IDs.
It also binds replay to the persisted split ID, rejecting a decision replayed
through a different split identity.
The observation store also fixes final evidence files to mode `0600`.

The Hermes, static semantic writeback, operation graph, feedback store, and
activation/rollback foundations are complete. The former Phase 2J/2K work is
complete only as legacy retrieval-threshold infrastructure; extraction-prompt
adaptation remains a later effect experiment. Stage 1, the Stage 2A–2I policy
infrastructure, and deterministic runtime binding are closed. A first
third-stage feasibility fixture now records six-layer parent/candidate
interventions, strict feedback projection, process feedback, N+1 hypothesis
identity, and restart-safe content-free evidence. Extraction is currently
`optimization-ready` in that fixture; the other layers remain
`validation-only` until outcome variation is observed. The latest regression
counts are RSIMem `723 passed` and vendored PAST-Bench `399 passed, 2 skipped`.

Extraction-prompt Stage 1A through 1H are complete. Legacy threshold artifacts now
have incompatible identity and no resource-cost activation gates. Static
semantic compilation runs directly from a trusted completed-task snapshot,
independent of keep/evict evaluation; session end creates no second attempt,
and a persistent content-free compilation receipt prevents replayed model
calls. A versioned canonical source projection now binds the exact prompt
messages, stable IDs, tool closures, deterministic truncation, request and
receipt identity, operation artifact, and prompt input digest. A host-neutral
prompt slot and the real Mem0-flat adapter bind the extraction artifact at the
completion boundary. Composite and matched manifests freeze update, retrieval,
route, boundary, backend, framework, and model profile. Plain static extraction
is the explicit parent; native writers and background review are disabled, and
run-scoped receipt audit rejects unowned semantic drift. Source/set/fact
feedback, family-specific opportunity/use/outcome contracts, prompt-oriented
validation, formal extraction manifests, and raw-vector analysis are complete.
The accepted low-cost plain-parent smoke produced two source records, two exact-
joined feedback records, 50 successful physical requests, and an issue-free
audit. Detailed evidence and excluded attempts are recorded in
[`extraction_stage1_acceptance_20260828.md`](extraction_stage1_acceptance_20260828.md).
SM02 boundary feedback now normalizes human-readable recipient names at the
resolver boundary, while matching exact normalized IDs so near-collision names
cannot be mislabeled as harmful.

A single provider-connectivity smoke against the configured OpenAI-compatible
endpoint also completed successfully; it records only raw usage in the ignored
`outputs/provider_connectivity_smoke_20260829.json` and is not a benchmark or
effect replicate.

The first real-provider parent-only SM01 feedback pilot completed three clean
replicates. It produced 24 primary examples, all `unresolved`, so the strict
optimizer gate correctly returned `NO_PROPOSAL` without a candidate. Full raw
usage, trace, ledger, and audit evidence is recorded in
[`extraction_stage3_s1_feedback_20260829.md`](extraction_stage3_s1_feedback_20260829.md).

The current Stage 3B/3D process-signal work joins clean parent pilots without
using evaluation scores. The new SM02 and SM05 reruns use the latest
family-free process projection and replay-stable logical-case store. They
confirm lifecycle observability and tool-closure joins, but both remain
`STOP_NO_SIGNAL`: extraction is shadow `pending`, no artifact/use/outcome
chain is attributable, and no N+1 candidate is unlocked. Detailed dated
reports are [`extraction_stage2_sm02_process_signal_final_20260830.md`](extraction_stage2_sm02_process_signal_final_20260830.md) and
[`extraction_stage2_sm05_process_signal_20260830.md`](extraction_stage2_sm05_process_signal_20260830.md). Extraction remains the first
layer to open only after a signal-ready case; the other five layers stay
shadow/validation-only.

Process-signal analysis is now bound at both batch and case boundaries: each
formal batch freezes a result-independent protocol, every emitted case carries
the protocol ID, replicate ID, and observation window, and the analyzer rejects
missing, drifted, or cross-replicate cases before census. The deterministic
six-layer fixture also emits explicit content-free mechanism explanations for
each intervention; useful/missed/harmful outcomes without a complete evidence
chain remain unresolved rather than entering a negative denominator.

The follow-up SM02 process-signal pilot exercised the generalized family
launcher and both configured endpoints. Two primary-provider replicates passed
audit, while one primary and one backup attempt failed with HTTP 503/capacity
errors; incomplete attempts remain excluded from optimizer and activation. Raw
usage and process-corpus diagnostics are recorded in
[`extraction_stage3_sm02_process_pilot_20260829.md`](extraction_stage3_sm02_process_pilot_20260829.md).

A subsequent clean-worktree retry tested the same family against each endpoint
separately. The primary retry stopped after a provider-capacity failure and the
backup retry produced empty think-only responses with missing usage on every
trace; both audits therefore failed closed as `incomplete_model_usage`. Their
manifests and raw traces are retained, but neither attempt is a task negative
or optimizer input. See
[`extraction_stage3_sm02_provider_attempts_20260829_v3_v4.md`](extraction_stage3_sm02_provider_attempts_20260829_v3_v4.md).
The same report includes a deduplicated process-signal census: the primary
attempt still emitted all six policy-stage events, while the backup emitted
only retrieval, exposure, and task-failure events. This diagnostic is not a
formal training or validation batch.
The subsequent clean primary-endpoint train rerun
`sm02-feedback-20260829-rerun-main` completed three audited replicates and
reconstructed 24 primary feedback records (12 `missed`, 12 `unresolved`). A
the first v2 optimizer completion was malformed and the strict parser rejected
it; a single recovery request under the new v2 contract produced an abstract
schema-valid candidate after the persistence fix. Static safety and the
deterministic extraction suite pass, but the candidate remains a proposal and
has not entered independent SM03 validation. Details are in
[`extraction_stage3_sm02_feedback_rerun_20260829.md`](extraction_stage3_sm02_feedback_rerun_20260829.md).
The reusable `rsimem.provider_probe` entry now performs a bounded completion
check without exposing credentials or response content; it reports endpoint,
content, and usage availability only and remains outside benchmark accounting.
Its result contract rejects inconsistent manually constructed states, so a
successful probe always has a 2xx status and a failed probe always has an
explicit diagnostic code.
Both formal extraction launchers now run the probe after manifest/preflight
registration and before the first benchmark task, persisting only the safe
`provider_probe.json` result and stopping early on failure.
This fail-fast path was verified in a clean detached worktree with a local HTTP
fixture: the launcher exited before creating any PAST-Bench task trace.

The PAST-Bench Hermes adapter now transports only content-free process-corpus
identity (`event_ids` and a stable digest) in `StepResponse`; it never exposes
official score or grader fields to the policy path. An evaluator-free static
writeback fixture verifies trigger, source, extraction, admission, commit,
exposure, and task-outcome events, receipt joins, and process-ledger auditing.
The focused contract is in `tests/test_rsimem_hermes_adapter.py`; current
an earlier process-audit snapshot recorded RSIMem `676 passed` and PAST-Bench
`397 passed, 2 skipped`; the current regression is reported above.

Both formal extraction launchers now run `audit_process_events()` before
persisting `process_corpus.json`; malformed terminal receipt joins or
stage-specific failure semantics fail the attempt instead of entering analysis.
The standard `rsimem.audit` report also emits `processEvidence` rows and marks
the run failed when a process ledger cannot be structurally or semantically
audited; policy-bound process events are additionally joined against the
corresponding policy-decision ledger and fail closed when that ledger is absent
or missing the referenced decision. Standard audit also checks each event's
`runId`, `variant`, `traceId`, `taskId`, `familyId`, and `stage` against the
comparison trace context.
Raw lifecycle usage buckets now reject booleans and non-integer values while
preserving provider-unknown buckets as `null`.
Matched validation evidence assembly now requires a valid, content-free
`process_corpus.json` for every completed slot and verifies its declared split,
family, task-template, manifest digest, and process-event audit before building
activation evidence.
Process-corpus evaluator isolation now normalizes field names across casing and
separator conventions, preventing camelCase or hyphenated score/answer keys from
crossing the learner boundary.
Process audit terminal semantics now also require an execution receipt for
`REJECTED` events; non-executing `SKIPPED`/`DEFERRED` decisions remain explicit
and receipt-free by contract.
Policy `SKIP`/`DEFER` reasons are projected through a bounded mapping so
`absence` and `unsupported_boundary` remain distinguishable from generic
`decision_observed` process evidence.
`JsonProcessCorpusStore` now uses the same lock-protected atomic persistence
semantics as the process-event ledger; concurrent writers reserve one canonical
corpus instead of silently overwriting one another.
Hermes process feedback identity now has an explicit restart replay fixture:
the event set, canonical ID ordering, and digest remain unchanged after a
second bridge instance observes the same completed boundary.
The complete six-layer deterministic replay chain is also persisted and
reloaded through the process ledger; every event remains auditable and
idempotent after restart.
Native-bypass handling now records a second failed process event when the
fallback native call also fails, preserving both `adapter_failure` and the
stage-specific `retrieval_failure`/`tool_failure`/`injection_failure` reason.
The standard process audit also verifies that each policy-bound event's host
event matches the `triggerEventId` recorded by the policy ledger.
Unbound `SKIPPED`/`DEFERRED` process events now require an explicit `absence` or
`unsupported_boundary` reason; otherwise the event is rejected as unexplained.
Policy decision envelopes now reject a `RUN` action carrying a non-executing
status, and content-free audit rejects execution receipts on `SKIPPED` or
`DEFERRED` decisions.  Pending, failed, and rejected run states remain
available for normal scheduling and recovery.
The persistent idempotency store also has a regression guard proving that a
malformed receipt cannot be overwritten by a later reservation; it fails closed
before any mutation reservation is recorded.
The process-feedback ledger now has an equivalent concurrent-writer regression
test: one canonical event is reserved across eight writers and all retries are
idempotent.
Policy-decision evidence uses the same lock-protected atomic replacement and
now has a concurrent-writer regression test as well; identical evidence events
remain a single canonical record across independent ledger instances.

## Completed Work

### Repository And Benchmark Foundation

- [x] Define RSIMem as the experiment and evaluation repository for memory-mediated recursive self-improvement.
- [x] Select PAST-Bench as the first long-horizon benchmark and document the selection rationale.
- [x] Vendor PAST-Bench and Hermes Agent under `benchmarks/past-bench` with upstream attribution and licenses.
- [x] Establish a Python 3.11 development environment and reproducible local installation procedure.
- [x] Preserve task definitions, graders, answer keys, and native persistence semantics.
- [x] Add a repeatable GPT-Luna smoke launcher for `SM01_preference_adoption`.
- [x] Add a secret-free Python/dependency/state/provider preflight and verify a clean temporary installation.
- [x] Freeze a fail-closed matched experiment manifest with resolved model, judge, task budgets, environment versions, revisions, persistence isolation, restart identity, and append-only attempt evidence.

### Baseline And Usage Evidence

- [x] Run matched Hermes native-persistence and no-persistence variants.
- [x] Verify the initial persistence contrast: `1.000` versus `0.400` evaluation score in the infrastructure smoke.
- [x] Instrument every exposed Hermes model request with request-level usage evidence.
- [x] Record input, output, cache-read, cache-write, reasoning, request, retry, duration, status, provider, model, and API-mode fields.
- [x] Preserve unknown usage as `null` rather than inferring zero.
- [x] Assign stable billing execution IDs and deduplicate shared cold executions.
- [x] Reconcile request-level token buckets and request counts with every `TraceEnd` aggregate.

### Lifecycle Ledger And Audit

- [x] Implement `rsimem-ledger` for episode outcomes, model usage, model calls, tools, memory operations, memory injections, and storage snapshots.
- [x] Keep ledger evidence free of memory text, prompts, responses, authorization headers, credentials, and user-specific absolute paths.
- [x] Implement `rsimem-audit` to validate usage reconciliation, billing identity uniqueness, evidence completeness, and privacy constraints.
- [x] Produce a request-accounted end-to-end smoke whose 17 traces and 71 physical model requests reconcile exactly.

### Typed Memory Architecture

- [x] Adopt the standard semantic, episodic, and procedural memory taxonomy.
- [x] Define host-neutral artifacts, queries, hits, mutations, results, experiences, messages, resources, capabilities, and lifecycle events.
- [x] Separate `MemoryBackend`, which stores and retrieves typed memory, from `MemoryCompiler`, which distills completed experience into typed mutations.
- [x] Implement explicit backend registration and one selected route per memory kind.
- [x] Validate backend ownership, returned memory kind, mutation action, revision, and operation capabilities at runtime boundaries.
- [x] Emit content-free query, retrieval, injection, and mutation evidence.

### Hermes Native Adapters

- [x] Map semantic memory to `MEMORY.md` and `USER.md` with Hermes namespaces, delimiter, character budgets, file locking, and atomic writes.
- [x] Map episodic memory to read-only `state.db` message retrieval through SQLite FTS5.
- [x] Map procedural memory to `skills/**/SKILL.md` with progressive resources under `references`, `templates`, `scripts`, and `assets`.
- [x] Support semantic and procedural add, update, delete, query, revision conflict, and safe-path behavior.
- [x] Verify the adapter implementation with isolated temporary Hermes homes and a real FTS5 schema.

### Lifecycle Control Plane

- [x] Define host-neutral context segments, evaluation requests, cadence events, and joint context/memory signals.
- [x] Separate evaluation frequency from evaluator implementation through a configurable scheduler.
- [x] Add an injected JSON LLM evaluator boundary that can later be replaced by a topic-aware local model.
- [x] Add a conservative deterministic evaluator for behavior-equivalence tests.
- [x] Reject incomplete evaluator output and attempts to evict active context segments.
- [x] Keep observer-facing lifecycle evaluation evidence free of raw context content.

### Snapshot And Dry-Run Foundation

- [x] Define immutable context snapshots with stable segment IDs, revisions, active/current protection, token totals, and provenance.
- [x] Preserve Hermes tool call/result closures as atomic planning units.
- [x] Add a deterministic `SM01_preference_adoption` Hermes fixture and semantic preference evaluator.
- [x] Build content-free `WritebackPlan` validation with stale-revision rejection and idempotency keys.
- [x] Build a dry-run coordinator that records plan and mutation identifiers without changing Hermes or memory backend state.
- [x] Protect unresolved segments and require a real current turn or `None`.
- [x] Require revisioned update targets, backend update capability, compiler versioning, and persistent idempotency receipts.
- [x] Resolve model UPDATE hints through the selected backend registry, verify artifact ownership and existence, and bind the stored revision before plan creation.
- [x] Carry structured `ExitEvidence` and the complete update hint tuple into `WritebackPlan`.
- [x] Include every compiler-relevant `ExitEvidence` field in the canonical idempotency identity and strictly require boolean eviction safety.
- [x] Atomically reserve dry-run idempotency receipts under one store lock so concurrent coordinators cannot both accept the same plan.
- [x] Fail closed on malformed idempotency receipts and conflicting ledger event payloads.

The receipt reservation closes the dry-run check/reserve race. Real memory
mutation still requires explicit pending/committed receipt states and crash
recovery before any exactly-once claim is justified.

### Storage-Boundary Deterministic Equivalence Baseline

- [x] Define explicit `native`, `native+ledger`, and `native+adapter+ledger` modes with direct native behavior as the default.
- [x] Add a factory for constructing the three-route Hermes typed runtime from an isolated home.
- [x] Add observer-only evidence to the direct-native storage helper for the `native+ledger` fixture variant.
- [x] Verify identical deterministic semantic rendering, episodic FTS views, and procedural skill resources across native and adapter helpers.
- [x] Preserve native semantic entry order and Hermes episodic surrounding-context behavior in the adapter.
- [x] Join content-free snapshot, plan, validation, and dry-run events into the existing ledger schema.

Storage-boundary deterministic equivalence baseline completed. Real Hermes execution equivalence has not been established.

### Deterministic Hermes Execution-Surface Baseline

- [x] Invoke Hermes' real `AIAgent._build_system_prompt` with native and adapter-backed frozen semantic memory snapshots.
- [x] Dispatch the real `session_search` tool against native `SessionDB` and an adapter-backed DB contract with a deterministic summarizer.
- [x] Dispatch the real `skills_list` and `skill_view` tools, including a linked resource, against native and adapter-projected skill directories.
- [x] Verify identical model-visible output for all four surfaces across `native`, `native+ledger`, and `native+adapter+ledger`.
- [x] Verify content-free query, retrieval, and injection evidence for instrumented variants.
- [x] Rebuild runtimes and verify stable execution reports and artifact IDs.
- [x] Define explicit `fail_closed` and observable `bypass_native` adapter failure policies.

Deterministic Hermes execution-surface equivalence baseline completed. The
fixture replaces external session summarization with a deterministic function
and does not make model API calls, so matched live-model Hermes execution
equivalence has not yet been established.

### Deterministic PAST-Bench Agent-Loop Baseline

- [x] Carry explicit `native`, `native+ledger`, and `native+adapter+ledger` modes through the PAST-Bench sequence and runtime configuration.
- [x] Attach the bridge after Hermes agent construction and restore every wrapped surface before runtime cleanup.
- [x] Execute `HermesAdapter.step`, `_run_agent`, semantic prompt reads, session search, and skill reads in one deterministic fixture.
- [x] Verify identical final model-visible fixture output across all three execution modes.
- [x] Keep direct native free of RSIMem evidence and emit content-free evidence for both ledger modes.
- [x] Separate comparison variant identity from RSIMem execution mode.
- [x] Discover episode-local RSIMem JSONL evidence from comparison-owned trace paths and join it through strict run, variant, trace, task, family, and stage validation.
- [x] Reject malformed JSONL, misplaced evidence, unknown runtime fields, and conflicting event IDs.
- [x] Persist runtime evidence incrementally with flush and fsync instead of waiting for bridge shutdown.
- [x] Project procedural backend hits into an isolated Hermes skills root before native-format rendering and security checks.
- [x] Project episodic pagination, filters, session lineage, metadata, and full conversation expansion through the backend.
- [x] Match Hermes FTS5 query normalization, empty-role filtering, corrupt structured-field handling, and assistant-only reasoning replay.

This proves deterministic PAST-Bench adapter-loop equivalence without an
external model call. It does not prove live-model behavioral equivalence or
establish a nondeterminism tolerance.

### Phase 2B Prompt And Ingestion Infrastructure

- [x] Freeze v1 prompt, ingestion, and atomic operation evidence contracts.
- [x] Pin and attribute the MemBase Mem0 prompt source while keeping its runtime, datasets, and evaluation code outside RSIMem.
- [x] Add canonical route-specific ingestion requests, structured results/failures, deterministic fixture ingestion, and auditable raw usage.
- [x] Add a bounded, content-free append-only operation evidence graph with offline joins, privacy checks, observer-failure isolation, and tracing overhead reports.
- [x] Prove planning/executor separation and leave Hermes backend bytes unchanged in fixtures.

Known boundary: prompt completion and pass-through ingestion are fixture-only;
the operation graph's mutation fixture is synthetic. No real model-based memory
construction, validator, transaction executor, backend mutation, or live-run
operation tracing is enabled.

### Phase 2C Validation And Security Boundary

- [x] Add a content-free host-neutral validation result and fail-closed candidate pipeline.
- [x] Bind source, scope, validity, backend ownership, target ownership, revision, and actual target digest through trusted runtime state.
- [x] Enforce the semantic namespace, Hermes character budget, metadata/resource, duplicate/conflict, durable-category, and security allow/reject matrix.
- [x] Keep episodic/procedural mutation disabled and prove every rejected candidate leaves backend mutation count at zero.

Known boundary: target ownership persistence and committed receipts do not exist
outside the Phase 2D isolated receipt store. Semantic category and denylist checks enforce the current deterministic
contract but do not establish model-generated memory quality or comprehensive
prompt-injection resistance.

### Phase 2D Transaction And Recovery

- [x] Add pending/committed/failed/rolled-back receipts with atomic reservation, CAS transitions, target locking, and durable ownership projection.
- [x] Execute validated semantic ADD/UPDATE/DELETE/NONE in isolated Hermes backends through apply, reread, verify, and commit.
- [x] Recover five crash windows deterministically, preserve blocked unknown state, and audit corruption, orphan, revision, digest, and storage evidence.
- [x] Gate logical exit on verified committed memory at task/session boundaries while keeping physical rewrite disabled.

Known boundary: the JSON/`flock` transaction layer is a single-host fixture
implementation. It is default-disabled and not connected to PAST-Bench or a
live Hermes home. Unknown ownership and unsafe compensation remain blocked for
operator/recovery handling rather than being guessed.

### Phase 2E Semantic Construction, SM01 Loop, And Attribution

- [x] Implement Mem0-flat durable fact extraction, bounded related-memory retrieval, and internal ADD/UPDATE/DELETE/NONE decisions over fixed Hermes semantic storage.
- [x] Complete deterministic SM01 learn -> validate -> mutate -> restart -> native prompt injection -> downstream-use with receipt and artifact revision audit.
- [x] Record the real source/extraction/retrieval/decision/resolution/validation/mutation/verification/future exposure/outcome chain in the content-free atomic operation graph.
- [x] Bind fact-extraction prompt, update-decision prompt, and retrieval parameters to their owned operations without retaining prompt or response text.
- [x] Add deterministic-first failure attribution, observation cutoffs, exposure eligibility, batch sampling/dedup, and default-disabled budgeted model fallback with separate policy-update usage.

Static PAST-Bench semantic writeback is complete for SM01: 9/9 rotated method attempts passed audit, with 81 unique traces and content-free ingestion/mutation evidence. Admission timing varied across replicates; delayed utility labels and adaptive policy updates remain later gates.

The frozen static utility gate is also complete for SM01. The accepted 3-replicate static/static-utility batch has 6/6 completed scheduled slots, 54 unique traces, exact utility/ingestion joins, and one stable gate/policy/schema identity across utility replicates. Results and limitations are recorded in [`static_utility_sm01_20260827.md`](static_utility_sm01_20260827.md); no quality-superiority or adaptive-policy claim is made.

The delayed-feedback lifecycle join, exposure-bias contracts, and deterministic stage gate are complete. Versioned content-free datasets preserve artifact-specific future evidence, observation cutoffs, extraction-owned labels, attribution provenance, exposure opportunity, candidate/filter disposition, integrity audit results, aggregate censoring reports, and frozen config identity. Source-, extraction-set-, and attributable fact-level prompt feedback now use separate opportunity/use/outcome contracts; raw resource usage remains outside label and activation payloads.

The former 2J.1 learner produces deterministic retrieval-threshold proposals and retains value as a legacy plumbing fixture. Its artifacts are explicitly marked `legacy_threshold_experiment` and cannot be loaded by the extraction-prompt runtime. The crash-safe artifact lifecycle store remains reusable infrastructure.

Former Phase 2J threshold infrastructure is complete at its deterministic implementation gate: time-ordered splits, offline screening, matched activation, rollback, active-artifact binding, and replay are implemented. It does not implement extraction prompt N -> N+1 and is not evidence of the current paper method.

Former Phase 2K.1 execution plumbing covers the legacy threshold preparation boundary. Offline screening and matched activation no longer use resource cost, but the resulting threshold artifact is not a production extraction artifact. Its launchers remain disabled infrastructure until extraction-specific manifests replace them.

Former Phase 2K.2 analysis plumbing is retained as legacy threshold experiment infrastructure. The extraction-specific plain-parent feedback launcher, prompt-oriented manifest, split audit, and analyzer are implemented independently. Raw request, token, storage, injection, recovery, and timing vectors remain accounting outputs and do not enter learning or activation. Adaptive extraction launch and ACTIVE artifact binding remain Phase 2 work.

### Extraction-Prompt Stage 1D Component Boundary

- [x] Give extraction, update, and retrieval independent component identities and combine them in one semantic policy manifest.
- [x] Register and bind the host-neutral `mem0-flat.semantic.extraction` slot at the real policy factory; rendered completion evidence carries the binding fingerprint.
- [x] Define `static-extraction-parent-v1` as the no-utility/no-cost parent while retaining utility modes only as legacy regression identities.
- [x] Reject matched parent/candidate drift in update, retrieval, route, task boundary, backend, framework, or model profile.
- [x] Remove the Hermes native memory tool for unset, empty, and non-empty toolset configurations while retaining native semantic prompt reads.
- [x] Disable memory/skill background review requests for RSIMem-managed semantic runs.
- [x] Distinguish RSIMem executor and operator-recovery writers in mutation receipts and reject native or disallowed writer contamination against a run-scoped storage baseline.

### Extraction-Prompt Stage 2A Policy Envelope

- [x] Define immutable root and child `ExtractionPromptPolicyArtifact` contracts with ordered stable rule IDs, structured edits, exact compiler replay, parent lineage, content digests, and optimizer provenance.
- [x] Freeze wrapper, input/output schema, placeholder, model-profile, body-length, slot, and compiler identities outside the adaptive body.
- [x] Protect source/safety exclusion and exact output-schema rules from generated replacement or deletion.
- [x] Persist proposal, active, rejected, and rolled-back artifacts in an independent crash-safe store that rejects threshold-store schemas, tampering, cycles, unknown parents, and multiple ACTIVE records.
- [x] Export the exact Mem0-flat root body as a serializable root artifact while preserving the accepted root component ID and provenance.
- [x] Bind serialized root and child artifacts through the Mem0-flat runtime bridge; only the extraction component changes in the composite semantic manifest.
- [x] Reload an ACTIVE child after store restart and reproduce the same binding, rendered prompt, and template digest in a fresh adapter.
- [x] Load the same serialized artifact through a fake adapter projection and fail closed to the trusted root on corrupt lifecycle state or slot-contract drift.

### Extraction-Prompt Stage 2B Optimizer Corpus

- [x] Project bounded source messages, exact extracted fact content, persisted artifact lineage, and deployment-observable opportunity/use/outcome evidence into an owner-controlled content-bearing corpus.
- [x] Bind every corpus example to the content-free source record, feedback record/dataset/example, operation IDs, mutation IDs, artifact IDs, and content digests.
- [x] Derive delayed content from the real `DeploymentObservation`; current input is digest-bound, use content is the actual final response, and outcome content is a deterministic completion/tool projection.
- [x] Preserve source/set/fact level, primary-unit identity, useful/harmful/missed/unresolved/censored labels, attribution confidence, reason codes, and component ownership.
- [x] Require complete useful and harmful/missed attribution chains; reject missing source, fact, operation, mutation, observation, digest, or future-cutoff joins.
- [x] Canonically order examples so the same frozen evidence produces the same corpus ID independent of input order.
- [x] Persist immutable train, validation, and future-test corpora under an explicit owner-controlled attempt root with atomic replacement, `0700` directories, `0600` files, and explicit retention.
- [x] Keep future-test reads unavailable until the declared extraction artifact is ACTIVE; optimizer and validator APIs cannot read another split.
- [x] Redact credentials, authorization headers, and machine paths only in the optimizer copy; mark all content as untrusted data and reject grader, answer-key, hidden-expectation, or judge evidence.
- [x] Audit public source/manifest/ledger/operation payloads for any corpus body leakage without treating stable IDs and digests as content leaks.

### Extraction-Prompt Stage 2C Controlled Optimizer

- [x] Freeze the optimizer system instruction, input/output schemas, actual model ID, model profile, temperature, output token budget, timeout, sample bounds, edit budget, and leakage n-gram size.
- [x] Render parent policy and train-only corpus into separate system and untrusted user messages grouped by useful, harmful, missed, unresolved, and censored extraction-set units.
- [x] Treat source/fact levels only as attribution annotations; only one extraction-set primary per future opportunity contributes evidence weight.
- [x] Accept only one `PROPOSE` or `NO_PROPOSAL` JSON object; rule edits require eligible primary evidence IDs and reason codes, and a model-provided compiled body is rejected.
- [x] Create the candidate exclusively through structured ADD/REPLACE/DELETE edits and the frozen policy compiler; generated provenance records corpus, cutoff, request/completion digests, model, config, and raw usage.
- [x] Return `NO_PROPOSAL` without a model call for no-signal, low-sample, conflicting, censored-only, unresolved-only, or non-extraction-owned evidence.
- [x] Protect durability, source-grounding/credential exclusion, and output-schema rules from generated replacement or deletion.
- [x] Reject family/benchmark shortcuts, fixed output columns, task/run/family identities, project-specific values, source long n-grams, prompt injection, credential exfiltration, and schema override.
- [x] Add an OpenAI-compatible provider client with zero SDK retry, separated system/user messages, JSON mode, frozen request parameters, and raw/unknown usage preservation; tests use a fake SDK and no provider call.
- [x] Replay identical captured completions into identical candidate artifacts and preserve optimizer usage outside the optimization objective.

### Extraction-Prompt Stage 2D Static Safety And Offline Validation

- [x] Validate candidate lineage, exact edit replay, slot/wrapper/schema compatibility, protected rules, body constraints, and forbidden adaptive instructions.
- [x] Run the eight-category deterministic extraction suite with strict `{facts: string[]}` parsing, retain/exclude expectations, and source-copy/prompt-leakage rejection.
- [x] Join static safety, deterministic suite, frozen split, artifact body digests, observations, and quality decisions by stable identity.
- [x] Compare parent and candidate at extraction-set level without official score or resource cost in the decision objective.
- [x] Report useful, harmful, coverage, empty, and missed ratios with explicit numerator, denominator, and unknown count.
- [x] Reject equal quality, insufficient resolved evidence, empty-output collapse, coverage collapse, safety failure, stale digests, incomplete split roles, and non-frozen candidate budgets.
- [x] Limit an offline acceptance to matched-trial eligibility; no offline decision API can activate a production artifact.

### Extraction-Prompt Stage 2E Deterministic Foundation

- [x] Separate rich policy artifact identity from the actual runtime prompt-component identity in matched decisions and replay.
- [x] Prepare a validation-only isolated ACTIVE store whose config cannot be loaded as a production profile.
- [x] Assemble content-free observations from formal validation manifests, source records, delayed live feedback, and run-level safety audits without reading official score or raw cost into the decision.
- [x] Require one completed run per predeclared slot, exact runtime artifact and run joins, complete feedback closure for every completed source, and paired model/budget/persistence identity.
- [x] Preserve useful, harmful, missed, unresolved, and censored set-level evidence; unresolved/censored sources remain outside resolved and missed denominators without disappearing from coverage evidence.
- [x] Apply strict useful-rate, harmful, coverage, empty, missed, safety, and intervention constraints before activation.
- [x] Persist matched decisions and rollback evidence; activation, rejection, restart, duplicate apply, crash recovery, operator rollback, and observed-safety rollback are deterministic and idempotent.
- [ ] Run the real independent PAST-Bench parent/candidate validation batch after Stage 2F binds the candidate at the extraction prompt boundary.

### Extraction-Prompt Stage 2F Runtime Binding And Fingerprint

- [x] Load the validation-only ACTIVE candidate through the real extraction policy store and bind it at the Mem0-flat fact-extraction call boundary.
- [x] Transport a content-free trial profile through PAST-Bench and revalidate an immutable attempt-local bundle without recording its machine source path.
- [x] Expose `prompt_slot(...)` as a one-line explicit registry entry with no global patching or alternate validation path.
- [x] Record rich policy, prompt component, binding, wrapper/schema, render input/output, semantic manifest, mutation, and persisted-artifact identities for every completed extraction.
- [x] Fail closed on configured/loaded artifact mismatch and slot, adapter, wrapper, schema, update, retrieval, route, boundary, backend, framework, or model drift.
- [x] Report no intervention separately when N and N+1 produce the same parsed extraction.
- [x] Reproduce the same runtime binding fingerprint after restart.

Detailed deterministic evidence is in [`extraction_stage2f_acceptance_20260828.md`](extraction_stage2f_acceptance_20260828.md). No live provider validation run or production activation is included in this milestone.

The first six-layer deterministic feasibility baseline is recorded in
[`policy_feasibility_baseline_20260829.md`](policy_feasibility_baseline_20260829.md).
It includes replay-stable parent/candidate identities, process feedback,
strict extraction-feedback projection, constrained N+1 hypothesis identity,
restart-safe future N+1 loading/intervention-path identity, and a crash-safe
content-free evidence ledger. The fixture classifies
Extraction as `optimization-ready` and the other five layers as
`validation-only`; fixture labels are not deployment labels and do not support
an uplift claim. It can be regenerated with
`python -m rsimem.memory.policy_feasibility_fixture` and verified against the
durable ledger after restart.

The runtime also emits an independent host-neutral process-feedback ledger
(`rsimem_process_feedback.jsonl`) for every RSIMem bridge attempt. Trigger,
source, extraction, admission, commit, retrieval, exposure, tool and task
outcome observations carry stable host-event, source-revision, policy-decision,
lineage, digest and receipt identities without copying memory or prompt text.
The JSONL store is lock-protected, atomically replaced and idempotent across
restart; `audit_process_events()` keeps retrieval misses, adapter failures,
tool failures, injection failures and task failures as distinct reason codes.
This is process-signal infrastructure, not a live adaptive-effect result.

### Verification Baseline

- [x] Pass all RSIMem tests: `723 passed`.
- [x] Pass the vendored PAST-Bench regression suite: `399 passed, 2 skipped`.
- [x] Pass Python import and compile checks.
- [x] Pass dependency validation with `pip check`.

## Next Milestone

### **Current: Third-stage deterministic policy feasibility**

The first candidate layer remains the semantic fact-extraction prompt. The
current gate first verifies that each Trigger/Source/Extraction/Admission/
Commit/Exposure decision is observable, controllable, replayable, and linked to
process feedback. Delayed deployment evidence may later update prompt body N to
N+1 while the operation prompt, retrieval configuration, route, invocation
boundary, backend, and model profile remain frozen. Raw resources are reported
separately from the method decision.

Stage 2F binds the validation-only ACTIVE extraction artifact to the actual Mem0-flat prompt call and records the complete activation fingerprint while freezing the completion client, model profile, update prompt, retrieval config, route, boundary, and backend. The real predeclared parent/proposal validation batch remains deferred until feasibility gates are sufficient. Candidate artifacts remain proposals and production activation remains blocked until that live gate passes.

The accepted run contains 17 unique physical traces, 68 fully accounted model requests, 34 task/session lifecycle chains, 28 exact native-shadow checks, and zero audit, privacy, projection, bypass, or lifecycle-rejection issues. Direct native remains the default. Phase 2 must preserve the frozen route and invocation boundary and remain opt-in until each later gate passes.

- [x] Define an experiment configuration that selects one backend for each memory kind and defaults to native Hermes behavior.
- [x] Add a factory that constructs the selected registry and runtime from an isolated experiment home.
- [x] Construct context snapshots at explicit task-completion and session-end boundaries; context-pressure remains unavailable until Hermes exposes trusted usage and threshold inputs.
- [x] Add default-disabled deterministic and injected-JSON lifecycle evaluator modes using Hermes request accounting and enforced provider timeout.
- [x] Identify and exercise the Hermes semantic prompt, episodic search, and skill read interception points in an isolated execution fixture; mutation interception remains gated on compiler validation.
- [x] Add an opt-in adapter execution path while retaining the direct native path as the control.
- [x] Forward runtime lifecycle events into the existing ledger through stable run, episode, session, and artifact identifiers.
- [x] Add host-neutral snapshot, evaluation, plan, revision, safety, and persistent-idempotency validation before allowing compiler-generated content to mutate Hermes files.
- [x] Define explicit fail-closed and native-bypass behavior for deterministic execution surfaces and the PAST-Bench runner.
- [x] Add an explicit SM01 live-run launcher that fixes the three execution modes, paired persistence control, failure policy, raw ledger/audit output, and replicate count.
- [x] Record that the current model runtime does not expose a controllable provider seed instead of labeling independent replicates as seeded runs.

Acceptance criteria:

- [x] The deterministic direct-native and adapter-native fixture variants receive identical task inputs and use identical Hermes storage semantics.
- [x] A deterministic fixture produces equivalent model-visible semantic memory, episodic search results, and procedural resources in both variants.
- [x] Every deterministic fixture query and injection appears once in lifecycle evidence without memory text leakage; mutation evidence remains gated on real writeback.
- [x] Restarting the deterministic fixture preserves the selected backend state and artifact identity behavior.
- [x] The full RSIMem and PAST-Bench regression suites continue to pass.

## Following Milestones

### Native Behavior Equivalence Experiment

- [x] Add `native`, `native+ledger`, and `native+adapter+ledger` storage-boundary variants.
- [x] Run matched deterministic fixtures before spending model tokens.
- [x] Invoke real Hermes memory prompt construction, `session_search`, `skills_list`, and `skill_view` in matched deterministic fixtures.
- [x] Verify deterministic restart persistence and explicit adapter failure bypass.
- [x] Complete and audit one live unseeded infrastructure replicate on the earlier hybrid adapter path; record the provider-failed attempt separately.
- [x] Run at least three matched independent `SM01_preference_adoption` replicates with fixed model, prompt, task order, sandbox, and budget; see `matched_phase1c_20260827.md`.
- [ ] Compare task score, pass rate, model requests, token buckets, tool calls, stored bytes, injected characters, and wall time.
- [ ] Establish an explicit tolerance for nondeterministic model variation and require zero unexplained accounting drift.
- [ ] Treat adapter instrumentation as behaviorally neutral only after the matched comparison passes.

### Memory Policy Baselines

- [x] Freeze Hermes routing and the semantic invocation boundary across all active policy variants.
- [x] Implement a deterministic pass-through ingestor for fixtures and contract validation.
- [x] Define provenance so every ingestion links back to its source episode, fixed route, internal operation, and later retrieval.
- [x] Locally reimplement Mem0 flat fact extraction, related-memory comparison, and internal ADD/UPDATE/DELETE/NONE policy from the pinned MemBase source.
- [x] Keep MemBase datasets, runners, evaluation code, graph store, and runtime imports outside RSIMem.
- [x] Add a bounded MemTrace-inspired, content-free atomic operation graph without importing MemTrace/smartcomment runtime or tracing arbitrary calls.
- [x] Attribute failures to extraction, internal decision, mutation, or retrieval subgraphs before updating policy.
- [x] Keep episodic/procedural mutation and adaptive learning disabled until their separate research gates select a method and matched baseline; typed read-surface and deterministic policy contracts remain available for feasibility replay.
- [x] Record ingestion model requests, latency, tokens, internal operations, rejected operations, and stored bytes as lifecycle cost.
- [x] Add validation, rollback, and idempotency tests for framework-produced internal operations.

### Static LightRSI Writeback

- [x] Invoke the Hermes semantic route at the same task/session boundary used by its control policy.
- [x] Expose only ingest/add externally; let the semantic policy produce ADD, UPDATE, DELETE, or NONE.
- [x] Preserve the former static utility objective as a superseded infrastructure experiment; it is not the current adaptive method.
- [x] Link source context, ingestion execution, internal operation, stored artifact, retrieval, injection, downstream execution, and outcome through stable lifecycle IDs.
- [x] Compare no persistence, native memory, and static LightRSI under matched settings.
- [ ] Verify that quality gains are reported together with ingestion-policy, storage, retrieval, injection, and recovery costs.

### Adaptive LightRSI Loop

- [x] Collect deployment-observable delayed feedback from retrieval, injection, task completion, tool behavior, retries, supersession, and non-use.
- [x] Preserve deployment-observable delayed feedback and strict attribution diagnostics without reading hidden task scores into the policy update path; legacy utility estimates remain historical infrastructure, not the current objective.
- [ ] Version and deploy semantic extraction prompt N+1 while freezing operation, retrieval, routing, and invocation components.
- [x] Validate proposed policy updates against held-out deployment evidence.
- [x] Support acceptance, rejection, rollback, and reproducible replay of every policy update.
- [ ] Compare static and adaptive extraction prompts on deployment-observable quality while reporting each raw resource dimension separately.

### PAST-Bench Task-Family Expansion

- [ ] Expand from `SM01` to the preselected semantic-relevant memory-ability families.
- [ ] Add semantic-relevant update-ability families.
- [ ] Add episodic/procedural families only after their method-selection gates pass; they do not block the semantic-first paper path.
- [ ] Keep Hermes native memory as the fixed backend across all current-paper experiments.
- [ ] Keep MemBase runtime/evaluation code, external memory backends, other hosts, and additional benchmarks outside the current implementation scope.

## Required Experiment Invariants

- [ ] Never modify benchmark task semantics, hidden expectations, answer keys, or grading criteria to improve results.
- [ ] Never expose hidden grader evidence to an ingestor/generator, backend, controller, or policy update.
- [ ] Keep raw resource quantities separate from provider prices.
- [ ] Account for controller, ingestion/generation, retrieval-policy, and policy-update calls as experiment cost rather than free preprocessing.
- [ ] Deduplicate physical model requests across shared executions before reporting cost.
- [ ] Keep memory content out of the lifecycle ledger and audit outputs.
- [ ] Pin benchmark, agent, LightRSI, model profile, judge profile, task manifest, and seed for every reported run.
- [ ] Preserve raw traces and derived tables so every aggregate can be recomputed.

## Immediate Execution Order

1. [x] Implement Mem0-flat fact extraction and internal ADD/UPDATE/DELETE/NONE decision using the frozen prompt/completion contracts.
2. [x] Define bounded related-memory retrieval parameters and bind candidate targets through the committed ownership resolver.
3. [x] Complete the SM01 learn -> ingest -> validate -> mutate -> restart -> native injection -> downstream-use fixture.
4. [x] Connect real extraction, retrieval, decision, validation, mutation, verification, injection, use/non-use, and outcome operation evidence.
5. [x] Run and audit the static SM01 matched comparison.
6. [x] Complete and audit the Stage 1H live plain-parent smoke; see `extraction_stage1_acceptance_20260828.md`.
7. [x] Implement the extraction policy envelope, independent artifact store, Mem0-flat root export, runtime binding, restart replay, and root fallback.
8. [x] Build the content-bearing optimizer corpus with exact audit joins and train/validation/future-test isolation.
9. [x] Implement the controlled extraction prompt optimizer with captured replay, provider adapter, evidence gating, and candidate leakage/safety rejection.
10. [x] Implement static safety and held-out offline validation for extraction prompt candidates.
11. [x] Implement the predeclared matched-trial contract, validation-only store, evidence assembler, activation, rejection, restart, and rollback gates.
12. [x] Bind the validation-only ACTIVE extraction artifact to the real Mem0-flat prompt call and record its runtime fingerprint.
13. [x] Build the deterministic six-layer feasibility replay/census with target-layer intervention and process-feedback lineage.
14. [x] Bind strict primary extraction feedback to feasibility chains and persist replay identities across restart.
15. [ ] Run a predeclared matched static-extraction/adaptive-extraction SM01 batch after feasibility gates permit it.
16. [ ] Produce dated extraction-method reports before making adaptation claims.

The backup-provider v10 plain-parent batch completed three clean replicates and
reconstructed the private optimizer corpus, but all 24 primary labels were
unresolved. The deterministic signal gate returned `NO_PROPOSAL` with zero
optimizer calls and no candidate. The matched Stage 2E item therefore remains
open; details are recorded in
[`extraction_stage2e_feedback_v10_20260828.md`](extraction_stage2e_feedback_v10_20260828.md).
It is not a blocker for the deterministic feasibility baseline, but real N+1
provider validation remains deferred until a family supplies sufficient
actionable signal.

The corrected plain-parent SM01 batch `s1-sm01-feedback-20260829-v9a` then
completed three clean real-provider replicates.  All audits and process
corpora passed; 24 primary feedback examples remained unresolved, so corpus
preparation returned `optimizerSignalReady=false` and the optimizer made zero
provider calls with `NO_PROPOSAL`.  This is a valid no-signal feedback pilot,
not an effect result.  Raw usage and disposition are recorded in
[`extraction_stage3_sm01_feedback_v9a_20260829.md`](extraction_stage3_sm01_feedback_v9a_20260829.md).

On 2026-08-29, a transient successful provider probe allowed a clean SM01
feedback attempt (`v5`) to start.  It exposed a runtime boundary bug: the
separate PAST-Bench reflection episode was incorrectly sent through semantic
extraction and failed closed for a missing invocation fingerprint.  Commit
`e22af5c` now keeps reflection process-only and reserves extraction for the
primary task-completed boundary, with a regression test.  A post-fix clean
attempt (`v6`) was gated before its first task by another provider HTTP 503;
neither attempt is eligible feedback or effect evidence.
The complete v5–v8 diagnostics and the v7 canonical process-corpus replay are
documented in [`extraction_stage3_sm01_feedback_attempts_20260829_v5_v8.md`](extraction_stage3_sm01_feedback_attempts_20260829_v5_v8.md).

The SM02 process-signal family subsequently completed three clean parent
replicates in `s1-sm02-feedback-20260829-v5`.  It yielded 8 contract-resolved
`missed` observations and 16 `unresolved` observations, with no
useful/harmful variation.  The content-bearing corpus was reconstructed and
the bounded optimizer request was built, but the provider returned malformed
optimizer JSON twice; strict parsing rejected both and produced no candidate.
This is recorded as provider/model-output diagnostics, not a task negative,
in [`extraction_stage3_sm02_feedback_v5_20260829.md`](extraction_stage3_sm02_feedback_v5_20260829.md).

After the provider recovered, a bounded retry confirmed that the endpoint's
`json_object` mode still omitted the required top-level `reason_codes`. The
adapter was changed in commit `d2d06fc` to send the frozen strict JSON Schema
response contract. The unchanged SM02 corpus then yielded one schema-valid
`PROPOSE`; candidate static safety passed and the artifact remains a proposal,
not ACTIVE or matched-validation evidence. The retry report is in
[`extraction_stage3_sm02_optimizer_retry_20260829.md`](extraction_stage3_sm02_optimizer_retry_20260829.md).

An independent SM05 process pilot (`s1-sm05-feedback-20260829-v1`) then
completed three clean nine-episode replicates. It produced 24 extraction-owned
`missed` and 12 `unresolved` primary observations, with 89 canonical process
events per replicate (267 in the batch-level process corpus). The optimizer
request initially exceeded the frozen input budget; the
deterministic replica compaction in commit `f194bf8` reduced it to 158,538
characters while retaining all primary IDs. The provider returned schema-valid
proposals, but the candidate content-safety gate rejected corpus-specific
value copying on both attempts, so no candidate was written. Details are in
[`extraction_stage3_sm05_optimizer_20260829.md`](extraction_stage3_sm05_optimizer_20260829.md).

Commit `3373a78` adds an explicit `ExtractionSplitPlan` contract and optional
matched-preflight validation. The current SM01, SM02, and SM05 pilots all
occupy train roles, so none of those manifests may be relabeled as validation.
An explicit held-out plan now reserves SM03 fact correction for validation and
SM04 rule migration for final test (`configs/extraction_split_plan_sm02_sm03_sm04.json`).
SM03 is registered as an extraction-only validation contract with its update
prompt frozen; the plan is identity bookkeeping only and no matched validation
has run yet ([`extraction_stage3_split_audit_20260829.md`](extraction_stage3_split_audit_20260829.md)).

## Update Policy

Update this file whenever a milestone is completed, its acceptance criteria change, or evidence reveals a new blocker. Mark an item complete only after its implementation, tests, and required experiment evidence are all available. Record detailed numerical results in a separate dated report and link that report here rather than embedding provisional paper results in the checklist.
