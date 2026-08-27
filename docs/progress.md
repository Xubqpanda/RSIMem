# RSIMem Progress

Last updated: 2026-08-27

This document tracks implementation progress, the current experimental boundary, and the next executable milestones. Research motivation and the full staged evaluation design remain in [`experiment_plan.md`](experiment_plan.md). The detailed lifecycle implementation sequence is in [`lifecycle_implementation_plan.md`](lifecycle_implementation_plan.md), and the complete two-stage serial implementation and acceptance requirements are in [`implementation_handoff_checklist.md`](implementation_handoff_checklist.md).

## Status Legend

- [x] Completed and verified.
- [ ] Not completed.
- **Current** identifies the milestone that should receive implementation effort next.

## Current State

RSIMem can run the vendored PAST-Bench with Hermes and GPT-Luna, account for every exposed model request, derive a privacy-safe lifecycle ledger, audit run completeness, and represent Hermes semantic, episodic, and procedural memory through typed backend contracts.

The typed memory runtime is connected to the PAST-Bench Hermes execution path behind an explicit opt-in mode. Direct native remains the default, and no RSIMem ingestor, real writeback mutation, static policy, or adaptive policy is active yet. The active implementation scope is now semantic-first: Mem0 flat policy logic over Hermes native semantic storage. Episodic and procedural adapters remain verified read surfaces, but their policy implementations are deferred until methods are selected.

Phase 1A environment/configuration freeze, Phase 1B deterministic read-path equivalence, and Phase 1C live matched SM01 read-path validation are complete. Phase 1D real Hermes lifecycle dry-run integration is the active gate; all Phase 2 implementation remains blocked until Phase 1 is frozen.

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

### Verification Baseline

- [x] Pass all RSIMem tests: `95 passed`.
- [x] Pass the vendored PAST-Bench regression suite: `384 passed, 2 skipped`.
- [x] Pass Python import and compile checks.
- [x] Pass dependency validation with `pip check`.

## Next Milestone

### **Current: Real Hermes Lifecycle Dry-Run Wiring**

The Phase 1C live matched gate passed on three order-rotated, independently unseeded replicates per mode. The dated report is [`matched_phase1c_20260827.md`](matched_phase1c_20260827.md). The immediate objective is Phase 1D: wire snapshot collection, lifecycle evaluation, and validated dry-run plans into explicit real Hermes task-completion and session-end events without memory mutation or context eviction.

- [x] Define an experiment configuration that selects one backend for each memory kind and defaults to native Hermes behavior.
- [x] Add a factory that constructs the selected registry and runtime from an isolated experiment home.
- [ ] Construct context snapshots at task completion and configured context-pressure boundaries.
- [ ] Add an opt-in lifecycle evaluator configuration with an injected model client and deterministic baseline.
- [x] Identify and exercise the Hermes semantic prompt, episodic search, and skill read interception points in an isolated execution fixture; mutation interception remains gated on compiler validation.
- [x] Add an opt-in adapter execution path while retaining the direct native path as the control.
- [x] Forward runtime lifecycle events into the existing ledger through stable run, episode, session, and artifact identifiers.
- [ ] Add host-neutral validation hooks before allowing compiler-generated content to mutate Hermes files.
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
- [ ] Run at least three matched independent `SM01_preference_adoption` replicates with fixed model, prompt, task order, sandbox, and budget.
- [ ] Compare task score, pass rate, model requests, token buckets, tool calls, stored bytes, injected characters, and wall time.
- [ ] Establish an explicit tolerance for nondeterministic model variation and require zero unexplained accounting drift.
- [ ] Treat adapter instrumentation as behaviorally neutral only after the matched comparison passes.

### Memory Policy Baselines

- [ ] Freeze Hermes routing and the semantic invocation boundary across all active policy variants.
- [ ] Implement a deterministic pass-through ingestor for fixtures and contract validation.
- [ ] Define provenance so every ingestion links back to its source episode, fixed route, internal operation, and later retrieval.
- [ ] Locally reimplement Mem0 flat fact extraction, related-memory comparison, and internal ADD/UPDATE/DELETE/NONE policy from the pinned MemBase source.
- [ ] Keep MemBase datasets, runners, evaluation code, graph store, and runtime imports outside RSIMem.
- [ ] Add a bounded MemTrace-inspired, content-free atomic operation graph without importing MemTrace/smartcomment runtime or tracing arbitrary calls.
- [ ] Attribute failures to extraction, internal decision, mutation, or retrieval subgraphs before updating policy.
- [ ] Keep episodic and procedural policy implementation disabled until their separate research gates select a method and matched baseline.
- [ ] Record ingestion model requests, latency, tokens, internal operations, rejected operations, and stored bytes as lifecycle cost.
- [ ] Add validation, rollback, and idempotency tests for framework-produced internal operations.

### Static LightRSI Writeback

- [ ] Invoke the Hermes semantic route at the same task/session boundary used by its control policy.
- [ ] Expose only ingest/add externally; let the semantic policy produce ADD, UPDATE, DELETE, or NONE.
- [ ] Define one fixed future-utility-per-cost objective for evaluating semantic construction/update and retrieval behavior.
- [ ] Link source context, ingestion execution, internal operation, stored artifact, retrieval, injection, downstream execution, and outcome through stable lifecycle IDs.
- [ ] Compare no persistence, native memory, and static LightRSI under matched settings.
- [ ] Verify that quality gains are reported together with ingestion-policy, storage, retrieval, injection, and recovery costs.

### Adaptive LightRSI Loop

- [ ] Collect deployment-observable delayed feedback from retrieval, injection, task completion, tool behavior, retries, supersession, and non-use.
- [ ] Estimate realized future utility without reading hidden task scores into the policy update path.
- [ ] Version semantic extraction, operation/consolidation, and retrieval policies without changing routing or invocation frequency.
- [ ] Validate proposed policy updates against held-out deployment evidence.
- [ ] Support acceptance, rejection, rollback, and reproducible replay of every policy update.
- [ ] Compare static and adaptive LightRSI on the cost-quality frontier.

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

1. Run matched independent replicates and compare quality, usage, tool, storage, injection, and timing evidence.
2. Define and enforce the acceptable live-model nondeterminism boundary.
3. Add provider-seed control only if the selected runtime exposes and verifies it.
4. Implement the validated Mem0-style semantic ingestor and backend mutation baseline.
5. Begin the static RSIMem writeback policy only after runtime equivalence is established.

## Update Policy

Update this file whenever a milestone is completed, its acceptance criteria change, or evidence reveals a new blocker. Mark an item complete only after its implementation, tests, and required experiment evidence are all available. Record detailed numerical results in a separate dated report and link that report here rather than embedding provisional paper results in the checklist.
