# RSIMem Progress

Last updated: 2026-08-25

This document tracks implementation progress, the current experimental boundary, and the next executable milestones. Research motivation and the full staged evaluation design remain in [`experiment_plan.md`](experiment_plan.md). The detailed lifecycle implementation sequence is in [`lifecycle_implementation_plan.md`](lifecycle_implementation_plan.md).

## Status Legend

- [x] Completed and verified.
- [ ] Not completed.
- **Current** identifies the milestone that should receive implementation effort next.

## Current State

RSIMem can run the vendored PAST-Bench with Hermes and GPT-Luna, account for every exposed model request, derive a privacy-safe lifecycle ledger, audit run completeness, and represent Hermes semantic, episodic, and procedural memory through typed backend contracts.

The new memory runtime is not yet connected to the PAST-Bench execution path. Hermes native tools remain the behavioral baseline, and no LightRSI admission, distillation, retrieval, or adaptive policy is active yet.

## Completed Work

### Repository And Benchmark Foundation

- [x] Define RSIMem as the experiment and evaluation repository for memory-mediated recursive self-improvement.
- [x] Select PAST-Bench as the first long-horizon benchmark and document the selection rationale.
- [x] Vendor PAST-Bench and Hermes Agent under `benchmarks/past-bench` with upstream attribution and licenses.
- [x] Establish a Python 3.11 development environment and reproducible local installation procedure.
- [x] Preserve task definitions, graders, answer keys, and native persistence semantics.
- [x] Add a repeatable GPT-Luna smoke launcher for `SM01_preference_adoption`.

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

### Deterministic Native Equivalence

- [x] Define explicit `native`, `native+ledger`, and `native+adapter+ledger` modes with direct native behavior as the default.
- [x] Add a factory for constructing the three-route Hermes typed runtime from an isolated home.
- [x] Verify identical semantic prompt blocks, episodic search results with surrounding context, and procedural skill resources across native and adapter reads.
- [x] Preserve native semantic entry order and Hermes episodic surrounding-context behavior in the adapter.
- [x] Join content-free snapshot, plan, validation, and dry-run events into the existing ledger schema.

### Verification Baseline

- [x] Pass all RSIMem tests: `32 passed`.
- [x] Pass the vendored PAST-Bench regression suite: `380 passed, 2 skipped`.
- [x] Pass Python import and compile checks.
- [x] Pass dependency validation with `pip check`.

## Next Milestone

### **Current: Opt-In Memory Runtime Integration**

The immediate objective is to connect the typed memory runtime and lifecycle control plane to experiments without changing the native Hermes baseline. Follow [`lifecycle_implementation_plan.md`](lifecycle_implementation_plan.md) for the staged contracts, snapshot collector, dry-run plan, writeback, and acceptance order.

- [x] Define an experiment configuration that selects one backend for each memory kind and defaults to native Hermes behavior.
- [x] Add a factory that constructs the selected registry and runtime from an isolated experiment home.
- [ ] Construct context snapshots at task completion and configured context-pressure boundaries.
- [ ] Add an opt-in lifecycle evaluator configuration with an injected model client and deterministic baseline.
- [ ] Identify and document the exact Hermes semantic write, episodic search, skill read, and skill mutation interception points.
- [ ] Add an opt-in adapter execution path while retaining the direct native path as the control.
- [ ] Forward runtime lifecycle events into the existing ledger through stable run, episode, session, and artifact identifiers.
- [ ] Add host-neutral validation hooks before allowing compiler-generated content to mutate Hermes files.
- [ ] Ensure adapter failures bypass or fail explicitly according to experiment configuration and never silently corrupt native state.

Acceptance criteria:

- [ ] The direct-native and adapter-native variants receive identical task inputs and use identical Hermes storage semantics.
- [ ] A deterministic fixture produces equivalent model-visible semantic memory, episodic search results, and procedural resources in both variants.
- [ ] Every adapter query, injection, and mutation appears once in lifecycle evidence without memory text leakage.
- [ ] Restarting between episodes preserves the selected backend state and artifact identity behavior.
- [ ] The full RSIMem and PAST-Bench regression suites continue to pass.

## Following Milestones

### Native Behavior Equivalence Experiment

- [x] Add `native`, `native+ledger`, and `native+adapter+ledger` storage-boundary variants.
- [x] Run matched deterministic fixtures before spending model tokens.
- [ ] Run at least three matched `SM01_preference_adoption` seeds with fixed model, prompt, task order, sandbox, and budget.
- [ ] Compare task score, pass rate, model requests, token buckets, tool calls, stored bytes, injected characters, and wall time.
- [ ] Establish an explicit tolerance for nondeterministic model variation and require zero unexplained accounting drift.
- [ ] Treat adapter instrumentation as behaviorally neutral only after the matched comparison passes.

### Compiler Baselines

- [ ] Implement a deterministic pass-through compiler for fixtures and contract validation.
- [ ] Define candidate provenance so every compiled mutation links back to its source episode and context-exit event.
- [ ] Implement fixed semantic, episodic, and procedural compiler baselines without adaptive LightRSI decisions.
- [ ] Represent Text2Skill and SkillCreator as procedural compilers rather than storage backends.
- [ ] Record compiler model requests, latency, tokens, accepted mutations, rejected mutations, and stored bytes as lifecycle cost.
- [ ] Add validation, rollback, and idempotency tests for compiler-produced mutations.

### Static LightRSI Writeback

- [ ] Define task-aligned context-exit candidates without using hidden benchmark labels or grader evidence.
- [ ] Implement one fixed joint action space: discard, retain as episodic, distill as semantic, distill as procedural, update, or supersede.
- [ ] Define the first fixed utility-per-cost criterion used consistently for writeback and later retrieval.
- [ ] Link source context, compiler execution, stored artifact, retrieval, injection, downstream execution, and outcome through stable lifecycle IDs.
- [ ] Compare no persistence, native memory, and static LightRSI under matched settings.
- [ ] Verify that quality gains are reported together with controller, compilation, storage, retrieval, injection, and recovery costs.

### Adaptive LightRSI Loop

- [ ] Collect deployment-observable delayed feedback from retrieval, injection, task completion, tool behavior, retries, supersession, and non-use.
- [ ] Estimate realized future utility without reading hidden task scores into the policy update path.
- [ ] Version the joint context-exit, memory-form, and retrieval policy.
- [ ] Validate proposed policy updates against held-out deployment evidence.
- [ ] Support acceptance, rejection, rollback, and reproducible replay of every policy update.
- [ ] Compare static and adaptive LightRSI on the cost-quality frontier.

### Backend And Benchmark Expansion

- [ ] Add Mem0 as the first external storage and retrieval backend.
- [ ] Add LangMem or Graphiti after the adapter contract is stable.
- [ ] Keep every comparison paired as native backend versus the same backend with LightRSI.
- [ ] Expand from `SM01` to all memory-ability families.
- [ ] Add update-ability, procedural-reuse, and proactive-information-gathering families.
- [ ] Evaluate a second long-horizon benchmark only after PAST-Bench accounting and lifecycle attribution are stable.

## Required Experiment Invariants

- [ ] Never modify benchmark task semantics, hidden expectations, answer keys, or grading criteria to improve results.
- [ ] Never expose hidden grader evidence to a compiler, backend, controller, or policy update.
- [ ] Keep raw resource quantities separate from provider prices.
- [ ] Account for controller and compiler calls as experiment cost rather than free preprocessing.
- [ ] Deduplicate physical model requests across shared executions before reporting cost.
- [ ] Keep memory content out of the lifecycle ledger and audit outputs.
- [ ] Pin benchmark, agent, LightRSI, model profile, judge profile, task manifest, and seed for every reported run.
- [ ] Preserve raw traces and derived tables so every aggregate can be recomputed.

## Immediate Execution Order

1. Connect the explicit modes and lifecycle collector to the opt-in PAST-Bench/Hermes execution path.
2. Join typed memory query, retrieval, injection, and mutation events to the existing ledger.
3. Verify restart persistence and explicit adapter failure behavior.
4. Implement validated compiler and backend mutation baselines.
5. Begin the static RSIMem writeback policy only after runtime equivalence is established.

## Update Policy

Update this file whenever a milestone is completed, its acceptance criteria change, or evidence reveals a new blocker. Mark an item complete only after its implementation, tests, and required experiment evidence are all available. Record detailed numerical results in a separate dated report and link that report here rather than embedding provisional paper results in the checklist.
