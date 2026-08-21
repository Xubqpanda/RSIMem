# RSIMem Progress

Last updated: 2026-08-21

This document tracks implementation progress, the current experimental boundary, and the next executable milestones. Research motivation and the full staged evaluation design remain in [`experiment_plan.md`](experiment_plan.md).

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

### Verification Baseline

- [x] Pass all RSIMem tests: `17 passed`.
- [x] Pass the vendored PAST-Bench regression suite: `380 passed, 2 skipped`.
- [x] Pass Python import and compile checks.
- [x] Pass dependency validation with `pip check`.

## Next Milestone

### **Current: Opt-In Memory Runtime Integration**

The immediate objective is to connect the typed memory runtime to experiments without changing the native Hermes baseline.

- [ ] Define an experiment configuration that selects one backend for each memory kind and defaults to native Hermes behavior.
- [ ] Add a factory that constructs the selected registry and runtime from an isolated experiment home.
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

- [ ] Add `native`, `native+ledger`, and `native+adapter+ledger` experiment variants.
- [ ] Run matched deterministic fixtures before spending model tokens.
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

1. Add opt-in configuration and runtime construction without changing the benchmark path.
2. Build deterministic native-versus-adapter equivalence fixtures.
3. Connect one memory surface at a time: semantic, episodic, then procedural.
4. Join adapter lifecycle events to the ledger and extend the audit.
5. Run matched native behavior equivalence experiments.
6. Implement deterministic compiler baselines.
7. Begin the static LightRSI writeback policy only after equivalence is established.

## Update Policy

Update this file whenever a milestone is completed, its acceptance criteria change, or evidence reveals a new blocker. Mark an item complete only after its implementation, tests, and required experiment evidence are all available. Record detailed numerical results in a separate dated report and link that report here rather than embedding provisional paper results in the checklist.
