# Lifecycle Implementation Plan

This document records the implementation path from the current lifecycle control plane to an end-to-end LightRSI memory writeback experiment. The plan keeps the native Hermes behavior as the control and introduces each new layer behind an opt-in path.

## Target Architecture

```text
Host adapter
    |
    v
Context snapshot collector
    |
    v
Evaluation scheduler
    |
    v
Context evaluator
    |
    v
Lifecycle policy
    |
    v
Writeback coordinator
    |
    v
Memory compiler
    |
    v
Memory runtime and selected backend
    |
    v
Retrieval, injection, and delayed feedback
```

Each layer has one responsibility:

- **Host adapter** reads native messages, task boundaries, tool state, active-turn state, and token accounting from Hermes or a future host.
- **Context snapshot collector** converts native messages into stable `ContextSegment` values with session, task, revision, completion, and tool-closure metadata.
- **Evaluation scheduler** decides when evaluation is allowed. It does not know whether the evaluator is an LLM or a local model.
- **Context evaluator** estimates completion, future reuse, freshness, unresolved state, and a suggested memory form from a snapshot.
- **Lifecycle policy** converts the assessment and lifecycle cost model into a joint context and memory decision.
- **Writeback coordinator** turns an accepted decision into a validated, provenance-linked writeback plan.
- **Memory compiler** distills an experience into semantic, episodic, or procedural mutations.
- **Memory runtime and backend** validate and apply mutations while preserving each backend's native storage and retrieval behavior.
- **Feedback collector** joins retrieval, injection, task, tool, retry, and cost evidence for later policy updates.

## Evaluation Cadence

The first policy evaluates at task completion and session end. Context-pressure evaluation is enabled only when the host supplies a token threshold. Turn-interval and tool-boundary evaluation remain opt-in. Evaluation is not performed per token because the evaluator's own cost could exceed the context savings.

The first cadence comparison should contain:

```text
task completion only
task completion plus context pressure
fixed turn interval
```

The scheduler must not advance its state when evaluation fails. A retry must be possible without silently changing the active context.

## Implementation Stages

### Stage 1: Snapshot And Plan Contracts

- [x] Define a `ContextSnapshot` that groups segments, active IDs, task/session identity, context revision, and token totals.
- [x] Define a `WritebackPlan` containing context actions, memory actions, memory kind, provenance, and policy version.
- [x] Preserve tool call/result closure and reject plans that split a closure.
- [x] Require active, current-turn, unresolved, and open-tool segments to remain in context.
- [x] Require `current_turn_id` to reference a real turn or be `None`.
- [x] Require update targets, expected memory revisions, compiler versions, and backend capabilities.
- [x] Resolve UPDATE hints through the selected backend registry and verify backend ownership, artifact existence, stored revision, allowlist membership, and update capability before creating an executable plan.
- [x] Carry structured `ExitEvidence` from the signal into the writeback plan.
- [x] Add deterministic contract tests for duplicate IDs, stale revisions, incomplete decisions, and unsafe eviction.

### Stage 2: Hermes Snapshot Collector

- [x] Read a host-neutral Hermes transcript fixture and current task state through an isolated adapter.
- [x] Assign stable segment IDs that do not depend on list position alone.
- [x] Identify completed task blocks, current user turn, and tool call/result pairs.
- [x] Preserve supplied token counts and report their total to the scheduler.
- [x] Add the `SM01_preference_adoption` fixture without spending model tokens.

### Stage 3: Dry-Run Writeback Coordinator

- [x] Convert validated lifecycle signals into a plan without changing Hermes files.
- [x] Link every candidate to `session_id`, `task_id`, `evaluation_id`, `segment_id`, and `policy_version`.
- [x] Enforce target-aware idempotency and support persistent receipts across coordinator restarts.
- [x] Include all compiler-relevant exit evidence in the canonical idempotency identity while keeping equivalent reevaluations stable.
- [x] Derive the logical plan ID from the idempotency identity while retaining evaluation IDs in provenance and audit events.
- [x] Strictly validate deterministic eviction safety as a boolean contract.
- [x] Atomically reserve persistent dry-run receipts before accepting a mutation simulation.
- [x] Validate that `add` and `update` decisions declare a compatible memory kind.
- [x] Emit content-free plan and validation evidence.
- [x] Fail closed on malformed idempotency receipts and conflicting ledger event payloads.

### Stage 4: Storage-Boundary Equivalence Baseline

- [x] Add an opt-in experiment configuration selecting native or RSIMem adapter execution, defaulting to direct native behavior.
- [x] Run `native`, `native+ledger`, and `native+adapter+ledger` variants on a deterministic storage-boundary fixture.
- [x] Add observer-only evidence to the direct-native storage helper used by `native+ledger`.
- [x] Confirm identical deterministic semantic rendering, episodic FTS views, and procedural resources at the storage-helper boundary.
- [ ] Call Hermes' real memory prompt construction, `session_search`, `skills_list`, and `skill_view` paths.
- [ ] Confirm restart persistence, artifact identity, and failure bypass behavior.
- [ ] Run matched variants through the PAST-Bench execution path.
- [x] Keep the direct Hermes path unchanged as the control.

### Stage 5: Validated Memory Writeback

- [ ] Invoke the compiler only after the lifecycle decision accepts a candidate.
- [ ] Route semantic facts and preferences to semantic memory, situated task records to episodic memory, and reusable workflows to procedural memory.
- [ ] Validate compiler output before backend mutation.
- [ ] Persist memory before evicting the source context.
- [ ] Add revision checks, idempotency, failure bypass, and rollback tests.
- [ ] Add pending/committed receipt states and crash recovery around real backend mutations; dry-run reservation alone is not an exactly-once guarantee.
- [ ] Account for compiler model calls, tokens, latency, storage bytes, and rejected mutations.

### Stage 6: Static RSIMem Evaluation

- [ ] Run the fixed writeback policy on `SM01_preference_adoption` first.
- [ ] Compare no persistence, native Hermes, and static RSIMem with matched task order, model, seed, and budget.
- [ ] Report task score, persistence gap, memory retrieval, injected tokens, model calls, tool calls, retries, controller cost, compiler cost, storage cost, and wall time.
- [ ] Expand to procedural and update families only after the first family is fully attributable.

### Stage 7: Adaptive Policy

- [ ] Collect delayed evidence for retrieval, injection, actual use, task outcome, tool behavior, retries, supersession, non-use, and lifecycle cost.
- [ ] Estimate realized future utility without using hidden grader labels in the policy path.
- [ ] Version the joint context-exit and memory-form policy.
- [ ] Validate proposed policy versions on held-out episodes before activation.
- [ ] Support rejection, rollback, and reproducible replay.

## Safe Execution Order

The writeback path must follow this order:

```text
snapshot
  -> evaluate
  -> validate
  -> compile
  -> validate mutation
  -> persist memory
  -> confirm persistence
  -> evict context
```

An evaluator or compiler failure must never cause the source context to be evicted. A backend mutation failure must leave the native context available for bypass or retry. Raw memory content may be present in evaluator and compiler inputs, but observer-facing ledger events must contain only IDs, actions, counts, sizes, and reason codes.

## First End-To-End Acceptance Case

The first end-to-end case is `SM01_preference_adoption`:

```text
learn episode
  -> capture Hermes context
  -> task_completed evaluation
  -> identify the TSV preference
  -> create a semantic writeback plan
  -> persist MEMORY.md
  -> start a fresh eval_near session
  -> retrieve and inject the preference
  -> record task outcome and lifecycle cost
```

Acceptance requires native behavior to remain unchanged, the memory artifact to survive restart, the new session to retrieve it, and every lifecycle stage to be joinable through stable IDs. The ledger must reconcile all model and tool usage without storing memory text.

## Current Boundary

The scheduler, evaluator protocol, JSON LLM evaluator, conservative evaluator, lifecycle controller, snapshot contracts, deterministic Hermes fixture, and dry-run writeback coordinator are implemented in `rsimem.lifecycle`. Storage-boundary deterministic equivalence baseline completed. This does not establish Hermes execution equivalence: real prompt construction, `session_search`, `skills_list`, `skill_view`, restart, failure bypass, and matched PAST-Bench execution remain unverified. Snapshot and writeback events can be joined to the existing ledger schema without context content. Compiler execution, adaptive policy, and real memory mutation also remain future stages. No PAST-Bench task definition or hidden grading contract should be changed while these stages are implemented.
