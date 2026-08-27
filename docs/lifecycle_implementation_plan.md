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
Semantic ingestor
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
- **Context evaluator** estimates completion, future reuse, freshness, and unresolved state from a snapshot.
- **Lifecycle policy** supplies evidence to a fixed Hermes memory route; phase two does not learn route selection or invocation timing.
- **Writeback coordinator** turns an accepted decision into a validated, provenance-linked writeback plan.
- **Semantic ingestor** locally reimplements Mem0 flat extraction and internal operation selection over the fixed Hermes semantic route.
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
- [x] Call Hermes' real memory prompt construction, `session_search`, `skills_list`, and `skill_view` paths in a deterministic execution-surface fixture.
- [x] Confirm deterministic restart persistence, artifact identity, and explicit fail-closed or native-bypass behavior.
- [x] Run all three modes through a deterministic PAST-Bench Hermes adapter-loop fixture without external model calls.
- [x] Complete one live-model infrastructure replicate on the earlier hybrid adapter path and retain its clean and failed accounting evidence.
- [ ] Run at least three order-rotated live-model replicates on the completed semantic, episodic, and procedural projection path.
- [x] Keep the direct Hermes path unchanged as the control.

### Stage 5: Validated Semantic Memory Writeback

- [ ] Invoke the semantic ingestor only at the frozen Hermes semantic boundary.
- [ ] Keep all Hermes routing fixed and do not predict memory form; episodic/procedural policy implementation remains disabled.
- [ ] Expose ingest/add externally and treat framework-internal ADD/UPDATE/DELETE/NONE as observable outcomes.
- [ ] Validate every internal operation before backend mutation.
- [ ] Persist memory before evicting the source context.
- [ ] Add revision checks, idempotency, failure bypass, and rollback tests.
- [ ] Add pending/committed receipt states and crash recovery around real backend mutations; dry-run reservation alone is not an exactly-once guarantee.
- [ ] Account for ingestor model calls, tokens, latency, storage bytes, and rejected mutations.
- [ ] Record content-free atomic operations for extraction, related-memory retrieval, internal decision, target resolution, validation, mutation, verification, future retrieval, injection, use, and outcome.

### Stage 6: Static RSIMem Evaluation

- [ ] Run the fixed writeback policy on `SM01_preference_adoption` first.
- [ ] Compare no persistence, native Hermes, and static RSIMem with matched task order, model, budget, and provider randomness controls when available.
- [ ] Report task score, persistence gap, memory retrieval, injected tokens, model calls, tool calls, retries, controller cost, ingestor cost, storage cost, and wall time.
- [ ] Expand to semantic-relevant update families only after the first family is fully attributable.

### Stage 7: Adaptive Policy

- [ ] Collect delayed evidence for retrieval, injection, actual use, task outcome, tool behavior, retries, supersession, non-use, and lifecycle cost.
- [ ] Estimate realized future utility without using hidden grader labels in the policy path.
- [ ] Version semantic extraction/update/consolidation and retrieval policies while keeping route and cadence fixed.
- [ ] Use attributed failure subgraphs to update the responsible operation policy instead of broadcasting every task failure to all prior memory operations.
- [ ] Validate proposed policy versions on held-out episodes before activation.
- [ ] Support rejection, rollback, and reproducible replay.

## Safe Execution Order

The writeback path must follow this order:

```text
snapshot
  -> evaluate
  -> validate
  -> ingest
  -> validate mutation
  -> persist memory
  -> confirm persistence
  -> evict context
```

An evaluator or ingestor failure must never cause the source context to be evicted. A backend mutation failure must leave the native context available for bypass or retry. Raw memory content may be present in evaluator and ingestor inputs, but observer-facing ledger events must contain only IDs, actions, counts, sizes, and reason codes.

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

The scheduler, evaluator protocol, JSON LLM evaluator, conservative evaluator, lifecycle controller, snapshot contracts, deterministic Hermes fixture, and dry-run writeback coordinator are implemented in `rsimem.lifecycle`. Storage-boundary, deterministic Hermes execution-surface, and deterministic PAST-Bench adapter-loop baselines are complete. Versioned ingestion, validation, and a default-disabled isolated semantic transaction/recovery fixture are also complete; the live host does not invoke them yet. One live unseeded infrastructure replicate is recorded in [`matched_20260827.md`](matched_20260827.md), but it predates full procedural and episodic projection and used a fixed mode order. It does not establish live-model equivalence. Mem0-flat model construction, live mutation activation, and adaptive policy remain future stages. No PAST-Bench task definition or hidden grading contract should be changed while these stages are implemented.
