# Lifecycle Implementation Plan

This document records the implementation path from the lifecycle control plane to an end-to-end RSIMem memory writeback experiment. The semantic compiler now consumes a trusted completed-task snapshot directly. Context evaluation remains an optional observer/control-plane path and is not a prerequisite for extraction.

## Target Architecture

```text
Host adapter
    |
    v
Completed-task snapshot collector
    |
    v
Semantic compilation trigger
    |
    v
Extraction source projection
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
- **Completed-task snapshot collector** converts native messages into stable `ContextSegment` values with session, task, revision, completion, and tool-closure metadata.
- **Semantic compilation trigger** accepts only trusted `task_completed`; failed, active, unresolved, current-turn, and open-tool sources fail closed.
- **Context evaluator and writeback coordinator** remain optional context-management infrastructure. Their keep/evict plans cannot enable, disable, or alter semantic compilation.
- **Extraction source projection** freezes allowed roles, stable source IDs, tool-closure atomicity, metadata allowlisting, ordering, content bounds, and deterministic truncation.
- **Semantic ingestor** consumes exactly that projection and locally reimplements Mem0 flat extraction and internal operation selection over the fixed Hermes semantic route.
- **Memory runtime and backend** validate and apply mutations while preserving each backend's native storage and retrieval behavior.
- **Feedback collector** joins deployment-observable opportunity, explicit use, and outcome evidence for later policy updates. Raw resource usage remains a separate accounting join.

## Evaluation Cadence

Semantic compilation occurs only at trusted task completion. Session end performs cleanup and does not create another semantic plan. Context-pressure, turn-interval, tool-boundary, and evaluator cadence remain separate context-management concerns.

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

- [x] Invoke the semantic ingestor directly at the frozen Hermes completed-task boundary without requiring an eviction evaluator or plan.
- [x] Persist a content-free compilation receipt before model execution so same-source replay and restart do not repeat extraction.
- [x] Keep session-end cleanup from creating a second semantic compilation attempt.
- [x] Bind one canonical extraction source projection and digest to the request, receipt, source operation artifact, and rendered extraction prompt.
- [x] Keep all Hermes routing fixed and do not predict memory form; episodic/procedural policy implementation remains disabled.
- [x] Expose ingest/add externally and treat framework-internal ADD/UPDATE/DELETE/NONE as observable outcomes.
- [x] Validate every internal operation before backend mutation.
- [x] Persist memory before evicting the source context.
- [x] Add revision checks, idempotency, failure bypass, and rollback tests.
- [x] Add pending/committed receipt states and crash recovery around real backend mutations; dry-run reservation alone is not an exactly-once guarantee.
- [x] Account for ingestor model calls, tokens, latency, storage bytes, and rejected mutations.
- [x] Record content-free atomic operations for extraction, related-memory retrieval, internal decision, target resolution, validation, mutation, verification, future retrieval, injection, use, and outcome.
- [x] Bind a host-neutral extraction prompt slot at the real Mem0-flat policy factory and carry its binding fingerprint into the completion request.
- [x] Freeze update, retrieval, route, task boundary, backend, framework, and model profile in extraction-only matched policy manifests.
- [x] Keep plain static extraction free of the legacy utility gate and lifecycle-evaluator dependency.
- [x] Remove the native Hermes memory writer and background review surfaces while preserving semantic prompt reads.
- [x] Identify executor/recovery mutation writers and reject semantic state changes not explained by allowed committed receipts.

Stage 5 is complete for the isolated deterministic path. The extraction-owned feedback and prompt-oriented validation contracts are also complete. A successful live plain-parent smoke remains the Stage 1H close condition; physical context rewrite remains disabled.

### Stage 6: Static RSIMem Evaluation

- [x] Run the historical fixed writeback policy on `SM01_preference_adoption`; retain it as infrastructure evidence.
- [ ] Compare no persistence, native Hermes, and static RSIMem with matched task order, model, budget, and provider randomness controls when available.
- [ ] Report task score, persistence gap, memory retrieval, injected tokens, model calls, tool calls, retries, controller cost, ingestor cost, storage cost, and wall time.
- [ ] Expand to semantic-relevant update families only after the first family is fully attributable.

### Stage 7: Extraction-Prompt Adaptation

- [ ] Collect extraction-owned delayed evidence for opportunity, explicit memory-specific use, outcome, supersession, conflict, unresolved state, and censoring.
- [ ] Estimate delayed extraction utility without hidden grader labels or resource usage in the policy path.
- [ ] Version only the semantic extraction prompt while keeping update, retrieval, route, cadence, backend, and model profile fixed.
- [ ] Use attributed failure subgraphs to update the responsible operation policy instead of broadcasting every task failure to all prior memory operations.
- [ ] Validate proposed policy versions on held-out episodes before activation.
- [ ] Support rejection, rollback, and reproducible replay.

## Safe Execution Order

The semantic writeback path must follow this order:

```text
snapshot
  -> validate completed-task source
  -> reserve compilation receipt
  -> ingest
  -> validate mutation
  -> persist memory
  -> confirm persistence
  -> natural task exit
```

An evaluator or ingestor failure must never cause the source context to be evicted. A backend mutation failure must leave the native context available for bypass or retry. Raw memory content may be present in evaluator and ingestor inputs, but observer-facing ledger events must contain only IDs, actions, counts, sizes, and reason codes.

## First End-To-End Acceptance Case

The first end-to-end case is `SM01_preference_adoption`:

```text
learn episode
  -> capture Hermes context
  -> trusted task_completed snapshot
  -> semantic compilation
  -> persist MEMORY.md
  -> start a fresh eval_near session
  -> retrieve and inject the preference
  -> record task outcome and lifecycle cost
```

Acceptance requires native behavior to remain unchanged, the memory artifact to survive restart, the new session to retrieve it, and every lifecycle stage to be joinable through stable IDs. The ledger must reconcile all model and tool usage without storing memory text.

## Current Boundary

The scheduler, evaluator protocol, lifecycle controller, snapshot/writeback contracts, Hermes execution-surface baselines, transactional semantic mutation, Mem0-flat construction, delayed-feedback foundation, and legacy threshold activation/rollback infrastructure are implemented. Semantic compilation is independent of context eviction and runs once at trusted task completion with restart-safe compilation receipts; physical context rewrite remains disabled. The canonical extraction source projection, host-neutral extraction slot, Mem0-flat binding, composite component identity, plain static parent, native-writer isolation, semantic mutation audit, source/set/fact feedback semantics, prompt-oriented validation, and extraction-specific experiment manifest/analyzer are complete. Stage 1H still requires one successful low-cost live plain-parent smoke. No PAST-Bench task definition, official score, answer key, hidden grading contract, or resource-cost scalar may enter prompt optimization.
