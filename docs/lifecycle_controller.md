# Context Lifecycle Controller

Phase 1 freezes the host-neutral lifecycle contracts at `LIFECYCLE_CONTRACT_SCHEMA_VERSION=1`. The accepted interface and evidence paths are summarized in [`phase1_acceptance_20260827.md`](phase1_acceptance_20260827.md). Phase 2 extensions must reject schema mismatch and preserve the v1 source, provenance, revision, safety, and idempotency semantics rather than silently reinterpret existing plans.

## Purpose

The lifecycle controller is the decision layer between an agent's active context and the typed memory runtime. It evaluates context candidates at explicit lifecycle boundaries and produces one joint signal for context retention and future memory writeback.

The lifecycle controller remains a control-plane component and does not itself mutate Hermes or change default PAST-Bench behavior. Downstream isolated fixtures now consume its validated signals through Mem0-flat ingestion, validation, transactional semantic mutation, restart injection, and content-free operation attribution; live activation remains opt-in and gated.

## Boundaries

```text
host events + context snapshot
        |
        v
EvaluationScheduler  -- when should we evaluate?
        |
        v
ContextEvaluator      -- what should this experience become?
        |
        v
LifecycleController   -- validate and publish the joint signal
        |
        +--> future writeback coordinator
        +--> content-free lifecycle evidence
```

`EvaluationScheduler` is independent of the evaluator implementation. The default production boundary is `JsonLlmContextEvaluator`, which accepts an injected model completion function. A topic-aware local model can implement the same `ContextEvaluator` protocol without changing scheduling, contracts, or memory backends. `ConservativeContextEvaluator` is a deterministic no-op baseline for behavior-equivalence tests.

## Evaluation Frequency

The initial cadence is event-driven rather than every token:

| Trigger | Default | Intended use |
|---|---:|---|
| `task_completed` | on | Evaluate completed work and decide writeback. |
| `session_end` | on | Finalize candidates not handled earlier. |
| `context_pressure` | on when a threshold is configured | Protect the context budget before the host overflows. |
| `turn_interval` | off unless configured | Periodic checkpoints for long-running tasks. |
| `tool_boundary` | off | Optional boundary after tool-heavy phases. |
| `manual` | explicit | Debugging and user-triggered cleanup. |

`min_turns_between_evaluations` prevents duplicate evaluations at the same boundary. Failed evaluator calls do not advance scheduler state, so the host can retry. The current implementation does not yet estimate token counts itself; the host supplies `context_tokens` from its native accounting.

## Joint Signal

Every evaluated segment receives both actions:

```text
context_action: retain | evict
writeback_action: defer | discard | add | update
memory_kind: semantic | episodic | procedural | null
utility_estimate: [0, 1]
confidence: [0, 1]
reason_codes: content-free machine-readable evidence
```

The evaluator must return exactly one signal per input segment. Active, current-turn, unresolved, and open-tool segments cannot be evicted. `add` requires an explicit memory kind without an existing target. For `update`, the evaluator supplies only update hints/mode. The trusted resolver uses the selected `MemoryBackendRegistry` route, verifies update capability and the backend allowlist, loads each candidate artifact through that backend, and binds the stored revision. Fabricated, missing, revisionless, ambiguous, and unsupported candidates are rejected. Compiler version is supplied by the host runtime, not trusted from model output. A retained segment cannot be marked for discard.

The idempotency identity hashes all evidence that can affect compiler-produced memory: completion status and evidence, unresolved state, scope, temporal validity, reusable facts, reusable procedures, and update hints. Stable source IDs, context revision, target identity, policy version, and compiler version remain explicit parts of the key. `plan_id` is derived only from this idempotency identity, so it names the stable logical plan. Evaluation IDs stay in audit provenance and events but do not make an otherwise equivalent retry a new plan or mutation. `safe_to_evict` is a deterministic runtime boolean and is validated strictly.

The signal is a prediction, not a proof of future value. Later retrieval, injection, task, tool, retry, and cost events will provide delayed feedback for the adaptive policy. Raw context remains in the evaluator request only; observer-facing lifecycle evidence contains IDs, counts, actions, and reason codes rather than memory text.

The generic joint-signal fields remain useful for dry-run safety and future hosts, but the current PAST-Bench paper path does not let this evaluator choose semantic versus episodic versus procedural routing or predict ADD versus UPDATE. Phase two freezes Hermes' native route and invocation boundary, sends one ingest/add request to the selected route-specific policy, and records that policy's internal ADD, UPDATE, DELETE, or NONE outcome.

## Current Usage

The package is available under `rsimem.lifecycle`:

- `contracts.py`: context segments, cadence, exit signals, and evaluator protocols.
- `snapshot.py`: stable snapshot identity, provenance, turn state, and deterministic safety state.
- `scheduler.py`: event-to-evaluation cadence decisions.
- `evaluators.py`: injected JSON LLM adapter and deterministic baseline.
- `controller.py`: scheduling, validation, and observer notification.
- `writeback.py`: revisioned plans, backend-bound update resolution, compiler-input-aware idempotency, atomic dry-run receipt reservation, and dry-run coordination.

Atomic reservation prevents two dry-run coordinators from accepting the same
plan concurrently. A separate default-disabled isolated executor now provides
pending/committed receipts, target locking, reread verification, and conservative
crash recovery for semantic mutation. This is a single-host fixture contract,
not a distributed exactly-once claim or live PAST-Bench activation.

The real Hermes prompt builder, `session_search`, `skills_list`, and `skill_view` surfaces pass deterministic native/adapter equivalence checks with observer-only instrumentation, restart-stable identities, and explicit failure policy. The isolated SM01 writeback fixture now additionally connects semantic extraction through future use/outcome and deterministic-first attribution. The next implementation step is the pre-registered static SM01 matched comparison; current evidence still does not establish live-model writeback quality or broad PAST-Bench equivalence.
