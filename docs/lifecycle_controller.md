# Context Lifecycle Controller

## Purpose

The lifecycle controller is the decision layer between an agent's active context and the typed memory runtime. It evaluates context candidates at explicit lifecycle boundaries and produces one joint signal for context retention and future memory writeback.

The first implementation is intentionally a control-plane scaffold. It does not mutate Hermes, invoke a memory compiler, or change PAST-Bench behavior. A later writeback coordinator will consume the validated signals and account for compilation, storage, retrieval, and injection costs.

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

The evaluator must return exactly one signal per input segment. Active, current-turn, unresolved, and open-tool segments cannot be evicted. `add` requires an explicit memory kind without an existing target. For `update`, the evaluator supplies only update hints/mode; a trusted target resolver must search an allowlisted backend, bind one artifact and expected revision, and the coordinator rejects ambiguous or unsupported candidates. Compiler version is supplied by the host runtime, not trusted from model output. A retained segment cannot be marked for discard.

The signal is a prediction, not a proof of future value. Later retrieval, injection, task, tool, retry, and cost events will provide delayed feedback for the adaptive policy. Raw context remains in the evaluator request only; observer-facing lifecycle evidence contains IDs, counts, actions, and reason codes rather than memory text.

## Current Usage

The package is available under `rsimem.lifecycle`:

- `contracts.py`: context segments, cadence, exit signals, and evaluator protocols.
- `snapshot.py`: stable snapshot identity, provenance, turn state, and deterministic safety state.
- `scheduler.py`: event-to-evaluation cadence decisions.
- `evaluators.py`: injected JSON LLM adapter and deterministic baseline.
- `controller.py`: scheduling, validation, and observer notification.
- `writeback.py`: revisioned plans, target-aware idempotency, persistent receipts, and dry-run coordination.

The next implementation step is calling real Hermes prompt, session-search, and skill surfaces under observer-only instrumentation, followed by restart and failure-bypass checks. The current result is a storage-boundary deterministic equivalence baseline, not execution equivalence.
