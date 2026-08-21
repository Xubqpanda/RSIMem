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

The evaluator must return exactly one signal per input segment. Active segments cannot be evicted. `add` and `update` require an explicit memory kind, while a retained segment cannot be marked for discard. This makes the relationship between eviction and memory writeback explicit before any backend mutation is attempted.

The signal is a prediction, not a proof of future value. Later retrieval, injection, task, tool, retry, and cost events will provide delayed feedback for the adaptive policy. Raw context remains in the evaluator request only; observer-facing lifecycle evidence contains IDs, counts, actions, and reason codes rather than memory text.

## Current Usage

The package is available under `rsimem.lifecycle`:

- `contracts.py`: context snapshots, cadence, signals, and evaluator protocols.
- `scheduler.py`: event-to-evaluation cadence decisions.
- `evaluators.py`: injected JSON LLM adapter and deterministic baseline.
- `controller.py`: scheduling, validation, and observer notification.

The next implementation step is an opt-in Hermes integration that constructs a context snapshot at a task-aligned exit boundary and forwards the controller result to a validated compiler/writeback coordinator. Native Hermes behavior remains the control until deterministic equivalence tests pass.
