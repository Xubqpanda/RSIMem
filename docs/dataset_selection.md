# Dataset Selection

## Decision

The first RSIMem evaluation of LightRSI uses PAST-Bench only.

LoCoMo and LongMemEval are useful long-horizon conversational QA benchmarks, but they do not exercise the execution effects central to LightRSI: tool calls, retries, state mutation, procedural reuse, failed tasks, and cross-session agent improvement. Strong results on those datasets would primarily support a long-conversation retrieval claim rather than the intended memory-mediated recursive self-improvement claim.

RealMem and other memory QA environments are also outside the initial scope. They can be reconsidered as diagnostic evaluations only if a later experiment needs an isolated memory-quality test.

## Why PAST-Bench

PAST-Bench evaluates ordered task-family sequences in fresh sessions. Earlier episodes create information or experience that later episodes can reuse. Its protocol provides:

- 26 task families and 204 ordered episodes;
- memory, procedural reuse, proactive information gathering, and update abilities;
- matched persistence-on and persistence-off controls;
- real tools and sandbox execution;
- task scores and mechanism-level evidence;
- model tokens, wall-clock time, and execution traces.

This makes it possible to test whether a memory policy actually lowers global agent cost, or merely shifts cost into additional model calls, tools, retries, latency, or failed episodes.

## Source

PAST-Bench is maintained externally under Apache-2.0:

```text
Repository: https://github.com/Gen-Verse/PAST-Bench
Paper:     arXiv:2608.04003
Local:     /mnt/20t/xubuqiang/Study/PAST-Bench
```

The runtime requires Python 3.11+, Docker, and an LLM API profile. RSIMem should integrate through PAST-Bench's public task/runtime interfaces rather than vendor the benchmark source.

## Initial Protocol

The primary experiments fix the agent framework and base model, then vary only persistence and memory coordination:

```text
No persistence
Native persistence
Native persistence + ledger only
Eviction-aware admission
Retrieval-feedback invocation
Full LightRSI
```

Hermes is the initial agent because PAST-Bench includes an in-process adapter and explicit persistence controls. Hermes+ and a second agent framework are follow-up generalization tests, not prerequisites for the first implementation.

## Rollout Order

```text
1. One memory-ability family as an end-to-end smoke test
2. All memory-ability families
3. Update-ability families to test stale/conflicting memory
4. Procedural-reuse families
5. Proactive-information-gathering families
6. Full 26-family paired evaluation
```

Every stage must preserve matched task, model, tool environment, session order, and execution budget across policy variants.

## Scope Boundary

PAST-Bench is the primary benchmark, not part of RSIMem. RSIMem owns:

- context-eviction and retrieval-feedback event contracts;
- backend and host adapters;
- lifecycle accounting;
- coordination policies;
- experiment configuration and statistical analysis.

PAST-Bench owns task definitions, sandbox services, agent runtimes, grading, and persistence-control semantics.

Using a single benchmark initially keeps the engineering effort aligned with the research claim. A second interactive agent benchmark should be added only after the PAST-Bench pipeline and main empirical findings are stable.
