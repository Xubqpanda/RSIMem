# Experiment Plan

## Research Question

Can an agent recursively improve its future behavior through memory while ensuring that retained experience repays its full lifecycle cost?

The primary claim is not that LightRSI creates fewer memories.

The claim is that LightRSI improves the cost--quality frontier of existing memory backends through delayed retrieval and execution feedback.

## Controlled Setting

The initial experiments use PAST-Bench with a fixed agent, model, tool environment, task order, and execution budget.

The first smoke family is `memory_ability/SM01_preference_adoption`.

Hermes native memory is the first backend because PAST-Bench already provides its runtime and persistence controls.

The smoke run uses one seed, while reported experiments use at least three independent seeds.

PAST-Bench expectations, answer keys, and grading rubrics remain evaluation-only and never enter LightRSI's policy updates.

## Primary Variants

| Variant | Purpose |
|---|---|
| No persistence | Measure performance without cross-session memory. |
| Native backend | Establish the original memory backend baseline. |
| Native backend + ledger | Verify that instrumentation is behaviorally neutral. |
| Static LightRSI | Test context-exit coordination and memory admission without recursive updates. |
| Adaptive LightRSI | Test the full delayed-feedback policy update loop. |

The first implementation stops at these five variants.

Additional component ablations are added only after the end-to-end result is stable.

## Metrics

Task quality uses the native PAST-Bench task score, persistence gap, and mechanism evidence.

Raw resource accounting records model input and output tokens, cache usage, model calls, tool calls, retries, wall time, memory operations, stored records and bytes, retrieved records, injected tokens, and controller overhead.

Derived metrics include total cost, cost per successful episode, future utility per lifecycle cost, and the cost--quality frontier.

Raw resource quantities are retained separately from provider prices so that monetary results can be recomputed.

## Rollout

### Stage 0: Reproduce the Baseline

Run PAST-Bench's no-persistence and native-persistence variants on `SM01_preference_adoption` without RSIMem changes.

Record the benchmark commit, agent commit, model profile, judge profile, task manifest, seed, and runtime configuration.

The stage passes when both variants complete and their official trace and grading outputs can be reproduced.

### Stage 1: Add the Ledger

Instrument Hermes memory writes, reads, injections, model calls, tools, retries, latency, and storage without changing decisions.

Compare native persistence with native persistence plus ledger under matched settings.

The stage passes when all required events are present and instrumentation does not materially change task behavior or resource usage.

### Stage 2: Add Static LightRSI

At task-aligned context exits, use one fixed policy to choose discard, add, or update before invoking the native writer.

Keep backend-native storage and retrieval unchanged.

The stage passes when every candidate can be linked to its memory operation and later retrieval through stable experiment identifiers.

### Stage 3: Close the Recursive Loop

Aggregate deployment-observable feedback from later retrieval, injection, tool execution, retries, completion, and lifecycle cost.

Use this feedback to propose versioned updates to the context-exit and memory-writing policies.

Validate, accept, or roll back each proposal without using hidden benchmark grading information.

The stage passes when policy versions are reproducible and adaptive LightRSI can be compared against the static policy.

### Stage 4: Expand Task Coverage

Run all five memory-ability families first.

Then add update-ability families, followed by procedural-reuse and proactive-information-gathering families.

Only after these stages are stable should the full 26-family evaluation begin.

### Stage 5: Expand Memory Backends

Add Mem0 first, then LangMem, using MemBase's layer implementations behind an interactive PAST-Bench adapter.

Add MemOS later because its current MemBase layer does not support a unified in-place update operation.

For every backend $B$, the primary comparison remains native $B$ versus $B$ with LightRSI under matched settings.

## Repository Boundaries

RSIMem contains PAST-Bench integration, memory-backend adapters, experiment configurations, launchers, event collection, statistical analysis, and table or figure generation.

RSIMem does not vendor PAST-Bench, copy hidden evaluation contracts, or fork backend implementations.

Reusable production components belong in the LightRSI framework repository, while RSIMem pins their exact commit for reproducibility.

The local MemBase worktree currently contains uncommitted changes, so RSIMem must pin a clean commit or a dedicated branch before using it in reported experiments.
