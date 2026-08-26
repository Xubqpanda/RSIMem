# Experiment Plan

## Research Question

Can an agent recursively improve its future behavior through memory while ensuring that retained experience repays its full lifecycle cost?

The primary claim is not that LightRSI creates fewer memories.

The claim is that LightRSI improves the cost--quality frontier of existing memory backends through delayed retrieval and execution feedback.

## Controlled Setting

The initial experiments use PAST-Bench with a fixed agent, model, tool environment, task order, and execution budget.

The first smoke family is `memory_ability/SM01_preference_adoption`.

Hermes native memory is the first backend because PAST-Bench already provides its runtime and persistence controls.

The current paper scope fixes Hermes native memory as the only backend and PAST-Bench as the only benchmark. MemBase's runtime and evaluation pipeline and the external backends integrated through it are out of scope, although pinned memory-layer source may be used as an attributed algorithm reference for local reimplementation.

The initial infrastructure smoke uses one run. The current provider does not expose a verifiable seed, so reported experiments use at least three independently sampled, explicitly unseeded replicates and rotate execution order.

PAST-Bench expectations, answer keys, and grading rubrics remain evaluation-only and never enter LightRSI's policy updates.

## Primary Variants

| Variant | Purpose |
|---|---|
| No persistence | Measure performance without cross-session memory. |
| Native backend | Establish the original memory backend baseline. |
| Native backend + ledger | Verify that instrumentation is behaviorally neutral. |
| Static LightRSI | Test a fixed semantic construction/update/retrieval policy without recursive updates. |
| Adaptive LightRSI | Test operation-attributed delayed-feedback updates to the same fixed semantic policy. |

The first implementation stops at these five variants.

Additional component ablations are added only after the end-to-end result is stable.

## Metrics

Task quality uses the native PAST-Bench task score, persistence gap, and mechanism evidence.

Raw resource accounting records model input and output tokens, cache usage, model calls, tool calls, retries, wall time, memory operations, stored records and bytes, retrieved records, injected tokens, and controller overhead.

Derived metrics include total cost, cost per successful episode, future utility per lifecycle cost, and the cost--quality frontier.

Raw resource quantities are retained separately from provider prices so that monetary results can be recomputed.

## Phase 1C Read-Path Analysis Protocol

This protocol is frozen before the full-projection live matched results are observed. One analysis unit is a successful `(replicate, execution mode)` run over the complete ordered SM01 family and its paired no-persistence control. At least three successful units are required per mode. Provider-failed attempts remain in the manifest and failure summary but do not enter clean-run quality or resource aggregates.

The integrity gate is exact rather than statistical: every successful run must pass `rsimem-audit`; physical request reconciliation, model-visible adapter input divergence, unexplained task/order/budget/state differences, silent bypass, and missing runtime evidence must each equal zero. Any non-zero value blocks Phase 1C regardless of quality scores.

For the small unseeded sample, the report presents every raw replicate value plus the median and full range by mode. It does not use a significance test, confidence interval, post-hoc tolerance, or an equivalence claim based on failure to reject a null hypothesis. Paired mode differences are shown within each rotated replicate for:

- with-persistence and without-persistence task score, pass rate, and persistence gap;
- model request count, input/output/cache-read/cache-write/reasoning tokens, retries, and model duration;
- tool calls, stored bytes, retrieved records, injected characters, ledger event count, and wall time.

Episode-level attribution follows fixed evidence rules. A changed task manifest, episode order, budget, initial state digest, or model-visible input digest is configuration or adapter divergence. A provider transport/status failure is provider failure. When those invariants match and model-visible inputs are identical, differences in model outputs, tool choices, tokens, or timing are reported as unseeded model/provider variation, not adapter causation. Ambiguous evidence remains unexplained and fails the stage gate.

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

At the fixed Hermes semantic task/session boundary, expose one external ingestion operation and let the locally reimplemented Mem0-style policy decide ADD, UPDATE, DELETE, or NONE internally.

Keep routing, invocation frequency, backend-native storage, and model-visible retrieval surfaces unchanged. The first semantic implementation locally reimplements the flat-memory construction path and prompts from Mem0 as vendored in MemBase, without importing MemBase's datasets, runners, or evaluation code.

The stage passes when every semantic ingestion can be linked through an atomic operation graph from source and extraction to internal operation, mutation, later retrieval, use, and outcome.

### Stage 3: Close the Recursive Loop

Aggregate deployment-observable feedback from later retrieval, injection, tool execution, retries, completion, and lifecycle cost.

Use operation-attributed feedback to propose versioned updates to semantic extraction, conflict-resolution, consolidation, and retrieval policies while keeping route selection and invocation boundaries fixed.

Validate, accept, or roll back each proposal without using hidden benchmark grading information.

The stage passes when policy versions are reproducible and adaptive LightRSI can be compared against the static policy.

### Stage 4: Expand Task Coverage

Run semantic-relevant memory-ability families first.

Then add semantic-relevant update-ability families. Episodic and procedural families remain out of the active implementation sequence until their method-selection gates pass.

Only after these stages are stable should the need for broader PAST-Bench coverage be decided from the supported claim; full 26-family coverage is not an automatic requirement.

### Stage 5: Deferred Typed-Memory Expansion

After the semantic claim is established, separately decide whether an episodic method and a trajectory-to-skill method are mature enough for matched PAST-Bench evaluation. Their implementation is not required to complete the semantic-first paper path.

Do not add MemBase, Mem0, LangMem, or Graphiti as runtime external backends, and do not add another host or benchmark. The locally reimplemented Mem0 flat construction policy is a controlled algorithm baseline over Hermes storage, not a Mem0 backend integration.

The primary comparison remains no persistence, native Hermes, static LightRSI over Hermes, and adaptive LightRSI over Hermes under matched settings.

## Repository Boundaries

RSIMem contains benchmark sources under `benchmarks/`, benchmark integration, memory-backend adapters, experiment configurations, launchers, event collection, statistical analysis, and table or figure generation. Each vendored benchmark retains its upstream license, notice, and provenance metadata.

RSIMem may modify benchmark runtime and telemetry code needed for reproducible experiments, but it does not change task semantics, hidden evaluation contracts, answer keys, or grading criteria.

Reusable production components belong in the LightRSI framework repository, while RSIMem pins their exact commit for reproducibility.
