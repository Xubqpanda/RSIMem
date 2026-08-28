# Experiment Plan

## Research Question

Can delayed deployment feedback improve a semantic memory extraction prompt
from version N to N+1 and thereby improve future task behavior?

The primary claim is not that LightRSI creates fewer memories.

The first claim is that RSIMem improves extraction behavior through delayed
observable opportunity, use, and outcome feedback while the rest of the memory
pipeline remains frozen. Resource usage is reported separately.

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
| Adaptive RSIMem | Test delayed-feedback updates to the semantic extraction prompt only. |

The first implementation stops at these five variants.

Additional component ablations are added only after the end-to-end result is stable.

The first production learner updates only the deployable semantic extraction
prompt body. The update prompt, retrieval configuration, backend, route,
invocation boundary, model profile, and writer remain frozen. The existing
retrieval-threshold learner is a legacy infrastructure experiment and cannot
produce an extraction runtime artifact.

## Metrics

Task quality uses the native PAST-Bench task score, persistence gap, and mechanism evidence.

Raw resource accounting records model input and output tokens, cache usage, model calls, tool calls, retries, wall time, memory operations, stored records and bytes, retrieved records, injected tokens, and controller overhead.

No heterogeneous lifecycle-cost scalar enters learning or activation. Reports
retain the raw resource vector and may present each resource dimension beside
quality without combining unlike units.

Raw resource quantities are retained separately from provider prices so that monetary results can be recomputed.

The former five-method threshold report is legacy infrastructure. The
extraction-specific report will retain every replicate value, compute matched
static-to-adaptive deltas, and keep configured budgets separate from realized
request, token, storage, injection, and timing vectors.

Claim eligibility is fail-closed. One audited SM01 batch can provide implementation evidence for the fixed semantic route, unified objective, and operation-attributed update, but it cannot by itself establish statistical quality superiority, a second recursive policy iteration, or cross-family generalization.

## Phase 1C Read-Path Analysis Protocol

This protocol is frozen before the full-projection live matched results are observed. One analysis unit is a successful `(replicate, execution mode)` run over the complete ordered SM01 family and its paired no-persistence control. At least three successful units are required per mode. Provider-failed attempts remain in the manifest and failure summary but do not enter clean-run quality or resource aggregates.

The integrity gate is exact rather than statistical: every successful run must pass `rsimem-audit`; physical request reconciliation, same-call adapter/native projection mismatches, unexplained task/order/budget/initial-state differences, silent bypass, unresolved injection, and missing runtime evidence must each equal zero. Adapter runs must contain at least one native-shadow projection check. Any violation blocks Phase 1C regardless of quality scores.

For the small unseeded sample, the report presents every raw replicate value plus the median and full range by mode. It does not use a significance test, confidence interval, post-hoc tolerance, or an equivalence claim based on failure to reject a null hypothesis. Paired mode differences are shown within each rotated replicate for:

- with-persistence and without-persistence task score, pass rate, and persistence gap;
- model request count, input/output/cache-read/cache-write/reasoning tokens, retries, and model duration;
- tool calls, stored bytes, retrieved records, injected characters, ledger event count, and wall time.

Episode-level attribution follows fixed evidence rules. A changed task manifest, episode order, budget, initial state, or adapter/native return value at the same read call is configuration or adapter divergence. A provider transport/status failure is provider failure. When those invariants match and every adapter call returns exactly the native shadow value, differences across independently sampled runs in model outputs, later memory state, tool choices, tokens, or timing are reported as unseeded model/provider variation, not adapter causation. Ambiguous evidence remains unexplained and fails the stage gate.

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

Aggregate extraction-owned feedback from later exposure opportunity, explicit
memory-specific use, supersession/conflict, and observable task outcomes.

Use bounded, operation-attributed feedback to propose a versioned semantic
extraction prompt N+1 while keeping update, retrieval, route, invocation,
backend, and model components fixed.

Validate, accept, or roll back each proposal without using hidden benchmark grading information.

The stage passes when policy versions are reproducible and adaptive LightRSI can be compared against the static policy.

Threshold-oriented learner, validation, activation, rollback, and runtime
binding gates are retained as legacy infrastructure. Extraction-owned feedback,
prompt-oriented validation, and formal manifest/analyzer contracts are complete.
Deployable extraction artifacts, optimizer-driven runtime binding, and matched
live comparison remain Phase 2 work.

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
