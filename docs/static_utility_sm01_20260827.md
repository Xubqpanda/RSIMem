# Static Utility SM01 Gate - 2026-08-27

## Scope

This report closes Phase 2H.3 for the frozen semantic policy objective. It
compares two opt-in methods over three rotated, independent, unseeded
replicates of `SM01_preference_adoption`:

- `static-rsimem`: the Phase 2E fixed Mem0-flat baseline.
- `static-utility-rsimem`: the same route, lifecycle boundary, source
  trajectory contract, extraction prompt, operation prompt, and invocation
  schedule, with the frozen utility gate enabled.

The model was `gpt-5.6-luna` at temperature 0.0 through the configured Luna
endpoint. The judge was disabled. Official task scores were produced after
execution and were not available to extraction, utility scoring, mutation, or
retrieval policy code. The provider exposes no controllable seed, so score and
resource differences remain independent-run variation.

Accepted batch:

- Batch: `outputs/static_utility_sm01/hermes_luna/static_utility_sm01_20260827_v1`
- Experiment ID: `f8279c9be05dcac67a67423e4f2337f0860971f4fc4609764aa8b3cd374b4b81`
- RSIMem commit: `6f41add95945abbebace31476d21970a3be51394`
- PAST-Bench commit/tree: `79a5f7ae7aacf17b8ae6ef8fa8a0f22c83c55a9f` / `96c22c4d7f6823b0544ee244651f28e7790978fa`
- Scheduled slots: 6 completed, 0 method failures; 54 unique traces.
- Audit: all 6 reports have `ok=true`; zero accounting issues, credential
  hits, memory-text leaks, unresolved injections, adapter bypasses, or policy
  identity changes.

One earlier attempt in the first static slot is retained with
`failureStage=launcher_timeout`. The launcher was mistakenly invoked with a
one-second outer command timeout before the provider run could complete. Its
isolated directory has no completed result and is excluded from all method
aggregates; the scheduled slot completed as attempt 2.

## Results

Primary score and pass rate exclude the separate reflection trace. Resource
quantities include every unique physical trace and request. Values are means
over three replicates.

| Method | Primary score | Pass rate | Persistence gap | Requests | Input tokens | Output tokens | Retries | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Static RSIMem | 0.3700 | 0.0000 | 0.0000 | 48.0 | 70,564 | 5,492 | 0 | 319.4 s |
| Static utility | 0.4106 | 0.0000 | 0.0933 | 46.7 | 67,319 | 5,654 | 0 | 310.2 s |

Static utility scores were `[0.3700, 0.4917, 0.3700]`; static baseline scores
were `[0.3700, 0.3700, 0.3700]`. The utility mean difference comes entirely
from replicate 2. Three unseeded replicates, zero hard passes, and model-driven
trajectory variation do not support a quality-superiority claim.

## Fixed Invocation And Cost

Both methods produced exactly, per run:

- 6 semantic ingestion executions;
- 7 ingestion-policy model requests;
- 1 planned `ADD` and 5 `NONE` outcomes;
- zero provider retries.

Mean ingestion-only raw cost was:

| Method | Input tokens | Output tokens | Duration | Model requests |
|---|---:|---:|---:|---:|
| Static RSIMem | 16,058 | 961.7 | 37.7 s | 7 |
| Static utility | 14,953.7 | 961.7 | 33.4 s | 7 |

The invocation count is the matched invariant. Token and duration values are
not expected to be identical across unseeded runs because task trajectories
and model outputs differ. Provider-omitted ingestion cache and reasoning
buckets remain unknown rather than being treated as complete zero values. No
provider-price conversion is reported.

## Utility Evidence

Each utility run emitted 6 content-free `static_utility_decisions` events, one
for every ingestion execution. Across all three replicates:

- ingestion/utility execution joins were exact: 18 expected, 18 observed;
- gate digest was always
  `161f616239ee27d0f179791fd38f67a080de50e5e1848ad7975dc64b6f0a1a14`;
- gate version was always `mem0-flat-static-utility-gate-v1`;
- feature schema was always `semantic-static-utility-features-v1`;
- scorer policy was always `semantic-static-utility-policy-v1`;
- each run produced one accepted generation decision and one accepted internal
  operation decision.

The live SM01 trajectories did not exercise a retrieval utility decision.
After the one durable fact was admitted, later boundary extraction calls
returned no durable facts and therefore did not enter related-memory retrieval.
Shared-objective retrieval filtering/ranking remains covered by deterministic
Mem0-flat integration fixtures, not by this live batch.

## Memory Outcome

Static baseline injected no semantic memory in all three runs. Static utility
injected memory three times in replicate 2 and zero times in replicates 1 and
3. Peak semantic memory storage was similar: 170.7 bytes mean for static and
172 bytes for static utility. This is admission-timing variation, not evidence
of an online policy update: the utility artifact identity remained frozen and
no current-run outcome was read by the scorer.

Physical context rewrite remained disabled in both methods, and no saved-token
quantity was reported. This batch demonstrates an auditable frozen static
policy only; it does not demonstrate delayed-feedback learning, adaptive policy
updates, recursive self-improvement, or statistical quality gains.

## Reproduction

Raw evidence and derived `analysis.json` remain under the ignored batch
directory. The content-free analysis is reproducible with:

```bash
RSIMEM_STATIC_METHOD_SET=utility \
RSIMEM_REPLICATES=3 \
RSIMEM_BATCH_ID=static_utility_sm01_20260827_v1 \
scripts/run_luna_static_sm01.sh

PYTHONPATH=src .venv/bin/python -m rsimem.static_utility_analysis \
  outputs/static_utility_sm01/hermes_luna/static_utility_sm01_20260827_v1 \
  --output outputs/static_utility_sm01/hermes_luna/static_utility_sm01_20260827_v1/analysis.json
```

`GPT_LUNA_API_KEY` must be supplied through the process environment. It is not
stored in the launcher, manifest, ledger, report, analysis output, or git
history.
