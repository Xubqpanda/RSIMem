# Stage 0C Native Attribution Cleanup Audit

Date: 2026-09-05

## Decision

The `native-attribution-repair-v1` protocol supersedes the earlier
five-condition sensitivity and extraction-first execution lines. Existing
source, tests, configs, reports, and outputs from those lines remain available
for deterministic replay and historical audit, but they are not current
experiment entrypoints and cannot contribute quality observations to the new
attribution corpus.

No tracked historical evidence is deleted in this pass. The cleanup is a
research-boundary change: new provider execution must be prepared from
`NativeAttributionRunManifest` and must use `native_static` only until Stage 0D
acceptance is complete.

## Asset Classification

| Asset group | Disposition | Current use |
| --- | --- | --- |
| `src/rsimem/sensitivity*.py`, `src/rsimem/past_sensitivity_*.py` | `HISTORICAL_REPLAY_ONLY` | Reproduce and audit the superseded five-condition pilots; do not launch new main-line runs. |
| `configs/sensitivity/` | `HISTORICAL_REPLAY_ONLY` | Preserve public-source oracle seed identity and negative evidence; not an Analysis 1 repair registry. |
| `docs/sensitivity_*` and ignored `outputs/sensitivity/` | `EVIDENCE_KEEP` | Historical execution and infrastructure evidence only. |
| Extraction experiment/preparation/matched modules and configs | `LEGACY_METHOD_FIXTURE` | Retain tests and pure-process contract evidence; no extraction N+1 or held-out batch is authorized. |
| Adaptive preparation/activation/analysis modules | `DEFERRED_INFRASTRUCTURE` | Retain state, validation, and accounting primitives; not a current policy or quality claim. |
| Lifecycle, provenance, revision, idempotency, CAS, rollback, evidence-plane, and usage modules | `GENERALIZE_AND_KEEP` | Shared contracts for native attribution, repair, and later feedback experiments. |
| `native_attribution_protocol.py`, `native_attribution_run.py`, `native_attribution_launcher.py`, `native_anchor.py`, `native_scheduler.py` | `CURRENT_MAIN_LINE` | Stage 0 isolation and the gated native failure-attribution path. |

The console scripts for extraction and adaptive tooling remain installed
because tests and replay workflows depend on them. Their presence is not an
authorization signal. The current provider-facing entrypoint must be manifest
bound through `prepare_native_attribution_launch()`; the old sensitivity pilot
launcher must not be used for new runs.

## Preserved Boundaries

- Dataset, grader, task fixtures, and historical output formats are unchanged.
- Grader fields, hidden answers, future evaluation data, and official scores
  remain outside updater and method views.
- Costs and token usage remain reporting fields only.
- Historical `unresolved`, provider failure, and incomplete-usage observations
  remain negative or infrastructure evidence; they are not relabeled.
- The immutable native anchor and isolated repair branches remain the only
  authorized basis for future one-axis repair experiments.

## Stage 0D Gate

Service fixture isolation passed sequential and concurrent contract tests. The
real manifest-bound SM01 smoke then reconstructed and verified trace identity,
state/home/session/artifact paths and digests, service fixture identity and
ports, provider/model identity, and complete request usage. The full serial
verification suite also passed. Details are in
[`stage0d_native_smoke_20260905.md`](stage0d_native_smoke_20260905.md); Stage 0D
is complete.

## Provider Status

A bounded connectivity probe on 2026-09-05 returned HTTP 200 from the primary
OpenAI-compatible endpoint with model `gpt-5.6-luna`, the expected response,
and a usage object. This is connectivity evidence only and does not enter a
benchmark denominator or reopen a superseded experiment.
