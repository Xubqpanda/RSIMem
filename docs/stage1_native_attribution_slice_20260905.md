# Stage 1 Native Attribution Slice

Date: 2026-09-05  
Protocol: `native-attribution-repair-v1`  
Corpus: `outputs/native_attribution/stage1-cross-panel-slice-20260905/attribution_corpus_v2.json`

## Scope

This is the first bounded native-static attribution slice after the Stage 0
isolation gate. It includes accepted SM02 and PC01 runs. EP01 and PG01 attempts
were retained as infrastructure evidence but excluded because at least one
model request had incomplete usage after provider HTTP 503 responses.

## Result

| Measure | Value |
| --- | ---: |
| Accepted runs | 2 |
| Accepted observations | 10 |
| Attribution candidates | 10 |
| Unresolved candidates | 10 |
| Actionable candidates | 0 |
| Evidence completeness | 0.5091 |
| Repair cases authorized | 0 |

Decision: `STOP_NO_ACTIONABLE_SIGNAL`.

The deterministic summary is produced by
`rsimem.native_attribution_report.build_attribution_report()`. Its report ID is
derived from the frozen corpus identity and its fields are limited to counts,
ratios, lifecycle labels, memory kinds, families, and exclusion reasons.
The frozen corpus reload and report reconstruction were verified directly from
disk; report ID: `native-attribution-report.849141cb52594f52f952d18cf292b03626773ebb`.
The report also exposes a content-free case index and evidence-reference map
for all ten unresolved candidates, so reviewer work can be performed against
stable IDs without copying task text.

The deterministic contracts, event identity, state digests, retrieval metadata,
tool closure references, and content-free corpus are valid for this slice. The
observed evidence does not uniquely distinguish a repair axis. In particular,
no candidate is converted into formation, persistence, retrieval, or
application failure solely from the task result.

## Boundary

This report is an attribution and infrastructure report, not a quality result.
It does not authorize Analysis 1 repair, N+1 execution, adaptive policy, or
feedback-granularity experiments. Provider probes are kept separate from the
benchmark denominator; the primary endpoint still exhibits intermittent HTTP
503 responses and usage-incomplete attempts.

The next valid transition requires either a new accepted native slice with
complete process evidence and a unique repair axis, or an explicit protocol
review. Until then, unresolved and infrastructure-excluded cases remain
negative evidence and are not used to train an updater.
