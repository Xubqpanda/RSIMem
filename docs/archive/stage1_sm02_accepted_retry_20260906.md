# Stage 1 SM02 Accepted Retry

Date: 2026-09-06  
Protocol: `native-attribution-repair-v1`  
Family: `SM02_constraint_retention`  
Replicate: `1`

## Execution

This was one manifest-bound `native_static` run using the local runtime and
`--no-judge`. All five tasks completed with complete model usage and passed the
native execution audit. Content-free artifacts are stored under
`outputs/native_attribution/stage1-sm02-retry-20260906/`.

The runtime emitted a `Honcho init failed: No module named
honcho_integration` diagnostic, but the run continued and its manifest-bound
usage, trace, service, and state checks passed. This diagnostic is retained as
runtime evidence and is not labeled as a Memory failure.

## Result

| Measure | Value |
| --- | ---: |
| Accepted runs | 1 |
| Observations | 5 |
| Candidates | 5 |
| Unresolved candidates | 5 |
| Actionable candidates | 0 |
| Panel | semantic |

Corpus: `native-corpus.3b4ad5017ab5d8c9251e8f7c0513dafa86a95acb`  
Report: `native-attribution-report.ed629ca1f06a0e8db1d677e5be1f13f47cdac041`

The Stage 2 decision remains `STOP_NO_ACTIONABLE_SIGNAL`. This accepted run
adds complete semantic process evidence, but no observation uniquely identifies
a repair axis. It is an incremental accepted slice and does not replace the
frozen cross-panel corpus.
