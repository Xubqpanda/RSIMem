# Stage 1 EP01 Accepted Retry

Date: 2026-09-06  
Protocol: `native-attribution-repair-v1`  
Family: `EP01_prior_case_recall`  
Replicate: `1`  

## Execution

This was one manifest-bound `native_static` run using the local runtime and
`--no-judge`. The provider returned complete model usage for all five tasks.
The run passed the native execution audit and was accepted by the attribution
batch assembler. Content-free artifacts are stored under
`outputs/native_attribution/stage1-ep01-retry-20260906/`.

## Result

| Measure | Value |
| --- | ---: |
| Accepted runs | 1 |
| Observations | 5 |
| Candidates | 5 |
| Unresolved candidates | 5 |
| Actionable candidates | 0 |
| Panel | episodic |

Corpus: `native-corpus.f3c9a7392f3f9b21ce6de9599ca3114899cd6442`  
Report: `native-attribution-report.c0a8fdd26a5671b9dbdcfcfe463e30c1facfa765`

The Stage 2 decision remains `STOP_NO_ACTIONABLE_SIGNAL`. The accepted run
adds complete episodic process evidence, but its observations do not uniquely
identify a repair axis. No score, grader output, or final response text is
used as an attribution label.

This is an incremental accepted slice and does not replace the previously
frozen cross-panel corpus.
