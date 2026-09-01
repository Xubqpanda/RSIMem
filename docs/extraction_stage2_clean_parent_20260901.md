# Stage 2 clean-parent process-signal reruns - 2026-09-01

This report records the fresh provider-backed, fixed-parent reruns required by
the implementation handoff checklist after the provider probe recovered.  The
runs are process-signal feasibility evidence only.  They are not candidate,
held-out, activation, matched-effect, or benchmark-quality experiments.

No grader output, answer key, hidden expectation, official score, or resource
cost is used as a process label.  Token, request, latency, storage, and cache
fields are retained as raw accounting only.

## Frozen protocol and provenance

Both batches use the same parent execution profile:

- Hermes with `native+ledger` and static semantic extraction;
- `with_persistence`, lifecycle evaluator disabled, native memory writer
  disabled, background review disabled, and judge disabled;
- model `gpt-5.6-luna` at the configured OpenAI-compatible endpoint;
- three replicates, one fixed parent method per replicate;
- batch-time completion probe before the first task;
- task-specific train split, frozen protocol, and `logical_case_v1` deduplication.

| batch | family | split | experiment ID | RSIMem commit | PAST-Bench commit | status |
| --- | --- | --- | --- | --- | --- | --- |
| `s2-sm02-clean-parent-20260901-v1` | `SM02_constraint_retention` | train | `3ed3a33ecb3c3cbdafaf6560bf26f792eb947b6502d0ffbd88f85f77eda3d019` | `a15dbf87688501e8e70f881a95e1a8a4f4a2be60` | `235fd8e26ed2752b11a568d6d2b28cbfec0ef6a6` | 3/3 completed, audit clean |
| `s2-sm05-clean-parent-20260901-v1` | `SM05_weak_trigger_preference_adoption` | train | `e05120b6e5e06fa141db76e873c7c32d0ab25b4b50a160323da0605b46e3dd05` | `a15dbf87688501e8e70f881a95e1a8a4f4a2be60` | `235fd8e26ed2752b11a568d6d2b28cbfec0ef6a6` | 3/3 completed, audit clean |

The authoritative manifests and raw artifacts remain in the ignored
`outputs/extraction_feedback/hermes_luna/<batch>/` directories.  Both manifests
record a passing provider probe, clean trees, frozen split identity, and
append-only attempt history.

## Raw runtime census

| batch | replicate | process events | model requests | input tokens | output tokens | cache-read tokens | peak stored bytes | audit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SM02 | r01 | 125 | 35 | 84,810 | 4,164 | 13,056 | 98,533 | clean |
| SM02 | r02 | 146 | 41 | 102,962 | 4,671 | 0 | 98,533 | clean |
| SM02 | r03 | 142 | 41 | 96,205 | 4,867 | 66,048 | 106,725 | clean |
| SM05 | r01 | 193 | 62 | 148,130 | 5,293 | 58,880 | 180,449 | clean |
| SM05 | r02 | 191 | 58 | 127,775 | 5,557 | 68,096 | 160,050 | clean |
| SM05 | r03 | 181 | 55 | 91,981 | 5,102 | 73,984 | 131,290 | clean |

All per-replicate audits report `ok=true`, no audit issues, no projection
mismatch, and no native bypass.  Reasoning-token and ingestion-cache buckets
that are not supplied by the provider remain unknown where reported by the
analysis artifact.  These values are not policy signals.

## Pure-process and logical-case results

| batch | pure-process events | optimizer examples | logical cases | physical observations | case status | replicate consistency |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| SM02 | 422 | 21 (all `unresolved`) | 25 | 48 | 3 `observable_only`, 3 `diagnostic_only`, 19 `censored` | 1.0 |
| SM05 | 565 | 33 (all `unresolved`) | 31 | 63 | 2 `observable_only`, 7 `diagnostic_only`, 22 `censored` | 1.0 |

Both optimizer corpus artifacts have `evidence_plane=pure_process`,
`evidence_source=runtime_observation`, no benchmark/grader fields,
`process_signal_gate=no_signal`, zero optimization cases, and no hypothesis
digest.  The separate benchmark-audit summaries retain their own labels for
instrumentation diagnostics only: SM02 has 24 audit-only unresolved records;
SM05 has 12 audit-only missed and 24 audit-only unresolved records.  None of
these audit labels enters the pure optimizer corpus.

The resulting pure-process decision is:

```text
SM02: STOP_NO_SIGNAL
SM05: STOP_NO_SIGNAL
```

The corpora can reconstruct source, extraction, persistence, retrieval,
exposure, tool call/result, and observable task boundaries, but no case has a
complete extraction-owned chain that is both attributable and abstractable:

```text
durable source
-> executed extraction
-> persisted artifact or complete set binding
-> retrieval and exposure
-> memory-specific use
-> exact observable outcome
-> benchmark-independent policy edit
```

No Extraction N+1, held-out validation, ACTIVE pointer, adaptive effect, or
joint six-layer intervention is authorized by these reruns.  The two batches
close the finite Stage 2 family attempt with process observability established
but extraction attribution signal still insufficient.

## Verification

The code revision recorded in both manifests passed the deterministic baseline:

```text
RSIMem: 1101 passed
PAST-Bench: 401 passed, 2 skipped (run from benchmarks/past-bench)
compileall: passed
pip check: passed
bash -n scripts/*.sh: passed
tracked-secret scan: passed
git diff --check: passed
```

