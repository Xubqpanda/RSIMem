# Stage 2 production pure-process reruns — 2026-08-31

This note records the first production runs after the automatic pure-process
source/feedback wiring and the PAST-Bench fact matcher were installed.  It is
an infrastructure and signal audit, not a candidate effect experiment.  The
runs use `native+ledger`, static extraction, judge disabled, three replicates,
and the configured OpenAI-compatible provider with model `gpt-5.6-luna`.

No grader output, answer key, hidden expectation, official score, or resource
cost is used as a process label.  Token/request counts below are raw resource
accounting only.

## Provenance

| batch | family | split | RSIMem / PAST-Bench revision | status |
| --- | --- | --- | --- | --- |
| `sm02-pure-production-20260831-v3` | `SM02_constraint_retention` | train | `e554274` / `e554274` | 3/3 completed, all audits clean |
| `sm05-pure-production-20260831-v2` | `SM05_weak_trigger_preference_adoption` | train | `e554274` / `e554274` | 3/3 completed, all audits clean |

The authoritative manifests and artifacts are kept outside the repository in
the clean worktree under
`outputs/extraction_feedback/hermes_luna/<batch>/`.  Each manifest records a
clean tree, frozen protocol, provider probe, and append-only attempt history.

## Runtime and resource census

| batch | replicate | process events | model requests | input tokens | output tokens | audit |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SM02 v3 | r01 | 146 | 41 | 106,841 | 4,666 | clean |
| SM02 v3 | r02 | 134 | 38 | 95,234 | 4,407 | clean |
| SM02 v3 | r03 | 154 | 45 | 93,747 | 5,873 | clean |
| SM05 v2 | r01 | 195 | 58 | 139,971 | 5,210 | clean |
| SM05 v2 | r02 | 193 | 59 | 126,318 | 4,911 | clean |
| SM05 v2 | r03 | 185 | 54 | 117,961 | 4,966 | clean |

All per-replicate audits report `ok=true`, no issues, no projection mismatch,
and no native bypass.  These rows must not be interpreted as quality or reward
comparisons.

## Automatic pure-process dataflow

The production path now executes:

```text
task completion
  -> pure source record + provenance anchor
  -> future-task opportunity/use/tool observations
  -> pure feedback record + provenance anchor
  -> cross-task joined process-signal case
```

The current repository also passes the post-rerun focused/full regression
baseline (RSIMem `1053 passed`; vendored PAST-Bench `399 passed, 2 skipped`),
with compileall, dependency, shell-syntax, secret-scan, and diff checks clean.

SM02 produced 15 source records, 7 feedback records, and 1 complete
artifact-set binding per replicate.  Each replicate has 18 persisted facts with
application semantic keys.  The binding is produced only when the transient
owner matcher can classify every persisted member against the frozen public
application contract; the pure record stores IDs, keys, digests, and
provenance, not fact text.

SM05 produced 21 source records and 11 feedback records per replicate.  Its
preference keys are not part of the frozen public notes application contract,
so the matcher correctly returned no per-fact keys and no artifact-set
bindings.  This is a fail-closed result, not missing instrumentation.

## Signal result

All feedback records in both batches remain `unresolved`.  The batch-level
optimizer corpora were generated successfully, but both report:

```text
process_signal_gate = no_signal
process_signal_optimization_count = 0
process_signal_hypothesis_digest = null
```

SM02 contains 25 protocol-bound physical cases after cross-replicate
deduplication; SM05 contains 31.  Joined cases are replayable and no longer
become censored merely because an unrelated tool result has an `UNKNOWN`
application-success schema.  Incomplete typed feedback and explicit
`observation_censored` markers still close a case conservatively.

The correct decision for both batches is therefore:

```text
STOP_NO_SIGNAL
```

No extraction N+1, held-out validation, adaptive effect batch, or six-layer
joint intervention was generated.  Trigger, source selection, admission,
commit, and exposure remain fixed/shadow or validation-only until an
extraction-owned, replay-stable opportunity → persisted artifact/set →
retrieval → exposure → memory-specific use → observable outcome chain exists.

## Strict compound-matcher rerun (SM02 v4)

After the initial production reruns, SM02 was repeated with the complete
compound semantic-role matcher enabled.  The run was performed in the same
clean detached tree and used the same frozen provider/model and feedback
protocol; only the matcher implementation changed.  All three replicates
completed with clean audits:

| replicate | process events | source records | feedback records | complete bindings | process-case statuses |
| ---: | ---: | ---: | ---: | ---: | --- |
| r01 | 146 | 15 | 7 | 1 | 3 `observable_only`, 4 `diagnostic_only`, 9 `censored` |
| r02 | 140 | 15 | 7 | 1 | 3 `observable_only`, 4 `diagnostic_only`, 9 `censored` |
| r03 | 144 | 15 | 7 | 1 | 3 `observable_only`, 4 `diagnostic_only`, 9 `censored` |

The batch contains 25 logical cases and 48 physical observations after the
registered `logical_case_v1` join.  Every feedback record remains
`unresolved`; the pure optimizer corpus has 21 examples and reports
`process_signal_gate = no_signal` with zero optimization cases.  The strict
matcher therefore confirms the earlier conservative result rather than
creating a candidate: the benchmark-audit compound binding is observable,
but no extraction-owned useful, harmful, or missed chain is available for a
general prompt update.

Raw provider accounting for the three replicates is recorded in their
manifests and audit artifacts (41/38/41 model requests respectively); these
values are reporting-only and are not used as learner signal.

## Verification

The corresponding clean checkout passed:

```text
RSIMem: 1053 passed
PAST-Bench: 399 passed, 2 skipped
pip check: passed
compileall: passed
shell syntax: passed
secret scan: passed
git diff --check: passed
```
