# SM01 feedback attempts — 2026-08-29 (v5–v8)

This report records three post-probe attempts to run the plain static SM01
feedback batch plus one follow-up provider-only probe.  They are retained as
infrastructure diagnostics only.  No batch below is eligible for optimizer
input, matched validation, activation, or an effect claim.

## Attempt outcomes

| attempt | provider gate | task progress | terminal issue | disposition |
| --- | --- | --- | --- | --- |
| `v5` | passed (HTTP 200, content/usage available) | first replicate reached `learn_b` reflection | reflection episode incorrectly entered semantic extraction and lacked an invocation fingerprint | incomplete runtime attempt |
| `v6` | failed (HTTP 503) | no task | provider capacity failure before first task | no-trace diagnostic |
| `v7` | passed (HTTP 200, content/usage available) | first replicate completed the task sequence through control tasks | launcher audited an exact shared-cold duplicate process event before canonical corpus construction | incomplete evidence attempt |
| `v8` probe | failed (HTTP 503) | no batch started | provider capacity failure at the follow-up completion probe | provider-only diagnostic |

All attempts used a clean detached worktree, the fixed model
`gpt-5.6-luna`, plain `static-extraction-rsimem`, `native+ledger`, disabled
lifecycle evaluator, disabled native writer, and `with_persistence`.  Their
raw manifests, traces and provider probes are retained under:

```text
outputs/extraction_feedback/hermes_luna/s1-sm01-feedback-20260829-v5/
outputs/extraction_feedback/hermes_luna/s1-sm01-feedback-20260829-v6/
outputs/extraction_feedback/hermes_luna/s1-sm01-feedback-20260829-v7/
```

The v8 probe was run after v7; it failed before a batch directory was created
and therefore has no local output copy beyond the command result.

## Runtime fixes

The v5 failure exposed that PAST-Bench reflection is a separate review episode,
not a second completed-task extraction boundary.  Commit `e22af5c` makes the
Hermes bridge process reflection observations without invoking semantic
compilation or extraction projection, and adds a regression fixture.

The v7 failure exposed that shared-cold evidence is visible both in the nested
shared directory and in the attempt directory.  Commit `569e295` constructs
the canonical `ProcessCorpus` before auditing in both formal launchers.  Exact
duplicate event IDs are collapsed; conflicting payloads still fail closed.

The v7 trace was replayed with the corrected ordering:

```text
raw process events:       76
canonical process events: 75
process audit errors:      0
```

This replay validates evidence construction only; it does not turn the
incomplete provider attempt into a completed feedback replicate.

## Provider interpretation

The alternating HTTP 200/503 probes indicate intermittent provider capacity,
not a deterministic RSIMem failure.  Formal launchers remain intentionally
fail-closed: a failed completion probe starts no benchmark task, and a partial
task run is never silently converted into feedback or a quality denominator.
A new clean batch is required after sustained completion health, now with both
runtime fixes included.
