# SM02 provider attempts — 2026-08-29 (v3/v4)

This report records two isolated attempts to extend the `SM02_constraint_retention`
process-feedback pilot.  They are provider/connectivity evidence only; neither
batch is eligible for optimizer input, matched validation, activation, or an
effect claim.

## Configuration

- RSIMem revision: `e016261` (`refresh process chain replay count`)
- PAST-Bench revision: recorded in each batch manifest
- Family: `SM02_constraint_retention`
- Method: `static-extraction-rsimem`
- Persistence: `with_persistence`
- Judge: disabled
- Replicates requested: 3
- Endpoints were run in separate batches and were never pooled.

Both configured endpoints returned HTTP 200 for an authenticated `/models`
health probe immediately before the attempts.  A health response is not
treated as evidence that a completion batch is usable.

A later provider-only completion probe against the primary endpoint returned
HTTP 503 with no completion or usage payload.  No benchmark batch was started
from that probe, and it is not included in any experiment accounting.

The reusable probe was run again after the completion-probe gate was added; it
timed out with `transport_error` after 20 seconds (no HTTP status, content, or
usage).  This provider-only result likewise did not start a benchmark batch.

The latest probe still returns `http_error` with status 503 and no completion or
usage payload.  The formal launchers therefore remain correctly gated and no
new SM02 attempt is created from this check.

Follow-up provider-only probes on 2026-08-29 used the formal model
`gpt-5.6-luna` and the exact configured completion route.  The primary endpoint
again returned `http_error`/HTTP 503; the backup endpoint returned
`transport_error` without an HTTP status.  Neither probe exposed completion
content or usage, and neither started a benchmark task.  These results are
kept as connectivity diagnostics only and do not alter the batch accounting.

## Batch outcomes

| batch | endpoint | completed traces | requests | retries | input tokens | output tokens | audit | disposition |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `s1-sm02-process-20260829-v3` | primary | 8 | 37 | 5 | 36,979 | 3,475 | failed: 3 `incomplete_model_usage` | provider-capacity failure; incomplete |
| `s1-sm02-process-20260829-v4-backup` | backup | 9 | 27 | 0 | 0 | 0 | failed: 9 `incomplete_model_usage` | empty think-only responses; incomplete |

The primary endpoint returned useful content for most traces, but an HTTP 503
occurred during a reflection request and the resulting missing usage evidence
made the run fail closed.  The backup endpoint returned only empty think blocks;
the runtime exhausted its empty-content retries for each task.  Although those
requests were transport-successful, no provider usage was reported, so the
audit correctly rejects the batch rather than interpreting missing usage as
zero.

An additional endpoint probe explains the backup result: its configured root
URL serves the Sub2API web UI, while `/v1/chat/completions` returns an upstream
authentication error.  The backup key/base-URL pair therefore does not provide
a usable OpenAI-compatible completion route in this environment and should
not be retried as an experimental provider until its operator supplies a
working API route and credential.

## Process-signal diagnostic

The failed attempts were still inspected as content-free process evidence
after exact event-ID deduplication.  The primary retry retained 74 unique
events (policy-bound coverage 43/74, receipt-bound coverage 52/74), including
trigger, source selection, extraction, admission, commit, exposure, retrieval,
and task-outcome events.  The backup retry retained 37 unique events (14
exposure, 14 retrieval, and 9 task-outcome events); its empty responses caused
no formation policy chain to be emitted.  These counts are infrastructure
diagnostics only and do not enter the optimizer, strict attribution denominator,
or matched validation.

The complete attempt manifests and raw traces are retained under:

- `outputs/extraction_feedback/hermes_luna/s1-sm02-process-20260829-v3/`
- `outputs/extraction_feedback/hermes_luna/s1-sm02-process-20260829-v4-backup/`

No optimizer proposal, activation decision, or quality aggregate was produced.
The process events and failure reasons remain useful for infrastructure
diagnosis, but they are not task-level negative labels.  A new clean,
single-provider SM02 batch is required after sustained provider completion
health is established.

## Follow-up SM01 attempts — v5/v6

After a transient successful completion probe, a clean detached worktree
started `s1-sm01-feedback-20260829-v5`.  The first replicate reached the
`learn_b` reflection episode, then failed closed because the bridge attempted
to project a reflection as a new semantic extraction without an invocation
fingerprint.  This was an RSIMem runtime defect, not a provider label or task
outcome; the partial attempt is retained under
`/tmp/rsimem-formal-sm01-v5/outputs/` and is excluded from feedback and
accounting aggregates.

The defect was fixed in commit `e22af5c` by reserving semantic compilation for
the primary task-completed boundary.  A new clean attempt
`s1-sm01-feedback-20260829-v6` was registered against that commit, but its
pre-task completion probe immediately returned HTTP 503.  It therefore created
no task trace.  Both attempts remain diagnostics only; a fresh clean batch is
required after a sustained successful probe.
