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
