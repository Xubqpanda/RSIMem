# SM02 optimizer retry — 2026-08-29

The previously completed clean parent batch `s1-sm02-feedback-20260829-v5`
was reused without changing its corpus, task traces, or labels. The primary
provider completion probe returned HTTP 200 with non-empty content and usage.

The first retry used the old `json_object` response mode and again returned a
syntactically valid `PROPOSE` object missing the required top-level
`reason_codes` field. Strict parsing rejected it. A bounded diagnostic retry
confirmed the same shape across three completions; response bodies were not
stored or reported.

The provider supports the frozen JSON Schema response mode. The optimizer
adapter now sends `json_schema` with the versioned
`EXTRACTION_OPTIMIZER_OUTPUT_SCHEMA` and `strict=true` (commit `d2d06fc`).
With that boundary, the same request produced a schema-complete `PROPOSE`
with one edit and complete usage. The resulting candidate is stored only as
an extraction proposal; static candidate safety passed. It is not ACTIVE,
has not entered matched validation, and does not constitute an uplift claim.

Artifacts (owner-controlled/ignored output):

`outputs/extraction_feedback/hermes_luna/s1-sm02-feedback-20260829-v5/proposal-provider-retry-20260829-v3/`

The provider's backup endpoint was probed separately and returned invalid JSON,
so it was not used for optimizer execution. No schema relaxation or manual
candidate construction was performed.
