# Stage 1 EP/PG Provider Attempt

Date: 2026-09-05  
Batch: `stage1-ep-pg-v3-20260905`  
Protocol: `native-attribution-repair-v1`

## Outcome

The primary provider passed a five-request preflight probe (`5/5` HTTP 200,
non-empty content, and usage). A new manifest-bound EP01/PG01 batch was
prepared with isolated state, Hermes home, artifact, trace, service ports, and
`--runtime local`.

The EP01 single-replicate attempt was stopped during the second task after the
runner entered repeated judge `APIConnectionError` retries and made no bounded
progress. The attempt did not produce a complete sequence result and is
classified as infrastructure evidence, not a quality observation. PG01 was not
started. The run-specific process and service were terminated and its ports
were confirmed released.

## Decision

This attempt does not change the frozen Stage 1 corpus and does not satisfy the
SM/EP/PC/PG cross-panel gate. It is excluded from the quality denominator under
provider/runner infrastructure failure. No repair case or Stage 2 transition is
authorized.
