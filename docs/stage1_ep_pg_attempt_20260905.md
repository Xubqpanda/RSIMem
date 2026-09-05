# Stage 1 EP/PG Provider Attempt

Date: 2026-09-05  
Batch: `stage1-ep-pg-v3-20260905`  
Protocol: `native-attribution-repair-v1`

## Outcome

The primary provider passed a five-request preflight probe (`5/5` HTTP 200,
non-empty content, and usage). A new manifest-bound EP01/PG01 batch was
prepared with isolated state, Hermes home, artifact, trace, service ports, and
`--runtime local`.

The first EP01 attempt was stopped during the second task after repeated judge
`APIConnectionError` retries. The launcher was then corrected to pass
`--runtime local --no-judge`, and the same isolated run completed all five
tasks. Its final evaluation task still encountered repeated provider HTTP 503
responses, so the aggregate result reported incomplete model usage and failed
the manifest-bound audit. Both attempts are infrastructure evidence, not
quality observations. PG01 was not started. Run-specific processes and
services were terminated and their ports were confirmed released.

## Decision

This attempt does not change the frozen Stage 1 corpus and does not satisfy the
SM/EP/PC/PG cross-panel gate. It is excluded from the quality denominator under
provider/runner infrastructure failure. No repair case or Stage 2 transition is
authorized. The `--no-judge` launcher change is covered by deterministic tests,
but it does not make an incomplete provider run acceptable.
