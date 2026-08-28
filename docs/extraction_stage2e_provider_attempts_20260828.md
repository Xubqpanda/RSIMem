# Extraction Stage 2E Provider Attempt Record

Date: 2026-08-28

## Decision

Stage 2E live matched validation remains incomplete. No extraction candidate was
generated or activated.

The new plain-parent feedback path was executed from clean detached worktrees at
RSIMem commit `f5b3a03242338f5d5d2ec5c2b2252ed3a7a7dad4`. The attempts are retained
as provider/infra failures and are excluded from optimizer input.

## Attempts

- `feedback-sm01-20260828-v1` exposed an API-key file parsing error. The provider
  returned HTTP 401. The manifest closed the run as a provider failure.
- `feedback-sm01-20260828-v2` used the correctly extracted credential token, but
  the provider repeatedly returned HTTP 503 capacity errors. The run closed as
  an audit failure with zero semantic ingestion events.
- `feedback-sm01-20260828-v3` independently reproduced the HTTP 503 capacity
  failure. One agent call succeeded, 27 calls failed, and semantic ingestion
  remained zero.
- `feedback-sm01-20260828-v4` passed both preflight checks but was terminated by
  the local execution-tool timeout before the first attempt completed. Its open
  attempt was explicitly closed as `operator_interrupted`; it is not classified
  as provider evidence.
- `feedback-sm01-20260828-v5` ran to an audited terminal state from RSIMem commit
  `232b4f06a8573074663f86c0fff9afff9772415d`. Four of 34 agent calls succeeded
  and 30 failed with HTTP 503. All nine traces had incomplete model usage,
  semantic ingestion remained zero, and the attempt closed as `failed/audit`.

For every completed authenticated provider-capacity attempt, source records,
private optimizer captures, and live extraction-feedback records all remained
absent. None can enter the resolved denominator, optimizer corpus, offline
validation, or matched trial.

## Resume Gate

Resume from a new clean detached worktree and a new immutable batch ID only after
the frozen provider/model profile can complete the plain-parent sequence. A
resumed batch must produce current source schema records, private capture joins,
and at least the configured minimum actionable primary examples. The optimizer
must return `NO_PROPOSAL` if the resolved-signal threshold remains unmet.

Provider capacity does not authorize changing the model profile, lowering the
resolved-signal gate, reusing legacy schema-v2 evidence, or substituting a
deterministic candidate.
