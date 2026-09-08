# Stage 1 Three-Panel Merged Slice

> Status: `historical/superseded` (2026-09-08). This merged native-attribution
> slice is retained as evidence and does not define the current experiment.

Date: 2026-09-06  
Protocol: `native-attribution-repair-v1`

The accepted EP01 retry was merged with the previously frozen SM02/PC01
accepted slice using `merge_native_attribution_corpora()`. The merge requires a
common protocol and rejects duplicate accepted runs, observations, candidates,
or accepted/excluded identity overlap.

## Result

| Measure | Value |
| --- | ---: |
| Accepted runs | 3 |
| Observations | 15 |
| Candidates | 15 |
| Unresolved candidates | 15 |
| Actionable candidates | 0 |
| Panels | semantic, episodic, procedural |

Corpus: `native-corpus.e23d5342d06a972240a7064df73891b07967d872`  
Report: `native-attribution-report.a5ee4014f6a2226b05ab2b30179b9950de6e1389`

The merged report contains five new manifest-bound `replicate_id=1` EP01
observations and ten older v2 observations whose replicate field predates the
schema upgrade and remains `unknown`. This identity difference is documented
schema provenance, not a new replicate claim.

The Stage 2 decision remains `STOP_NO_ACTIONABLE_SIGNAL`; no candidate has a
unique evidence-backed repair axis and no reviewer coverage exists. This is a
three-panel accepted slice, not the required SM/EP/PC/PG four-panel gate.
