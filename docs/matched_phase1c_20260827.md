# Phase 1C Live Matched Read-Path Validation

Date: 2026-08-27

This report applies the pre-registered Phase 1C protocol in `experiment_plan.md` to the first complete live batch with same-call native-shadow projection verification. It validates the current SM01 read-path infrastructure; it does not evaluate memory generation, mutation policy, lifecycle dry-run, or adaptive RSIMem.

## Decision

Phase 1C passes for `memory_ability/SM01_preference_adoption`.

- Machine gate: `stageGatePassed=true`, `issues=[]`.
- Successful runs: 3 per mode, 9 total.
- Failed attempts in the accepted batch: 0.
- Every run: 17 unique traces, audit clean, request accounting reconciled, 0 model retries.
- Adapter runs: 28 exact same-call projection checks each, 84 total; 0 mismatch, 0 native bypass, 0 unresolved injection.
- Privacy across all runs: 0 memory-text leaks, 0 credential-pattern hits, 0 absolute source paths.
- Task manifest, episode order, budget, initial state, model, judge, runtime, and persistence isolation matched.

The provider did not expose a verifiable seed. These are independent unseeded replicates, not seeded trials. The exact projection checks establish that the adapter returned the same semantic system-prompt values as native Hermes at each observed read call. Differences in separately sampled model outputs and resource use are therefore reported as unseeded model/provider variation.

## Configuration

- Accepted batch: `outputs/matched/hermes_luna_sm01/20260827_073620`
- Experiment ID: `ea556ca289803dd6727961861da44232df0ae87ef9a375f3258c3a325a2ff4bd`
- RSIMem commit: `24def0684481fbb14ad8db5781c31c5115182d29`
- Last PAST-Bench change: `fe631ca951e8d28777f2f3cd7381036b97dbbbcc`
- PAST-Bench subtree: `43c5a348fcc0f3d1b81bb25f592c8086ea263d8a`
- Dirty state: RSIMem false; PAST-Bench false.
- Agent/model: `hermes-luna` / `gpt-5.6-luna`; temperature `0.0`.
- Judge: disabled.
- Failure policy: `fail_closed`.
- Projection verification: enabled.
- Per-task budget: task-manifest `max_turns=20`, timeout `300s`.

Actual order rotated as scheduled:

1. `native`, `native+ledger`, `native+adapter+ledger`
2. `native+ledger`, `native+adapter+ledger`, `native`
3. `native+adapter+ledger`, `native`, `native+ledger`

## Quality

Every mode and replicate produced the same evaluation outcome:

| Variant | Eval score | Pass rate |
|---|---:|---:|
| With persistence | 1.000 | 1.000 |
| Without persistence | 0.400 | 0.000 |
| Persistence gap | 0.600 | 1.000 |

The episode-level scores were also identical across all nine runs. Cold was `0.6`; learn A was `1.0` with persistence and `0.56` without; learn B was `1.0` with and `0.6` without; eval-near and eval-far were `1.0` with and `0.4` without; controls were `0.8`, `1.0`, and `0.6`; reflection was `0.0`. There is no episode quality divergence to attribute to execution mode in this batch.

## Raw Resources

Values are `median [min, max]` over three independent runs. They are raw quantities, not provider-priced cost.

| Metric | Native | Native + ledger | Native + adapter + ledger |
|---|---:|---:|---:|
| Requests | 68 [68, 72] | 68 [66, 68] | 69 [68, 70] |
| Input tokens | 94,284 [88,558, 103,479] | 90,058 [89,430, 99,880] | 94,619 [93,363, 123,338] |
| Output tokens | 6,790 [6,785, 6,810] | 6,441 [6,345, 6,462] | 6,503 [6,413, 7,070] |
| Cache-read tokens | 53,760 [38,400, 59,904] | 47,616 [41,984, 50,688] | 47,104 [36,864, 49,152] |
| Cache-write tokens | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| Reasoning tokens | 3,356 [3,341, 3,443] | 3,197 [3,145, 3,212] | 3,133 [3,064, 3,551] |
| Retries | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| Tool-call views | 59 [56, 59] | 56 [55, 57] | 57 [57, 59] |
| Retrieved record views | unavailable | 8 [8, 11] | 4 [4, 4] |
| Injected chars | 582 [549, 603] | 513 [513, 612] | 564 [540, 606] |
| Peak stored bytes | 201 [194, 250] | 400 [171, 402] | 286 [255, 422] |
| Wall seconds | 498.67 [483.42, 509.43] | 474.92 [459.06, 475.39] | 444.24 [443.87, 484.93] |
| Ledger events | 190 [187, 194] | 250 [248, 250] | 253 [253, 256] |

Direct native has no typed retrieval event, so retrieved record views remain unknown rather than being reported as zero. Ledger-only observer reads and adapter-owned frozen snapshot reads have different physical evidence counts; exact same-call return-value checks, not those counts, are the model-visible parity criterion. No post-hoc tolerance or significance claim is made from three unseeded samples.

## Excluded Development Batches

Two earlier batches are retained but excluded from the accepted aggregate:

- `20260827_051318` predated native-shadow verification. It exposed missing user-profile injection matching and included one local launcher-timeout partial attempt.
- `20260827_070609` enabled projection checks but the ledger attribute allowlist rejected the new `equivalent` field. The adapter calls observed before ledger construction had zero mismatch, but the batch failed closed at the ledger stage.

The earlier `20260827_010856` hybrid-path infrastructure replicate remains documented separately in `matched_20260827.md`. None of these batches is silently merged into the accepted sample.

## Limitations

SM01 exercises the live semantic system-prompt path. Episodic search and procedural skill projection remain covered by deterministic native/adapter execution fixtures, not by this live family. The sample is small and unseeded, so this report supports the Phase 1C infrastructure gate, not a broad statistical equivalence claim across models, providers, families, episodic queries, or procedural skill use.
