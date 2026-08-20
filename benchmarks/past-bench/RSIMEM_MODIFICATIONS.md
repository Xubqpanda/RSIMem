# RSIMem Modifications

This copy originates from <https://github.com/Gen-Verse/PAST-Bench> and retains the upstream Apache-2.0 `LICENSE` and `NOTICE` files.

RSIMem vendors the benchmark to make experiment runtime changes reproducible. Local modifications must remain scoped to integration, telemetry, and lifecycle accounting unless a protocol change is explicitly documented. Task definitions, answer keys, grading criteria, and persistence-control semantics are not modified for the initial study.

## Local Changes

- Added request-level model usage evidence for runtime adapters and traces.
- Connected Hermes canonical provider usage to the benchmark telemetry contract.
