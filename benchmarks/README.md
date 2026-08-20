# Benchmarks

Benchmark implementations are kept beside one another so RSIMem can version the exact runtime instrumentation used by each experiment without mixing benchmark-specific code into the LightRSI method package.

| Directory | Purpose | Upstream |
|---|---|---|
| `past-bench` | Initial long-horizon memory and persistence evaluation | <https://github.com/Gen-Verse/PAST-Bench> |

PAST-Bench is distributed under Apache-2.0. Its original `LICENSE` and `NOTICE` files remain in `benchmarks/past-bench`. RSIMem modifications are limited to experiment integration, runtime telemetry, and accounting unless a study explicitly documents a protocol change.

A second RSI benchmark should be added as another directory under `benchmarks/`, with its own provenance and license files.
