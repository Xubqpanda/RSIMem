# Current Experiment Results

Status: `26-family B0/B2 full-trajectory validation accepted`  
Last updated: 2026-09-13  
Protocol: `adamem-trajectory-baseline-v1`

本表只记录当前冻结协议下的主结果。每个 family-condition 包含 3 个独立
accepted replicate，数值为 family-level matched Near/Far N+1 evaluation 的
宏平均；显示格式为 `Near / Far`。未在本协议下运行的条件保持空白。

## Table 1: Full-Suite Quality

每个数值为 3 个 accepted replicate 的均值；括号内为标准差。数值显示为
`Near / Far`，分数越高越好。

| Layer | Method | Semantic (7) | Episodic (3) | Procedural (10) | Proactive retrieval (6) | Overall (26) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| L0 | AllMemoryOff | 0.3962 (0.0101) / 0.3943 (0.0151) | 0.3133 (0.0793) / 0.3083 (0.0880) | 0.3560 (0.0556) / 0.3560 (0.0556) | 0.2968 (0.0668) / 0.3060 (0.0707) | 0.3482 (0.0012) / 0.3493 (0.0013) |
| L1 | HermesNative |  |  |  |  |  |
| L1 | Mem0Static (B0) | 0.9625 / 0.9681 | 0.5607 / 0.6066 | 0.5278 / 0.5323 | 0.6888 / 0.6983 | 0.7327 / 0.7356 |
| L2 | Mem0 + AdaMem terminal (B1) |  |  |  |  |  |
| L2 | Mem0 + AdaMem full trajectory (B2) | 0.9684 / 0.9693 | 0.5795 / 0.6063 | 0.5645 / 0.5427 | 0.6936 / 0.7077 | 0.7448 / 0.7411 |
| L2 | HermesNative + AdaMem |  |  |  |  |  |
| L3 | Mem0 + AdaMem + RSIMem |  |  |  |  |  |
| L3 | HermesNative + AdaMem + RSIMem |  |  |  |  |  |

## Table 2: Update Behavior

| Method | Update rate | Abstention rate | Rollback rate | Harmful update rate | Updater input tokens | Updater output tokens | Updater latency | N+1 delta vs. B0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mem0Static (B0) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.000 |
| Mem0 + AdaMem terminal (B1) | not run | not run | not run | not run | not run | not run | not run | not run |
| Mem0 + AdaMem full trajectory (B2) | 58/78 (74.36%) | 20/78 (25.64%) | 0/78 (0%) | 9/78 (11.54%) | 11,357.7 | 229.1 | 4,833.3 ms | +0.01986 |
| Mem0 + AdaMem + RSIMem |  |  |  |  |  |  |  |  |

## Formal Evidence

Manifest:
`outputs/adamem_mem0_full_trajectory_26family_20260912/formal_manifest.json`  
Aggregate:
`outputs/adamem_mem0_full_trajectory_26family_20260912/provider_runs/validation_aggregate.json`  
Manifest digest:
`3a558d406e19e95f06268762a5fdb0f25a99c8c6e4ec3595acefd7a57bcba40b`

The accepted comparison contains 26 families, 52 accepted condition-family
batches, 52 batch audits, 26 matched audits, and 156 accepted runs: 78 B0 and
78 B2. All 78 B2 runs have complete updater usage. The B0/B2 difference is a
descriptive result for this frozen protocol, not an unconditional improvement
claim; B2 includes 9/78 harmful updates.

B1 terminal feedback, Hermes + AdaMem, and RSIMem-enhanced conditions were not
run in this protocol.
