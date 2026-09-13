# Current Goal

代码重构的 Stage 1 freeze 和 Stage 2 的 26-family `B0 Mem0Static` 对比
`B2 Mem0 + AdaMem full trajectory` 验证已完成。正式实验固定三次 accepted
matched replicate、相同 task/model/backend identity、deployment-visible trajectory、
完整 request usage 和隔离运行目录。

Stage 2 产生 156 个 accepted runs（B0 78、B2 78），详细结论见
[`adamem_mem0_full_trajectory_26family_results_20260913.md`](adamem_mem0_full_trajectory_26family_results_20260913.md)。
下一阶段仍需单独设计和运行 B1 terminal、Hermes + AdaMem 与 RSIMem-enhanced
条件；它们没有包含在已完成比较中。B2 的 `9/78` harmful updates 也要求后续
工作优先分析更新安全性，而不是把当前均值差异视为普遍改进。
