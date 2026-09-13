# RSIMem 下一阶段 Checklist

最后更新：2026-09-12
状态：`已完成（当前 checklist 的 Stage 1 与 Stage 2 B0/B2 主链路）`

## 总目标

当前执行清单只覆盖两个相互独立的阶段：

1. 完成代码重构，并冻结规范的运行时边界；
2. 运行第一组 `Mem0 + AdaMem` 实验，用于验证重构后的代码。

历史上的 `NoMemory` 实际表示 `SemanticDisabled`，不能再被当作完全关闭 Memory 的控制组。正式的 `AllMemoryOff` baseline 和三 backend baseline 已经冻结，证据记录在 `docs/current_experiment_results.md` 和 `docs/all_memory_off_baseline_20260911.md` 中。

Stage 1 的验收门通过前，不启动 Stage 2。Hermes + AdaMem、Episodic/
Procedural baseline 和我们自己的方法属于后续路线，不作为当前 checklist
的执行内容。

## Stage 1：代码重构与运行时边界

目标：在保持当前行为的前提下，让代码更容易扩展，并将实验代码与可复用
运行时代码分离。

### 1. 包所有权

- [x] 当前 AdaMem 代码位于 `experiments/adamem/`。
- [x] 当前 base-memory 代码位于 `experiments/base_memory/`。
- [x] 历史协议代码位于 `experiments/legacy/`。
- [x] benchmark、Hermes、evaluation、lifecycle 和 memory 等可复用代码
  在 `src/rsimem/` 下拥有规范归属位置。
- [x] 已删除根目录下的兼容性转发模块。
- [x] 确认所有生产代码 import 都指向规范归属位置，测试 workaround
  不能成为运行时依赖。

### 1.1 目录与模块功能边界

重构不能只停留在顶层目录搬迁。目录树本身必须能够表达架构：

- [x] 一个目录对应一个清晰的架构职责，例如 `protocol`、`runtime`、
  `launch`、`audit`、`analysis`、`backends`、`lifecycle`、`feedback` 和
  `policy`。
- [x] 一个模块只负责一个稳定的功能边界。launcher、runtime execution、
  audit、aggregate、contract 和 adapter 不能混成一个模块。
- [x] `memory_systems/` 按 memory kind 组织，而不是把所有具体 system
  实现堆在同一个平目录中：
  ```text
  memory_systems/
  ├── semantic/
  │   ├── mem0_flat/
  │   └── hermes_native/
  ├── episodic/
  │   └── hermes_native/
  ├── procedural/
  │   └── hermes_native/
  └── registry.py
  ```
- [x] 通过 registry 或 capability manifest 声明每个具体 system 支持的
  memory kind；当前 `mem0_flat` 只声明 semantic，`hermes_native` 声明
  semantic、episodic 和 procedural。
- [x] `memory/` 只放跨 system、跨 memory kind 的通用 contract、lifecycle、
  feedback、attribution、policy 和 audit，不把 Mem0-specific 逻辑混入其中。
- [x] 对实验协议目录和 memory runtime 目录中的大平目录进行功能拆分，
  但只在文件确实属于不同架构层时拆分。
- [x] 保持依赖方向明确：contract 和 shared primitive 不能依赖 launcher
  或 analysis；analysis 只能消费已记录的 evidence，不能控制 runtime。
- [x] 不为了减少单个目录的文件数量而机械拆分。只有形成稳定职责边界时，
  拆分才是有效的。
- [x] 增加架构测试或静态 reachability 检查，检测反向依赖和跨层归属错误。

### 2. Runtime 与 Memory surface contract

- [x] 定义统一的 runtime surface policy，明确 semantic、episodic、
  procedural 和 profile 是否可用。
- [x] 将 `expected_signal`/family label 与 runtime tool availability
  分离。family 标签不能隐式开启其他 memory 类型。
- [x] 在 run manifest 中明确记录 `AllMemoryOff`、native memory 和 Mem0
  semantic backend 的 surface policy。
- [x] 增加由 Hermes integration、PAST-Bench routing 和 audit 共用的
  canonical capability check。
- [x] 让 semantic、episodic、procedural 和 profile 操作输出 typed
  evidence，不能依赖 backend 特有的日志解析。

### 3. Lifecycle hook 与 evidence 边界

- [x] 明确 formation、persistence、retrieval、exposure/injection、use
  和 outcome 的 canonical hook 点。
- [x] 定义稳定的 event contract，至少包含 run ID、episode ID、memory
  type、operation、decision、result、provenance 和 failure status。
- [x] 显式记录缺失或失败的操作，不能让“没有操作”和“没有采集到日志”
  无法区分。
- [x] 通过严格 identity 将 hook event 与 model call、tool call、ledger
  record 和最终 outcome 关联起来。
- [x] 按 contract 要求，禁止 hidden grader score、future evaluation answer
  和不应暴露的 memory text 进入 updater/evidence payload。

### 4. Adapter、Launcher 与 Audit

- [x] 在行为可比较的范围内，让 backend adapter 实现统一的 typed
  read/write/retrieval contract。
- [x] 让实验 launcher 依赖 canonical runtime contract，不能依赖其他实验
  的内部实现细节。
- [x] 让每个 launcher 写入不可变的 manifest、condition、代码 revision、
  config digest 和隔离目录 identity。
- [x] 对未授权的 semantic、episodic、procedural 或 profile 操作增加
  fail-closed audit。
- [x] 为 `AllMemoryOff`、native、Mem0、EP、PC、PG 和 mixed sequence 增加
  focused contract test。

### 5. 重构验证

- [x] 在最终重构后重新运行 architecture/reachability audit。
- [x] 运行 `python -m compileall -q src experiments`。
- [x] 运行 RSIMem 测试和指定的 PAST-Bench 测试。
- [x] 运行 `pip check`、wheel build、secret scan 和 `git diff --check`。
- [x] 为 Stage 2 的 Mem0/AdaMem manifest 运行 provider-free dry run。
- [x] 记录一个干净的代码 revision，作为 Stage 1 冻结点。

### Stage 1 验收门

Stage 1 只有在以下条件全部满足时才算完成：

- canonical import 和 CLI entry point 在没有兼容性 shim 的情况下正常工作；
- 每个目录有清晰的架构职责，每个模块有稳定的功能边界；
- 可以仅通过 manifest 和 trace 恢复每个 memory surface 的可用状态；
- hook/evidence record 具有稳定的 identity 和 leakage boundary；
- deterministic、package、architecture 和 audit 检查全部通过；
- 已记录供 Stage 2 使用的准确代码 revision。

## Stage 2：Mem0 + AdaMem 验证实验

目标：用一组聚焦的 `Mem0 + AdaMem full trajectory` 实验，端到端验证重构后的 runtime、launcher、evidence plane 和 updater boundary。

### 1. 固定实验定义

- [x] 固定 `Mem0Static` 作为 semantic backend。
- [x] 固定 AdaMem policy-update adapter 及其允许的 patch space。
- [x] 固定 base model、task manifest、train/evaluation split、budget、temperature、provider endpoint 和 validation protocol。
- [x] 将验证 baseline 定义为 `B0 Mem0Static` 对比 `B2 Mem0 + AdaMem full trajectory`。
- [x] 将 terminal feedback（`B1`）和其他 feedback 粒度放到后续分析实验，不作为验证主实验链路的前置条件。
- [x] `AllMemoryOff` 和 `HermesNative` 只作为背景 baseline；除非在同一 method protocol 下重跑，否则不把它们混入 AdaMem method delta。

### 2. 执行与审计

- [x] 运行一次 provider-free manifest/materialization 检查。
- [x] 运行一次覆盖 SM、EP、PC 和 PG routing 的代表性 smoke。
- [x] 确认 B0 与 B2 除声明的 full-trajectory update 条件外，task、model 和 backend identity 完全一致。
- [x] 确认 B2 只能接收声明的 deployment-visible trajectory，不能访问 hidden grader score 或 future evaluation answer。
- [x] 确认 policy update 绑定到正确的 suffix run，且不修改 retrieval、routing、task prompt 或 model configuration。
- [x] 每个 replicate 都必须具备完整 request usage、隔离目录和 accepted phase status。

### 3. 首批正式实验

- [x] 先在 SM01-SM03 上运行三次 accepted matched replicate，比较静态 B0 与 full-trajectory B2。
- [x] 每完成一个 family batch 就进行 audit，再启动下一个 batch。
- [x] 保留失败 attempt 作为 provenance，但不得纳入 aggregate。
- [x] 分别记录 update、abstention、rollback、harmful-update、policy-diff、updater token、latency 和 task quality。
- [x] 如果出现 identity drift、usage 不完整、未授权 evidence 或 policy binding 无效，则停止扩展到完整 suite。

### 4. 结果与决策

- [x] 生成 machine-readable aggregate 和 per-family appendix。
- [x] 将 full-trajectory update behavior 与 quality uplift 分开报告；saturated family 不能支持 effectiveness claim。
- [x] 规划后续 feedback-granularity 分析，在同一冻结 substrate 上比较 terminal、full-trajectory 和 structured RSIMem feedback。
- [x] 检查 EP/PC 分数变化是否伴随 semantic operation，区分 cross-surface effect 与无法解释的 residual variance。
- [x] 根据 SM01-SM03 的 accepted batch、matched audit、usage 和 policy
  binding 结果，确认重构后的 runtime 具备扩展到完整 26-family 的前置条件。
- [x] 为完整 26-family 单独建立 B0/B2-only formal manifest；绑定当前冻结
  source manifest、配置 digest、代码 revision 和新的输出根目录，不覆盖
  SM01-SM03 的已验收结果。
- [x] 对 full-suite manifest 做 provider-free preflight：确认 family 数为
  26、family identity 唯一、配置 digest 正确、replicate 数为 3，且 condition
  恰好为 B0 `Mem0Static` 和 B2 `Mem0 + AdaMem full trajectory`，不得混入
  B1 terminal。
- [x] 对 26 个 family 逐一执行 routing/materialization dry run，确认 SM、EP、
  PC、PG 的 cutover label、surface policy、source sequence 和 output root
  均无漂移。
- [x] 通过 dry run 后，按 family 顺序运行正式 provider batch；每个
  family-condition 固定 3 个隔离 replicate，单个 batch 内最多并发 3 个
  replicate，前一 batch 未通过 audit 时不得启动下一 batch。
- [x] 每个正式 batch 完成后检查 accepted status、完整 task/updater usage、
  isolated state/trace/artifact/mem0 roots、code revision 和 policy binding；
  失败 attempt 保留 provenance，但不计入 aggregate。
- [x] 26-family 全部完成后生成 B0/B2 aggregate 和 per-family appendix，核对
  预期为 52 个 accepted condition-family batches、156 个 accepted runs，
  再更新 `progress.md`、`current_experiment_results.md` 和 `main_table.md`。
- [x] 只能根据 accepted frozen artifact 更新 `current_experiment_results.md`、`progress.md` 和 `main_table.md`。

### Stage 2 验收门

Stage 2 只有在 B0/B2 comparison 具备 accepted matched replicate、完整的 audit 和 usage evidence、有效的 policy binding，并且明确说明结果能证明什么以及不能证明什么时，才算完成。

## 停止条件

- 如果某个 memory surface 与 manifest 声明不一致地可用，立即停止。
- 如果 comparison 内 task、model、budget、fixture、代码 revision 或 backend identity 发生漂移，立即停止。
- 如果 updater 收到禁止的 evaluation evidence 或 memory text，立即停止。
- 如果 usage 不完整、policy binding 不明确，或 accepted run 无法根据 manifest 和隔离目录复现，立即停止。

## 后续路线（当前 checklist 不执行）

- 完成 `Mem0 + AdaMem` 后，再运行 `HermesNative + AdaMem`。
- 完成 semantic backend baseline 后，再系统运行 episodic 和 procedural
  baseline。
- 汇总并检查所有 baseline 的轨迹、memory operation、failure point 和
  policy update 行为。
- 完成上述 baseline 分析后，再实现并运行我们自己的 RSIMem-enhanced 方法。

## 当前状态

AllMemoryOff baseline、三 backend baseline、canonical package migration 和
Stage 1 freeze 已完成。Stage 2 的冻结 26-family B0/B2 比较也已完成：52 个
accepted condition-family batches、52 个 batch audits、26 个 matched audits 和
156 个 accepted runs（B0 78、B2 78）。正式 manifest、preflight、dry run 和
aggregate 位于 `outputs/adamem_mem0_full_trajectory_26family_20260912/`；完整
结果及限制见 `adamem_mem0_full_trajectory_26family_results_20260913.md`。

该结论只覆盖 `Mem0Static` 与 `Mem0 + AdaMem full trajectory`。B1 terminal、
Hermes + AdaMem、Episodic/Procedural 专用方法，以及 RSIMem-enhanced 方法均
尚未在这一协议下运行，不能从本结果推断其效果。B2 有 `9/78` harmful updates，
因此不得表述为始终改进。

跑实验的 API 使用：/mnt/20t/xubuqiang/Study/api_key.md
