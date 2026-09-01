# RSIMem PAST-Bench Foundation Checklist

最后更新：2026-09-01

## 0. 文档定位

本文是 RSIMem 当前唯一的实现与验收主清单。旧版 checklist 研究的是“能否根据 deployment-visible process evidence 自动修改 semantic extraction prompt”，但 SM02/SM05 的真实 clean-parent 实验均得到 `STOP_NO_SIGNAL`。这说明 extraction-only 设定过度依赖细粒度 attribution，也无法覆盖 semantic、episodic、procedural 三类 memory 的不同更新机制。因此，旧版尚未完成的真实 N+1 prompt、held-out prompt validation 和 matched prompt effect 不再继续执行。

RSIMem 当前研究问题是：

> 在冻结基础模型参数的条件下，什么类型、什么粒度的反馈，足以驱动 semantic、episodic 和 procedural memory 产生可靠、可归因且能够泛化的自我改进？

当前只执行四个串行阶段：

0. **仓库清理与基线冻结。** 删除 extraction-only 主线遗留的冗余代码、配置、脚本、测试和重复文档，同时保护通用 runtime、lifecycle、evidence 和历史否定结果。
1. **研究协议冻结。** 固化三类 memory、六个 lifecycle control surfaces、PAST family 映射、对照组、指标、数据边界和允许声明的 claim。
2. **通用实验架构。** 建立 `BenchmarkAdapter`、`HostAdapter`、`MemoryMethodAdapter` 和 `FeedbackCondition` 四个独立边界，并拆分现有 Hermes/PAST 单体接线。
3. **PAST memory sensitivity。** 在接入 AdaMem、MemQ 和 Recuris 前，证明 PAST 的 SM、EP、PC families 分别对对应 memory 类型有真实且机制一致的敏感性。

阶段 4 至阶段 7 不属于当前执行范围。阶段 3 完成后，才决定是否接入三种方法、运行 feedback sufficiency、构造 `w/ RSIMem` 和增加外部 benchmark。

状态约定：

- `√`：已有实现与可审计证据，可作为稳定依赖。
- `□`：尚未完成，是当前执行项。
- `部分完成`：已有可复用实现，但未满足新的跨类型口径。
- `停止`：旧方向不再继续，不得按原计划恢复。
- `延后`：是否执行取决于阶段 3 的结果。

## 1. 冻结边界

### 1.1 当前实验范围

- Benchmark 固定为 vendored PAST-Bench。
- Host 固定为 Hermes。
- 同一比较块内固定基础模型、provider、sampling、任务预算、工具预算和 retry policy。
- 研究对象是 external persistent memory，不在线更新执行 Agent 的基础模型参数。
- Semantic、episodic 和 procedural 是同级 memory 类型；working context、feedback 和 policy metadata 不是第四类 memory。
- Cost、token、latency、storage 和 API call 只用于报告，不进入当前 updater 学习信号。
- PAST grader、hidden expectation、答案和 official score 只允许进入 final evaluation plane。

### 1.2 三类 Memory 的操作性定义

分类依据持久化 memory unit 的内容，不依据论文名称、更新算法或下游用途。

| 类型 | 持久化内容 | 必须保留的身份 | 典型用途 |
| --- | --- | --- | --- |
| Semantic | 去情境化的事实、偏好、规则和约束 | subject、scope、validity、source provenance | 回答事实、遵守偏好和约束 |
| Episodic | 发生在具体任务、时间或环境中的经历 | episode、context、outcome、provenance | 回忆相似案例和迁移经历 |
| Procedural | 可复用 SOP、skill、方法和动作模式 | applicability、steps、version、validation | 执行任务、恢复失败和调用技能 |

混合系统必须声明 `primary_kind` 和 transform。例如 RGMem 可以声明为 `episodic -> semantic consolidation`，但未隔离 L1 profile 时不能报告为纯 semantic 方法。

### 1.3 后续首选方法

阶段 0 至阶段 3 不实现真实方法，但预先冻结首选方法，避免看完 sensitivity 结果后任意换方法。

| 类型 | 方法 | 固定论文身份 | 归类 |
| --- | --- | --- | --- |
| Semantic | AdaMem | arXiv `2606.21144` | semantic write-policy adaptation |
| Episodic | MemQ | arXiv `2605.08374` | episodic credit assignment |
| Procedural | Recuris | arXiv `2608.24876` | procedural skill/harness evolution |

不得将 arXiv `2606.05684` 的另一篇 AdaMEM 与 `2606.21144` 混用。SelfMem、SAGE、RGMem、UMEM、RoMeRL 和 GSEM 只保留为 secondary candidates。

### 1.4 六个 Lifecycle Control Surfaces

旧六层 policy 不再被视为六个必须联合优化的模块，而被保留为统一诊断坐标：

| Surface | 核心问题 | Semantic | Episodic | Procedural |
| --- | --- | --- | --- | --- |
| Trigger | 何时产生 candidate | 何时检查新事实 | 何时关闭 episode | 何时触发 skill distillation |
| Source Selection | 使用哪些证据 | 用户消息和 tool result | 轨迹、状态和 outcome | 成功、失败和恢复步骤 |
| Construction | 构造成什么 | 事实、偏好和规则 | 带 provenance 的 episode | SOP、skill 和适用条件 |
| Admission/Maintenance | 如何接受和维护 | ADD/UPDATE/MERGE/NOOP | 去重、保留和 credit | 新建、修补、替换和失效 |
| Commit/Versioning | 何时持久化生效 | revision、scope、rollback | episode commit 和 lineage | skill version 和 activation |
| Retrieval/Exposure | 何时提供给 Agent | 检索并注入事实 | 召回相似 episode | 选择、调用或注入 skill |

这些 surface 用于记录方法修改了什么、失败发生在哪里、反馈能否归因到目标 surface。当前不要求每个方法修改全部 surface，也不声明“六层都能自优化”。

### 1.5 三个证据平面

1. **Pure Process Plane** 只包含部署自然可见的上下文、memory lifecycle、retrieval、exposure、tool call/result、状态变化和用户反馈。
2. **Benchmark Audit Plane** 可以使用预注册 family contract 检查实现与机制，但 label 必须标记为 `benchmark_audit_only`。
3. **Final Evaluation Plane** 只在实验冻结并运行完成后读取 official score，不得回写 updater 或下一轮 proposal。

Evidence 必须带 plane、source identity、observation cutoff 和 provenance。归因不确定时保持 `unresolved`、`unknown` 或 `censored`，不得用 benchmark 先验补标签。

## 2. 可复用资产与停止项

### 2.1 保留的通用资产

- √ PAST-Bench vendoring、安装、preflight、runner、grader isolation 和 usage accounting。
- √ Hermes session、native semantic/episodic/procedural storage-boundary fixtures 和工具调用基础。
- √ `ContextSnapshot`、source projection、tool closure、revision、provenance 和 idempotency。
- √ Transaction validation、CAS、receipt、restart recovery 和 rollback。
- √ Operation graph、process corpus、logical case、artifact set 和 crash-safe store。
- √ Pure process、benchmark audit、final evaluation 三平面类型与拒绝边界。
- √ Opportunity、retrieval、exposure、tool closure 和 observable outcome 基础事件。
- √ Revocation、schema version、stale artifact fail-closed 和 secret scan。
- √ `unresolved`、`censored`、`not_exposed`、`injected_not_used`、`retrieval_miss`、`unknown_usage`。
- √ SM02/SM05 clean-parent 的 `STOP_NO_SIGNAL` 结果和 case audit。

### 2.2 泛化后复用

- `extraction-owned diagnosis` 改为 `method-owned/surface-owned diagnosis`。
- `extraction prompt artifact` 改为 versioned `MethodStateArtifact` 或 `MemoryPolicyArtifact`。
- `N+1 prompt` 改为方法 state、memory bank 或 policy state 的 `N+1` revision。
- `extraction behavior variation` 改为目标 memory surface 的 state/action variation。
- Mem0-flat path 只作为 semantic fixture，不再作为框架默认方法。
- 旧六层 deterministic/shadow fixture 只保留 contract 和 intervention 思想。

### 2.3 明确停止

- 停止：使用现有 SM02/SM05 corpus 生成 extraction prompt N+1。
- 停止：旧 checklist 的 extraction-only held-out、activation 和 matched effect。
- 停止：继续追加 SM family，直到偶然找到 extraction-owned signal。
- 停止：把 deterministic action variation 当作真实 policy improvement。
- 停止：把 lifecycle cost units 当作 optimizer reward。

## 3. 阶段 0：仓库清理与基线冻结

阶段 0 不改变研究行为，而是把 extraction-only 原型清理为承载三类 memory 实验的最小可信基线。

### 0A：冻结清理前基线

- √ 记录 Git commit、Python、依赖锁状态、Hermes commit 和 PAST vendored identity，见 [`baseline_manifest_20260901.json`](baseline_manifest_20260901.json)。仓库当前没有 lock file，manifest 固定记录 pip-freeze digest。
- √ 在 clean tree 运行 RSIMem 全量测试、PAST 测试、compileall、dependency、shell syntax、secret scan 和 diff check。
- √ 生成 baseline manifest，记录 test count、skips、关键 fixture identity 和验收命令。
- √ 记录公共 CLI、Python imports、config entry points 和 artifact schemas。
- √ 记录 `hermes_past_bridge.py` 的职责和调用图，建立阶段 2 拆分基线。

反向验收：

- √ 无 clean baseline manifest 时禁止删除；manifest 的 `deletionAuthorized` 当前为 `false`。
- √ 工作树、依赖或 PAST identity 不一致时 preflight 失败；`python -m rsimem.baseline` 校验 manifest、clean tree、commit drift、pip-freeze、requirements、Python/package/import origin 和 vendored tree digest，缺失或漂移均 fail closed。
- √ 不得通过修改 baseline 数字掩盖测试失败；manifest 保存原始命令结果和固定 source commit。

### 0B：资产分类

√ 每个候选文件已标记为 `KEEP`、`GENERALIZE`、`DELETE` 或 `EVIDENCE_KEEP`，理由和已知依赖者见 [`asset_inventory_20260901.md`](asset_inventory_20260901.md)。该表只完成初始分类，不授权删除。

优先审计：

- `src/rsimem/extraction_*`、`adaptive_*`、`matched_analysis.py` 及对应测试。
- `src/rsimem/memory/extraction_*`、`adaptive_*`、`pure_extraction*`、旧 prompt optimizer 和 activation。
- `configs/extraction_feedback_*`、`extraction_split_plan_*`、`extraction_validation_*`。
- `scripts/run_luna_adaptive_*`、`run_luna_extraction_*`、旧 SM01/static/matched launchers。
- `docs/extraction_*`、`phase*`、`matched_*`、`static_*` 和 provider attempt 流水账。
- 只服务于已撤销 candidate、旧 corpus 或停止 N+1 的 fixtures。

强制保留：

- Generic contracts、artifact identity、revocation、evidence plane、lifecycle event 和 rollback。
- `current_checkpoint_20260901.md`、`case_analysis.md` 和解释 `STOP_NO_SIGNAL` 所需 evidence index。
- Raw outputs 的 append-only identity；可以移出默认路径，但不得篡改或用新 schema 重写。
- PAST task data、grader 和原始 benchmark 文档。

反向验收：

- □ 不能仅凭文件名含 `extraction` 就删除，通用 contract 必须先迁移调用者。
- □ 不能删除 revocation entry 后让旧 candidate 重新可加载。
- □ 不能删除解释旧判断和 `STOP_NO_SIGNAL` 的唯一 evidence。
- □ Dataset、grader、task prompt 和原始 fixture 不参与格式清理。

### 0C：删除与收敛

- □ 对 `DELETE` 文件完成 import/call-site 审计，确保无 production、test、CLI 或 packaging 引用。
- □ 删除停止的 N+1 launcher、无调用 wrapper、重复 config 和仅服务撤销 artifact 的代码。
- □ 删除 dead tests，但保留保护 generic contract 的反向测试。
- □ 将重复流水账收敛为少量 current-state 文档后删除旧报告；Git 历史负责归档，不创建大型 `legacy/`。
- □ 删除生成物和缓存，不跟踪 `__pycache__`、`.pyc`、`.pytest_cache`、临时 HOME、日志或 editable-install metadata。
- □ 更新 exports、CLI help、README、docs index 和 `.gitignore`，不留旧路径。

### 0D：清理后验收

- □ Generic runtime、Hermes/PAST smoke、lifecycle replay、revocation、restart 和 rollback 与 baseline 等价。
- □ Tracked imports 可解析，CLI help 可执行，配置引用文件存在。
- □ `rg` 不存在指向已删除模块、脚本、配置和文档的路径。
- □ Generic corpus 和 `STOP_NO_SIGNAL` evidence 仍可读；停止入口必须明确不可调用。
- □ 记录删除文件数、净减少行数、保留测试数和公共接口变化。
- □ 全量验收通过，工作树 clean，清理提交可独立 review/revert。

## 4. 阶段 1：研究协议冻结

阶段 1 先明确“测什么、怎么分、结果能说明什么”，避免看到结果后修改分类和指标。

### 1A：Memory Taxonomy Contract

- □ 实现版本化 `MemoryKind` 和 `MemoryUnitDescriptor`。
- □ Descriptor 包含 kind、content schema、scope、source provenance、temporal identity、applicability、version 和 owner method。
- □ 支持 `primary_kind + secondary_kind/transform`，但主表实验只能指定一个 target kind。
- □ Feedback、Q-value、quality、policy state 与 memory content 分离建模。
- □ 为事实、具体 episode、SOP、混合 profile 和 condition-strategy rule 建立正反 fixture。

### 1B：Lifecycle Surface Contract

- □ 实现版本化六类 `MemoryLifecycleSurface`。
- □ Event 声明 producer、owner、memory kind、surface、input/output IDs、revision、cutoff 和 plane。
- □ Method descriptor 声明读取、修改和仅观察哪些 surface。
- □ 非方法拥有的 surface 失败不得给该 updater 错误 credit。
- □ 旧 extraction operation 只映射为 Construction，不伪造其他 surface。

### 1C：PAST Family Applicability Matrix

PAST 当前 26 个 family 预注册如下：

| Target | Families | 数量 | 用途 |
| --- | --- | --- | --- |
| Semantic | `SM01` preference、`SM02` constraint、`SM03` correction、`SM04` migration、`SM05` weak trigger、`SM06` exception pollution、`SM07` scoped migration | 7 | semantic sensitivity |
| Episodic | `EP01` prior-case、`EP02` exception-list、`EP03` recall-then-modify | 3 | episodic sensitivity |
| Procedural | `PC01_sop_bootstrap_01..06`、`PC02_sop_patch_01..02`、`PC03_latent_rule_induction_01`、`PC04_failure_to_rule_01` | 10 | procedural sensitivity |
| Auxiliary | `PG01..PG06` proactive information gathering | 6 | process-feedback 辅助分析 |

- □ 记录每个 family 的 task sequence、stages、metric、memory opportunity、target kind 和 confounders。
- □ Family ID 只用于 audit、split 和报告，不进入 method-visible payload。
- □ Inclusion/exclusion 在运行前冻结，不能根据分数改变。
- □ SM、EP、PC 分 panel 报告，不直接平均异构原始分数。

### 1D：比较层级与隔离

```text
Level 0: Vanilla Hermes / no cross-task persistence
Level 1: Hermes native static memory
Sensitivity: type-matched oracle, shortcut/current-input, wrong-mechanism
Level 2: Existing method                 [阶段 4，延后]
Level 3: Existing method w/ RSIMem       [阶段 6，延后]
```

- □ 每次只允许一个 target kind 改变，其他两类冻结。
- □ 每个 condition 使用独立 state directory，禁止跨 condition 污染。
- □ Learn/validation/final 按 task-template/family group 隔离。
- □ 固定 model、provider、wrapper、tool budget、turns、sampling 和 retry。
- □ Oracle 只用于 sensitivity 上界，不进入 updater corpus。

### 1E：指标与解释规则

Primary：PAST official metric 和 paired delta，按三类 panel 报告。

Mechanism：formation/retention coverage、retrieval、exposure、attributable use、unknown use、correct/harmful update、abstention、negative transfer 和 surface failure distribution。

- □ 正式运行前冻结 practical improvement threshold、replicates 和 paired statistical procedure。
- □ 总分变化不能单独证明目标 surface 改进。
- □ 无提升不能直接判方法失败，必须先验证 oracle、activation 和 exposure。
- □ Unknown usage 单独报告，不并入 useful/harmful。
- □ Raw resource vector 只报告，不作为 learning reward。

### 1F：阶段 1 完成条件

- □ Taxonomy、surface、family matrix、comparison、split 和指标已版本化冻结。
- □ 三类 fixture 和混合系统反向 fixture 通过。
- □ 26 个 family 均有 target/auxiliary/excluded 身份和理由。
- □ Protocol manifest 在首个 run 前生成 digest，后续修改创建新版本。
- □ 论文 abstract/introduction/background 与该协议一致，不再以 extraction prompt 为主线。

## 5. 阶段 2：通用实验架构

阶段 2 让 benchmark、host、method 和 feedback 独立替换。真实 AdaMem、MemQ、Recuris 在阶段 4 接入；本阶段使用 deterministic fake adapters。

### 2A：BenchmarkAdapter

职责：枚举 case/split、重置和推进环境、提供 public capability schema、在 final plane 评分、生成 audit-only annotation。

- □ 定义 host-neutral request/response/event contract，不引用 Hermes 类。
- □ 将 PAST loader、family/stage、environment 和 grader 封装到 PAST adapter。
- □ 区分 public task state、audit-only contract 和 final-only score。
- □ 保持 raw prompt、grader 和 reference 不变。
- □ Hidden answer、grader field 或 family-derived key 进入 Host/method 时 fail closed。
- □ Adapter 前后环境 transition 和 official score 等价。

### 2B：HostAdapter

职责：Hermes session、模型、工具、native memory、context、usage、restart 和 state isolation。

- □ 定义 `HostCapabilities`，声明 memory surface、tool closure、usage、restart 和 snapshot。
- □ Hermes payload 转 canonical event，原生内容不进入 content-free ledger。
- □ BenchmarkAdapter 不读取 Hermes state，MethodAdapter 不调用 PAST grader。
- □ Native bypass、method-managed 和 no-persistence 有独立 identity。
- □ 缺失/重复/跨 session tool result 和 restart drift fail closed 或 censored。

### 2C：MemoryMethodAdapter

统一接口：

```text
describe_capabilities
prepare_run
start_episode
observe_event
finalize_episode
snapshot_state
propose_update
validate_update
activate_update
rollback_update
```

- □ Descriptor 声明 primary/secondary kind、owned surfaces、required feedback、Host capabilities、state schema、lineage、online update、validation 和 rollback。
- □ Method state 与 memory content 分离。
- □ Update 声明 target surface、affected artifacts、base revision、cutoff 和 expected behavior change。
- □ Unsupported capability 明确返回 unsupported，不静默改算法。
- □ Semantic fake method 不得修改 episodic/procedural state。
- □ Method 不得读取 final score、hidden expectation 或 cutoff 后 evidence。
- □ Stale revision、duplicate activation 和 invalid rollback fail closed。

### 2D：FeedbackCondition

| ID | Condition | Updater 可见信息 |
| --- | --- | --- |
| F0 | Frozen | 无反馈 |
| F1 | Terminal outcome | 部署可见 success/failure/outcome |
| F2 | Unstructured trajectory | F1 加完整可见轨迹 |
| F3 | Structured lifecycle | F2 加 canonical memory events |
| F4 | Artifact-grounded | F3 加 exact provenance/use/outcome joins |
| F5 | Counterfactual | F4 加预注册 replay/intervention |

- □ 每个 condition 通过 allowlist 构造 updater view。
- □ 低 condition 不能经 nested metadata、raw payload 或 pointer 读取高 condition。
- □ Feedback artifact 记录 condition、schema、cutoff、plane 和 digest。
- □ F5 只使用 train/development cases，不反复查询 final held-out。

### 2E：Canonical Evidence 与 Ownership

- □ Event 覆盖 candidate、constructed memory、admission、commit、retrieval、exposure、use、tool、outcome、proposal 和 activation。
- □ Event 包含 run/session/task、kind、surface、owner、revision、parents、cutoff 和 plane。
- □ Exposure、behavioral consistency 和 attributable use 严格区分。
- □ Native/其他 method 可以阻止 false attribution，但不能替目标 method 获得 credit。
- □ Semantic set、episode provenance 和 skill invocation 复用统一 parent/child contract。

### 2F：拆分 Hermes-PAST 单体

当前 `src/rsimem/hermes_past_bridge.py` 约 3853 行，同时承担 benchmark、Hermes、memory、evidence、feedback 和 report 职责。

- □ PAST task/family/grader 迁入 BenchmarkAdapter。
- □ Hermes session/tool/native memory 迁入 HostAdapter。
- □ Canonical event、corpus 和 attribution join 留在 framework core。
- □ Mem0/extraction path 迁入 semantic fixture/backend。
- □ 旧入口提供兼容层或明确迁移错误，不静默运行旧协议。
- □ 拆分前后用 golden trace 比较 events、outcome、usage 和 state digest。

### 2G：阶段 2 完成条件

- □ 四个边界有 typed contract、capability descriptor 和反向测试。
- □ Fake semantic/episodic/procedural methods 可在同一 PAST/Hermes harness 独立运行。
- □ F0-F5 有字段 allowlist 和 contamination tests。
- □ Grader 无法经 Host、method、metadata 或 pointer 泄漏。
- □ Hermes-PAST bridge 完成职责拆分且 golden trace 等价。
- □ Restart、revision、idempotency、rollback、secret scan 和 telemetry 验收通过。
- □ 本阶段不要求真实三种方法，不用 fake method 分数声明效果。

## 6. 阶段 3：PAST Memory Sensitivity

阶段 3 回答 PAST 的 SM、EP、PC 是否能稳定区分“没有该类 memory”和“拥有正确该类 memory”。只有通过该 gate，后续方法无提升才可解释。

### 3A：Family Eligibility

- □ 验证 26 个 family 的 learn -> persistence -> future opportunity -> evaluation 时间顺序。
- □ Future input 完整重述目标 memory 时标记 `current_input_confounded`。
- □ Target memory 必须存在可干预的结果路径，不能只影响无关格式。
- □ Control 必须真正隔离 persistence、shortcut 和 wrong mechanism。
- □ Split 不共享答案、特定值、文件名或可直接记忆的 surface token。
- □ Exclusion 在结果前冻结并记录理由。

### 3B：五个 Sensitivity Conditions

| Condition | 目的 |
| --- | --- |
| `no_persistence` | 无跨任务 memory 下界 |
| `native_static` | Hermes 原生静态 memory |
| `type_matched_oracle` | 正确目标类型 memory 的可达上界 |
| `shortcut/current_input` | 检查是否绕过 persistence |
| `wrong_mechanism` | 检查是否只是额外文本带来提升 |

Oracle 要求：

- □ 在 run 前由 audit plane 生成、冻结并标记 `oracle_only`。
- □ 只含目标机制需要的最小信息，不含标准答案、grader 指令和输出模板。
- □ Semantic oracle 是事实/偏好/规则；episodic oracle 保留 episode identity；procedural oracle 是 SOP/skill。
- □ Oracle 不进入 pure corpus、method updater、retrieval learning 或后续 proposal。

### 3C：Matched Execution

- □ 相同 task、seed、model、provider、budget、tool environment 和 retry 下配对。
- □ 每个 condition 使用全新 state directory，并验证无残留 memory。
- □ Replicates、failure exclusion 和 incomplete usage 规则预先冻结。
- □ 记录 source、state、retrieval、exposure、tool closure、outcome 和 final plane。
- □ Provider/infrastructure failure 不进入质量 denominator，但保留 attempts audit。
- □ 平衡或随机化执行顺序，避免 condition 与时间/provider drift 固定相关。

### 3D：Type-Isolated Panels

Semantic：分 preference、constraint、correction、migration、pollution；验证 scope/validity/correction；固定 episodic 和 procedural。

Episodic：分 prior-case、exception-list、recall-then-modify；验证 context/outcome/provenance；固定 semantic 和 procedural。

Procedural：分 SOP bootstrap、patch、latent rule、failure-to-rule；验证 skill selection/invocation；固定 semantic 和 episodic。

- □ 三个 panel 均报告 official metric、paired delta、oracle coverage、retrieval、exposure、use 和 unknown use。
- □ 不把三个 panel 的异构 raw metric 直接平均。

### 3E：Sensitivity Gate

每个类型独立给出：

- `SENSITIVE`：oracle 相对 no-persistence 达到预注册 practical improvement，replicate 一致，shortcut/wrong-mechanism 不能解释主要提升。
- `PARTIALLY_SENSITIVE`：只有部分预注册 families 通过，后续只在通过 family 上运行方法。
- `INSENSITIVE`：oracle 无稳定改善，不能用该 panel 否定 memory method。
- `INVALID`：环境、泄漏、adapter 或 control 失败，修复后重新预注册。

- □ 报告 family-level paired delta、replicate variation 和 oracle coverage。
- □ 区分 dataset insensitivity、memory 未形成、未检索、未曝光和 Agent non-use。
- □ 报告 shortcut/wrong-mechanism，排除额外 token 解释。
- □ `INSENSITIVE` 作为合法负结果保留，不降低 gate 制造 sensitivity。

### 3F：阶段 3 决策出口

- 三类通过：阶段 4 接入 AdaMem、MemQ、Recuris，运行 type-matched diagonal experiments。
- 部分通过：PAST 继续作为共同主数据集，主 claim 限于通过类型；未通过类型寻找外部 benchmark。
- 仅 procedural 通过：PAST 作为 procedural 主数据集，semantic/episodic 使用专用 benchmark。
- 全部未通过：停止方法接入，检查 Hermes adapter、oracle 和 PAST suitability。

### 3G：阶段 3 完成条件

- □ 26 个 family 的 eligibility 和 inclusion 在结果前冻结。
- □ 三个 panel 完成五个 conditions 或有预注册不可运行理由。
- □ 每个 panel 有明确 sensitivity 状态。
- □ Oracle 与 method-visible evidence 隔离，final score 未进入 updater。
- □ Manifest、trace、state、attempt audit 和 report 可重建全部结论。
- □ 根据结果另写阶段 4 至阶段 7 checklist，不在本文提前标记后续完成。

## 7. 阶段 4 至阶段 7 的暂定方向

以下只记录方向，不是当前执行任务：

- 延后阶段 4：忠实接入 AdaMem、MemQ、Recuris，先做 native fidelity smoke，再做 PAST 对角实验。
- 延后阶段 5：同一方法内控制 F0-F5，研究 feedback-sufficiency frontier。
- 延后阶段 6：构造 `Original Method + RSIMem Feedback Adapter + Validation Gate`，运行 equal-compute、oracle 和 held-out controls。
- 延后阶段 7：根据结果选择外部 benchmark，完成跨 benchmark 验证和论文报告。

阶段 3 前不实现方法专属 adapter、不承诺 SOTA、不选择 external benchmark，也不把旧 extraction N+1 当作 `w/ RSIMem`。

## 8. 当前状态

截至 2026-09-01：

- √ PAST/Hermes runtime、usage、generic memory contracts、lifecycle、provenance、安全和 rollback 可复用。
- √ Pure process、benchmark audit 和 final evaluation 已隔离。
- √ SM02/SM05 clean-parent 已完成，均为 `STOP_NO_SIGNAL`。
- √ Extraction-only 主线在研究决策上停止，不再生成 N+1。
- 部分完成：三类 storage-boundary 和六 surfaces 有 fixture，但未按新 taxonomy/ownership 验收。
- 部分完成：PAST/Hermes 可运行，但仍集中在大型 `hermes_past_bridge.py`。
- √ 阶段 0A baseline 冻结和 0B 逐文件资产分类已完成；`baseline_preflight`
  可在 cleanup 前 fail closed。
- □ 阶段 0C/0D 尚未执行删除、收敛和清理后等价验收。
- □ 阶段 1 未生成冻结 protocol manifest 和完整 family matrix。
- □ 阶段 2 未完成四个 adapter contracts 和单体拆分。
- □ 阶段 3 未运行三类 oracle sensitivity matrix。
- □ AdaMem、MemQ、Recuris 未接入；这是阶段 4，不是当前缺陷。

## 9. 标准验收命令

RSIMem 根目录：

```bash
PYTHONPATH=src .venv/bin/python -m rsimem.baseline --manifest docs/baseline_manifest_20260901.json --repo-root .
.venv/bin/python -m pytest -q tests
.venv/bin/python -m compileall -q src tests
.venv/bin/python -m pip check
.venv/bin/python -m rsimem.secret_scan
git diff --check
bash -n scripts/*.sh
```

Vendored PAST-Bench：

```bash
../../.venv/bin/pytest -q
```

每阶段还要求 clean tree、isolated temporary HOME、restart fixture、tracked-secret scan、schema/revision/revocation/state isolation 验收。正式 provider batch 前运行 bounded completion probe，但 probe 不进入质量统计。不要从 RSIMem 根目录直接运行 `pytest benchmarks/past-bench`，避免错误解析 Hermes-plus 顶层 `agent` package。

## 10. 总体验收原则

1. **先证明 benchmark 需要 memory，再评价 method。** Oracle 不工作时，方法无提升不能解释为方法失败。
2. **先证明更新发生并被使用，再解释分数。** Task delta 不能单独证明目标 surface 改进。
3. **分类依据 memory content。** 混合系统必须声明 primary target 和 transform。
4. **六个 surfaces 用于诊断，不要求联合优化。** 方法只对拥有的 surface 负责。
5. **负结果合法。** `STOP_NO_SIGNAL`、`INSENSITIVE`、`unresolved` 和 abstention 不得被改写为正标签。
6. **Benchmark knowledge 与 method feedback 隔离。** Family ID、grader、答案、oracle 和 official score 不进入 updater。
7. **后续由前三阶段决定。** 阶段 3 完成前不提前实现或宣称阶段 4 至阶段 7。
