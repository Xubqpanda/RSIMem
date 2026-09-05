# RSIMem PAST-Bench Research Checklist

最后更新：2026-09-05

## 0. 研究目标和执行原则

RSIMem 当前不是先设计一个更强的 extraction prompt，而是分析 Memory 如何实现可靠的 recursive self-improvement。核心问题是：

> 在 Memory 自进化过程中，什么类型、什么粒度的反馈，足以把真实执行中的失败归因连接到可验证的 Memory 改进方向？

当前实验必须围绕 native static 的真实缺陷展开，而不是通过删除组件观察分数下降。执行逻辑固定为：

```text
native static execution
    -> lifecycle evidence
    -> failure attribution
    -> one-axis repair
    -> same downstream task
    -> paired headroom
    -> feedback-to-update experiment
```

本 checklist 是新的主执行协议，旧版本中的五条件 sensitivity、shortcut 和 wrong-mechanism 不再是当前主线。旧结果只作为历史 pilot 保留，不得继续据此写出 Memory 表示或 prompt 注入的因果结论。

执行原则：

- 先修复状态隔离和 evidence 完整性，再跑正式结果。
- 能并行的 family、replicate、trace audit 和 case review 尽早并行。
- 共享资源必须隔离：每个 run 使用独立 port、HOME、state directory、trace directory 和 service fixture。
- 不允许多个实验复用一个已启动但 fixture 不明的 mock service。
- 所有实验先生成 immutable manifest，再启动 provider batch。
- 正式质量统计排除 provider、服务和 usage infrastructure failure，但保留 attempt audit。
- 不以最终分数单独决定 Memory failure；`non_memory_failure` 和 `unresolved` 是合法结果。

## 1. 固定研究边界

### 1.1 当前范围

- Benchmark：vendored PAST-Bench。
- Host：Hermes。
- Memory：Semantic、Episodic、Procedural 三类同级对象；feedback、policy 和 context 不是第四类 Memory。
- 基线：Hermes `native_static`，另保留 `no_persistence` 作为背景下界。
- 基础模型参数冻结；当前研究更新外部 Memory、Memory policy 或 self-improvement state。
- 当前论文先使用 Hermes 一个 host；其他 host 属于后续泛化实验。
- cost、token、latency、storage 和 API call 只做报告字段，不能作为当前 Memory updater 的 reward。

### 1.2 Lifecycle vocabulary

```text
trigger/source
    -> formation
    -> persistence
    -> maintenance
    -> retrieval
    -> exposure/application
    -> downstream action/outcome
```

| Surface | 需要回答的问题 |
| --- | --- |
| `trigger/source` | 当前交互或历史证据是否提供了形成 Memory 的机会 |
| `formation` | 是否形成了正确、完整、范围合适的 Memory candidate |
| `persistence` | candidate 是否成功提交，并在后续 session 可见 |
| `maintenance` | 是否正确处理更新、冲突、过期、撤销和污染 |
| `retrieval` | 是否在需要时检索，并选中正确 Memory |
| `exposure/application` | 正确 Memory 是否进入模型可用上下文并影响行为 |
| `downstream` | 工具、模型推理或任务执行是否仍然失败 |

这些 surface 是统一诊断坐标，不要求每一种 Memory 或方法都优化全部 surface。

### 1.3 Evidence planes

- `pure_process`：部署时自然可见的 context、Memory event、retrieval、injection、tool call/result、用户反馈和状态变化。
- `benchmark_audit`：离线使用预注册 task contract、grader 和 reference 检查结果，仅用于审计。
- `final_evaluation`：实验结束后读取 official score 和 hidden evaluation。

`benchmark_audit` 和 `final_evaluation` 的字段不得进入 Agent、Memory updater 或 policy optimizer。所有 evidence 带有 `plane`、`source`、`cutoff`、`revision` 和 digest。

## 2. 阶段 0：重置协议、修复隔离和清理旧主线

阶段 0 是所有后续实验的前置条件。该阶段可以将审计任务并行，但协议变更和正式 batch 必须串行冻结。

### 0A. 串行冻结 clean baseline

- [x] 记录 RSIMem commit、Python/依赖、Hermes commit、PAST identity、provider 配置 schema 和当前 CLI。
- [x] 运行现有测试、compileall、pip check、secret scan、shell syntax 和 `git diff --check`。
- [x] 生成新的 `baseline_manifest`，固定 source digest、test result、fixture digest 和 runner version。
- [x] 记录当前五条件 pilot 为 `historical_exploratory_only`，不再作为新协议的质量数据。
- [x] 写入新的 protocol ID，例如 `native-attribution-repair-v1`。

### 0B. 并行修复运行隔离

以下任务可并行开发，但必须共享同一份 contract test：

- [x] Service isolation：每个 task/run 使用独立 port 或独立 service process；健康检查必须校验 fixture identity/digest，不能只校验 HTTP 200。
- [x] Service lifecycle：task 结束时停止本 run 创建的 service；外部已有进程若 fixture 不匹配必须拒绝复用。
- [x] State isolation：每个 `family x replicate x condition` 使用独立 Hermes HOME、state、session、artifact 和 trace 目录。
- [x] Anchor isolation：所有 repair 从同一个 immutable native post-learn anchor 派生，repair 之间互不写回。
- [x] Provider scheduling：支持 bounded concurrency、重试上限、429/5xx 分类和按 run 的 usage 完整性检查。
- [x] Manifest：记录 run、task、family、replicate、port、fixture digest、home digest、model、provider、seed 和 protocol ID。
- [x] Failure handling：provider/service/usage 失败标记为 infrastructure attempt，不进入 task quality denominator。

### 0C. 并行清理和迁移

- [x] 审计 extraction-only launcher、proposal CLI、旧 prompt optimizer、旧 shortcut/wrong-mechanism fixture 和重复 report。
- [x] 通用 lifecycle、provenance、revision、idempotency、rollback、evidence-plane 和 usage accounting 必须保留。
- [x] 旧 extraction API 若仍被通用代码依赖，改成 method/surface-neutral interface；不能直接删除调用者。（当前 native 主线无该依赖；旧 API 仅保留为 legacy method fixture。）
- [x] dataset、grader、原始 fixture 和历史 negative evidence 不做格式重写。
- [x] 生成物、缓存和 provider secrets 不进入 tracked source。
- [x] 更新 `progress.md`，把旧阶段标记为 superseded，把新三阶段主线写清楚。

### 0D. 阶段 0 验收

- [x] 连续启动两个使用不同 notes fixture 的 task，第二个 task 不能读到第一个 task 的 note。
- [x] 并发启动多个同类 task，所有 audit 中的 service fixture digest 与 manifest 一致。
- [x] 任意 run 可以从 manifest 重建 trace、state、fixture 和 provider usage。
- [x] clean baseline、隔离 contract tests 和完整 smoke 通过后，才允许正式运行 Analysis 2。

## 3. 阶段 1：Analysis 2，Native Failure Attribution

这里虽然在论文中称为 Analysis 2，但工程上先执行。目标是分析 native static 的自然失败，而不是人为删除组件。

### 1A. 统一 observation contract

- [x] 为每个 native episode 记录 lifecycle event：source、candidate、formation、admission、commit、maintenance、retrieval、exposure、application、tool 和 outcome。
- [x] 每条 Memory event 带 `event_id`、`owner`、`memory_kind`、`surface`、`input_ids`、`output_ids`、`revision`、`parents`、`cutoff`、`plane` 和 digest。
- [x] 记录 Memory state before/after 的 digest、entry identity、scope、validity、provenance、revision 和 commit status。
- [x] 记录 retrieval query、candidate IDs、selected ID、injection status、injection position 和 application surface；query digest 与 artifact ID 集合经过严格校验，且不保存原文。
- [x] 记录 tool calls/results、service audit、最终输出和 task component score；score 只进入离线 audit。
- [x] 缺失的事件标记为 `not_observed`，不能默认为“没有发生”。

### 1B. 并行 trace extraction

按 panel 并行处理，每个 worker 只读 immutable trace：

仓库现在提供 `rsimem-assemble-native-attribution`，按 immutable manifest
逐 run 执行 native audit，并将失败 run 只记录为 content-free exclusion；它
不会把 provider failure、grader 或最终输出复制进 attribution corpus。可选的
`--audit PATH` 会单独持久化所有 accepted/excluded run identity，即使整批没有
accepted run 也不会丢失 infrastructure attempt。该入口完成了
trace-to-corpus 的确定性接线，但不替代下面要求的真实 accepted runs。

- [ ] Semantic：SM01-SM07，重点检查事实、偏好、约束、迁移、过期和 scope。
- [ ] Episodic：EP01-EP03，重点检查事件、上下文、outcome、provenance 和 prior-case recall。
- [ ] Procedural：PC families，重点检查 SOP/skill formation、version、activation 和 invocation。
- [ ] PG：PG01-PG06 作为 retrieval-centric 辅助 family，重点检查何时意识到需要历史并主动检索。
- [ ] 每个 panel 至少并行抽取多个 replicate 和成功/失败样本，避免先做完一个 family 才开始下一个。

### 1C. Failure taxonomy

每个 case 只有一个 primary failure surface，可有多个 secondary observation：

| Label | 判定规则 |
| --- | --- |
| `formation_missing` | 应形成 candidate，但没有 candidate |
| `formation_incorrect` | candidate 存在，但内容、scope、provenance 或 procedure 不正确 |
| `persistence_failed` | candidate 正确，但未提交、版本不一致或后续不可见 |
| `maintenance_stale` | 旧、过期或撤销内容仍被视为有效 |
| `maintenance_conflict` | 冲突 entry 共存且没有正确消解 |
| `maintenance_pollution` | 一次性、无关或错误内容被持久化 |
| `retrieval_missed` | 正确 Memory 存在，但没有触发、召回或注入 |
| `retrieval_wrong` | 检索发生，但选中错误、过期或无关 Memory |
| `application_ignored` | 正确 Memory 已暴露，但没有影响后续行为 |
| `non_memory_failure` | 主要问题来自工具、服务、任务理解或通用推理 |
| `unresolved` | 现有 process evidence 不足以唯一归因 |

判定顺序为：formation -> persistence/maintenance -> retrieval -> application -> non-memory。不能仅凭低分给 Memory 归因。

### 1D. 并行 case review

- [x] 自动规则先生成候选 attribution，不直接生成最终标签。
- [x] reviewer packet/record 独立于冻结 corpus 持久化，且只包含 ID、标签和 evidence refs；review 结果不会覆盖 deterministic candidate。
- [x] reviewer record 在写入前绑定 frozen corpus、candidate ID 和 evidence-ref allowlist；未知 candidate/evidence 会 fail closed。
- [x] review store 支持 canonical reload，并可统计 review coverage、reviewer 分歧和 adjudication 覆盖；未达到双 reviewer 门槛时不生成最终标签。
- [x] candidate/observation 绑定 manifest 的 `replicate_id`；历史 corpus 缺失该字段时显式报告为 `unknown`，不从 case ID 猜测。
- [ ] 至少两名 reviewer 独立检查代表性 case；分歧 case 进入 adjudication。
- [x] 每个标签引用 event ID、artifact digest、revision、tool index 或 snapshot digest；无法形成稳定引用的 case 保留为 unresolved。
- [x] 每个 case 生成 `candidate_repair_axis`，若不能唯一映射则标记 `is_actionable=false`。
- [x] hidden answer、future-test answer、grader instruction 和 official score 不得出现在 updater view；content-free corpus/review packet 有对应污染测试。

输出 schema：

```text
case_id
family_id
replicate_id
memory_kind
primary_failure_surface
secondary_observations
evidence_refs
confidence
candidate_repair_axis
is_actionable
review_status
```

### 1E. Analysis 2 指标和验收

- [x] `attribution_coverage`、`unresolved_rate`、`evidence_completeness`；由 `rsimem-report-native-attribution` 从冻结 corpus 重建。
- [x] 各 surface 的 case 数、比例、panel 分布和 family 分布；由 frozen-corpus report 重建。
- [x] `cross_family_consistency` 和 `non_memory_exclusion_rate` 已可计算；前者是描述性分组指标，不替代 reviewer/adjudication 的归因准确率。
- [x] `actionability_rate`：能否映射到唯一 repair axis；当前 slice 为 `0.0`，因此不开放 repair。
- [x] 报告成功 case、明确 Memory failure、non-memory failure 和 unresolved case 的实例；完整 failure taxonomy 对未出现类别显式报告为零。
- [ ] 至少完成一个跨 SM/EP/PC/PG 的 audit slice，再决定是否扩大到全量。（当前 SM02/PC01 已完成；EP01/PG01 因 usage-incomplete 排除，不能宣称跨四 panel。）
- [x] Analysis 2 的输出冻结成版本化 attribution corpus，作为 Analysis 1 的唯一输入；当前 slice 为 `10 observations / 10 unresolved / 0 actionable`，因此触发 `STOP_NO_ACTIONABLE_SIGNAL`，不进入 repair。
- [x] Stage 1 -> Stage 2 gate 已确定性实现：必须同时具备 high-confidence actionable candidate、合法 repair axis、双 reviewer case、通过 event/evidence contract 校验和非 unresolved-only corpus；`not_observed` 是合法自然执行状态，不被误判为证据缺失。
- [x] report CLI 支持 `--review-store`，会重新校验 reviewer records 并将双 reviewer 覆盖绑定到 Stage 2 gate；单 reviewer 或半覆盖不会开放 repair。

## 4. 阶段 2：Analysis 1，Native Improvement Headroom

本阶段验证 Analysis 2 找到的 failure 是否确实对应 native static 的可优化空间。它不做删除式 ablation，而做同一任务上的局部反事实 repair。

### 2A. 串行选择 repair cases

仓库提供 `select_native_repair_cases()` 与 corpus-bound case-list payload。
它只接受 high-confidence、唯一合法 repair axis、完整 evidence 且完成双
reviewer 覆盖的 attribution；不会读取 score 或生成 oracle 内容。当前
`STOP_NO_ACTIONABLE_SIGNAL` corpus 因没有 eligible case，无法进入该入口。
case list 已支持 canonical reload 和 append-once store，便于后续在 gate
开启后冻结 `case list / repair axis / reference boundary`。

- [ ] 只选择 `confidence` 足够高、evidence 完整、`is_actionable=true` 的 case。
- [ ] 按 failure surface 和 Memory kind 分桶，避免只选择最容易的 semantic case。
- [ ] 预先冻结 case list、repair axis、reference source、allowed changes 和 expected behavior。
- [ ] unresolved 和 non-memory case 保留为 negative control，但不强行做 Memory repair。

### 2B. 并行实现局部 oracle repair

每个 repair 从相同 native anchor 派生，不能改变其他 lifecycle surface：

| Repair | 只允许改变 | 适用问题 |
| --- | --- | --- |
| `oracle_formation` | candidate 内容、完整性、scope 或 provenance | 漏写、错写、错误抽取 |
| `oracle_persistence` | commit、revision、跨 session 可见性 | 正确 candidate 未持久化 |
| `oracle_maintenance` | stale、冲突、污染和错误覆盖状态 | 保存后状态错误 |
| `oracle_retrieval` | trigger、query、ranking、target selection 或 retrieval timing | 未召回或召回错误 |
| `oracle_application` | 正确 entry 的 exposure/application surface | 已召回但 Agent 未使用 |

每个 repair 必须声明：

- `base_native_state_digest`；
- `repair_axis`；
- `repair_payload_digest`；
- changed artifact IDs；
- explicitly unchanged artifact IDs；
- expected behavior change；
- rollback path。

### 2C. 并行运行 matched validation

- [ ] `native_static` 和每个 oracle repair 使用完全相同的 evaluation task、fixture、工具、模型、预算、grader 和 seed。
- [ ] 每个 `case x repair x replicate` 使用独立 service process、port、HOME 和 trace directory。
- [ ] repair 不得读取 evaluation answer、hidden grader 或正式 task score。
- [ ] 先运行 bounded smoke，再并行运行各 repair bucket 的正式 replicate。
- [ ] condition 顺序随机化或平衡化，避免 provider drift 与某个 repair 固定相关。
- [ ] infrastructure failure 单独重试；质量分母只包含 accepted runs。

### 2D. Headroom 指标

核心是同一 native case 的 paired improvement：

```text
formation_headroom   = score(oracle_formation)   - score(native_static)
persistence_headroom = score(oracle_persistence) - score(native_static)
maintenance_headroom = score(oracle_maintenance) - score(native_static)
retrieval_headroom   = score(oracle_retrieval)   - score(native_static)
application_headroom = score(oracle_application) - score(native_static)
```

同时报告：

- task score 和 completion/robustness/safety 等 component delta；
- case-level repair success rate；
- content coverage、stale/conflict contamination、target retrieval hit、wrong hit；
- exposure/application use rate；
- regression、side effect 和 non-memory residual failure。

只有当某个 repair 在对应 attribution bucket 上稳定提升，才认定该 surface 是 native static 的可优化维度。oracle 无提升同样是结果：说明 attribution 不完整、该 surface 不是主要瓶颈，或失败不属于 Memory policy。

### 2E. Reference state 边界

- [ ] formation/maintenance reference 只来自公开 learn/update input、公开 task contract 和预注册 schema。
- [ ] 不使用 evaluation answer、hidden expectation、future-test content 或 official score 编写 reference。
- [ ] reference 保存结构化 keys、scope、validity、provenance、retrieval target 和 allowed changes，不只保存无身份文本。
- [ ] oracle artifact 只进入 benchmark audit plane，不进入 pure process corpus 或 updater。

### 2F. Analysis 1 验收

- [ ] 每个 repair case 能追溯到 Analysis 2 的 attribution record。
- [ ] 每个 repair 只有一个 declared changed axis。
- [ ] native 与 repair 的 downstream task 完全 matched。
- [ ] 报告 paired delta、case distribution、replicate variation 和 regression。
- [ ] 生成 native improvement map：surface、failure count、headroom、confidence、适用 Memory kind。
- [ ] 只将 evidence-supported 且 repair 有真实提升的 surface 交给下一阶段。

## 5. 阶段 3：Feedback-to-Repair Sufficiency

前三阶段中的“阶段 3”是方法验证阶段，只有 Analysis 2/1 完成后才开始。目标是验证不同粒度的部署可见反馈，能否自动完成“归因 -> 修复目标”的连接。

### 3A. 并行构造反馈条件

所有条件使用相同 native cases、同一个 meta-agent、相同 update budget 和 validation gate：

| Condition | Meta-Agent 可见内容 |
| --- | --- |
| `F0_terminal` | 终态成功/失败和部署可见 outcome |
| `F1_trajectory` | F0 加完整可见对话和 tool trace |
| `F2_lifecycle` | F1 加 canonical lifecycle events |
| `F3_artifact_grounded` | F2 加 Memory artifact、revision、provenance 和 use join |
| `F4_counterfactual` | F3 加预注册 replay/intervention observation |

F4 是诊断上界，不代表真实部署反馈。每种 feedback view 使用 field allowlist、cutoff、plane 和 digest，做 contamination test。

### 3B. 并行运行 proposal

- [ ] 每个 feedback condition 在独立 state 上生成 candidate update。
- [ ] candidate update 必须声明 target surface、affected artifacts、base revision、expected behavior 和 rollback。
- [ ] updater 不得直接读取 final score、hidden label 或 future-test result。
- [ ] 同一 native case 的不同 feedback condition 不能共享已激活的 update。
- [ ] update proposal、validation 和 activation 分离记录，防止“生成即成功”。

### 3C. 指标和验收

- [ ] actionable diagnosis rate；
- [ ] surface attribution accuracy/consistency；
- [ ] candidate acceptance rate；
- [ ] 真实 `N+1` task gain 和 held-out gain；
- [ ] regression、harmful update、rollback 和 abstention rate；
- [ ] unresolved/censored rate；
- [ ] meta-agent input tokens、调用次数和 latency 作为报告字段。

若结构化 process evidence 相比 terminal-only feedback 能稳定生成正确 repair，并在真实 `N+1` 上提升，才说明 feedback 粒度能够连接错误归因和优化方向。

## 6. 并行执行矩阵

### 可以并行的工作

- service/state isolation、manifest、trace schema、fixture audit、usage audit；
- SM、EP、PC、PG 四个 panel 的 native trace extraction；
- 不同 family 和 replicate 的 case review；
- 已冻结 attribution 后，不同 repair axis 的 oracle implementation；
- F0-F4 feedback view 的 schema 和 contamination tests；
- report aggregation、paired statistics 和 case visualization。

### 必须串行的工作

- clean baseline -> isolation fix -> formal run；
- protocol freeze -> native execution -> attribution corpus freeze；
- attribution corpus freeze -> repair case selection；
- native anchor freeze -> repair materialization；
- repair result -> feedback sufficiency；
- feedback sufficiency -> multi-round RSI。

### 并行资源规则

- 每个 worker 通过 run manifest 分配唯一 port range。
- 每个 worker 拥有独立 `HOME`、Hermes state、service fixture、trace 和 artifact root。
- provider 并发由全局 semaphore 控制，默认从低并发开始，根据 429/503 和 usage completeness 调整。
- 任一 worker 发现 fixture digest、state digest 或 revision 不匹配，立即 fail closed，不继续计入结果。
- 汇总脚本只读取 accepted immutable manifests，不扫描目录猜测结果。

## 7. 后续阶段，不在当前 checklist 提前实现

前三阶段通过后再决定：

- 接入 Hermes native 之外的 Memory method adapter；
- 在 semantic、episodic、procedural 上运行 method-specific self-improvement；
- 运行多轮 `M0 -> M1 -> M2` 的 recursive improvement；
- 跨 benchmark 或跨 host 泛化；
- feedback compression 和 Meta-Agent cost study。

当前不承诺每个 Memory kind 都有提升，不承诺所有六个 surface 都可优化，也不把一次 oracle repair 结果称为完整 RSI。

## 8. 总体验收出口

- [x] 阶段 0：环境、fixture、state、usage 和 manifest 隔离可信。
- [ ] 阶段 1 / Analysis 2：native failure 有 evidence-backed attribution，无法归因的 case 被保留为 unresolved。
- [ ] 阶段 2 / Analysis 1：局部 repair 能测量 native static 各 surface 的真实 headroom。
- [ ] 阶段 3：不同粒度 feedback 能否把 attribution 映射为可执行 repair 得到实证结论。
- [ ] 所有结论可由 immutable manifest、trace、state digest、audit 和 report 重建。
- [ ] 任何 negative result、abstention、unresolved 和 infrastructure failure 都单独报告，不被改写为 positive evidence。
