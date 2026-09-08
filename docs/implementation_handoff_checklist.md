# RSIMem PAST-Bench: Current Checklist

最后更新：2026-09-08

## 当前目标

当前先不尝试证明完整的 multi-round recursive self-improvement，也不先设计复杂的 failure attribution 方法。先建立可解释的 Memory/RSI 分层 baseline，再回答第一个 RSI 问题：

> 对同一个 Memory RSI updater，完整的、部署可见的 Agent execution trajectory，是否比稀疏的 terminal feedback 更能帮助它更新 Semantic Memory policy，并在匹配的后续 `N+1` 任务上带来可测变化？

RSIMem 后续要研究如何把完整轨迹组织为更有效的 feedback representation；但只有先证明完整轨迹本身存在可用的 RSI signal，才值得继续做这一步。

当前最小闭环：

```text
one static Memory backend executes a train sequence
    -> collect one feedback view
    -> same RSI meta-agent updates or abstains on policy P_n
    -> validate and activate/reject P_{n+1}
    -> run matched N+1 sequence
    -> compare the outcome
```

## 固定边界

- Benchmark：PAST-Bench；host：Hermes；当前只做 Semantic Memory，不宣称其结论覆盖 Episodic 或 Procedural。
- 当前先接 `Mem0 adapted to Hermes` static backend，再接 `Mem0 + AdaMem adapted to Hermes`；Hermes native semantic memory 保留为独立 base-memory baseline，并在 Mem0 路径跑通后接入相同的 RSI adapter contract。
- 同一比较中固定 meta-agent、base model、temperature、token/update budget、policy update space、train/N+1 split、fixture 和 grader。
- meta-agent 只能读取部署可见的 `pure_process` 信息。hidden answer、grader、official score 与 future-test 内容不得进入 meta-agent、analyst 或 updater。
- token、API call、latency、storage 只做报告指标，不作为 updater reward。
- PG 是 `proactive_information_gathering` 任务族，测试主动检索/触发，不是第四类 Memory；当前不作为 gate，也不进入第一版 Semantic extraction-policy 主实验。

### Baseline 分层

当前论文的对照按能力分层，而不是把所有方法放在一个平面上：

| 层级 | 条件 | 当前状态 |
| --- | --- | --- |
| L0 | `NoMemory` | 需要重新按当前 manifest 跑一次 |
| L1 | `HermesNative`、`Mem0Static` | 先跑 Mem0Static；HermesNative 作为同协议 base-memory baseline |
| L2 | `Mem0 + AdaMem`、`HermesNative + AdaMem` | 先完成 Mem0 + AdaMem；后者在 adapter contract 稳定后接入 |
| L3 | `Mem0 + AdaMem + RSIMem`、`HermesNative + AdaMem + RSIMem` | 当前不实现；只在 L2 有可信 RSI signal 后开始 |

这里的 `+ RSIMem` 指向 AdaMem 提供更好的 feedback representation，不替 AdaMem 生成 patch，也不改变 AdaMem 的 patch space、validation 或 rollback。每条 backend 路径都要有对应的 static 条件；不能用 Mem0 + AdaMem 的结果替代 Mem0Static，或用 HermesNative 的结果替代 Mem0 的结果。

## Stage -1：清理与冻结文档入口

目标是降低当前仓库的认知负担，不删除研究证据。完成后，只需要阅读本 checklist、当前 progress 和明确链接的 adapter audit，不应从旧的 extraction-only 或 native-attribution 文档推导当前实验路线。

- [x] 审计 `docs/`：逐份标记为 `current`、`historical`、`superseded`、`reference` 或 `generated evidence`。
- [x] 保留并更新三个权威入口：本 checklist、`docs/progress.md` 的当前状态摘要、AdaMem upstream/adaptation audit；它们之间必须互相链接。
- [x] 将旧 extraction-only、five-condition sensitivity、deterministic native-attribution 主线文档移入 `docs/archive/` 或在文件首段加醒目的 `historical/superseded` 状态；保留原文件名、日期和不可变实验结果。历史文档已统一移入 `docs/archive/`，并修复相对链接。
- [x] 删除真正冗余的草稿、重复的计划和无引用的临时说明；不删除 dataset/task/grader 文档、baseline manifest、accepted/excluded attempt、原始 trace 或可复现实验报告。2026-09-08 redundancy audit 未发现可安全删除的文件；无引用文件均为 dated evidence，详见 `document_status_inventory_20260908.md`。
- [x] 更新 `docs/README.md` 或新建 `docs/README.md`：列出当前入口、归档原则和历史证据位置。
- [x] 对移动后的 Markdown 链接运行检查，避免 current docs 指向失效路径。

### Stage -1 验收

- 新加入的开发者可以仅从 `docs/README.md` -> 本 checklist -> `progress.md` 找到当前工作与下一步。
- 当前文档不再把 `STOP_NO_ACTIONABLE_SIGNAL`、四 panel attribution gate 或旧 five-condition matrix 表述为 RSI 实验的前置条件。
- 所有历史结果仍可按原路径或明确的 archive 链接访问。

## Stage 0：冻结最小可信比较协议

当前证据：Stage 0 的 Mem0Static、base-memory smoke、AdaMem adapter smoke 和
SM01 B0/B1/B2 formal batch 已实现并审计；见
`docs/progress.md` 与 `outputs/adamem_formal_20260908/`。历史 formal batch
生成时尚未持久化 meta-agent provider usage，因此 updater raw-token 条目仍
保持未完成，不能用 suffix task usage 代替。

现有隔离和审计基础设施可以复用，但不再围绕 deterministic attribution corpus 扩展开发。先完成同一协议下的 static Memory backend baseline，避免把 backend 更换误判为 RSI 效果。

- [x] 每个 run 的 service/port、Hermes HOME、state、session、artifact、trace 和 fixture 隔离。
- [x] immutable manifest、usage audit、provider/service failure exclusion、lifecycle/provenance/revision 记录。
- [x] historical five-condition sensitivity、shortcut、wrong-mechanism 只保留为 exploratory history，不进入当前质量结论。
- [x] 将 Mem0 的 semantic memory workflow 适配到 Hermes，形成独立的 `Mem0Static` backend：固定实现 source/commit、extraction model/embedding/version、storage root、write path、search path、retrieval injection point 和 state reset。可复用 RSIMem 现有 `mem0_flat`，也可调用官方 Mem0 API；选择必须在 manifest 和 adapter audit 中固定。
- [x] 在 adapter audit 中记录与 AdaMem 上游 Mem0 runtime 的实现差异；差异仅允许位于 Hermes host/storage/retrieval integration，不得改变“policy-controlled extraction -> Mem0-style semantic write -> later retrieval”的机制，也不得复用 AdaMem-Bench story data 或 upstream experiment runner。
- [x] 跑 `NoMemory`、`HermesNative`、`Mem0Static` 的单 family smoke；三个条件共享 PAST task、模型、预算、fixture、grader、state isolation 和 manifest。
- [x] 验证 Mem0Static 的 write/retrieval/injection 真实发生并可审计；它不是把所有历史对话直接塞入 context 的 Full Context shortcut。
- [x] 新建最小 RSI experiment manifest，固定：train sequence、matched N+1 sequence、模型配置、meta-agent prompt、update budget、policy update space、feedback view schema 和 output locations。
- [x] 明确 policy update space：第一版只允许替换 versioned semantic extraction policy；不能同时改 retrieval、host route、backend、task prompt 或模型配置。
- [x] 定义 updater request 的字段 allowlist，并写 contamination test：请求中不能含 grader、official score、hidden answer、future evaluation 内容或 task-specific reference answer。
- [x] 跑 `Mem0Static`、`Mem0 + AdaMem-terminal`、`Mem0 + AdaMem-full-trajectory` 各一次的 end-to-end smoke，确认 AdaMem 条件中只有 feedback view 不同，其他 execution identity 完全一致。

首批条件：

| Condition | RSI updater 可见信息 |
| --- | --- |
| `B0_mem0_static` | Mem0 backend，保持 `P_n`，不运行 AdaMem updater |
| `B1_mem0_adamem_terminal` | Mem0 + AdaMem；updater 仅见 train episode 的部署可见终态 outcome 与用户/环境反馈 |
| `B2_mem0_adamem_full_trajectory` | Mem0 + AdaMem；B1 加完整部署可见对话、工具调用、Memory operation、Memory state 变化和可见输出轨迹 |

### Stage 0 验收

- `NoMemory`、`HermesNative`、`Mem0Static` 的 base-memory comparison 使用相同任务、模型、预算和 manifest。
- B0/B1/B2 使用相同的 train/N+1 split、模型、预算和 AdaMem update space。
- 每个 run 可以从 manifest 重建，并能校验 fixture/state/trace/usage identity。
- meta-agent request 审计通过，无评测泄漏。
- 任何 provider/service/usage 失败只记录为 infrastructure attempt，不进入质量比较。
- smoke 只需每条件一次 accepted run；smoke 通过不代表效果结论。

## Stage 1：AdaMem-first Semantic RSI 与 trajectory baseline

SM01 formal B0/B1/B2 已各有 3 个 accepted replicate；当前结果只支持
full-trajectory policy-update signal，不支持 N+1 quality uplift。正式 raw
resource comparison 需在新 batch 中使用 `updater_usage.json`。

目标：先确认 `B2_mem0_adamem_full_trajectory` 是否能够让 AdaMem 的 Semantic policy updater 产生可审计的 `P_n -> P_{n+1}`，并在 matched N+1 上表现出与 `B0_mem0_static`、`B1_mem0_adamem_terminal` 不同的结果。

### 1. 固定上游与适配边界

唯一允许作为当前上游参考的仓库是：

```text
/mnt/20t/xubuqiang/Study/AdaMem
remote: https://github.com/galaxyChen/AdaMem.git
paper: AdaMem: Learning What to Remember with Adaptive Memory Policies for Personalized Agents
arXiv: 2606.21144
commit: 9dba25d (Camera-Ready public release)
```

从 AdaMem 上游只提取以下四项机制：

```text
policy schema
-> reflection input formatter
-> JSON policy patch and patch application
-> policy rendering into memory extraction instructions
```

不要迁移 AdaMem-Bench、其 synthetic story 数据、评测脚本、judge、Full Context/Ideal Memory baseline 或 upstream 的 API 配置。RSIMem 仍使用 Hermes + vendored PAST-Bench。Mem0 runtime 可为 Hermes integration 选择合适实现，但选择后必须固定，且不能改变 AdaMem 的核心更新因果链。

特别注意：AdaMem 上游的 QA record 包含 `gold_answer` 与 `golden_feedback`，这是其自有 benchmark 的公开反馈设定。PAST-Bench adapter 不得读取、复制或间接传递这些字段；任何 grader、official score、hidden answer、future evaluation 内容都不能进入 reflection/updater request。

### 2. 接入 AdaMem updater

- [x] 在 `docs/` 写一页 upstream/adaptation audit：上述 commit、AdaMem policy schema、reflection prompt 输入/输出、patch fields、render point、failure/no-op 行为，以及与 Hermes/Mem0 adapter 的逐项映射和差异。
- [x] 在 RSIMem 新建独立 AdaMem adapter；不得修改上游仓库，也不得把上游源码直接复制进 Hermes runtime。
- [x] 复刻 AdaMem 的 policy contract：旧 policy + feedback -> JSON patch -> 新 policy 或 `NO_UPDATE`；解析/shape/model 失败必须保持旧 policy。
- [x] 第一版将 policy render 到 Mem0 的 custom extraction-instruction boundary；不得改变 Mem0 retrieval、Hermes host route、task prompt 或模型配置。
- [x] 将 AdaMem diagnosis 输入抽象成可替换 feedback adapter：`terminal`、`full_trajectory` 只改变输入信息，不改变 patch space、validation 或 rollback。
- [x] 先实现 `AdaMem-native` input adapter，得到方法 fidelity baseline；不能把新增 trajectory 字段提前混入原方法。
- [x] 定义唯一的 semantic extraction policy artifact：render point、policy version、base revision、允许动作和 rollback path。
- [x] 三个条件共用同一个 AdaMem updater prompt、模型、update budget 和 policy update space；输入为空或不足时允许 `NO_UPDATE`。
- [x] 保存 policy version、request digest、proposal digest、validation result、activation/rejection、rollback 和原因码。
- [x] policy validation 拒绝把当前 task-specific answer 直接写成 policy；更新必须是可用于后续任务的泛化规则。

### 3. 先做 fidelity smoke

- [x] 用一个不含 PAST hidden evaluation 的最小 deterministic fixture，验证 policy patch 的 apply/no-op/parse-failure/rollback 行为。
- [x] 用一个 PAST Semantic family 运行 `Mem0 + AdaMem-native` smoke，确认 policy 确实在后续 Mem0 extraction 中 render 生效，并保存完整 policy version 链。
- [x] 审计 AdaMem-native request：只有允许的部署可见反馈字段；禁止字段出现即 fail closed。
- [x] 比对 `Mem0Static` 与 `Mem0 + AdaMem-native`：除 AdaMem policy update 本身外，运行 identity、模型、预算、任务、state isolation 和 Mem0 retrieval route 不得漂移。

如果 AdaMem 的原始实现无法在不改变其 update space 的前提下替换 feedback 输入，则停止直接改 AdaMem，记录耦合边界，并实现一个只复刻其 policy-update contract 的薄 `SemanticRSI fallback`。fallback 只作为工程控制变量，不能与 AdaMem-native 结果混称。

### 4. 正式运行、三次 replicate 与并行规则

- [x] 正式比较每个 `family x condition` 至少取得 3 个 accepted replicate；每个 replicate 都完整运行 train -> update/abstain -> matched N+1。provider/service/usage 失败不计入 3 次，但必须保留 attempt audit，并以新的隔离 run 重试。
- [x] replicate 指独立的完整执行，不是对同一次 policy proposal 或同一份 N+1 输出重复评分。每个 replicate 必须拥有独立 policy state、Mem0 storage/collection、Hermes HOME、session、service/port、fixture copy、trace/artifact root 和 run manifest。
- [x] 所有开发、adapter 调试、fidelity、文档检查、single-family smoke 和 condition 切换均串行执行。先串行完成一个 family 的 B0/B1/B2 smoke 和 manifest review，再开始正式 batch。
- [x] 正式 batch 的唯一并发单位是同一个 `family x condition` 的 3 个 replicate：固定同时启动 3 个独立 run。一个三-replicate batch 完成、审计并冻结后，才启动下一个 condition 或 family；不跨 family 或 condition 并发。
- [x] 全局 provider-backed run 上限固定为 3，不随机器资源提高。出现 429/503、usage-incomplete 或延迟急剧上升时，停止启动新的 batch；失败 replicate 仅在该 batch 完全结束后，以新的隔离 run 串行或重新组成三-replicate batch 重试。
- [x] 不同 replicate/condition 不共享 policy、Mem0 collection、Hermes state、service process、fixture directory、trace 或 artifact root；唯一允许共享的是只读代码、冻结任务源和只读模型配置。
- [x] 记录每个 batch 的启动/结束时间、provider health、condition 顺序和重试原因。不同 family 的 B0/B1/B2 batch 顺序轮换，避免长期 provider drift 总是与同一 condition 相关。
- [x] 额外保留 `AdaMem-native` 输入条件，确认 adapter 接线没有改变 AdaMem 原始行为；它是方法 fidelity baseline，不替代 B0/B1/B2。
- [x] 对每个 case 保存 `P_n`、`P_{n+1}`、是否更新、N+1 score/component delta、proposal/acceptance/rollback 与资源记录。
- [x] 聚合时只使用 accepted matched pairs；报告全部 3 个 raw replicate、均值/标准差和 paired delta，不用 infrastructure failure 填补缺失值。若某个 condition 未达到 3 个 accepted replicate，不报告正式主结论。
- [x] 同时报告 proposal rate、abstention rate、acceptance rate、N+1 gain、harmful update、rollback rate 和 updater input token。

### Stage 1 复核

完成后只讨论下面三个问题：

1. `Mem0Static` 是否是可信的独立 semantic backend，而不是 Full Context 或 AdaMem-Bench 的替代物？
2. AdaMem-native 是否被忠实复现，且新增 adapter 没有改变原始 update space？
3. `B2_mem0_adamem_full_trajectory` 是否比 `B0_mem0_static` / `B1_mem0_adamem_terminal` 更能产生有效、可复现的 policy update？
4. 若有 gain，它是否来自 feedback information，而不是模型、预算、任务、prompt 或泄漏差异？

若 Mem0Static 不可信，先修复 backend adapter；若 AdaMem-native 无法复现，先修复 fidelity，不进入 feedback comparison。若 `B2_mem0_adamem_full_trajectory` 不能产生任何可信 signal，暂停后续 RSIMem method 设计，先修正 AdaMem 输入边界、Mem0 extraction render point 和 train/N+1 split。若 B2 有 signal，再单独写下一版 checklist，决定是否比较 lifecycle/artifact/LLM analyst feedback view、做机制分析、接入 PG，并把同一 AdaMem policy contract 移植到 HermesNative。
