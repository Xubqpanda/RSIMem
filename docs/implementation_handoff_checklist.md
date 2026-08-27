# RSIMem 实现与验收总清单

最后更新：2026-08-27

## 1. 总体目标

这份 checklist 统一记录 RSIMem 两个阶段的串行实现和验收要求。

- 第一阶段建立可复现、行为中立、可审计的 PAST-Bench + Hermes 实验环境，并将 lifecycle control plane 以 dry-run 方式接入真实 agent loop。
- 第二阶段先完成 semantic memory 的真实 generation、transactional writeback、future retrieval、delayed feedback 和 adaptive memory policy。Episodic 与 procedural 只保留稳定接口和研究闸门，选定方法后再进入实现主线。

两个阶段严格串行完成。第一阶段全部通过后才能开始第二阶段；每个子任务只有在功能、测试、证据和文档均通过验收后，才能开始下一个子任务。

状态约定：

- `□`：尚未开始或尚未通过验收。
- `进行中`：已经开始，但不能作为后续任务的稳定依赖。
- `√`：功能、测试和证据均已通过验收。

## 2. 当前研究范围

### 2.1 Benchmark 与 Host

- 当前只使用 PAST-Bench。
- 当前只使用 Hermes host。
- 当前不接入其他 benchmark、host 或 agent framework。
- 不修改 PAST-Bench task semantics、episode order、grader、answer key 或 hidden evaluation contract 来改善结果。

### 2.2 Memory Backend

当前选择 Hermes native storage substrate，而不是把 MemBase 作为 runtime backend：

| Memory kind | Hermes native substrate | 第一阶段 | 第二阶段 |
|---|---|---|---|
| Semantic | `MEMORY.md` / `USER.md` | 验证 native read、system-prompt rendering 和 adapter equivalence。 | **当前实现主线**：生成、更新和检索 durable fact、preference、rule 与 constraint。 |
| Episodic | `state.db` session/message history | 验证 native FTS retrieval、session projection 和 adapter equivalence。 | **暂缓**：只保留 read-only contract；选定 episode segmentation、retention 和 retrieval 方法后再实现。 |
| Procedural | `skills/**/SKILL.md` 及 resources | 验证 native list/view、resource projection 和 adapter equivalence。 | **暂缓**：只保留 contract；选定 Context2Skill、Text2Skill、SkillCreator 或其他 trajectory-to-skill 方法后再实现。 |

MemBase 的 dataset、runner 和 evaluation pipeline 不进入 RSIMem。仅将本地 `/mnt/20t/xubuqiang/Study/MemBase` 中的 memory-layer 实现作为算法参考，在 RSIMem 内重写所需的最小逻辑，不在运行时 import 或调用 MemBase。

第一个实现选择 Mem0 flat-memory algorithm：关闭 graph store，只保留 fact extraction、related-memory retrieval、内部 `ADD/UPDATE/DELETE/NONE` 决策和 flat retrieval。选择 Mem0 而不是 NaiveRAG，是因为 NaiveRAG 只做分块与向量检索，没有 model-based memory extraction 或 add-time update；选择 Mem0 而不是 LangMem，是因为 Mem0 的 prompt 和核心实现已经 vendored 在 MemBase 中，更容易固定版本、审计和本地重写。

MemBase 的统一 layer API 覆盖多种主导表示形态，不能将其中所有系统都称为 semantic memory：

| MemBase layer | 主导表示形态 | 与当前实现的关系 |
|---|---|---|
| Mem0 | 原子事实、偏好和 profile 型 semantic memory；flat 模式也保留 add-time conflict resolution。 | 当前第一个 semantic algorithm reference。 |
| LangMem | 由对话抽取并维护的 semantic records。 | 后续 semantic baseline 候选。 |
| HippoRAG2 | 文档、实体和关系组织形成的 relational semantic memory。 | 后续 relational retrieval baseline 候选。 |
| Long-Context | 原始消息流组成的 bounded trajectory/context。 | 更接近 raw episodic/context baseline，不是 semantic compiler。 |
| NaiveRAG | 对消息块或文档块做 embedding retrieval。 | 更接近 uncompiled episodic/document retrieval baseline。 |
| A-MEM | 带时间、上下文、标签、链接和演化历史的 atomic agentic notes。 | Episodic 与 semantic 的混合形态。 |
| MemOS | Reader 抽取 long-term/user memory，再以 tree structure 组织。 | Hybrid textual memory，不能视为单一类别。 |
| EverMemOS | Event/episode、profile 和 foresight 共同组成的 personal memory。 | 明确的 episodic + semantic/profile + prospective hybrid。 |

该表描述的是各实现的主导表示形态，不是互斥 taxonomy。Memory system 名称、存储结构和认知类别不是一一对应关系；当前论文只先实现其中边界最清楚、最容易做 matched comparison 的 flat semantic path。

### 2.3 阶段边界

第一阶段允许：

- 读取 Hermes native memory。
- 将 native memory 投影到 host-neutral typed contracts。
- 构建 context snapshot。
- 调用 deterministic evaluator 或受控的 lifecycle evaluator。
- 生成并验证 dry-run writeback plan。
- 记录 content-free lifecycle、usage 和 accounting evidence。

第一阶段禁止：

- 执行 semantic、episodic 或 procedural memory compiler。
- 调用 memory distillation prompt 生成新 memory。
- 对 `MEMORY.md`、`USER.md`、`state.db` 或 `skills/**` 执行真实 mutation。
- 根据 dry-run plan 删除或重写真实 model context。
- 根据 delayed feedback 更新 prompt、规则、scorer 或 policy version。
- 宣称已经实现 memory generation、memory evolution 或 recursive self-improvement。

第二阶段只有在第一阶段全部通过后才允许：

- 执行版本化 semantic ingestion prompt。
- 执行 validation、transactional mutation、reread verification 和 recovery。
- 在固定 Hermes semantic route 和固定 invocation boundary 下运行 semantic memory policy。
- 根据统一 future-utility-per-cost objective 评估并更新 semantic generation/update/retrieval policy。
- 根据 deployment-observable delayed feedback 提议和激活下一 policy version。
- Episodic/procedural 只有通过各自研究闸门后，才复用同一 transaction、ledger、feedback 和 activation infrastructure。

### 2.4 统一设计哲学

Hermes backend、Hermes native memory routing、policy invocation schedule 和基础模型参数在当前论文中保持固定。当前 semantic-first 实现更新两类对象：

- Memory state：artifact 被新增、更新、保留、淘汰、召回或 supersede。
- Memory policy：每条固定 route 内负责 memory extraction、内部 conflict resolution、consolidation 和 retrieval ranking 的 prompt、规则、threshold 或轻量 scorer。

Hermes native routing 固定为：semantic 由 memory review/tool path 处理，episodic 由 session persistence/search path 处理，procedural 由 skill review/manager path 处理。Hermes 并不存在一个统一的 semantic/episodic/procedural classifier，因此不将其描述为“自动三分类 compiler”。当前只优化 semantic route；另外两条 route 继续作为行为中立的 adapter/read-path contract 存在。

RSIMem 对 memory framework 只暴露统一 `ingest/add` 操作，不在 framework 外部预测 ADD 还是 UPDATE。一次 ingestion 可以由内部 policy 产生 `ADD`、`UPDATE`、`DELETE` 或 `NONE`，这些内部 operation 必须作为 framework outcome 被记录、计费和审计。该设计减少重复决策，但不等于系统禁止 update。

完整方法闭环为：

```text
context and task outcome
  -> enter a fixed Hermes memory route at a fixed boundary
  -> run the selected route-specific memory policy
  -> internally add, update, delete, or retain memory
  -> validate and persist
  -> retrieve and inject in later tasks
  -> observe use, non-use, outcome, supersession and cost
  -> generate delayed utility labels
  -> propose and validate policy version N+1
  -> activate version N+1 in the same fixed route and boundary
```

只完成 memory writeback 时，只能声明 static memory lifecycle。Policy N+1 确实由 delayed feedback 产生、通过 held-out validation 并改善同一固定 route 的后续 memory behavior 后，才能声明 memory-mediated self-improvement。至少完成两次可重放 policy iteration，并让新 evidence 继续进入下一轮更新后，才能使用 recursive self-improvement 表述。

## 3. 当前已完成的基础

- √ 将 PAST-Bench 与 Hermes vendored 到 `benchmarks/past-bench`，保留 upstream attribution 和 license。
- √ 建立 Python 3.11 development environment 和本地安装流程。
- √ 实现 request-level model usage accounting、ledger 和 audit。
- √ 定义 semantic、episodic、procedural memory contract、backend registry 和 runtime。
- √ 实现 Hermes semantic、episodic、procedural native-format adapter。
- √ 定义 `native`、`native+ledger`、`native+adapter+ledger` 三种显式执行模式，direct native 保持默认。
- √ 实现 storage-boundary deterministic equivalence fixture。
- √ 实现 Hermes system prompt、`session_search`、`skills_list` 和 `skill_view` execution-surface fixture。
- √ 实现 deterministic PAST-Bench Hermes agent-loop fixture。
- √ 对齐 episodic FTS5 normalization、filter、session lineage、conversation projection 和损坏字段行为。
- √ 将 runtime evidence 按事件 flush 和 fsync，不等待进程正常退出后统一落盘。
- √ 定义 `ContextSnapshot`、`WritebackPlan`、revision、provenance 和 dry-run idempotency receipt。
- √ 当前回归基线为 RSIMem `90 passed` 与 PAST-Bench `384 passed, 2 skipped`。

## 4. 全局验收原则

### 4.1 行为中立

- direct native 必须保持默认开启。
- 关闭 RSIMem 时，Hermes 的输入、存储、prompt、tool surface 和输出路径不能改变。
- `native+ledger` 只能增加 observer evidence，不能改变 memory decision。
- `native+adapter+ledger` 必须保持 Hermes model-visible memory surface 等价。
- fallback 和 bypass 必须显式配置并产生 evidence，不能静默降级。

### 4.2 安全与隐私

- grader、answer key、hidden score 和未来 episode 信息不能进入 evaluator 或 dry-run policy。
- ledger、audit、receipt、reason code 和异常 trace 不能包含 context、memory、prompt、response、credential 或用户绝对路径原文。
- active/current/unresolved segment 和 open tool closure 必须保持 protected。
- evaluator、adapter 或 evidence persistence 失败不能改变 Hermes memory 或 active context。

### 4.3 可复现性

- 每次运行记录 RSIMem commit、PAST-Bench commit/tree、dirty state、model profile、judge profile、task manifest、budget 和执行顺序。
- provider 不支持 seed 时，明确记录 independent unseeded replicate，不能将其描述为 seeded run。
- 所有 experiment mode 使用隔离的 HOME、state directory、trace directory 和 persistence state。
- raw resource quantity 与 provider price 分开保存。

### 4.4 Prompt、Policy 与 Accounting

- Lifecycle evaluator、route-specific ingestor/generator、retrieval scorer 和 policy updater 使用显式版本与 content digest。
- Policy version 由 host configuration 选择，模型不能声明或覆盖当前版本。
- Prompt template 可以进入 manifest；包含 context/memory 的 rendered prompt 和 model response 原文不能进入 ledger 或公共 report。
- Evaluator、ingestor/generator、retrieval scorer、policy updater、retry、fallback 和 recovery 的模型调用全部进入 usage accounting。
- Unknown usage 保持 `null`，不能推断为零。
- Shared physical execution 只计费一次，但允许关联多个 lifecycle observation。

### 4.5 Memory Safety

- Memory 必须先完成 validation、persistence 和 reread verification，之后才允许 source context 逻辑退场。
- UPDATE 必须绑定 backend、artifact ID 和 expected revision，不允许 last-write-wins 覆盖未知新版本。
- Generated procedural memory 必须通过 Hermes-compatible security scan。
- Raw framework output 不能绕过 host-neutral validation 直接写入 Hermes。
- Evaluator、ingestor/generator、validator、backend 或 recovery failure 不能造成 memory 半写入或 context 不可恢复删除。

## 5. 标准验收命令

每个子任务至少运行聚焦测试。每个串行工作块结束前必须运行完整回归：

```bash
cd /path/to/RSIMem
.venv/bin/python -m compileall -q src tests
.venv/bin/pytest -q tests
.venv/bin/pip check
git diff --check origin/main..HEAD

cd benchmarks/past-bench
../../.venv/bin/pytest -q
```

PAST-Bench tests 必须从 `benchmarks/past-bench` 目录运行。从 RSIMem 根目录直接收集 `benchmarks/past-bench/tests` 会解析到冲突的顶层 `agent` module，不能将该入口的 collection error 当成产品回归。

每个子任务的验收材料包括：

- 精确 commit ID 和 dirty-worktree 状态。
- 实际执行的命令和结果摘要。
- fixture 的输入、模式、预期和输出路径。
- failure、fallback、retry 和 restart 的 machine-readable evidence。
- 对 benchmark semantics 未修改的确认。

## 6. 第一阶段 1A：环境与配置冻结

目标：保证任意后续实验都建立在同一套可定位、可隔离和可复现的环境上。

### 1A.1 安装与依赖

功能需求：

- √ 从 clean checkout 建立 `.venv` 并安装 RSIMem 与 vendored PAST-Bench。
- √ 固定 Python major/minor requirement，并检查 `pip check`。
- √ 文档化 RSIMem 和 PAST-Bench 两套正确测试入口。
- √ 增加一个不包含 secret 的 environment preflight command 或脚本，检查 Python、依赖、必须目录和可选 provider 配置。

验收需求：

- √ clean temporary HOME 下 preflight 能通过。
- √ 缺失依赖、错误 Python、不可写 state directory 和缺失 provider configuration 分别给出明确错误。
- √ preflight 不打印 API key、Authorization header 或完整机器专属路径。

### 1A.2 Experiment Configuration

功能需求：

- √ 显式选择 `native`、`native+ledger` 或 `native+adapter+ledger`。
- √ direct native 为默认模式。
- √ 配置 failure policy、evidence path、replicate count 和 matched mode order。
- √ 将最终使用的 model、judge、budget、task family、execution mode 和 persistence isolation 汇总到单一 manifest schema。

验收需求：

- √ manifest 缺少关键字段、包含未知 mode 或指向 dirty benchmark 时 fail closed。
- √ manifest 记录实际执行值，而不是只记录用户请求值。
- √ restart 后使用同一 manifest 能定位同一 experiment identity，但不会覆盖已有 attempt evidence。

### 1A.3 阶段闸门

- √ 在 clean temporary HOME 下完成安装、preflight、RSIMem tests 和 PAST-Bench tests。
- √ 所有配置和版本信息可从 manifest 重建。
- √ 环境检查不调用真实模型，也不产生 memory mutation。

1A 验收记录（2026-08-27）：

- 实现 commits：`1f174ab`（secret-free preflight）、`e1eafcd`（validated manifest/restart attempts）、`5e6a306`（resolved runtime/environment binding）。
- Clean temporary venv 使用 Python 3.11，从当前 checkout 安装 `RSIMem`、`PAST-Bench[mock,sandbox,dev]` 与 vendored `hermes-agent`；无 provider credential 时 optional preflight 通过。
- Clean temporary HOME 回归：RSIMem `90 passed`；从 `benchmarks/past-bench` 运行 PAST-Bench `384 passed, 2 skipped`。
- Manifest v2 记录实际 registry model/base URL、run-config runtime/temperature/judge、逐 task `max_turns`/timeout、完整 installed distribution version map、代码 revision、dirty state、轮换顺序与隔离策略。
- 已知限制：preflight 只验证 provider 配置存在，不发网络请求验证 provider 可用性；这是为了保证本阶段环境检查不调用模型。RSIMem dirty state 会被记录，PAST-Bench subtree dirty state会直接拒绝。

## 7. 第一阶段 1B：Deterministic Read-Path 等价性冻结

依赖：1A 通过。

目标：冻结三个执行模式在没有模型随机性时的行为等价性。

### 1B.1 Storage 与 Execution Surface

功能需求：

- √ 比较 semantic rendering、episodic FTS view 和 procedural resource projection。
- √ 调用 Hermes 原生 system-prompt builder、`session_search`、`skills_list` 和 `skill_view`。
- √ 覆盖 pagination、filters、session lineage、full conversation 和 linked skill resource。
- √ 覆盖 restart-stable artifact identity。

验收需求：

- √ 三种模式产生完全相同的 model-visible deterministic output。
- √ adapter evidence 不包含 memory text。
- √ fail-closed 与 native-bypass 均有明确测试。
- √ episodic query normalization 和 malformed structured field 与 native Hermes 一致。

### 1B.2 PAST-Bench Agent Loop

功能需求：

- √ 在 deterministic fixture 中经过真实 `HermesAdapter.step` 和 `_run_agent` 路径。
- √ fixture 同时触发 semantic、episodic 和 procedural read surface。
- √ episode-local evidence 通过 run、variant、trace、task、family 和 stage 关联到 ledger。

验收需求：

- √ 三种模式的最终 model-visible fixture output 相同。
- √ direct native 不产生 RSIMem runtime evidence。
- √ 两个 ledger mode 的 query、retrieval 和 injection 各记录一次，不重不漏。
- √ malformed、misplaced 或 conflicting evidence 被 audit 拒绝。

### 1B.3 阶段闸门

- √ RSIMem 全量回归通过。
- √ PAST-Bench 全量回归通过。
- √ deterministic read-path equivalence 已有可重复 fixture evidence。

## 8. 第一阶段 1C：Live Matched Read-Path 验证

依赖：1B 通过。

目标：证明 typed adapter read path 在真实模型运行中没有引入可归因的系统性行为变化。

### 1C.1 Matched Replicates

功能需求：

- √ 使用同一 model profile、judge profile、task manifest、budget、sandbox 和 provider 配置运行三个 execution mode。
- √ 每个 mode 至少完成 3 个 independent replicate。
- √ replicate 之间轮换 mode order，不能始终让同一 mode 最先执行。
- √ 记录每个 attempt 的 scheduled order、actual order、failure stage 和 output directory。

验收需求：

- √ 所有成功 run 通过 `rsimem-audit`，usage reconciliation 和 physical request deduplication 为零漂移。
- √ 三种模式不存在未解释的 task input、task order、storage state 或 budget 差异。
- √ adapter mode 不存在静默 bypass、静默空召回或 evidence 丢失。
- √ failed provider attempt 与 successful run 分开保存，不能删除失败 evidence。

### 1C.2 Nondeterminism Boundary

功能需求：

- √ 比较 task score、pass rate、model requests、各 token bucket、tool calls、retries、stored bytes、injected chars 和 wall time。
- √ 对逐 episode 差异进行 attribution，区分模型随机性、provider failure 和 adapter-caused divergence。
- √ 在观察结果前定义统计汇总方式，不能根据结果临时选择有利指标。

验收需求：

- √ adapter-caused model-visible input difference 必须为零。
- √ accounting drift 必须为零。
- √ 不用一次 run 或任意拍脑袋阈值宣称“完全等价”。
- √ 形成 dated report，明确样本量、失败 attempt、限制和是否通过阶段闸门。

### 1C.3 阶段闸门

只有 deterministic equivalence 继续通过、live runs 全部可审计且没有 adapter-caused divergence 时，才能进入 1D。provider instability 导致 replicate 不完整时，本工作块保持未完成，不提前开发后续接线。

1C 验收记录（2026-08-27）：

- Accepted batch：`outputs/matched/hermes_luna_sm01/20260827_073620`；RSIMem `24def06`；每 mode 3 个 order-rotated independent unseeded replicate。
- Machine analysis：`stageGatePassed=true`、`issues=[]`；9 个 audit 全部通过，所有 run 17 traces，0 retry，0 accounting/privacy issue。
- Adapter：每 replicate 28 个 same-call native-shadow projection check，共 84 个；0 mismatch、0 bypass、0 unresolved injection。
- Quality：所有 mode/replicate 的 with-persistence evaluation `1.0/100%`，without-persistence `0.4/0%`；逐 episode score 与 pass 完全一致。
- Dated report：`docs/matched_phase1c_20260827.md`。排除的开发批次和 live semantic-only 限制在报告中单列，未并入 accepted aggregate。

## 9. 第一阶段 1D：真实 Hermes Lifecycle Dry-Run 接线

依赖：1C 通过。

目标：在真实 PAST-Bench Hermes loop 中构建 snapshot、运行 lifecycle evaluator 并产生 validated dry-run plan，但不生成或写入 memory。

### 1D.1 Host Lifecycle Boundary

功能需求：

- □ 在 task completion 和 session end 接入明确 host event，不通过自然语言猜测任务是否完成。
- □ context-pressure 只有在 Hermes 提供可信 token total 和 threshold 时启用。
- □ turn interval 和 tool boundary 保持默认关闭。
- □ 同一 boundary 的重复 callback 产生相同 logical identity。

验收需求：

- □ deterministic fixture 精确触发一次 task-completed 和一次 session-end evaluation。
- □ duplicate callback、retry 和 restart 不重复接受同一 logical evaluation。
- □ disabled mode 不构建 snapshot、不调用 evaluator、不改变 Hermes 行为。

### 1D.2 Live Context Snapshot

功能需求：

- □ 从真实 Hermes message、session 和 task state 构造 `ContextSnapshot`。
- □ stable segment ID 不依赖消息列表位置。
- □ 正确标记 current turn、active、completed、unresolved 和 tool call/result closure。
- □ snapshot revision 覆盖所有会影响 lifecycle decision 的结构化字段。
- □ token total 等于 segment token count 之和；unknown usage 不能伪装成零。

验收需求：

- □ 同一 transcript 重启后产生相同 segment ID、snapshot ID 和 revision。
- □ 新增 turn 后旧 segment ID 保持稳定，snapshot revision 改变。
- □ orphan tool result、duplicate tool call、open tool call、缺失 current turn 和 stale task state fail closed。
- □ raw context 只存在于 snapshot/evaluator runtime boundary，不进入 ledger。

### 1D.3 Evaluator Configuration

功能需求：

- □ 配置显式选择 deterministic evaluator 或 injected JSON lifecycle evaluator，默认关闭。
- □ evaluator request 携带 snapshot revision、protected IDs、trigger、turn index 和 host-selected policy version。
- □ policy version 由 host configuration 固定，模型不能声明或覆盖版本。
- □ evaluator timeout、invalid JSON、missing signal、unknown segment 和 unsafe action 产生结构化 rejection。
- □ evaluator failure 不推进 scheduler state，允许使用同一 boundary retry。

验收需求：

- □ deterministic evaluator 在固定 fixture 上完全可重现。
- □ mocked evaluator 覆盖 valid、malformed、partial、timeout、exception 和 policy-version override。
- □ evaluator failure 不产生 accepted plan，不产生 mutation event，也不改变下一轮 native request。
- □ lifecycle evaluator 的模型调用进入 usage accounting，但 prompt/response 原文不进入 ledger。

### 1D.4 Dry-Run Plan 与 Evidence

功能需求：

- □ 将通过验证的 lifecycle signal 转换为 revisioned `WritebackPlan`。
- □ plan 关联 run、episode、session、task、snapshot、evaluation、segment 和 policy version。
- □ 使用 persistent idempotency receipt，重复处理同一 logical input 只能接受一次 dry-run。
- □ plan validation 保护 current、active、unresolved 和 open tool closure。
- □ dry-run 只记录本来会执行的 action，不调用 compiler 或 backend mutation。

验收需求：

- □ stale revision、duplicate plan、malformed receipt、ambiguous update target 和 unsafe eviction 被拒绝。
- □ 并发 dry-run coordinator 只有一个能够 reserve 同一 idempotency key。
- □ restart 后 duplicate plan 被识别，receipt corruption fail closed。
- □ Hermes memory files、state DB、skills 和 forwarded model context 在 dry-run 前后字节级不变。

### 1D.5 阶段闸门

- □ 真实 Hermes loop 能产出 snapshot、evaluation、validated dry-run plan 和完整 content-free evidence。
- □ disabled、success、failure、retry 和 restart 路径均有 deterministic test。
- □ 三个 execution mode 的 read-path equivalence 继续通过。
- □ RSIMem 与 PAST-Bench 全量回归通过。
- □ 没有真实 memory generation、memory mutation 或 context eviction。

## 10. 第一阶段 1E：最终验收与冻结

依赖：1D 通过。

目标：把第一阶段冻结为第二阶段可直接依赖的稳定实验底座。

### 1E.1 End-To-End Infrastructure Acceptance

功能需求：

- □ 从 clean temporary HOME 启动 PAST-Bench Hermes。
- □ 加载固定 experiment manifest。
- □ 完成 selected task sequence、typed memory reads、snapshot、evaluation、dry-run plan、ledger 和 audit。
- □ 模拟一次 evaluator failure 和一次 restart，并完成可审计恢复。

验收需求：

- □ direct native 行为保持不变。
- □ adapter mode 的 model-visible memory 与 native 等价。
- □ 所有 physical model requests、tool calls、memory reads 和 dry-run lifecycle events 可重建且不重不漏。
- □ 所有 observer-facing evidence 通过 privacy audit。
- □ Hermes memory 与 active context 没有被第一阶段代码修改。

### 1E.2 第二阶段输入契约冻结

功能需求：

- □ 冻结 `ContextSnapshot`、lifecycle evaluation、`WritebackPlan`、provenance、revision、idempotency 和 usage schema 的第一阶段版本。
- □ 记录第二阶段可以依赖的 API、版本、fixture 和 evidence path。
- □ 列出所有已知限制，但不在第一阶段提前实现第二阶段功能。

验收需求：

- □ 第二阶段可以从一个 validated plan 获得 route-specific ingestor 所需 source reference 和 structured exit evidence。
- □ schema version mismatch、unknown required field 和 stale revision 有明确拒绝语义。
- □ 第一阶段 report 不包含任何 memory quality 或 recursive improvement claim。

### 1E.3 第一阶段完成条件

只有以下条件全部成立，才能将第一阶段标记为完成：

- □ 1A-1E 所有子任务均已勾选并有对应 commit。
- □ 最终 RSIMem 与 PAST-Bench 全量测试通过。
- □ live matched read-path validation 通过。
- □ 真实 Hermes lifecycle dry-run acceptance 通过。
- □ ledger、audit、restart、failure 和 privacy evidence 完整。
- □ 第一阶段没有执行任何真实 memory generation、mutation 或 context eviction。

## 11. 第二阶段前置条件

第二阶段实现真实 memory generation 与 self-improvement。开始前必须满足：

- □ 1A-1E 全部完成。
- □ Live matched read-path validation 通过，未发现 adapter-caused model-visible divergence。
- □ 真实 Hermes lifecycle dry-run 已接通 snapshot、evaluation、validated plan、ledger 和 audit。
- □ `ContextSnapshot`、`WritebackPlan`、provenance、revision、idempotency 和 usage schema 已冻结版本。
- □ Direct native 仍是默认路径，第二阶段功能全部通过显式配置启用。
- □ RSIMem 与 PAST-Bench 全量回归保持通过。

任一前置条件未满足时，不开始 ingestor/generator、backend mutation 或 policy update。

## 12. 第二阶段 2A：固定 Routing 与 Ingestion Contract

目标：冻结 Hermes 既有 memory routing 和调用边界，当前只为 semantic route 暴露统一 ingestion，不让 RSIMem 重复判断 memory form 或 ADD/UPDATE。

建议修改范围：`src/rsimem/lifecycle/contracts.py`、`src/rsimem/lifecycle/writeback.py`、`docs/lifecycle_controller.md` 和 contract tests。

### 2A.1 Fixed Route And Invocation

功能需求：

- □ 固定 semantic、episodic、procedural 三条 Hermes native route，不增加统一 memory-form classifier；当前只启用 semantic policy implementation。
- □ 固定 semantic route 的 invocation boundary、输入投影和输出 surface，所有 semantic policy version 使用相同触发条件。
- □ Semantic route 对外只接受 `ingest(experience)`；episodic/procedural contract 保持可构造但不启用 policy implementation。
- □ 不在 RSIMem 外层预测 ADD、UPDATE、DELETE 或 NONE。
- □ 明确 natural session exit、logical context exit 和 physical context rewrite；第一版使用 natural boundary。

验收需求：

- □ 相同 semantic experience 在所有 policy variant 中进入相同 route 和 invocation boundary。
- □ Semantic input 不会被外层改路由为 episodic 或 procedural。
- □ Disabled mode 完全恢复 Hermes native routing 与 invocation。
- □ Open tool closure、active turn 和 unresolved task 不能进入 completed-experience ingestion。

### 2A.2 Add-Only External Contract

功能需求：

- □ 外部 ingestion contract 只包含 source experience、fixed route、policy version、provenance 和 idempotency identity。
- □ 定义 host-neutral `SemanticMemoryPolicy` interface：接收 ingestion request 与受控的 existing-memory candidate reader，返回 `MemoryIngestResult`；接口不暴露 Hermes file path、Mem0/Qdrant object 或 provider-native payload。
- □ 第一个 registered policy provider 命名为 `mem0_flat`；registry 显式绑定 policy/framework version 和 capability，unknown provider fail closed。
- □ Memory framework 内部允许产生 `ADD`、`UPDATE`、`DELETE` 或 `NONE` operation。
- □ Internal operation、target、old/new digest、cost 和 reason code 作为 ingestion outcome 记录。
- □ RSIMem 不重复实现一套独立 ADD-versus-UPDATE predictor。

验收需求：

- □ 同一 ingestion retry 不重复应用内部 operation。
- □ Internal UPDATE/DELETE 仍受 revision、validation、transaction 和 recovery 约束。
- □ `NONE` 与 framework failure 可明确区分。
- □ Framework 若不支持 add-time update，capability 中必须明确声明，不能假装支持。
- □ Fake semantic policy 与 `mem0_flat` 可以在不改 coordinator/executor 的前提下替换，证明 policy interface 与具体算法解耦。

### 2A.3 Policy Ownership

功能需求：

- □ Policy、framework、prompt 和 feature schema version 由 runtime 绑定。
- □ Internal model output 可以建议 operation，但不拥有 revision、policy identity 或 activation 权限。
- □ Target resolution、revision binding、validation 和 safety decision 保持 deterministic/trusted。

验收需求：

- □ Model 尝试伪造 version、backend、artifact ID 或 revision 时 fail closed 或被严格忽略。
- □ 同一 policy 与 deterministic input 产生相同 logical plan identity。
- □ Policy version 变化时 decision provenance 改变，source snapshot identity 不改变。

### 2A.4 阶段闸门

- □ Fixed routing、add-only ingestion、internal operation 和 ownership semantics 形成设计记录。
- □ Contract 正反向测试全部通过。
- □ 尚未调用 compiler 或执行 backend mutation。

## 13. 第二阶段 2B：Prompt 与 Memory Ingestion 基础设施

依赖：2A 通过。

目标：参考 MemBase 中的 Mem0 flat algorithm，建立可版本化、可计费、可验证的 ingestion 边界，但暂不写入 Hermes。

建议修改范围：新增 `src/rsimem/memory_systems/mem0_flat/`，扩展 `src/rsimem/memory/contracts.py`、`src/rsimem/ledger.py` 和 ingestion tests。只重写 memory algorithm，不复制 MemBase dataset、runner、evaluation 或 tracing framework。

### 2B.1 Prompt Artifact

功能需求：

- □ 定义 prompt ID、version、template digest、input schema、output schema、model profile 和 policy version。
- □ Mem0 fact extraction、Mem0 internal operation decision 和可选 semantic retrieval scorer 使用统一 prompt contract。
- □ Prompt template 与 rendered prompt 分离。
- □ Prompt artifact 可进入 manifest；rendered prompt 不进入 ledger、receipt 或 report。
- □ 记录 MemBase commit `d2aca6c7abcb1d67b331586cb834495d037fa3a6`、原 prompt 路径、MIT attribution 和本地修改 digest。
- □ 不原样保留 Mem0 中与 memory construction 无关或不适合 PAST-Bench 的指令，例如要求谎称信息来自公开互联网的回答规则。

验收需求：

- □ Digest 与 template 内容确定性绑定。
- □ 缺失 schema、model profile 或 version 的 artifact 构造失败。
- □ Rendered prompt 中的 sentinel text 不出现在 observer-facing evidence。
- □ Fake completion client 可在无 provider 环境下验证 contract。

### 2B.2 Ingest Request 与 Result

功能需求：

- □ `MemoryIngestRequest` 包含 source、fixed route、exit evidence、scope、validity、framework version、policy version 和 provenance。
- □ 外部 request 不携带预先决定的 ADD/UPDATE operation 或 target。
- □ `MemoryIngestResult` 包含 execution ID、status、ordered internal operations、usage、reason codes 和 content digests。
- □ Internal UPDATE/DELETE operation 必须返回 candidate target，trusted resolver 再绑定真实 artifact 和 revision。
- □ Framework 只能返回 host-neutral operation/artifact，不能返回 Hermes/OpenAI/Anthropic native payload。
- □ 区分 rejected、failed、successful NONE 和 successful mutation。

验收需求：

- □ Missing provenance、source、route 或 version 被拒绝。
- □ Stale snapshot、unknown route、ambiguous internal target、duplicate operation 和 invalid resource 被拒绝。
- □ Canonical identity 不受 key order 影响，但任一 framework-relevant evidence 变化都会改变。
- □ Timeout、invalid JSON 和 exception 转为结构化 failure，不泄漏 source content。

### 2B.3 Deterministic Ingestor

功能需求：

- □ 实现仅供 fixture 使用的 deterministic pass-through ingestor。
- □ 支持 semantic ingestion 内部产生 ADD/UPDATE/DELETE/NONE。
- □ 不调用模型、不读取 grader、不读取隐藏状态。
- □ Ingestion planning 完成后 backend 保持不变。

验收需求：

- □ 相同 request 重启前后生成相同 execution、mutation 和 digest。
- □ Malformed candidate、stale target 和 unsupported capability 被拒绝。
- □ Ingestor success/failure usage 可进入 ledger 和 audit。

### 2B.4 Atomic Memory Operation Graph

设计参考：MemTrace 使用 `comment_variable`、`comment_op`、`comment_op_scope` 和 `comment_mutation` 将 Mem0 的 extraction、operation decision、mutation 与 retrieval 组织成 operation-variable graph，并按 attributed operation 将失败反馈分配给 fact-extraction prompt 或 update-decision prompt。RSIMem 只借鉴“原子步骤可归因”的思想，不复制其 runtime、全量变量快照、XML subgraph、source-code metadata 或默认 LLM attribution 流程。MemTrace optimization 示例可使用 golden answer 构造失败反馈；RSIMem 不复用这一监督边界，PAST-Bench hidden grader 继续严格隔离。

功能需求：

- □ 只为 source observation、fact extraction、related-memory retrieval、internal operation decision、target resolution、validation、mutation、reread verification、future query、retrieval、injection、use 和 downstream outcome 等预定义高价值边界记录 operation；不做 per-token、任意函数调用或完整 Python call graph tracing。
- □ 提供最小 instrumentation API，例如 `operation_scope`、`record_artifact` 和 `record_mutation`；允许用 decorator/context manager 降低接线成本，但 operation identity、parent edge 和 failure semantics 由显式 contract 决定，不能依赖函数名或调用栈猜测。
- □ 每次 operation 记录 `operation_id`、parent operation IDs、typed input/output artifact IDs、run/episode/session、policy/prompt/framework version、status、reason code、latency、model usage 和 retry identity。
- □ Variable/artifact node 只记录 stable ID、kind、schema version、content digest、byte/token size、revision 和 provenance reference；原文只留在 owner-controlled source/backend。
- □ Mutation edge 记录 target ID、expected revision、before/after digest、internal `ADD/UPDATE/DELETE/NONE` 和 receipt ID，不复制 before/after content。
- □ 支持一条 source 产生多个 fact、一条 fact 检索多个 related memories、多个 proposal 归并为一个 mutation，以及一个 artifact 多次 retrieval/use。
- □ Operation graph 从 append-only lifecycle evidence 派生，不建立第二套可修改的事实来源。
- □ 在线 critical path 只追加定长/有界 event；parent traversal、subgraph materialization、failure grouping 和 policy-target join 在 episode 后离线执行。
- □ 支持 `minimal`、`sampled` 和 `diagnostic` tracing level；论文主实验默认 `minimal`，`diagnostic` 只用于受控失败分析，不能静默启用。
- □ Tracing 为 observer-only；graph writer、digest、serialization 或 flush failure 不能改变 ingestion decision、backend mutation 或 model-visible memory。

验收需求：

- □ Deterministic Mem0 fixture 可重建 `source -> extraction -> related retrieval -> decision -> mutation -> verification` 子图。
- □ Future-use fixture 可继续连接 `artifact -> retrieval -> injection -> use/non-use -> outcome`，且 operation/artifact identity 不重不漏。
- □ Parallel model calls、retry、NONE、rejected proposal、failed mutation 和 restart recovery 不会错误合并为同一 operation。
- □ Ledger/graph privacy audit 证明不存在 raw source、memory、query、prompt、response、credential 或绝对路径。
- □ 强制注入 tracing failure 时，memory result 与 tracing-disabled control 等价，并产生独立的 observer failure evidence 或显式 audit gap。
- □ 分别报告 tracing-disabled、minimal 和 diagnostic 的 event count、serialized bytes、CPU time、wall-time overhead 和 peak memory；超过预先配置预算时降级为 minimal 并标记 attribution gap。

### 2B.5 阶段闸门

- □ Prompt、request、result、deterministic ingestion 和 atomic operation graph contract 冻结版本。
- □ Ingestor 与 executor 已证明解耦。
- □ 尚未执行真实 backend mutation。

## 14. 第二阶段 2C：Mutation Validation 与 Security Boundary

依赖：2B 通过。

目标：在任何 memory-framework output 接触 Hermes 前建立统一 validation pipeline。

建议修改范围：新增 `src/rsimem/memory/validation.py`，扩展 backend descriptor、ingest result、coordinator 和 tests。

### 2C.1 Host-Neutral Validation

- □ 校验 mutation kind、action、backend capability、namespace、target ownership 和 expected revision。
- □ 校验 content、metadata allowlist、resource path/count/size 和 provenance consistency。
- □ Validation result 只包含 reason code、digest 和 size，不包含原文。
- □ Path traversal、absolute path、duplicate resource、oversized content、invalid namespace 和 stale revision 全部拒绝。
- □ Validation failure 不调用 `backend.mutate`，不创建 committed receipt。

### 2C.2 Semantic Validation

- □ Semantic 校验 `memory/user` namespace、entry delimiter、字符预算、duplicate 和 conflict。
- □ Semantic entry 仅允许 durable fact、preference、rule 和 constraint，不接受完整 transcript、tool payload 或 skill resource。
- □ Prompt injection、credential、machine path、fabricated target、changed source 和 cross-run target 被拒绝。
- □ Episodic/procedural validator 保持 disabled；没有对应 research gate 和 fixture 时不能通过 generic mutation path 写入。

验收需求：

- □ Semantic memory 有完整 allow/reject matrix。
- □ Safe semantic fixture 通过。
- □ Disabled episodic/procedural mutation 被 capability gate 拒绝。

### 2C.3 阶段闸门

- □ Semantic memory validation 正反向 fixture 完整。
- □ Security failure、revision conflict 和 unsupported action 均 fail closed。
- □ 任何 rejected output 都没有修改 backend。

## 15. 第二阶段 2D：事务化 Mutation Executor

依赖：2C 通过。

目标：将 dry-run coordinator 升级为 crash-safe、idempotent、可恢复的真实 executor。

建议修改范围：扩展 `src/rsimem/lifecycle/writeback.py`，新增 executor/recovery module，扩展 backend、ledger 和 audit tests。

### 2D.1 Receipt State Machine

- □ 定义 `pending`、`committed`、`failed` 和 `rolled_back`。
- □ Pending receipt 记录 idempotency key、attempt、backend、target、pre-revision、mutation digest 和 provenance。
- □ Receipt 原子写入并加锁，同一 logical mutation 只能有一个 active executor。
- □ Unknown、malformed、digest conflict 和 target conflict fail closed。

验收需求：

- □ 两个并发 executor 只有一个 reserve 成功。
- □ Duplicate retry 不重复 ADD 或 UPDATE。
- □ Receipt corruption 与 orphan artifact 被 audit 发现。

### 2D.2 Apply、Verify 与 Recovery

- □ 严格执行 `validate -> reserve pending -> mutate -> reread -> verify -> commit receipt`。
- □ ADD 验证实际 artifact ID、kind、digest、resources 和 revision。
- □ UPDATE 使用 expected revision CAS，storage bytes 从实际结果计算。
- □ Backend accepted 但 reread 不一致时不能 commit。
- □ 覆盖 reserve 后、backend call 前、backend write 后、verification 前和 receipt commit 前五个 crash point。
- □ Restart 后区分未执行、已执行未 commit、失败和状态未知。
- □ 可证明安全时 commit/rollback；状态不明时阻止同目标后续 mutation。

验收需求：

- □ Semantic ADD、UPDATE、DELETE 和 NONE 成功路径通过。
- □ Revision conflict、permission error、disk failure、partial write 和 reread mismatch 不产生 committed receipt。
- □ 五个 crash fixture 均有确定、幂等的恢复结果。
- □ 外部 actor 修改 revision 时停止自动 rollback。

### 2D.3 Context Exit Gate

- □ 只有 committed 且 reread-verified 的 memory 才允许 source logical exit。
- □ 任一 failure 保留 source context/reference。
- □ 第一版只在 natural task/session boundary 执行 logical exit。
- □ Physical rewrite 保持关闭，直到 host contract 能证明 revision、tool closure 和 rollback。
- □ Report 区分 natural exit、logical exit 和 physical rewrite。

### 2D.4 阶段闸门

- □ Transaction、idempotency、recovery 和 audit 全部通过。
- □ 真实 mutation 默认关闭，只在 isolated fixture 显式启用。
- □ Direct native 和第一阶段 read-path regression 继续通过。

## 16. 第二阶段 2E：Mem0-Style Semantic Memory 与 SM01 闭环

依赖：2D 通过。

目标：在 `SM01_preference_adoption` 完成第一条真实 semantic writeback、restart、retrieval、injection 和 downstream-use 链路。

### 2E.1 Mem0 Flat Ingestion

- □ 基于 Mem0 `FACT_RETRIEVAL_PROMPT` 重写 PAST-Bench semantic fact/preference extraction prompt。
- □ 基于 Mem0 `DEFAULT_UPDATE_MEMORY_PROMPT` 实现 related-memory comparison 和内部 `ADD/UPDATE/DELETE/NONE` decision。
- □ 关闭 graph store，不复制 MemBase runner/evaluation，只实现 flat semantic construction 与 retrieval。
- □ Add-time related-memory candidate reader 对 Hermes semantic entries 建立受控的 flat search projection；它只服务 internal operation decision，不替换 Hermes future-task eager injection surface。
- □ 明确 candidate retrieval 的 embedding/model、top-k、threshold、index revision 和 rebuild semantics，并纳入 policy version 与 usage accounting。
- □ 将 durable fact、preference、rule 和 constraint提取为最小独立 entry，不复制完整轨迹、失败尝试、tool noise 或 unresolved information。
- □ Internal UPDATE/DELETE suggestion 经 trusted resolver 绑定唯一 target/revision；ambiguous target fail closed。
- □ Duplicate 使用 NONE，superseded artifact 保留 provenance 和 replacement relation。

验收需求：

- □ SM01 fixture 通过统一 ingest 请求生成一条最小、可重新注入的 preference。
- □ Temporary、contradictory、unknown-owner 和 unresolved candidate 被拒绝或 defer。
- □ 重复 experience 不新增 duplicate entry。
- □ 新信息、冲突信息、重复信息和应删除信息分别覆盖内部 ADD、UPDATE、NONE 和 DELETE。
- □ Prompt malformed、timeout 和 hallucinated target fail closed。

### 2E.2 SM01 End-To-End

- □ Learn episode 在固定 semantic route/boundary 产生 snapshot 和 ingest request，不额外判断 memory form。
- □ Mem0-style ingestor、validation 和 executor 写入 isolated `MEMORY.md` 或 `USER.md`。
- □ 新进程/fresh session 通过 Hermes native prompt builder 注入 memory。
- □ 连接 source、ingestion、internal operation、mutation、artifact、retrieval、injection 和 downstream task。

验收需求：

- □ Restart 后 artifact 存在且 digest/revision 可验证。
- □ Model-visible prompt 不由测试直接注入答案。
- □ Disabled mode 恢复 direct native behavior。
- □ Ledger 可重建链路但不包含 preference 原文。
- □ Failure 不导致 source evidence 丢失。

### 2E.3 Operation-Level Attribution

- □ 将 Mem0-style ingestion 拆为 extraction、related-memory retrieval、internal decision、target resolution、validation、mutation 和 reread verification operation。
- □ 将未来 semantic query、candidate retrieval、injection、use/non-use 和 task outcome 接回同一 artifact revision。
- □ 为每个 optimizable operation 标注 policy parameter/prompt field ownership，至少区分 fact-extraction prompt、update-decision prompt 和 retrieval parameters。
- □ Attribution 输出只引用 operation/artifact/policy ID 和 failure category，不包含原始 memory 或模型响应。
- □ 同一失败可以归因到一个或多个 candidate operation，但必须记录 attribution method、confidence、evidence window 和 version。
- □ Attribution 采用分层策略：先使用 contract violation、receipt、retrieval/use 和 outcome evidence 做 deterministic attribution；只有失败样本无法定位且明确启用时，才允许调用预算受限的 LLM attribution。
- □ Successful、NONE、未曝光和 censored 样本不默认触发 LLM attribution；批量 attribution 必须采样、去重并设置 calls/tokens/wall-time 上限。
- □ Attribution model 的请求、token、重试、延迟和费用单独计入 policy-update cost，不能算作免费离线处理。

验收需求：

- □ Extraction 漏事实、错误 UPDATE target、重复 ADD、retrieval miss 和 retrieved-but-unused 分别落到不同 operation/failure category。
- □ 最终 task failure 不会无条件归因给最后一次回答或全部历史 memory operation。
- □ Attribution 只能使用失败发生时已可观测 evidence，不能读取 grader answer、未来 episode 或 held-out outcome。
- □ 关闭 attribution 不改变 ingestion、retrieval 和任务结果。
- □ Deterministic attribution 足够时不会发出 attribution model request；预算耗尽时保留 unresolved attribution，不使用全轨迹统一归因兜底。

### 2E.4 Static SM01 Comparison 与闸门

- □ 比较 no persistence、native Hermes 和 static semantic writeback。
- □ 固定 model、judge、budget、task order、sandbox 和 persistence isolation。
- □ 报告 task score、persistence gap、usage、ingestion-policy cost、storage、retrieval、injection、retry 和 wall time。
- □ 每个 variant 完成规定 replicate 并通过 audit。
- □ 第一条 semantic lifecycle 可复现、可审计、无 leakage。
- □ 尚未宣称 adaptive self-improvement。

## 17. 延后范围 2F：Episodic Memory Research Gate

状态：不属于当前 semantic-first 串行主线，不阻塞 2H-2K。只有完成方法选择和 matched-evaluation 设计后，才拆分为实现 checklist。

目标：先回答“什么是值得保留和复用的 episode、如何分段、如何根据 outcome 检索”，避免将 transcript retention 错当成已经优化 episodic memory。

### 2F.1 候选方法与可借鉴机制

| 方法方向 | 主要机制 | 可借鉴内容 | 当前限制 |
|---|---|---|---|
| EM-LLM | 依据 surprise 等信号在线切分 event，并做 event-level organization/retrieval。 | Episode boundary、event adjacency 和跨 event retrieval。 | 主要面向无限上下文，不直接解决 agent outcome attribution。 |
| Generative Agents / MemoryBank | 基于 relevance、importance、recency、reflection 或 forgetting/reinforcement 管理经历。 | Retention score、时间衰减和强化基线。 | Reflection 可能把 episode 编译成 semantic summary，taxonomy 边界较模糊。 |
| Reflexion / ExpeL | 保存失败、反馈和成功轨迹，并在后续任务复用 verbal experience。 | Outcome-aware experience representation。 | 产物常接近 strategy/lesson，可能属于 semantic 或 procedural memory。 |
| EverMemOS | Boundary detection 后形成 event/episode，并联合 profile 与 foresight 检索。 | Event segmentation、episode metadata 和多阶段 retrieval。 | 是 hybrid system，不能作为纯 episodic 对照直接解释。 |
| Causal Episodic Memory / MERIT | 只使用已结束的历史 episode，区分 verified correction 与 unsuccessful direction，并按 failure type 做 hybrid retrieval。 | Temporal eligibility、positive/negative outcome、failure-typed retrieval 和 causal availability audit。 | 当前证据集中在 Text-to-SQL repair，迁移到 PAST-Bench 前需独立验证。 |

### 2F.2 解锁条件

- □ 给出 operational definition：episodic artifact 必须保留 situated event、时间/任务边界和 outcome，而不只是去上下文化的事实或步骤。
- □ 选定一个 PAST-Bench family，证明它需要 episode-level experience，且不能由同预算 semantic fact baseline 等价解决。
- □ 冻结 segmentation、admission、positive/negative outcome、temporal eligibility、retrieval 和 stale handling contract。
- □ 决定是引用 Hermes native session/message 作为 source of truth，还是生成独立 episode artifact；两种方案均禁止无审计复制 transcript。
- □ 设计 no episodic、raw episode retrieval、chosen episodic method 和 matched semantic baseline。
- □ 明确官方 grader 只用于 evaluation，不能作为在线 memory polarity、admission 或 retrieval 的隐藏信号。
- □ 完成 privacy、restart、deduplication、current-session self-retrieval 和 source-deletion failure semantics。

只有以上项目完成后，才允许实现 episodic mutation/retrieval；此前保持 adapter read-only，不能为追求“三类齐全”提前写一个缺少研究假设的版本。

## 18. 延后范围 2G：Procedural Memory Research Gate

状态：不属于当前 semantic-first 串行主线，不阻塞 2H-2K。

目标：选定 trajectory-to-skill 方法后，再将可复用执行经验编译为 Hermes native skill；当前不自造一个无法与论文 baseline 横向比较的 procedural prompt。

### 2G.1 解锁条件

- □ 对 Context2Skill、Text2Skill、SkillCreator 和直接 trajectory distillation 做统一输入、输出、反馈、依赖和 license 对比。
- □ 选定一个 PAST-Bench procedural-reuse family，冻结 native skill control 与 generated skill variant。
- □ 定义 reusable workflow 与 one-off command 的判定、适用条件、验证步骤、失败恢复和边界条件。
- □ 定义 `SKILL.md`、references、templates、scripts、assets 的完整 resource transaction 和 security scan。
- □ 明确 compiler plugin 与 Hermes storage backend 分离，外部仍只调用 ingest/add。
- □ 设计 generation、restart、progressive disclosure、actual use 和 downstream outcome 的 operation graph。

通过研究闸门后再实现 transactional persistence、`skills_list`/`skill_view` reuse 和 matched evaluation，并复用 2B-2D 已验证的版本、validation、receipt 与 recovery infrastructure。

## 19. 第二阶段 2H：统一 Static Memory Policy Objective

依赖：2E 通过。2F/2G 为延后研究范围，不是当前依赖。

目标：在固定 semantic route 和固定 invocation boundary 下，用同一 future-utility-per-cost objective 评价 semantic policy 的 extraction、internal update/consolidation 和 retrieval behavior，不学习何时调用哪条 policy。

### 2H.1 Feature 与 Utility Contract

- □ 定义 completion、unresolved、scope、validity、recency、reuse、conflict、storage/retrieval/injection cost 和 recovery risk。
- □ 区分 host-observed、model-predicted 和 delayed feature。
- □ Feature schema 版本化并定义 missing-value semantics。
- □ Hidden benchmark score 不属于 policy feature。
- □ 定义 predicted benefit、full lifecycle cost 和 uncertainty/risk。
- □ Generation/update quality 与 retrieval ranking 使用相同 future-utility semantics。
- □ 第一版使用固定规则或可解释非参数化 scorer。

验收需求：

- □ Feature extraction deterministic fixture 通过。
- □ Missing、unknown、out-of-range 和 future-dated feature 有明确处理。
- □ 成本升高且 utility 不变时，policy score 不应提高。
- □ Utility 升高且成本不变时，policy score 不应下降。
- □ Unknown cost、no history、low confidence 和 conflict 使用 conservative fallback。

### 2H.2 Fixed Invocation 下的 Static Policy

- □ Semantic route 使用固定 Mem0-style extraction、internal operation 和 flat retrieval policy。
- □ 所有 static variant 接收相同 semantic route、boundary 和 source trajectory。
- □ Static policy 在整个 run 中冻结，不使用当前 run 反馈在线更新。
- □ Logical exit 只有在 persistence commit 后生效。
- □ Physical rewrite disabled 时不报告 saved tokens；启用时根据真实 forwarded context 计算。

验收需求：

- □ 相同 route、input、feature 和 version 产生相同 internal operation/retrieval output。
- □ Policy version 不改变 route 或 invocation count。
- □ Unsupported internal operation 在 capability/validation 阶段过滤。
- □ Natural、logical 和 physical exit evidence 分开统计。

### 2H.3 阶段闸门

- □ Semantic generation/internal operation/retrieval outcome 使用统一 utility semantics。
- □ Static LightRSI 可在 selected semantic-relevant PAST-Bench families 运行并审计。
- □ Policy 在 run 内保持冻结。

## 20. 第二阶段 2I：Delayed Feedback Dataset

依赖：2H 通过。

目标：构建可用于更新 memory policy 的无泄漏 delayed feedback 数据。

### 2I.1 Lifecycle Join 与 Label

- □ 从 atomic operation graph 连接 source、extraction、related-memory retrieval、internal operation、mutation、artifact revision、query、retrieval、injection、task、tool、supersession 和 recovery。
- □ 支持一个 artifact 多次 retrieval、一次 query 多个 hit 和共享 physical execution。
- □ 明确 observation window、cutoff 和 policy version。
- □ 显式记录 non-retrieval、retrieved-not-injected、injected-not-used、superseded 和 censored。
- □ Label 只使用后续 retrieval、injection、tool behavior、retry、completion、supersession、non-use 和 lifecycle cost。
- □ Official PAST-Bench score 只用于最终 evaluation，不进入 policy-update label。
- □ 定义 positive、negative、unresolved 和 censored utility。
- □ 为每个 label 保留 attributed operation IDs 和最小 failure subgraph，使 extraction、operation decision 与 retrieval policy 能分别接收反馈。

验收需求：

- □ Referential-integrity audit 发现 orphan、duplicate、revision mismatch 和 future leakage。
- □ 同一 evidence 重复生成 canonical-equivalent dataset。
- □ 四类 label fixture 通过，修改 attribution window 生成新版本且不覆盖旧数据。
- □ Dataset 不包含 raw memory、prompt、response 或 hidden evidence。
- □ 删除无关 operation 后 label 不变；删除被归因 operation 后 integrity audit 失败，防止把整条 trajectory 无差别作为训练样本。

### 2I.2 Exposure Bias

- □ 记录 artifact 是否有曝光机会、是否进入 candidate set 和是否被 policy 过滤。
- □ 区分“没有价值”和“没有曝光机会”，不把未召回直接标为负 utility。
- □ 记录 propensity 或 deterministic eligibility，支持偏差分析。
- □ Missing propensity 时禁止使用需要 propensity correction 的 estimator。
- □ Dataset report 显示 observation count 和 censoring rate。

### 2I.3 阶段闸门

- □ Feedback dataset 可重建、可审计、无 leakage。
- □ Utility label 与 official score 严格隔离。
- □ Static policy、feature schema 和 label schema 冻结版本。

## 21. 第二阶段 2J：Adaptive Memory Policy

依赖：2I 通过。

目标：利用 delayed feedback 更新轻量 policy，不修改 Hermes backend 或基础模型参数。

### 2J.1 Policy Learner 与 Artifact

- □ 首先实现可解释 learner，例如 Bayesian estimate、regularized linear scorer 或 contextual bandit。
- □ Learner 只读取冻结 feature/label dataset。
- □ 记录 training config、seed、feature、objective 和 regularization。
- □ Low-sample、missing feature 和 distribution shift 使用 conservative fallback。
- □ Policy learner 只更新 semantic route 内 extraction、operation、consolidation 或 retrieval parameters/prompt，不更新 route selector 或 invocation schedule。
- □ Prompt/parameter update 使用 attributed failure subgraph 聚合，不把所有 task failure 同时归因给全部历史 operation。
- □ Policy artifact 包含 version、parent、dataset version、schema、parameters、prompt refs、metrics 和 digest。
- □ Proposal、validated、active、rejected 和 rolled_back 使用显式状态。

验收需求：

- □ 相同 dataset/config/seed 产生相同 artifact。
- □ Hidden grader evidence 不进入 learner input。
- □ Tampering、unknown parent、schema mismatch 和 missing provenance 被拒绝。
- □ Restart 后 active policy 唯一且可验证。

### 2J.2 Validation、Activation 与 Rollback

- □ 按时间或 task group 划分 train/validation，避免 future leakage。
- □ 与 frozen static policy 在同一 held-out evidence 比较。
- □ 预先定义 quality、cost、stability 和 uncertainty acceptance criterion。
- □ 未通过 criterion 的 proposal 保持 rejected。
- □ 原子切换 active policy pointer，每个 decision 记录实际 version。
- □ 支持 operator rollback 和 automatic safety rollback。

验收需求：

- □ Split identity、cutoff 和 episode membership 可审计。
- □ Acceptance decision 可重放，不根据 test official score 选择 policy。
- □ Activation crash 不产生两个 active policy。
- □ Repeated activation/rejection/rollback 幂等，restart 后稳定。

### 2J.3 阶段闸门

- □ Policy N+1 由 deployment-observable feedback 生成。
- □ N+1 通过 held-out validation 后才激活。
- □ Activation、rejection、rollback 和 replay 全部可审计。
- □ 满足 memory-mediated adaptive self-improvement 的实现定义。

## 22. 第二阶段 2K：PAST-Bench 实验与论文验收

依赖：2J 通过。

目标：仅在 PAST-Bench 内验证 static/adaptive LightRSI 的效果、成本、机制和泛化性。

### 2K.1 Variants 与 Family 顺序

- □ 比较 no persistence、native Hermes、native + ledger、static LightRSI 和 adaptive LightRSI。
- □ 所有 variant 使用 matched model、judge、budget、task order、sandbox 和 persistence isolation。
- □ 当前依次运行 semantic-relevant memory-ability 和 update-ability families。
- □ Procedural-reuse 只有在 2G 解锁并实现后加入；episodic-specific family 只有在 2F 解锁并实现后加入。
- □ 每增加 family，先冻结 memory mapping、allowed signal、forbidden evidence、expected lifecycle 和 acceptance。
- □ 只在核心 families 稳定后决定是否运行完整 family set。

验收需求：

- □ 每个 variant 完成预先规定的 independent replicate。
- □ 所有 run 通过 usage、privacy、identity 和 lifecycle audit。
- □ Failed run 与 provider failure 单独报告，不删除。

### 2K.2 Metrics 与 Ablation

- □ 报告 task score、pass rate、persistence gap 和 mechanism evidence。
- □ 报告 model tokens/calls、tools、retry、latency 和 wall time。
- □ 报告 ingestion/generation policy、storage、retrieval、injection、policy update 和 recovery cost。
- □ 报告 artifact/bytes、retrieval/injection/use/non-use、supersession 和 rollback。
- □ 报告 future utility per lifecycle cost 和 cost-quality frontier，但不让 economics 掩盖 quality claim。
- □ Ablate Mem0 internal update、operation-level attribution、unified utility objective、delayed feedback、generation-policy update、retrieval-policy update 和 lifecycle cost。

验收需求：

- □ Raw quantities 与 derived metrics 可从 evidence 重算。
- □ Provider price 变化不要求重新运行。
- □ Adaptive gain 不来自更高 budget、未计费调用或 leakage。
- □ 每个 ablation 只改变一个因素并保持 backend/model/task matched。

### 2K.3 Claim Gate

- `Fixed-route semantic memory optimization`：Hermes native routing 保持不变，semantic 真实闭环通过。
- `Unified memory policy objective`：generation/internal update 与 retrieval outcome 使用同一 future-utility semantics。
- `Operation-attributed policy improvement`：Policy update 可以定位到 extraction、internal decision 或 retrieval operation，而不是只使用无差别 episode reward。
- `Memory-mediated self-improvement`：Policy N+1 来自 delayed feedback，并影响未来 task decision。
- `Recursive self-improvement`：至少两次可重放 policy iteration，且新 evidence 继续进入下一轮更新。
- `Generalization within PAST-Bench`：多个预先选定 family 在同一 Hermes backend 完成 matched evaluation。

不满足对应 gate 时，删除或弱化相关 claim。

### 2K.4 第二阶段完成条件

- □ 当前主线 2A-2E、2H-2K 均有 commit、tests 和 evidence；2F/2G 只有解锁并纳入论文 claim 后才计入完成条件。
- □ Static/adaptive variants 完成 matched PAST-Bench evaluation。
- □ Lifecycle 与 usage 可重建且无 leakage。
- □ Policy proposal、activation 和 rollback 可重放。
- □ 论文 claim 与实际通过的 gate 一致。
- □ 未接入范围外 backend、host 或 benchmark。

## 23. 总体串行执行规则

- 当前 semantic-first 主线严格按照 `1A -> 1B -> 1C -> 1D -> 1E -> 2A -> 2B -> 2C -> 2D -> 2E -> 2H -> 2I -> 2J -> 2K` 推进。
- 2F/2G 是延后研究闸门，不阻塞 semantic 主线；只有选定方法、冻结 matched baseline 并更新 checklist 后才进入实现。
- 同一工作块内严格按子任务编号推进。
- 每个子任务使用独立 commit，包含实现、failure semantics、测试和验收材料。
- 当前子任务未通过时，不提前实现后续任务。
- 发现 contract 缺陷时，先更新当前 contract 和 checklist，不通过后续模块绕开问题。
- 不只完成 happy path，也不能把 failure、restart、privacy、accounting 或 leakage test 留到以后。
- 持续更新 checklist；只有功能、测试、证据和文档全部通过后，才勾选对应任务。

## 24. 单个任务记录模板

```text
Task ID:
Objective:
In scope:
Out of scope:
Modified modules:
Contract/version changes:
Prompt/policy changes:
Configuration and default:
Failure semantics:
Transaction/restart semantics:
Privacy and leakage impact:
Accounting impact:
Focused tests:
Negative tests:
Integration fixture:
Full regression result:
Experiment evidence:
Known limitations:
Commit:
```
