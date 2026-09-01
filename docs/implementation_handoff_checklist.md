# RSIMem Semantic Memory Policy Adaptation Checklist

最后更新：2026-09-01

## 1. 文档定位

本文是 RSIMem 当前唯一的实现与验收主清单。旧版 checklist 已由 Git 历史保留，不再在本文重复历史修复流水账、过期测试计数和已失效的实验判断。

当前只回答一个核心研究问题：

> 在不读取 grader、标准答案、hidden expectation 和 official score 的情况下，能否根据形成记忆时的上下文，以及后续真实发生的持久化、检索、注入、工具调用和任务结果，识别 semantic memory extraction policy 的问题，并生成可泛化的 policy 修改？

执行严格分为三个串行阶段：

1. **阶段一：补齐通用 process attribution 基建。** 将 pure process、benchmark audit 和 final evaluation 分离，补齐 opportunity/use、set-level provenance、tool exact join 和 logical case identity。
2. **阶段二：验证真实 process signal 是否足够。** 在 PAST-Bench/Hermes 上重新收集 clean parent evidence，逐 case 判断可观测、可诊断和可优化；这一阶段允许得到“信号不足”的否定结论。
3. **阶段三：运行 extraction prompt N+1 在线适应实验。** 只有阶段二找到可信且可泛化的优化信号后，才允许生成 candidate、做 held-out validation、激活并运行 matched effect experiment。

状态约定：

- `√`：当前范围内已经有代码、回归和可审计证据。
- `□`：尚未完成，是当前或后续串行任务。
- `进行中`：已有部分实现，但还不能作为稳定依赖。
- `延后`：不属于当前 semantic extraction 主线。
- `撤销`：历史 artifact 或结论已被后续审计判定无效，不得复用。

## 2. 冻结范围与 Claim

### 2.1 当前研究范围

- Benchmark 只使用 vendored PAST-Bench。
- Host 只使用 Hermes。
- 第一种 memory 只研究 semantic memory。
- Backend 使用 Hermes native semantic storage，即 `MEMORY.md` / `USER.md`。
- Formation implementation 使用 RSIMem 内的 Mem0-flat semantic path，不在运行时 import MemBase。
- 当前唯一开放的 learned policy 是 semantic extraction prompt。
- Trigger、source selection、internal update/admission、commit scheduling、retrieval/exposure、route、backend 和模型参数在第一轮 effect experiment 中全部固定。
- Cost、token、latency 和 storage 只用于 accounting/report，不进入 policy learning signal。

### 2.2 当前不做的内容

- 延后：episodic memory policy。
- 延后：procedural/skill memory policy。
- 延后：context eviction、physical rewrite 和 token-saving optimization。
- 延后：retrieval threshold、retrieval prompt 和 update prompt adaptation。
- 延后：六层 policy 联合优化。
- 延后：Codex、Claude Code、OpenClaw 和其他 Host。
- 延后：其他 benchmark、跨 benchmark 泛化和跨 Host 泛化。
- 延后：N+2、连续多轮 recursive self-improvement 和长期在线部署。

### 2.3 最低可发表 Claim

第一条必须先证明的 claim 是：

> Deployment-visible process evidence 中存在能够诊断 semantic extraction policy 缺陷、并指导通用 prompt 修改的信号，而不需要访问 benchmark grader 或未来标准答案。

只有阶段三完成后，才能进一步声明：

> RSIMem 使用过去 deployment process 生成并验证 extraction prompt N+1，N+1 在未来 matched tasks 中改变了 memory formation behavior，并取得更好的任务结果。

当前不得声明：

- 六层 policy 已经 optimization-ready。
- 已经完成 recursive self-improvement。
- 旧 SM02 candidate 是可信 N+1。
- 当前 process signal 已经足够。
- deterministic fixture 等价于真实 provider effect。

## 3. 三个证据平面

### 3.1 Pure Process Plane

Pure process plane 只能读取真实部署自然可见的信息：

```text
source context
-> extraction request/output
-> persisted memory or empty result
-> retrieval query/result
-> exposure/injection
-> tool call/result
-> task completion or other observable outcome
```

允许内容：

- 用户、Assistant 和 tool 真实可见的 bounded context，其中 durable source eligibility 只能由可信 role/evidence 判断。
- Extraction 输出、过滤状态、mutation、artifact identity 和 revision。
- Retrieval、exposure、tool execution、用户纠正、重试、取消和任务完成状态。
- Application-owned、公开且版本化的业务 policy schema，例如资源访问权限或 API 成功条件。

禁止内容：

- PAST-Bench family ID 被用来查找正确答案或自动声明 opportunity。
- `eval_near`、`eval_far` 等 benchmark stage 被当作任务语义。
- Grader、answer key、hidden expectation、official score 或 judge feedback。
- Validation/test 中尚未发生的 future evidence。
- Cost/accounting 数值作为 optimizer reward。

### 3.2 Benchmark Audit Plane

Benchmark audit plane 可以使用预注册的 PAST-Bench family contract，检查 instrumentation 和 attribution implementation 是否符合已知任务设计。它可以回答“在已知 SM02 规则的情况下，resolver 是否错误地产生 missed”，但不能回答“真实部署能否自动发现这条规则”。

要求：

- √ 所有依赖 `family_id`、benchmark stage、TSV parser、固定 recipient 或固定日期的 label 显式标记为 `benchmark_audit_only`。
- √ `benchmark_audit_only` evidence 不得进入通用 optimizer 的 high-confidence bucket。
- √ Audit report 必须分别报告 pure-process diagnosis 和 benchmark-contract diagnosis，不能合成一个 useful/missed 数字。

### 3.3 Final Evaluation Plane

Final evaluation plane 可以在 candidate 冻结、实验完成后读取 official score，只用于论文效果评估。

要求：

- √ Learner、proposal、offline safety 和 activation API 无法访问 grader、answer key 和 official score。
- √ Final reporter 明确记录 score 的读取时间晚于 candidate freeze 和 run completion。
- √ Official score 不回写 process corpus、optimizer corpus 或下一轮 proposal input。

## 4. 当前可复用资产

以下内容已经完成，不重新实现：

- √ PAST-Bench/Hermes vendoring、安装、preflight、runner 和 usage accounting。
- √ Semantic memory backend contract 和 Hermes native semantic adapter。
- √ Mem0-flat extraction、internal `ADD/UPDATE/DELETE/NONE` 和 semantic mutation path。
- √ `ContextSnapshot`、source projection、tool closure、revision、provenance 和 idempotency。
- √ Transaction validation、CAS、receipt、reread verification、restart recovery 和 rollback。
- √ Extraction prompt slot、host-neutral adapter、artifact store、ACTIVE pointer 和 runtime binding fingerprint。
- √ Content-free operation graph、policy decision ledger、process corpus 和 crash-safe stores。
- √ Extraction source/capture、owner-controlled content-bearing corpus 和 content-free exact identity 基础。
- √ Candidate schema、prompt safety、anti-copy、anti-shortcut、offline validation 和 activation/rollback contract。
- √ 六层 policy 的 deterministic/shadow contract、intervention 和 replay fixture。
- √ `unresolved`、`censored`、`not_exposed`、`injected_not_used`、`retrieval_miss` 等保守状态语义。

这些资产只能证明 runtime 和实验框架具备承载能力，不能证明真实 process attribution 或 online adaptation 已经成功。

## 5. 已知失效证据

- 撤销：`s1-sm02-feedback-20260829-v5` 的 8 条 historical `missed`。
- 撤销：`sm02-feedback-20260829-rerun-main` 的 12 条 historical `missed`。
- 撤销：基于上述 SM02 missed 生成的 proposal candidate 和后续 offline-validation 语义结论。
- 撤销：`s1-sm05-feedback-20260829-v1` 的 24 条 historical `missed`。
- 撤销：旧 process census 中“Extraction 已 optimization-ready”的结论。
- 保留：SM01 的 24 条 `unresolved`，它是合法的 no-signal pilot，不是负样本。
- 保留：失败 provider/runtime attempts，只用于 infrastructure diagnosis，不进入质量统计。

旧 artifact 保持 append-only，不原地改写。新代码不得读取它们生成 proposal、validation input 或论文数字。

## 6. 阶段一：补齐通用 Process Attribution 基建

阶段一完成前，不运行新的正式 proposal、held-out validation 或 adaptive effect batch。

### 1A：分离三个证据平面

功能需求：

- √ 为 evidence 和 label 增加明确的 plane/source identity：`pure_process`、`benchmark_audit`、`final_evaluation`。
- √ Process corpus writer 禁止写入 family-specific semantic label、grader 字段和 official score。
- √ Benchmark adapter 可以保留 family contract，但输出必须标记 `benchmark_audit_only`。
- √ Optimizer input builder 默认只接受 pure-process evidence；benchmark audit 只能作为诊断注释或预注册安全约束。
- √ Final reporter 与 learner 使用不同数据类型和读取入口。

反向验收：

- √ 删除 family ID 和 stage 后，pure process corpus 仍可完整重建。
- √ 将 benchmark audit label 传给通用 high-confidence learner 时必须 fail closed。
- √ 将 official score 注入 process/optimizer payload 时 schema 必须拒绝。

### 1B：通用 Opportunity Contract

当前问题：Hermes bridge 根据 `family_id + stage` 直接把 contract 的 `memory_scope_keys` 写入 `task_semantic_keys`，这相当于预先告诉 resolver 当前 future task 需要什么。

功能需求：

- √ Opportunity 必须由 deployment-visible evidence 产生：当前输入、environment/resource state、tool schema、用户请求或 application-owned policy schema。
- √ 定义 host-neutral `OpportunityEvidence`，至少包含 evidence ID、source surface、semantic requirement、observation time、operation ID 和 provenance。
- √ Application-owned schema 必须公开、版本化并在 run 前冻结，不能从 grader 或 family lookup 动态构造。
- √ 没有可验证 opportunity 时保持 `unresolved`，不能因为任务失败自动创建 missed。
- √ PAST-Bench adapter 第一版可提供公开 task/tool schema adapter，但不能传入 hidden expectation 或标准答案。

反向验收：

- √ 仅改变 `family_id` 或 `stage` 不得改变 pure-process opportunity。
- √ 当前输入已重新提供同一信息时标记 `current_input_confounded`。
- √ 只有 task completion、没有语义需求时不得生成 opportunity。
- √ 使用 hidden grader 构造 opportunity 的 fixture 必须被拒绝。

### 1C：通用 Use 与 Outcome Attribution

当前问题：最终输出出现 TSV 或工具调用没有违规，不等于 Agent 使用了某条过去 memory。

功能需求：

- √ 定义 `MemoryUseEvidence`，显式绑定过去 artifact/artifact set、retrieval、injection、downstream operation 和 observation cutoff。
- √ 区分 exposure、behavioral consistency 和 attributable use；前两者不能自动升级为 use。
- √ 定义 application-observable outcome contract，例如 tool success、state transition、用户确认或任务 completion；纯字符串匹配默认只能作为 weak hypothesis。
- √ Use/outcome exact join 缺任何节点时保持 `unresolved`。
- √ 其他 source 写入的 memory 可以解释行为或阻止 false missed，但不能替当前 extraction 获得 useful/harmful credit。

反向验收：

- √ 其他 source artifact 不能替当前 extraction 获得 useful credit。
- √ Future binding 不能为当前 source artifact 伪造 semantic key。
- √ 仅 eager injection、仅最终格式匹配、仅 task completion 均不能生成 attributable use。
- √ Tool failure、Agent non-use 和 retrieval failure 分别归入自己的 component，不惩罚 extraction。

### 1D：Set-Level Semantic Provenance

当前问题：一个 durable rule 可能被 extraction 拆成多个 facts；SM02 已证明单 fact matcher 会产生 false missed，而把完整 semantic key 复制给每个 fact 又会产生 false useful。

功能需求：

- √ 定义版本化 `ArtifactSetSemanticBinding`，包含 semantic unit ID、member artifact IDs、member fact IDs、completeness、source digest 和 provenance。
- √ Set binding 只能由 extraction-set evidence 或显式 matcher 产生，不能通过简单字符串拼接后把 key 复制给所有 member。
- √ 只有完整 member set 被检索并曝光时才允许绑定联合规则；partial retrieval/exposure 保持 `unresolved`。
- √ Set-level useful/missed 只计算一个 primary unit，不按 member fact 数重复奖励或惩罚。
- √ Matcher 如使用 LLM，必须独立版本化、冻结、禁止访问 grader/answer，并输出可审计 equivalence evidence。

反向验收：

- √ 未分类的非空 extraction 不得生成 high-confidence missed。
- √ 外部传入 stale missed evidence 不能绕过 builder gate。
- √ Partial/unclassified exposure 不得生成 useful。
- √ 三个 facts 共同表达完整规则且完整曝光时可以生成一个 set-level evidence。
- √ 缺少任一 member、member 来自其他 source 或 matcher ambiguous 时保持 unresolved。

### 1E：Tool Call/Result Exact Join

当前问题：owner evidence 能看到部分 `notes_*` 行为，但统一 process corpus 还没有通用 tool call/result closure 和 memory-use join。

功能需求：

- √ 每次真实 tool call 和 result 都生成独立 content-free `ProcessEvent`。
- √ Event 至少包含 call ID、result ID、tool-name digest、success/failure、retry identity、host event、task/session、source revision、policy lineage 和 receipt identity。
- √ Call/result 必须一对一闭合；missing、duplicate、orphaned 和 type mismatch fail closed 或明确 censored。
- √ 原始 arguments/returns 只保存在 owner-controlled evidence；公共 corpus 只保存允许字段、digest 和必要的结构化状态。
- √ Memory exposure/use operation 可以 exact join 到其后的 tool call/result，不能只依赖同一 task ID。

反向验收：

- √ 成功、失败、重试、缺失 result、重复 call ID 和跨 task call/result 均有 fixture。
- √ `notes_share` 之外至少有一个不同工具协议 fixture，证明 contract 不是 family-specific。
- √ Tool result 中的原始用户内容和凭据不会进入 content-free corpus。

### 1F：Logical Case Identity 与统计独立性

当前问题：同一 source/future 组合会因多个 retrieval boundary 和 replicate 生成多条 primary feedback，physical count 不能当作独立样本量。

功能需求：

- √ 定义稳定 `logical_case_id`，至少绑定 frozen policy、source task/template、source extraction set、future task/template 和 observation window。
- √ Request-level future boundary 使用独立 observation ID，但属于同一 semantic case 时共享 logical case ID。
- √ Replicate ID 不进入 logical semantic identity，只进入 physical observation identity。
- √ Analyzer 同时报告 physical observations、logical cases、replicate consistency 和 conflict rate。
- √ Proposal builder 按 logical case 加权，不能按重复 observation 或 fact 数放大奖励。

反向验收：

- √ 同一 case 三个 replicate 只计一个 logical case 和三个 physical observations。
- √ 同一 task 中重复 retrieval 不增加 semantic sample count。
- √ 不同 source、future task 或 frozen policy 必须产生不同 logical case ID。
- √ Replicate 结果冲突时 case 标记 ambiguous，不通过多数投票静默制造标签。

### 1G：Stale Artifact 与 Schema 迁移

功能需求：

- √ Feedback、optimizer corpus、candidate 和 validation artifact 记录 attribution schema version 和 evidence plane。
- √ 旧 schema artifact 默认拒绝，不按新语义静默读取。
- √ 为已撤销 batch/candidate 建立 content-free revocation registry 或等价 denylist identity。
- √ Proposal、preflight 和 activation 在模型调用前拒绝 stale/revoked evidence。
- √ 历史报告保留，但新汇总必须引用 [`case_analysis.md`](case_analysis.md) 的后续审计结论。

反向验收：

- √ 旧 SM02/SM05 corpus 不能进入 proposal builder。
- √ 旧 SM02 candidate 不能进入 validation 或 ACTIVE pointer。
- √ Revocation registry 损坏、缺失或 identity 冲突时 fail closed。

这里的历史身份验收分为两层：仓库外保留的旧 raw 文件首先因 schema
不兼容而被拒绝；`configs/revocations.jsonl` 再以 content-free
ID/digest denylist 覆盖这些身份在当前 schema 下被重新呈现的情况。当前
测试验证了两层行为，但没有把 ignored raw output 当作可运行的训练输入。

当前 denylist 使用 `rsimem-revocation-registry-v2`。历史 artifact 使用
`scope=legacy_untyped`，只保留 artifact ID、artifact schema version 和
digest，`evidence_plane`/`evidence_source` 明确为 `null`，因此不会为旧
artifact 猜测不存在的 provenance；这类身份会在任意 typed lookup 下 fail
closed。新建 revocation entry 使用 `scope=typed`，必须提供合法的
evidence plane/source。registry、lock、symlink、malformed record 和 identity
conflict 均继续 fail closed。

### 1H：阶段一关闭条件

- √ 1A-1G 的功能和反向测试全部通过。
- √ Pure process corpus 在没有 family ID、stage 和 grader 的情况下可以重建完整 lifecycle。
- √ Opportunity、use、outcome、artifact set 和 tool event 都有稳定 operation/provenance join。
- √ Analyzer 能正确区分 stage diagnosis、benchmark audit label 和 pure-process attribution。
- √ Stale artifact 无法进入 learner、validation 和 activation。
- √ RSIMem、PAST-Bench、compileall、dependency、shell syntax、secret scan 和 restart fixture 全部通过。

## 7. 阶段二：真实 Process-Signal 可行性验收

阶段二的目标不是生成 candidate，而是判断 process signal 是否真实存在。结果可以是通过、部分通过或不通过。

### 2A：冻结分析协议

- √ 在看新结果前冻结训练 families、task-template groups、provider/model、replicate、observation window 和 case dedup 规则；`ProcessSignalAnalysisProtocol` 在任务运行前由 manifest 派生并冻结。
- √ 第一轮仍优先 SM02，但只把 application-visible recipient/resource schema 当作 opportunity source，不使用 family lookup 自动填 key。
- √ SM01/SM05 作为 preference/output 稀疏信号对照，不把格式失败自动归因到 extraction。
- √ 旧 SM01/SM02/SM05 raw batches 只用于 bug audit，不复用为新 corpus；schema/revocation gate 会在 proposal、validation 和 activation 前拒绝它们。
- √ Pre-register 至少一个无信号预期 case；协议固定 `case.no_signal.v1`，SM01 unresolved pilot 用于验证 resolver 不强制产出标签。

### 2B：收集 Clean Parent Evidence

- √ 使用 fixed parent extraction prompt，其他所有 policy layer 固定。
- √ 每个正式 batch 要求 clean RSIMem tree、clean PAST-Bench tree、唯一 batch ID、固定 manifest 和 append-only attempts。
- √ Provider probe 只判断可运行性，不进入 task quality 或 process label。
- √ 每个 logical case 收集 source、extraction、persisted memory、retrieval、exposure、tool call/result 和 observable outcome。
- √ 每个失败/partial run 保留 diagnosis，但不进入完整 observation denominator。

2B 的 clean parent evidence 见 [`extraction_stage2_production_reruns_20260831.md`](extraction_stage2_production_reruns_20260831.md) 和本次 fresh rerun 报告 [`extraction_stage2_clean_parent_20260901.md`](extraction_stage2_clean_parent_20260901.md)。两组新 batch 均完成 audit；它们仍只提供 process observability，不提供 extraction-owned signal。

### 2C：逐 Case 三层分析

每个 logical case 必须回答：

1. **可观测**：source、extraction、persisted memory、retrieval/exposure 和行为结果是否全部可重建？
2. **可诊断**：能否排除 retrieval、exposure、Agent non-use、tool failure、current-input confounding 和其他 memory 的影响，将问题归因到 extraction？
3. **可优化**：能否把诊断抽象为不包含 task ID、具体答案、recipient 名称、文件名和 family 规则的 prompt 修改？

当前 clean rerun 已实现并持久化这些 case 状态；SM02/SM05 的现有 case 分别为 `observable_only`、`diagnostic_only` 或 `censored`，没有 `optimization_signal`。因此 2C 的 case classification 已完成，2D 的 signal sufficiency 仍未通过。

每个 case 的状态只能是：

- `optimization_signal`：三层全部通过。
- `diagnostic_only`：可观测且能定位 stage，但不能生成通用 policy edit。
- `observable_only`：只能重建过程。
- `censored`：证据窗口不完整。
- `invalid`：evidence identity、privacy 或 schema 不可信。

√ 当前 clean parent rerun 已为每个 logical case 持久化 source、extraction、
persistence、retrieval、exposure、outcome、stage-diagnosis、replicate 和
observation-window 字段，并据此给出三层判定；现有 cases 均未达到
`optimization_signal`。

### 2D：信号充分性 Gate

阶段二通过至少需要：

- □ 至少一个 logical case 提供可信 extraction-owned diagnosis（当前未满足；现有 cases 没有完整 artifact/use/outcome chain）。
- □ 至少一个不同 logical case 支持相同或兼容的抽象 policy 修改，避免单例过拟合（当前未满足，因为上一项未满足）。
- □ 该修改不包含 benchmark shortcut，并通过 source-value、task-ID 和长 n-gram safety gate（候选尚未生成，暂不适用）。
- √ Replicate consistency 可报告，且没有由重复 observation 伪造样本量。
- √ 至少一个 no-signal/ambiguous case 被正确保留为 unresolved，证明 resolver 不会强制产出标签。

若不满足，阶段二结论必须是以下之一：

- Process observability 成立，但 extraction attribution signal 不足。
- 当前 family 的 signal 不适合 extraction，可换预注册 family 再做一次有限尝试。
- Deployment process 本身不足以支持当前 self-improvement claim，需要缩小论文 claim 或增加合法 application feedback。

不得通过修改 grader、降低 attribution gate、把 unresolved 变成 harmful/missed 或在看结果后增加 family-specific parser 来通过阶段二。

### 2E：阶段二交付物

- √ 更新 [`case_analysis.md`](case_analysis.md)，按 logical case 给出三层结论和 evidence index。
- √ 生成 pure-process signal census，与 benchmark-audit census 分开。
- √ 记录所有可泛化 policy hypotheses 及其支持/反对 cases（当前没有获得可信 hypothesis）。
- √ 明确给出 `STOP_NO_SIGNAL` 决策；`GO_TO_N_PLUS_1` 未解锁。

## 8. 阶段三：Extraction Prompt N+1 在线适应

只有阶段二为 `GO_TO_N_PLUS_1` 时执行。

### 3A：Candidate 生成

- √ Optimizer 只读取冻结时间点之前的 training pure-process evidence；pure corpus store 只向 optimizer 暴露 `train` 且已通过 signal/revocation gate 的 corpus。
- √ 输入包含 parent prompt、bounded source/process evidence、stage diagnosis 和 logical-case weighting。
- √ Benchmark audit evidence 只能作为安全约束，不能提供具体答案或 family shortcut。
- √ Candidate 只修改 extraction prompt body，update、retrieval、trigger、source、admission、commit 和 exposure 全部保持冻结。
- √ Proposal budget、重试次数和 candidate selection 数量在运行前冻结。

### 3B：Static Safety 与 Offline Validation

- √ Candidate 保持 extraction JSON schema、grounding、durability、privacy 和 no-transcript-copy contract。
- √ Candidate 不包含训练 source 具体值、task ID、family ID、固定答案或长 n-gram。
- √ Candidate 通过 deterministic extraction regression，但 fixture 结果不计为真实效果。
- √ Artifact、parent lineage、slot、model profile、wrapper、update prompt 和 retrieval config identity 完整。
- √ Safety 或 evidence provenance 任一 unknown/invalid 时 reject。

3A/3B 的 √ 只表示 proposal/safety/offline contract 和 replay 已验收；由于阶段二仍为 `STOP_NO_SIGNAL`，当前没有 live N+1 candidate 可进入后续实验。

### 3C：Held-Out Validation

- √ Train、validation 和 final test 按 family/task-template group 隔离，同一 manifest digest 不跨 split。
- √ Parent/candidate 使用相同 model、budget、sandbox、persistence、Host 和非 extraction policy；matched preflight 会拒绝 drift。
- □ Candidate 必须实际被渲染，并至少改变一次 extraction output 或 persisted memory behavior。
- □ Validation 同时报告 task-level end-to-end outcome、pure-process diagnosis、benchmark audit 和 unresolved/censored。
- □ Official score 仅在 run 完成后由 final reporter 读取。
- □ Validation 只按预注册 budget 执行，不能反复查看 held-out 后修改 prompt。

### 3D：Activation、Matched Effect 与 Rollback

- √ 只有 held-out validation 通过后才写 ACTIVE pointer。
- √ Restart 后 runtime 加载同一 N+1 artifact 和 binding fingerprint。
- □ Future matched batch 对比 static N 与 adaptive N+1。
- □ Primary paper outcome 使用 PAST-Bench 正式 task metric；process evidence用于解释机制，不替代效果指标。
- □ 同时报告 coverage、empty extraction、harmful、non-use、unknown usage 和 raw resource vector。
- √ N+1 无提升、行为未改变、证据不完整或 safety regression 时自动 reject/rollback；当前仅有 deterministic/replay 证据。

### 3E：阶段三完成条件

- □ `N -> past process evidence -> abstract hypothesis -> N+1` lineage 可重建。
- □ N+1 只改变 extraction prompt，并在真实 future run 中改变 formation behavior。
- □ Held-out 和 final matched evidence 不存在 template leakage。
- □ Adaptive N+1 的预注册 primary outcome 高于 static N，且安全/coverage gate 不退化。
- □ 结果可以在 clean tree 和 isolated temporary HOME 下复现。

完成后只声明一次 observed online extraction-policy adaptation，不声明 N+2 或通用 recursive self-improvement。

## 9. 当前问题清单

| 优先级 | 问题 | 当前影响 | 对应任务 |
| --- | --- | --- | --- |
| P0 | Pure process 与 benchmark family contract 混合 | 无法证明信号来自真实 deployment process | 1A、1B |
| P0 | Opportunity 由 family/stage 自动声明 | 存在 benchmark semantic leakage | 1B |
| P0 | Use 主要由最终字符串或 family parser 推断 | 无法证明过去 memory 导致行为 | 1C |
| P0 | 跨 fact semantic unit 无显式完整性 | SM02 产生 false missed，简单修复又可能产生 false useful | 1D |
| P0 | Tool call/result 未完整进入统一 process corpus | 无法排除 tool failure 或完成 exact attribution | 1E |
| P0 | 缺少 logical case identity | Physical feedback 被误当作独立样本 | 1F |
| P0 | 旧 corpus/candidate 未有统一 revocation gate | 失效 evidence 可能被后续脚本误用 | 1G |
| P1 | 真实 provider runs 没有可信 extraction-owned signal | 不能生成 N+1 | 阶段二 |
| P1 | 真实 runs 没有 candidate action variation | Deterministic feasibility 不能证明真实 effect | 阶段三 |
| P2 | Trigger/Source/Admission/Commit/Exposure 只有 shadow feasibility | 不支持六层联合 claim | 延后 |

## 10. 标准验收命令

在 RSIMem 根目录运行：

```bash
.venv/bin/python -m pytest -q tests
.venv/bin/python -m compileall -q src tests
.venv/bin/python -m rsimem.secret_scan
git diff --check
bash -n scripts/*.sh
```

在 vendored PAST-Bench 目录运行：

```bash
../../.venv/bin/pytest -q
```

每个阶段还必须运行：

- 对应 focused tests。
- Tracked-secret scan。
- Isolated temporary-HOME restart fixture。
- Artifact schema/revocation preflight。
- 正式 provider batch 前的 completion probe。

不要从 RSIMem 根目录运行 `pytest benchmarks/past-bench`；该方式可能错误解析 Hermes-plus 的顶层 `agent` package。

## 11. 当前状态

### 当前口径（2026-09-01）

本节以下内容是当前执行口径；前文的“问题清单”保留为阶段性风险
登记，不能单独解读为这些问题仍未修复。

Provider gate、验证基线和下一批恢复顺序的完整记录见
[`current_checkpoint_20260901.md`](current_checkpoint_20260901.md)。

- 已完成：pure-process、benchmark-audit、final-evaluation 三个证据平面
  的 contract、持久化、重放和 revocation 边界。
- 已完成：Hermes 任务完成后的 source -> future observation -> feedback ->
  process-signal 自动接线；source、feedback、event archive 和 capture
  均使用 append-only/restart-safe 存储。
- 已完成：semantic、episodic、procedural 的 Hermes storage-boundary
  adapter fixture；adapter semantic exposure 复用同一次检索命中，native
  bypass 不会伪造 RSIMem future trace。
- 已完成：六层 policy 的 deterministic/shadow contract、action variation、
  replay case 和机制解释；这只证明 feasibility，不证明真实收益。
- 六层当前口径：Extraction 是第一个允许在 future signal gate 满足后尝试
  N+1 的层；Trigger、Source selection、Admission、Commit 和 Exposure 仍为
  diagnostic/validation-only，不做联合 policy 干预。
- 当前结论：SM02/SM05 clean parent process-signal census 均为
  `STOP_NO_SIGNAL`，因此不生成 N+1，也不启动 held-out 或 adaptive effect
  experiment。
- 当前状态：primary provider bounded probe 和 SM02/SM05 fresh clean-parent
  batches 均完成；两组结果都是 `STOP_NO_SIGNAL`，未生成 candidate。任何
  后续 batch 仍须在注册后、首个 task 前重新执行 probe。
- 下一步：provider 恢复后重新执行预注册 clean parent batch，并重新核验
  source、extraction、persisted memory、retrieval、exposure、tool
  call/result 和 observable outcome 的完整链路。

截至 2026-09-01：

- √ Runtime、semantic writeback、prompt artifact、ledger、replay、安全和 deterministic feasibility 基础可复用。
- √ 已完成现有 SM01、SM02、SM05 case 的后续归因审计，见 [`case_analysis.md`](case_analysis.md)。
- √ 已增加 user-only source detection、conservative missed gate 和 source-specific artifact attribution 防线。
- √ 当前执行入口已分离 pure process、benchmark audit 和 final evaluation；SM02/SM05 clean parent rerun 已生成 replay-stable process-signal census。
- √ 阶段二的 deterministic/process-signal 验收已完成当前两组 train-family rerun；两组均为 `STOP_NO_SIGNAL`，旧 batch 不得作为新 signal census 的输入。
- √ 已完成 3B/3D 的 deterministic/shadow decision contract、process signal、action variation 和可回放 case 整理；这不等于真实 policy intervention 或 uplift。
- □ 真实 N+1、held-out、activation 和 matched effect 仍未开放。
- □ 阶段三未解锁；当前不存在可信、可验证的 N+1 candidate。

最近验证基线：RSIMem `.venv/bin/python -m pytest -q tests` 为 `1101 passed`，vendored PAST-Bench 为 `401 passed, 2 skipped`；compileall、pip check、shell syntax、tracked-secret scan 和 git diff --check 均通过。`configs/revocations.jsonl` 现为 v2 content-free historical denylist seed，formal matched launcher 会在 batch 输出目录创建并锁定副本，避免工作树污染；五个已撤销 SM02/SM05 corpus/candidate identity 均在 proposal、validation 与 activation 边界前 fail closed。最近补强了 policy/process/operation/lifecycle evidence 的重载、锁和 tool-closure fail-closed 验证；adapter semantic exposure 现在复用同一次检索命中，native bypass 不会伪造 RSIMem future trace。最新 provider probe 为 HTTP `200`、非空内容和 usage；SM02/SM05 fresh clean-parent batch 均 audit-clean 但阶段二仍为 `STOP_NO_SIGNAL`，阶段三未解锁。
