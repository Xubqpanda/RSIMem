# RSIMem Memory Policy Online Adaptation 实现与验收清单

最后更新：2026-08-29

## 1. 文档定位

本文是 RSIMem 当前唯一的实现与验收主清单。旧 checklist 中已经完成的 Hermes、PAST-Bench、semantic writeback、transaction、operation graph、feedback store、activation 和 rollback 基础设施继续保留，但不再沿用“future utility per cost + retrieval threshold adaptation”作为论文主线。

当前工作严格分为三个串行阶段：

- 第一阶段：修正现有实现中偏离 extraction-prompt adaptation 的目标、契约、证据和实验 gate。
- 第二阶段：实现六层 memory policy 的可观测、可替换和可验收基建；先固定其余层验证 extraction adaptation，再逐层开放 trigger、admission 和 exposure policy。
- 第三阶段：对六层 policy 做可优化性验收，确认每层是否有足够的 process signal、可控 action space、可回放 intervention 和具体收益 case；只有通过可行性验收的层，才进入后续真实效果实验。

状态约定：

- `□`：尚未完成或尚未通过验收。
- `进行中`：已开始，但不能作为后续任务的稳定依赖。
- `√`：功能、测试、真实证据和文档均已通过。
- `保留`：已有实现可复用，但不代表当前论文核心任务完成。
- `延后`：不属于本轮串行路径。

## 2. 冻结研究范围与 Claim

### 2.1 当前范围

- Benchmark 只使用 vendored PAST-Bench。
- Host 只使用 Hermes。
- Memory backend 只使用 Hermes native semantic storage，即 `MEMORY.md` / `USER.md`。
- Memory algorithm 使用 RSIMem 内重写的 Mem0-flat semantic path，不在运行时 import MemBase。
- 最终研究对象是 semantic memory formation and exposure policy；第一轮实验先只开放 extraction policy，随后按单层增量顺序开放其余 policy。
- Extraction optimization core必须与memory backend和Host解耦；Mem0-flat/Hermes只是第一个adapter和实验载体，不属于optimizer contract。
- 第一轮实验固定 Internal `ADD/UPDATE/DELETE/NONE` prompt、related-memory retrieval、future retrieval/injection surface、route、invocation boundary、backend 和基础模型参数；第二阶段只为后续打开这些 policy 的受控接口，不在没有 matched ablation 时同时改变它们。
- Episodic memory、procedural memory、context eviction、physical rewrite、其他 host 和其他 benchmark 全部延后。

### 2.2 当前方法定义

```text
completed Hermes experience
  -> frozen semantic compilation boundary
  -> extraction prompt N
  -> fixed Mem0-flat update/writeback
  -> future eager exposure / observable use / downstream outcome
  -> delayed end-to-end feedback + strict attribution diagnostics
  -> extraction-only proposal N+1
  -> held-out validation
  -> production activation
  -> future matched PAST-Bench evaluation
```

### 2.3 Policy Update Signal

当前反馈分成两个用途不同的通道：

- **Exploratory end-to-end feedback**：使用训练 split 内的完整执行轨迹、memory 状态、exposure 行为和 deployment-observable outcome，让 optimizer 探索 extraction-only 的改进方向。它可以包含 retrieval 和 agent 行为噪声，只能作为 hypothesis/proposal，不能直接把整条失败轨迹标成 extraction error。
- **Strict attribution feedback**：使用版本化 family contract 和 deterministic resolver 生成 `useful/harmful/missed/unresolved/censored`，只把证据链完整且能归因到 extraction 的样本作为高置信诊断、约束或训练 evidence。

允许进入 exploratory extraction optimizer 的训练信号：

- Extraction 当时可见的 bounded completed context。
- Policy N 实际生成的 extracted fact set。
- Persisted memory、exposure/injection、retrieval result、tool calls/results、最终回答和 deployment-observable task outcome。
- Strict resolver 生成的 useful、harmful、missed、unresolved 和 censored evidence，以及 stage diagnosis 和 attribution confidence。
- Training split 内预声明允许使用的 task-level end-to-end outcome；它用于提出候选和选择后续实验方向，不伪装成 extraction-owned label。

允许进入 formal extraction optimizer 的高置信信号：

- Fact 或 extraction set 后续是否有 exposure opportunity。
- 后续是否被注入、显式使用、supersede、证明冲突或关联到 deployment-observable outcome。
- 受 observation cutoff、censoring 和 attribution confidence 约束的 missed-extraction evidence。
- 只有 source、extraction、persistence、exposure、use 和 outcome 链条满足对应 contract 时，才进入 resolved attribution bucket。

禁止进入 optimizer 的信号：

- Official PAST-Bench score、grader、answer key、hidden expectation 或 judge feedback。
- Validation/test batch 中尚未到达的 future evidence。
- Model calls、tokens、latency、storage、recovery 或任何合成 cost unit。
- Route selection、invocation schedule、backend selection 或模型参数更新。
- 未在 manifest 中声明为 deployment-observable 的隐藏标签，不能通过字段改名绕过 future-test 隔离。

### 2.4 Cost 与效果边界

- Cost 只属于 evaluation/accounting plane，不属于 policy learning plane。
- 正式报告保留 model calls、各 token bucket、retry、wall time、storage bytes、injected chars 和 recovery duration 等 raw vector。
- 不把 request、token、byte、character 和 millisecond 直接相加为可用于结论的 `lifecycleCostUnits`。
- Provider 未返回的 usage bucket 保持 unknown；当前廉价 provider 的 unknown 不阻塞功能开发，正式实验使用支持完整 usage 的 provider 或在报告中保留 unknown。

### 2.5 当前最低 Claim

当前阶段的最低目标不是一次性完成六层 online optimization，而是证明六层 policy 是否具备可优化性：

1. 每层都有明确的 host-neutral decision contract和真实/确定性输入。
2. 每层至少有一个 parent/candidate decision对，且两者只改变该层行为。
3. 每层都有不依赖 hidden grader 的 process feedback，能够观察 intervention 后的状态或行为差异。
4. 每层至少给出一个具体 case，说明该层的 policy decision可能改善 memory formation、memory exposure或执行可靠性。
5. Decision、feedback和结果可以跨 restart 重建并通过 safety gate。

完成以上条件只能声明 `optimization-ready`，不能声明六层已经取得真实 aggregate uplift。

在可行性验收通过后，第一条真实效果目标才是一次 online extraction-policy adaptation：

1. Policy N 在过去 deployment 中产生可审计 delayed feedback。
2. 只使用过去 evidence 生成 extraction prompt N+1。
3. N+1 通过独立 held-out validation 后被激活。
4. 未来 run 实际加载 N+1，并至少改变一次 extraction output 或 persisted memory behavior。
5. Adaptive N+1 的 matched aggregate primary task score 高于 static N。

完成以上条件可以声明 observed online extraction-policy adaptation。N+2、repeated/recursive self-improvement、跨 family 泛化和统计显著 superiority 不属于当前完成条件。

六层联合 policy 的更高 claim 必须另行满足增量 ablation，不能由 extraction-only 的结果直接推出：

```text
固定 Host 能力和安全边界
  -> Trigger policy
  -> Source selection policy
  -> Extraction policy
  -> Admission policy
  -> Commit scheduling policy
  -> Exposure policy
  -> delayed task outcome
```

其中 transaction atomicity、CAS、revision、schema、tool closure、credential boundary 和 rollback 是不可学习的 runtime safety invariant；所谓 commit policy 只允许优化“何时提交/是否延迟/是否重试”等安全边界内的调度，不允许模型改写事务实现。

### 2.6 六层 Policy 的职责边界

六层 policy 的可复用 contract 必须与具体 Host 和 benchmark 分离。Host adapter 提供真实事件、snapshot、revision、memory read/write hook 和安全能力；RSIMem runtime 产生 policy decision；PAST-Bench adapter 只提供 task、future outcome 和 benchmark-specific feedback contract。

| 层级 | 可学习的 decision | 必须由 runtime/adapter 固定的部分 |
| --- | --- | --- |
| Trigger | `RUN/SKIP/DEFER`、评估频率、下一次边界 | 真实 Host event、task state、事件顺序 |
| Source | 本次 formation 使用哪些已完成 context segment | snapshot、message order、tool closure、revision和内容边界 |
| Extraction | 提取哪些 durable facts以及如何表达 | prompt slot、模型调用、输出 schema和source grounding |
| Admission | 候选是否进入 `ADD/NONE/UPDATE` | backend capability、mutation validator和禁止操作 |
| Commit | 立即提交、延迟提交或安全重试 | transaction atomicity、CAS、receipt、rollback和writer identity |
| Exposure | 何时注入、选择哪些 memory、排序、格式和预算 | read/injection hook、协议位置、active/current turn保护 |

六层都必须支持 `observe -> decide -> execute -> record -> replay`，但第一版不要求六层同时由模型学习。每个 decision 至少记录 `decision_id`、`policy_version`、`source_revision`、`reason_codes`、输入 digest、输出 digest和实际执行结果。

### 2.7 第一版 Useful Signal 与优化目标

严格 attribution 通道不让一个通用LLM直接判断“这条memory是否有用”。每个family必须在运行前注册版本化`OpportunityContract`、`UseContract`和`OutcomeContract`，再由统一resolver将deployment-observable evidence映射为label。探索性端到端通道可以让受控LLM阅读完整训练轨迹并提出当前开放层的 policy 修改，但它的判断属于 hypothesis/proposal，不直接成为 `useful/harmful/missed` 标签：

```text
useful
  = observed opportunity
  + explicit use attributable to the extraction output
  + successful observable outcome

harmful
  = extraction-owned unsupported/transient/conflicting memory
  or explicit use followed by a deterministically attributable harmful outcome

missed
  = bounded source中存在可精确绑定的durable information
  + policy N没有生成等价fact
  + future task出现可观测的明确需求
  + failure可归因于该信息缺失而不是retrieval/application/model failure
```

统一判定规则：

- Eager system-prompt injection本身不是opportunity或use；只有future task与memory scope/key匹配时才产生opportunity。
- Use signal必须对应past memory携带而当前future task输入没有重新提供的信息；如果当前用户消息、工具结果或环境状态已经直接给出同一事实/规则，则该行为不能归因于memory，最多形成set-level unresolved evidence。
- `injected_not_used`在严格 attribution 通道默认是`unresolved`；完整轨迹仍进入 exploratory corpus，并记录为 downstream non-use/stage-diagnosis evidence，但不能自动改名为 extraction failure。
- Memory被正确提取和注入但Agent没有使用时，不自动惩罚extraction prompt；该问题归入future application/retrieval component或unresolved。
- Observation window不完整、没有相关future task、多个artifact无法唯一归因、证据冲突或只有模型主观判断时，严格通道分别记为`censored`或`unresolved`，不进入resolved denominator；探索性通道保留原始轨迹，但不增加resolved attribution计数。
- 多fact共同影响一次future outcome且无法拆分贡献时，只生成一个extraction-set-level label；不把同一次成功复制成多个fact-level positive。Fact-level label只在artifact/use/outcome可以唯一绑定时生成。
- Primary optimization unit是一次completed source对应的extraction set及其future opportunity，不是fact数量，防止把一个fact拆成多条来放大reward。

Strict attribution channel 的 diagnostic objective 保持为 resolved observed useful rate：

```text
resolved_useful_rate = useful_set_count / (useful_set_count + harmful_set_count)
observed_harmful_rate = harmful_set_count / nonempty_extraction_set_count
nonempty_coverage = nonempty_extraction_set_count / matched_completed_source_count
empty_extraction_rate = empty_extraction_set_count / matched_completed_source_count
high_confidence_missed_rate = missed_set_count / missed_assessable_source_count
```

所有分母为0时对应metric记为unknown，不能伪装成0。`unresolved`和`censored`不进入resolved useful rate分母，但必须完整报告。Non-empty coverage只用于与同一matched source上的parent比较，不表示每个completed task都应该产生memory；允许candidate正确输出空集合，但不能通过大面积清空输出来刷高useful比例。Proposal不能仅靠少提取或不提取提高比例，activation还必须同时满足以下预注册约束：

- Resolved sample count达到`minimum_resolved_examples`，分母为零时禁止proposal和activation。
- Candidate的`resolved_useful_rate`严格高于parent，最小delta必须在查看validation前冻结且大于0。
- `observed_harmful_rate`不高于parent；unsupported、transient和conflicting extraction分别报告，不能相互抵消。
- Non-empty extraction coverage不低于parent乘以预注册的`minimum_coverage_ratio`，empty extraction rate不得超过预注册上限。
- 高置信度missed extraction rate不高于parent；没有可靠missed contract时该family不得伪造该指标，只能标为unknown。
- Schema、safety、prompt leakage和native-writer contamination failure必须为0。
- Cost、token、latency和storage不进入上述目标或约束，只随实验结果报告。

Exploratory end-to-end channel 不使用上述 resolved useful rate 作为唯一 gate。它使用运行前冻结的 deployment-observable task outcome，例如任务完成、工具调用成功、格式/约束满足和可观察业务结果，并同时报告 extraction、admission、exposure、unresolved 和 censored 状态。其目标是发现“只改变当前开放 policy layer 是否可能改善未来结果”的候选方向，不把所有失败归因给该 layer。

SM01第一版strict contract必须明确：只有预注册的`eval_near/eval_far`报告任务构成TSV preference opportunity；future task输入没有重新提供TSV preference时，合法四列表头及非空数据行才构成memory-specific use signal；task completion及预注册的非grader输出/工具条件构成outcome signal。单artifact时允许fact-level归因，多artifact时最多生成set-level evidence。仅检测到TSV格式而没有memory-specific opportunity，当前turn已直接要求TSV，或只因最终回答缺少TSV，均不能自动把某条已正确提取的memory判为harmful。探索性通道可以把“最终没有合法TSV但任务完成/失败”的完整轨迹交给 optimizer 分析，但必须标记为 end-to-end hypothesis。

### 2.8 两层反馈与候选生成边界

端到端 feedback 首先验证“未来结果是否包含可指导当前 policy 改进的模式”，strict attribution 再验证“该模式是否足以归因给某一层”。两者不能混成一个 label：

```text
raw trajectory + end-to-end outcome
  -> exploratory hypothesis
  -> policy candidate for the currently open layer
  -> offline safety/replay
  -> held-out matched validation

source/extraction/persistence/exposure/use/outcome evidence
  -> strict attribution diagnosis
  -> high-confidence constraint or training evidence
```

因此，`unresolved` 不再阻止 exploratory candidate 生成，但仍然不能增加 resolved useful rate、不能直接成为 negative label，也不能绕过 matched validation。正式 candidate 生成必须记录本次开放的 layer set；如果同时改变多个 layer，必须明确标记为 joint candidate。

### 2.9 Process-First Runtime Feedback

真实开源部署通常没有 benchmark grader、gold answer 或 hidden expectation，因此通用 RSIMem runtime 的主反馈必须是 process feedback，output correctness 只能作为可选的应用层反馈。通用 process event 至少包括：

- Host lifecycle event、task/turn/tool/session boundary和context-pressure状态。
- 本次 trigger 是否 `RUN/SKIP/DEFER`，source选择了哪些segment，是否发生truncation或duplicate suppression。
- Extraction request/output fingerprint、候选fact数量、空结果、过滤结果和 admission `ADD/NONE/UPDATE`。
- Commit success/rejection/duplicate、revision/CAS结果、rollback和恢复状态。
- Retrieval query/result、selected artifact、exposure mode、injection位置/预算和context revision。
- Tool call/result、失败/重试、重复询问、用户纠正、人工接管、任务取消和可观察完成状态。

Process event本身只能说明“发生了什么”，不自动说明语义上是否正确。Runtime可以用它产生 weak reward、stage diagnosis 和 policy hypothesis；只有应用提供的显式反馈或无隐藏答案依赖的 contract 才能提升为高置信 label。PAST-Bench grader只由最终 reporter读取，不能进入通用 policy learner。

验收要求：

- √ 每个 process event都能绑定到 Host event、policy decision、source revision和实际执行receipt（pending/skip/defer 决策按其非执行语义保留无 runtime receipt，并由 policy decision identity 绑定）。
- √ absence、non-use、tool failure、retrieval miss、injection failure和task failure使用不同 reason code，不把它们统一标为 extraction failure。
- √ 未提供output evaluator的真实部署仍能运行 trigger、formation、exposure和process-level feedback闭环；PAST-Bench evaluator-free Hermes fixture 已验证完整 stage corpus 和 audit。
- √ PAST-Bench adapter同时输出 content-free process corpus identity（event IDs/digest）和 evaluation-only official score；二者不共享 learner 输入对象。

## 3. 可复用的已完成基础

以下能力保留，不重新实现：

- √ PAST-Bench 与 Hermes vendoring、安装、preflight 和 GPT-Luna runner。
- √ `native`、`native+ledger`、`native+adapter+ledger` 显式执行模式与 direct-native 默认路径。
- √ Request-level model usage、billing execution identity、ledger 和 audit。
- √ Semantic、episodic、procedural typed contracts 与 Hermes native adapters；当前只启用 semantic mutation。
- √ `ContextSnapshot`、tool closure、revision、provenance、writeback plan 和 idempotency foundation。
- √ Mem0-flat extraction、related-memory comparison 和 internal `ADD/UPDATE/DELETE/NONE` implementation。
- √ Mutation validation、transaction receipt、CAS、reread verification、restart recovery 和 rollback foundation。
- √ Content-free atomic operation graph 与 source/extraction/mutation/future operation lineage。
- √ Delayed-feedback dataset identity、cutoff、censoring、exposure、stage gate 和 immutable store foundation。
- √ Adaptive artifact lifecycle state、atomic ACTIVE pointer、rejection、rollback 和 restart-safe store foundation。
- √ Matched experiment manifest、replicate rotation、attempt retention、persistence isolation 和 five-method analysis foundation。
- √ 已完成 static SM01 三方法实验；结果继续保存在 [`static_sm01_20260827.md`](static_sm01_20260827.md)。
- √ 已完成 static utility infrastructure 实验；它保留为历史工程证据，不再定义当前 adaptive objective。

当前已验证基线：RSIMem `345 passed`；PAST-Bench `390 passed, 2 skipped`；compileall、`pip check` 和 diff check 通过。该数字只表示重写 checklist 前的代码基线，后续每个任务必须记录新的实际结果。

## 4. 审计发现：必须在第一阶段修正的问题

### D01：Adaptive Objective 仍错误命名为 Utility Per Cost

当前 `ADAPTIVE_POLICY_OBJECTIVE` 使用 `delayed-future-utility-per-cost-v1`，adaptive feature 列表包含 `raw_resource_usage`，offline/matched validator 也使用 cost gate。这与当前冻结方法冲突。

修正目标：新 adaptive schema 只表达 delayed future utility；resource usage 只留在 audit/report，不进入 learner 和 activation decision。

### D02：当前 Production Learner 只更新 Retrieval Threshold

`adaptive_preparation.py` 只创建 `retrieval_accept_threshold`，runtime config 和 binder 也只支持数值 threshold。该路径只能证明 online-update plumbing，不能证明 extraction prompt optimization。

修正目标：threshold 路径保留为 legacy infrastructure，不再被 launcher 命名为正式 `adaptive-rsimem`；核心 ACTIVE artifact 必须是 extraction prompt artifact。

### D03：Prompt Artifact 只有 Digest，没有可部署 Template Body

现有 `PromptArtifact` 描述静态代码内模板的 identity，`AdaptivePolicyArtifact.prompt_refs` 只保存引用，无法持久化、验证和部署模型生成的新 prompt 内容。`PromptSourceProvenance` 也只支持固定 Git upstream，不适合 generated child prompt。

修正目标：新增带 parent lineage、完整可部署 policy body、生成 provenance 和 content digest 的 extraction prompt artifact；不能把 prompt 文本伪装成 numeric parameter。

### D04：Extraction 与 Update Prompt 被强制共用 Policy Version

`Mem0FlatSemanticPolicy` 要求 fact prompt 与 operation prompt 的 `policy_version` 完全相同。只升级 extraction prompt 时，要么错误地改写 update prompt version，要么无法构造 policy。

修正目标：每个 component 拥有独立 artifact identity，composite semantic policy manifest 显式组合 extraction prompt digest、update prompt digest、retrieval config digest、route 和 boundary。

### D05：Semantic Compilation 依赖 Eviction Plan

当前真实 semantic writeback 只有在 lifecycle evaluator 生成 `ContextAction.EVICT + WritebackAction.ADD/UPDATE` plan 后才运行。Extraction 是否执行因此被一个与当前论文无关的 context-eviction decision 控制。

修正目标：在固定 completed-task boundary 上直接构建 semantic compilation request；context eviction 和 physical rewrite 保持 disabled，不能成为 extraction prerequisite。

### D06：Plan Source 与 Prompt 实际输入不一致

当前 deterministic evaluator 的 plan 通常只绑定第一个 completed user segment，但 `StaticSemanticWritebackRuntime` 将 snapshot 中全部 messages 放入 `MemoryExperience`，extraction prompt 实际读取未被 plan source IDs 完整绑定的内容。

修正目标：定义唯一 `ExtractionSourceProjection`；其 message IDs、内容 digest、prompt input、provenance、idempotency identity 和 optimizer corpus必须完全一致。

### D07：Task-Completed 与 Session-End 语义不一致

当前 lifecycle 在 task-completed 和 session-end 都生成 plan，但 semantic writeback 只消费 task-completed 结果。Session-end plan 会产生多余 dry-run evidence，使 checklist 声称的 invocation boundary 与真实 extraction boundary 不一致。

修正目标：第一版 extraction 只使用 `task_completed`；同一 completed experience 只能编译一次。Session-end 仅负责 cleanup，不生成第二个 semantic compilation attempt。

### D08：Feedback Dataset 看不到 NONE、无 Fact 和 Missed Extraction

当前 builder 只遍历具有非 `NONE` mutation 和 target artifact 的 operation。没有提取 fact、fact 被过滤、internal `NONE` 或应提取但遗漏的 source 都不会形成 training example。

修正目标：新增 source-level 和 extraction-set-level example；即使没有 mutation，也必须记录 extraction result、future observation window 和 unresolved/missed status。

### D09：Content-Free Dataset 无法直接优化 Prompt

Operation graph 只保存 ID、digest 和 metadata，适合审计但不包含 source text、old extraction output 或 future evidence content。Prompt optimizer 无法仅凭这些字段提出规则修改。

修正目标：保留 content-free audit dataset，同时新增 owner-controlled、ignored、content-bearing optimizer corpus；两者必须通过稳定 ID/digest exact join。

### D10：SM01 Feedback Resolver 不是通用 Use Evidence

当前 `sm01_tsv_v1` 通过 final response 是否包含四列 TSV 推断 reuse，并在 future surface 多于一个 memory artifact 时整体 censor。Hermes semantic path实际是 blank query 后将全部 memory eager 注入，不是 selective retrieval。

修正目标：显式记录 `exposure_mode=eager_system_prompt`；不把 eager exposure 称为 selective retrieval，不把 injected-but-not-used 自动标为 negative；支持 extraction-set-level outcome，并只在证据可唯一归因时生成 fact-level label。

### D11：Offline 与 Matched Validation 仍是 Threshold-Specific

Offline validator 使用 threshold 对 0/1 target 的 squared error；matched validator 使用 positive-rate、cost ratio、stability 和 uncertainty。这些 contract 不能直接评价 prompt artifact。

修正目标：为 extraction prompt 单独定义 validation observation、quality metric 和 decision schema，不复用 threshold error 作为 prompt quality。

### D12：Matched Gate 包含无效或虚假的条件

- `minimum_quality_delta=0` 允许零提升通过。
- `maximum_cost_ratio=1` 将 cost 错误地作为 activation gate。
- Live assembler 将 `stability_failure=False` 和 `uncertainty=0.0` 硬编码为已知值。
- 异构 request/token/storage 被相加为 lifecycle cost。

修正目标：Prompt activation 只使用可真实重建的 delayed extraction quality 与安全 gate；未测量字段删除或标为 unknown，不能使用常量冒充证据。

### D13：新 Live Run 被映射成旧 Validation Example ID

Matched assembler 按 replicate 位置把新 live run 包装成旧 split 的 `example_id/episode_id`，实际 derived example identity 只放在旁路 source metadata 中。

修正目标：每个 live observation 使用自身真实 run/task/episode/extraction identity；parent/proposal 通过 predeclared pair ID、task manifest digest、replicate slot 和 execution profile配对，不冒用历史 example identity。

### D14：Feedback Collection 使用带 Cost Gate 的 Parent Policy

`FEEDBACK_METHOD_VARIANTS` 当前只有 `static-utility-rsimem`，正式 adaptive methods 中 static/adaptive 也都启用 utility gate。因此旧 feedback 并非来自“只冻结 extraction prompt、其余行为一致”的 parent policy。

修正目标：新 parent variant 使用 plain static Mem0-flat extraction/writeback，不启用 cost utility gate；N 与 N+1 唯一允许变化的是 extraction policy body。

### D15：Formal Feedback Batch 没有强制 Clean Tree

Adaptive final launcher要求 clean tree，但 static feedback launcher允许 dirty RSIMem tree并只在 manifest 中记录。这样 training evidence 无法对应唯一源码状态。

修正目标：所有正式 feedback、validation 和 final evaluation batch 都要求 clean RSIMem tree、clean PAST tree、固定 commit/tree 和不可复用 batch ID。

### D16：缺少 Prompt Activation Fingerprint

当前 audit 只能证明 active policy version 出现在 utility evidence，不能证明新的 extraction prompt template 被实际渲染、改变输出并影响 persisted memory。

修正目标：建立 `ACTIVE artifact -> rendered prompt -> extraction output -> mutation/artifact -> future exposure/use/outcome` 的逐级 fingerprint 和计数。

### D17：Native Hermes Writer 隔离不是 Fail-Closed

当前 static writeback 会从显式 `enabled_toolsets` 中移除 `memory`，并通过 `skip_memory=False` 保留 native memory prompt read surface。在现有 PAST 配置中 memory tool 与 background review 没有触发，但当 `enabled_toolsets` 为空列表时，Hermes 的工具解析可能回退到全部默认工具，使 native memory writer重新出现。

修正目标：Static/adaptive RSIMem显式禁用 native `memory` tool、memory nudge和background memory review，同时保留只读system-prompt injection；不能依赖某个task恰好提供非空toolset。

### D18：按 Replicate 切分仍会重复同一 Task Template

当前SM01 feedback、validation和final launcher虽然使用不同run与replicate，但每次都重放同一组固定PAST-Bench task文本。Prompt optimizer如果已经看过过去replicate中的eval source/final response，再在相同task template上测试，会形成benchmark task leakage；unseeded model variation不能替代数据独立性。

修正目标：Formal prompt experiment按family或task-template group隔离train、validation和final test；同一task manifest digest的重复replicate只能属于同一个split。单一SM01重复实验只作为pipeline pilot，不作为最终paper effect evidence。

### D19：缺少 Host-Neutral Prompt Slot 与 Backend Adapter Contract

当前 Mem0-flat 虽然允许构造时传入 `fact_prompt`，但 prompt identity、默认模板、policy construction 和 runtime binding 都写在 Mem0-flat 实现内部。系统没有统一方式让一个外部 memory implementation 声明“这个运行时入口是 semantic extraction prompt”，也不能证明仅给出的源码路径或 module symbol 就是实际模型调用所消费的 prompt。现有 TypeScript LightRSI runtime 也不能通过 npm import 直接替换 Python memory framework 内部对象。

修正目标：定义 host-neutral `PromptSlot` 和 `MemoryPromptAdapter` contract。Backend 只需在真实 prompt consumption boundary 注册一个稳定 slot，optimizer、artifact store、validation 和 activation 只依赖 slot contract。第一版以 Python SDK/API 完成 Mem0-flat adapter；未来迁入 LightRSI 时共享语言无关的 artifact schema，并通过 Python SDK 或 IPC/sidecar 接入 Python backend，而不是依赖源码路径 monkey-patch。

## 5. 第一阶段：修正既有偏离与证据缺陷

第一阶段完成前，不运行新的 adaptive live batch，不生成论文效果结论。

### 1A：重置方法与文档边界

功能需求：

- √ 将 adaptive objective 从 utility-per-cost 改为 delayed extraction utility，并升级 schema/version，禁止静默解释旧 artifact。
- √ 从 learner feature ownership、offline validation 和 matched activation 中移除 resource cost。
- √ 将 retrieval-threshold learner 标记为 `legacy_threshold_experiment` 或等价非论文路径。
- √ 在 README、`progress.md`、experiment plan 和代码 docstring 中统一“第一版只优化 extraction prompt”。
- √ 将 2J/2K 已完成表述改成“threshold infrastructure complete，extraction adaptation pending”。
- √ 保留历史实验报告，不重写历史结果，但为旧 utility experiment 增加 superseded-method note。

测试与验收：

- √ 全仓搜索不再把 `future-utility-per-cost` 描述为当前方法目标。
- √ 新 adaptive config 无 `cost_weight`、`maximum_cost_ratio` 或 `lifecycleCostUnits` learner field。
- √ 旧 threshold artifact 不会被新 extraction runtime 当作兼容 artifact 加载。

### 1B：解耦 Semantic Compilation 与 Context Eviction

功能需求：

- √ 新增显式 `SemanticCompilationTrigger.TASK_COMPLETED` 或等价 contract。
- √ 由 trusted Hermes completion boundary 直接触发 semantic compilation，不要求 EVICT plan。
- √ Context eviction、physical rewrite 和 saved-token accounting 保持 disabled/not applicable。
- √ Failed、active、unresolved 或 open-tool-closure experience 不进入 completed semantic compilation。
- √ Session-end 不再生成第二个 semantic plan；同一 task/source revision 重放返回同一 receipt，不重复调用 extraction model。

测试与验收：

- √ 在无 eviction evaluator 时，completed task 仍能执行一次 semantic extraction/writeback。
- √ 开启或关闭 dry-run eviction observer 不改变 extraction request、model calls、mutation 或 memory bytes。
- √ Task-completed 后再 close session 不产生第二次 extraction operation。
- √ Active/current turn、open tool call 和 failed task 均 fail closed 且 backend mutation 为零。

### 1C：冻结一致的 Extraction Source Projection

功能需求：

- √ 定义版本化 `ExtractionSourceProjection`，明确允许的 roles、tool closure、message order、content bounds 和 metadata allowlist。
- √ 第一版投影使用完整 completed task experience，但不包含 system prompt、hidden grader、answer key、benchmark metadata 或 session 外消息。
- √ `source_message_ids` 必须精确覆盖传给 extraction prompt 的每一条 message，不多不少。
- √ Source projection digest 同时进入 request identity、idempotency receipt、operation artifact、prompt render evidence 和 optimizer corpus join。
- √ Assistant claim 与 tool output可以作为带 role/type 的上下文输入，但 prompt contract继续禁止把未经用户或工具证据支持的 assistant acknowledgement 当 durable fact。
- √ Source 大于预算时使用版本化、确定性的裁剪规则并记录 truncation，不允许调用方静默截断。

测试与验收：

- √ 增删、重排或修改任一投影 message 都会改变 source digest 和 request identity。
- √ 未被 source IDs 绑定的 sentinel 不会出现在 rendered extraction prompt。
- √ Tool call/result closure 不会被拆分；超预算裁剪保持 closure 原子性。
- √ 同一 snapshot/task 重启前后生成相同 projection 和 digest。

### 1D：拆分 Component Identity 与 Static Parent Policy

功能需求：

- √ Extraction prompt、update prompt 和 retrieval config分别拥有独立 version/digest。
- √ 定义版本化、host-neutral `PromptSlotDescriptor`：至少包含 `slot_id`、`memory_kind=semantic`、`policy_stage=extraction`、input/output schema digest、frozen wrapper digest、model profile和owner adapter identity。
- √ 定义 `MemoryPromptAdapter` 最小接口：列出 slot、读取 root artifact、校验 replacement artifact、将 artifact绑定到真实 policy factory，并返回actual binding fingerprint；optimizer不得 import Hermes、Mem0-flat或PAST-Bench实现。
- √ Slot注册发生在真实prompt consumption boundary；源码文件路径或Python dotted symbol只能作为可选provenance，不能作为runtime replacement contract。
- √ 第一个 `Mem0FlatPromptAdapter` 将 `mem0-flat.semantic.extraction` 显式绑定到 `Mem0FlatSemanticPolicy.fact_prompt`；未注册、重复slot、owner不匹配或目标不可替换时在模型调用前fail closed。
- √ 新增 composite semantic policy manifest，明确 route、boundary、backend、framework 和三个 component identity。
- √ `Mem0FlatSemanticPolicy` 允许 extraction prompt N+1 与 frozen update prompt N 组合，但 composite digest 必须变化。
- √ Operation context记录 composite policy version；每个 operation 额外记录自己实际使用的 component artifact identity。
- √ 新 static parent variant关闭 utility/cost gate，只运行 frozen extraction prompt + frozen update/retrieval。
- √ Static/adaptive mode显式将native `memory` tool加入disabled set，并将background memory review关闭；`skip_memory=False`只用于读取和渲染RSIMem写入的native semantic files。
- √ Native Hermes、RSIMem executor和operator recovery的每次semantic mutation拥有可区分writer identity；正式static/adaptive run只允许RSIMem committed receipt对应的writer。

测试与验收：

- √ 只改 extraction body时，只有 extraction component digest与 composite digest变化。
- √ 使用同一host-neutral fake adapter可在不import Hermes/PAST-Bench的测试中完成root读取、replacement绑定、render和fingerprint校验。
- √ Mem0-flat adapter只需一次显式注册即可加载root或ACTIVE extraction artifact；删除注册、给出错误slot ID或仅提供源码路径时不得静默报告绑定成功。
- √ 实际completion request记录的render fingerprint必须与adapter返回的binding fingerprint一致，不能只证明artifact被store读取。
- √ Update prompt、retrieval config、route、boundary、backend 或 model profile漂移会被 matched manifest拒绝。
- √ Legacy static utility mode仍可回归，但不进入新 parent/adaptive launcher。
- √ 在`enabled_toolsets=None`、空列表和非空列表三种配置下，static/adaptive model surface都不包含native `memory` writer，且background review model request为零。
- √ 检测到无RSIMem receipt的semantic file mutation时audit失败。

### 1E：修正 Feedback Evidence 与 Label Semantics

功能需求：

- √ Feedback builder同时生成 source-level、extraction-set-level 和可归因 fact-level examples。
- √ 新增版本化`OpportunityContract`、`UseContract`和`OutcomeContract` registry；contract声明family、eligible stage、memory scope matcher、允许读取的deployment surface、use parser、outcome parser和ambiguity semantics。
- √ Use resolver比较past-memory semantic key与current future input projection；current turn已重新提供同一信息时标记`current_input_confounded`，不得生成memory-attributed useful。
- √ Primary label unit固定为`source_id + extraction_set_id + future_opportunity_id`；fact-level只能作为唯一归因时的诊断和optimizer example，不能重复增加primary reward。
- √ `NONE`、empty extraction、filtered fact、failed mutation 和 no-mutation source均保留为 example，而不是被丢弃。
- √ Eager system-prompt exposure与 selective retrieval使用不同枚举；当前 Hermes semantic path记录前者。
- √ `injected_not_used` 在没有显式负面证据时标为 unresolved，不自动标为 negative。
- √ Useful只来自`opportunity + explicit attributable use + successful observable outcome`；三段operation identity缺一不可。
- √ Harmful只来自extraction-owned unsupported/transient/conflicting memory，或explicit use后的deterministically attributed harmful outcome；正确memory未被Agent采用不属于extraction harmful。
- √ 没有曝光机会、多个 artifact 无法区分贡献、观察窗口不完整或信号冲突时标为 unresolved/censored。
- √ Missed extraction必须同时绑定source span/digest、empty或不等价extraction result、future opportunity和absence-attributed outcome；任一缺失都保持unresolved。
- √ 等价fact检查第一版优先使用family contract的deterministic normalized key/value或结构约束；若未来使用LLM matcher，必须独立版本化、冻结并禁止访问grader/answer，不能在当前SM01 pilot中临时加入。
- √ Resource usage 从 label payload 中移出，单独保存在 accounting join。
- √ Feedback resolver使用版本化family contract registry；每个train/validation/final family在运行前声明允许读取的deployment surface、use signal和ambiguity规则，未知family或缺失contract在模型调用前失败。

测试与验收：

- √ 覆盖 useful、harmful、empty、filtered、NONE、missed、ambiguous-multi-artifact、not-exposed 和 censored fixtures。
- √ 删除opportunity、use或successful outcome任一证据后，useful label必须消失；仅有eager injection不能补足三段证据。
- √ 同一次set-level success包含三个facts时，primary useful count仍为1；只有唯一绑定的fact-level diagnostic可以额外存在。
- √ 正确fact已提取但未来Agent未使用时，不能生成extraction harmful或missed label。
- √ SM01 多 artifact 不再让整次 extraction 无条件失效；允许 set-level label，fact-level不确定时保持 unresolved。
- √ 删除 future use/outcome evidence 后 positive label不能继续存在。
- √ Eager injection但无显式使用时不能生成 negative label。
- √ Official score、grader、answer 和 expectation 在 label builder API 中不可达。
- √ SM01、SM02、SM05等被选family各自具有正向、负向、ambiguous和censored contract fixture；不能把TSV parser复用于非TSV family。
- √ SM02 resolver在 feedback boundary 统一规范化人名/recipient ID（大小写、空格和连字符），并按精确 normalized ID 判定 advisory boundary，避免相似字符串误判。

### 1F：重建 Prompt-Oriented Validation Contract

功能需求：

- √ 新 validation observation使用真实 live run/task/episode/extraction identity，不复用历史 example ID。
- √ Parent/proposal pair由 `pair_id + replicate + task_manifest_digest + model_profile + budget + persistence_state` 精确绑定。
- √ Activation primary quality固定为set-level `resolved_useful_rate=useful/(useful+harmful)`；unresolved/censored不进入分母，也不能伪装成0。
- √ Acceptance config显式冻结`minimum_resolved_examples`、大于0的`minimum_useful_rate_delta`、`maximum_harmful_rate_delta`、`minimum_coverage_ratio`、`maximum_empty_rate`和`maximum_missed_rate_delta`，并固定每个metric的上述numerator/denominator定义。
- √ Candidate必须同时通过strict useful-rate improvement与全部anti-collapse/safety constraints；任何一项unknown且该项被配置为required时拒绝。
- √ Cost不参与 activation；raw cost仅随 decision report输出。
- √ Stability和uncertainty只有在真实计算时才进入 report；否则为 unknown并不参与 gate。
- √ Proposal generation次数和 candidate selection budget预先冻结，不能反复查看同一 validation 后无限改 prompt。
- √ Split unit使用family/task-template group而不是run ID；相同task manifest digest不能跨train、validation和final test。

测试与验收：

- √ 旧/new identity冒用、pair错位、task digest漂移和variant缺失均 fail closed。
- √ Equal quality、无 resolved evidence、仅随机 score波动或无实际 extraction intervention都不能激活。
- √ Candidate输出全空、只保留一个高置信fact、将一个fact拆成多条或把unresolved从分母删除但不报告时均不能提高可接受quality。
- √ Useful率提高但harmful、empty或high-confidence missed任一越界时保持REJECTED。
- √ Positive delayed quality且所有安全条件通过时可以激活，即使 resource cost更高。
- √ Validation decision可从 raw observation重建，且不读取 official task score。
- √ 将同一SM01 task template放入不同split时，split audit必须拒绝。

### 1G：修正 Experiment Manifest、Launcher 与 Analyzer

功能需求：

- √ 新方法名称明确为 `static-extraction-rsimem` 和 `adaptive-extraction-rsimem`，不复用 `adaptive_utility` 造成语义混淆。
- √ Feedback collector使用 plain static parent extraction policy，不启用 utility gate。
- √ Formal feedback、validation 和 final batch都要求 clean RSIMem/PAST tree并记录 commit/tree。
- √ Manifest记录 source-projection version、extraction parent/active artifact、update prompt digest、retrieval config digest、feedback contract和proposal budget。
- √ Manifest记录opportunity/use/outcome contract digest、primary objective schema、primary unit、resolved denominator和全部anti-collapse threshold。
- √ Manifest记录split role、family/task-template group ID和task manifest digest，阻止同一template跨split复用。
- √ Analyzer删除异构 `lifecycleCostUnits` 和由其派生的 `futureUtilityPerCost`；保留 raw vector。
- √ Analyzer传播 usage completeness；unknown bucket不参与 delta和claim。
- √ Analyzer增加 prompt activation funnel：eligible、rendered N+1、changed extraction、changed mutation/artifact、future exposure、use和outcome。
- √ Analyzer分别报告eligible opportunity、useful、harmful、missed、unresolved、censored、non-empty coverage、empty rate和schema/safety failures；不只输出一个aggregate score。

测试与验收：

- √ Dirty tree、旧 artifact、wrong component digest、错误 method mapping和重复 batch ID在模型调用前失败。
- √ Static/adaptive 两侧除 extraction artifact外的 manifest identity完全相同。
- √ Unknown usage不阻止 quality analysis，也不会被计算成真实零。
- √ 无 changed extraction时，analyzer拒绝 operation-attributed adaptation claim。

### 1H：第一阶段回归与关闭条件

- √ 所有 Stage 1A-1G 功能与反向测试完成。
- √ Direct native、native+ledger和plain static semantic behavior回归通过。
- √ Deterministic fixture证明一次且仅一次 completed-task extraction。
- √ 一个低成本 live static smoke证明 source projection、plain parent policy、feedback set和raw accounting可重建；见[`extraction_stage1_acceptance_20260828.md`](extraction_stage1_acceptance_20260828.md)。
- √ `progress.md`、experiment plan和本文状态同步。
- √ 记录完整 RSIMem、PAST-Bench、compileall、`pip check`、shell syntax、diff和secret scan结果。

第一阶段关闭后才能开始第二阶段的 policy optimization infrastructure；第三阶段的真实 adaptive live run仍需等待第二阶段所有 required contract和replay gate通过。

## 6. 第二阶段：六层 Memory Policy 优化基建

第二阶段的目标不是立即让六层都由模型联合学习，而是为每一层建立统一的 decision contract、adapter boundary、证据记录、回放和 fail-closed 行为。第一版实验可以只打开 extraction；没有这些基础设施时，不允许把 Host 固定行为误称为 RSIMem policy。以下 2A-2I 的 `√/□` 只表示基建状态，不表示第三阶段的真实效果已经完成。

### 2A：统一 Memory Formation/Exposure Policy Contract

功能需求：

- √ 定义 host-neutral `TriggerEvent`、`TriggerDecision`、`SourceSelectionDecision`、`ExtractionDecision`、`AdmissionDecision`、`CommitDecision` 和 `ExposureDecision`，每个 decision 有 schema version、稳定 ID、policy version、source revision、reason codes、输入/输出 digest和执行状态。
- √ 定义 `MemoryFormationPolicy` 和 `MemoryExposurePolicy` 接口；runtime只依赖接口，不 import Hermes、PAST-Bench或具体 memory backend。
- √ 明确 `RUN/SKIP/DEFER` 的语义：SKIP不执行 extraction，DEFER必须记录下一次允许边界，RUN必须绑定本次 source projection；未知或冲突状态 fail closed。
- √ 将六层 decision 与最终 `MemoryMutationReceipt`、injection receipt、future feedback通过 stable lineage 关联，支持从 Host event 重建整条 formation/exposure 链。
- √ 将不可学习的安全不变量从 policy action 中隔离：active/current保护、tool closure、schema、CAS、transaction、rollback、credential和writer identity不得由模型输出。
- √ 为 fixed policy、single-layer adaptive policy 和 joint policy 定义兼容但不可混淆的 artifact identity；联合 candidate必须声明实际开放的 layer set。

测试与验收：

- √ 缺少 decision、revision、source digest、policy identity或执行 receipt时，audit拒绝该 episode。
- √ `SKIP` 不产生 extraction model request，`DEFER` 不产生 mutation，`RUN` 至少产生可关联的 source/extraction record。
- √ 任意 policy decision 不能绕过 schema、CAS、tool closure、active/current或rollback safety gate。
- √ 同一 event/revision 重放生成相同 decision identity，不重复执行 mutation。

2A 实现记录：`src/rsimem/memory/policy_contracts.py` 提供统一 contract、SafetyBoundary、PolicyLineage、PolicyArtifactIdentity 和 audit validator；正反与 replay 测试见 `tests/test_policy_contracts.py`，对应 commit `ca174d4`。
后续补充的执行状态反向测试还要求 `RUN` 不得伪装成 `SKIPPED/DEFERRED`，并拒绝非执行决策携带 runtime receipt；该约束保持 pending/failed/rejected recovery 状态可用。
`JsonProcessFeedbackLedger` 的并发 writer 也已通过锁与原子替换反向测试，重复 event 只保留一个 canonical record。
`JsonPolicyDecisionLedger` 同样通过独立实例并发 writer 测试，重复 policy evidence 不会扩增 ledger 或覆盖冲突 payload。

### 2B：Host Adapter 与 Lifecycle Trigger Policy

功能需求：

- √ Hermes adapter将真实 task、turn、tool、context-pressure和session事件映射到统一 `TriggerEvent`；没有真实事件来源的 trigger明确标记 `unsupported`，不伪造。
- √ 保留当前 `task_completed -> RUN` 作为 fixed parent policy；将 `session_end`、`turn_interval`、`tool_boundary`、`context_pressure` 和 `manual` 先接入 shadow-only观测，不改变正式 parent行为。
- √ Trigger policy支持 `RUN/SKIP/DEFER`、最小间隔、pending source、下一次 eligible boundary和duplicate suppression；source revision变化必须重新判断。
- √ Host adapter只提供事件、snapshot和执行 hook；trigger strategy、阈值、频率和skip/defer理由归RSIMem policy artifact管理。
- √ 记录每次候选 trigger，包括未执行的 SKIP/DEFER，防止实验只看到已写入样本而忽略被跳过的样本。

测试与验收：

- √ task completion、session end、tool boundary和context pressure均有正反 fixture；unsupported trigger不能静默当作支持。
- √ shadow-only trigger不改变模型调用、extraction、mutation、memory文件或future exposure。
- √ 相同 source revision 的重复事件只产生一次 RUN；新 revision可产生新的 formation attempt。
- √ active、current、failed task和open tool closure不能因为 trigger policy 而进入 semantic mutation。

2B 当前实现记录：`HostTriggerAdapter`/`HermesTriggerEventAdapter` 对真实 snapshot identity、task state、turn index、tool boundary、pressure token 和 manual authorization 做显式 supported gate；`DeterministicTriggerPolicy` 保留 task-completed parent，其余事件 shadow-only。Hermes bridge 已提供 `on_session_end`、`on_turn_interval`、`on_tool_boundary`、`on_context_pressure` 和 `on_manual_trigger` 入口，并持久化 trigger observation；open tool closure 可在 shadow snapshot 中保留，但不会进入 source selection 或 semantic mutation。正反与 replay fixture 位于 `tests/test_trigger_policy.py`、`tests/test_hermes_shadow_boundaries.py` 和 `tests/test_hermes_integration.py`。

当前 replay/audit 记录：`JsonPolicyDecisionLedger` 已包含 `variant/traceId/familyId/stage`，`audit_policy_evidence` 校验 policy evidence 与 lifecycle snapshot 的身份 join；`DeterministicPolicyReplay` 可在无模型、无 backend mutation 的条件下重建六层 decision、lineage 和 mutation/injection receipt join。该 harness 是 deterministic feasibility 基线，不等同于六层 live adaptive policy。

### 2C：Source Selection 与 Context Projection Policy

功能需求：

- √ 在现有 `ExtractionSourceProjection` 之上增加可审计的 source selection decision，区分“Host可见内容”和“本次policy选择的内容”。
- √ 支持 whole completed task、selected completed segments和增量 revision三种固定 projection mode；第一版保持 whole completed task作为 parent。
- √ 记录被选择、被跳过、因 active/current/tool closure/预算而拒绝的 segment IDs及reason codes。
- √ Source selection不能读取 hidden grader、answer key、future test或benchmark-only metadata。
- √ Source digest、selected IDs、truncation和projection mode进入 extraction request、idempotency和feedback lineage。

测试与验收：

- √ 选择集合变化会改变 source digest和extraction identity。
- √ source selection不能拆开 tool call/result closure，不能包含当前 active turn。
- √ 同一 snapshot/revision回放得到相同 selection；超预算时行为确定且有记录。

2C 当前实现记录：`DeterministicSourceSelectionPolicy` 生成可审计 selected/skipped/rejected decision；`ExtractionSourceProjector` 和 semantic ingest builder 接受 selected IDs 并再次执行 role、unresolved、active/current、tool-closure、budget safety gate。Hermes static writeback 已绑定该 decision，相关测试见 `tests/test_source_selection_policy.py`、`tests/test_extraction_source.py`。hidden/future metadata isolation 和完整 feedback lineage 仍需在后续 feasibility audit 中复核。

### 2D：Extraction Policy Envelope 与 Artifact

为降低 generated prompt 改坏 schema或安全边界的风险，第一版不允许模型重写整个 wire prompt。Prompt拆为：

```text
frozen system/safety/schema wrapper
  + adaptive extraction-policy body
  + frozen source_messages / exit_evidence slots
```

功能需求：

- √ 定义不可变 `ExtractionPromptPolicyArtifact`，保存 adaptive body、parent artifact、version、digest和生成 provenance。
- √ Adaptive body使用版本化`ExtractionPolicySpec`表示为有稳定rule ID的有序规则列表；frozen compiler将spec确定性编译成实际prompt body。
- √ Child artifact保存parent spec digest、结构化rule edits、compiled body和compiler digest；重新应用edits必须逐字重建同一compiled body。
- √ Artifact只引用host-neutral `slot_id` 和slot contract digest，不保存Python callable、module object、Hermes request或Mem0-flat内部类型。
- √ Artifact记录 frozen wrapper digest、input/output schema digest、required placeholders、model profile和最大长度。
- √ Generated provenance记录 optimizer model/config、training corpus ID/cutoff、proposal request digest、completion digest和usage。
- √ Baseline Mem0-flat extraction prompt被导入为 root artifact N，不依赖代码常量才能部署。
- √ Root artifact由adapter导出并可独立序列化；同一artifact可由fake adapter加载，证明其生命周期不依赖Mem0-flat源码位置。
- √ Artifact store复用既有 crash-safe lifecycle思想，但使用独立 schema，不能与 numeric threshold artifact混读。
- √ Prompt body不得包含 `$source_messages`、`$exit_evidence` 或其他模板控制字符；placeholder只属于 frozen wrapper。

测试与验收：

- √ Root、child、unknown parent、cycle、digest mismatch、oversize和schema mismatch测试通过。
- √ Duplicate rule ID、unknown replace/delete target、protected rule修改、no-op edit和edit/body replay mismatch均被拒绝。
- √ Exact artifact跨restart可重载并生成相同 rendered prompt。
- √ Artifact篡改、多个 ACTIVE或错误 wrapper digest时fail closed到root static prompt。

### 2D.1：Admission Policy 与 ADD/NONE/UPDATE Decision

功能需求：

- √ 将 extraction output 与 admission decision 显式拆开：记录候选 fact IDs、被接受/过滤的 fact IDs、`ADD/NONE/UPDATE`、reason codes和当前 backend revision。
- √ 定义 host-neutral `AdmissionPolicy` 接口；Mem0-flat 当前内部 operation prompt作为 fixed parent admission policy接入，不让其隐式承担 extraction 结果解释。
- √ 保留 backend 的 duplicate、conflict、temporary、unsupported、empty和unresolved safety rules；policy只能在允许集合内选择 action。
- √ 将“没有提取到事实”和“提取到了但 admission 选择 NONE”记录为不同状态，二者不得共用一个空 extraction label。
- √ candidate不得通过全量ADD、重复ADD或全量NONE获得虚假的 coverage/useful-rate提升。

测试与验收：

- √ extraction有候选但 admission=NONE、extraction为空且 admission=NONE、ADD被backend拒绝、重复ADD和UPDATE冲突均有独立 fixture。
- √ admission decision可由 input fact digest、existing revision和policy artifact重放。
- √ 不支持 UPDATE 的 backend不能接受 update decision；不支持 rollback 的 backend不能被声明为可安全执行的adaptive admission。

2D.1 当前实现记录：`DeterministicAdmissionPolicy` 位于 `src/rsimem/memory/admission_policy.py`，Mem0-flat static writeback 的 extraction trace 会生成 admission evidence，并绑定 backend revision、operation IDs 与 receipt lineage。`AdmissionCensus` 与 `validate_admission_candidate()` 现在按稳定 source ID 对齐 parent/candidate，拒绝 blanket ADD、blanket NONE、duplicate-marked ADD 和 coverage collapse；空 extraction 的 NONE 与 duplicate candidate 保持不同 reason。对应反向测试见 `tests/test_admission_policy.py`。第三阶段仍需在正式 family census 中报告该 gate 的实际分布。

### 2D.2：Commit Scheduling 与 Mutation Safety Boundary

功能需求：

- √ 将“是否立即提交、是否等待后续 boundary、是否安全重试”建模为 commit scheduling decision；事务实现、CAS、receipt和rollback保持固定。
- √ 支持 pending/deferred commit 的持久状态、过期 revision、重启恢复、取消和最终状态；不能因为进程退出丢失待提交决策。
- √ 每次 commit schedule 记录触发 event、目标 mutation IDs、expected revision、执行 boundary和最终 receipt。
- √ 第一版正式实验固定立即提交；commit scheduling 只做 deterministic/shadow infrastructure，不能与 extraction candidate同时开放。

测试与验收：

- √ deferred commit不会提前修改backend，重启后可恢复或明确标记失败。
- √ stale revision、CAS失败、重复提交、进程崩溃和rollback均不产生半提交 memory。
- √ commit scheduler 执行前必须经过显式 mutation validator；validator 缺失或拒绝时持久化 `FAILED`，且不会调用 apply，避免绕过 mutation safety 或把失败伪装成成功。

2D.2 当前实现记录：`CommitScheduler`/`JsonCommitScheduleStore` 提供 crash-safe pending/deferred schedule、CAS revision gate、cancel/failure/terminal idempotency；`execute()` 现在要求显式 `mutation_validator`，真实 backend validator 仍由现有 transactional executor 持有，scheduler 不宣称替代该 safety boundary。
`JsonIdempotencyReceiptStore.reserve_if_absent()` 同样使用排他锁和原子替换；损坏 receipt 在 reservation 路径 fail-closed，不能被新 mutation 覆盖。

### 2D.3：Exposure 与 Context-Memory Interaction Policy

功能需求：

- √ 将 memory exposure 从 backend read 和 Host injection hook 中拆出 `ExposureDecision`，至少记录 `RUN/SKIP`、selected artifact IDs、排序、注入位置、预算和reason codes。
- √ 区分 eager system-prompt、selective retrieval、tool-mediated read和not-exposed；不能把 Host 提供的注入能力误报成 policy 已选择的 retrieval。
- √ Hermes adapter提供真实 injection boundary和context revision；RSIMem决定何时、注入哪些memory以及如何组成memory block。
- √ 第一版固定当前 eager exposure、注入位置、格式和预算；exposure policy先做 shadow replay，再进入单独 matched ablation。
- √ exposure decision 可接收 runtime-owned `SafetyBoundary`；schema/CAS/transaction/rollback/credential/writer 任一安全条件失效时 fail closed 为无注入 `SKIP`，并继续由固定 context/tool-closure 与 token budget 边界约束，不能伪造 memory source。

测试与验收：

- √ `SKIP` 不产生注入，`RUN` 的 artifact IDs与实际注入内容 exact join，not-exposed不能产生memory-use label。
- √ eager与selective、空memory、多个artifact、预算裁剪和注入失败均有独立 fixture。
- √ exposure decision变化而formation policy不变时，audit能准确标记为 exposure intervention。
- √ 注入前后 context revision、artifact digest和render fingerprint可重建；重启后不重复注入或丢失 active pointer。

2D.3 当前实现记录：`DeterministicExposurePolicy` 与 `InjectionReceipt` 位于 `src/rsimem/memory/exposure_policy.py`；Hermes `_PromptMemoryStore.format_for_system_prompt` 记录真实 artifact IDs、context revision 和 render fingerprint，并写入 policy evidence ledger。无效 `SafetyBoundary` 会在注入前返回空 `SKIP`；真实 selective/tool-mediated matched ablation 尚未开放。

### 2E：Content-Bearing Extraction Optimizer Corpus

Audit dataset继续content-free；optimizer corpus只存在于owner-controlled ignored output中。

功能需求：

- √ 每个 example保存 bounded source projection、policy N extracted fact set、persisted fact lineage，以及opportunity/use/outcome三段delayed evidence。
- √ Example显式标记`useful/harmful/missed/unresolved/censored`、label level（source/set/fact）、attribution confidence、reason codes和component ownership。
- √ Useful example只在三段证据完整时进入resolved optimizer bucket；harmful和missed必须携带各自的可重建归因链。
- √ 同时保存对应content-free dataset/example/operation/artifact IDs与digests，支持exact join。
- √ Corpus区分train、validation和future-test batch；future-test内容在N+1激活前不可达。
- √ Corpus文件使用attempt-local路径、最小权限和显式retention policy，不进入Git、通用ledger或共享trace。
- √ Credential、authorization header和机器路径在进入optimizer provider前通过专用secret boundary处理；不改变正常agent context，只保护optimizer副本。
- √ Source content被作为untrusted data结构化传入optimizer，不能覆盖optimizer system instruction。

测试与验收：

- √ 同一frozen source/evidence重建canonical-equivalent corpus。
- √ Content-free audit与content-bearing corpus任一join缺失、冲突或future-dated时fail closed。
- √ Tracked-source/manifest/ledger中不存在corpus正文。
- √ Corpus中不存在official grader、answer key、hidden expectation或future-test内容。

### 2F：Extraction Prompt Optimizer

第一版使用一个受控LLM meta-optimizer，根据历史example总结“什么应该提取、什么不应该提取”的规则。Optimizer不直接自由重写整个prompt，而是对parent `ExtractionPolicySpec`生成结构化rule edits，再由frozen compiler生成replacement policy body。

功能需求：

- √ 冻结optimizer system instruction、input schema、output schema、model profile、temperature、token budget和timeout。
- √ 输入包含parent policy body、bounded source/trajectory evidence、deployment-observable task outcome，以及按useful/harmful/missed/unresolved/censored分类的strict diagnostics。
- √ Optimizer input按source/set/fact和episode层级分组，显式说明`unresolved/censored`不是negative；同一set不能按fact数重复加权，完整episode outcome也不能被改名为extraction-owned error。
- √ Output只允许`ADD_RULE/REPLACE_RULE/DELETE_RULE` edits、每个edit的evidence example IDs和结构化reason codes；candidate body由frozen compiler生成，不接受模型提供的第二份不一致body。
- √ Protected durability、source-grounding、credential和schema规则只能位于frozen wrapper或protected rule set，optimizer不能删除或弱化。
- √ Exploratory objective明确要求在固定下游组件下寻找可验证的deployment-observable outcome改善方向；strict objective继续约束harmful、coverage、empty和missed，不输入cost或official task score。
- √ Formal policy update默认只生成一个candidate；若未来增加K candidates，K和selection rule必须在查看validation结果前冻结。
- √ Strict attribution path在无resolved signal、只有censored evidence或attribution不足时返回`NO_PROPOSAL`；exploratory end-to-end path在满足预注册的完整episode/结果多样性/预算条件时可以保留完整训练轨迹并生成受限 hypothesis，但必须标记为unresolved/noisy evidence，且不得直接激活candidate。
- √ Optimizer调用usage单独记录，但不作为optimizer目标或candidate排序依据。

测试与验收：

- √ Useful-only、harmful、missed、conflicting、low-sample和no-signal fixtures通过。
- √ Optimizer将`injected_not_used`误当negative、将application failure归因给extraction或复制一次set success给多个facts时，proposal contract拒绝。
- √ 相同captured optimizer completion生成相同artifact；不要求重新调用随机模型逐字复现。
- √ Candidate body不能复制training source中的用户事实、task ID、答案、专有值或长n-gram；prompt必须学习规则，不能充当memory store。
- √ Candidate rule不得出现SM01、TSV、固定列名、项目名或其他family-specific shortcut，除非该词原本属于frozen generic root contract；命中shortcut时拒绝而不是交给validation碰运气。
- √ Prompt injection、credential exfiltration、schema override和benchmark-specific shortcut candidate被拒绝。

### 2G：Static Safety 与 Offline Policy Validation

功能需求：

- √ Contract validator检查body长度、字符、forbidden instruction、wrapper/schema digest和parent lineage。
- √ Deterministic extraction suite覆盖durable preference、constraint、temporary request、unresolved claim、assistant-only acknowledgement、tool evidence、credential/path和empty source。
- √ Candidate必须保持严格JSON `{facts: string[]}` output contract。
- √ Offline validation在独立historical split上比较parent/candidate的deployment-observable outcome与layer-specific intervention，不使用official score。
- √ Strict diagnostics按set-level计算resolved useful rate，同时检查harmful、non-empty coverage、empty extraction和high-confidence missed；end-to-end exploratory quality单独报告，不与strict rate混成一个指标。
- √ 所有ratio同时输出numerator、denominator和unknown count；resolved denominator不足时拒绝，不能只报告百分比。

测试与验收：

- √ Candidate不能降低所有输出为空来通过negative-only样本。
- √ Candidate只提取一个高置信fact时，即使useful rate为100%，只要coverage低于冻结floor也必须拒绝。
- √ Candidate不能通过复制source或输出完整transcript提高recall。
- √ 对于需要正式激活的candidate，offline quality不严格高于parent时保持REJECTED；exploratory hypothesis只能进入matched trial，不能直接激活。
- √ Offline accepted只允许进入matched trial，不可直接写production ACTIVE。

### 2H：Matched Trial、Activation 与 Rollback

功能需求：

- □ 在独立PAST-Bench validation batch中轮换运行parent N与proposal N+1。
- √ Pair使用相同family、episode manifest、model、budget、home seed state和feedback contract。
- √ Activation只看预注册的deployment-observable task outcome、strict attribution diagnostics、anti-collapse constraints和安全审计，不看optimizer不可达的official task score。
- √ Exploratory candidate可以在strict resolved sample不足时进入 matched trial，但不能直接激活；正式激活必须有 matched outcome improvement，并通过适用的harmful、coverage、empty、missed和安全 gate。若某项 strict metric 为unknown，只能在该项被预先声明为非必需时继续。
- √ Production activation原子切换唯一ACTIVE extraction artifact。
- √ Operator rollback恢复parent N；自动rollback只在有真实定义的safety violation时触发。

测试与验收：

- √ Trial config不能被official final launcher误当production config。
- √ Activation crash不会产生两个ACTIVE artifact。
- √ Rejection、重复activation、restart和rollback幂等。
- √ Decision记录真实pair IDs、artifact digests、六层 decision fingerprints、U/H/M/unresolved/censored counts、各ratio分子分母、deployment-outcome delta、coverage、quality delta、constraint results和reason codes。

Matched evidence assembler 现在还要求每个 completed validation slot 具有
`process_corpus.json`，并校验其 split/family/template/manifest identity、process
event audit 和 evaluator/score 字段隔离；缺失或篡改时在 activation assembly 前
fail closed。

### 2I：Runtime Prompt Binding 与 Activation Fingerprint

功能需求：

- √ Runtime通过 `slot_id -> MemoryPromptAdapter` registry解析唯一ACTIVE extraction artifact，由adapter组合frozen wrapper并注入真实policy factory。
- √ 对开发者暴露的一行适配入口等价于显式注册一个slot，例如`prompt_slot("mem0-flat.semantic.extraction", default=..., schemas=...)`；该便利API必须落到同一adapter contract，不能使用全局monkey-patch。
- √ Static N和adaptive N+1使用相同completion client、model profile、update prompt、retrieval config和backend。
- √ 每次extraction operation记录actual extraction artifact ID/version/body digest、wrapper digest和render input digest。
- √ Extracted fact、mutation和persisted memory lineage回连actual artifact。
- √ Audit输出`eligible -> rendered -> changed extraction -> changed artifact -> future exposure -> use/outcome` funnel。

测试与验收：

- √ Config声明N+1但runtime加载N时fail closed。
- √ Store中的slot ID、adapter owner、contract digest、wrapper digest或schema digest任一不匹配时，在extraction model调用前fail closed到明确配置的root policy；formal adaptive run不得静默fallback后继续标记为adaptive。
- √ N/N+1产生相同extraction时记录no intervention，不伪装成changed decision。
- √ N+1改变extraction但update/retrieval/component identity漂移时matched audit失败。
- √ Restart后actual artifact fingerprint保持一致。

## 7. 第三阶段：逐层 Policy 实验与 PAST-Bench 验收

第二阶段完成后，六层 policy 都具备统一 contract、adapter boundary、evidence、replay 和 fail-closed 行为，但正式实验不同时打开六层。每一轮只新增一个可学习层，其他层、Host、backend、model、task manifest、budget、persistence 和协议 surface 保持不变。探索性端到端 feedback 可以用于发现 candidate，但任何 candidate 都必须经过独立 matched validation。

### 3A：Deterministic End-To-End Gate

截至 2026-08-29，已完成第一份 deterministic/shadow feasibility fixture，并提供可执行入口 `python -m rsimem.memory.policy_feasibility_fixture`：completed snapshot 同时包含 durable 与 temporary 信息，parent/candidate replay 共享 event、revision、backend 和 lineage；Extraction case 覆盖 useful 与 missed 的完整证据链，缺失任一链节点会 fail-closed 降级为 `unresolved`。该 fixture 结果与限制记录在 [`policy_feasibility_baseline_20260829.md`](policy_feasibility_baseline_20260829.md)，不构成真实 provider 或 PAST-Bench uplift 证据。

- √ `build_extraction_feedback_fixture()` 构造一个过去context中含durable与temporary信息、未来任务只使用durable信息的 SM01 fixture；`MISSED` 分支由 Policy N 的空 extraction 结果表示，`USEFUL` 分支由同一 durable fact 的显式曝光和使用表示。
- √ Policy N 在 deterministic fixture 中产生至少一个可归因问题（遗漏 durable fact），并由注册的 family contract 解析为 `MISSED`，不是手工写入 reward 标签。
- √ Fixture分别构造完整`opportunity -> use -> successful outcome` useful链和`source -> no equivalent extraction -> future demand -> absence-attributed outcome` missed链（当前为 fixture-local opaque IDs，不是部署标签）。
- √ 删除任一useful/missed链节点后label退化为unresolved，而不是继续贡献optimizer reward。
- √ End-to-end feedback fixture 保留 source、future opportunity/use/outcome operation join、exposure、extraction-set 和 fact-level 轨迹；strict resolver 同时保留 `UNRESOLVED/CENSORED` 诊断，resolved primary 才能投影 extraction hypothesis。
- √ Delayed feedback构建optimizer corpus并生成受限的、只针对当前开放层的N+1 hypothesis；`extraction_proposal` 现在持久化经 parent/corpus/ownership gate 校验的 `feasibility-hypothesis.json`，没有真实 uplift 也不判定 feasibility 失败。
- √ N+1 proposal 已通过 optimizer projection、parent/corpus/ownership gate、candidate static safety、deterministic extraction suite、offline quality gate 和 extraction slot boundary validation；candidate 只注册为 proposal，是否激活属于后续效果实验，不属于本阶段最低验收。
- √ Future fixture 可以从 restart 后的 extraction policy store 加载 N+1 proposal，并通过 `FeasibilityInterventionPath`/JSONL ledger 记录 projection、parent/candidate、target layer 和 replay fingerprint；是否改善 deployment-observable outcome仍属于后续效果实验。
- √ fixture 到 feasibility projection 的链路不读取 grader/answer，不使用 cost 信号，也不修改 update/retrieval policy；content-bearing source 只停留在 owner-controlled fixture/corpus 边界。
- √ restart 后的 proposal reload、`NO_PROPOSAL`（无 candidate）、rejection（不可转 ACTIVE）和 rollback（清空 ACTIVE pointer）反向路径均通过；真实 mutation crash recovery 仍由后续 matched/runtime gate 负责。

验收：只有结构化证据可以重建`N -> past feedback -> N+1 hypothesis -> target-layer intervention`，并确认candidate只改变预注册的policy layer时，才可以把该层标记为 `optimization-ready`。真实 provider uplift 仍属于后续效果实验，严格 attribution 不再是探索性 candidate 的唯一前置门槛。

### 3B：六层 Policy 可优化性验收

在六层 matched-intervention focused test 加入后，RSIMem 当前验证计数为
`675 passed`；下文较早的 `656 passed` 是历史 process-audit 快照。
完整 process-chain restart replay 测试后，历史计数曾更新为 `676 passed`；当前
RSIMem `.venv` 回归为 `703 passed`，PAST-Bench 为 `397 passed, 2 skipped`；
SM03 held-out split preflight 记录在
[`extraction_stage3_sm03_heldout_preflight_20260829.md`](extraction_stage3_sm03_heldout_preflight_20260829.md)。

第一轮六层 deterministic census 已建立：每层至少有一个 parent/candidate replay case，Extraction 因同时具备 useful/missed resolved outcome 暂列 `optimization-ready`；Trigger、Source selection、Admission、Commit、Exposure 因 outcome variation 不足暂列 `validation-only`。每个 intervention 现在由 `ProcessFeedback` 绑定 event、source revision、parent/candidate decision、receipt 和 before/after digest，并由独立 feasibility ledger 跨 restart 保存；ledger 支持 `verify_case()`，缺失或冲突 receipt fail closed；有 process signal 的 case 还生成绑定 past feedback 与目标层的 `PolicyHypothesis`，并持久化完整 hypothesis payload，显式覆盖 `N -> feedback -> N+1 -> intervention` identity。Hermes bridge 另外写入独立的 `rsimem_process_feedback.jsonl`，通过 host-neutral `ProcessEvent` 记录 trigger、source、extraction、admission、commit、retrieval、exposure、tool 和 task-outcome 的 content-free fingerprints；事件以文件锁和原子替换跨 restart 保持幂等，stage-specific failure reason 不再统一折叠为 extraction failure。PAST-Bench `StepResponse` 现在只携带 process event IDs/digest，不携带 grader 或 score 字段；evaluator-free fixture 验证了 trigger、formation、exposure、task-outcome 的完整事件闭环和 cross-ledger audit。真实 extraction feedback 只能通过 `feedback_chain_from_extraction_example()`、`LayerIntervention.from_extraction_feedback()`、`build_extraction_feedback_interventions()` 和 `build_optimizer_corpus_interventions()` 投影 primary useful/missed 链，其他状态继续保留为 unresolved/censored 诊断；census 同时报告 U/H/M、unresolved/censored 原始计数、ambiguity 计数和 zero-denominator unknown。当前新增的 `build_extraction_feedback_fixture()` 将 durable/temporary past context 与 future task 的 registered SM01 contract 串成可重放 useful/missed case，并由 `FeasibilityInterventionPath` ledger 记录 N+1 replay identity。该结果只是可行性状态，不代表任何层已完成在线优化；后续仍需补充完整 PAST process corpus、独立 matched validation 和失败/provider run 保留。此前 `656 passed` 是历史快照；当前验证为 RSIMem `.venv` `689 passed`、PAST-Bench `397 passed, 2 skipped`。

当前第三阶段的主要任务是 feasibility，不要求一次性完成六层 online optimization。每层分别完成以下验收：

1. √ 定义该层的 decision contract、可控 action space和固定安全边界。
2. √ 证明真实 Host event 或 deterministic fixture 能产生该层输入，并且 process feedback 可以记录 intervention 前后差异。
3. √ 构造至少一个 parent/candidate case，candidate只改变该层且能通过 replay 重建。
4. √ 记录该层的 signal coverage、action variation、outcome variation、ambiguous/unresolved比例和主要缺口。
5. √ 只有 signal 和 action 足够支持下一步实验时，才把该层列为 `optimization-ready`；否则列为 `diagnostic-only` 或 `validation-only`，不强行训练。

推荐先完成六层的 deterministic/shadow feasibility，再选择最有信号的层做真实 adaptive effect experiment。正式效果实验仍需冻结 split、model、budget、replicate、source projection、feedback contract、optimizer config和anti-collapse criteria。

Family规则：

- 第一轮可以用SM01完成pipeline pilot，因为它已有显式TSV reuse contract；该pilot不能作为最终paper effect evidence。
- Formal train、validation和final test使用互斥的semantic family/task-template group，例如在查看结果前从SM01、SM02、SM05中冻结各自角色；具体分配必须由feedback-signal可用性预检决定，而不是按效果挑选。
- 若training family只能提供单一positive或大量ambiguous/censored evidence，不能把这些样本伪造为strict negative；应保留为process/end-to-end exploratory evidence，并在查看final test前按预注册规则调整training family集合。
- Update-ability family不用于第一版update-prompt优化；若其任务能提供extraction质量信号，只能在明确冻结update prompt的前提下作为extraction验证family。

最终效果验收：

- □ 所有variant使用matched task manifest、model、budget、order、sandbox和persistence isolation。
- √ deterministic feasibility cases 与首个真实 provider feedback pilot 均完成预设 replay/trace/audit；provider pilot 的完整 attempt、raw usage 和 no-signal 结果单独保存在 [`extraction_stage3_s1_feedback_20260829.md`](extraction_stage3_s1_feedback_20260829.md)，没有把 unresolved 当成 negative。
- √ 每个 deterministic candidate 至少一次改变目标层的 decision或输入/输出 fingerprint；不要求真实任务分数提升。
- √ `LayerFeasibilityCensus` 报告每层 process-signal coverage、action variation、outcome variation、unresolved/censored 比例、ambiguity 和具体失败原因。
- √ census 报告 U/H/M/unresolved/censored 原始计数以及 resolved useful rate 的分子分母；extraction-specific coverage、empty、missed 和 unknown 分母由 `ExtractionOfflineValidationDecision` 的 ratio evidence 单独报告，不把 unknown silently drop。
- √ raw calls/tokens/retry/latency/storage 由 lifecycle `RawResourceUsage` 记录，injection/recovery 由对应 receipt/ledger event 记录并独立 join；当前不生成未经定义的混合 cost 结论。
- □ 只有后续效果实验才能声明observed uplift、layer superiority或联合版本优势。
- □ 不要求N+2，不声明recursive self-improvement。

### 3C：逐层解锁矩阵、联合 Policy 与论文边界

随后新增 raw-usage contract 反向测试后，RSIMem 当前回归计数更新为
`665 passed`；下方 `656 passed` 记录保留为此前 process-audit 基线。
matched process-corpus gate 反向测试后，当前计数进一步更新为 `666 passed`。
rejected-terminal receipt 反向测试后，当前计数更新为 `667 passed`。
policy reason-code 保真反向测试后，当前计数更新为 `671 passed`。
policy-to-host trigger join 反向测试后，当前计数更新为 `672 passed`。
process corpus 并发 writer 反向测试后，当前计数更新为 `673 passed`。
unbound skip reason 反向测试后，当前计数更新为 `674 passed`。

本轮在新增 process-feedback、process-corpus 与 admission anti-collapse
回归后，RSIMem 当前验证计数为 `665 passed`；PAST-Bench 仍为 `397 passed,
2 skipped`。旧段落中的 `649 passed` 是本轮改动前的 census 基线。

完成可优化性验收后，后续真实效果实验的候选解锁顺序为：

| 阶段 | 新开放的 policy layer | 其余层 | 主要问题 |
| --- | --- | --- | --- |
| S0 | 无，static parent | 全部固定 | 复现 no-persistence、native 和 static RSIMem基线 |
| S1 | Extraction | Trigger、Source、Admission、Commit、Exposure固定 | future feedback能否指导 extraction policy |
| S2 | Trigger | Source、Extraction、Admission、Commit、Exposure按上一阶段固定 | 写入时机是否带来独立提升 |
| S3 | Source selection | Trigger和前序层固定 | 选择哪些上下文进入 formation是否有价值 |
| S4 | Admission | Trigger、Source、Extraction、Commit、Exposure固定 | ADD/NONE/UPDATE policy是否改善 memory quality |
| S5 | Exposure | formation各层固定 | 注入时机、选择、排序和上下文组成是否改善使用 |
| S6 | Joint | 允许联合优化的层集合预先声明 | 组合效果是否超过最佳单层版本 |

当前建议的第一条真实路线是 `S0 -> S1 -> S2 -> S4 -> S5 -> S6`；Source selection先保持固定，只有当 source projection成为明确瓶颈时再加入 `S3`。这不是跳过 Source 层，而是避免第一版同时引入新的上下文裁剪变量。若某一层未通过 feasibility，后续实验可以停在该层并将其列为 future work。

每个 S 阶段必须包含：

- □ 固定 parent policy、唯一新增的开放 layer、candidate artifact和decision budget。
- □ 同一 task manifest 上的 matched parent/candidate，独立 replicate、相同模型、budget、sandbox、persistence和Host adapter。
- □ layer-specific intervention fingerprint，证明 candidate确实改变了该层，而非隐式改变其他层。
- □ task-level end-to-end outcome、strict attribution diagnosis、coverage/empty/non-use和安全结果的完整报告。
- □ 与 static parent、上一阶段版本和最佳单层版本的比较；只报告实际完成的层级 claim。

第一版只运行与extraction claim直接相关的ablation：

- □ Static parent prompt vs adaptive extraction prompt。
- □ Delayed-feedback optimizer vs 不使用delayed feedback的generic prompt rewrite。
- □ Operation/source attribution vs 无差别episode feedback。
- □ 包含missed-extraction evidence vs 只观察已提取fact的future evidence。
- □ Constrained structured rule edits vs unconstrained free-form prompt rewrite。
- □ Fixed `task_completed` trigger vs adaptive trigger。
- □ Fixed exposure vs adaptive exposure。
- □ Best single-layer variant vs joint formation/exposure policy。

以下ablation标记为deferred/not applicable：

- Update-prompt optimization。
- Retrieval-threshold或retrieval-prompt optimization。
- Lifecycle-cost optimization。
- Episodic/procedural memory policy。
- N+2 recursive iteration。
- Cross-host或cross-benchmark generalization。

### 3D：PAST-Bench Family 的 Process-Signal 预检

以下判断基于当前 family/task 定义和可观察工具协议，是运行前的设计预检，不把 hidden grader 当成 process signal。正式选择 training family 前必须执行一次 process-signal census，报告每个 family 的 episode-level coverage、action variation、source-to-outcome chain coverage和ambiguity rate。

| Family | 主要 memory substrate | 可观察 process signal | 当前稀疏性判断 | 适合的 policy 实验 |
| --- | --- | --- | --- | --- |
| `EP01_prior_case_recall` | episodic/session recall | `session_search`、retrieved session artifact、后续 notes/tool action、task completion | 中低 ambiguity；检索调用很清楚，但是否回答正确仍部分落在 output | Exposure、retrieval和episodic recall；不作为第一版 semantic extraction train family |
| `EP02_exception_list_recall` | episodic/session recall | `config_list/get`、`config_update` 的 integration IDs、status、exception字段和重复/错误更新 | 高 process density；关键行为直接体现在 tool arguments 和 state mutation | Episodic admission、exposure和action policy；适合作为 process feedback pilot |
| `SM01_preference_adoption` | semantic preference | notes tool调用、share success、memory injection；格式偏好主要只在最终 TSV/文本中显现 | 低；当前 v10 的 24 条 primary feedback 全部 unresolved，不能只靠 process 判定格式偏好是否被采用 | Extraction端到端 feasibility或最终 offline evaluation，不作为 strict process-only train source |
| `SM02_constraint_retention` | semantic constraint | `notes_share` 的 recipient IDs、调用成功/失败、完成状态 | 中高；边界遵守直接体现在 action 参数，但必须修复完整 advisory roster contract | 第一优先 semantic process-feedback training family |
| `SM05_weak_trigger_preference_adoption` | semantic preference under weak trigger | notes tool调用和share结果；T2/T1偏好采用仍主要体现在输出格式 | 低到中；可观察到触发和写入过程，但偏好本身的后续采用信号稀疏 | Trigger/extraction diagnostic sibling，不作为第一轮唯一训练依据 |

Family 选择原则：

- `EP01/EP02` 的 process signal 强，不代表它们可以直接训练 semantic extraction；memory substrate 必须和开放的 policy layer匹配。
- `SM02` 最适合先验证 semantic process feedback，因为关键约束会体现在 `notes_share` 的真实 recipient 参数，而不是只有最终文本格式。
- `SM01/SM05` 仍可用于端到端结果和 output-based final evaluation，但不能把格式不符合直接转成 extraction failure。
- 在任何 family 上生成 candidate 前，先统计每条episode是否有 trigger、source、extraction、admission、commit、retrieval、exposure、tool/outcome记录，以及这些字段是否存在变化；全是同一个值的字段不能被称为有效学习信号。
- 如果一个 family 的 process chain coverage不足，先补 adapter/fixture；如果 chain完整但 action variation和outcome variation不足，换 family或把它降级为 validation-only，不修改标签门槛制造信号。

## 8. 全局安全、复现与证据要求

- 不修改PAST-Bench task semantics、episode order、grader、answer key或hidden evaluation contract来改善结果。
- Official score只在final evaluation完成后由reporter读取，learner、validator和activation API不可达。
- 所有formal batch要求clean tree、固定commit/tree、唯一batch ID和append-onlyattempt history。
- Raw trace和content-bearing optimizer corpus不提交Git；content-free manifest、audit和derived report可独立审计。
- Prompt optimizer和extraction model的每次物理请求都计入usage，不把离线优化视为免费。
- Adapter、ledger、corpus writer或audit失败不能静默改变agent/memory behavior；formal experiment中证据缺失fail closed。
- 每个schema变更升级version，旧artifact必须显式migrate或拒绝，不能按新语义静默加载。
- “一行适配”是开发者在真实prompt调用边界进行一次显式slot注册，不表示LightRSI可仅凭任意源码路径可靠修改第三方框架。Python backend第一版使用Python SDK；npm/TypeScript侧只消费共享artifact contract或通过IPC调用，不直接patch Python对象。
- 每个任务使用独立commit，包含功能、failure semantics、正反测试和文档状态更新。

## 9. 标准验收命令

在RSIMem仓库根目录运行：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src tests
.venv/bin/python -m pip check
git diff --check
bash -n scripts/*.sh
```

在vendored PAST-Bench目录运行：

```bash
../../.venv/bin/python -m pytest -q
```

每个阶段还必须运行对应focused tests、tracked-secret scan和一次isolated temporary-home restart fixture。真实provider命令只在代码、fixture和preflight全部通过后执行。
正式 provider batch 前可运行 [`provider_probe.md`](provider_probe.md) 中的
completion probe；probe 只判断 endpoint 是否返回非空 completion，不替代
formal usage/process-corpus audit，且失败结果不得进入训练或质量统计。
`run_luna_extraction_feedback_sm01.sh` 与 `run_luna_extraction_matched.sh` 已在
manifest/preflight 注册后、首个 benchmark task 前自动执行该 gate，并将
content-free 结果保存为 batch 内的 `provider_probe.json`。
clean detached worktree 的本地 HTTP fixture 已验证失败 probe 会在首个
PAST-Bench task 前退出且不产生 task trace。

## 10. 三阶段完成定义

### 第一阶段完成

- √ D01-D19均有明确代码修复或正式deferred理由。
- √ Stage 1A-1H全部通过。
- √ Static extraction path不依赖eviction、source/provenance一致、每task只执行一次。
- √ Feedback可覆盖empty/NONE/missed且不误用eager exposure。
- √ Validation不使用cost、硬编码证据或伪造历史identity。
- √ 新实验launcher已冻结plain static parent和prompt-oriented manifest。

### 第二阶段完成

- √ 六层 policy 的 core contract、Host adapter boundary、decision evidence、replay和安全不变量全部通过（deterministic/shadow scope；不等同于真实 adaptive effect）。
- √ Trigger、Source、Extraction、Admission、Commit scheduling和Exposure均可以被fixed policy观测、回放和做 matched intervention；`tests/test_policy_feasibility.py::test_every_layer_case_has_matched_process_intervention_identity` 对六层逐一校验 event、revision、decision、before/after digest 和 action variation。该结论限定为 deterministic/shadow feasibility，不等同于真实 provider effect。
- √ Extraction artifact、trigger/admission/exposure decision和formation lineage可以跨restart重建；完整六层 replay process chain 已通过 JSON ledger 重启读取、逐事件幂等重写和 audit 验证。
- □ 第二阶段不要求任何 adaptive layer已经取得真实效果；真实效果属于第三阶段。

### 第三阶段完成

- √ S0 baseline和六层 feasibility cases完成，并保留完整process/end-to-end feedback（deterministic/shadow scope；真实 provider batch 另行记录）。
- √ 每层都有明确的 `optimization-ready`、`diagnostic-only` 或 `validation-only` 结论。
- √ 每个 `optimization-ready` 层都有至少一个可回放的 parent/candidate case、process signal、action variation和收益假设。
- √ 六层的 Host adapter、decision、execution receipt、lineage和failure semantics可以独立审计。
- □ 论文只声明六层 policy 的可优化性和具体 case；真实 uplift、单层 superiority、联合效果和跨 family 泛化列为后续效果实验结论。

## 11. 当前执行入口

当前实现入口已完成第二阶段六层 Memory Policy 优化基建，并进入第三阶段 deterministic/shadow feasibility 验收；尚未进入真实 adaptive effect run。当前 census 只将 Extraction fixture 暂列 `optimization-ready`，其余层为 `validation-only`，不能宣称六层 aggregate uplift 或真实 execution equivalence。
当前已完成 2A 的 host-neutral contract 验收，并完成 2B/2C/2D 子层的 deterministic policy baseline（trigger、source selection、admission、commit scheduling、exposure）；Hermes lifecycle 已记录 trigger/source decision，并通过独立的 content-free policy evidence ledger 持久化 decision identity、revision、digest、lineage 和 receipt join；静态 semantic writeback 已将 extraction/admission/commit 与真实 ingestion/mutation receipt 关联，Hermes system-prompt boundary 已将 exposure 与 injection receipt 关联；shadow trigger 正反 fixture、跨重启 policy evidence replay、统一 cross-ledger audit、source selection 到 extraction projection 的实际绑定均已通过。第三阶段 deterministic feasibility runner、strict feedback projection、process/hypothesis lineage 和 durable replay ledger 已完成首轮 baseline，但上述实现尚未替代 Hermes 正式 runtime 的 admission/commit/exposure 执行，也不能作为 2B-2D 的最终 adaptive policy 验收。下一步是接入真实 delayed feedback corpus，补足各层 resolved process/outcome variation，并在满足 gate 后进行 matched effect experiment。
第一阶段 1A-1H、原 extraction artifact/runtime binding 基础和低成本 live plain-parent acceptance 已完成，记录在
[`extraction_stage1_acceptance_20260828.md`](extraction_stage1_acceptance_20260828.md)。
已有 extraction artifact 的 deterministic contract、content-free evidence assembler、atomic activation、
rejection、restart、rollback、runtime binding和activation fingerprint基础可以复用，但不代表六层 policy 基建或第三阶段效果已经完成。验收记录见
[`extraction_stage2f_acceptance_20260828.md`](extraction_stage2f_acceptance_20260828.md)。
Stage 2E plain-parent feedback 已通过显式注册的 backup provider 完成 3 个 clean replicate。v10 的
source/capture/feedback exact join和private corpus可重建，但24个primary label全部为unresolved，
actionable count为0，低于冻结门槛2；optimizer以0次模型调用返回`NO_PROPOSAL`，未生成candidate。
这被接受为 strict attribution 的 no-signal pilot，不再阻止 exploratory end-to-end feasibility；但第三阶段仍必须先逐层完成六层 decision/evidence feasibility case。验收记录见
[`extraction_stage2e_feedback_v10_20260828.md`](extraction_stage2e_feedback_v10_20260828.md)；
此前provider失败记录见[`extraction_stage2e_provider_attempts_20260828.md`](extraction_stage2e_provider_attempts_20260828.md)。
2026-08-29 的后续 clean-worktree SM02 重试仍未形成 eligible batch：主 endpoint
在首个 replicate 的 reflection 请求遇到容量错误，备用 endpoint 的 9 条 trace
均为 think-only 空响应且缺少 usage。两次 attempt 均以
`incomplete_model_usage` fail-closed，完整 manifest/raw trace 仅作为 provider
diagnostic 保留，详见
[`extraction_stage3_sm02_provider_attempts_20260829_v3_v4.md`](extraction_stage3_sm02_provider_attempts_20260829_v3_v4.md)。
同日 SM01 的后续 clean-worktree 尝试又发现两处真实运行边界问题：v5 将独立
reflection episode 错误送入 semantic extraction，已由 `e22af5c` 修复并加入
回归测试；v7 在 shared-cold 与 attempt 目录中看见同一份 process event，已由
`569e295` 在两个 formal launcher 中先构造 canonical `ProcessCorpus`、再审计
完全相同的 event ID。v6 及 post-v7 probe 仍因 provider HTTP 503 在首个 task
前 fail-closed。完整记录见
[`extraction_stage3_sm01_feedback_attempts_20260829_v5_v8.md`](extraction_stage3_sm01_feedback_attempts_20260829_v5_v8.md)。
修复后 `s1-sm01-feedback-20260829-v9a` 已完成 3 个 clean parent replicate，
231 条 canonical process event 和全部 raw usage 均可重建；24 条 primary
feedback 全部为 `unresolved`，strict corpus 返回 `NO_PROPOSAL`，因此仍未
产生可进入 matched validation 的 N+1 candidate。结果见
[`extraction_stage3_sm01_feedback_v9a_20260829.md`](extraction_stage3_sm01_feedback_v9a_20260829.md)。
随后 SM02 process-signal family 的 `s1-sm02-feedback-20260829-v5` 已完成 3 个
clean parent replicate，产生 8 条 contract-resolved `missed` 和 16 条
`unresolved`；没有 useful/harmful variation，不能直接生成可激活 candidate。
其 content-bearing corpus 已通过重复上下文压缩保持在冻结 optimizer 输入预算
内；早期 `json_object` 响应两次缺少 `reason_codes`，严格 schema 拒绝。随后
provider 恢复后，commit `d2d06fc` 将 adapter 固定为冻结的 strict JSON Schema
response contract，同一 corpus 成功产生一个经过 static-safety 检查的 proposal
candidate。candidate 仍未 ACTIVE、未进入 matched validation，也不构成 uplift
证据。详细记录见
[`extraction_stage3_sm02_feedback_v5_20260829.md`](extraction_stage3_sm02_feedback_v5_20260829.md)
和
[`extraction_stage3_sm02_optimizer_retry_20260829.md`](extraction_stage3_sm02_optimizer_retry_20260829.md)。

随后独立的 SM05 process pilot `s1-sm05-feedback-20260829-v1` 完成了三个
clean parent replicate，得到 24 条 extraction-owned `missed`、12 条
`unresolved` 和每 replicate 89 条（batch 共 267 条）canonical process event。
该 family 的 request 在
replicate 展开后超过 160k 字符；commit `f194bf8` 增加仅在预算溢出时触发的
deterministic replica compaction，保留全部 primary IDs 和 delayed identity，
将 request 控制在 158,538 字符。两次 schema-valid optimizer completion 都因
复制 corpus-specific value 被 static safety gate 拒绝，未生成 candidate；
结果见
[`extraction_stage3_sm05_optimizer_20260829.md`](extraction_stage3_sm05_optimizer_20260829.md)。

commit `3373a78` 新增显式 `ExtractionSplitPlan` contract，并让 matched
preflight 在提供 split plan 时校验 validation family、template group 和
task manifest digest。当前 SM01、SM02、SM05 的 pilot 全部已经占用 train
role，因此不得把这些 train manifest 重命名为 validation。当前已冻结一个
候选 held-out 计划：SM02 作为 train、SM03_fact_correction 作为 validation、
SM04_rule_migration 作为 final_test，计划文件为
`configs/extraction_split_plan_sm02_sm03_sm04.json`；SM03 的
`sm03_fact_correction_v1` contract 仅用于冻结 update prompt 下的 extraction
validation。split 审计见
[`extraction_stage3_split_audit_20260829.md`](extraction_stage3_split_audit_20260829.md)。
正式 feedback/matched-preflight launcher 都强制提供该 split plan；feedback
只能使用 plan 的 `train` assignment，validation/final family 误传给 feedback
会在模型调用前 fail closed；省略计划同样会失败。该计划只解除 split identity
阻塞，不代表 validation 已执行；后续仍需
candidate trial profile、clean trees、provider probe、完整 process corpus
和 matched parent/proposal run。
后续顺序仍严格按照：

```text
1A -> 1B -> 1C -> 1D -> 1E -> 1F -> 1G -> 1H
   -> 2A -> 2B -> 2C -> 2D -> 2D.1 -> 2D.2 -> 2D.3
   -> 2E -> 2F -> 2G -> 2H -> 2I
   -> 3A/3B deterministic/shadow census（首轮已完成）
   -> 3A process feedback/replay gate 与各层补充 case
   -> 3B(Trigger/Source/Extraction/Admission/Commit/Exposure feasibility)
   -> 3C(S0/S1/.../S6 optional effect experiments)
```

任何上游contract缺陷必须在当前阶段修正，不通过后续模块、脚本参数或手工数据绕过。

跑实验的API key放在/mnt/20t/xubuqiang/Study/api_key.md，可以去那边查看。
