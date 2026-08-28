# RSIMem Extraction-Prompt Online Adaptation 实现与验收清单

最后更新：2026-08-28

## 1. 文档定位

本文是 RSIMem 当前唯一的实现与验收主清单。旧 checklist 中已经完成的 Hermes、PAST-Bench、semantic writeback、transaction、operation graph、feedback store、activation 和 rollback 基础设施继续保留，但不再沿用“future utility per cost + retrieval threshold adaptation”作为论文主线。

当前工作严格分为两个串行阶段：

- 第一阶段：修正现有实现中偏离 extraction-prompt adaptation 的目标、契约、证据和实验 gate。
- 第二阶段：实现由 delayed future feedback 驱动的 extraction prompt N -> N+1，并在未来 PAST-Bench matched run 中验证效果。

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
- 当前只优化 semantic fact-extraction prompt。
- Extraction optimization core必须与memory backend和Host解耦；Mem0-flat/Hermes只是第一个adapter和实验载体，不属于optimizer contract。
- Internal `ADD/UPDATE/DELETE/NONE` prompt、related-memory retrieval、future retrieval/injection surface、route、invocation boundary、backend 和基础模型参数保持冻结。
- Episodic memory、procedural memory、context eviction、physical rewrite、其他 host 和其他 benchmark 全部延后。

### 2.2 当前方法定义

```text
completed Hermes experience
  -> frozen semantic compilation boundary
  -> extraction prompt N
  -> fixed Mem0-flat update/writeback
  -> future eager exposure / observable use / downstream outcome
  -> delayed extraction feedback
  -> extraction prompt proposal N+1
  -> held-out validation
  -> production activation
  -> future matched PAST-Bench evaluation
```

### 2.3 Policy Update Signal

允许进入 extraction prompt optimizer 的信号：

- Extraction 当时可见的 bounded completed context。
- Policy N 实际生成的 extracted fact set。
- Fact 或 extraction set 后续是否有 exposure opportunity。
- 后续是否被注入、显式使用、supersede、证明冲突或关联到 deployment-observable outcome。
- 受 observation cutoff、censoring 和 attribution confidence 约束的 missed-extraction evidence。

禁止进入 optimizer 的信号：

- Official PAST-Bench score、grader、answer key、hidden expectation 或 judge feedback。
- Validation/test batch 中尚未到达的 future evidence。
- Model calls、tokens、latency、storage、recovery 或任何合成 cost unit。
- Route selection、invocation schedule、backend selection 或模型参数更新。

### 2.4 Cost 与效果边界

- Cost 只属于 evaluation/accounting plane，不属于 policy learning plane。
- 正式报告保留 model calls、各 token bucket、retry、wall time、storage bytes、injected chars 和 recovery duration 等 raw vector。
- 不把 request、token、byte、character 和 millisecond 直接相加为可用于结论的 `lifecycleCostUnits`。
- Provider 未返回的 usage bucket 保持 unknown；当前廉价 provider 的 unknown 不阻塞功能开发，正式实验使用支持完整 usage 的 provider 或在报告中保留 unknown。

### 2.5 当前最低 Claim

第一版最低目标是一次真实 online extraction-policy adaptation：

1. Policy N 在过去 deployment 中产生可审计 delayed feedback。
2. 只使用过去 evidence 生成 extraction prompt N+1。
3. N+1 通过独立 held-out validation 后被激活。
4. 未来 run 实际加载 N+1，并至少改变一次 extraction output 或 persisted memory behavior。
5. Adaptive N+1 的 matched aggregate primary task score 高于 static N。

完成以上条件可以声明 observed online extraction-policy adaptation。N+2、repeated/recursive self-improvement、跨 family 泛化和统计显著 superiority 不属于当前完成条件。

### 2.6 第一版 Useful Signal 与优化目标

第一版不让一个通用LLM直接判断“这条memory是否有用”。每个family必须在运行前注册版本化`OpportunityContract`、`UseContract`和`OutcomeContract`，再由统一resolver将deployment-observable evidence映射为label：

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
- `injected_not_used`默认是`unresolved`。只有预注册contract能证明存在使用机会且缺失行为属于extraction component时，才允许产生negative/missed signal。
- Memory被正确提取和注入但Agent没有使用时，不自动惩罚extraction prompt；该问题归入future application/retrieval component或unresolved。
- Observation window不完整、没有相关future task、多个artifact无法唯一归因、证据冲突或只有模型主观判断时，分别记为`censored`或`unresolved`，不进入resolved denominator。
- 多fact共同影响一次future outcome且无法拆分贡献时，只生成一个extraction-set-level label；不把同一次成功复制成多个fact-level positive。Fact-level label只在artifact/use/outcome可以唯一绑定时生成。
- Primary optimization unit是一次completed source对应的extraction set及其future opportunity，不是fact数量，防止把一个fact拆成多条来放大reward。

第一版primary objective冻结为resolved observed useful rate：

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

SM01第一版contract必须明确：只有预注册的`eval_near/eval_far`报告任务构成TSV preference opportunity；future task输入没有重新提供TSV preference时，合法四列表头及非空数据行才构成memory-specific use signal；task completion及预注册的非grader输出/工具条件构成outcome signal。单artifact时允许fact-level归因，多artifact时最多生成set-level evidence。仅检测到TSV格式而没有memory-specific opportunity，当前turn已直接要求TSV，或只因最终回答缺少TSV，均不能自动把某条已正确提取的memory判为harmful。

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

第一阶段关闭后才能开始 prompt optimizer 或新的 adaptive live run。

## 6. 第二阶段：实现 Extraction Prompt N -> N+1

### 2A：Extraction Policy Envelope 与 Artifact

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

### 2B：Content-Bearing Extraction Optimizer Corpus

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

### 2C：Extraction Prompt Optimizer

第一版使用一个受控LLM meta-optimizer，根据历史example总结“什么应该提取、什么不应该提取”的规则。Optimizer不直接自由重写整个prompt，而是对parent `ExtractionPolicySpec`生成结构化rule edits，再由frozen compiler生成replacement policy body。

功能需求：

- √ 冻结optimizer system instruction、input schema、output schema、model profile、temperature、token budget和timeout。
- √ 输入包含parent policy body，以及按useful/harmful/missed/unresolved分类的bounded training examples。
- √ Optimizer input按source/set/fact层级分组，显式说明`unresolved/censored`不是negative，且同一set不能按fact数重复加权。
- √ Output只允许`ADD_RULE/REPLACE_RULE/DELETE_RULE` edits、每个edit的evidence example IDs和结构化reason codes；candidate body由frozen compiler生成，不接受模型提供的第二份不一致body。
- √ Protected durability、source-grounding、credential和schema规则只能位于frozen wrapper或protected rule set，optimizer不能删除或弱化。
- √ Optimizer objective明确要求提高future observed useful proportion，同时保持harmful、coverage、empty和missed constraints；不输入cost或official task score。
- √ Formal policy update默认只生成一个candidate；若未来增加K candidates，K和selection rule必须在查看validation结果前冻结。
- √ 无resolved signal、只有censored evidence或attribution不足时返回`NO_PROPOSAL`，不为了推进实验强行改prompt。
- √ Optimizer调用usage单独记录，但不作为optimizer目标或candidate排序依据。

测试与验收：

- √ Useful-only、harmful、missed、conflicting、low-sample和no-signal fixtures通过。
- √ Optimizer将`injected_not_used`误当negative、将application failure归因给extraction或复制一次set success给多个facts时，proposal contract拒绝。
- √ 相同captured optimizer completion生成相同artifact；不要求重新调用随机模型逐字复现。
- √ Candidate body不能复制training source中的用户事实、task ID、答案、专有值或长n-gram；prompt必须学习规则，不能充当memory store。
- √ Candidate rule不得出现SM01、TSV、固定列名、项目名或其他family-specific shortcut，除非该词原本属于frozen generic root contract；命中shortcut时拒绝而不是交给validation碰运气。
- √ Prompt injection、credential exfiltration、schema override和benchmark-specific shortcut candidate被拒绝。

### 2D：Static Safety 与 Offline Prompt Validation

功能需求：

- √ Contract validator检查body长度、字符、forbidden instruction、wrapper/schema digest和parent lineage。
- √ Deterministic extraction suite覆盖durable preference、constraint、temporary request、unresolved claim、assistant-only acknowledgement、tool evidence、credential/path和empty source。
- √ Candidate必须保持严格JSON `{facts: string[]}` output contract。
- √ Offline validation在独立historical split上比较parent/candidate extraction utility，不使用official score。
- √ Validation按set-level计算resolved useful rate，并同时检查harmful、non-empty coverage、empty extraction和high-confidence missed；不允许只靠少提取获得虚假提升。
- √ 所有ratio同时输出numerator、denominator和unknown count；resolved denominator不足时拒绝，不能只报告百分比。

测试与验收：

- √ Candidate不能降低所有输出为空来通过negative-only样本。
- √ Candidate只提取一个高置信fact时，即使useful rate为100%，只要coverage低于冻结floor也必须拒绝。
- √ Candidate不能通过复制source或输出完整transcript提高recall。
- √ Offline quality不严格高于parent时保持REJECTED。
- √ Offline accepted只允许进入matched trial，不可直接写production ACTIVE。

### 2E：Matched Trial、Activation 与 Rollback

功能需求：

- □ 在独立PAST-Bench validation batch中轮换运行parent N与proposal N+1。
- √ Pair使用相同family、episode manifest、model、budget、home seed state和feedback contract。
- √ Activation只看deployment-observable set-level useful rate、anti-collapse constraints和安全审计，不看official task score。
- √ Proposal必须达到strict useful-rate delta、resolved sample下限，并同时通过harmful、coverage、empty、missed gate才激活；equal、unknown、conflicting或任一约束失败保持REJECTED。
- √ Production activation原子切换唯一ACTIVE extraction artifact。
- √ Operator rollback恢复parent N；自动rollback只在有真实定义的safety violation时触发。

测试与验收：

- √ Trial config不能被official final launcher误当production config。
- √ Activation crash不会产生两个ACTIVE artifact。
- √ Rejection、重复activation、restart和rollback幂等。
- √ Decision记录真实pair IDs、artifact digests、U/H/M/unresolved/censored counts、各ratio分子分母、coverage、quality delta、constraint results和reason codes。

### 2F：Runtime Prompt Binding 与 Activation Fingerprint

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

### 2G：Deterministic End-To-End Gate

- □ 构造一个过去context中含durable与temporary信息、未来任务只使用durable信息的fixture。
- □ Policy N产生至少一个可归因问题，例如遗漏durable fact或提取temporary fact。
- □ Fixture分别构造完整`opportunity -> use -> successful outcome` useful链和`source -> no equivalent extraction -> future demand -> absence-attributed outcome` missed链。
- □ 删除任一useful/missed链节点后label退化为unresolved，而不是继续贡献optimizer reward。
- □ Delayed feedback构建optimizer corpus并生成N+1。
- □ N+1通过offline/matched fixture validation并被激活。
- □ Future fixture实际加载N+1，改变extraction和persisted memory，并改善deployment-observable outcome。
- □ 全链不读取grader/answer，不使用cost信号，不修改update/retrieval policy。
- □ Restart、rejection、no-proposal和rollback反向路径全部通过。

验收：只有结构化证据可以精确重建`N -> past feedback -> N+1 -> future changed extraction/outcome`时，才进入真实provider实验。

### 2H：真实 PAST-Bench Online Adaptation

正式实验严格按以下顺序执行，每一步审计通过后才能进入下一步：

1. 冻结train/validation/final family或task-template split、model、budget、replicate、source projection、opportunity/use/outcome contracts、optimizer config、resolved useful-rate objective和全部anti-collapse acceptance criterion。
2. 使用static parent N运行至少3个independent unseeded feedback replicates。
3. 构建并审计optimizer corpus；若没有足够resolved extraction signal，停止并更换预声明的semantic memory family，不降低gate。
4. 生成一个candidate N+1并完成offline validation。
5. 运行独立parent/proposal matched validation batch；只有strict positive resolved useful-rate delta且全部anti-collapse/safety constraints通过才激活。
6. 使用production ACTIVE N+1运行未来matched final batch，同时运行static N control。
7. 汇总no persistence、native Hermes、static N和adaptive N+1；native+ledger只作为accounting control。

Family规则：

- 第一轮可以用SM01完成pipeline pilot，因为它已有显式TSV reuse contract；该pilot不能作为最终paper effect evidence。
- Formal train、validation和final test使用互斥的semantic family/task-template group，例如在查看结果前从SM01、SM02、SM05中冻结各自角色；具体分配必须由feedback-signal可用性预检决定，而不是按效果挑选。
- 若training family只能提供单一positive或大量ambiguous/censored evidence，不允许强行训练prompt；应在查看final test前调整预声明的training family集合，不降低resolved-signal gate。
- Update-ability family不用于第一版update-prompt优化；若其任务能提供extraction质量信号，只能在明确冻结update prompt的前提下作为extraction验证family。

最终效果验收：

- □ 所有variant使用matched task manifest、model、budget、order、sandbox和persistence isolation。
- □ 每个variant完成预设replicate；failed/provider run保留并单独报告。
- □ N+1在真实run中至少一次改变extraction并影响persisted memory。
- □ Adaptive N+1 aggregate primary task score严格高于matched static N。
- □ 报告每个replicate、task score、pass rate、persistence gap和activation funnel。
- □ 报告U/H/M/unresolved/censored原始计数、resolved useful rate及其分子分母、coverage和empty rate；不把unknown silently drop。
- □ 报告raw calls/tokens/retry/latency/storage/injection/recovery，不生成无定义的混合cost结论。
- □ 若只有aggregate正向均值，可声明observed uplift，不声明统计显著superiority。
- □ 不要求N+2，不声明recursive self-improvement。

### 2I：Ablation 与论文边界

第一版只运行与extraction claim直接相关的ablation：

- □ Static parent prompt vs adaptive extraction prompt。
- □ Delayed-feedback optimizer vs 不使用delayed feedback的generic prompt rewrite。
- □ Operation/source attribution vs 无差别episode feedback。
- □ 包含missed-extraction evidence vs 只观察已提取fact的future evidence。
- □ Constrained structured rule edits vs unconstrained free-form prompt rewrite。

以下ablation标记为deferred/not applicable：

- Update-prompt optimization。
- Retrieval-threshold或retrieval-prompt optimization。
- Lifecycle-cost optimization。
- Episodic/procedural memory policy。
- N+2 recursive iteration。
- Cross-host或cross-benchmark generalization。

## 7. 全局安全、复现与证据要求

- 不修改PAST-Bench task semantics、episode order、grader、answer key或hidden evaluation contract来改善结果。
- Official score只在final evaluation完成后由reporter读取，learner、validator和activation API不可达。
- 所有formal batch要求clean tree、固定commit/tree、唯一batch ID和append-onlyattempt history。
- Raw trace和content-bearing optimizer corpus不提交Git；content-free manifest、audit和derived report可独立审计。
- Prompt optimizer和extraction model的每次物理请求都计入usage，不把离线优化视为免费。
- Adapter、ledger、corpus writer或audit失败不能静默改变agent/memory behavior；formal experiment中证据缺失fail closed。
- 每个schema变更升级version，旧artifact必须显式migrate或拒绝，不能按新语义静默加载。
- “一行适配”是开发者在真实prompt调用边界进行一次显式slot注册，不表示LightRSI可仅凭任意源码路径可靠修改第三方框架。Python backend第一版使用Python SDK；npm/TypeScript侧只消费共享artifact contract或通过IPC调用，不直接patch Python对象。
- 每个任务使用独立commit，包含功能、failure semantics、正反测试和文档状态更新。

## 8. 标准验收命令

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

## 9. 两阶段完成定义

### 第一阶段完成

- √ D01-D19均有明确代码修复或正式deferred理由。
- √ Stage 1A-1H全部通过。
- √ Static extraction path不依赖eviction、source/provenance一致、每task只执行一次。
- √ Feedback可覆盖empty/NONE/missed且不误用eager exposure。
- √ Validation不使用cost、硬编码证据或伪造历史identity。
- √ 新实验launcher已冻结plain static parent和prompt-oriented manifest。

### 第二阶段完成

- □ Stage 2A-2I全部通过或按当前claim明确标记not applicable。
- □ Exact extraction prompt N+1可生成、验证、激活、加载、回滚和重放。
- □ 真实future run证明N+1改变extraction与memory behavior。
- □ Adaptive aggregate primary task score高于matched static。
- □ 论文claim严格限制为实际通过的online extraction-policy adaptation证据。

## 10. 当前执行入口

当前实现入口已返回 **Stage 2E：独立 PAST-Bench live matched validation trial**。
Stage 2E的真实PAST validation run仍保持未完成。Stage 1A-1H、
Stage 2A、Stage 2B、Stage 2C 和 Stage 2D 已经通过，低成本 live plain-parent acceptance 记录在
[`extraction_stage1_acceptance_20260828.md`](extraction_stage1_acceptance_20260828.md)。
Stage 2D只证明candidate通过静态安全、deterministic extraction suite和独立historical
split offline gate；其accepted状态只能进入matched trial，不代表production activation。
Stage 2E的decision、trial-only config、content-free evidence assembler、atomic activation、
rejection、restart和rollback contract已经通过deterministic验收。Stage 2F也已完成
deterministic runtime binding、PAST validation-only transport、activation fingerprint和
matched drift gate；验收记录见
[`extraction_stage2f_acceptance_20260828.md`](extraction_stage2f_acceptance_20260828.md)。
当前必须执行Stage 2E独立matched validation batch，不能直接进入Stage 2G或production final run。
Stage 2E plain-parent feedback已从clean detached worktree发起；认证后的两次运行均因provider
HTTP 503 capacity失败，且没有生成source/capture/feedback evidence，因此不得进入optimizer。
运行记录见[`extraction_stage2e_provider_attempts_20260828.md`](extraction_stage2e_provider_attempts_20260828.md)。
后续顺序仍严格按照：

```text
1A -> 1B -> 1C -> 1D -> 1E -> 1F -> 1G -> 1H
   -> 2A -> 2B -> 2C -> 2D -> 2E(contract) -> 2F
   -> 2E(live matched trial) -> 2G -> 2H -> 2I
```

任何上游contract缺陷必须在当前阶段修正，不通过后续模块、脚本参数或手工数据绕过。

跑实验的API key放在/mnt/20t/xubuqiang/Study/api_key.md，可以去那边查看。
