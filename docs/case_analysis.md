# Process-Signal Case Analysis

日期：2026-08-30（含 clean parent rerun 更新；2026-09-01 implementation checkpoint supersedes earlier infrastructure wording）

> **Current implementation note (2026-09-01).**  The earlier statements in
> this historical analysis that tool-level exact joins, logical-case
> persistence, or pure-process separation were incomplete are superseded by
> the current runtime checkpoint.  Those deterministic/runtime contracts are
> now implemented and covered by the `1101 passed` RSIMem suite and the
> `401 passed, 2 skipped` vendored PAST-Bench suite.  The scientific conclusion
> is unchanged: current clean parent cases remain `STOP_NO_SIGNAL`, and no
> extraction-owned N+1 signal or effect claim is established.

## 1. 核心问题与结论

本文分析 RSIMem 当前已经完成的 PAST-Bench/Hermes case，回答论文现阶段最重要的前提问题：在不读取 grader、标准答案、hidden expectation、official score 和 cost/accounting value 的情况下，仅根据形成记忆时的上下文，以及后续真实发生的 extraction、持久化、检索、注入、工具调用和任务结果，能否识别 memory policy 的问题，并推导出可泛化的优化方向。

分析链路固定为：

```text
source context
-> extraction request/output
-> persisted memory or empty result
-> retrieval query/result
-> exposure/injection
-> tool call/result or downstream behavior
-> terminal task observation
```

截至当前日志，结论分为三层：

| 层次 | 当前结论 | 证据边界 |
| --- | --- | --- |
| 可观测 | 成立（当前 runtime） | 当前 lifecycle event、source、memory、retrieval、exposure、tool call/result 和 task outcome 可以通过稳定 identity 重建；历史批次的 projection 缺口只作为审计背景保留 |
| 可诊断 | 阶段诊断部分成立，extraction 诊断尚未成立 | 纯 process 能区分 `not_exposed`、`injected_not_used`、`retrieval_miss` 等阶段现象；当前 useful/missed resolver 还依赖 family-specific audit contract，且 SM01/SM02/SM05 都没有可信的 extraction-owned resolved chain |
| 可优化 | 尚未由真实 provider case 证明 | 当前没有一组可以安全生成 N+1 extraction policy 的真实反馈；旧 SM02 candidate 基于失效标签，不能使用 |

因此，当前最重要的工作不是扩大 optimizer，也不是立即跑 adaptive uplift，而是验证 deployment-visible process 是否真的包含足够的优化信号。这个问题如果成立，RSIMem 才能主张 online memory-policy self-improvement；如果不成立，也必须准确报告缺失的是哪段证据，而不能用 grader 反馈或错误归因制造训练标签。

## 2. 分析范围

### 2.1 可用于语义分析的完成批次

| batch | family | replicate | historical primary feedback | 当前 disposition |
| --- | --- | ---: | --- | --- |
| `s1-sm01-feedback-20260829` | `SM01_preference_adoption` | 3 | 24 unresolved | 早期完整 no-signal batch；被 v9a 的修正 runtime 重复验证，不单独增加样本量 |
| `s1-sm01-feedback-20260829-v9a` | `SM01_preference_adoption` | 3 | 24 unresolved | 当前 SM01 authoritative pilot |
| `s1-sm02-feedback-20260829-v5` | `SM02_constraint_retention` | 3 | 8 missed, 16 unresolved | historical missed 已撤销，必须重跑 |
| `s1-sm05-feedback-20260829-v1` | `SM05_weak_trigger_preference_adoption` | 3 | 24 missed, 12 unresolved | historical missed 已撤销，必须重跑 |
| `sm02-feedback-20260829-rerun-main` | `SM02_constraint_retention` | 3 | 12 missed, 12 unresolved | historical missed 与 proposal 均已撤销 |

四批当前 process census 使用的 clean parent runs 是 SM01 v9a、SM02 v5、SM05 v1 和 SM02 rerun；早期 SM01 batch 没有当前版本的 canonical `process_corpus.json`，只作为 no-signal strict-feedback 的重复证据。所有这些 batch 都是 fixed parent policy，不是 parent/candidate matched comparison。

2026-09-01 的 authoritative fresh provider-backed reruns 已替代下表中的
2026-08-30 process-only counts，详见
[`extraction_stage2_clean_parent_20260901.md`](extraction_stage2_clean_parent_20260901.md)。

### 2.1.1 Historical clean process rerun（2026-08-30）

为避免继续使用已撤销的 SM02/SM05 historical labels，使用当前
process-signal builder 和纯 process projection 在干净 detached worktree
中重新执行了两个 fixed parent-only batch：

| batch | family | replicate | pure-process events | logical cases | status | disposition |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `sm02-clean-process-20260830` | SM02_constraint_retention | 3 | 379 | 9 | 9 `observable_only` | `STOP_NO_SIGNAL` |
| `sm05-clean-process-20260830-v2` | SM05_weak_trigger_preference_adoption | 3 | 495 | 10 | 8 `observable_only`, 2 `censored` | `STOP_NO_SIGNAL` |

两批均为 audit-clean、无 candidate intervention；Trigger、Source
selection、Extraction 均保持 shadow `pending`。SM02 的详细记录见
[`extraction_stage2_sm02_process_signal_final_20260830.md`](extraction_stage2_sm02_process_signal_final_20260830.md)，SM05 见
[`extraction_stage2_sm05_process_signal_20260830.md`](extraction_stage2_sm05_process_signal_20260830.md)。它们只增加 process
observability 证据，不把 benchmark-audit 的 `missed`/`unresolved` 标签
转成 learner signal。

### 2.1.2 Fresh provider-backed clean parent (2026-09-01)

SM02 和 SM05 按冻结 parent profile 各运行 3 个 replicate；每批都通过
provider preflight、usage/audit、privacy 和 process-corpus 校验。新的
batch-level `logical_case_v1` census 为：

| batch | pure-process events | optimizer examples | logical cases | physical observations | status | replicate consistency |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `s2-sm02-clean-parent-20260901-v1` | 422 | 21 (`unresolved`) | 25 | 48 | 3 `observable_only`, 3 `diagnostic_only`, 19 `censored` | 1.0 |
| `s2-sm05-clean-parent-20260901-v1` | 565 | 33 (`unresolved`) | 31 | 63 | 2 `observable_only`, 7 `diagnostic_only`, 22 `censored` | 1.0 |

两批的 pure optimizer corpus 都固定为 `evidence_plane=pure_process`、
`process_signal_gate=no_signal`、zero optimization cases。SM05 的 12 条
`missed` 仅存在于独立 benchmark-audit projection；它们没有进入 pure
corpus。故本次 2D 决策仍为 `STOP_NO_SIGNAL`，不生成 N+1。

后续的 `s2-sm02-clean-parent-20260901-v2` 尝试不进入上表或任何 signal
census：replicate 1 audit-clean，replicate 2 首次 provider
`InternalServerError` 后重试通过，replicate 3 因 fail-closed 的
`skip/defer extraction` 触发 `incomplete_model_usage`。该批次只保留为
provider/runtime 诊断，不能按 partial replicate 或 process evidence 使用。

### 2.2 不能进入语义结论的尝试

SM01 v5/v6/v7/v8 以及 SM02 v2/v3/v4 等失败或不完整尝试只用于定位 provider、reflection boundary、duplicate corpus 和 usage-audit 问题。它们没有完整且合格的 observation window，不能进入 optimizer、resolved denominator、matched validation 或任务质量结论。

Deterministic six-layer fixture 共覆盖 7 个 case，可以证明 decision contract 可观察、可干预、可回放和 fail closed；fixture-local useful/missed 不是来自真实 Hermes deployment，因此只能作为 feasibility evidence，不能证明真实 process signal 足够。

### 2.3 Evidence 使用边界

- Content-free `process_corpus.json`、policy decision ledger、memory event ledger 和 receipts 用于确认事件是否发生、顺序是否正确、身份能否 join 和是否可以 replay。
- Owner-controlled source/extraction capture 与 extraction feedback artifact 用于人工审计当时模型可见的 source、extraction output、persisted artifact、future opportunity、use 和 outcome；它们不进入 Git、公共 ledger 或 benchmark learner 输入。
- PAST-Bench grader、answer key、expectations、official score、hidden future label 和 cost/accounting value不参与 process-signal 判断；official score 只允许最终 reporter 在独立 evaluation plane 使用。
- 本文是对历史 artifact 的后续归因审计。早期报告中“SM02/SM05 missed 有效”“Extraction optimization-ready”或“SM02 candidate trusted”的表述已被本文撤销；原始报告和 artifact 保留为审计历史，但不得继续作为当前结论。

### 2.4 Benchmark Audit Contract 不等于纯 Process Signal

当前 strict resolver 虽然不读取 grader 和 answer key，但仍包含以下 PAST-Bench-specific 先验：

1. `family_id + eval_near/eval_far` 会直接把 contract 的全部 `memory_scope_keys` 写入 `task_semantic_keys`，因此 opportunity 不是从当前用户输入、工具状态或环境变化中独立检测出来的。
2. Resolver 硬编码 TSV 表头、priority/date format、Ava Chen recipient boundary 和 Phoenix date 等 family-specific parser。
3. `family_id` 和 stage 是 benchmark adapter metadata；真实部署通常没有这些标签。

这些 contract 可用于离线审计“已知任务语义下 attribution implementation 是否正确”，但不能作为论文中“仅凭 deployment process 就发现优化方向”的证据。本文因此同时报告两条证据平面：

- **Pure process plane**：只使用 source、memory lifecycle、retrieval/exposure、tool call/result 和 task completion。它目前可以完成可观测和 stage diagnosis，不能完成 extraction-owned diagnosis。
- **Benchmark audit plane**：额外使用预注册 family contract 检查 attribution 边界。它已经发现 false missed，但这些 family parser 不能进入通用 learner 或作为可泛化 policy edit 的内容。

## 3. 三层判定标准

### 3.1 可观测

一个 case 只有在 source、extraction、持久化、future retrieval/exposure、工具行为和 terminal observation 能通过稳定 identity 重建时才算可观测。`decision_observed`、HTTP success、`task_completed` 或 memory injection 只说明某个阶段发生过，不能单独说明 extraction 正确或错误。

### 3.2 可诊断

只有同时满足下列条件，问题才可以归因到 extraction：

1. Source 中确实存在可以跨任务复用的 durable information。
2. Policy N 的 extraction output 缺少等价信息，或者生成了明确错误、短暂或冲突的 memory。
3. Future task 对该信息存在 deployment-visible opportunity。
4. Retrieval、exposure、tool execution 和 Agent application 没有先失败，或者其失败已经被明确归入其他 component。
5. Opportunity、memory use 和 outcome 之间存在稳定、可重建的 operation join。
6. 当前 future input 没有直接重新提供同一信息，否则行为不能归因于过去 memory。

如果 memory 已正确生成但没有被检索、没有被注入，或者 Agent 看到了却没有使用，应分别记录为 `retrieval_miss`、`not_exposed`、`injected_not_used` 或 `unresolved`，不能直接惩罚 extraction。`Unresolved` 不是 harmful，也不是 missed。

### 3.3 可优化

一个 case 只有在诊断结果能够抽象成不依赖 task ID、具体答案、recipient、文件名和 benchmark rule 的规则时，才支持 policy optimization。例如，“用户明确声明一条未来持续适用的边界规则，extraction 应保留其 scope、condition 和 prohibition”是通用 policy 方向；“SM02 必须记住 Ava Chen”是 benchmark 具体值，不能进入 candidate。

Candidate 还必须经过 source-value、task-ID、benchmark-shortcut 和长 n-gram safety gate，并在独立 held-out split 上只验证一次。Schema-valid proposal 不等于有效 N+1。

## 4. Case-by-Case 分析

### 4.1 SM01：TSV preference adoption

**Case 构成。** `SM01_COLD_001` 是无持久化先验的 baseline；`SM01_LEARN_A_001` 和 `SM01_LEARN_B_001` 形成并强化 TSV、四列顺序、priority normalization 和 date format；`SM01_EVAL_NEAR_001` 与 `SM01_EVAL_FAR_001` 检查近距离和远距离复用；三个 control case 分别检查 shortcut、普通 control 和 no-persistence。只有 learn source 到 eval opportunity 的链路与 extraction feedback 有关，cold/control 不应被当作 extraction 训练标签。

**可观测。** 通过。Source capture 能看到用户声明，extraction output 和持久化 artifact 可定位，eval 阶段的 retrieval/exposure、最终格式和 task outcome 可以关联。早期 batch 和 v9a 都独立得到 24 条 unresolved，说明 no-signal 结果可重复。

**可诊断。** 不通过。V9a 的 primary reasons 主要是 `use_not_bound_to_memory` 和 `outcome_not_attributable`。最终响应出现 TSV 不能证明模型使用了过去 memory；当前请求可能重新要求格式，eager injection 也不能单独证明使用。没有 useful、harmful 或 high-confidence missed chain。

此外，“eval task 需要 TSV preference”来自 SM01 family contract，而不是 pure process 自动发现；即使未来 resolver 产生 useful，也只能先算 benchmark audit evidence，不能直接算通用 online signal。

**可优化。** 不通过。当前日志不能支持可靠 prompt edit，SM01 应保留为 no-signal pilot，而不是为了生成 candidate 把 unresolved 改成负标签。

### 4.2 SM02：external advisory boundary

**Case 构成。** `SM02_COLD_001` 是 baseline；`SM02_LEARN_A_001` 与 `SM02_LEARN_B_001` 形成并强化 external advisory roster、内部 planning note 禁止外发和仅 employee 可接收的联合边界；`SM02_EVAL_NEAR_001` 与 `SM02_EVAL_FAR_001` 产生后续分享机会；三个 control case 检查 shortcut、普通 control 和 no-persistence。

**真实 extraction 行为。** Learn source 中的联合边界被模型拆成三条 independently persisted facts：一条描述 roster membership，一条描述不得向 external advisory roster 分享，一条描述只有 employees 可以接收。旧 detector 要求单条 fact 同时包含具体 entity 与 prohibition，因此三条 fact 都没有 semantic key。

**真实 future 行为。** Eval case 中存在 request-level retrieval、injection 和 `notes_list -> notes_get -> notes_share` 行为。旧 feedback 中的 `exposure_mode=not_exposed` 实际表示“没有找到可与 source semantic key 精确绑定的 artifact”，不等价于“Host 没有注入任何 memory”；raw memory evidence 显示相关持久化 artifact 曾进入 future execution。`notes_share success=false` 也可在 owner evidence 中看到，但尚未通过统一 tool call/result ProcessEvent 与具体 memory use 完成 exact join。

**为什么旧 missed 无效。** 旧 resolver 把“source detector 找到目标 key、单 fact detector 没找到 key、future outcome 失败”解释为 extraction miss，但真实 extraction 已经持久化了联合规则，只是投影器无法表示跨多个 fact 的语义等价。随后出现的 8/12 historical missed 是 representation/attribution failure，不是 extraction failure。

**可观测。** 历史批次的大部分证据已存在。当前 runtime 已补齐统一 process corpus 的 tool-level exact join；本节保留的旧缺口只说明为什么该批次的 historical missed 不能复用。

**可诊断。** 当前不通过。既不能把缺少单 fact key 解释为 extraction miss，也不能把部分 artifact 的出现解释为整个联合规则已使用。当前修复采取保守策略：存在未分类的非空 fact 时不得生成 high-confidence missed；在显式 set binding 完成前，也不得把联合 key 复制到每个 fact 来制造 useful。

SM02 opportunity 与 recipient correctness 仍由 family-specific roster parser 定义；pure process 只能看到分享对象、成功/失败和 memory lifecycle，不能在没有业务 policy schema 的情况下知道哪位 recipient 应被禁止。因此这批 case 能验证 audit contract，不能证明真实部署会自动发现同一语义规则。

**可优化。** 当前不通过。SM02 rerun 生成的 proposal 依赖 stale missed labels，即使 schema 和静态 safety 通过，也不能称为有效 N+1，不能进入 SM03 validation。

### 4.3 SM05：weak-trigger preference adoption

**Case 构成。** `SM05_COLD_001` 是 baseline；`SM05_LEARN_A_001`、`SM05_LEARN_B_001` 和 `SM05_UPDATE_001` 提供显式、弱触发和稳定化 preference source；`SM05_EVAL_NEAR_001` 与 `SM05_EVAL_FAR_001` 检查近距离和远距离复用；三个 control case 检查 shortcut、普通 control 和 no-persistence。

**真实 variation。** 不同 replicate 的 extraction output 有非空、部分 preference 和空输出变化，Admission 也出现 ADD/NONE status variation；这说明 lifecycle 中存在 output variation，但不是 learned intervention，也不自动形成质量标签。

**为什么旧 missed 无效。** 旧 source detector 扫描整个 projected trace，assistant 或 tool output 中出现 TSV、priority 和 date 也会被误当成用户声明的 durable preference。由此生成的 24 条 historical missed 不能证明 extraction 遗漏了用户信息。

**可观测。** 通过。Source、extraction variation、memory activity、future eval、tool activity 和 outcome 都可查看。

**可诊断。** 当前不通过。修复后 source eligibility 只读取 user-role messages；旧 artifact 不可在原地重标。`injected_not_used` 和 `not_exposed` 仍可作为 stage diagnosis，但当前没有 useful/harmful variation，也不能据此惩罚 extraction。

SM05 的 TSV、priority 和 date parser 同样是 benchmark audit contract。Pure process 能看到 extraction output variation 和后续执行，但不能凭 `family_id` 预先假定 future task 需要哪些 preference。

**可优化。** 当前不通过。必须使用 user-only source detection 重新生成 feedback；旧 batch 只能作为 projection bug 的反例和后续 validation substrate。

### 4.4 失败与不完整 case

Provider 503、empty think-only response、reflection 被错误当成新 extraction boundary、shared-cold duplicate audit 顺序错误和 incomplete usage 都属于 infrastructure diagnosis。它们证明 fail-closed 路径能阻止不完整证据进入 learner，但不提供 memory policy 的正负反馈。

## 5. 计数与独立性

`Primary feedback count` 不是独立语义样本量。一个 eval task 会针对多个 prior source record 和多个 request-level future observation 生成反馈，三个 replicate 又重复同一 benchmark task template；同一 source record 甚至可能在同一 task 的不同 retrieval boundary 上重复出现。因此 24 条 primary observation 不能写成 24 个独立发现。

历史日志尚未持久化显式 `logical_case_id`，所以本文对应的旧批次只报告 physical observations 和 family/stage structure，不伪造精确的 logical-case 数。当前 runtime 已持久化 logical case；正式重跑同时报告：

1. Physical feedback observations。
2. 按 source task、future task、future boundary 和 frozen policy 去重的 logical cases。
3. 同一 logical case 的 replicate consistency。

## 6. 六层 Policy 的 Process-Signal 结论

| layer | 真实日志可观测 | 真实 action variation | 当前可诊断 | 当前可优化 |
| --- | --- | --- | --- | --- |
| Trigger | 是 | 否，parent 基本固定 `RUN` | 否，缺少 skip/delay 对照 | 否，保持 fixed/shadow |
| Source selection | 是 | 否，parent 使用固定 completed-task projection | 否，source omission 会与 extraction 混淆 | 否，保持 fixed |
| Extraction | 是 | 有模型输出 variation，没有有效 candidate intervention | 暂不成立；SM02/SM05 historical missed 已失效，现有 resolver 仍依赖 family contract | 暂不成立，先补通用 opportunity/use evidence，再重跑 |
| Admission | 是 | 有 ADD/NONE status variation，不是 learned intervention | 不能与 extraction empty/conflict 分离 | 否，需要 matched case |
| Commit | 是 | 没有 scheduling intervention | 只能诊断 revision、receipt 和执行可靠性 | 否，只做 safety/shadow |
| Exposure | 是 | 有 RUN/SKIP，但主要来自 fixed behavior 和 memory existence | 可以定位 exposure failure，不能证明 learned policy | 否，需要 selective matched case |

Deterministic fixture 说明六层都可以被独立干预和回放，但真实 provider 日志目前只证明 lifecycle observability，没有证明任一层已经 optimization-ready。

## 7. 当前归因修复

本次审计对应的代码防线是：

1. Source semantic-key detection 只读取 user-role messages，禁止 assistant/tool output 反向制造 durable source。
2. 非空 extraction 中只要存在未分类 fact，在没有 deterministic equivalence proof 时就不能生成 high-confidence missed。
3. 外部传入的 stale `MissedExtractionEvidence` 也必须经过同一保守 gate，不能绕过 builder。
4. 联合规则的 semantic key 不复制到每个 persisted fact；否则只曝光其中一个成员也可能被错误判为完整规则已使用。
5. Useful/harmful 采用 source-specific artifact attribution：其他 source 写入的 memory 可以阻止误报 missed，但不能替当前 extraction 获得正负反馈；future binding 也不能为当前 source artifact 伪造 semantic key。
6. `never be shared` 和 `must never be shared` 等已观察到的自然表达进入规范化 detector，但仍受 user-only 和 exact contract scope 限制。

这些修复只用于 benchmark audit plane 撤销错误标签，不会把历史 artifact 原地改写成新标签，也不宣称已经完成通用 opportunity discovery、use attribution 或 set-level semantic binding。

历史 artifact 还受到 v2 content-free revocation registry 保护。当前登记的
五个身份使用 `scope=legacy_untyped`：artifact ID、schema version 和
digest 保持权威，但缺失的历史 provenance 以 `null` 表示而不作推断；对
这些身份的任意 typed lookup 都会被拒绝。后续新增撤销项必须使用
`scope=typed` 并通过 evidence plane/source 校验；registry 损坏或 identity
冲突时在 learner、validation 和 activation 之前 fail closed。

## 8. Process Signal 何时才算足够

一个真实 extraction case 至少需要满足：

```text
durable user source observed
+ parent extraction output observed
+ exact persisted artifact or explicit artifact-set binding
+ future opportunity derived from deployment-visible evidence or an application-owned policy schema
+ current input not confounded
+ retrieval/exposure state known
+ memory-specific use observed
+ tool/outcome exact join complete
+ diagnosis can be abstracted without benchmark values
```

应用可以提供公开、版本化的业务 policy schema，例如“此资源只允许 employee 接收”，但不能由 RSIMem learner 读取 benchmark hidden answer 或通过 family ID 查表获得。没有 application-owned schema 时，只能把 end-to-end behavior 作为 exploratory hypothesis，不能生成 high-confidence extraction label。

只有出现至少一组可信的 parent/candidate extraction variation，并且 persisted memory、future use 和 outcome 随 intervention 发生可重建变化，才允许生成并验证 N+1。单纯看到任务失败、没有检索、没有注入、最终格式错误、family parser 命中或 schema-valid proposal 都不够。

## 9. 后续执行边界（2026-09-01 checkpoint）

1. √ Benchmark adapter 的 family/stage metadata 已与 pure process corpus 分栏；依赖 family parser 的 label 标记为 `benchmark_audit_only`，不进入通用 learner。
2. √ 通用 opportunity/use contract 已落地：opportunity 来自当前输入、环境/资源状态、工具 schema 或 application-owned policy schema；use 绑定过去 artifact 对当前行为的实际贡献。
3. √ Set-level semantic binding 已落地；partial retrieval/exposure 保持 `unresolved`。
4. √ 真实 tool call/result 已投影为 content-free `ProcessEvent`，并带 exact closure、policy lineage 和 receipt join。
5. √ `logical_case_id` 已持久化，区分 physical observation、logical case 和 replicate consistency。
6. √ 已使用 user-only source detection、保守 missed gate、opportunity/use contract 和 set binding 重跑 SM02/SM05；新 census 仍无 extraction-owned signal，旧 corpus、旧 missed 和旧 candidate 不复用。
7. √ 在出现可信 extraction-owned chain 前保持 `STOP_NO_SIGNAL`；不生成 Extraction N+1、held-out validation 或 adaptive effect batch，其他层继续 shadow/validation-only。
8. □ 若后续 case 仍只有 unresolved、censored、not-exposed 或 injected-not-used，继续报告“当前 deployment process 不提供足够 extraction attribution”，不得把这些状态改成 harmful 或 missed。

## 10. 最终兜底结论

RSIMem 当前已经证明：真实 Agent execution 可以提供比最终 score 更细的 memory lifecycle evidence，并且 process evidence 与 benchmark audit contract 的交叉检查足以发现 attribution pipeline 自身的错误。SM02/SM05 的 false missed 正是通过 source、persisted memory、future exposure 和 tool evidence 被撤销的，这说明 process analysis 本身是有意义的。

RSIMem 当前尚未证明：仅依靠现有 pure process evidence 就能稳定诊断 extraction 并产生有效 policy update。SM01 是可信的 no-signal case；SM02 和 SM05 的旧 actionable signal 已失效；现有 strict resolver 仍依赖 benchmark-specific family contract；其他五层只有 observability 或 deterministic feasibility。

因此这项兜底验收的真实价值不是保证实验一定提升，而是建立一个不可跳过的研究边界：先证明 process 中存在可诊断、可泛化的优化信号，再进行 online policy adaptation；如果信号不足，就明确补哪段 observation contract，而不是依赖 grader、标准答案或 benchmark-specific rule 伪造自进化。

## Evidence Index

- [`extraction_stage2_clean_parent_20260901.md`](extraction_stage2_clean_parent_20260901.md)：authoritative fresh SM02/SM05 clean-parent rerun；两批均为 `STOP_NO_SIGNAL`。
- [`extraction_stage2_sm02_process_signal_final_20260830.md`](extraction_stage2_sm02_process_signal_final_20260830.md)：historical SM02 process-only rerun；9 个 logical case 均为 `observable_only`。
- [`extraction_stage2_sm05_process_signal_20260830.md`](extraction_stage2_sm05_process_signal_20260830.md)：historical SM05 process-only rerun；8 个 logical case 为 `observable_only`，2 个为 `censored`。
- [`extraction_stage3_s1_feedback_20260829.md`](extraction_stage3_s1_feedback_20260829.md)：早期 SM01 no-signal batch。
- [`extraction_stage3_sm01_feedback_v9a_20260829.md`](extraction_stage3_sm01_feedback_v9a_20260829.md)：当前 SM01 authoritative no-signal batch。
- [`extraction_stage3_sm02_feedback_v5_20260829.md`](extraction_stage3_sm02_feedback_v5_20260829.md)：历史 SM02 batch；其中 missed 有效性的旧结论已被本文撤销。
- [`extraction_stage3_sm05_optimizer_20260829.md`](extraction_stage3_sm05_optimizer_20260829.md)：历史 SM05 batch；其中 extraction-owned missed 的旧结论已被本文撤销。
- [`extraction_stage3_sm02_feedback_rerun_20260829.md`](extraction_stage3_sm02_feedback_rerun_20260829.md)：历史 SM02 rerun；其中 trusted candidate 的旧结论已被本文撤销。
- [`extraction_stage3_process_signal_census_20260829.md`](extraction_stage3_process_signal_census_20260829.md)：历史 process census；其中 Extraction optimization-ready 的旧结论已被本文撤销。
- [`policy_feasibility_baseline_20260829.md`](policy_feasibility_baseline_20260829.md)：deterministic six-layer feasibility，不是实际效果证据。
- [`extraction_stage3_sm01_feedback_attempts_20260829_v5_v8.md`](extraction_stage3_sm01_feedback_attempts_20260829_v5_v8.md)：SM01 不完整尝试及 runtime/provider diagnosis。
- [`extraction_stage3_sm02_process_pilot_20260829.md`](extraction_stage3_sm02_process_pilot_20260829.md)：SM02 不完整 pilot。
- [`extraction_stage3_sm02_provider_attempts_20260829_v3_v4.md`](extraction_stage3_sm02_provider_attempts_20260829_v3_v4.md)：SM02 provider/runtime diagnostics。
