# Six-layer policy feasibility deterministic baseline

日期：2026-08-29

本文记录第三阶段 3A/3B 的第一份 deterministic/shadow feasibility evidence。它验证的是 decision 是否可观察、可控制、可回放，以及是否存在可归因的 process/outcome 反馈；它不是 PAST-Bench aggregate uplift，也不是在线 adaptive policy 实验。

表中的 `useful/harmful/missed` 链使用 fixture-local 的不透明 evidence IDs 来演示 contract 和统计边界；它们不是从真实 Hermes deployment 或 SM01 strict resolver 读取的标签，不能直接进入正式 optimizer corpus 或效果结论。真实 corpus 接入后必须用已注册的 opportunity/use/outcome contract 替换这些 fixture 链，并保留 unresolved/censored。

## Fixture

Fixture 使用一个 completed Hermes-style snapshot，包含一个 durable preference（TSV 输出偏好）和一个 temporary formatting request。`DeterministicPolicyReplay` 在相同 event、snapshot revision、backend descriptor 和 lineage 下生成 parent/candidate。candidate artifact 只声明一个目标 policy layer；replay audit 必须先通过，才允许构造 `LayerIntervention`。

覆盖的 intervention：

- Trigger：parent `RUN`，candidate 关闭 `task_completed` 后 `SKIP`。
- Source selection：parent 选择整个 completed task，candidate 只选择 durable segment。
- Extraction：parent/candidate 改变 candidate fact set。
- Admission：parent `ADD`，candidate 使用显式 target 和 revision 执行 `UPDATE`。
- Commit：只改变 mutation ID，保持 formation decision 不变。
- Exposure：只改变 selected artifact 集合。

每个 case 都保留 `event -> decision -> receipt/lineage` 的 content-free identity。用例还覆盖：缺少 useful/missed 证据链节点时自动降级为 `unresolved`，candidate 不改变目标层时拒绝，重复 case ID 时拒绝，以及 parent/candidate replay audit 失败时拒绝。

当前 feasibility evidence ledger schema 为 v2；此前只保存 ID 的 v1 记录不会被静默迁移，而是明确 fail closed。

`build_extraction_feedback_fixture()` 提供了同一 past snapshot 的两个确定性
delayed outcome：`MISSED` 分支让 Policy N 省略 durable TSV preference，未来
任务需要该 preference 但没有 memory exposure；`USEFUL` 分支保留该 fact、通过
eager exposure 注入并记录成功的 `notes_share`。两个分支都通过已注册的
SM01 opportunity/use/outcome contract 和 operation join 生成 source、set、fact
以及 attribution evidence，随后可由
`build_extraction_feedback_interventions()` 投影为 content-free feasibility
case。missed chain 使用独立的 logical absence ID，避免与 outcome operation
重复占用同一 hypothesis evidence identity。

`ProcessFeedback` 进一步将 intervention 绑定到 event、source revision、目标层 parent/candidate decision、执行 receipt 集合和 before/after output digest；`JsonFeasibilityEvidenceLedger` 以原子替换和文件锁持久化该 identity。重启读取、重复写入、损坏记录和冲突记录均 fail closed。ledger 只保存 ID、digest、状态和 reason，不保存 snapshot 或 memory 正文。

每个有 process signal 的 case 还会生成 `PolicyHypothesis`：它把 past feedback IDs、parent artifact、candidate artifact 和唯一 target layer 固定成稳定的 N+1 proposal identity。hypothesis 不能引用 intervention 外部的 feedback，也不能跨层或复用相同 artifact。

为避免 opaque fixture ID 被误接入正式训练，`feedback_chain_from_extraction_example()` 只接受真实 `ExtractionFeedbackExample` 的 primary extraction-set useful/missed 记录，并要求其 operation identity 完整；fact-level、harmful、unresolved 和 censored 记录返回空链，继续留在诊断桶。

`LayerIntervention.from_extraction_feedback()` 将同一规则用于构造 extraction intervention：只有 primary extraction-set example 可以进入，fact-level example 会被拒绝；不完整 useful/missed 链会自动降级为 unresolved。

`build_extraction_feedback_interventions()` 提供批量入口，直接消费已有 feedback dataset 的 examples，跳过非 primary source/fact projection，并对 primary example ID 去重；因此后续 Stage 2E corpus 可以直接复用这条链。

## Census

| layer | cases | signal coverage | action variation | useful | harmful | missed | unresolved | censored | complete feedback | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| trigger | 1 | 1/1 | 1/1 | 0 | 0 | 0 | 1 | 0 | 0 | validation-only |
| source_selection | 1 | 1/1 | 1/1 | 0 | 0 | 1 | 0 | 0 | 1 | validation-only |
| extraction | 2 | 2/2 | 2/2 | 1 | 0 | 1 | 0 | 0 | 2 | optimization-ready |
| admission | 1 | 1/1 | 1/1 | 0 | 1 | 0 | 0 | 0 | 0 | validation-only |
| commit | 1 | 1/1 | 1/1 | 1 | 0 | 0 | 0 | 0 | 1 | validation-only |
| exposure | 1 | 1/1 | 1/1 | 1 | 0 | 0 | 0 | 0 | 1 | validation-only |

这里的 `optimization-ready` 只表示该层已有 process signal、目标层 action variation、至少两类 resolved outcome，以及完整可回放 evidence chain。它不表示该层已经在真实任务上提升分数。其余层保留为 `validation-only`，因为当前 fixture 没有足够的 outcome variation；没有将 unknown 或 unresolved 当成 negative。

每层报告同时保留 U/H/M、unresolved、censored 原始计数以及 `resolvedUsefulRate`；resolved denominator 为 U+H，分母为零时输出 unknown（JSON `null`）。

`LayerFeasibilityCensus` 还记录 process-signal、action/outcome variation、
ambiguity 和失败 reason。coverage、empty、missed 等 extraction-specific
指标不从 policy fixture 猜测，而由 offline validation 的
`OfflineRatioEvidence` 提供 numerator/denominator/unknown。raw calls/tokens/
retry/latency/storage 继续使用 lifecycle `RawResourceUsage`，injection 与
recovery 使用独立 ledger events；这些维度不被压成一个混合成本分数。

## 验证

```text
PYTHONPATH=src pytest -q tests/test_policy_feasibility.py tests/test_policy_replay.py
32 passed

.venv/bin/pytest -q tests
645 passed

cd benchmarks/past-bench && ../../.venv/bin/pytest -q
397 passed, 2 skipped

.venv/bin/python -m compileall -q src
.venv/bin/python -m pip check
git diff --check
```

完整回归已在仓库 `.venv` 项目实验环境中完成。系统 Python 3.12 环境缺少 Hermes/PAST-Bench 运行时依赖，不能用系统解释器复现这些结果；实验命令应使用 `.venv`，PAST-Bench 测试应从其目录运行。

实现入口：`src/rsimem/memory/policy_feasibility.py`；测试 fixture：`tests/test_policy_feasibility.py`。

## Executable runner

同一 deterministic fixture 现在可直接运行，不依赖 pytest 私有 helper：

```text
.venv/bin/python -m rsimem.memory.policy_feasibility_fixture \\
  --output /tmp/policy-feasibility.json \\
  --evidence /tmp/policy-feasibility.jsonl
```

runner 会生成 7 个 case、写入 content-free evidence ledger，并在重复运行时复用相同 record IDs；它仍然只产生 feasibility/shadow 证据，不调用 provider。

Provider connectivity 另做过一次单请求 smoke（不属于 feasibility 或
PAST-Bench replicate）；仅 raw usage 保存在 ignored
`outputs/provider_connectivity_smoke_20260829.json`，没有保留 prompt/response
正文。

Stage 2E 的 `rsimem.extraction_proposal` 现在也会写出
`feasibility-hypothesis.json`。该文件只包含 optimizer result/request、parent
and candidate artifact identity、training corpus digest、actionable evidence
IDs 和 reason codes；`NO_PROPOSAL` 明确没有 candidate。proposal 写入前会
通过 `project_optimizer_result()` 的 parent/corpus/ownership/schema gate，因而
不会把一个未验证的 optimizer-result.json 当成 N+1 intervention。

`tests/test_policy_feasibility_nplus1.py` 进一步把 proposal 接到
`ExtractionCandidateStaticValidator`、deterministic extraction suite、offline
quality gate 和 `JsonExtractionPolicyStore` 的 restart reload。验证结果是
`ACCEPTED_FOR_MATCHED_TRIAL`，candidate 仍保持 `PROPOSAL` 状态，未被本阶段
提前切换为 ACTIVE；真实 future-task intervention 与 matched effect 仍待后续
阶段完成。该测试还用 `FeasibilityInterventionPath` 将 projection 与
target-layer replay fingerprint 写入独立 JSONL ledger，重启读取和重复写入
均保持幂等；ledger 只保存 identity、状态和 reason，不保存 source 或 memory
正文。同一集成测试还覆盖了 `NO_PROPOSAL`、rejected candidate、restart 后拒绝
再次激活，以及 ACTIVE candidate 的 operator rollback；这些路径只验证 policy
artifact state machine 和 evidence identity，不把它们误报为真实 memory mutation
recovery。
