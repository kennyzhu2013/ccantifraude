# 与 `kennyzhu2013/hangyekaAntiFraud` 的关系、对比与融合建议

对比对象：<https://github.com/kennyzhu2013/hangyekaAntiFraud>（描述「重庆行业卡质检」，
主分支 `main`，末次提交 `46be494 fix rules`）。下文简称 **HY**，本项目简称 **QC**。

本文结论中的量化数据来自 `scripts/cross_eval_hangyeka.py` 的交叉实测（两侧均只跑离线
确定性通道，不调用 LLM），可复现。

---

## 1. 两者是什么关系

### 1.1 同一个业务需求，同一份规范文档（已实证同源）

QC 的 `knowledge/spec.md` 与 HY 的 `重庆行业卡质检复核规范V1.1_20260518.md` 是**同一份
规范文档**的两个版本。按空白归一后做长句集合比对：

| 指标 | 数值 |
|---|---|
| 长句（>12 字）重合 | **577** 条 |
| QC `spec.md` 长句总数 | 645 |
| HY `V1.1` 长句总数 | 612 |

小节标题序列亦逐一对应（`目的` / `原则` / `复核规范` / `安全策略` / `违规场景识别` /
`引流第三方平台` / … / `违规场景涉诈判断与补充`）。HY 文档首行标题为
《中间号安全策略复核标注规范》，正是 QC `README.md` 中声明对齐的规范名称。

**违规场景类目体系完全一致**（8 类）：引流第三方平台、贷款相关、法律服务、
企业营销与招商服务、商品推销、商业地产、违规催收、其他。

### 1.2 同一个需求出发的两条技术路线（`思路.txt` 在此分叉）

两个仓库根目录都有 `思路.txt`，但内容不同——它们是同一需求下的两份**不同方案**：

| 仓库 | `思路.txt` 主题 |
|---|---|
| QC | 「持续演进、自我自治（Self-Evolving & Autonomous）的质检 Agent」，核心是反射机制 + 动态知识固化，按 `learn-claude-code` 哲学实现 |
| HY | 「重庆行业卡反诈质检策略 Agent 实施计划」，Python + FastAPI + 规则引擎 + LLM 单 Agent |

HY 的 `思路.txt` 里记录了需求主文档路径 `d:\AI\通话质检agent\重庆行业卡质检复核规范V1.1_20260518.md`
与思路目录 `d:\AI\通话质检agent\docs\ideas`，说明二者出自**同一工作目录下的同一批需求资料**。

### 1.3 输入 / 输出契约高度雷同

| 维度 | QC | HY |
|---|---|---|
| 输入 | ASR 转写文本，`left:` 主叫 / `right:` 被叫 | 同（另支持 `conversation[]` 结构化输入） |
| 是否违规 | `is_violation` | `is_violation` |
| 是否涉诈 | `is_fraud` | `is_fraud` |
| 风险等级 | `risk_level`（合规/低/中/高） | `risk_level`（正常/低/中/高） |
| 类目 | `scene_category` / `scene_subtype` | `violation_type`（+ `rule_hits[].subtype`） |
| 说明 | `explanation` | `explanation` + `summary` |
| 证据 | `evidence_quotes` | `evidence` |
| 置信度 | `confidence` | `confidence` |
| 人工复核 | `review_flags`（信号列表） | `needs_human_review` + `review_reason`（枚举） |

判定原则也一致：**就高不就低**；**涉诈单独标注且一律高风险**；说明格式为
**「违规/涉诈类型 + 简要分析」**。

### 1.4 同样的 LLM 接入方式与实测模型

两侧都走 OpenAI 兼容协议（`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`），
且都用 DeepSeek 系列做过真实模型评测（QC 用 `deepseek-v4-pro`/`deepseek-v4-flash`，
HY 的 `tests/fixtures/e2e_eval/e2e_real_report.json` 记录模型为 `deepseek-v4-flash`）。

### 1.5 规范版本存在代差，且原始件分别在两边

| | QC | HY |
|---|---|---|
| 规范 Markdown | `knowledge/spec.md`（V1.1 **+ 对客反诈新规范增补**，34 小节，含各场景「典型话术举例和分析」） | `重庆行业卡质检复核规范V1.1_20260518.md`（23 小节，仅 V1.1） |
| 规范原始件 | **无** | `重庆行业卡对客反诈复核规范.xlsx`（Excel 原始件） |
| 涉诈场景数 | **9** | 8 |

QC 的 `spec.md` 多出的小节即新规范增补的三个涉诈场景（`贷款相关-ab贷`、
`引导贷款用户添加第三方微信`、`套路运诈骗`）与全部「典型话术举例和分析」。
HY 虽然规范 md 停留在 V1.1，但其 `chongqing_rules.yaml` 已按 xlsx 补齐了这三个场景的规则。

**唯一的类目覆盖缺口**：`验证码/短信转发诈骗` 仅 QC 有（`knowledge/rules.json` 的
`fraud_scenarios`），HY 规则库中 `验证码` 零命中。QC 的 `README.md` 把它标为
「高危场景，已作为最高优先级规则修复」。

---

## 2. 架构分歧全景

两者的分歧集中在**确定性层怎么实现**与**交付形态**两条轴上。

| 维度 | QC（本项目） | HY |
|---|---|---|
| 设计哲学 | harness：`Agent = LLM + Tools + Knowledge + Retrieval + Memory`，智能在模型 | 规则确定性优先 + LLM 裁决，硬约束在代码 |
| 交付形态 | CLI + Python API（`scripts/inspect_text.py` / `batch_eval.py` / `evolve.py`），**无服务层** | FastAPI REST：`POST /api/v1/audit/transcript`、`GET /healthz` |
| 确定性层 | 关键词子串列表 + 自由文本 `judgment_method`/`risk_rules`/`disambiguation`（供 LLM 消费）；`qc_agent/heuristic.py` 另有硬编码正则 | **可执行 YAML DSL**：`all`/`any`/`none` + `speaker` + `window`，45 条规则、16 类目 |
| 合规豁免 | `compliant_subtypes` + `business_decision_table`（**纯文本，仅注入 prompt**） | **8 条 `risk_level: 合规` 可执行规则**，`merge_hits` 将其排除出 top 选择 |
| 说话人约束 | 仅在 prompt 里说明 `left`/`right` 角色，匹配与校验都对全文 | 子句级 `speaker: agent\|customer` |
| 时序约束 | 无 | `window: N` 轮次滑动窗口 |
| 就高不就低 | LLM 负责 + `agent._finalize` 确定性兜底（涉诈→高风险→违规） | `rule_engine.merge_hits`（规则内）+ `audit_pipeline._fuse`（规则 vs LLM）**双层代码强制** |
| 检索 | 字符 n-gram TF-IDF（`qc_agent/retrieval.py`）+ 人工判例 kNN RAG（`case_store.py`，带 `exclude_id` 防自泄漏） | **无检索**；`few_shots.json` 按命中 category 静态筛选，取前 3 条 |
| 知识按需加载 | 17 个技能文件渐进披露（`knowledge/skills/*.md`），触发词 + TF-IDF top-k 路由 | 注入命中 category 下全部分支的 `decision_notes` |
| 自动演进 | `qc_agent/reflect.py` + `scripts/evolve.py`：候选词暂存区、`--promote`/`--discard`、A/B/C 冲突分桶、错题本回写技能文件 | 无；靠人工改 YAML + 补黄金样本 |
| 幻觉防护 | `qc_agent/verify.py` **事后**校验证据是否真在原文（失败则升级复核） | `llm_agent._sanity_check` **事前**校验 + 带错误提示重试（最多 3 次） |
| 复核分流 | `review_flags`：硬信号（证据未命中/启发式涉诈冲突/JSON 修复）+ 软信号（kNN 冲突/涉诈路由冲突/低置信）+ 自一致性采样 | `review_reason` 四值枚举：`rule_hit_fraud`/`rule_llm_conflict`/`low_confidence`/`anomaly_only` |
| 不确定性度量 | **明确放弃自报置信度**（实测饱和在 0.95–1.0），改用独立视角旁证 + K=2 重采样一致性 | `confidence < 0.6` 阈值（`LOW_CONF_THRESHOLD`） |
| 异常检测 | 无（`qc_agent/dedup.py` 的 SimHash 只用于降本去重） | **PyOD 双通道设计**：`AnomalyResult` schema 与 prompt 纪律已预留，`ANOMALY_ENABLED=false`，实现步骤 7–10 未完成 |
| 成本控制 | 结果缓存（内容哈希 + 磁盘持久化）、SimHash 近重复去重、超长截断、fast/tool 双模 | 无缓存/去重；仅长对话轮次裁剪（>80 轮保留头 15 + 尾 10 + 命中±3） |
| 依赖 | 零必需依赖（`openai` 可选） | fastapi/uvicorn/pydantic/PyYAML/openai/httpx |
| 测试 | 85 项单测，全部离线可跑 | 70 passed + 1 skipped：44 条黄金样本参数化 + FakeLLM 融合矩阵 + 100 条标注集 e2e |
| CI | 无 | 无 |

---

## 3. 交叉实测：两个确定性通道确实互补

复现命令（两侧均不调 LLM）：

```bash
git clone https://github.com/kennyzhu2013/hangyekaAntiFraud /tmp/hangyekaAntiFraud
python3 scripts/cross_eval_hangyeka.py --repo /tmp/hangyekaAntiFraud
```

### 方向 A：HY 规则引擎跑 QC 的 `data/eval_fresh_cases.csv`（28 条手写回归集）

| 指标 | HY 规则引擎 | QC 启发式 | 并集 |
|---|---|---|---|
| 违规召回（21 正例） | 14 | 16 | **17** |
| 涉诈召回（9 正例） | 6 | 8 | 8 |
| 合规负例误报（7 负例） | 0 | 0 | 0 |
| 类目正确（可评 21） | 13 | 16 | — |

### 方向 B：QC 启发式跑 HY 的 `tests/fixtures/cases.json`（44 条黄金样本）

| 指标 | HY 规则引擎 | QC 启发式 | 并集 |
|---|---|---|---|
| 违规召回（33 正例） | 33 | 27 | 33 |
| 涉诈召回（13 正例） | 13 | 11 | 13 |
| **合规负例误报（11 负例）** | **0** | **3** | 3 |
| 类目正确（可评 33） | 33 | 25 | — |

> **口径说明（重要）**：两个测试集各自是其所属项目的「主场」——`fresh-28` 是按 QC 的
> `spec.md`/`rules.json` 手写的回归集，`golden-44` 是 HY 每次改规则必须全绿的回归门禁。
> 因此**绝对分数不可直接用于优劣排序**，有意义的信号是下面的具体失效模式，以及方向 A 中
> 「并集召回 17 > 单侧 16/14」所体现的互补性。

### 3.1 QC 侧暴露的真实缺陷：合规豁免只写在文本里，确定性通道兜不住

方向 B 中 QC 启发式的 3 条误报，全部落在规范明确写为**合规**的场景上：

| 黄金样本 | 话术要点 | 规范口径 | QC 启发式输出 | HY 输出 |
|---|---|---|---|---|
| `taobao_instant_compliant` | 给商家开通淘宝闪购联盟功能 | 合规（`spec.md` 淘宝闪购联盟功能开启） | 低风险/企业营销与招商服务 | 正常 |
| `operator_compliant` | 运营商客服，宽带续费、携号转网 | 合规（`spec.md` 运营商业务推销） | 低风险/商品推销 | 正常 |
| `shop_rent_compliant` | 商铺包租需签合同并付推广费 | 合规（`spec.md` 商铺包租提前收取推广费） | 中风险/商业地产 | 正常 |

关键在于：**QC 已经知道这三条口径**，而且写了三遍——

1. `knowledge/spec.md`：「淘宝闪购联盟功能开启（合规）」「运营商业务推销（合规）」「商铺包租提前收取推广费（合规）」；
2. `knowledge/rules.json` 的 `business_decision_table` 三行 `risk_level: 合规`；
3. `knowledge/rules.json` 的 `compliant_subtypes`（企业营销、商业地产）。

但这三处**全是自由文本，只被渲染进 prompt**（`knowledge_base.slim_brief()` /
`rules_brief()`），`qc_agent/heuristic.py` 完全不消费它们——该文件里唯一的合规豁免是
硬编码的 `_COMPLIANT_WECHAT` 正则。于是只要 LLM 不可用（或触发启发式兜底），
这些豁免就全部失效。

HY 把同样三条口径写成了可执行规则 `CQ-MKT-TAOBAO-INSTANT-001`、
`CQ-RETAIL-OPERATOR-001`、`CQ-ESTATE-SHOP-RENT-COMPLIANT-001`（`risk_level: 合规`,
`priority: 10`），`merge_hits` 只从 `risk_level != "合规"` 的命中里挑 top，
因此负例误报为 0/11。

### 3.2 HY 侧暴露的缺口：类目覆盖与话术变体

| 样本 | 人工标签 | HY 规则引擎 | QC 启发式 |
|---|---|---|---|
| `fresh01` | 机票退、改签诈骗（涉诈） | 高风险 / **引流第三方平台** / 非涉诈 | 高风险 / 机票退、改签诈骗 / 涉诈 |
| `fresh03` | 证券投资类，引导投资（涉诈） | **正常（漏判）** | 高风险 / 证券投资类 / 涉诈 |
| `fresh09` | 贷款相关，引导用户平台操作提现 | **正常（漏判）** | 低风险 / 贷款相关 |
| `fresh16` | 商品推销，赠送 POS 机 | **正常（漏判）** | 低风险 / 商品推销 |

`fresh01` 是 `CQ-FRAUD-FLIGHT-001` 的 `all` 子句未全部命中而下坠到引流规则，
`fresh03`（工作室每日推票 + 加老师微信 + 体验群）是 `CQ-FRAUD-STOCK-001` 未覆盖的变体。
这类「同类目内的话术变体」正是 QC 的关键词广覆盖 + 判例 RAG 的强项。

### 3.3 反向：HY 也能补 QC 的漏判

`fresh20`（冒充市中级法院诉讼服务中心催收）：HY 的 `CQ-COLLECT-IMPERSONATE-001`
判高风险/违规催收（正确），QC 启发式判**合规（漏判）**。方向 B 中 QC 启发式还漏判了
`ab_loan_high_risk`、`debt_optimization_high_risk`、`stock_daily_yield_fraud`、
`loan_housefund_medium_risk`、`legal_refund_high_risk`、`prefee_loan_low_risk` 共 6 条，
而这些在 HY 都是靠 `all` 组合条件稳定命中的。

---

## 4. QC 可以向 HY 借鉴什么

按性价比排序。

### P0 — 把合规豁免变成可执行的否决层

**问题**：见 §3.1，实测 3/11 负例误报。合规口径散落在 `spec.md`/`business_decision_table`/
`compliant_subtypes` 三处自由文本中，只有 LLM 读得懂。

**建议**：给 `rules.json` 增加机器可判定的豁免表（或让 `heuristic.py` 直接消费
`business_decision_table` 中 `risk_level == "合规"` 的行 + `compliant_subtypes`），
在启发式给出违规结论前做一次**否决检查**。这不需要引入完整 DSL——最小改动是让
决策表的合规行成为启发式的一等公民，顺带消除「同一口径写三遍」的一致性风险。

进一步（可选）：把这层否决也接进 `_finalize`，作为 LLM 结论的合规兜底，与现有
「涉诈→高风险」的确定性升级对称（当前只有向上升级，没有向下豁免）。

### P1 — `speaker` 作用域约束

QC 的 `README.md` 记录过一个真实误判：「引流第三方平台『我加你微信』（外呼方执行=合规）
被误判为『你加我微信』（用户执行=高风险）方向」，修复方式是加一条自由文本 `disambiguation`。
HY 对同一问题的解法是**确定性的**——`引流第三方平台` 被拆成 4 条 speaker 作用域规则：

| 规则 | 风险 |
|---|---|
| `CQ-DIVERT-WECHAT-USER-INITIATED`（`speaker: agent` 引导用户操作 + `none` 排除「我加你」） | 高风险 |
| `CQ-DIVERT-WECHAT-AGENT-INITIATED`（外呼方主动加） | 合规 |
| `CQ-DIVERT-WECHAT-THIRDPARTY-LOAN`（第三方加 + 贷款场景） | 低风险 |
| `CQ-DIVERT-WECHAT-THIRDPARTY-OTHER`（第三方加 + 非贷款） | 合规 |

QC 目前的 `heuristic.py`、`verify.py`、关键词匹配都对**全文**做子串匹配，不区分说话人；
`left:`/`right:` 只在 prompt 里做角色说明。至少在证据校验与启发式两处引入说话人切分，
是低成本高收益的改动。

### P2 — 测试纪律：正反例 + FakeLLM 融合矩阵

HY 的两条做法值得直接搬：

1. **黄金样本带反例**：`tests/fixtures/cases.json` 每条含 `expect_rules` 与
   **`expect_not_rules`**，并写明「每条规则合入必须随附正反测试用例」。QC 的 85 项单测
   覆盖很广，但缺少「这条话术**不应**命中某类目」的显式负例断言。
2. **FakeLLM 融合矩阵**：`RuleMirrorFakeLLM` 镜像规则候选结论，用于**只测融合/后处理路径**
   而不测模型质量（`test_pipeline.py` 11 项）。QC 的 `_finalize` + 两级门控是纯确定性逻辑，
   完全可以用同样手法覆盖「LLM 判正常但硬信号触发」等分支组合，不必依赖真实模型。

### P3 — HTTP 服务层

QC 只有 CLI 与 Python API。HY 的 `app/main.py` + `app/api/audit.py` 是一份很薄的
FastAPI 外壳（启动时加载规则库并 fail-fast，`/healthz` + 单条质检接口），
若 QC 要接入生产调用方，可直接照搬形态，把 `QcAgent.inspect()` 包一层即可。

### P4 — `review_reason` 枚举与 `review_flags` 互补

QC 的 `review_flags` 是**信号集合**（诊断用，`scripts/analyze_flags.py` 已能统计触发率），
HY 的 `review_reason` 是**单值分流原因**（队列路由用）。两者不冲突：可在
`InspectionResult` 上增加一个由 flags 归约出的主原因，便于人工复核队列按原因分组与
计量（例如「规则冲突」优先于「低置信」）。

### P5 — ASR 错字归一表

HY 的 `app/knowledge/normalization.yaml` 是一张显式的同义/错字表（微信←威信/薇信、
征信←争信、二维码←二惟码/二维玛、服务号←公众号、企业微信←企微/工作微信），
配合 **raw/norm 双视图**：`norm_turns` 用于匹配，`raw_turns` 用于展示证据，index 严格对齐。
QC 的 `retrieval._normalize()` 只去标点空白，不做 ASR 错字修正——补一张表可直接提升
关键词与证据校验的命中率，且成本极低。

### P6 — `sanity_check` 前置修复

QC 的 `verify.py` 是**事后**校验（证据未命中 → 升级 tool loop 复核，代价是一次完整重跑）。
HY 的 `_sanity_check` 在拿到 LLM 输出的当场校验业务硬约束（涉诈必违规必高风险、
违规必须有证据且证据必须是原文子串、`summary` 必须是「类型：分析」格式），
不通过就把错误信息与修复提示追加进 messages 重试。两种策略可以叠加：
先就地重试一次，仍失败再走升级，能省下相当一部分升级开销。

### P7 — PyOD 异常检测通道

QC 的 `README.md` TODO 第 1 条希望「收集网上诈骗案例喂给 AI 发现新规则」。HY 的 V2 文档
（`思路V2_引入PyOD的质检策略Agent实施方案.md` 等三份）给出了一个更工程化的答案：
用离群检测发现**未知话术**，且严格限定其权限——异常 ≠ 违规，只允许单向置 `needs_human_review`、
小幅加 `confidence`、给复核队列排序，**禁止**改 `violation_type`/`risk_level`。
其中 `anomaly_only` 路径（规则未命中 + LLM 判正常 + 高异常分 → 结论仍正常但送人工复核）
正好补上 QC 现有六个信号都覆盖不到的一类漏判。注意 HY 这部分**只有设计没有实现**
（步骤 7–10 未做，`requirements.txt` 里没有 pyod/sklearn），可借鉴的是设计与权限纪律。

---

## 5. HY 可以向 QC 借鉴什么

### P0 — 规范升级与类目补齐

- 补 `验证码/短信转发诈骗` 涉诈类目（HY 规则库中 `验证码` 零命中，QC 视其为最高优先级高危场景）；
- 把规范 md 同步到 QC `spec.md` 的口径（含各场景「典型话术举例和分析」），
  这些典型话术正好可直接转成黄金样本；
- QC `rules.json` v1.4 记录的口径修订（ab贷升级为涉诈、引导贷款用户加第三方微信、
  套路运、芝麻信用分→手机租赁套路贷、帮退律所费用收一半=高风险等）可用于校对 YAML 规则。

### P1 — 判例 RAG 取代静态 few-shot

HY 的 `few_shots.json` 是 10 条静态样例，按命中 category 筛选后取前 3 条。QC 的
`case_store.py` 从人工标注 CSV 建 TF-IDF 索引做 kNN 召回，并用 `exclude_id` 排除自身
防评估泄漏。方向 A 中 HY 漏判的 `fresh03`/`fresh09`/`fresh16` 都属「同类目话术变体」，
正是判例召回最能补的一类。

### P2 — 成本控制

HY 无缓存、无去重。QC 的结果缓存（内容哈希 + 知识指纹 namespace + 磁盘持久化）在 827 条
语料上做到「重跑约 1s」；SimHash 近重复聚类每簇只调一次 LLM。HY 的 `e2e_real_report.json`
显示 100 条 real 评测耗时 182.56s，接一层缓存即可让回归评测近乎免费。

### P3 — 标签治理与自动演进

QC 的 `scripts/evolve.py` 默认**只导出冲突不改规则**，把冲突分成
A（建议回标为合规）/B（真违规）/C（待人工复核）三桶下发回标；演进出的高危词先进
`candidate_keywords` 暂存区，经 `--promote` 人工审核才进生产。HY 目前用
`acceptable_categories` 软匹配来容忍标签噪声，但没有治理闭环——在人工标签有噪声的
真实数据上（QC 实测 827 条中 93 条冲突、其中约 70 条是人工误标），这套工作流是刚需。

### P4 — 零依赖启发式兜底

HY 在 LLM 连续失败后走 `_fallback_verdict()`：`is_violation=false`、`confidence=0`、
送人工复核——即**放弃判定**。QC 的 `HeuristicInspector` 在无 LLM 时仍能给出完整结论
（fresh-28 上违规召回 0.889）。HY 已有可执行规则引擎，其实离「规则直出结论」只差一层
薄封装（等价于本文交叉实测里的 HY runner），比返回空结论更有价值。

### P5 — 用旁证替代置信度阈值

HY 用 `confidence < 0.6` 触发复核。QC 明确做过三条路的实验并给出结论：自报置信度
饱和在 0.95–1.0（灰区话术照样给 0.95）、终答 token logprob 被推理模型坍缩、
K=3 重采样一致率可用但需选择性使用。因此改用**独立视角旁证**（证据校验失败、
启发式与 LLM 冲突、kNN 判例冲突、涉诈路由冲突）分硬/软两级门控。这个结论对 HY 直接适用——
HY 的 `review_rate` 在 fake/real 上分别是 75%/77%，说明当前阈值把绝大多数样本都送了人工，
分流几乎没有筛选作用。

### P6 — 技能渐进披露降 token

QC 把 17 个类目各做成一个技能文件，system prompt 只常驻技能目录（每技能一行触发词 +
描述）与全局不变量，完整细则按 top-k 路由随待检文本注入，prompt 缩减约 37% 且字节稳定
利于前缀缓存。HY 注入命中 category 下的**全部**分支 `decision_notes`，规则库继续膨胀后
这块会线性增长。

---

## 6. 若要真正融合成一个系统

两者的强项几乎不重叠，目标形态是：**HY 的确定性骨架 + QC 的知识/检索/演进/门控**。

```
                    HTTP 层（HY app/main.py + api/audit.py）
                                 │
                    预处理：说话人切分 + ASR 错字归一（HY preprocess + normalization.yaml）
                                 │  raw/norm 双视图
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
  可执行规则层                 检索层                  异常通道（HY V2 设计）
  HY rule_engine DSL      QC case_store kNN            PyOD，仅可置
  + 合规豁免规则          + QC skills top-k 路由        needs_human_review
  + merge_hits 就高        + spec 小节召回
        └────────────────────────┼────────────────────────┘
                                 ▼
                    LLM 裁决（QC prompts 技能注入 + HY sanity_check 带错重试）
                                 ▼
                    融合硬约束（HY _fuse + QC _finalize 归一 + 合规否决）
                                 ▼
                    质量门控（QC 硬/软信号 + 自一致性 → review_flags/review_reason）
                                 ▼
                    演进闭环（QC reflect/evolve：冲突分桶 → 候选词暂存 → 人工晋升 → 回写规则/技能）
```

分阶段落地（按依赖顺序，不含工期估算）：

1. **对齐知识源**。以 QC `spec.md` + `rules.json` v1.4 为规范事实源，把 HY 的 45 条 YAML 规则
   逐条对照校验，补 `验证码/短信转发诈骗`；把 QC 的 `business_decision_table` 合规行与
   `compliant_subtypes` 编译成 HY 形态的 `risk_level: 合规` 规则。改动面：两边 `knowledge/`。
2. **在 QC 内引入确定性豁免与 speaker 约束**（即 §4 的 P0/P1）。这一步不依赖 HY 代码，
   可独立完成，直接消掉实测中的 3/11 负例误报。改动面：`qc_agent/heuristic.py`、
   `knowledge/rules.json`、可选 `qc_agent/verify.py`。
3. **合流确定性层**。把 HY 的 `rule_engine` + `evidence_extractor` 作为 QC 的一个新工具/前置
   通道接入，规则候选结论作为 prompt 上下文与 `_finalize` 的兜底下界（当前 QC 只有
   「涉诈→高风险」的向上兜底，缺少「规则已判高风险但 LLM 判正常」的向上兜底，
   这正是 HY `_fuse` 的 `rule_llm_conflict` 分支）。改动面：新增依赖 PyYAML，
   `qc_agent/agent.py`、`qc_agent/tools.py`。
4. **合流测试**。QC 85 项 + HY 44 条黄金样本（含 `expect_not_rules` 负例）+ FakeLLM 融合矩阵，
   并给两边补 CI（目前都没有 `.github/`）。
5. **可选：异常通道**。按 HY V2 文档实现 PyOD 通道（步骤 7–10），严格遵守
   「异常只影响复核与排序，不影响类型与等级」的权限纪律。

如果不打算合并代码库，最低成本的互利做法是：**共享 `knowledge/` 与测试语料**——
规范文档、决策表、黄金样本、标注语料都是与架构无关的资产，而两边恰好各持有对方缺的部分
（HY 有 xlsx 原始件与 44 条带反例的黄金样本，QC 有新规范口径、17 个技能文件与冲突治理产物）。

---

## 7. 两者共同的短板

1. **ASR 数字/谐音混淆未归一**。HY 的示例 transcript 里就有「幺零九八」这类微信号读法，
   但 `normalization.yaml` 只处理错字，不处理中文数字；QC 侧也没有。两边都靠正则容忍
   （如 HY 砍头息规则写成 `'[一1壹]万[^。]{0,12}[八8]千'`），不是系统解法。
2. **语义检索缺失**。QC 是纯字符 n-gram TF-IDF（`README.md` TODO 第 3 条已列为待办），
   HY 无检索。同义改写话术（`fresh03` 即典型）对字面检索不友好。
3. **无 CI**。两边都没有 `.github/`，回归全靠本地手动执行。
4. **人工标签噪声**只有部分对策。QC 有治理工具但仍未完成回标；HY 用
   `acceptable_categories` 软匹配掩盖，指标可能偏乐观。

---

## 附：实测复现

```bash
# 1) 取对比仓库
git clone https://github.com/kennyzhu2013/hangyekaAntiFraud /tmp/hangyekaAntiFraud

# 2) HY 侧依赖（QC 侧零依赖）
python3 -m venv /tmp/hy-venv && /tmp/hy-venv/bin/pip install -r /tmp/hangyekaAntiFraud/requirements.txt

# 3) 交叉实测
/tmp/hy-venv/bin/python scripts/cross_eval_hangyeka.py \
    --repo /tmp/hangyekaAntiFraud --out /tmp/cross_eval.json

# 4) 各自基线
python3 -m unittest tests.test_qc_agent                      # QC: 85 passed
cd /tmp/hangyekaAntiFraud && /tmp/hy-venv/bin/python -m pytest tests/ -q   # HY: 70 passed, 1 skipped
```
