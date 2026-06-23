# 重庆行业卡反诈语音质检 Agent

基于 [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) 的 **harness 工程** 思路实现的反诈质检 Agent。

> **核心理念（来自 learn-claude-code）**：`Agent = Model(LLM) + Harness`。
> 智能来自模型本身，而不是靠堆叠 if-else / 节点图 / prompt 瀑布把『智能拼出来』。
> 我们要做的，是给模型造一辆能在「重庆行业卡反诈质检」这个具体领域里跑起来的车——
> 即 **工具（Tools）+ 知识（Knowledge）+ 观察（Observation）+ 检索（Retrieval）+ 记忆（Memory）**。

输入：通话录音转写的 ASR 文本（`left:` 主叫/外呼方，`right:` 被叫/用户）。
输出：对齐《中间号安全策略复核标注规范》的结构化复核结论——
**正常/违规场景判断 + 风险等级判断 + 判断说明**，供人工二次复核。

---

## 1. 它解决什么问题

重庆行业卡用于银行贷款营销、贷款催收、零售推销、房产中介等外呼。为保证合法合规，
需要根据通话 ASR 文本识别该通话是否 **违规 / 高风险涉诈**。本项目把规范里的：

- **8 类违规场景**：引流第三方平台、贷款相关、法律服务、企业营销与招商服务、商品推销、商业地产、违规催收、其他；
- **5 类涉诈场景**（均为高风险，需单独标注诈骗）：机票退改签、手机租赁套路贷、证券投资类、网贷退息退费、个体工商户年报补录收费；
- **就高不就低** 的判定原则与各子场景的风险等级规则；

固化成一个 **可检索的知识库 + 工具集**，交给模型按需调用与推理。

---

## 2. 架构：一个恒定的 Agent Loop + 可插拔 Harness

```
                          输入：通话 ASR 转写文本
                                   │
                                   ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                     核心质检 Agent (agent loop)                │
   │   while True:                                                  │
   │     resp = LLM.chat(messages, tools)                          │
   │     if 无工具调用: 输出 JSON 结论; break                       │
   │     执行工具 -> 把结果 append 回 messages -> 继续              │
   └───────────────┬──────────────────────────────────────────────┘
                   │ 模型按需调用工具（harness 提供能力）
     ┌─────────────┼───────────────────────────────┐
     ▼             ▼              ▼                 ▼
 search_spec   get_scenario  retrieve_similar   web_search_fraud
 (规范小节)    (场景规则)     _cases (人工判例)  (主体/IP/套路核查)
     │             │              │                 │
     ▼             ▼              ▼                 ▼
   知识库 KnowledgeBase     人工标注语料 CaseStore   （可注入外网后端）
   spec.md + rules.json     数千条 CSV (RAG few-shot)
                   │
                   ▼
        结构化质检结论 InspectionResult
                   │
                   ▼
   ┌──────────────────────────────────────────────┐
   │     反射/演进 Agent (ReflectAgent)             │  ← 与人工标签不符时触发
   │   对比 [Agent结论] vs [人工comment]            │
   │   提炼新高危词/错题本 -> 写回 rules.json        │  自治闭环：越用越准
   └──────────────────────────────────────────────┘
```

**无 LLM 也能跑**：未配置 `LLM_API_KEY`（或未安装 `openai`）时，自动回退到
`HeuristicInspector`——一个完全基于知识库关键词 + 检索判例的确定性基线，
保证零依赖离线可运行、可测试、可作为 LLM 的兜底。

### 与 learn-claude-code 的机制对应

| learn-claude-code 机制 | 本项目落地 |
|---|---|
| s01 Agent Loop | `qc_agent/agent.py` 的 `_inspect_with_llm` 恒定循环 |
| s02 Tool Use（加工具=加 handler） | `qc_agent/tools.py` 的 `ToolRegistry` 分发表 |
| s07 Skill / 按需加载知识 | `search_spec` / `get_scenario` 按需拉取规范小节，而非整篇塞入上下文 |
| s09 Memory（选择/提炼/固化） | `qc_agent/reflect.py` 把错题与新话术固化进 `rules.json` |
| s10 System Prompt 运行时拼装 | `qc_agent/prompts.py` 分节拼接 |
| s11 Error Recovery | LLM 失败 / JSON 不可解析 -> 启发式兜底 |
| RAG few-shot | `qc_agent/case_store.py` 召回最相似人工判例对齐口径 |

---

## 3. 目录结构

```
.
├── knowledge/
│   ├── spec.md            # 质检规范原文（被切成可检索小节）
│   └── rules.json         # 结构化反诈知识库（场景/风险规则/关键词/错题本，可被自动演进）
├── data/
│   └── sample_cases.csv   # 人工复核标注样例（data_id,content,comment）
├── qc_agent/
│   ├── agent.py           # 核心 agent loop（LLM）+ 启发式回退入口
│   ├── llm.py             # OpenAI 兼容 LLM 客户端封装
│   ├── tools.py           # harness 工具集 + 分发表 + tool schema
│   ├── prompts.py         # system prompt 运行时拼装
│   ├── knowledge_base.py  # 规范解析 + 动态规则加载/保存 + 检索
│   ├── case_store.py      # 人工标注 CSV 语料库 + RAG 检索
│   ├── retrieval.py       # 零依赖中文字符 n-gram TF-IDF 检索
│   ├── heuristic.py       # 离线启发式质检（兜底基线）
│   ├── reflect.py         # 反射/自治演进 + 冲突扫描（标签治理）
│   ├── labels.py          # 人工标签归一化（粗标签 -> 规范类目）
│   ├── cache.py           # 结果缓存（内容哈希，磁盘持久化，线程安全）
│   ├── dedup.py           # 近重复 SimHash 聚类去重
│   ├── schema.py          # 结构化输出契约 InspectionResult
│   └── config.py          # 配置（环境变量 / .env）
├── scripts/
│   ├── inspect_text.py    # 单通质检 CLI（--tools/--fast）
│   ├── batch_eval.py      # 批量质检 + 类目/涉诈准确率（--dedup/--cache/--workers）
│   └── evolve.py          # 标签治理：导出『规范vs人工标签』冲突（--apply 才改规则）
└── tests/
    └── test_qc_agent.py   # 离线可跑通的单测（20 项）
```

---

## 4. 快速开始

```bash
# 零依赖即可离线运行（启发式模式）
python3 scripts/inspect_text.py "left:我是投顾客服，联合上海证券创建官方福利群，关注官方接待员的服务号，打开微信..."

# 批量评估（输出逐条结果 + Precision/Recall/F1）
python3 scripts/batch_eval.py --csv data/sample_cases.csv --out results.csv

# 跑单测
python3 -m unittest tests.test_qc_agent -v
```

输出示例：

```
复核标签：涉诈-高风险-证券投资类
是否违规：是    是否涉诈：是
风险等级：高风险
判断说明：证券投资类（涉诈，高风险）。命中特征：投顾、证券、福利群、官方接待员、服务号...
```

### 启用真实 LLM（推荐）

```bash
pip install -r requirements.txt
cp .env.example .env      # 填写 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
python3 scripts/inspect_text.py -v -f call.txt           # 默认快速模式
python3 scripts/inspect_text.py --tools -f call.txt      # 完整 agentic tool loop
```

`LLM_BASE_URL` 兼容任意 OpenAI 协议服务：OpenAI、Claude 兼容网关、通义千问 DashScope 兼容模式、DeepSeek、GLM 等。

### 两种质检模式

| 模式 | 触发 | 机制 | 适用 |
|---|---|---|---|
| **快速模式（默认）** | `QC_USE_TOOLS=false` | 检索增强单轮：预先把相似人工判例 + 规范小节注入 prompt，1 次 LLM 调用直接出结论 | 大批量质检，省时省钱 |
| **工具模式** | `QC_USE_TOOLS=true` 或 `--tools` | 完整 agent loop：模型自主多轮调用 `search_spec`/`get_scenario`/`retrieve_similar_cases` 调查后出结论 | 疑难/高价值复核 |
| **两阶段** | `QC_ESCALATE_BELOW_CONFIDENCE=0.7` | 快速模式出结论，置信度低于阈值时自动升级到工具模式复核 | 兼顾成本与精度 |

---

## 5. 接入你自己的数据

1. **几千条人工复核 CSV**：放到某路径，设 `QC_CASES_PATH=your.csv`（列：`data_id,content,comment`）。
   系统会自动建索引，作为 few-shot 判例库召回，对齐人工口径。
2. **网上几十个网站的诈骗案例**：
   - 离线方式：把套路总结成场景/关键词补进 `knowledge/rules.json`，或直接把案例文本追加进 CSV（`comment` 写明类型）后跑 `scripts/evolve.py`，由反射 Agent 自动沉淀；
   - 联网方式：给 `ToolRegistry(web_search=...)` 注入一个检索后端（如企查查/爱企查主体核查、小红书/抖音套路搜索），模型即可在 loop 中调用 `web_search_fraud` 做主体/IP/套路核查。

### 标签治理 / 自治演进

```bash
# 默认：只导出『规范判定 vs 人工标签』冲突，交人工裁决（不改规则，安全）
python3 scripts/evolve.py --csv data/your_labeled.csv --out conflicts.csv --cache --workers 8

# 谨慎：让反射 Agent 把冲突自动沉淀为新规则（高危词 + 错题本）
python3 scripts/evolve.py --csv data/your_labeled.csv --apply
```

- **冲突导出（默认）**：扫描全量，找出模型判定与人工 `comment` 不一致的样本，导出含冲突类型、
  模型判定/说明/原文片段的 CSV，供人工裁决以哪边为准——这是真实噪声标签下的稳妥做法。
- **`--apply`（自治演进）**：把案例归类到最匹配场景、提炼高危词写回，并把『错题』沉淀进
  `rules.json` 的 `evolved_examples`（下次作为 few-shot 注入），实现零人工改代码的规则自演进。
  注意：人工标签有噪声时谨慎使用，避免学到错误信号。

---

## 6. 输出字段（InspectionResult）

| 字段 | 含义 |
|---|---|
| `is_violation` | 是否违规（含涉诈），正常场景为 false |
| `is_fraud` | 是否涉诈（涉诈须单独标注，风险等级一律高风险） |
| `risk_level` | 合规 / 低风险 / 中风险 / 高风险 |
| `scene_category` / `scene_subtype` | 场景大类 / 子场景 |
| `explanation` | 判断说明，格式【违规/涉诈类型+分析】 |
| `detected_features` / `evidence_quotes` | 命中特征 / 原文证据 |
| `confidence` / `analysis_thought` | 置信度 / 推理与参考依据 |

---

## 7. 在真实数据上的效果与优化建议

在 **827 条真实行业卡反诈人工标注语料**（DeepSeek `deepseek-v4-pro`，快速模式）上评估：

| 指标 | 全量 827 条 | 说明 |
|---|---|---|
| 速度 | 首跑约 21min（并发 8）；**重跑命中缓存 ≈1s** | 缓存让重复评估近乎零成本 |
| 违规检出召回 | 0.886（94 漏判） | 漏判绝大多数是『招商加盟/阿里国际站』 |
| 类目准确率 | 0.863 | 受人工标签噪声拖累 |
| 涉诈判定准确率 | 0.871 | 同上 |

**关键发现（已用 `evolve.py` 全量复核确认）**：114 条冲突（13.8%）里，94 条是『模型判正常、人工判违规』，
而其中 **70 条含『招商』、67 条含『加盟』、8 条『阿里国际站』** ——这些按规范明确是 **合规**
（品牌招商加盟、邀请阿里国际电商开店、外呼方主动加个人微信），却被人工粗标成 `引导投资/引导投资理财`。
**模型实际严格按规范判（对规范一致性约 96-98%）**，表观分数被噪声标签拉低。

### 已落地的优化

1. **快速检索增强单轮模式**：多轮工具往返压成 1 次调用，延迟/成本约降一半（默认）。
2. **批量并发 + 失败重试退避**：`batch_eval --workers N`，23s/条 → ~2s/条；瞬时错误指数退避重试。
3. **结果缓存**（`--cache` / `QC_CACHE_PATH`）：按内容哈希持久化，重跑/增量近乎零成本（827 条重跑约 1s）；周期落盘防中断丢进度。
4. **近重复去重**（`--dedup`，SimHash）：每簇只调一次 LLM，其余复用；本数据集阈值 0.9 约省 30%（ASR 词面差异大，去重收益有限，按数据集而定）。
5. **子类目判别增强**：`rules.json` 增加 `disambiguation` 消歧规则 + 强化手机租赁套路贷关键词，降低『涉诈判对但子类目判错』。
6. **超长转写截断**（`QC_MAX_CONTENT_CHARS`）/ **评估自泄漏修复**（相似判例排除自身）/ **标签归一化**（`labels.py`）/ **两阶段置信度升级**。

### 标签治理工作流（建议1 落地）

由于人工标签存在噪声，`scripts/evolve.py` **默认只导出冲突、不改规则**：

```bash
python3 scripts/evolve.py --csv data/real_cases.csv --out conflicts.csv --cache --workers 8
```

产出 `conflicts.csv`（每条含：人工标签、冲突类型、模型判定/说明/原文片段、建议动作），供人工裁决：
**以规范判定为准、还是修正人工标签**。裁决后再决定是否更新 `rules.json`/规范。
确需让反射 Agent 自动把冲突沉淀为规则时，显式加 `--apply`（谨慎，避免学到噪声）。

### 业务口径判定表（已编码进 `rules.json` 的 `business_decision_table`，prompt 最高优先级）

| 场景 | 判定 | category | subtype | risk_level |
|---|---|---|---|---|
| 品牌招商加盟（棋牌/零食店加盟、阿里国际站开店） | 合规 | 正常 | - | 合规 |
| AI获客 / GEO / 小红书推广 / 排名 / 引流 / 年付半年付 | 违规 | 企业营销与招商服务 | AI推广获客服务 | 低风险 |
| 会展 / 招商会 / 展位 且 有预付款/定金/服务费 | 违规 | 企业营销与招商服务 | 会展营销提前收费 | 低风险 |
| 会展 / 招商会 / 展位 但 无提前收费 | 合规 | 正常 | - | 合规 |
| 股票/证券/基金/理财/投顾 + 群聊/服务号/领取服务 | 涉诈 | 证券投资类 | 证券投资引流 | 高风险 |
| 混合：表面招商/推广，实质引导炒股群/证券服务号/理财 | 涉诈（就高不就低） | 证券投资类 | 证券投资引流 | 高风险 |

已用真实 LLM 验证 6 类场景均按上表输出（含『混合话术→证券投资类』的就高不就低）。
注：人工旧标签把合规的招商加盟标成了 `引导投资理财`，按此口径属**标签错误**，应在
`conflicts.csv` 中改判为合规；改标后表观准确率即可对齐到模型的规范一致性（约 96-98%）。

### 仍可继续的优化

1. **按口径回标**：用 `conflicts.csv` 把招商加盟/会展无收费类的人工标签修正为合规，重评。
2. **子类目 few-shot 增强**：对手机租赁套路贷 vs 引流/贷款继续补判别样本。
3. **可观测性**：记录 token 用量与置信度分布，按置信度阈值路由人工复核。

## 8. 设计取舍

- **零必需依赖**：检索、解析、调度、兜底全部用标准库实现，确保任何环境开箱即跑；`openai` 仅为启用真实 LLM 的可选项。
- **知识与代码分离**：判定规则集中在 `knowledge/`，改规则不改代码；反射 Agent 直接演进知识文件。
- **harness 而非 workflow**：工具是原子、可组合的，是否调用、调用顺序由模型决定，而非硬编码流程，符合 learn-claude-code 的核心主张。
