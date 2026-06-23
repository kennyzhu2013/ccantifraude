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
│   ├── reflect.py         # 反射/自治演进 Agent
│   ├── schema.py          # 结构化输出契约 InspectionResult
│   └── config.py          # 配置（环境变量 / .env）
├── scripts/
│   ├── inspect_text.py    # 单通质检 CLI
│   ├── batch_eval.py      # 批量质检 + 与人工标签对比评估
│   └── evolve.py          # 自治演进 CLI（从语料沉淀新规则）
└── tests/
    └── test_qc_agent.py   # 离线可跑通的单测（16 项）
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

### 自治演进（自我进化）

```bash
python3 scripts/evolve.py --csv data/your_labeled.csv --limit 500
```

当主 Agent 判定与人工 `comment` 不一致时，反射 Agent 会：
- 把该案例归类到最匹配的场景，提炼新增高危关键词写回对应场景；
- 把『错题』沉淀进 `rules.json` 的 `evolved_examples`，下次作为 few-shot 注入主 Agent。
从而实现 **零人工改代码** 的规则自演进，契合诈骗话术高频更迭的特性。

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

在 **827 条真实行业卡反诈人工标注语料**（DeepSeek，快速模式，并发 6）上抽样评估：

| 指标 | 结果（抽样 60 条，已排除检索自泄漏） |
|---|---|
| 速度 | 单轮 ~11s/条；并发后 **~2s/条**（工具模式约 23s/条） |
| 违规检出召回 | 0.93 |
| 类目准确率 | 0.92 |
| 涉诈判定准确率 | 0.93 |

**逐条复盘后的关键发现**：模型『错误』里 **绝大多数其实是人工标签噪声**——
例如 `品牌招商加盟`（棋牌室/零食店加盟、阿里国际站开店）按规范明确为 **合规**、
`外呼人员主动添加个人微信` 也为 **合规**，但人工把它们粗标成了 `引导投资/引导投资理财`。
**模型实际是严格按规范判的（对规范一致性 ~98%）**，被噪声标签拉低了表观分数。
只有少量是真实可改进项（如某『手机租赁套路贷』被正确识别为涉诈高风险，但子类目误归到『引流第三方平台』）。

### 已落地的优化

1. **快速检索增强单轮模式**：把多轮工具往返压成 1 次调用，延迟/成本约降一半（默认开启）。
2. **批量并发 + 失败重试退避**：`batch_eval --workers N`，23s/条 → ~2s/条；瞬时错误自动退避重试。
3. **超长转写截断**：保留头尾（`QC_MAX_CONTENT_CHARS`），控制 token（真实样本最长 1.2 万字）。
4. **评估自泄漏修复**：相似判例检索排除样本自身，避免把标准答案喂给模型。
5. **标签归一化 + 类目/涉诈准确率**：`qc_agent/labels.py` 对齐粗粒度人工口径，评估更可信。
6. **两阶段置信度升级**：低置信度自动从快速模式升级到工具模式复核。

### 建议的后续优化（按收益排序）

1. **标签治理（收益最大）**：当前 827 条中 733 条是 `引导投资/引导投资理财`，且与规范存在冲突
   （把合规的招商加盟标成引导投资）。建议：①明确业务口径——招商加盟投资到底算不算违规，
   据此更新 `rules.json`/规范；②用反射 Agent（`scripts/evolve.py`）自动把『规范判定 vs 人工标签』
   冲突样本挑出来交人工再标，沉淀干净的 few-shot。
2. **子类目判别增强**：对易混场景（手机租赁套路贷 vs 引流/贷款）补充判别关键词与 few-shot，
   降低『涉诈判对但子类目判错』。
3. **成本进一步下降**：内容指纹去重（大量近重复投顾话术）、结果缓存、对明显正常样本用启发式预筛，
   仅把不确定样本交给 LLM。
4. **可观测性**：记录 token 用量与置信度分布，按置信度阈值路由人工复核。

## 8. 设计取舍

- **零必需依赖**：检索、解析、调度、兜底全部用标准库实现，确保任何环境开箱即跑；`openai` 仅为启用真实 LLM 的可选项。
- **知识与代码分离**：判定规则集中在 `knowledge/`，改规则不改代码；反射 Agent 直接演进知识文件。
- **harness 而非 workflow**：工具是原子、可组合的，是否调用、调用顺序由模型决定，而非硬编码流程，符合 learn-claude-code 的核心主张。
