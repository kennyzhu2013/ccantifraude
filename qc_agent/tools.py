"""Harness 工具集 + 分发表。

遵循 learn-claude-code：『新增工具 = 新增一个 handler』，agent loop 本身不变。
模型通过这些工具按需感知领域知识与历史判例，再自行推理给出结论。
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .case_store import CaseStore
from .knowledge_base import KnowledgeBase


class ToolRegistry:
    def __init__(
        self,
        kb: KnowledgeBase,
        cases: Optional[CaseStore] = None,
        top_k: int = 3,
        web_search: Optional[Callable[[str], str]] = None,
    ):
        self.kb = kb
        self.cases = cases
        self.top_k = top_k
        self._web_search = web_search
        self._handlers: Dict[str, Callable[..., str]] = {
            "search_spec": self._search_spec,
            "list_scenarios": self._list_scenarios,
            "get_scenario": self._get_scenario,
            "load_skill": self._load_skill,
            "retrieve_similar_cases": self._retrieve_similar_cases,
            "web_search_fraud": self._web_search_fraud,
        }

    # ---------- 工具实现 ----------
    def _search_spec(self, query: str, top_k: Optional[int] = None) -> str:
        sections = self.kb.search_spec(query, top_k=top_k or self.top_k)
        if not sections:
            return "未检索到相关规范小节。"
        out = []
        for s in sections:
            body = s.content if len(s.content) <= 1500 else s.content[:1500] + "…"
            out.append(f"### {s.full_title}\n{body}")
        return "\n\n".join(out)

    def _list_scenarios(self) -> str:
        return json.dumps(self.kb.list_scenarios(), ensure_ascii=False, indent=2)

    def _get_scenario(self, category: str) -> str:
        sc = self.kb.get_scenario(category)
        if not sc:
            avail = self.kb.list_scenarios()
            return f"未找到场景『{category}』。可选场景：{json.dumps(avail, ensure_ascii=False)}"
        return json.dumps(sc, ensure_ascii=False, indent=2)

    def _load_skill(self, name: str) -> str:
        if not self.kb.skills_available:
            return "技能库未启用，请改用 get_scenario 获取场景规则。"
        skill = self.kb.skills.get(name)
        if skill is None:
            names = "、".join(s.name for s in self.kb.skills.skills)
            return f"未找到技能『{name}』。可选技能：{names}"
        return f"### 技能：{skill.name}（{skill.bucket}·{skill.risk}）\n{skill.body}"

    def format_similar_cases(self, hits: Sequence[Tuple[Any, float]]) -> str:
        """把 (判例, 相似度) 列表渲染为 few-shot 块。

        独立出来供快速模式复用已检索好的结果，避免同一通话重复检索判例库。
        """
        if not self.cases or len(self.cases) == 0:
            return "暂无人工标注判例库。"
        if not hits:
            return "未检索到相似人工判例。"
        out = []
        for case, score in hits:
            out.append(
                f"- 相似度{score:.2f} | 人工标签：{case.comment or '无'}\n  内容片段：{case.short(260)}"
            )
        return "检索到的相似人工复核判例（few-shot 参考）：\n" + "\n".join(out)

    def _retrieve_similar_cases(
        self, text: str, top_k: Optional[int] = None, exclude_id: Optional[str] = None
    ) -> str:
        if not self.cases or len(self.cases) == 0:
            return "暂无人工标注判例库。"
        hits = self.cases.retrieve_with_scores(
            text, top_k=top_k or self.top_k, exclude_id=exclude_id
        )
        return self.format_similar_cases(hits)

    def _web_search_fraud(self, query: str) -> str:
        if self._web_search is None:
            return (
                "联网检索未启用（当前环境无外网或未注入检索后端）。"
                "可在 ToolRegistry 注入 web_search 回调以核查公众号/小程序主体、IP归属、套路分享。"
            )
        try:
            return self._web_search(query)
        except Exception as exc:  # pragma: no cover
            return f"联网检索失败：{exc}"

    # ---------- 分发 ----------
    def dispatch(self, name: str, arguments: Dict[str, Any]) -> str:
        handler = self._handlers.get(name)
        if handler is None:
            return f"未知工具：{name}"
        try:
            return handler(**arguments)
        except TypeError as exc:
            return f"工具参数错误（{name}）：{exc}"
        except Exception as exc:  # pragma: no cover
            return f"工具执行异常（{name}）：{exc}"

    # ---------- OpenAI tool schema ----------
    def openai_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_spec",
                    "description": "检索《重庆中间号安全策略复核标注规范》中与查询最相关的小节原文（违规/涉诈场景定义、风险等级、判断方法、典型话术）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "检索关键词或问题，如『引流 服务号 高风险』"},
                            "top_k": {"type": "integer", "description": "返回小节数，默认3"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_scenarios",
                    "description": "列出全部违规场景与涉诈场景类目，用于先了解判定空间。",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_scenario",
                    "description": "获取某个场景类目的结构化定义：判断方法、风险等级规则、关键词。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string", "description": "场景类目名，如『引流第三方平台』『证券投资类』"}
                        },
                        "required": ["category"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "load_skill",
                    "description": "加载某场景类目的完整判定技能（判断方法、风险等级规则、业务口径判定、易混消歧、错题本）。判定前应优先加载系统提示技能目录中与通话最相关的 1-3 个技能。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "技能名（即场景类目名），如『手机租赁套路贷诈骗』"}
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "retrieve_similar_cases",
                    "description": "在数千条人工复核标注语料中检索与当前通话最相似的历史判例及其人工标签，作为 few-shot 对齐参考。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "当前通话转写文本（或其关键片段）"},
                            "top_k": {"type": "integer", "description": "返回判例数，默认3"},
                        },
                        "required": ["text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search_fraud",
                    "description": "联网核查通话中提到的公众号/服务号/小程序/APP 主体经营异常、IP归属，或搜索是否有相关诈骗套路分享（需注入检索后端，离线环境不可用）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "核查查询词"}
                        },
                        "required": ["query"],
                    },
                },
            },
        ]
