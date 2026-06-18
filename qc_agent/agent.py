"""核心质检 Agent。

体现 learn-claude-code 的核心模式：一个恒定的 agent loop。
    while True:
        resp = llm.chat(messages, tools)
        if 无工具调用: break
        执行工具，把结果 append 回 messages，继续循环
模型决定何时调用工具、何时停止；harness 只负责执行模型的请求。

无 LLM（未配置 Key / 未装 openai）时回退到 HeuristicInspector，保证可离线运行。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .case_store import CaseStore
from .config import Config, DEFAULT_CONFIG
from .heuristic import HeuristicInspector
from .knowledge_base import KnowledgeBase
from .llm import LLMClient
from .prompts import build_system_prompt
from .schema import InspectionResult, RiskLevel
from .tools import ToolRegistry

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        m = _JSON_BLOCK.search(text)
        candidate = m.group(0) if m else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


class QcAgent:
    def __init__(
        self,
        config: Optional[Config] = None,
        kb: Optional[KnowledgeBase] = None,
        cases: Optional[CaseStore] = None,
        web_search=None,
        verbose: bool = False,
    ):
        self.config = config or DEFAULT_CONFIG
        self.kb = kb or KnowledgeBase(self.config.spec_path, self.config.rules_path)
        self.cases = cases if cases is not None else CaseStore(self.config.cases_path)
        self.llm = LLMClient(self.config)
        self.tools = ToolRegistry(
            self.kb, self.cases, top_k=self.config.retrieve_top_k, web_search=web_search
        )
        self.heuristic = HeuristicInspector(self.kb, self.cases, top_k=self.config.retrieve_top_k)
        self.verbose = verbose

    @property
    def mode(self) -> str:
        return "llm" if self.llm.available else "heuristic"

    # ---------- 对外主入口 ----------
    def inspect(self, content: str, data_id: Optional[str] = None) -> InspectionResult:
        if not self.llm.available:
            return self.heuristic.inspect(content, data_id=data_id)
        try:
            return self._inspect_with_llm(content, data_id=data_id)
        except Exception as exc:  # pragma: no cover - LLM 调用失败时兜底
            if self.verbose:
                print(f"[warn] LLM 质检失败，回退启发式：{exc}")
            res = self.heuristic.inspect(content, data_id=data_id)
            res.analysis_thought = (res.analysis_thought + f"\n[LLM失败回退] {exc}").strip()
            return res

    # ---------- LLM agent loop ----------
    def _inspect_with_llm(self, content: str, data_id: Optional[str] = None) -> InspectionResult:
        system_prompt = build_system_prompt(self.kb)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "请对以下通话转写文本进行反诈质检，按系统要求先调查再输出 JSON 结论。\n\n"
                    f"【通话转写文本】\n{content}"
                ),
            },
        ]
        tool_schemas = self.tools.openai_schemas()

        final_text = ""
        for turn in range(self.config.max_tool_turns):
            msg = self.llm.chat(messages, tools=tool_schemas, tool_choice="auto")
            tool_calls = msg["tool_calls"]

            if not tool_calls:
                final_text = msg["content"]
                break

            # 记录 assistant 的工具调用意图。
            messages.append(
                {
                    "role": "assistant",
                    "content": msg["content"] or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]},
                        }
                        for tc in tool_calls
                    ],
                }
            )
            for tc in tool_calls:
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                if self.verbose:
                    print(f"[turn {turn}] 调用工具 {tc['name']}({args})")
                result = self.tools.dispatch(tc["name"], args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    }
                )
        else:
            # 工具回合用尽，强制要求出结论。
            messages.append(
                {"role": "user", "content": "请立即停止调用工具，仅输出最终 JSON 结论。"}
            )
            msg = self.llm.chat(messages, tools=None)
            final_text = msg["content"]

        payload = _extract_json(final_text)
        if payload is None:
            # 模型未给出可解析 JSON，回退启发式并附上模型原文。
            res = self.heuristic.inspect(content, data_id=data_id)
            res.source = "llm+heuristic"
            res.analysis_thought = (
                f"[LLM未返回可解析JSON，启发式兜底]\n模型原文：{final_text[:500]}"
            )
            return res

        payload.setdefault("source", "llm")
        res = InspectionResult.from_dict(payload)
        res.data_id = data_id
        # 一致性约束：涉诈一律高风险；任何非合规等级即视为违规。
        if res.is_fraud:
            res.risk_level = RiskLevel.HIGH
            res.is_violation = True
        if res.risk_level.rank > 0:
            res.is_violation = True
        return res
