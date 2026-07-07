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

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cache import ResultCache
from .case_store import CaseStore
from .config import Config, DEFAULT_CONFIG
from .heuristic import HeuristicInspector
from .knowledge_base import KnowledgeBase
from .llm import LLMClient
from .prompts import (
    _JSON_FORCE,
    _JSON_REPAIR,
    build_fast_user_message,
    build_system_prompt,
    build_system_prompt_fast,
)
from .schema import InspectionResult, RiskLevel
from .tools import ToolRegistry


def _truncate(content: str, max_chars: int) -> str:
    """超长转写保留头尾，控制 token；中间以省略标记替代。"""
    if max_chars <= 0 or len(content) <= max_chars:
        return content
    head = int(max_chars * 0.6)
    tail = max_chars - head
    return content[:head] + "\n…（中间省略）…\n" + content[-tail:]

_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_INSPECTION_KEYS = frozenset(
    {"is_violation", "is_fraud", "risk_level", "scene_category", "explanation"}
)


def _looks_like_inspection_result(obj: Dict[str, Any]) -> bool:
    return bool(_INSPECTION_KEYS & obj.keys())


def _try_parse_json(candidate: str) -> Optional[Dict[str, Any]]:
    text = (candidate or "").strip()
    if not text:
        return None
    variants = [text, _TRAILING_COMMA.sub(r"\1", text)]
    for variant in variants:
        try:
            obj = json.loads(variant)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _extract_balanced_object(text: str, start: int) -> Optional[str]:
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出中提取质检 JSON，兼容 markdown 包裹、前后说明文字与嵌套字段。"""
    if not text:
        return None

    candidates: List[str] = []
    for m in re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE):
        candidates.append(m.group(1))
    for i, ch in enumerate(text):
        if ch == "{":
            balanced = _extract_balanced_object(text, i)
            if balanced:
                candidates.append(balanced)

    best: Optional[Dict[str, Any]] = None
    for candidate in candidates:
        obj = _try_parse_json(candidate)
        if obj is None:
            continue
        if _looks_like_inspection_result(obj):
            return obj
        if best is None:
            best = obj
    return best


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
        self.cache: Optional[ResultCache] = None
        if self.config.cache_path:
            # 知识指纹纳入缓存命名空间：rules/规则/判定表变更后旧缓存自动失效，避免陈旧结论。
            kb_fp = hashlib.sha1(self.kb.rules_brief().encode("utf-8")).hexdigest()[:10]
            self.cache = ResultCache(
                Path(self.config.cache_path), self.config.llm_model, f"{self.mode}:{kb_fp}"
            )

    @property
    def mode(self) -> str:
        return "llm" if self.llm.available else "heuristic"

    def flush_cache(self) -> None:
        if self.cache is not None:
            self.cache.flush()

    # ---------- 对外主入口 ----------
    def inspect(
        self,
        content: str,
        data_id: Optional[str] = None,
        use_tools: Optional[bool] = None,
    ) -> InspectionResult:
        if not self.llm.available:
            return self.heuristic.inspect(content, data_id=data_id)
        tool_mode = self.config.use_tools if use_tools is None else use_tools

        cache_ns = "tools" if tool_mode else "fast"
        if self.cache is not None:
            cached = self.cache.get(cache_ns + "\n" + content)
            if cached is not None:
                res = InspectionResult.from_dict(cached)
                res.data_id = data_id
                return res

        try:
            if tool_mode:
                res = self._inspect_with_llm(content, data_id=data_id)
            else:
                res = self._inspect_fast(content, data_id=data_id)
                # 两阶段：快速模式低置信度时升级到完整工具回路复核。
                threshold = self.config.escalate_below_confidence
                if threshold > 0 and 0 < res.confidence < threshold:
                    if self.verbose:
                        print(f"[escalate] 置信度 {res.confidence:.2f} < {threshold}，升级工具模式复核")
                    res = self._inspect_with_llm(content, data_id=data_id)
        except Exception as exc:  # pragma: no cover - LLM 调用失败时兜底
            if self.verbose:
                print(f"[warn] LLM 质检失败，回退启发式：{exc}")
            res = self.heuristic.inspect(content, data_id=data_id)
            res.analysis_thought = (res.analysis_thought + f"\n[LLM失败回退] {exc}").strip()
            return res

        if self.cache is not None and res.source.startswith("llm"):
            self.cache.set(cache_ns + "\n" + content, res.to_dict())
        return res

    # ---------- 快速模式：检索增强单轮 ----------
    def _inspect_fast(self, content: str, data_id: Optional[str] = None) -> InspectionResult:
        clipped = _truncate(content, self.config.max_content_chars)
        # 排除当前样本自身，避免评估时把『标准答案』当相似判例泄漏给模型。
        similar_block = self.tools.dispatch(
            "retrieve_similar_cases", {"text": clipped, "exclude_id": data_id}
        )
        spec_block = self.tools.dispatch(
            "search_spec", {"query": clipped[:300], "top_k": 3}
        )
        messages = [
            {"role": "system", "content": build_system_prompt_fast(self.kb)},
            {
                "role": "user",
                "content": build_fast_user_message(clipped, similar_block, spec_block),
            },
        ]
        msg = self.llm.chat(messages, tools=None)
        return self._finalize(msg["content"], content, data_id, source="llm-fast", messages=messages)

    # ---------- LLM agent loop ----------
    def _inspect_with_llm(self, content: str, data_id: Optional[str] = None) -> InspectionResult:
        system_prompt = build_system_prompt(self.kb)
        clipped = _truncate(content, self.config.max_content_chars)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "请对以下通话转写文本进行反诈质检，按系统要求先调查再输出 JSON 结论。\n\n"
                    f"【通话转写文本】\n{clipped}"
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
            messages.append({"role": "user", "content": _JSON_FORCE})
            msg = self.llm.chat(messages, tools=None)
            final_text = msg["content"]

        return self._finalize(final_text, content, data_id, source="llm", messages=messages)

    def _repair_json_with_llm(
        self, messages: List[Dict[str, Any]], partial_text: str
    ) -> Optional[Dict[str, Any]]:
        """工具回路结束后 JSON 不可解析时，追加一轮无工具修复请求。"""
        retry_messages = list(messages)
        if (partial_text or "").strip():
            retry_messages.append({"role": "assistant", "content": partial_text})
        retry_messages.append({"role": "user", "content": _JSON_REPAIR})
        try:
            msg = self.llm.chat(retry_messages, tools=None)
        except Exception as exc:
            if self.verbose:
                print(f"[warn] JSON 修复请求失败：{exc}")
            return None
        payload = _extract_json(msg.get("content") or "")
        if payload is None and self.verbose:
            print("[warn] JSON 修复后仍不可解析")
        return payload

    # ---------- 结果归一化 ----------
    def _finalize(
        self,
        final_text: str,
        content: str,
        data_id: Optional[str],
        source: str,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> InspectionResult:
        payload = _extract_json(final_text)
        if payload is None and messages is not None:
            payload = self._repair_json_with_llm(messages, final_text)
            if payload is not None:
                source = f"{source}-repair"

        if payload is None:
            # 模型未给出可解析 JSON，回退启发式并附上模型原文。
            res = self.heuristic.inspect(content, data_id=data_id)
            res.source = "llm+heuristic"
            res.analysis_thought = (
                f"[LLM未返回可解析JSON，启发式兜底]\n模型原文：{final_text[:500]}"
            )
            return res

        payload.setdefault("source", source)
        res = InspectionResult.from_dict(payload)
        res.source = source
        res.data_id = data_id
        # 一致性约束：涉诈一律高风险；任何非合规等级即视为违规。
        if res.is_fraud:
            res.risk_level = RiskLevel.HIGH
            res.is_violation = True
        if res.risk_level.rank > 0:
            res.is_violation = True
        return res
