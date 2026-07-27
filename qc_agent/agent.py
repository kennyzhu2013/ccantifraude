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
from .verify import verify_evidence


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
        self.kb = kb or KnowledgeBase(
            self.config.spec_path,
            self.config.rules_path,
            skills_dir=self.config.skills_dir if self.config.use_skills else None,
        )
        self.cases = cases if cases is not None else CaseStore(self.config.cases_path)
        self.llm = LLMClient(self.config)
        self.tools = ToolRegistry(
            self.kb, self.cases, top_k=self.config.retrieve_top_k, web_search=web_search
        )
        self.heuristic = HeuristicInspector(self.kb, self.cases, top_k=self.config.retrieve_top_k)
        self.verbose = verbose
        self.cache: Optional[ResultCache] = None
        if self.config.cache_path:
            # 知识指纹纳入缓存命名空间：rules/技能变更后旧缓存自动失效，避免陈旧结论。
            fp_src = self.kb.rules_brief()
            if self.kb.skills_available:
                fp_src += self.kb.skills.digest()
            kb_fp = hashlib.sha1(fp_src.encode("utf-8")).hexdigest()[:10]
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
                reason = self._escalation_reason(res, content)
                if reason:
                    if self.verbose:
                        print(f"[escalate] {reason}，升级工具模式复核")
                    res = self._inspect_with_llm(content, data_id=data_id)
                    res.source = "llm-escalated"
                    res.analysis_thought = (
                        f"[升级复核触发原因] {reason}\n" + res.analysis_thought
                    ).strip()
        except Exception as exc:  # pragma: no cover - LLM 调用失败时兜底
            if self.verbose:
                print(f"[warn] LLM 质检失败，回退启发式：{exc}")
            res = self.heuristic.inspect(content, data_id=data_id)
            res.analysis_thought = (res.analysis_thought + f"\n[LLM失败回退] {exc}").strip()
            return res

        if self.cache is not None and res.source.startswith("llm"):
            self.cache.set(cache_ns + "\n" + content, res.to_dict())
        return res

    # ---------- 升级信号 ----------
    def _escalation_reason(self, res: InspectionResult, content: str) -> str:
        """快速模式结论的可疑信号：命中任一即升级到工具回路复核。

        实测 confidence 饱和在 0.95-1.0，置信度阈值区分度差；改用确定性信号：
        ① 证据校验失败（违规/涉诈结论但引用未在原文命中，few-shot 照抄的特征）；
        ② 启发式命中涉诈关键词但 LLM 判正常（潜在漏判）；
        ③ 输出经 JSON 修复或启发式兜底（结论可靠性存疑）。
        """
        signals = []
        if self.config.escalate_on_signals:
            if res.evidence_verified is False:
                signals.append("违规结论的证据引用未在原文命中")
            if not res.is_violation:
                heur = self.heuristic.inspect(content)
                if heur.is_fraud:
                    signals.append(
                        f"启发式命中涉诈关键词（{heur.scene_category}）但LLM判正常"
                    )
            if res.source.endswith("-repair") or res.source == "llm+heuristic":
                signals.append("输出经JSON修复/兜底")
        threshold = self.config.escalate_below_confidence
        if threshold > 0 and 0 < res.confidence < threshold:
            signals.append(f"置信度{res.confidence:.2f}低于阈值{threshold}")
        return "；".join(signals)

    # ---------- 快速模式：检索增强单轮 ----------
    def _disambig_block(self, clipped: str) -> str:
        """分层注入：system prompt 只常驻消歧标题索引，命中的正文随待检文本给出。"""
        entries = self.kb.relevant_disambiguation(clipped, top_k=self.config.disambig_top_k)
        return "\n".join(f"- {e}" for e in entries)

    def _skills_block(self, clipped: str) -> str:
        """技能路由：按通话内容选 top-k 场景技能，注入完整判定细则。"""
        if not self.kb.skills_available:
            return ""
        routed = self.kb.skills.route(clipped, top_k=self.config.skills_top_k)
        if not routed:
            return ""
        if self.verbose:
            names = ", ".join(f"{s.name}({sc:.1f})" for s, sc in routed)
            print(f"[skills] 路由命中: {names}")
        return self.kb.skills.render([s for s, _ in routed])

    def _inspect_fast(self, content: str, data_id: Optional[str] = None) -> InspectionResult:
        clipped = _truncate(content, self.config.max_content_chars)
        # 排除当前样本自身，避免评估时把『标准答案』当相似判例泄漏给模型。
        similar_block = self.tools.dispatch(
            "retrieve_similar_cases", {"text": clipped, "exclude_id": data_id}
        )
        spec_block = self.tools.dispatch(
            "search_spec", {"query": clipped[:300], "top_k": 3}
        )
        skills_block = self._skills_block(clipped)
        # 技能已内嵌各自类目的消歧规则，启用技能时只补充未随技能注入的部分。
        disambig_block = "" if skills_block else self._disambig_block(clipped)
        messages = [
            {"role": "system", "content": build_system_prompt_fast(self.kb)},
            {
                "role": "user",
                "content": build_fast_user_message(
                    clipped, similar_block, spec_block, disambig_block, skills_block
                ),
            },
        ]
        msg = self.llm.chat(messages, tools=None)
        return self._finalize(msg["content"], content, data_id, source="llm-fast", messages=messages)

    # ---------- LLM agent loop ----------
    def _inspect_with_llm(self, content: str, data_id: Optional[str] = None) -> InspectionResult:
        system_prompt = build_system_prompt(self.kb)
        clipped = _truncate(content, self.config.max_content_chars)
        user_parts = [
            "请对以下通话转写文本进行反诈质检，按系统要求先调查再输出 JSON 结论。"
        ]
        disambig_block = self._disambig_block(clipped)
        if disambig_block:
            user_parts.append(
                "【与本通话相关的易混场景消歧规则（重要，先按此消歧再定类目）】\n" + disambig_block
            )
        user_parts.append(f"【通话转写文本】\n{clipped}")
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(user_parts)},
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
        # 类目归一化：模型偶发把真实类目落在 subtype，主类目写成泛化词
        # （如『诈骗』）或错误大类（如涉诈结论主类目写『其他』）——提升 subtype。
        sub = (res.scene_subtype or "").strip().strip("-/")
        violation_cats = {
            s.get("category") for s in self.kb.rules.get("violation_scenarios", [])
        }
        fraud_cats = {
            s.get("category") for s in self.kb.rules.get("fraud_scenarios", [])
        }
        generic = {"诈骗", "涉诈", "违规", "违规场景", "涉诈类型"}
        if sub in violation_cats | fraud_cats:
            if res.scene_category in generic or (
                res.is_fraud and res.scene_category not in fraud_cats and sub in fraud_cats
            ):
                res.scene_category, res.scene_subtype = sub, ""
        # 证据校验：违规/涉诈结论的引用必须能在待检原文中命中（拦截 few-shot 照抄）。
        if res.is_violation and res.evidence_quotes:
            ratio, missing = verify_evidence(res.evidence_quotes, content)
            res.evidence_verified = ratio > 0
            if not res.evidence_verified:
                res.analysis_thought = (
                    "[证据校验失败] evidence_quotes 均未在待检原文中命中，"
                    "结论可能照抄了相似判例/规范话术，需人工留意。\n" + res.analysis_thought
                ).strip()
                if self.verbose:
                    print(f"[verify] 证据未命中原文: {missing[:2]}")
        return res
