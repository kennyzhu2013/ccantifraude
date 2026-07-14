"""离线启发式质检（无 LLM 时的回退路径）。

它不是『真正的 Agent』——按 learn-claude-code 的观点，智能应来自模型。
但它给 harness 提供一个可离线跑通、可被测试、可作为 LLM 兜底的确定性基线，
其规则与权重全部来自知识库（rules.json + 检索到的人工判例），便于解释与演进。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .case_store import CaseStore, LabeledCase
from .knowledge_base import KnowledgeBase
from .schema import InspectionResult, RiskLevel

# 违规场景的默认风险等级（与规范一致；可被高危触发词升级）。
_VIOLATION_DEFAULT_RISK = {
    "引流第三方平台": RiskLevel.HIGH,
    "贷款相关": RiskLevel.LOW,
    "法律服务": RiskLevel.LOW,
    "企业营销与招商服务": RiskLevel.LOW,
    "商品推销": RiskLevel.LOW,
    "商业地产": RiskLevel.MEDIUM,
    "违规催收": RiskLevel.LOW,
    "其他": RiskLevel.HIGH,
}

# 高危触发词：命中即把对应类目升级为高风险。
_HIGH_RISK_TRIGGERS = {
    "贷款相关": ["砍头息", "债务优化", "到手", "解冻金", "操作提现", "提额降息", "关闭会员", "芝麻信用分"],
    "法律服务": ["帮退律所", "一半服务费", "律所费用"],
    "商品推销": [],
    "违规催收": ["上门", "全网公开", "纪检", "公检法", "冒充", "法院人员", "司法调节站", "诉讼中心", "限制高消费"],
    "企业营销与招商服务": ["虚开", "成本票", "核定征收"],
}

# 合规信号（个人微信、外呼人员主动添加 → 不应判为引流违规）。
_COMPLIANT_WECHAT = re.compile(r"(我加(你|您|一下)|我这边加|我备注|我加个(你|您)|经理.{0,4}(待会|稍后|马上).{0,4}加)")
# 引导用户主动添加 / 非个人微信 → 高风险引流。
_ACTIVE_ADD = re.compile(
    r"(你|您).{0,6}(打开|点开).{0,3}微信|搜索一下|报(一下|个)号码|添加到通讯录|"
    r"关注.{0,4}(服务号|公众号)|领取服务|置顶|免打扰|扫(一下|码|二维码)|"
    r"下载|点(击|一下).{0,3}链接|屏幕共享|会议软件|加(个|入).{0,3}群|福利群"
)


def _count_hits(norm_text: str, keywords: List[str]) -> Tuple[int, List[str]]:
    hit = [k for k in keywords if k and k in norm_text]
    return len(hit), hit


class HeuristicInspector:
    def __init__(self, kb: KnowledgeBase, cases: Optional[CaseStore] = None, top_k: int = 3):
        self.kb = kb
        self.cases = cases
        self.top_k = top_k

    def inspect(self, content: str, data_id: Optional[str] = None) -> InspectionResult:
        text = content or ""
        norm = text.replace("\n", "")

        candidates: List[Tuple[RiskLevel, int, str, str, List[str], bool]] = []
        # (risk, score, category, subtype, features, is_fraud)

        # 涉诈场景（高风险，涉诈）。
        for sc in self.kb.rules.get("fraud_scenarios", []):
            cat = sc.get("category", "")
            n, hits = _count_hits(norm, sc.get("keywords", []))
            if n >= 2:
                candidates.append((RiskLevel.HIGH, n, cat, "", hits, True))

        # 违规场景。
        for sc in self.kb.rules.get("violation_scenarios", []):
            cat = sc.get("category", "")
            n, hits = _count_hits(norm, sc.get("keywords", []))
            if n < 1:
                continue
            risk = _VIOLATION_DEFAULT_RISK.get(cat, RiskLevel.LOW)
            subtype = ""

            if cat == "引流第三方平台":
                if _ACTIVE_ADD.search(norm):
                    risk = RiskLevel.HIGH
                    subtype = "引导用户主动添加/非个人微信"
                elif _COMPLIANT_WECHAT.search(norm) and not _ACTIVE_ADD.search(norm):
                    # 外呼人员主动添加个人微信 → 合规，不计入违规候选。
                    continue
            for trg in _HIGH_RISK_TRIGGERS.get(cat, []):
                if trg in norm:
                    risk = RiskLevel.HIGH
                    break

            # 至少 2 个关键词命中或命中高危触发，才计为违规，降低误报。
            if n >= 2 or risk == RiskLevel.HIGH:
                candidates.append((risk, n, cat, subtype, hits, False))

        similar = self.cases.retrieve(text, top_k=self.top_k, exclude_id=data_id) if self.cases else []

        if not candidates:
            res = InspectionResult(
                is_violation=False,
                is_fraud=False,
                risk_level=RiskLevel.COMPLIANT,
                scene_category="正常",
                explanation="正常场景，未识别到违规话术，直接判定正常提交。",
                confidence=0.4,
                source="heuristic",
                data_id=data_id,
            )
            res.analysis_thought = self._format_similar(similar)
            return res

        # 就高不就低：先比风险等级，再比命中数。
        candidates.sort(key=lambda c: (c[0].rank, c[1]), reverse=True)
        risk, score, cat, subtype, feats, is_fraud = candidates[0]

        scenario = self.kb.get_scenario(cat) or {}
        method = scenario.get("judgment_method", "")
        tag = "涉诈" if is_fraud else "违规"
        detail = f"{cat}" + (f"：{subtype}" if subtype else "")
        explanation = f"{detail}（{tag}，{risk.value}）。命中特征：{('、'.join(feats[:6]))}。{method}"

        res = InspectionResult(
            is_violation=True,
            is_fraud=is_fraud,
            risk_level=risk,
            scene_category=cat,
            scene_subtype=subtype,
            explanation=explanation,
            detected_features=feats,
            evidence_quotes=self._extract_quotes(text, feats),
            confidence=min(0.5 + 0.1 * score, 0.95),
            analysis_thought=self._format_similar(similar),
            source="heuristic",
            data_id=data_id,
        )
        return res

    @staticmethod
    def _extract_quotes(text: str, features: List[str], max_quotes: int = 3) -> List[str]:
        quotes: List[str] = []
        for feat in features:
            idx = text.find(feat)
            if idx >= 0:
                start = max(0, idx - 12)
                end = min(len(text), idx + len(feat) + 18)
                quotes.append(text[start:end].replace("\n", " ").strip())
            if len(quotes) >= max_quotes:
                break
        return quotes

    @staticmethod
    def _format_similar(similar: List[LabeledCase]) -> str:
        if not similar:
            return ""
        parts = ["参考的相似人工判例："]
        for c in similar:
            parts.append(f"- [{c.comment or '无标签'}] {c.short(80)}")
        return "\n".join(parts)
