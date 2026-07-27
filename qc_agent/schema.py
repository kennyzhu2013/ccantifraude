"""质检结构化输出契约。

对齐《中间号安全策略复核标注规范》的复核产物：
    正常/违规场景判断 + 风险等级判断 + 判断说明。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class RiskLevel(str, Enum):
    COMPLIANT = "合规"
    LOW = "低风险"
    MEDIUM = "中风险"
    HIGH = "高风险"

    @classmethod
    def from_text(cls, value: Optional[str]) -> "RiskLevel":
        if not value:
            return cls.COMPLIANT
        v = value.strip()
        mapping = {
            "合规": cls.COMPLIANT,
            "正常": cls.COMPLIANT,
            "无": cls.COMPLIANT,
            "低": cls.LOW,
            "低风险": cls.LOW,
            "中": cls.MEDIUM,
            "中风险": cls.MEDIUM,
            "高": cls.HIGH,
            "高风险": cls.HIGH,
        }
        return mapping.get(v, cls.COMPLIANT)

    @property
    def rank(self) -> int:
        return {
            RiskLevel.COMPLIANT: 0,
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
        }[self]


@dataclass
class InspectionResult:
    """单通通话的质检结论。"""

    is_violation: bool = False
    is_fraud: bool = False
    risk_level: RiskLevel = RiskLevel.COMPLIANT
    scene_category: str = "正常"
    scene_subtype: str = ""
    explanation: str = "正常场景，未识别到违规话术。"
    detected_features: List[str] = field(default_factory=list)
    evidence_quotes: List[str] = field(default_factory=list)
    confidence: float = 0.0
    analysis_thought: str = ""
    source: str = "llm"  # llm | heuristic
    data_id: Optional[str] = None
    # 证据校验结果：None=未校验；False=违规/涉诈结论但证据引用未在原文命中（需人工留意）。
    evidence_verified: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["risk_level"] = self.risk_level.value
        return d

    def to_json(self, ensure_ascii: bool = False, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)

    @property
    def label(self) -> str:
        """人类可读的一行复核标签。"""
        if not self.is_violation:
            return "正常"
        tag = "涉诈" if self.is_fraud else "违规"
        cat = self.scene_category
        if self.scene_subtype:
            cat = f"{cat}/{self.scene_subtype}"
        return f"{tag}-{self.risk_level.value}-{cat}"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "InspectionResult":
        risk = RiskLevel.from_text(payload.get("risk_level"))
        is_violation = bool(payload.get("is_violation", risk.rank > 0))
        return cls(
            is_violation=is_violation,
            is_fraud=bool(payload.get("is_fraud", False)),
            risk_level=risk,
            scene_category=str(payload.get("scene_category", "正常") or "正常"),
            scene_subtype=str(payload.get("scene_subtype", "") or ""),
            explanation=str(payload.get("explanation", "") or ""),
            detected_features=list(payload.get("detected_features", []) or []),
            evidence_quotes=list(payload.get("evidence_quotes", []) or []),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            analysis_thought=str(payload.get("analysis_thought", "") or ""),
            source=str(payload.get("source", "llm") or "llm"),
            data_id=payload.get("data_id"),
            evidence_verified=payload.get("evidence_verified"),
        )
