"""知识库：质检规范（spec.md）+ 动态反诈规则（rules.json）。

体现 learn-claude-code 的『按需加载知识』：规范被切成可检索的小节，
模型在需要时通过工具拉取对应小节，而不是把整篇规范一次性塞进上下文。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .retrieval import TfidfIndex


@dataclass
class SpecSection:
    title: str
    level: int
    content: str
    path: List[str]  # 祖先标题链

    @property
    def full_title(self) -> str:
        return " / ".join(self.path + [self.title]) if self.path else self.title


def _parse_markdown_sections(md_text: str) -> List[SpecSection]:
    """按 markdown 标题切分为小节，记录标题层级与祖先链。"""
    lines = md_text.splitlines()
    sections: List[SpecSection] = []
    stack: List[str] = []  # (level, title) 简化为按 level 维护
    level_titles: Dict[int, str] = {}

    cur_title = "前言"
    cur_level = 0
    cur_path: List[str] = []
    buf: List[str] = []

    def flush():
        content = "\n".join(buf).strip()
        if cur_title.strip() and (content or sections == []):
            sections.append(
                SpecSection(
                    title=cur_title.strip(),
                    level=cur_level,
                    content=content,
                    path=list(cur_path),
                )
            )

    heading_re = re.compile(r"^(#{1,6})\s+(.*)$")
    for line in lines:
        m = heading_re.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip().strip("#").strip("*").strip()
            level_titles[level] = title
            for lv in list(level_titles.keys()):
                if lv > level:
                    level_titles.pop(lv, None)
            cur_path = [level_titles[lv] for lv in sorted(level_titles) if lv < level]
            cur_title = title
            cur_level = level
            buf = []
        else:
            buf.append(line)
    flush()
    return [s for s in sections if s.title]


class KnowledgeBase:
    def __init__(self, spec_path: Path, rules_path: Path):
        self.spec_path = Path(spec_path)
        self.rules_path = Path(rules_path)
        self.sections: List[SpecSection] = []
        self.rules: Dict[str, Any] = {}
        self._index = TfidfIndex()
        self._load()

    # ---------- 加载 ----------
    def _load(self) -> None:
        if self.spec_path.exists():
            md = self.spec_path.read_text(encoding="utf-8")
            self.sections = _parse_markdown_sections(md)
            corpus = [f"{s.full_title}\n{s.content}" for s in self.sections]
            if corpus:
                self._index.fit(corpus)
        self.rules = self.load_rules()

    def load_rules(self) -> Dict[str, Any]:
        if self.rules_path.exists():
            return json.loads(self.rules_path.read_text(encoding="utf-8"))
        return {"violation_scenarios": [], "fraud_scenarios": [], "evolved_examples": []}

    def save_rules(self, rules: Optional[Dict[str, Any]] = None) -> None:
        data = rules if rules is not None else self.rules
        self.rules_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.rules = data

    # ---------- 查询 ----------
    def search_spec(self, query: str, top_k: int = 3) -> List[SpecSection]:
        hits = self._index.search(query, top_k=top_k)
        return [self.sections[i] for i, _ in hits]

    def list_scenarios(self) -> Dict[str, List[str]]:
        return {
            "违规场景": [s.get("category", "") for s in self.rules.get("violation_scenarios", [])],
            "涉诈场景": [s.get("category", "") for s in self.rules.get("fraud_scenarios", [])],
        }

    def get_scenario(self, category: str) -> Optional[Dict[str, Any]]:
        for bucket in ("violation_scenarios", "fraud_scenarios"):
            for sc in self.rules.get(bucket, []):
                if sc.get("category") == category:
                    out = dict(sc)
                    out["_bucket"] = bucket
                    return out
        return None

    def all_keywords(self) -> Dict[str, List[str]]:
        """category -> keywords，供启发式回退使用。"""
        out: Dict[str, List[str]] = {}
        for bucket in ("violation_scenarios", "fraud_scenarios"):
            for sc in self.rules.get(bucket, []):
                out[sc.get("category", "")] = list(sc.get("keywords", []))
        return out

    def rules_brief(self) -> str:
        """生成注入 system prompt 的精简规则摘要（按需加载的『总览索引』）。"""
        lines: List[str] = []
        lines.append(f"判定原则：{self.rules.get('principle', '')}")
        lines.append(f"输出契约：{self.rules.get('output_contract', '')}")
        lines.append("【违规场景及风险等级】")
        for sc in self.rules.get("violation_scenarios", []):
            lines.append(f"- {sc.get('category')}：{sc.get('judgment_method', '')}")
            for r in sc.get("risk_rules", []):
                lines.append(f"    · {r}")
        lines.append("【涉诈场景（均为高风险，需标注为诈骗）】")
        for sc in self.rules.get("fraud_scenarios", []):
            lines.append(f"- {sc.get('category')}：{sc.get('judgment_method', '')}")
        table = self.rules.get("business_decision_table", [])
        if table:
            lines.append("【业务口径判定表（最高优先级，命中即按此输出 category/subtype/risk_level）】")
            for row in table:
                sub = row.get("subtype") or "-"
                lines.append(
                    f"- {row.get('场景')} => {row.get('判定')}｜category={row.get('category')}"
                    f"｜subtype={sub}｜risk_level={row.get('risk_level')}"
                )
        disambig = self.rules.get("disambiguation", [])
        if disambig:
            lines.append("【易混场景子类目判别（重要，先按此消歧再定类目）】")
            for d in disambig:
                lines.append(f"- {d}")
        evolved = self.rules.get("evolved_examples", [])
        if evolved:
            lines.append("【已自动演进沉淀的边缘案例（错题本）】")
            for ex in evolved[-5:]:
                lines.append(f"- {ex.get('label', '')}: {ex.get('analysis', '')}")
        return "\n".join(lines)
