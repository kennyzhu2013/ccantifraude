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
from .skills import SkillLibrary


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


def _disambig_title(entry: str) -> str:
    """提取消歧条目短标题，作为 system prompt 常驻索引（正文按需注入）。"""
    if entry.startswith("【"):
        end = entry.find("】")
        if end > 0:
            return entry[: end + 1]
    head = entry.split("：", 1)[0]
    return head if len(head) <= 40 else head[:40] + "…"


class KnowledgeBase:
    def __init__(
        self,
        spec_path: Path,
        rules_path: Path,
        extra_spec_paths: Optional[List[Path]] = None,
        skills_dir: Optional[Path] = None,
    ):
        self.spec_path = Path(spec_path)
        self.rules_path = Path(rules_path)
        self.extra_spec_paths = [Path(p) for p in (extra_spec_paths or [])]
        self.sections: List[SpecSection] = []
        self.rules: Dict[str, Any] = {}
        self._index = TfidfIndex()
        self._disambig_index = TfidfIndex()
        self.skills: Optional[SkillLibrary] = (
            SkillLibrary(skills_dir) if skills_dir else None
        )
        self._load()

    @property
    def skills_available(self) -> bool:
        return self.skills is not None and self.skills.available

    # ---------- 加载 ----------
    def _load(self) -> None:
        self.sections = []
        paths = [self.spec_path] + self.extra_spec_paths
        for path in paths:
            if not path.exists():
                continue
            md = path.read_text(encoding="utf-8")
            parsed = _parse_markdown_sections(md)
            # 额外规范文件加前缀，避免与 V1.1 同名小节冲突。
            if path != self.spec_path:
                prefix = path.stem
                for s in parsed:
                    s.path = [prefix] + list(s.path)
            self.sections.extend(parsed)
        corpus = [f"{s.full_title}\n{s.content}" for s in self.sections]
        if corpus:
            self._index.fit(corpus)
        self.rules = self.load_rules()
        # 消歧规则单独建索引：正文不再全量常驻 prompt，而是按通话文本检索命中后注入。
        disambig = self.rules.get("disambiguation", [])
        if disambig:
            self._disambig_index.fit(disambig)

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

    def relevant_disambiguation(self, text: str, top_k: int = 6) -> List[str]:
        """检索与通话文本最相关的消歧规则全文（分层注入的『按需加载』部分）。"""
        entries = list(self.rules.get("disambiguation", []))
        if not entries or not (text or "").strip():
            return []
        hits = self._disambig_index.search(text, top_k=top_k)
        return [entries[i] for i, _ in hits]

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
                    # 演进暂存区未经人工审核，不暴露给 LLM 工具/启发式消费。
                    out.pop("candidate_keywords", None)
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

    def slim_brief(self) -> str:
        """skills 模式的 system prompt 常驻部分：全局不变量 + 技能目录。

        场景细则（判断方法/风险规则/类目内消歧）不再全量常驻，由技能路由按需注入；
        业务口径判定表含大量『合规』行（正常场景守门），保持全局常驻。
        """
        lines: List[str] = []
        lines.append(f"判定原则：{self.rules.get('principle', '')}")
        lines.append(f"输出契约：{self.rules.get('output_contract', '')}")
        table = self.rules.get("business_decision_table", [])
        if table:
            lines.append("【业务口径判定表（最高优先级，命中即按此输出 category/subtype/risk_level）】")
            for row in table:
                sub = row.get("subtype") or "-"
                line = (
                    f"- {row.get('场景')} => {row.get('判定')}｜category={row.get('category')}"
                    f"｜subtype={sub}｜risk_level={row.get('risk_level')}"
                )
                if row.get("备注"):
                    line += f"｜备注：{row['备注']}"
                lines.append(line)
        if self.skills_available:
            lines.append(
                "【场景技能目录（17 个判定技能；与本通话相关技能的完整判定规则会随待检文本注入，"
                "以注入的技能细则为准定类目与风险等级）】"
            )
            lines.append(self.skills.catalog())
        # 未归属任何技能的跨域消歧规则保持全局常驻（技能内已含各自类目的消歧）。
        skill_names = {s.name for s in self.skills.skills} if self.skills_available else set()
        orphan = [
            d for d in self.rules.get("disambiguation", [])
            if not any(name in d for name in skill_names)
        ]
        if orphan:
            lines.append("【通用消歧规则】")
            lines.extend(f"- {d}" for d in orphan)
        return "\n".join(lines)

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
                line = (
                    f"- {row.get('场景')} => {row.get('判定')}｜category={row.get('category')}"
                    f"｜subtype={sub}｜risk_level={row.get('risk_level')}"
                )
                note = row.get("备注")
                if note:
                    line += f"｜备注：{note}"
                lines.append(line)
        disambig = self.rules.get("disambiguation", [])
        if disambig:
            lines.append(
                "【易混场景子类目判别索引（仅列标题；与当前通话相关条目的全文会随待检文本一并给出，先按其消歧再定类目）】"
            )
            for d in disambig:
                lines.append(f"- {_disambig_title(d)}")
        evolved = self.rules.get("evolved_examples", [])
        if evolved:
            lines.append("【已自动演进沉淀的边缘案例（错题本）】")
            for ex in evolved[-5:]:
                lines.append(f"- {ex.get('label', '')}: {ex.get('analysis', '')}")
        return "\n".join(lines)
