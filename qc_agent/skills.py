"""场景技能库（Agent Skills / 渐进式披露）。

learn-claude-code 的 skills 思路：system prompt 只常驻一份**技能目录**
（每个场景一行触发词+描述），每次质检按通话内容**路由**出 top-k 个技能，
把它们的完整判定知识（判断方法/风险规则/消歧/错题本）随待检文本注入。

收益：
- 规则碰撞面从『全局×全局』缩小到『同通话命中的技能之间』；
- 演进（错题本）按类目落盘到对应技能文件，不污染无关场景；
- system prompt 保持短且字节稳定，利于 LLM 前缀缓存。

技能文件格式（knowledge/skills/*.md）：
    ---
    name: 手机租赁套路贷诈骗
    bucket: 涉诈场景
    risk: 高风险
    triggers: 租赁, 芝麻信用分, 回收, 买断
    description: 租手机/商品下单变现做资金周转
    ---
    ## 判断方法
    ...
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .retrieval import TfidfIndex

_ERRATA_START = "<!-- errata:start -->"
_ERRATA_END = "<!-- errata:end -->"


@dataclass
class Skill:
    name: str
    bucket: str = ""          # 违规场景 | 涉诈场景
    risk: str = ""            # 默认风险等级描述
    triggers: List[str] = field(default_factory=list)
    description: str = ""
    body: str = ""            # frontmatter 之后的完整正文
    path: Optional[Path] = None

    def catalog_line(self) -> str:
        trig = "、".join(self.triggers[:8])
        return f"- {self.name}｜{self.bucket}·{self.risk}｜触发词：{trig}｜{self.description}"


def parse_skill_md(text: str, path: Optional[Path] = None) -> Optional[Skill]:
    """解析带 frontmatter 的技能文件（零依赖的 YAML-lite：仅 key: value 行）。"""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.DOTALL)
    if not m:
        return None
    meta_text, body = m.group(1), m.group(2).strip()
    meta: Dict[str, str] = {}
    for line in meta_text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    name = meta.get("name", "").strip()
    if not name:
        return None
    triggers_raw = meta.get("triggers", "").strip().strip("[]")
    triggers = [t.strip() for t in re.split(r"[,，、]", triggers_raw) if t.strip()]
    return Skill(
        name=name,
        bucket=meta.get("bucket", ""),
        risk=meta.get("risk", ""),
        triggers=triggers,
        description=meta.get("description", ""),
        body=body,
        path=path,
    )


class SkillLibrary:
    """技能加载、目录生成与按通话内容路由。"""

    def __init__(self, skills_dir: Path):
        self.skills_dir = Path(skills_dir)
        self.skills: List[Skill] = []
        self._index = TfidfIndex()
        self._load()

    def _load(self) -> None:
        self.skills = []
        if not self.skills_dir.is_dir():
            return
        for path in sorted(self.skills_dir.glob("*.md")):
            skill = parse_skill_md(path.read_text(encoding="utf-8"), path=path)
            if skill:
                self.skills.append(skill)
        if self.skills:
            corpus = [
                f"{s.name}\n{' '.join(s.triggers)}\n{s.description}\n{s.body}"
                for s in self.skills
            ]
            self._index.fit(corpus)

    def __len__(self) -> int:
        return len(self.skills)

    @property
    def available(self) -> bool:
        return bool(self.skills)

    def get(self, name: str) -> Optional[Skill]:
        for s in self.skills:
            if s.name == name:
                return s
        return None

    def catalog(self) -> str:
        """system prompt 常驻的技能目录（每技能一行）。"""
        return "\n".join(s.catalog_line() for s in self.skills)

    def route(self, text: str, top_k: int = 3) -> List[Tuple[Skill, float]]:
        """按通话文本路由技能：触发词命中数 + 正文 TF-IDF 相似度加权。"""
        if not self.skills or not (text or "").strip():
            return []
        tfidf: Dict[int, float] = dict(self._index.search(text, top_k=len(self.skills)))
        scored: List[Tuple[int, float]] = []
        for i, s in enumerate(self.skills):
            trig_hits = sum(1 for t in s.triggers if t and t in text)
            score = min(trig_hits, 5) * 1.0 + tfidf.get(i, 0.0) * 3.0
            if score > 0:
                scored.append((i, score))
        scored.sort(key=lambda x: -x[1])
        return [(self.skills[i], sc) for i, sc in scored[:top_k]]

    def render(self, skills: List[Skill], max_chars_each: int = 2200) -> str:
        """将路由命中的技能渲染为注入块。"""
        parts = []
        for s in skills:
            body = s.body
            if len(body) > max_chars_each:
                body = body[:max_chars_each] + "…"
            parts.append(f"### 技能：{s.name}（{s.bucket}·{s.risk}）\n{body}")
        return "\n\n".join(parts)

    def append_errata(self, name: str, line: str) -> bool:
        """把一条错题/边缘案例写入对应技能文件的错题本区（演进定向落盘）。"""
        skill = self.get(name)
        if not skill or not skill.path or not skill.path.exists():
            return False
        text = skill.path.read_text(encoding="utf-8")
        if _ERRATA_START not in text:
            text = text.rstrip() + f"\n\n## 错题本（自动演进，人工复核后合入正式规则）\n{_ERRATA_START}\n{_ERRATA_END}\n"
        entry = f"- {line.strip()}"
        if entry in text:
            return True
        text = text.replace(_ERRATA_END, f"{entry}\n{_ERRATA_END}")
        skill.path.write_text(text, encoding="utf-8")
        self._load()
        return True

    def digest(self) -> str:
        """技能内容摘要（纳入缓存指纹：技能变更后旧缓存自动失效）。"""
        import hashlib

        h = hashlib.sha1()
        for s in self.skills:
            h.update(s.name.encode("utf-8"))
            h.update(s.body.encode("utf-8"))
        return h.hexdigest()[:10]
