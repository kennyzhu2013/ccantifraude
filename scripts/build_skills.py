#!/usr/bin/env python3
"""从 rules.json 生成/更新 knowledge/skills/*.md 技能文件。

- rules.json 仍是场景规则的编辑入口（启发式关键词、判定表等继续消费它）；
- 本脚本把每个场景转换为一个技能文件：frontmatter（触发词/描述）+ 正文
  （判断方法、风险规则、该类目相关的业务口径判定行与消歧规则）；
- 幂等：重复运行输出一致；技能文件中的错题本区（errata 标记之间）会被保留，
  供反射 Agent 定向落盘、人工复核后再合入 rules.json。

用法：python scripts/build_skills.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qc_agent.skills import _ERRATA_END, _ERRATA_START  # noqa: E402

RULES_PATH = ROOT / "knowledge" / "rules.json"
SKILLS_DIR = ROOT / "knowledge" / "skills"


def _slug(name: str) -> str:
    return re.sub(r"[/\\:*?\"<>|\s]+", "_", name)


def _first_sentence(text: str, limit: int = 60) -> str:
    head = re.split(r"[。；;]", text or "", 1)[0].strip()
    return head if len(head) <= limit else head[:limit] + "…"


def _extract_errata(existing: str) -> str:
    m = re.search(re.escape(_ERRATA_START) + r"(.*?)" + re.escape(_ERRATA_END), existing, re.DOTALL)
    return m.group(1).strip("\n") if m else ""


def build_skill_md(sc: dict, bucket: str, rules: dict, errata: str) -> str:
    name = sc["category"]
    risk = sc.get("risk_level") or "按风险规则分级"
    triggers = [k for k in sc.get("keywords", []) if k][:16]
    description = _first_sentence(sc.get("judgment_method", ""))

    lines = [
        "---",
        f"name: {name}",
        f"bucket: {bucket}",
        f"risk: {risk}",
        f"triggers: {', '.join(triggers)}",
        f"description: {description}",
        "---",
        "",
        "## 判断方法",
        "",
        sc.get("judgment_method", "").strip(),
    ]

    risk_rules = sc.get("risk_rules", [])
    if risk_rules:
        lines += ["", "## 风险等级规则", ""]
        lines += [f"- {r}" for r in risk_rules]

    subtypes = sc.get("subtypes", {})
    if subtypes:
        lines += ["", "## 标准子类目", ""]
        lines += [f"- {k}：{v}" for k, v in subtypes.items()]
    compliant = sc.get("compliant_subtypes", [])
    if compliant:
        lines += ["", "## 合规情形（不判本类违规）", ""]
        lines += [f"- {c}" for c in compliant]

    table_rows = [
        row for row in rules.get("business_decision_table", [])
        if row.get("category") == name or name in (row.get("场景") or "")
    ]
    if table_rows:
        lines += ["", "## 业务口径判定（最高优先级）", ""]
        for row in table_rows:
            sub = row.get("subtype") or "-"
            line = (
                f"- {row.get('场景')} => {row.get('判定')}｜category={row.get('category')}"
                f"｜subtype={sub}｜risk_level={row.get('risk_level')}"
            )
            if row.get("备注"):
                line += f"｜备注：{row['备注']}"
            lines.append(line)

    disambig = [d for d in rules.get("disambiguation", []) if name in d]
    if disambig:
        lines += ["", "## 易混场景消歧（先按此消歧再定类目）", ""]
        lines += [f"- {d}" for d in disambig]

    lines += [
        "",
        "## 错题本（自动演进，人工复核后合入正式规则）",
        _ERRATA_START,
    ]
    if errata:
        lines.append(errata)
    lines += [_ERRATA_END, ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="从 rules.json 生成技能文件")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)

    expected_files = set()
    changed = 0
    for bucket_key, bucket_name in (
        ("violation_scenarios", "违规场景"),
        ("fraud_scenarios", "涉诈场景"),
    ):
        for sc in rules.get(bucket_key, []):
            name = sc.get("category", "")
            if not name:
                continue
            path = SKILLS_DIR / f"{_slug(name)}.md"
            expected_files.add(path.name)
            errata = _extract_errata(path.read_text(encoding="utf-8")) if path.exists() else ""
            content = build_skill_md(sc, bucket_name, rules, errata)
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                changed += 1
                if args.dry_run:
                    print(f"[dry-run] would write {path.name}")
                else:
                    path.write_text(content, encoding="utf-8")
                    print(f"wrote {path.name}")

    # 清理 rules.json 中已不存在的类目对应的技能文件。
    for path in SKILLS_DIR.glob("*.md"):
        if path.name not in expected_files:
            if args.dry_run:
                print(f"[dry-run] would remove stale {path.name}")
            else:
                path.unlink()
                print(f"removed stale {path.name}")
                changed += 1

    print(f"done: {len(expected_files)} skills, {changed} changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
