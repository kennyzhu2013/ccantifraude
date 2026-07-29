#!/usr/bin/env python3
"""分析 batch_eval 输出 CSV 中质量门控信号的触发率与命中率。

用法：
    python scripts/analyze_flags.py --csv results.csv
    python scripts/analyze_flags.py --csv results.csv --show-samples
统计口径：
  - 触发率：触发该信号的行数 / 总行数。
  - 命中率：触发该信号的样本中判错的比例（判错 = 违规判定错 / 类目错 / 涉诈判定错，
    与 batch_eval._report 口径一致）。
  - 基线错误率：未触发任何信号样本的判错比例，用于对比信号是否真的富集了错误。
用途：依据各信号的实际触发率/命中率微调 QC_ROUTE_FRAUD_SIGNAL_MIN、
QC_KNN_SIGNAL_MIN_SIM、QC_SOFT_CONFIDENCE_BELOW 等门控阈值。
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc_agent.labels import is_compliant_label  # noqa: E402

# 采样一致维持原判的软信号带此前缀（见 agent._quality_gated_fast）。
PASSED_PREFIX = "[已复核通过] "

# 信号文案 -> 信号类型（与 agent._hard_signals / _soft_signals 的输出文案对应）。
SIGNAL_PATTERNS = [
    ("硬信号:证据未命中原文", r"证据引用未在原文命中"),
    ("硬信号:启发式涉诈冲突", r"启发式命中涉诈关键词"),
    ("硬信号:JSON修复兜底", r"输出经JSON修复/兜底"),
    ("硬信号:置信度低于硬阈值", r"低于硬阈值"),
    ("软信号:kNN判例冲突", r"高相似历史判例"),
    ("软信号:涉诈路由冲突", r"路由到涉诈技能"),
    ("软信号:低自报置信度", r"自报置信度"),
    ("采样分歧升级", r"自一致性采样结论不一致"),
]

_ROUTE_SCORE = re.compile(r"score=([0-9.]+)")


def classify(flag: str) -> str:
    for name, pattern in SIGNAL_PATTERNS:
        if re.search(pattern, flag):
            return name
    return "其他"


def row_is_error(row: dict) -> bool:
    """样本是否判错：违规判定错 / 类目错 / 涉诈判定错，任一即算。"""
    comment = (row.get("human_comment") or "").strip()
    expected_v = bool(comment) and not is_compliant_label(comment)
    pred_v = str(row.get("is_violation", "")).strip().lower() == "true"
    if pred_v != expected_v:
        return True
    # 类目仅在人工标签给出可接受类目集时参与判错（与 batch_eval 一致）。
    if (row.get("acceptable_categories") or "").strip():
        if str(row.get("category_correct", "")).strip().lower() != "true":
            return True
    fraud_correct = str(row.get("fraud_correct", "")).strip()
    if fraud_correct and fraud_correct.lower() != "true":
        return True
    return False


def _rate(n: int, total: int) -> str:
    return f"{n}/{total} = {n / total:.3f}" if total else f"{n}/0 = -"


def main() -> int:
    parser = argparse.ArgumentParser(description="质量门控信号触发率/命中率统计")
    parser.add_argument("--csv", required=True, help="batch_eval 输出的结果 CSV")
    parser.add_argument("--show-samples", action="store_true", help="逐条列出触发信号的样本")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"错误：找不到结果文件 {csv_path}", file=sys.stderr)
        return 2
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("错误：结果文件为空", file=sys.stderr)
        return 2
    if "review_flags" not in rows[0]:
        print("错误：CSV 缺少 review_flags 列（需用带门控版本的 batch_eval 重新评估）", file=sys.stderr)
        return 2

    total = len(rows)
    flagged_rows = []  # (row, is_error, [(信号类型, 是否已复核通过, 原始文案)])
    unflagged_errors = 0
    unflagged_total = 0
    # 每种信号 -> 统计桶
    stats = defaultdict(lambda: {"rows": 0, "errors": 0, "passed": 0, "escalated": 0})
    route_scores = {"判错": [], "判对": []}

    for row in rows:
        raw = (row.get("review_flags") or "").strip()
        err = row_is_error(row)
        if not raw:
            unflagged_total += 1
            unflagged_errors += 1 if err else 0
            continue
        parsed = []
        for flag in raw.split("；"):
            flag = flag.strip()
            if not flag:
                continue
            passed = flag.startswith(PASSED_PREFIX)
            text = flag[len(PASSED_PREFIX):] if passed else flag
            parsed.append((classify(text), passed, text))
        flagged_rows.append((row, err, parsed))
        # 同一样本同种信号只计一行；升级/复核通过按信号粒度记录。
        seen = set()
        for name, passed, text in parsed:
            if name not in seen:
                seen.add(name)
                stats[name]["rows"] += 1
                stats[name]["errors"] += 1 if err else 0
            if passed:
                stats[name]["passed"] += 1
            else:
                stats[name]["escalated"] += 1
            if name == "软信号:涉诈路由冲突":
                m = _ROUTE_SCORE.search(text)
                if m:
                    route_scores["判错" if err else "判对"].append(float(m.group(1)))

    flagged_total = len(flagged_rows)
    flagged_errors = sum(1 for _, err, _ in flagged_rows if err)
    escalated_total = sum(
        1 for row, _, _ in flagged_rows if "escalated" in str(row.get("source", ""))
    )

    print("=" * 64)
    print(f"样本总数：{total} | 数据：{csv_path}")
    print(f"总触发率（review_flags 非空）：{_rate(flagged_total, total)}")
    print(f"实际升级 tool loop（source=llm-escalated）：{escalated_total} 条")
    print(f"触发样本命中率（判错比例）  ：{_rate(flagged_errors, flagged_total)}")
    print(f"未触发样本基线错误率        ：{_rate(unflagged_errors, unflagged_total)}")
    print("-" * 64)
    print(f"{'信号类型':<18}{'触发率':>16}{'命中率':>16}{'升级':>6}{'复核通过':>8}")
    # 按预定义顺序输出，未触发的信号也列出便于对比。
    for name, _ in SIGNAL_PATTERNS:
        s = stats.get(name)
        if s is None:
            print(f"{name:<20}{_rate(0, total):>18}{'-':>16}{0:>6}{0:>9}")
            continue
        print(
            f"{name:<20}{_rate(s['rows'], total):>18}"
            f"{_rate(s['errors'], s['rows']):>18}{s['escalated']:>6}{s['passed']:>9}"
        )
    if "其他" in stats:
        s = stats["其他"]
        print(
            f"{'其他':<20}{_rate(s['rows'], total):>18}"
            f"{_rate(s['errors'], s['rows']):>18}{s['escalated']:>6}{s['passed']:>9}"
        )

    if route_scores["判错"] or route_scores["判对"]:
        print("-" * 64)
        print("涉诈路由冲突分数分布（用于微调 QC_ROUTE_FRAUD_SIGNAL_MIN）：")
        for tag, scores in route_scores.items():
            if scores:
                print(
                    f"  {tag}：n={len(scores)} min={min(scores):.2f} "
                    f"max={max(scores):.2f} mean={sum(scores) / len(scores):.2f}"
                )

    if args.show_samples and flagged_rows:
        print("-" * 64)
        print("触发信号的样本明细：")
        for row, err, parsed in flagged_rows:
            mark = "×判错" if err else "√判对"
            flags = "；".join(
                (PASSED_PREFIX + t if p else t) for _, p, t in parsed
            )
            print(f"  [{mark}] {row.get('data_id', '')} | {row.get('source', '')} | {flags}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
