#!/usr/bin/env python3
"""批量质检 + 与人工标签对比评估。

用法：
    python scripts/batch_eval.py --csv data/sample_cases.csv --out results.csv
评估口径：以 CSV 的 comment 非空作为『人工判定为违规/涉诈』的代理标签，
统计违规检出的精确率/召回率/F1（粗粒度），并导出逐条结果。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc_agent import Config, QcAgent  # noqa: E402
from qc_agent.case_store import CaseStore  # noqa: E402
from qc_agent.reflect import expected_violation_from_comment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="重庆行业卡反诈语音质检 - 批量评估")
    parser.add_argument("--csv", help="输入 CSV（默认用配置中的 cases 路径）")
    parser.add_argument("--out", default="results.csv", help="输出结果 CSV")
    parser.add_argument("--limit", type=int, help="只评估前 N 条")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    config = Config()
    csv_path = Path(args.csv) if args.csv else config.cases_path
    if not csv_path.exists():
        print(f"错误：找不到数据文件 {csv_path}", file=sys.stderr)
        return 2

    store = CaseStore(csv_path)
    agent = QcAgent(config=config, cases=store, verbose=args.verbose)
    print(f"质检模式：{agent.mode} | 样本数：{len(store)} | 数据：{csv_path}")

    items = store.cases[: args.limit] if args.limit else store.cases
    tp = fp = fn = tn = 0
    rows = []
    for i, case in enumerate(items, 1):
        res = agent.inspect(case.content, data_id=case.data_id)
        expected = expected_violation_from_comment(case.comment)
        pred = res.is_violation
        if pred and expected:
            tp += 1
        elif pred and not expected:
            fp += 1
        elif not pred and expected:
            fn += 1
        else:
            tn += 1
        rows.append(
            {
                "data_id": case.data_id,
                "human_comment": case.comment,
                "pred_label": res.label,
                "is_violation": res.is_violation,
                "is_fraud": res.is_fraud,
                "risk_level": res.risk_level.value,
                "scene_category": res.scene_category,
                "scene_subtype": res.scene_subtype,
                "explanation": res.explanation,
                "confidence": round(res.confidence, 3),
            }
        )
        if args.verbose:
            print(f"[{i}/{len(items)}] {case.data_id} 人工『{case.comment}』 -> {res.label}")

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["data_id"])
        writer.writeheader()
        writer.writerows(rows)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print("=" * 50)
    print("违规检出评估（comment 非空 = 人工违规）：")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")
    print(f"结果已写入：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
