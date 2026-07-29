#!/usr/bin/env python3
"""批量质检 + 与人工标签对比评估。

用法：
    python scripts/batch_eval.py --csv data/real_cases.csv --sample 60 --workers 6
    python scripts/batch_eval.py --csv data/real_cases.csv --dedup --cache   # 降本：去重+缓存
评估指标：
  - 违规检出：以 comment 非空为人工违规代理标签，统计 P/R/F1。
  - 类目准确率：预测 scene_category 是否落入人工标签归一化后的可接受类目集。
  - 涉诈准确率：人工标签可判定涉诈时，比对 is_fraud。
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc_agent import Config, QcAgent  # noqa: E402
from qc_agent.case_store import CaseStore  # noqa: E402
from qc_agent.dedup import cluster_texts, group_by_cluster  # noqa: E402
from qc_agent.labels import (  # noqa: E402
    category_matches,
    expected_is_fraud,
    is_compliant_label,
    normalize_label,
)
from qc_agent.schema import InspectionResult  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="重庆行业卡反诈语音质检 - 批量评估")
    parser.add_argument("--csv", help="输入 CSV（默认用配置中的 cases 路径）")
    parser.add_argument("--out", default="results.csv", help="输出结果 CSV")
    parser.add_argument("--limit", type=int, help="只评估前 N 条")
    parser.add_argument("--sample", type=int, help="均匀抽样 N 条评估（优先于 limit）")
    parser.add_argument("--workers", type=int, help="并发数（默认取配置 batch_concurrency）")
    parser.add_argument("--tools", action="store_true", help="使用完整 agentic tool loop（更慢更准）")
    parser.add_argument("--dedup", action="store_true", help="近重复聚类去重，仅对代表样本调用 LLM")
    parser.add_argument("--dedup-threshold", type=float, default=0.9, help="去重相似度阈值")
    parser.add_argument("--cache", action="store_true", help="启用结果缓存（按内容哈希）")
    parser.add_argument("--cache-path", default=".cache/qc_results.jsonl", help="缓存文件路径")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    config = Config()
    if args.cache and not config.cache_path:
        config.cache_path = args.cache_path
    csv_path = Path(args.csv) if args.csv else config.cases_path
    if not csv_path.exists():
        print(f"错误：找不到数据文件 {csv_path}", file=sys.stderr)
        return 2

    store = CaseStore(csv_path)
    agent = QcAgent(config=config, cases=store, verbose=args.verbose)
    use_tools = True if args.tools else config.use_tools
    workers = args.workers or config.batch_concurrency

    if args.sample and args.sample < len(store):
        step = max(1, len(store) // args.sample)
        items = store.cases[::step][: args.sample]
    else:
        items = store.cases[: args.limit] if args.limit else store.cases

    print(
        f"质检模式：{agent.mode} | 工具模式：{use_tools} | 并发：{workers} | "
        f"去重：{args.dedup} | 缓存：{bool(config.cache_path)} | "
        f"样本：{len(items)}/{len(store)} | 数据：{csv_path}"
    )

    def run_one(case):
        return case, agent.inspect(case.content, data_id=case.data_id, use_tools=use_tools)

    # 去重：先聚类，仅对代表样本调用 LLM，其余复用代表结论。
    if args.dedup:
        cluster_of = cluster_texts([c.content for c in items], threshold=args.dedup_threshold)
        groups = group_by_cluster(cluster_of)
        reps = list(groups.keys())
        print(f"去重：{len(items)} 条 -> {len(reps)} 个代表簇（节省 {len(items) - len(reps)} 次 LLM 调用）")
    else:
        cluster_of = list(range(len(items)))
        groups = {i: [i] for i in range(len(items))}
        reps = list(range(len(items)))

    rep_results = {}
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, items[r]): r for r in reps}
        for fut in as_completed(futures):
            r = futures[fut]
            _, res = fut.result()
            rep_results[r] = res
            done += 1
            if args.verbose or done % 10 == 0:
                print(f"  [{done}/{len(reps)}] 代表 {items[r].data_id} -> {res.label}", flush=True)
            if done % 50 == 0:
                agent.flush_cache()
    elapsed = time.time() - t0
    agent.flush_cache()

    rows = []
    for idx, case in enumerate(items):
        rep = cluster_of[idx]
        base = rep_results[rep]
        if rep == idx:
            res = base
        else:
            res = InspectionResult.from_dict(base.to_dict())
            res.data_id = case.data_id
            res.source = base.source + "(dedup)"
        # 人工标签明确为正常/合规时，违规判定已由 TP/TN 统计覆盖，
        # 不再参与类目/涉诈准确率（否则『正常，对本人催收』会因关键词
        # 『催收』被归一化出可接受类目，制造伪冲突）。
        if is_compliant_label(case.comment):
            acceptable, exp_fraud = set(), None
        else:
            acceptable = normalize_label(case.comment)
            exp_fraud = expected_is_fraud(case.comment)
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
                "acceptable_categories": "|".join(sorted(acceptable)),
                "category_correct": category_matches(res.scene_category, acceptable),
                "expected_fraud": "" if exp_fraud is None else exp_fraud,
                "fraud_correct": "" if exp_fraud is None else (res.is_fraud == exp_fraud),
                "explanation": res.explanation,
                "confidence": round(res.confidence, 3),
                "review_flags": "；".join(res.review_flags),
                "source": res.source,
            }
        )

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["data_id"])
        writer.writeheader()
        writer.writerows(rows)

    _report(rows, len(reps), elapsed)
    print(f"结果已写入：{out_path}")
    return 0


def _report(rows, llm_calls, elapsed):
    tp = fp = fn = tn = 0
    cat_total = cat_correct = 0
    fraud_total = fraud_correct = 0
    for r in rows:
        comment = str(r["human_comment"])
        expected_v = bool(comment.strip()) and not is_compliant_label(comment)
        pred_v = bool(r["is_violation"])
        if pred_v and expected_v:
            tp += 1
        elif pred_v and not expected_v:
            fp += 1
        elif not pred_v and expected_v:
            fn += 1
        else:
            tn += 1
        if r["acceptable_categories"]:
            cat_total += 1
            cat_correct += 1 if r["category_correct"] else 0
        if r["fraud_correct"] != "":
            fraud_total += 1
            fraud_correct += 1 if r["fraud_correct"] else 0

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print("=" * 56)
    print(f"耗时：{elapsed:.1f}s | LLM 实际调用：{llm_calls} 次 | 评估样本：{len(rows)} 条")
    print("违规检出（comment 非空 = 人工违规）：")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn} | P={precision:.3f} R={recall:.3f} F1={f1:.3f}")
    if cat_total:
        print(f"类目准确率：{cat_correct}/{cat_total} = {cat_correct / cat_total:.3f}")
    if fraud_total:
        print(f"涉诈判定准确率：{fraud_correct}/{fraud_total} = {fraud_correct / fraud_total:.3f}")
    print("=" * 56)


if __name__ == "__main__":
    raise SystemExit(main())
