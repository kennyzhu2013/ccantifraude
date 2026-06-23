#!/usr/bin/env python3
"""标签治理 / 自治演进 CLI。

默认（安全模式）：扫描全量语料，导出『规范判定 vs 人工标签』冲突样本到 CSV，
供人工做标签治理。鉴于真实人工标签存在噪声，默认【不】自动修改 rules.json。

    python scripts/evolve.py --csv data/real_cases.csv --out conflicts.csv --cache

如需让反射 Agent 自动把冲突沉淀为新规则（高危词 + 错题本），显式加 --apply：

    python scripts/evolve.py --csv data/real_cases.csv --apply
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc_agent import Config, QcAgent  # noqa: E402
from qc_agent.case_store import CaseStore  # noqa: E402
from qc_agent.reflect import ReflectAgent  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="重庆行业卡反诈语音质检 - 标签治理/自治演进")
    parser.add_argument("--csv", help="人工标注 CSV（默认用配置中的 cases 路径）")
    parser.add_argument("--out", default="conflicts.csv", help="冲突样本导出路径")
    parser.add_argument("--limit", type=int, help="只处理前 N 条")
    parser.add_argument("--sample", type=int, help="均匀抽样 N 条")
    parser.add_argument("--workers", type=int, help="并发数")
    parser.add_argument("--tools", action="store_true", help="使用完整 agentic tool loop（更准更慢）")
    parser.add_argument("--cache", action="store_true", help="启用结果缓存")
    parser.add_argument("--cache-path", default=".cache/qc_results.jsonl")
    parser.add_argument("--apply", action="store_true", help="额外让反射 Agent 自动演进 rules.json（默认不改）")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = Config()
    if args.cache and not config.cache_path:
        config.cache_path = args.cache_path
    csv_path = Path(args.csv) if args.csv else config.cases_path
    if not csv_path.exists():
        print(f"错误：找不到数据文件 {csv_path}", file=sys.stderr)
        return 2

    store = CaseStore(csv_path)
    agent = QcAgent(config=config, cases=store)
    reflector = ReflectAgent(agent)
    workers = args.workers or config.batch_concurrency

    print(
        f"质检模式：{agent.mode} | 工具模式：{args.tools} | 并发：{workers} | "
        f"样本数：{len(store)} | 开始扫描『规范 vs 人工标签』冲突..."
    )
    result = reflector.scan_conflicts(
        store,
        use_tools=args.tools,
        workers=workers,
        limit=args.limit,
        sample=args.sample,
        verbose=not args.quiet,
    )
    conflicts = result["conflicts"]

    out_path = Path(args.out)
    if conflicts:
        with out_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(conflicts[0].keys()))
            writer.writeheader()
            writer.writerows(conflicts)

    print("=" * 56)
    print(f"扫描样本：{result['total']}  冲突样本：{len(conflicts)}  "
          f"（占比 {len(conflicts) / max(result['total'],1):.1%}）")
    type_counter = Counter(c["conflict_type"] for c in conflicts)
    print("冲突类型分布：")
    for t, n in type_counter.most_common():
        print(f"  {n:4d}  {t}")
    label_counter = Counter(c["human_comment"] for c in conflicts)
    print("冲突最多的人工标签 Top10：")
    for t, n in label_counter.most_common(10):
        print(f"  {n:4d}  {t[:40]}")
    if conflicts:
        print(f"\n冲突样本已导出：{out_path}（请人工复核：以规范判定为准还是修正人工标签）")

    if args.apply:
        print("\n--apply：让反射 Agent 自动演进 rules.json ...")
        stats = reflector.evolve_from_cases(store, limit=args.limit, verbose=not args.quiet)
        print(f"演进：判定分歧 {stats['conflicts']}，成功固化 {stats['evolved']} 条到 {config.rules_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
