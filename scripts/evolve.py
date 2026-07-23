#!/usr/bin/env python3
"""标签治理 / 自治演进 CLI。

默认（安全模式）：扫描全量语料，导出『规范判定 vs 人工标签』冲突样本到 CSV，
供人工做标签治理。鉴于真实人工标签存在噪声，默认【不】自动修改 rules.json。

    python scripts/evolve.py --csv data/real_cases.csv --out conflicts.csv --cache

如需让反射 Agent 自动把冲突沉淀为新规则（候选高危词 + 错题本），显式加 --apply：

    python scripts/evolve.py --csv data/real_cases.csv --apply

演进词仅进入 candidate_keywords 暂存区，不影响线上判定；人工审核后晋升/丢弃：

    python scripts/evolve.py --list-candidates                     # 查看待审核候选词
    python scripts/evolve.py --promote 贷款相关                    # 晋升该类目全部候选词
    python scripts/evolve.py --promote 贷款相关 --kw 砍头息 提前收费  # 只晋升指定词
    python scripts/evolve.py --discard 贷款相关                    # 丢弃该类目全部候选词
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
from qc_agent.knowledge_base import KnowledgeBase  # noqa: E402
from qc_agent.reflect import (  # noqa: E402
    ReflectAgent,
    list_candidate_keywords,
    review_candidates,
)


def _write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
    parser.add_argument("--apply", action="store_true", help="额外让反射 Agent 自动演进 rules.json（候选词进暂存区，默认不改）")
    parser.add_argument("--list-candidates", action="store_true", help="列出待审核的演进候选关键词后退出")
    parser.add_argument("--promote", metavar="类目", help="审核通过：把该类目候选词晋升进生产 keywords 后退出")
    parser.add_argument("--discard", metavar="类目", help="审核不通过：丢弃该类目候选词后退出")
    parser.add_argument("--kw", nargs="*", help="配合 --promote/--discard：只处理指定候选词（缺省全部）")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = Config()

    # ---- 候选词审核子命令（不需要语料/模型，直接操作 rules.json 后退出）----
    if args.list_candidates or args.promote or args.discard:
        kb = KnowledgeBase(config.spec_path, config.rules_path)
        if args.list_candidates:
            cands = list_candidate_keywords(kb)
            if not cands:
                print("暂存区为空：没有待审核的演进候选关键词。")
            for cat, kws in cands.items():
                print(f"  {cat}：{' / '.join(kws)}")
            return 0
        category = args.promote or args.discard
        promote = bool(args.promote)
        processed = review_candidates(kb, category, keywords=args.kw or None, promote=promote)
        action = "晋升进生产 keywords" if promote else "丢弃"
        if processed:
            print(f"【{category}】已{action} {len(processed)} 个候选词：{' / '.join(processed)}")
        else:
            print(f"【{category}】无匹配的候选词可处理。")
        return 0

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
        _write_csv(out_path, conflicts)

    print("=" * 56)
    print(f"扫描样本：{result['total']}  冲突样本：{len(conflicts)}  "
          f"（占比 {len(conflicts) / max(result['total'],1):.1%}）")

    # 按桶分文件导出（招商加盟回标合规 / 真违规 / 待人工复核）。
    bucket_counter = Counter(c["bucket"] for c in conflicts)
    print("\n分桶分布（用于下发回标）：")
    for b, n in sorted(bucket_counter.items()):
        bucket_rows = [c for c in conflicts if c["bucket"] == b]
        bucket_path = out_path.with_name(f"{out_path.stem}_{b}{out_path.suffix}")
        _write_csv(bucket_path, bucket_rows)
        print(f"  {n:4d}  {b}\n        -> {bucket_path}")

    type_counter = Counter(c["conflict_type"] for c in conflicts)
    print("\n冲突类型分布：")
    for t, n in type_counter.most_common():
        print(f"  {n:4d}  {t}")
    label_counter = Counter(c["human_comment"] for c in conflicts)
    print("冲突最多的人工标签 Top10：")
    for t, n in label_counter.most_common(10):
        print(f"  {n:4d}  {t[:40]}")
    if conflicts:
        print(f"\n全部冲突已导出：{out_path}（A 桶建议直接回标为合规；B 桶核对类目/涉诈；C 桶人工再确认）")

    if args.apply:
        print("\n--apply：让反射 Agent 自动演进 rules.json ...")
        stats = reflector.evolve_from_cases(store, limit=args.limit, verbose=not args.quiet)
        print(f"演进：判定分歧 {stats['conflicts']}，成功固化 {stats['evolved']} 条到 {config.rules_path}")
        print("提示：新提炼的关键词仅进入 candidate_keywords 暂存区，不影响线上判定；"
              "用 --list-candidates 查看，--promote/--discard 审核。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
