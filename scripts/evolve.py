#!/usr/bin/env python3
"""自治演进 CLI：跑反射 Agent，从人工标注语料中自动沉淀新规则。

用法：
    python scripts/evolve.py --csv data/sample_cases.csv --limit 200
运行后 knowledge/rules.json 会被自动更新（新增高危词 + 错题本）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc_agent import Config, QcAgent  # noqa: E402
from qc_agent.case_store import CaseStore  # noqa: E402
from qc_agent.reflect import ReflectAgent  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="重庆行业卡反诈语音质检 - 自治演进")
    parser.add_argument("--csv", help="人工标注 CSV（默认用配置中的 cases 路径）")
    parser.add_argument("--limit", type=int, help="只处理前 N 条")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = Config()
    csv_path = Path(args.csv) if args.csv else config.cases_path
    store = CaseStore(csv_path)
    agent = QcAgent(config=config, cases=store)
    reflector = ReflectAgent(agent)

    before = len(agent.kb.rules.get("evolved_examples", []))
    print(f"质检模式：{agent.mode} | 样本数：{len(store)} | 开始自治演进...")
    stats = reflector.evolve_from_cases(store, limit=args.limit, verbose=not args.quiet)
    after = len(agent.kb.rules.get("evolved_examples", []))

    print("=" * 50)
    print(f"处理样本：{stats['total']}  判定分歧：{stats['conflicts']}  成功演进：{stats['evolved']}")
    print(f"错题本沉淀：{before} -> {after}")
    print(f"规则库已更新：{config.rules_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
