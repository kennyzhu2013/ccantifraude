"""交叉实测：本项目启发式通道 vs hangyekaAntiFraud 规则引擎通道。

两个项目实现同一业务需求（重庆行业卡质检），确定性层走了不同路线：本项目用
关键词列表 + 自由文本口径（细则由 LLM 消费），hangyekaAntiFraud 用可执行的
YAML 组合规则 DSL（all/any/none + speaker + window）。

本脚本只跑两侧的**离线确定性通道**（不调用 LLM），用于量化二者的互补性：

  方向 A：hangyeka 规则引擎 跑本项目的 data/eval_fresh_cases.csv（手写回归集）
  方向 B：本项目启发式     跑 hangyeka 的 tests/fixtures/cases.json（黄金样本）

结论见 docs/与hangyekaAntiFraud项目对比与融合建议.md。

用法：
    git clone https://github.com/kennyzhu2013/hangyekaAntiFraud /tmp/hangyekaAntiFraud
    pip install pyyaml pydantic pydantic-settings   # hangyeka 侧依赖
    python3 scripts/cross_eval_hangyeka.py --repo /tmp/hangyekaAntiFraud
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

QC_ROOT = Path(__file__).resolve().parents[1]
if str(QC_ROOT) not in sys.path:
    sys.path.insert(0, str(QC_ROOT))

RISK_ORDER = {"合规": 0, "正常": 0, "低风险": 1, "中风险": 2, "高风险": 3}

Runner = Callable[[str], Dict]


def build_hangyeka_runner(repo: Path) -> Runner:
    sys.path.insert(0, str(repo))
    from app.schemas.audit import AuditRequest
    from app.services.preprocess import Normalizer, preprocess
    from app.services.rule_engine import RuleEngine, merge_hits

    engine = RuleEngine.from_yaml(repo / "app/knowledge/chongqing_rules.yaml")
    normalizer = Normalizer.from_yaml(repo / "app/knowledge/normalization.yaml")

    def run(transcript: str) -> Dict:
        ctx = preprocess(AuditRequest(transcript=transcript), normalizer)
        verdict = merge_hits(engine.evaluate(ctx.norm_turns))
        risk = verdict.candidate_risk or "正常"
        return {
            "category": verdict.candidate_category,
            "risk": risk,
            "is_fraud": bool(verdict.fraud_candidate),
            "is_violation": RISK_ORDER.get(risk, 0) > 0,
            "rule_ids": [h.rule_id for h in verdict.hits],
        }

    return run


def build_qc_runner() -> Runner:
    from qc_agent.heuristic import HeuristicInspector
    from qc_agent.knowledge_base import KnowledgeBase

    kb = KnowledgeBase(
        spec_path=QC_ROOT / "knowledge/spec.md",
        rules_path=QC_ROOT / "knowledge/rules.json",
        skills_dir=QC_ROOT / "knowledge/skills",
    )
    inspector = HeuristicInspector(kb)

    def run(transcript: str) -> Dict:
        res = inspector.inspect(transcript, include_similar=False)
        return {
            "category": res.scene_category,
            "risk": res.risk_level.value,
            "is_fraud": res.is_fraud,
            "is_violation": res.is_violation,
            "rule_ids": [],
        }

    return run


def direction_a(hy: Runner, qc: Runner) -> Tuple[Dict, List[Dict]]:
    """hangyeka 规则引擎跑本项目 fresh 回归集。"""
    from qc_agent import labels

    csv_path = QC_ROOT / "data/eval_fresh_cases.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    detail: List[Dict] = []
    for row in rows:
        comment = row["comment"]
        detail.append(
            {
                "data_id": row.get("data_id", ""),
                "comment": comment,
                "exp_violation": not labels.is_compliant_label(comment),
                "exp_fraud": labels.expected_is_fraud(comment),
                "acceptable": sorted(labels.normalize_label(comment)),
                "hy": hy(row["content"]),
                "qc": qc(row["content"]),
            }
        )
    return _summarize(detail, key_violation="exp_violation"), detail


def direction_b(hy: Runner, qc: Runner, repo: Path) -> Tuple[Dict, List[Dict]]:
    """本项目启发式跑 hangyeka 黄金样本（含合规负例）。"""
    cases = json.loads((repo / "tests/fixtures/cases.json").read_text(encoding="utf-8"))
    detail: List[Dict] = []
    for case in cases:
        exp_risk = case["expect_candidate_risk"]
        detail.append(
            {
                "data_id": case["name"],
                "comment": f"{exp_risk}/{case.get('expect_candidate_category')}",
                "exp_risk": exp_risk,
                "exp_violation": RISK_ORDER.get(exp_risk, 0) > 0,
                "exp_fraud": bool(case.get("expect_fraud_candidate")),
                "acceptable": (
                    [case["expect_candidate_category"]]
                    if case.get("expect_candidate_category")
                    else []
                ),
                "hy": hy(case["transcript"]),
                "qc": qc(case["transcript"]),
            }
        )
    return _summarize(detail, key_violation="exp_violation"), detail


def _summarize(detail: List[Dict], key_violation: str) -> Dict:
    pos = [d for d in detail if d[key_violation]]
    neg = [d for d in detail if not d[key_violation]]
    fraud_pos = [d for d in pos if d["exp_fraud"]]
    cat_pos = [d for d in pos if d["acceptable"]]

    def count(items, side, field):
        return sum(1 for d in items if d[side][field])

    return {
        "n": len(detail),
        "n_violation_pos": len(pos),
        "n_compliant_neg": len(neg),
        "n_fraud_pos": len(fraud_pos),
        "recall_violation": {
            "hangyeka": count(pos, "hy", "is_violation"),
            "qc": count(pos, "qc", "is_violation"),
            "union": sum(
                1 for d in pos if d["hy"]["is_violation"] or d["qc"]["is_violation"]
            ),
        },
        "false_positive_on_compliant": {
            "hangyeka": count(neg, "hy", "is_violation"),
            "qc": count(neg, "qc", "is_violation"),
            "union": sum(
                1 for d in neg if d["hy"]["is_violation"] or d["qc"]["is_violation"]
            ),
        },
        "recall_fraud": {
            "hangyeka": count(fraud_pos, "hy", "is_fraud"),
            "qc": count(fraud_pos, "qc", "is_fraud"),
            "union": sum(
                1 for d in fraud_pos if d["hy"]["is_fraud"] or d["qc"]["is_fraud"]
            ),
        },
        "category_correct": {
            "scored": len(cat_pos),
            "hangyeka": sum(
                1 for d in cat_pos if (d["hy"]["category"] or "") in d["acceptable"]
            ),
            "qc": sum(
                1 for d in cat_pos if (d["qc"]["category"] or "") in d["acceptable"]
            ),
        },
    }


def _print_report(title: str, stat: Dict, detail: List[Dict]) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(
        f"样本 {stat['n']}（违规正例 {stat['n_violation_pos']}，"
        f"合规负例 {stat['n_compliant_neg']}，涉诈正例 {stat['n_fraud_pos']}）"
    )
    for label, key in (
        ("违规召回", "recall_violation"),
        ("涉诈召回", "recall_fraud"),
        ("合规负例误报", "false_positive_on_compliant"),
    ):
        v = stat[key]
        print(
            f"  {label:<12} hangyeka={v['hangyeka']:2d}  本项目={v['qc']:2d}  并集={v['union']:2d}"
        )
    cat = stat["category_correct"]
    print(
        f"  {'类目正确':<12} hangyeka={cat['hangyeka']:2d}  本项目={cat['qc']:2d}"
        f"  （可评 {cat['scored']}）"
    )
    print("\n  --- 结论方向分歧逐条 ---")
    for d in detail:
        if (
            d["hy"]["is_violation"] != d["qc"]["is_violation"]
            or d["hy"]["is_fraud"] != d["qc"]["is_fraud"]
        ):
            print(
                f"    {d['data_id']:<30} 期望={d['comment'][:24]:<26} "
                f"HY={d['hy']['risk']}/{d['hy']['category']}/诈={d['hy']['is_fraud']}  "
                f"本={d['qc']['risk']}/{d['qc']['category']}/诈={d['qc']['is_fraud']}"
            )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default="/tmp/hangyekaAntiFraud",
        help="hangyekaAntiFraud 本地 clone 路径",
    )
    parser.add_argument("--out", default="", help="结果 JSON 输出路径（可选）")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if not (repo / "app/knowledge/chongqing_rules.yaml").exists():
        print(
            f"未找到 hangyekaAntiFraud 仓库：{repo}\n"
            "请先 clone：git clone https://github.com/kennyzhu2013/hangyekaAntiFraud "
            f"{repo}",
            file=sys.stderr,
        )
        return 2

    hy, qc = build_hangyeka_runner(repo), build_qc_runner()

    stat_a, detail_a = direction_a(hy, qc)
    _print_report("方向 A：hangyeka 规则引擎 跑本项目 data/eval_fresh_cases.csv", stat_a, detail_a)
    stat_b, detail_b = direction_b(hy, qc, repo)
    _print_report("方向 B：本项目启发式 跑 hangyeka tests/fixtures/cases.json", stat_b, detail_b)

    if args.out:
        payload = {
            "direction_a": {"stat": stat_a, "detail": detail_a},
            "direction_b": {"stat": stat_b, "detail": detail_b},
        }
        Path(args.out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"完整结果 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
