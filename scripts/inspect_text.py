#!/usr/bin/env python3
"""单通通话质检 CLI。

用法：
    python scripts/inspect_text.py "left:喂你好，我是*投顾客服..."
    python scripts/inspect_text.py --file call.txt
    echo "left:..." | python scripts/inspect_text.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc_agent import QcAgent  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="重庆行业卡反诈语音质检 - 单通检测")
    parser.add_argument("text", nargs="?", help="通话转写文本")
    parser.add_argument("--file", "-f", help="从文件读取通话文本")
    parser.add_argument("--data-id", help="数据ID（可选）")
    parser.add_argument("--json", action="store_true", help="仅输出 JSON")
    parser.add_argument("--tools", action="store_true", help="使用完整 agentic tool loop（更慢更准）")
    parser.add_argument("--fast", action="store_true", help="使用快速检索增强单轮模式")
    parser.add_argument("--verbose", "-v", action="store_true", help="打印工具调用过程")
    args = parser.parse_args()

    if args.file:
        content = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        content = args.text
    else:
        content = sys.stdin.read()

    content = content.strip()
    if not content:
        print("错误：未提供通话文本。", file=sys.stderr)
        return 2

    agent = QcAgent(verbose=args.verbose)
    use_tools = True if args.tools else (False if args.fast else None)
    result = agent.inspect(content, data_id=args.data_id, use_tools=use_tools)

    if args.json:
        print(result.to_json())
        return 0

    effective = "tools" if (use_tools or (use_tools is None and agent.config.use_tools)) else "fast"
    print(f"质检模式：{agent.mode}/{effective}（llm 需配置 LLM_API_KEY，否则启发式回退）")
    print("=" * 60)
    print(f"复核标签：{result.label}")
    print(f"是否违规：{'是' if result.is_violation else '否'}    是否涉诈：{'是' if result.is_fraud else '否'}")
    print(f"风险等级：{result.risk_level.value}")
    print(f"场景类目：{result.scene_category}" + (f" / {result.scene_subtype}" if result.scene_subtype else ""))
    print(f"判断说明：{result.explanation}")
    if result.detected_features:
        print(f"命中特征：{'、'.join(result.detected_features)}")
    if result.evidence_quotes:
        print("原文证据：")
        for q in result.evidence_quotes:
            print(f"  - {q}")
    print(f"置信度：{result.confidence:.2f}")
    if result.analysis_thought:
        print("推理/参考：")
        print("  " + result.analysis_thought.replace("\n", "\n  "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
