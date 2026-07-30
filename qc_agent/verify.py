"""证据校验：程序化验证 evidence_quotes 确实出自待检转写原文。

针对已观测到的『few-shot 照抄』失效模式——模型把检索到的相似判例/规范话术
当成待检文本下结论。此时其 evidence_quotes 往往不在原文中，可被确定性拦截，
比 prompt 告诫可靠。

匹配策略（对 ASR 文本友好）：
- 归一化（去空白与中英文标点）后做子串匹配；
- 长引用允许少量删改：按固定窗口分块，块命中率 >= 0.5 即视为命中；
- 引用中的省略号（… / ...）视为分段符，各段独立校验后取多数。
"""
from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from .retrieval import _normalize

_ELLIPSIS = re.compile(r"(?:…|\.{3,}|中间省略)+")
_CHUNK = 12


def _segment_hit(norm_seg: str, norm_content: str) -> bool:
    if not norm_seg:
        return True
    if norm_seg in norm_content:
        return True
    if len(norm_seg) <= _CHUNK:
        return False
    starts = list(range(0, len(norm_seg) - _CHUNK + 1, _CHUNK))
    if not starts:
        return False
    hits = sum(1 for i in starts if norm_seg[i : i + _CHUNK] in norm_content)
    return hits / len(starts) >= 0.5


def quote_in_content(quote: str, content: str, norm_content: str = None) -> bool:
    nc = norm_content if norm_content is not None else _normalize(content)
    segments = [s for s in _ELLIPSIS.split(quote or "") if s.strip()]
    if not segments:
        return False
    hits = sum(1 for seg in segments if _segment_hit(_normalize(seg), nc))
    return hits / len(segments) >= 0.5


def verify_evidence(quotes: Sequence[str], content: str) -> Tuple[float, List[str]]:
    """返回 (命中比例, 未命中的引用列表)。无引用时返回 (1.0, [])。"""
    quotes = [q for q in (quotes or []) if (q or "").strip()]
    if not quotes:
        return 1.0, []
    nc = _normalize(content)
    missing = [q for q in quotes if not quote_in_content(q, content, norm_content=nc)]
    return (len(quotes) - len(missing)) / len(quotes), missing
