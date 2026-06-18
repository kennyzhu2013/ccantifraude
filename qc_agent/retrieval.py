"""轻量中文检索引擎（零依赖）。

使用字符 n-gram 的 TF-IDF + 余弦相似度。中文无需分词即可获得稳定召回，
避免引入 jieba / scikit-learn / faiss 等重型依赖，保证 harness 可离线运行。
若后续需要更强语义召回，可在 `embed()` 处替换为向量模型而不影响上层接口。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

_PUNCT = re.compile(r"[\s，。！？、；：“”‘’（）()\[\]【】,.!?;:\"'\-—~·…]+")


def _normalize(text: str) -> str:
    return _PUNCT.sub("", text or "")


def char_ngrams(text: str, n_range: Tuple[int, int] = (2, 3)) -> List[str]:
    """生成字符 n-gram。对中文友好，对英文/数字也能命中片段。"""
    norm = _normalize(text)
    grams: List[str] = []
    lo, hi = n_range
    for n in range(lo, hi + 1):
        if len(norm) < n:
            if norm:
                grams.append(norm)
            continue
        grams.extend(norm[i : i + n] for i in range(len(norm) - n + 1))
    return grams


@dataclass
class _Doc:
    doc_id: int
    tf: Counter
    norm: float  # 预计算的向量模长（tf-idf 加权后）


class TfidfIndex:
    """字符 n-gram TF-IDF 索引。"""

    def __init__(self, n_range: Tuple[int, int] = (2, 3)):
        self.n_range = n_range
        self._docs: List[_Doc] = []
        self._idf: Dict[str, float] = {}
        self._raw: List[str] = []
        self._built = False

    def fit(self, documents: Sequence[str]) -> "TfidfIndex":
        self._raw = list(documents)
        tfs: List[Counter] = []
        df: Counter = Counter()
        for text in documents:
            tf = Counter(char_ngrams(text, self.n_range))
            tfs.append(tf)
            df.update(tf.keys())

        n = max(len(documents), 1)
        self._idf = {term: math.log((1 + n) / (1 + d)) + 1.0 for term, d in df.items()}

        self._docs = []
        for i, tf in enumerate(tfs):
            norm = math.sqrt(
                sum((freq * self._idf.get(term, 0.0)) ** 2 for term, freq in tf.items())
            )
            self._docs.append(_Doc(doc_id=i, tf=tf, norm=norm or 1.0))
        self._built = True
        return self

    def _query_vec(self, text: str) -> Tuple[Counter, float]:
        tf = Counter(char_ngrams(text, self.n_range))
        norm = math.sqrt(
            sum((freq * self._idf.get(term, 0.0)) ** 2 for term, freq in tf.items())
        )
        return tf, (norm or 1.0)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        if not self._built or not self._docs:
            return []
        q_tf, q_norm = self._query_vec(query)
        scored: List[Tuple[int, float]] = []
        for doc in self._docs:
            # 只遍历查询中出现的 term，稀疏点积。
            dot = 0.0
            for term, q_freq in q_tf.items():
                idf = self._idf.get(term)
                if not idf:
                    continue
                d_freq = doc.tf.get(term)
                if d_freq:
                    dot += (q_freq * idf) * (d_freq * idf)
            if dot <= 0:
                continue
            score = dot / (q_norm * doc.norm)
            scored.append((doc.doc_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
