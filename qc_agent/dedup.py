"""近重复聚类去重（SimHash，O(n) 量级）。

真实语料若含近重复话术，对每簇只跑一次 LLM、其余复用代表结论可降本。
采用 64 位 SimHash（基于字符 n-gram）：O(n) 算签名 + O(n×代表数) 海明距离比对，
远快于逐条余弦的 O(n²×词数)。

注意：行业卡 ASR 转写通常『同套路但不同措辞』，词面差异大，去重收益有限；
真正的降本主力是结果缓存 + 快速单轮 + 并发。本模块对真有近重复的数据集才显著。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from .retrieval import char_ngrams

_MASK64 = (1 << 64) - 1


def _hash64(token: str) -> int:
    # FNV-1a 64-bit，确定性、无外部依赖。
    h = 0xCBF29CE484222325
    for ch in token:
        h ^= ord(ch)
        h = (h * 0x100000001B3) & _MASK64
    return h


def simhash(text: str, n_range=(2, 3)) -> int:
    grams = char_ngrams(text, n_range)
    if not grams:
        return 0
    v = [0] * 64
    for g in grams:
        hv = _hash64(g)
        for b in range(64):
            v[b] += 1 if (hv >> b) & 1 else -1
    sig = 0
    for b in range(64):
        if v[b] > 0:
            sig |= 1 << b
    return sig


def _hamming(a: int, b: int) -> int:
    return bin((a ^ b) & _MASK64).count("1")


def cluster_texts(texts: Sequence[str], threshold: float = 0.9) -> List[int]:
    """返回每条文本的簇 id（簇 id = 代表样本下标）。

    threshold 为近似相似度，转换为最大海明距离 max_bits = round(64*(1-threshold))。
    """
    n = len(texts)
    if n == 0:
        return []
    max_bits = max(0, round(64 * (1.0 - threshold)))
    sigs = [simhash(t) for t in texts]
    cluster_of = [-1] * n
    reps: List[int] = []  # 代表下标
    for i in range(n):
        best_rep, best_dist = -1, 65
        for r in reps:
            d = _hamming(sigs[i], sigs[r])
            if d < best_dist:
                best_dist, best_rep = d, r
                if d == 0:
                    break
        if best_rep >= 0 and best_dist <= max_bits:
            cluster_of[i] = best_rep
        else:
            cluster_of[i] = i
            reps.append(i)
    return cluster_of


def group_by_cluster(cluster_of: Sequence[int]) -> Dict[int, List[int]]:
    groups: Dict[int, List[int]] = {}
    for idx, rep in enumerate(cluster_of):
        groups.setdefault(rep, []).append(idx)
    return groups
