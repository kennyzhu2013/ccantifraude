"""人工复核标注语料库（CSV）。

数据格式：data_id, content, comment
    - content：通话 ASR 转写文本（left:/right: 表示主叫/被叫）
    - comment：人工复核标签/说明（如『引导投资』）

作用：作为 RAG few-shot 检索源——质检时召回最相似的历史人工判例，
让模型『看着判过的例子』做判断，对齐人工口径，越用越准。
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .retrieval import TfidfIndex

csv.field_size_limit(min(sys.maxsize, 2_147_483_647))


@dataclass
class LabeledCase:
    data_id: str
    content: str
    comment: str

    def short(self, n: int = 220) -> str:
        c = self.content.replace("\n", " ")
        return c if len(c) <= n else c[:n] + "…"


class CaseStore:
    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        self.cases: List[LabeledCase] = []
        self._index = TfidfIndex()
        self._built = False
        if self.csv_path.exists():
            self.load()

    def load(self) -> "CaseStore":
        self.cases = []
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                content = (row.get("content") or "").strip()
                if not content:
                    continue
                self.cases.append(
                    LabeledCase(
                        data_id=(row.get("data_id") or "").strip(),
                        content=content,
                        comment=(row.get("comment") or "").strip(),
                    )
                )
        if self.cases:
            self._index.fit([c.content for c in self.cases])
            self._built = True
        return self

    def __len__(self) -> int:
        return len(self.cases)

    def retrieve(self, text: str, top_k: int = 3, exclude_id: Optional[str] = None) -> List[LabeledCase]:
        if not self._built:
            return []
        hits = self._index.search(text, top_k=top_k + (1 if exclude_id else 0))
        out: List[LabeledCase] = []
        for idx, _ in hits:
            case = self.cases[idx]
            if exclude_id and case.data_id == exclude_id:
                continue
            out.append(case)
            if len(out) >= top_k:
                break
        return out

    def retrieve_with_scores(self, text: str, top_k: int = 3):
        if not self._built:
            return []
        hits = self._index.search(text, top_k=top_k)
        return [(self.cases[i], s) for i, s in hits]
