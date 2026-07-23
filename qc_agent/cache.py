"""质检结果缓存（按内容哈希）。

大批量质检中存在大量完全重复/重跑场景，缓存可避免重复 LLM 调用，显著降本。
键 = sha256(content) + 模型 + 模式，保证换模型/换模式不串味。线程安全（配合并发）。
持久化为 JSONL，进程退出或显式 flush 时落盘。
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional


class ResultCache:
    def __init__(self, path: Path, model: str, mode: str):
        self.path = Path(path)
        self.namespace = f"{model}::{mode}"
        self._lock = threading.Lock()
        self._mem: Dict[str, Dict[str, Any]] = {}
        self._dirty = False
        self._load()

    def _key(self, content: str) -> str:
        h = hashlib.sha256((self.namespace + "\n" + (content or "")).encode("utf-8"))
        return h.hexdigest()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._mem[rec["key"]] = rec["value"]
        except Exception:
            # 缓存损坏不应影响主流程。
            self._mem = {}

    def get(self, content: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._mem.get(self._key(content))

    def set(self, content: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self._mem[self._key(content)] = value
            self._dirty = True

    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as f:
                for key, value in self._mem.items():
                    f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
            self._dirty = False

    def __len__(self) -> int:
        return len(self._mem)
