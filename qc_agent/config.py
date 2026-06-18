"""运行配置。

所有可调项集中在此，支持环境变量覆盖（可配合 .env 使用）。
默认值保证在『无 API Key / 无第三方依赖』时也能离线跑通（启发式回退）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


def _load_dotenv(path: Path) -> None:
    """极简 .env 加载，避免引入 python-dotenv 依赖。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Config:
    # ---- LLM（OpenAI 兼容协议，可接 Claude / Qwen / DeepSeek / GLM 等）----
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    )
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.0"))
    )
    llm_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "2048"))
    )
    max_tool_turns: int = field(
        default_factory=lambda: int(os.getenv("QC_MAX_TOOL_TURNS", "6"))
    )

    # ---- 知识与数据路径 ----
    spec_path: Path = field(default_factory=lambda: PROJECT_ROOT / "knowledge" / "spec.md")
    rules_path: Path = field(default_factory=lambda: PROJECT_ROOT / "knowledge" / "rules.json")
    cases_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("QC_CASES_PATH", str(PROJECT_ROOT / "data" / "sample_cases.csv"))
        )
    )

    # ---- 检索参数 ----
    retrieve_top_k: int = field(default_factory=lambda: int(os.getenv("QC_RETRIEVE_TOP_K", "3")))

    @property
    def llm_enabled(self) -> bool:
        """是否具备调用真实 LLM 的条件。否则走离线启发式回退。"""
        return bool(self.llm_api_key)


DEFAULT_CONFIG = Config()
