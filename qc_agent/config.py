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
    # 单次请求超时（秒）。防止个别请求挂死阻塞 worker（SDK 默认超时极长）。
    llm_timeout: float = field(
        default_factory=lambda: float(os.getenv("LLM_TIMEOUT", "90"))
    )
    max_tool_turns: int = field(
        default_factory=lambda: int(os.getenv("QC_MAX_TOOL_TURNS", "6"))
    )
    # 快速模式：检索增强单轮（预先注入相似判例+规范小节，免去多轮工具往返）。
    # 显著降低延迟与 token 成本，适合大批量质检；关闭则走完整 agentic tool loop。
    use_tools: bool = field(
        default_factory=lambda: os.getenv("QC_USE_TOOLS", "false").strip().lower()
        in ("1", "true", "yes", "on")
    )
    # 超长转写截断（保留头尾，控制 token）。0 表示不截断。
    max_content_chars: int = field(
        default_factory=lambda: int(os.getenv("QC_MAX_CONTENT_CHARS", "6000"))
    )
    # LLM 调用失败重试次数与退避基数（秒）。
    llm_max_retries: int = field(
        default_factory=lambda: int(os.getenv("QC_LLM_MAX_RETRIES", "3"))
    )
    llm_retry_backoff: float = field(
        default_factory=lambda: float(os.getenv("QC_LLM_RETRY_BACKOFF", "2.0"))
    )
    # 批量评估并发数。
    batch_concurrency: int = field(
        default_factory=lambda: int(os.getenv("QC_BATCH_CONCURRENCY", "4"))
    )
    # 两阶段策略：快速模式置信度低于该值时自动升级到完整 agentic tool loop 复核。
    # 0 表示关闭（默认）。体现 learn-claude-code 的『错误恢复/换条路』。
    escalate_below_confidence: float = field(
        default_factory=lambda: float(os.getenv("QC_ESCALATE_BELOW_CONFIDENCE", "0.0"))
    )
    # 结果缓存路径（按内容哈希）。空字符串=禁用。降本：避免重复/重跑 LLM 调用。
    cache_path: str = field(default_factory=lambda: os.getenv("QC_CACHE_PATH", ""))

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
    # 分层注入：随待检文本注入的相关消歧规则条数（system prompt 仅常驻标题索引）。
    disambig_top_k: int = field(default_factory=lambda: int(os.getenv("QC_DISAMBIG_TOP_K", "6")))

    @property
    def llm_enabled(self) -> bool:
        """是否具备调用真实 LLM 的条件。否则走离线启发式回退。"""
        return bool(self.llm_api_key)


DEFAULT_CONFIG = Config()
