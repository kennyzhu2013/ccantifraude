"""LLM 客户端封装（OpenAI 兼容协议）。

可对接 Claude / Qwen / DeepSeek / GLM / OpenAI 等任意兼容 /v1/chat/completions
的服务，只需配置 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL。
依赖 `openai` 包为可选项：未安装或未配置 Key 时 available=False，上层走启发式回退。
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .config import Config


class LLMClient:
    def __init__(self, config: Config):
        self.config = config
        self._client = None
        self._import_error: Optional[str] = None
        if config.llm_enabled:
            try:
                from openai import OpenAI  # type: ignore

                self._client = OpenAI(api_key=config.llm_api_key, base_url=config.llm_base_url)
            except Exception as exc:  # pragma: no cover - 依赖缺失路径
                self._import_error = str(exc)

    @property
    def available(self) -> bool:
        return self._client is not None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Dict[str, Any]:
        """返回归一化后的 assistant message：{'content', 'tool_calls', 'raw'}。"""
        if not self.available:
            raise RuntimeError(
                "LLM 不可用：请安装 openai 并配置 LLM_API_KEY。" + (self._import_error or "")
            )

        kwargs: Dict[str, Any] = {
            "model": self.config.llm_model,
            "messages": messages,
            "temperature": self.config.llm_temperature,
            "max_tokens": self.config.llm_max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        resp = self._create_with_retry(kwargs)
        msg = resp.choices[0].message
        tool_calls = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            tool_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            )
        return {
            "content": msg.content or "",
            "tool_calls": tool_calls,
            "raw": msg,
        }

    def _create_with_retry(self, kwargs: Dict[str, Any]):
        """对限流/网络/5xx 等瞬时错误做指数退避重试。"""
        attempts = max(1, self.config.llm_max_retries)
        last_exc: Optional[Exception] = None
        for i in range(attempts):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - 统一退避，最后一次抛出
                last_exc = exc
                if i == attempts - 1:
                    break
                time.sleep(self.config.llm_retry_backoff * (2 ** i))
        raise last_exc  # type: ignore[misc]
