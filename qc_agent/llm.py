"""LLM 客户端封装（OpenAI 兼容协议）。

可对接 Claude / Qwen / DeepSeek / GLM / OpenAI 等任意兼容 /v1/chat/completions
的服务，只需配置 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL。
依赖 `openai` 包为可选项：未安装或未配置 Key 时 available=False，上层走启发式回退。
"""
from __future__ import annotations

import random
import re
import time
from typing import Any, Dict, List, Optional

from .config import Config

# 客户端错误重试也不会成功（鉴权失败、请求超长、模型名写错），
# 重试只会让整批任务多付 N 倍请求与退避时间才失败。429 不在此列（限流应退避重试）。
_NON_RETRYABLE_STATUS = frozenset({400, 401, 403, 404, 405, 413, 422})

# SDK 未暴露结构化状态码时，从异常文案兜底解析（形如 "Error code: 401 - ..."）。
_STATUS_IN_MESSAGE = re.compile(r"(?:error\s+code|status(?:\s+code)?)\D{0,3}(\d{3})", re.IGNORECASE)


def _status_code(exc: Exception) -> Optional[int]:
    for obj in (exc, getattr(exc, "response", None)):
        code = getattr(obj, "status_code", None)
        if isinstance(code, int):
            return code
    m = _STATUS_IN_MESSAGE.search(str(exc))
    return int(m.group(1)) if m else None


def is_retryable(exc: Exception) -> bool:
    """瞬时故障（限流/5xx/超时/连接中断）才值得重试。"""
    code = _status_code(exc)
    if code is None:
        # 超时、连接重置等无状态码的网络异常，默认可重试。
        return True
    return code not in _NON_RETRYABLE_STATUS


class LLMClient:
    def __init__(self, config: Config):
        self.config = config
        self._client = None
        self._import_error: Optional[str] = None
        if config.llm_enabled:
            try:
                from openai import OpenAI  # type: ignore

                self._client = OpenAI(
                    api_key=config.llm_api_key,
                    base_url=config.llm_base_url,
                    timeout=config.llm_timeout,
                    max_retries=0,  # 重试由本类统一控制（含退避）
                )
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
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """返回归一化后的 assistant message：{'content', 'tool_calls', 'raw'}。

        temperature 可按调用覆盖（自一致性采样需 >0 的多样性）。
        """
        if not self.available:
            raise RuntimeError(
                "LLM 不可用：请安装 openai 并配置 LLM_API_KEY。" + (self._import_error or "")
            )

        kwargs: Dict[str, Any] = {
            "model": self.config.llm_model,
            "messages": messages,
            "temperature": (
                self.config.llm_temperature if temperature is None else temperature
            ),
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
        """对限流/网络/5xx 等瞬时错误做指数退避重试；客户端错误立即抛出。"""
        attempts = max(1, self.config.llm_max_retries)
        last_exc: Optional[Exception] = None
        for i in range(attempts):
            try:
                return self._client.chat.completions.create(
                    timeout=self.config.llm_timeout, **kwargs
                )
            except Exception as exc:  # noqa: BLE001 - 统一退避，最后一次抛出
                last_exc = exc
                if i == attempts - 1 or not is_retryable(exc):
                    break
                # 抖动：并发批量下所有 worker 同时撞限流，等长退避会同步重试再次撞墙。
                backoff = self.config.llm_retry_backoff * (2 ** i)
                time.sleep(backoff * (0.5 + random.random()))
        raise last_exc  # type: ignore[misc]
