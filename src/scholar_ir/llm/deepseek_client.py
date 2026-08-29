"""DeepSeek API client (OpenAI-compatible Chat Completions)."""

from __future__ import annotations

import os
from typing import List, Optional

from scholar_ir.config import (
    DEEPSEEK_API_BASE,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MAX_TOKENS,
    DEEPSEEK_MODEL,
    DEEPSEEK_TEMPERATURE,
    DEEPSEEK_THINKING,
)


def deepseek_configured() -> bool:
    key = (DEEPSEEK_API_KEY or "").strip()
    return bool(key) and key not in ("your_deepseek_api_key_here", "xxx")


def chat_completion(
    messages: List[dict],
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
    thinking: Optional[bool] = None,
) -> Optional[str]:
    """Call DeepSeek v4-flash (or configured model). Returns None on failure."""
    if not deepseek_configured():
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    temp = DEEPSEEK_TEMPERATURE if temperature is None else temperature
    max_tok = DEEPSEEK_MAX_TOKENS if max_tokens is None else max_tokens
    model_name = model or DEEPSEEK_MODEL or "deepseek-v4-flash"
    use_thinking = DEEPSEEK_THINKING if thinking is None else thinking

    extra_body = None
    if not use_thinking:
        # JSON slot extraction: disable thinking for lower latency/cost.
        extra_body = {"thinking": {"type": "disabled"}}

    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE)
        kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tok,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        text = (msg.content or "").strip()
        if text:
            return text
        # Some thinking responses may expose reasoning separately; ignore for JSON task.
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            return str(reasoning).strip()
        return None
    except Exception:
        return None
