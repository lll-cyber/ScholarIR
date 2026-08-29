"""Shared LLM chat helpers for Understanding (extract + expand)."""

from __future__ import annotations

import os
from typing import List, Optional


def call_llm(
    messages: List[dict],
    *,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> Optional[str]:
    """Try configured backends in order; return assistant text or None."""
    backend = os.getenv("SCHOLAR_IR_LLM_BACKEND", "auto").lower()
    getters = {
        "deepseek": (_try_deepseek,),
        "local": (_try_local_llm,),
        "openai": (_try_openai_cloud,),
        "auto": (_try_deepseek, _try_local_llm, _try_openai_cloud, _try_spar_local),
    }
    chain = getters.get(backend, getters["auto"])
    for getter in chain:
        text = getter(messages, temperature=temperature, max_tokens=max_tokens)
        if text:
            return text
    return None


def _try_deepseek(
    messages: List[dict],
    *,
    temperature: float,
    max_tokens: int | None,
) -> Optional[str]:
    try:
        from scholar_ir.llm.deepseek_client import chat_completion

        return chat_completion(
            messages,
            temperature=temperature,
            max_tokens=max_tokens or int(os.getenv("DEEPSEEK_MAX_TOKENS", "768")),
        )
    except Exception:
        return None


def _try_openai_cloud(
    messages: List[dict],
    *,
    temperature: float,
    max_tokens: int | None,
) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("SCHOLAR_IR_API_KEY") or ""
    if not api_key or api_key in ("your_openai_api_key_here", ""):
        return None
    if os.getenv("SCHOLAR_IR_LLM_URL") or os.getenv("LOCAL_LLM_BASE_URL"):
        return None

    base_url = os.getenv("OPENAI_ENDPOINT") or None
    model = os.getenv("OPENAI_MODEL") or os.getenv("SCHOLAR_IR_LLM_MODEL") or "gpt-4o-mini"
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
    except Exception:
        return None


def _try_local_llm(
    messages: List[dict],
    *,
    temperature: float,
    max_tokens: int | None,
) -> Optional[str]:
    try:
        from scholar_ir.llm.local_client import chat_completion

        return chat_completion(
            messages,
            temperature=temperature,
            max_tokens=max_tokens or int(os.getenv("SCHOLAR_IR_LLM_MAX_TOKENS", "768")),
        )
    except Exception:
        return None


def _try_spar_local(
    messages: List[dict],
    *,
    temperature: float,
    max_tokens: int | None,
) -> Optional[str]:
    model_name = os.getenv("SCHOLAR_IR_LOCAL_MODEL", "")
    if not model_name:
        return None
    try:
        from scholar_ir.vendor_spar.local_request_v2 import get_from_llm

        prompt = messages[-1]["content"] if messages else ""
        return get_from_llm(prompt, model_name=model_name) or ""
    except Exception:
        return None
