"""Local LLM client (OpenAI-compatible server or in-process HuggingFace)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from scholar_ir.config import (
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_MODEL,
    LOCAL_QWEN_GPTQ_PATH,
    LOCAL_QWEN_MAX_NEW_TOKENS,
    LOCAL_QWEN_TEMPERATURE,
)

_HF_MODEL = None
_HF_TOKENIZER = None


def _messages_to_prompt(messages: List[dict]) -> str:
    """Best-effort plain prompt if tokenizer has no chat template."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role.upper()}:\n{content}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


def chat_completion(
    messages: List[dict],
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Optional[str]:
    """Return assistant text, or None if backend unavailable."""
    temp = LOCAL_QWEN_TEMPERATURE if temperature is None else temperature
    max_new = LOCAL_QWEN_MAX_NEW_TOKENS if max_tokens is None else max_tokens

    if LOCAL_LLM_BASE_URL:
        return _chat_openai_compatible(messages, temperature=temp, max_tokens=max_new)

    if os.getenv("SCHOLAR_IR_USE_LOCAL_HF", "1").lower() not in ("0", "false", "no"):
        if LOCAL_QWEN_GPTQ_PATH.exists():
            return _chat_hf_local(messages, temperature=temp, max_new_tokens=max_new)

    return None


def _chat_openai_compatible(
    messages: List[dict],
    *,
    temperature: float,
    max_tokens: int,
) -> Optional[str]:
    try:
        from openai import OpenAI
    except ImportError:
        return None

    api_key = os.getenv("SCHOLAR_IR_LLM_API_KEY", "EMPTY")
    model = LOCAL_LLM_MODEL or "qwen2.5-32b-instruct-gptq"
    try:
        client = OpenAI(api_key=api_key, base_url=LOCAL_LLM_BASE_URL)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return None


def _load_hf_model():
    global _HF_MODEL, _HF_TOKENIZER
    if _HF_MODEL is not None and _HF_TOKENIZER is not None:
        return _HF_MODEL, _HF_TOKENIZER

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "Local HF model requires: pip install torch transformers accelerate "
            "(GPTQ: pip install auto-gptq or gptqmodel)"
        ) from e

    path = str(LOCAL_QWEN_GPTQ_PATH)
    _HF_TOKENIZER = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    _HF_MODEL = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    return _HF_MODEL, _HF_TOKENIZER


def _chat_hf_local(
    messages: List[dict],
    *,
    temperature: float,
    max_new_tokens: int,
) -> Optional[str]:
    try:
        model, tokenizer = _load_hf_model()
        import torch
    except Exception:
        return None

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = _messages_to_prompt(messages)
    else:
        text = _messages_to_prompt(messages)

    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=0.9,
        )
    new_tokens = out[0][inputs.input_ids.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
