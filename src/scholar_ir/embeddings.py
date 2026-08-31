"""Embedding client (OpenAI-compatible /v1/embeddings).

设计要点（参考 paperseek）：
  - 多模型 fallback：EMBEDDING_MODEL 支持逗号分隔，逐个尝试。
  - Key 复用：未配置 EMBEDDING_API_KEY 时回退到 DEEPSEEK_API_KEY。
  - 批量编码 + 进程内缓存，避免同一文本重复请求。
  - 任何失败都返回 None，由调用方降级，不阻断主流程。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from scholar_ir.config import (
    DEEPSEEK_API_KEY,
    EMBEDDING_API_BASE,
    EMBEDDING_API_KEY,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_CHARS,
    EMBEDDING_MODEL,
    EMBEDDING_TIMEOUT,
)

# 进程内缓存：text -> vector
_CACHE: Dict[str, List[float]] = {}
# 记录已确认可用的模型，避免每次都从头 fallback
_ACTIVE_MODEL: Optional[str] = None


def _api_key() -> str:
    return (EMBEDDING_API_KEY or DEEPSEEK_API_KEY or "").strip()


def _api_base() -> str:
    base = (EMBEDDING_API_BASE or "").strip().rstrip("/")
    if not base:
        return ""
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def embedding_configured() -> bool:
    """是否具备调用外部 embedding 服务的条件。"""
    return bool(_api_key() and _api_base())


def _model_candidates() -> List[str]:
    models = [m.strip() for m in (EMBEDDING_MODEL or "").split(",") if m.strip()]
    if _ACTIVE_MODEL and _ACTIVE_MODEL in models:
        # 已验证可用的模型优先
        models.remove(_ACTIVE_MODEL)
        models.insert(0, _ACTIVE_MODEL)
    return models


def _truncate(text: str) -> str:
    text = " ".join((text or "").split())
    if len(text) > EMBEDDING_MAX_CHARS:
        return text[:EMBEDDING_MAX_CHARS]
    return text


def _post_embeddings(model: str, texts: List[str]) -> Optional[List[List[float]]]:
    """调用一次 /v1/embeddings。失败返回 None。"""
    try:
        import requests
    except ImportError:
        return None

    url = f"{_api_base()}/embeddings"
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(
            url,
            headers=headers,
            json={"model": model, "input": texts},
            timeout=EMBEDDING_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        payload = resp.json()
    except Exception:
        return None

    data = payload.get("data")
    if not isinstance(data, list) or len(data) != len(texts):
        return None

    vectors: List[List[float]] = []
    for item in data:
        vec = item.get("embedding") if isinstance(item, dict) else None
        if not isinstance(vec, list) or not vec:
            return None
        vectors.append([float(v) for v in vec])
    return vectors


def encode(texts: List[str]) -> Optional[List[List[float]]]:
    """批量编码文本。任一环节失败返回 None（调用方降级）。"""
    global _ACTIVE_MODEL

    if not texts or not embedding_configured():
        return None

    normalized = [_truncate(t) for t in texts]
    missing = [t for t in set(normalized) if t and t not in _CACHE]

    if missing:
        models = _model_candidates()
        if not models:
            return None

        for model in models:
            all_ok = True
            fetched: Dict[str, List[float]] = {}
            for i in range(0, len(missing), max(1, EMBEDDING_BATCH_SIZE)):
                batch = missing[i : i + EMBEDDING_BATCH_SIZE]
                vectors = _post_embeddings(model, batch)
                if vectors is None:
                    all_ok = False
                    break
                for text, vec in zip(batch, vectors):
                    fetched[text] = vec
            if all_ok:
                _CACHE.update(fetched)
                _ACTIVE_MODEL = model
                break
        else:
            return None

    out: List[List[float]] = []
    for text in normalized:
        vec = _CACHE.get(text)
        if vec is None:
            # 空文本或未命中：用零向量占位，cosine 会得 0
            return None
        out.append(vec)
    return out


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """余弦相似度，结果裁剪到 [0, 1]。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))
