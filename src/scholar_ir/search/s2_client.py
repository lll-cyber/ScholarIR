"""Semantic Scholar Graph API client with global rate limiting.

Uses S2_API_KEY from env (via scholar_ir.config). Default: 1 request / second
(S2_RATE_LIMIT_RPS=1) for authenticated tier limits.

Docs: https://api.semanticscholar.org/api-docs/
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

from scholar_ir.config import S2_API_KEY, S2_RATE_LIMIT_RPS

logger = logging.getLogger(__name__)

S2_GRAPH_BASE = "https://api.semanticscholar.org/graph/v1"
S2_SEARCH_URL = f"{S2_GRAPH_BASE}/paper/search"
S2_PAPER_URL = f"{S2_GRAPH_BASE}/paper/{{paper_id}}"
S2_REFERENCES_URL = f"{S2_GRAPH_BASE}/paper/{{paper_id}}/references"
S2_CITATIONS_URL = f"{S2_GRAPH_BASE}/paper/{{paper_id}}/citations"

DEFAULT_FIELDS = (
    "paperId,externalIds,title,abstract,year,authors,url,"
    "citationCount,referenceCount,fieldsOfStudy,openAccessPdf"
)


class RateLimiter:
    """Minimum interval between consecutive API calls (thread-safe)."""

    def __init__(self, min_interval_s: float) -> None:
        self._min_interval = max(0.0, float(min_interval_s))
        self._lock = threading.Lock()
        self._last_at = 0.0

    @property
    def min_interval_s(self) -> float:
        return self._min_interval

    def wait(self) -> float:
        """Block until next slot; return seconds slept."""
        if self._min_interval <= 0:
            return 0.0
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_at
            sleep_s = self._min_interval - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
            self._last_at = time.monotonic()
            return max(0.0, sleep_s)


# Process-wide limiter for all S2 HTTP calls
_global_limiter = RateLimiter(1.0 / max(S2_RATE_LIMIT_RPS, 0.001))


def get_rate_limiter() -> RateLimiter:
    return _global_limiter


def reset_rate_limiter(min_interval_s: Optional[float] = None) -> RateLimiter:
    """Test helper: replace global limiter."""
    global _global_limiter
    if min_interval_s is None:
        min_interval_s = 1.0 / max(S2_RATE_LIMIT_RPS, 0.001)
    _global_limiter = RateLimiter(min_interval_s)
    return _global_limiter


def s2_configured() -> bool:
    key = (S2_API_KEY or "").strip().strip('"').strip("'")
    return bool(key and key not in ("xxx", "your_openai_api_key_here"))


def s2_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    key = api_key if api_key is not None else (S2_API_KEY or "")
    key = str(key).strip().strip('"').strip("'")
    if key and key not in ("xxx", "your_openai_api_key_here"):
        return {"x-api-key": key}
    return {}


@dataclass
class S2Response:
    ok: bool
    status_code: Optional[int]
    data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    waited_s: float = 0.0
    retries: int = 0


def _parse_retry_after(resp: requests.Response) -> float:
    raw = resp.headers.get("Retry-After") or resp.headers.get("retry-after") or ""
    try:
        return max(float(raw), 1.0)
    except (TypeError, ValueError):
        return 1.0


def _exponential_backoff(attempt: int, base: float = 2.0, max_wait: float = 30.0) -> float:
    """Return sleep seconds for attempt 0,1,2,..."""
    import random
    return min(base * (2 ** attempt) + random.uniform(0, 1), max_wait)


def s2_get(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
    api_key: Optional[str] = None,
    max_retries: int = 2,
    rate_limit: bool = True,
) -> S2Response:
    """GET with global rate limit and limited 429 retry."""
    hdrs = dict(headers or {})
    for k, v in s2_headers(api_key).items():
        hdrs.setdefault(k, v)

    waited = 0.0
    retries = 0
    last_err = ""

    for attempt in range(max_retries + 1):
        if rate_limit:
            waited += get_rate_limiter().wait()
        try:
            resp = requests.get(url, params=params, headers=hdrs or None, timeout=timeout)
        except requests.RequestException as e:
            last_err = str(e)
            if attempt < max_retries:
                retries += 1
                time.sleep(1.0)
                continue
            return S2Response(
                ok=False,
                status_code=None,
                error=last_err,
                waited_s=waited,
                retries=retries,
            )

        if resp.status_code == 429 and attempt < max_retries:
            retries += 1
            # Prefer server's Retry-After, then exponential backoff
            sleep_s = max(_parse_retry_after(resp), _exponential_backoff(attempt))
            logger.warning("S2 429 rate limit; sleeping %.1fs (attempt %s)", sleep_s, attempt + 1)
            time.sleep(sleep_s)
            # If we keep hitting 429, tighten the global limiter on the fly
            if attempt >= 1:
                current_interval = get_rate_limiter().min_interval_s
                new_interval = min(current_interval * 1.2, 8.0)
                reset_rate_limiter(new_interval)
                logger.warning("S2 tightened rate limiter to %.2fs between requests", new_interval)
            continue

        if resp.status_code != 200:
            return S2Response(
                ok=False,
                status_code=resp.status_code,
                error=(resp.text or "")[:500],
                waited_s=waited,
                retries=retries,
            )

        try:
            data = resp.json() if resp.content else {}
        except ValueError as e:
            return S2Response(
                ok=False,
                status_code=resp.status_code,
                error=f"invalid json: {e}",
                waited_s=waited,
                retries=retries,
            )
        if not isinstance(data, dict):
            data = {}
        return S2Response(
            ok=True,
            status_code=resp.status_code,
            data=data,
            waited_s=waited,
            retries=retries,
        )

    return S2Response(
        ok=False,
        status_code=429,
        error=last_err or "rate limited",
        waited_s=waited,
        retries=retries,
    )


def paper_search(
    query: str,
    *,
    limit: int = 10,
    year: Optional[str] = None,
    venue: Optional[str] = None,
    fields: str = DEFAULT_FIELDS,
    timeout: float = 30.0,
    api_key: Optional[str] = None,
    rate_limit: bool = True,
) -> Tuple[S2Response, List[Dict[str, Any]]]:
    """Search papers by relevance query. Returns (response_meta, raw paper dicts)."""
    params: Dict[str, Any] = {
        "query": (query or "").strip()[:300],
        "limit": max(1, min(int(limit), 100)),
        "fields": fields,
    }
    if year:
        params["year"] = year
    if venue:
        params["venue"] = venue

    resp = s2_get(
        S2_SEARCH_URL,
        params=params,
        timeout=timeout,
        api_key=api_key,
        rate_limit=rate_limit,
    )
    papers = list(resp.data.get("data") or []) if resp.ok else []
    return resp, papers


def get_paper(
    paper_id: str,
    *,
    fields: str = DEFAULT_FIELDS,
    timeout: float = 30.0,
    api_key: Optional[str] = None,
    rate_limit: bool = True,
) -> Tuple[S2Response, Optional[Dict[str, Any]]]:
    """Fetch one paper by S2 id / ArXiv: / DOI:."""
    pid = (paper_id or "").strip()
    if not pid:
        return S2Response(ok=False, status_code=None, error="empty paper_id"), None
    url = S2_PAPER_URL.format(paper_id=requests.utils.quote(pid, safe=":"))
    resp = s2_get(
        url,
        params={"fields": fields},
        timeout=timeout,
        api_key=api_key,
        rate_limit=rate_limit,
    )
    if not resp.ok:
        return resp, None
    return resp, resp.data if isinstance(resp.data, dict) else None


def get_paper_references(
    paper_id: str,
    *,
    fields: str = DEFAULT_FIELDS,
    limit: int = 10,
    timeout: float = 30.0,
    api_key: Optional[str] = None,
    rate_limit: bool = True,
) -> Tuple[S2Response, List[Dict[str, Any]]]:
    """Fetch papers referenced by the given paper."""
    pid = (paper_id or "").strip()
    if not pid:
        return S2Response(ok=False, status_code=None, error="empty paper_id"), []
    url = S2_REFERENCES_URL.format(paper_id=requests.utils.quote(pid, safe=":"))
    resp = s2_get(
        url,
        params={"fields": fields, "limit": max(1, min(int(limit), 1000))},
        timeout=timeout,
        api_key=api_key,
        rate_limit=rate_limit,
    )
    if not resp.ok:
        return resp, []
    refs = [
        item.get("citedPaper") or {}
        for item in list(resp.data.get("data") or [])
        if isinstance(item, dict)
    ]
    return resp, [r for r in refs if r]


def get_paper_citations(
    paper_id: str,
    *,
    fields: str = DEFAULT_FIELDS,
    limit: int = 10,
    timeout: float = 30.0,
    api_key: Optional[str] = None,
    rate_limit: bool = True,
) -> Tuple[S2Response, List[Dict[str, Any]]]:
    """Fetch papers that cite the given paper."""
    pid = (paper_id or "").strip()
    if not pid:
        return S2Response(ok=False, status_code=None, error="empty paper_id"), []
    url = S2_CITATIONS_URL.format(paper_id=requests.utils.quote(pid, safe=":"))
    resp = s2_get(
        url,
        params={"fields": fields, "limit": max(1, min(int(limit), 1000))},
        timeout=timeout,
        api_key=api_key,
        rate_limit=rate_limit,
    )
    if not resp.ok:
        return resp, []
    cits = [
        item.get("citingPaper") or {}
        for item in list(resp.data.get("data") or [])
        if isinstance(item, dict)
    ]
    return resp, [c for c in cits if c]
