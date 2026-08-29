"""Semantic Scholar adapt: SubQuery + slots → Graph API (via s2_client)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from scholar_ir.search.s2_client import (
    DEFAULT_FIELDS,
    S2_SEARCH_URL,
    paper_search,
    s2_configured,
)
from scholar_ir.types import PaperRef, SubQuery


@dataclass
class SemanticRequest:
    """Ready-to-fire Semantic Scholar /paper/search request."""

    params: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)
    text_used: str = ""
    source: str = "semantic"


def _year_range(filters: Dict[str, Any], slots: Dict[str, Any]) -> Optional[str]:
    """Build S2 `year` param like `2020-2024` or `2020-`."""
    yf = filters.get("year_from")
    yt = filters.get("year_to")
    if yf is None:
        yf = slots.get("year_from")
    if yt is None:
        yt = slots.get("year_to")
    if yf is None and yt is None:
        return None
    if yf is not None and yt is not None:
        return f"{int(yf)}-{int(yt)}"
    if yf is not None:
        return f"{int(yf)}-"
    return f"-{int(yt)}"


def _build_query_text(sub_query: SubQuery, slots: Dict[str, Any]) -> str:
    """Prefer subquery text; append method only when routed into filters."""
    text = (sub_query.text or "").strip()
    method = sub_query.filters.get("method")
    if method and method.lower() not in text.lower():
        text = f"{text} {method}".strip()
    return " ".join(text.split())[:300]


def adapt_semantic(
    sub_query: SubQuery,
    slots: Optional[Dict[str, Any]] = None,
    *,
    limit: int = 10,
    api_key: Optional[str] = None,
) -> SemanticRequest:
    """Map understanding output → Semantic Scholar query params (+ headers)."""
    from scholar_ir.search.s2_client import s2_headers

    slots = slots or {}
    filters = dict(sub_query.filters or {})
    text = _build_query_text(sub_query, slots)

    params: Dict[str, Any] = {
        "query": text,
        "limit": max(1, min(int(limit), 100)),
        "fields": DEFAULT_FIELDS,
    }

    year = _year_range(filters, slots)
    if year:
        params["year"] = year

    venue = filters.get("venue") or slots.get("venue")
    if venue:
        params["venue"] = venue if isinstance(venue, str) else ",".join(venue)

    headers = s2_headers(api_key)
    return SemanticRequest(params=params, headers=headers, text_used=text)


def _paper_to_ref(paper: Dict[str, Any]) -> PaperRef:
    arxiv_id = ""
    ext = paper.get("externalIds") or {}
    if isinstance(ext, dict) and ext.get("ArXiv"):
        arxiv_id = str(ext["ArXiv"]).split("/")[-1].split("v")[0]

    paper_id = arxiv_id or paper.get("paperId") or ""
    year = paper.get("year")
    if isinstance(year, str) and year.isdigit():
        year = int(year)

    return PaperRef(
        paper_id=str(paper_id),
        title=paper.get("title") or "",
        abstract=paper.get("abstract") or "",
        year=year if isinstance(year, int) else None,
        source="semantic",
        raw=paper,
    )


@dataclass
class SemanticSearchResult:
    papers: List[PaperRef]
    request: SemanticRequest
    status_code: Optional[int] = None
    error: str = ""
    waited_s: float = 0.0
    retries: int = 0


def search_semantic_detail(
    sub_query: SubQuery,
    slots: Optional[Dict[str, Any]] = None,
    *,
    limit: int = 10,
    timeout: float = 30.0,
    api_key: Optional[str] = None,
    rate_limit: bool = True,
) -> SemanticSearchResult:
    """adapt + rate-limited HTTP call."""
    req = adapt_semantic(sub_query, slots, limit=limit, api_key=api_key)
    resp, papers_raw = paper_search(
        req.params.get("query") or req.text_used,
        limit=int(req.params.get("limit") or limit),
        year=req.params.get("year"),
        venue=req.params.get("venue"),
        fields=str(req.params.get("fields") or DEFAULT_FIELDS),
        timeout=timeout,
        api_key=api_key,
        rate_limit=rate_limit,
    )
    out: List[PaperRef] = []
    for p in papers_raw:
        try:
            out.append(_paper_to_ref(p))
        except Exception:
            continue
    return SemanticSearchResult(
        papers=out,
        request=req,
        status_code=resp.status_code,
        error=resp.error,
        waited_s=resp.waited_s,
        retries=resp.retries,
    )


def search_semantic(
    sub_query: SubQuery,
    slots: Optional[Dict[str, Any]] = None,
    *,
    limit: int = 10,
    timeout: float = 30.0,
    api_key: Optional[str] = None,
    rate_limit: bool = True,
) -> List[PaperRef]:
    """adapt + HTTP call. Returns [] on failure."""
    return search_semantic_detail(
        sub_query,
        slots,
        limit=limit,
        timeout=timeout,
        api_key=api_key,
        rate_limit=rate_limit,
    ).papers


# Re-export for tests / docs
__all__ = [
    "S2_SEARCH_URL",
    "SemanticRequest",
    "SemanticSearchResult",
    "adapt_semantic",
    "search_semantic",
    "search_semantic_detail",
    "s2_configured",
]
