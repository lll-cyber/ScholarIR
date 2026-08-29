"""arXiv adapt: SubQuery + slots → Atom API query (via `arxiv` package)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import arxiv

from scholar_ir.types import PaperRef, SubQuery

_STOP = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "for",
    "to",
    "in",
    "on",
    "with",
    "about",
    "regarding",
    "please",
    "could",
    "you",
    "provide",
    "find",
    "papers",
    "paper",
    "that",
    "this",
    "from",
    "since",
    "what",
    "are",
    "is",
    "any",
    "some",
    "tell",
    "me",
}


@dataclass
class ArxivRequest:
    """Ready-to-run arXiv Search spec (package params, not HTTP query string)."""

    query: str
    max_results: int = 10
    id_list: List[str] = field(default_factory=list)
    sort_by: str = "relevance"  # relevance | submittedDate | lastUpdatedDate
    text_used: str = ""
    source: str = "arxiv"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "max_results": self.max_results,
            "id_list": list(self.id_list),
            "sort_by": self.sort_by,
        }


def _keywordize(text: str) -> str:
    text = " ".join((text or "").split())
    out: List[str] = []
    for raw in re.findall(r"[A-Za-z0-9\-]+", text):
        if raw.lower() in _STOP or len(raw) <= 1:
            continue
        out.append(raw)
        if len(out) >= 12:
            break
    return " ".join(out)[:200] if out else text[:200]


def _build_query_text(sub_query: SubQuery, slots: Dict[str, Any]) -> str:
    text = (sub_query.text or "").strip()
    # Only append method when Usage Decision put it in filters (not raw slots).
    method = sub_query.filters.get("method")
    if method and str(method).lower() not in text.lower():
        text = f"{text} {method}".strip()
    return _keywordize(text)


def _year_clause(filters: Dict[str, Any], slots: Dict[str, Any]) -> str:
    """arXiv submittedDate range: [YYYYMMDDHHMMSS TO YYYYMMDDHHMMSS]."""
    yf = filters.get("year_from")
    yt = filters.get("year_to")
    if yf is None:
        yf = slots.get("year_from")
    if yt is None:
        yt = slots.get("year_to")
    if yf is None and yt is None:
        return ""
    start = f"{int(yf)}0101000000" if yf is not None else "19910701000000"
    end = f"{int(yt)}1231235959" if yt is not None else "20991231235959"
    return f"submittedDate:[{start} TO {end}]"


def adapt_arxiv(
    sub_query: SubQuery,
    slots: Optional[Dict[str, Any]] = None,
    *,
    limit: int = 10,
    sort_by: str = "relevance",
) -> ArxivRequest:
    """Map understanding output → arXiv Search query string."""
    slots = slots or {}
    filters = dict(sub_query.filters or {})
    text = _build_query_text(sub_query, slots)

    # Prefer all: for multi-term keyword search (title+abs+comments).
    terms = [t for t in text.split() if t]
    if terms:
        # AND keywords; quote multi-token phrases lightly via all:
        body = " AND ".join(f"all:{t}" for t in terms[:8])
    else:
        body = "all:paper"

    year = _year_clause(filters, slots)
    query = f"({body}) AND {year}" if year else body

    return ArxivRequest(
        query=query,
        max_results=max(1, min(int(limit), 50)),
        sort_by=sort_by,
        text_used=text,
    )


def _sort_criterion(name: str) -> arxiv.SortCriterion:
    mapping = {
        "relevance": arxiv.SortCriterion.Relevance,
        "submittedDate": arxiv.SortCriterion.SubmittedDate,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
    }
    return mapping.get(name, arxiv.SortCriterion.Relevance)


def _entry_to_ref(result: arxiv.Result) -> PaperRef:
    entry = result.entry_id.rstrip("/").split("/")[-1]
    paper_id = entry.split("v")[0]
    year = result.published.year if result.published else None
    return PaperRef(
        paper_id=paper_id,
        title=(result.title or "").replace("\n", " ").strip(),
        abstract=(result.summary or "").replace("\n", " ").strip(),
        year=year,
        source="arxiv",
        raw={
            "entry_id": result.entry_id,
            "arxiv_id": paper_id,
            "categories": list(result.categories or []),
            "authors": [a.name for a in (result.authors or [])],
            "published": result.published.isoformat() if result.published else None,
            "pdf_url": result.pdf_url,
        },
    )


@dataclass
class ArxivSearchResult:
    papers: List[PaperRef]
    request: ArxivRequest
    status_code: Optional[int] = None  # 200 / None on client error
    error: str = ""


def _client(delay_seconds: float = 0.5) -> arxiv.Client:
    return arxiv.Client(page_size=25, delay_seconds=delay_seconds, num_retries=3)


def search_arxiv_detail(
    sub_query: SubQuery,
    slots: Optional[Dict[str, Any]] = None,
    *,
    limit: int = 10,
    sort_by: str = "relevance",
    delay_seconds: float = 0.5,
) -> ArxivSearchResult:
    req = adapt_arxiv(sub_query, slots, limit=limit, sort_by=sort_by)
    search = arxiv.Search(
        query=req.query,
        max_results=req.max_results,
        sort_by=_sort_criterion(req.sort_by),
        sort_order=arxiv.SortOrder.Descending,
    )
    try:
        results = list(_client(delay_seconds).results(search))
    except Exception as e:
        return ArxivSearchResult(papers=[], request=req, error=str(e))

    papers = []
    for r in results:
        try:
            papers.append(_entry_to_ref(r))
        except Exception:
            continue
    return ArxivSearchResult(papers=papers, request=req, status_code=200)


def search_arxiv(
    sub_query: SubQuery,
    slots: Optional[Dict[str, Any]] = None,
    *,
    limit: int = 10,
    sort_by: str = "relevance",
) -> List[PaperRef]:
    return search_arxiv_detail(
        sub_query, slots, limit=limit, sort_by=sort_by
    ).papers


def fetch_arxiv_by_ids(
    arxiv_ids: Sequence[str],
    *,
    delay_seconds: float = 0.5,
) -> List[PaperRef]:
    """补全/校验：按 arXiv id 拉题摘（去版本号）。"""
    cleaned: List[str] = []
    seen = set()
    for x in arxiv_ids:
        aid = (x or "").strip().lower().split("v")[0]
        if not aid or aid in seen:
            continue
        seen.add(aid)
        cleaned.append(aid)
    if not cleaned:
        return []

    search = arxiv.Search(id_list=cleaned, max_results=len(cleaned))
    try:
        results = list(_client(delay_seconds).results(search))
    except Exception:
        return []
    out: List[PaperRef] = []
    for r in results:
        try:
            out.append(_entry_to_ref(r))
        except Exception:
            continue
    return out


def search_arxiv_id_by_title(
    title: str,
    *,
    max_results: int = 5,
    delay_seconds: float = 0.5,
) -> Optional[str]:
    """Title → arXiv id（规范化后精确匹配；失败返回 None）。"""
    title = (title or "").strip()
    if not title:
        return None

    def norm(t: str) -> str:
        return re.sub(r"[^a-z0-9]", "", t.lower())

    target = norm(title)
    # Escape double quotes in title for ti: query
    safe = title.replace('"', "")
    search = arxiv.Search(
        query=f'ti:"{safe}"',
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    try:
        results = list(_client(delay_seconds).results(search))
    except Exception:
        return None
    for r in results:
        if norm(r.title or "") == target:
            return r.entry_id.rstrip("/").split("/")[-1].split("v")[0]
    # fallback: first hit if very close
    if results and target and target in norm(results[0].title or ""):
        return results[0].entry_id.rstrip("/").split("/")[-1].split("v")[0]
    return None
