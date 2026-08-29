"""OpenAlex adapt: SubQuery + slots → /works filter params."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from scholar_ir.config import OPENALEX_MAILTO
from scholar_ir.types import PaperRef, SubQuery

OPENALEX_WORKS_URL = "https://api.openalex.org/works"

# Light stopword strip so OA filter gets keyword-ish text (SPAR notes NL is weak).
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
}


@dataclass
class OpenAlexRequest:
    """Ready-to-fire OpenAlex /works request."""

    params: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)
    text_used: str = ""
    filter_parts: List[str] = field(default_factory=list)
    source: str = "openalex"


def _keywordize(text: str) -> str:
    text = " ".join((text or "").split())
    toks = re.findall(r"[A-Za-z0-9\-]+", text.lower())
    kept = [t for t in toks if t not in _STOP and len(t) > 1]
    # Prefer original casing for multiword phrase: join kept in order from original
    if not kept:
        return text[:200]
    # Rebuild from original tokens preserving order/case roughly
    out = []
    for raw in re.findall(r"[A-Za-z0-9\-]+", text):
        if raw.lower() in _STOP or len(raw) <= 1:
            continue
        out.append(raw)
        if len(out) >= 12:
            break
    return " ".join(out)[:200] if out else text[:200]


def _build_query_text(sub_query: SubQuery, slots: Dict[str, Any]) -> str:
    text = (sub_query.text or "").strip()
    method = sub_query.filters.get("method")
    if method and str(method).lower() not in text.lower():
        text = f"{text} {method}".strip()
    return _keywordize(text)


def _year_filters(filters: Dict[str, Any], slots: Dict[str, Any]) -> List[str]:
    yf = filters.get("year_from")
    yt = filters.get("year_to")
    if yf is None:
        yf = slots.get("year_from")
    if yt is None:
        yt = slots.get("year_to")
    parts: List[str] = []
    if yf is not None and yt is not None and int(yf) == int(yt):
        parts.append(f"publication_year:{int(yf)}")
    else:
        if yf is not None:
            parts.append(f"publication_year:>{int(yf) - 1}")
        if yt is not None:
            parts.append(f"publication_year:<{int(yt) + 1}")
    return parts


def adapt_openalex(
    sub_query: SubQuery,
    slots: Optional[Dict[str, Any]] = None,
    *,
    limit: int = 10,
    mailto: Optional[str] = None,
    require_oa: bool = False,
) -> OpenAlexRequest:
    """Map understanding output → OpenAlex /works params."""
    slots = slots or {}
    filters = dict(sub_query.filters or {})
    text = _build_query_text(sub_query, slots)

    filter_parts: List[str] = []
    if text:
        # OpenAlex filter values: commas separate filters; spaces ok inside search
        filter_parts.append(f"title_and_abstract.search:{text}")
    filter_parts.extend(_year_filters(filters, slots))

    venue = filters.get("venue") or slots.get("venue")
    if venue:
        v = venue if isinstance(venue, str) else ",".join(venue)
        filter_parts.append(f"primary_location.source.display_name.search:{v}")

    if require_oa:
        filter_parts.append("open_access.is_oa:true")

    params: Dict[str, Any] = {
        "filter": ",".join(filter_parts),
        "sort": "cited_by_count:desc",
        "per-page": max(1, min(int(limit), 50)),
        "page": 1,
    }
    mail = mailto if mailto is not None else (OPENALEX_MAILTO or "")
    if mail:
        params["mailto"] = mail

    return OpenAlexRequest(
        params=params,
        text_used=text,
        filter_parts=filter_parts,
    )


def _reconstruct_abstract(inv: Optional[Dict[str, List[int]]]) -> str:
    if not inv:
        return ""
    try:
        max_pos = max(p for positions in inv.values() for p in positions)
    except ValueError:
        return ""
    arr = [""] * (max_pos + 1)
    for word, positions in inv.items():
        for p in positions:
            if 0 <= p <= max_pos:
                arr[p] = word
    return " ".join(w for w in arr if w)


_ARXIV_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/|arxiv[:\s])(\d{4}\.\d{4,5})(?:v\d+)?",
    re.I,
)


def _extract_arxiv_id(work: Dict[str, Any]) -> str:
    candidates: List[str] = []
    oa_url = (work.get("open_access") or {}).get("oa_url") or ""
    if oa_url:
        candidates.append(oa_url)
    loc = work.get("primary_location") or {}
    for k in ("landing_page_url", "pdf_url"):
        if loc.get(k):
            candidates.append(loc[k])
    for loc in work.get("locations") or []:
        for k in ("landing_page_url", "pdf_url"):
            if loc.get(k):
                candidates.append(loc[k])
    ids = work.get("ids") or {}
    if isinstance(ids, dict):
        for v in ids.values():
            if isinstance(v, str):
                candidates.append(v)

    for c in candidates:
        m = _ARXIV_RE.search(c)
        if m:
            return m.group(1)
        # bare .../abs/1234.56789
        if "arxiv" in c.lower():
            tail = c.rstrip("/").split("/")[-1].split("v")[0]
            if re.match(r"^\d{4}\.\d{4,5}$", tail):
                return tail
    return ""


def _openalex_short_id(work: Dict[str, Any]) -> str:
    wid = work.get("id") or ""
    if isinstance(wid, str) and "/" in wid:
        return wid.rstrip("/").split("/")[-1]
    return str(wid)


def _work_to_ref(work: Dict[str, Any]) -> PaperRef:
    arxiv_id = _extract_arxiv_id(work)
    paper_id = arxiv_id or _openalex_short_id(work)
    year = work.get("publication_year")
    if isinstance(year, str) and year.isdigit():
        year = int(year)
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
    return PaperRef(
        paper_id=str(paper_id),
        title=work.get("title") or work.get("display_name") or "",
        abstract=abstract,
        year=year if isinstance(year, int) else None,
        source="openalex",
        raw=work,
    )


@dataclass
class OpenAlexSearchResult:
    papers: List[PaperRef]
    request: OpenAlexRequest
    status_code: Optional[int] = None
    error: str = ""


def search_openalex_detail(
    sub_query: SubQuery,
    slots: Optional[Dict[str, Any]] = None,
    *,
    limit: int = 10,
    timeout: float = 20.0,
    mailto: Optional[str] = None,
    require_oa: bool = False,
) -> OpenAlexSearchResult:
    req = adapt_openalex(
        sub_query, slots, limit=limit, mailto=mailto, require_oa=require_oa
    )
    try:
        resp = requests.get(
            OPENALEX_WORKS_URL,
            params=req.params,
            timeout=timeout,
            headers={"User-Agent": "ScholarIR/0.1 (research; mailto optional)"},
        )
    except requests.RequestException as e:
        return OpenAlexSearchResult(papers=[], request=req, error=str(e))

    if resp.status_code != 200:
        return OpenAlexSearchResult(
            papers=[],
            request=req,
            status_code=resp.status_code,
            error=(resp.text or "")[:200],
        )

    data = resp.json() if resp.content else {}
    works = data.get("results") or []
    out: List[PaperRef] = []
    for w in works:
        try:
            ref = _work_to_ref(w)
            if not ref.title:
                continue
            out.append(ref)
        except Exception:
            continue
    return OpenAlexSearchResult(
        papers=out, request=req, status_code=resp.status_code
    )


def search_openalex(
    sub_query: SubQuery,
    slots: Optional[Dict[str, Any]] = None,
    *,
    limit: int = 10,
    timeout: float = 20.0,
    mailto: Optional[str] = None,
    require_oa: bool = False,
) -> List[PaperRef]:
    return search_openalex_detail(
        sub_query,
        slots,
        limit=limit,
        timeout=timeout,
        mailto=mailto,
        require_oa=require_oa,
    ).papers
