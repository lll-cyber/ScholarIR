"""Cross-source paper identity & dedup.

Three backends (arxiv, openalex, semantic scholar) use independent id
systems. Without canonicalization, the same paper becomes 3 candidates:

  arxiv       paper_id = "2401.12345"          source = "arxiv"
  openalex    paper_id = "W4389791234"         source = "openalex"
  semantic    paper_id = "abc123def"           source = "semantic"

canonical_paper_id() pulls the most cross-source-stable handle in priority:

    1. arxiv id     (e.g. "2401.12345")
    2. DOI          (e.g. "10.1109/foo.2024.001")
    3. normalized title  (last resort, fuzzy-equivalent for short titles)

The returned key is namespaced ("arxiv:2401.12345") so two distinct ids
that happen to share a string never collide.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from scholar_ir.types import PaperRef

_ARXIV_RE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>]+)", re.I)


def _extract_arxiv_id(paper: PaperRef) -> Optional[str]:
    """Pull arxiv id from paper.paper_id, raw.externalIds.ArXiv, or raw.ids.arxiv.

    Returns the bare id (no version suffix), or None if not present.
    """
    pid = (paper.paper_id or "").strip()
    if pid:
        m = _ARXIV_RE.search(pid)
        if m:
            return m.group(1)

    raw: Dict[str, Any] = paper.raw or {}

    # S2 style: raw.externalIds.ArXiv
    ext = raw.get("externalIds")
    if isinstance(ext, dict):
        for k in ("ArXiv", "arXiv", "arxiv", "ARXIV"):
            v = ext.get(k)
            if isinstance(v, str):
                m = _ARXIV_RE.search(v)
                if m:
                    return m.group(1)

    # openalex style: raw.ids.arxiv (URL or bare id)
    ids = raw.get("ids")
    if isinstance(ids, dict):
        for k in ("arxiv", "ArXiv", "ARXIV"):
            v = ids.get(k)
            if isinstance(v, str):
                m = _ARXIV_RE.search(v)
                if m:
                    return m.group(1)

    # Some sources store it directly
    for k in ("arxiv_id", "arxivId", "arxiv"):
        v = raw.get(k)
        if isinstance(v, str):
            m = _ARXIV_RE.search(v)
            if m:
                return m.group(1)

    return None


def _extract_doi(raw: Dict[str, Any]) -> Optional[str]:
    """Pull DOI from raw dict (openalex: ids.doi, S2: externalIds.DOI)."""
    if not isinstance(raw, dict):
        return None

    # openalex ids block
    ids = raw.get("ids")
    if isinstance(ids, dict):
        for k in ("doi", "DOI", "Doi"):
            v = ids.get(k)
            if isinstance(v, str):
                m = _DOI_RE.search(v)
                if m:
                    return m.group(1).lower()

    # S2 externalIds
    ext = raw.get("externalIds")
    if isinstance(ext, dict):
        for k in ("DOI", "doi", "Doi"):
            v = ext.get(k)
            if isinstance(v, str):
                m = _DOI_RE.search(v)
                if m:
                    return m.group(1).lower()

    # Top-level doi
    for k in ("doi", "DOI", "Doi"):
        v = raw.get(k)
        if isinstance(v, str):
            m = _DOI_RE.search(v)
            if m:
                return m.group(1).lower()

    return None


def _normalize_title(title: str) -> str:
    """Loose title fingerprint: lowercase, alnum only, collapse whitespace."""
    if not title:
        return ""
    t = title.lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = " ".join(t.split())
    return t[:200]


def canonical_paper_id(paper: PaperRef, *, index: Optional[int] = None) -> str:
    """Return a cross-source stable id for the paper, namespaced.

    Order: arxiv > DOI > title.

    The returned string is unique per paper as long as the backend actually
    carries one of those identifiers. Title-only matches are a last resort
    and may collide on very generic titles. For papers with no id/title at
    all, pass `index` to disambiguate (e.g. enumerate before dedup).
    """
    arxiv = _extract_arxiv_id(paper)
    if arxiv:
        return f"arxiv:{arxiv}"

    raw = paper.raw if isinstance(paper.raw, dict) else {}
    doi = _extract_doi(raw)
    if doi:
        return f"doi:{doi}"

    title = _normalize_title(paper.title or "")
    if title:
        pid = (paper.paper_id or "").strip()
        if pid:
            # paper_id present but not arxiv/DOI → use paper_id as a second-tier
            # disambiguator to avoid title-only collisions on generic titles.
            return f"title:{title};pid:{pid}"
        return f"title:{title}"

    # No identifier at all — give each paper a unique sentinel.
    if index is None:
        pid = (paper.paper_id or "").strip()
        return f"raw:{pid or id(paper)}"
    return f"raw:{index}"


def deduplicate_papers(papers):
    """Deduplicate papers across sources using canonical_paper_id.

    Keeps the first occurrence; the rest are dropped. Order is preserved.
    """
    seen = set()
    out = []
    for i, p in enumerate(papers):
        key = canonical_paper_id(p, index=i)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out