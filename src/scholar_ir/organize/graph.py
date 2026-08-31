"""Citation relationship graph for organize stage (view=graph).

Builds an in-set citation graph over the final ranked papers.

Edge sources (in order, all optional):
  1. Local raw already on PaperRef (OpenAlex ``referenced_works``, S2 ``references``)
  2. OpenAlex work lookup (``referenced_works``)
  3. Crossref work references (DOI→DOI) when papers have DOIs / edges still empty
  4. Semantic Scholar references/citations — supplement when configured

S2 rate-limit alone must not zero out the graph: OpenAlex / Crossref / local
raw are independent fallbacks. arXiv has no structured cite graph here.

Policy:
  - Whether to *fetch* edges: ``build_graph`` + ≥2 papers (not intent).
  - Whether to *show* as graph: edge denseness after fetch (``view=auto``).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import requests

from scholar_ir.config import OPENALEX_MAILTO
from scholar_ir.eval import is_arxiv_id
from scholar_ir.search.dedup import _extract_arxiv_id, _extract_doi
from scholar_ir.search.s2_client import (
    get_paper_citations,
    get_paper_references,
    s2_configured,
)
from scholar_ir.types import PaperRef

_GRAPH_FIELDS = "paperId,externalIds,title,year"
_OA_ID_RE = re.compile(r"\b(W\d+)\b", re.I)
_OPENALEX_WORK_URL = "https://api.openalex.org/works/{key}"
_CROSSREF_WORK_URL = "https://api.crossref.org/works/{doi}"


def should_build_graph(options: Dict[str, Any], *, n_papers: int = 0) -> bool:
    """Decide whether to fetch citation edges.

    Intent is irrelevant. Default is to build whenever there are ≥2 papers.
    Opt out with ``build_graph=False``.
    """
    if "build_graph" in options:
        return bool(options["build_graph"])
    return int(n_papers) >= 2


def graph_dense_enough(graph: Dict[str, Any], options: Dict[str, Any] | None = None) -> bool:
    """True when in-set citation edges are dense enough to prefer graph view."""
    options = options or {}
    stats = graph.get("stats") or {}
    n_edges = int(stats.get("n_edges") or len(graph.get("edges") or []))
    n_nodes = int(stats.get("n_nodes") or len(graph.get("nodes") or []))
    min_edges = int(options.get("graph_min_edges", 1))
    min_density = float(options.get("graph_min_density", 0.0))
    if n_edges < min_edges:
        return False
    if n_nodes > 0 and min_density > 0.0 and (n_edges / n_nodes) < min_density:
        return False
    return n_edges > 0


def resolve_view(
    options: Dict[str, Any],
    graph: Dict[str, Any] | None = None,
) -> str:
    """Resolve display view from options + post-fetch graph denseness."""
    raw = str(options.get("view", "auto") or "auto").strip().lower()
    if raw == "list":
        return "list"
    if raw == "graph":
        return "graph"
    if graph and graph_dense_enough(graph, options):
        return "graph"
    return "list"


def _norm_alias(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _add_alias(bucket: Set[str], value: Optional[str]) -> None:
    if not value:
        return
    v = str(value).strip()
    if not v:
        return
    bucket.add(_norm_alias(v))
    for prefix in (
        "arxiv:",
        "doi:",
        "corpusid:",
        "https://doi.org/",
        "http://doi.org/",
        "https://openalex.org/",
        "http://openalex.org/",
        "openalex:",
    ):
        low = v.lower()
        if low.startswith(prefix):
            bucket.add(_norm_alias(v[len(prefix) :]))


def _norm_openalex_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    m = _OA_ID_RE.search(str(value))
    return m.group(1).upper() if m else None


def openalex_id(paper: PaperRef) -> Optional[str]:
    """Short OpenAlex work id (W…) if known on the paper."""
    raw = paper.raw or {}
    candidates: List[Any] = [raw.get("id"), paper.paper_id]
    ids = raw.get("ids")
    if isinstance(ids, dict):
        candidates.append(ids.get("openalex"))
    for c in candidates:
        oid = _norm_openalex_id(str(c) if c is not None else "")
        if oid:
            return oid
    return None


def openalex_lookup_key(paper: PaperRef) -> Optional[str]:
    """Path key for GET /works/{key}: W-id, DOI URL, or arxiv:XXXX."""
    oid = openalex_id(paper)
    if oid:
        return oid
    doi = _extract_doi(paper.raw or {})
    if doi:
        return f"https://doi.org/{doi}"
    arxiv = _extract_arxiv_id(paper)
    if arxiv:
        return f"arxiv:{arxiv}"
    return None


def paper_aliases(paper: PaperRef) -> Set[str]:
    """All known id strings that might identify this paper across sources."""
    out: Set[str] = set()
    _add_alias(out, paper.paper_id)

    arxiv = _extract_arxiv_id(paper)
    if arxiv:
        _add_alias(out, arxiv)
        _add_alias(out, f"ARXIV:{arxiv}")
        _add_alias(out, f"arxiv:{arxiv}")

    doi = _extract_doi(paper.raw or {})
    if doi:
        _add_alias(out, doi)
        _add_alias(out, f"DOI:{doi}")

    oid = openalex_id(paper)
    if oid:
        _add_alias(out, oid)
        _add_alias(out, f"https://openalex.org/{oid}")

    raw = paper.raw or {}
    pid = raw.get("paperId")
    if isinstance(pid, str) and pid.strip():
        _add_alias(out, pid)

    ext = raw.get("externalIds")
    if isinstance(ext, dict):
        for key in ("CorpusId", "DOI", "ArXiv", "arXiv", "ACL", "PubMed"):
            val = ext.get(key)
            if val is not None:
                _add_alias(out, str(val))
                if key == "CorpusId":
                    _add_alias(out, f"CorpusId:{val}")

    ids = raw.get("ids")
    if isinstance(ids, dict):
        for key in ("doi", "arxiv", "openalex", "mag"):
            val = ids.get(key)
            if isinstance(val, str):
                _add_alias(out, val)
                if key == "openalex":
                    oid2 = _norm_openalex_id(val)
                    if oid2:
                        _add_alias(out, oid2)

    return {a for a in out if a}


def aliases_from_s2_dict(raw: Dict[str, Any]) -> Set[str]:
    """Aliases for an S2 paper dict (reference/citation payload)."""
    if not isinstance(raw, dict):
        return set()
    ext = raw.get("externalIds") or {}
    arxiv_id = ""
    if isinstance(ext, dict) and ext.get("ArXiv"):
        arxiv_id = str(ext["ArXiv"]).split("/")[-1].split("v")[0]
    paper_id = arxiv_id or raw.get("paperId") or ""
    ref = PaperRef(
        paper_id=str(paper_id),
        title=raw.get("title") or "",
        year=raw.get("year") if isinstance(raw.get("year"), int) else None,
        source="semantic",
        raw=raw,
    )
    return paper_aliases(ref)


def s2_query_id(paper: PaperRef) -> str:
    """Best S2 Graph API paper id for lookups."""
    raw = paper.raw or {}
    pid = raw.get("paperId")
    if isinstance(pid, str) and pid.strip() and not is_arxiv_id(pid):
        if len(pid.strip()) >= 20 or pid.strip().isalnum():
            return pid.strip()

    ext = raw.get("externalIds")
    if isinstance(ext, dict):
        corpus = ext.get("CorpusId")
        if corpus is not None and str(corpus).strip():
            return f"CorpusId:{corpus}"
        arxiv = ext.get("ArXiv") or ext.get("arXiv")
        if isinstance(arxiv, str) and arxiv.strip():
            aid = arxiv.split("/")[-1].split("v")[0]
            return f"ARXIV:{aid}"
        doi = ext.get("DOI") or ext.get("doi")
        if isinstance(doi, str) and doi.strip():
            return f"DOI:{doi.strip()}"

    arxiv = _extract_arxiv_id(paper)
    if arxiv:
        return f"ARXIV:{arxiv}"

    doi = _extract_doi(raw)
    if doi:
        return f"DOI:{doi}"

    return (paper.paper_id or "").strip()


def _build_alias_index(papers: Iterable[PaperRef]) -> Dict[str, str]:
    """Map alias → canonical paper_id (first wins)."""
    index: Dict[str, str] = {}
    for paper in papers:
        canon = (paper.paper_id or "").strip()
        if not canon:
            continue
        for alias in paper_aliases(paper):
            index.setdefault(alias, canon)
    return index


def _register_aliases(index: Dict[str, str], paper_id: str, *values: Optional[str]) -> None:
    for v in values:
        if not v:
            continue
        for alias in paper_aliases(
            PaperRef(paper_id=paper_id, raw={"id": v} if _norm_openalex_id(v) else {})
        ):
            index.setdefault(alias, paper_id)
        # direct norms
        oid = _norm_openalex_id(v)
        if oid:
            index.setdefault(_norm_alias(oid), paper_id)
            index.setdefault(_norm_alias(f"https://openalex.org/{oid}"), paper_id)


def _match_selected(raw: Dict[str, Any], alias_index: Dict[str, str]) -> Optional[str]:
    for alias in aliases_from_s2_dict(raw):
        hit = alias_index.get(alias)
        if hit:
            return hit
    return None


def _match_openalex_ref(ref_id: str, alias_index: Dict[str, str]) -> Optional[str]:
    oid = _norm_openalex_id(ref_id)
    if not oid:
        return None
    for key in (_norm_alias(oid), _norm_alias(f"https://openalex.org/{oid}"), _norm_alias(ref_id)):
        hit = alias_index.get(key)
        if hit:
            return hit
    return None


def fetch_openalex_work(
    lookup_key: str,
    *,
    timeout: float = 20.0,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """GET OpenAlex work; return (work_dict, error)."""
    key = (lookup_key or "").strip()
    if not key:
        return None, "empty_lookup"
    url = _OPENALEX_WORK_URL.format(key=requests.utils.quote(key, safe=":/"))
    params: Dict[str, str] = {"select": "id,doi,ids,referenced_works"}
    if OPENALEX_MAILTO:
        params["mailto"] = OPENALEX_MAILTO
    try:
        resp = requests.get(
            url,
            params=params,
            timeout=timeout,
            headers={"User-Agent": "ScholarIR/0.1 (graph; mailto optional)"},
        )
    except requests.RequestException as e:
        return None, str(e)
    if resp.status_code != 200:
        return None, f"http_{resp.status_code}:{(resp.text or '')[:120]}"
    data = resp.json() if resp.content else {}
    return data if isinstance(data, dict) else None, ""


def fetch_crossref_reference_dois(
    doi: str,
    *,
    timeout: float = 20.0,
) -> Tuple[List[str], str]:
    """GET Crossref work references; return (cited_dois, error)."""
    doi = (doi or "").strip()
    if not doi:
        return [], "empty_doi"
    # strip URL prefix if present
    low = doi.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if low.startswith(prefix):
            doi = doi[len(prefix) :]
            break
    url = _CROSSREF_WORK_URL.format(doi=requests.utils.quote(doi, safe=""))
    mail = OPENALEX_MAILTO or "scholarir@localhost"
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": f"ScholarIR/0.1 (mailto:{mail})"},
        )
    except requests.RequestException as e:
        return [], str(e)
    if resp.status_code != 200:
        return [], f"http_{resp.status_code}:{(resp.text or '')[:120]}"
    data = resp.json() if resp.content else {}
    msg = data.get("message") if isinstance(data, dict) else None
    if not isinstance(msg, dict):
        return [], "bad_payload"
    out: List[str] = []
    for ref in msg.get("reference") or []:
        if not isinstance(ref, dict):
            continue
        d = ref.get("DOI") or ref.get("doi")
        if isinstance(d, str) and d.strip():
            out.append(d.strip().lower())
    return out, ""


def _match_doi(doi: str, alias_index: Dict[str, str]) -> Optional[str]:
    d = (doi or "").strip().lower()
    if not d:
        return None
    for key in (
        _norm_alias(d),
        _norm_alias(f"doi:{d}"),
        _norm_alias(f"https://doi.org/{d}"),
    ):
        hit = alias_index.get(key)
        if hit:
            return hit
    return None


def _edges_from_crossref(
    papers: List[PaperRef],
    alias_index: Dict[str, str],
    add_edge: Callable[[str, str, str], None],
    *,
    timeout: float = 20.0,
) -> Tuple[int, int, int]:
    """Crossref DOI→DOI references among selected papers."""
    hits = calls = errors = 0
    for paper in papers:
        source_id = (paper.paper_id or "").strip()
        doi = _extract_doi(paper.raw or {})
        if not source_id or not doi:
            continue
        calls += 1
        ref_dois, err = fetch_crossref_reference_dois(doi, timeout=timeout)
        if err:
            errors += 1
            continue
        for rd in ref_dois:
            target = _match_doi(rd, alias_index)
            if target:
                add_edge(source_id, target, "crossref")
                hits += 1
    return hits, calls, errors


def _edges_from_local_raw(
    papers: List[PaperRef],
    alias_index: Dict[str, str],
    add_edge: Callable[[str, str, str], None],
) -> int:
    """Use cite lists already attached to PaperRef.raw (no network)."""
    n = 0
    for paper in papers:
        source_id = (paper.paper_id or "").strip()
        if not source_id:
            continue
        raw = paper.raw or {}

        refs = raw.get("referenced_works")
        if isinstance(refs, list):
            for r in refs:
                target = _match_openalex_ref(str(r), alias_index)
                if target:
                    add_edge(source_id, target, "openalex_raw")
                    n += 1

        refs2 = raw.get("references")
        if isinstance(refs2, list):
            for item in refs2:
                cited: Any = item
                if isinstance(item, dict):
                    cited = item.get("citedPaper") or item
                if isinstance(cited, dict):
                    target = _match_selected(cited, alias_index)
                    if target:
                        add_edge(source_id, target, "s2_raw")
                        n += 1
    return n


def _edges_from_openalex(
    papers: List[PaperRef],
    alias_index: Dict[str, str],
    add_edge: Callable[[str, str, str], None],
    *,
    timeout: float = 20.0,
) -> Tuple[int, int, int]:
    """Fetch OpenAlex referenced_works for each paper. Returns (hits, calls, errors)."""
    hits = 0
    calls = 0
    errors = 0
    # Pass 1: resolve ids into alias_index so in-set W-id matching works
    cached_refs: Dict[str, List[str]] = {}
    for paper in papers:
        source_id = (paper.paper_id or "").strip()
        key = openalex_lookup_key(paper)
        if not source_id or not key:
            continue
        calls += 1
        work, err = fetch_openalex_work(key, timeout=timeout)
        if not work:
            errors += 1
            continue
        oid = _norm_openalex_id(str(work.get("id") or ""))
        doi = ""
        ids = work.get("ids") if isinstance(work.get("ids"), dict) else {}
        if isinstance(ids, dict) and ids.get("doi"):
            doi = str(ids["doi"])
        elif work.get("doi"):
            doi = str(work.get("doi"))
        _register_aliases(alias_index, source_id, oid, doi)
        refs = work.get("referenced_works") or []
        if isinstance(refs, list):
            cached_refs[source_id] = [str(r) for r in refs]

    # Pass 2: emit in-set edges (alias_index now includes resolved W-ids)
    for source_id, refs in cached_refs.items():
        for r in refs:
            target = _match_openalex_ref(r, alias_index)
            if target:
                add_edge(source_id, target, "openalex")
                hits += 1
    return hits, calls, errors


def _edges_from_s2(
    papers: List[PaperRef],
    alias_index: Dict[str, str],
    add_edge: Callable[[str, str, str], None],
    *,
    ref_limit: int,
    cit_limit: int,
    fetch_cits: bool,
) -> Tuple[int, int, int]:
    """S2 references/citations. Returns (hits, calls, errors)."""
    if not s2_configured():
        return 0, 0, 0
    hits = 0
    calls = 0
    errors = 0
    selected_ids = {p.paper_id for p in papers if p.paper_id}
    for paper in papers:
        query_id = s2_query_id(paper)
        source_id = paper.paper_id
        if not query_id or not source_id:
            continue

        if ref_limit > 0:
            resp, refs = get_paper_references(
                query_id, fields=_GRAPH_FIELDS, limit=ref_limit
            )
            calls += 1
            if not resp.ok:
                errors += 1
            for raw in refs:
                target = _match_selected(raw, alias_index)
                if target and target in selected_ids:
                    add_edge(source_id, target, "s2")
                    hits += 1

        if fetch_cits and cit_limit > 0:
            resp, cits = get_paper_citations(
                query_id, fields=_GRAPH_FIELDS, limit=cit_limit
            )
            calls += 1
            if not resp.ok:
                errors += 1
            for raw in cits:
                citing = _match_selected(raw, alias_index)
                if citing and citing in selected_ids:
                    add_edge(citing, source_id, "s2")
                    hits += 1
    return hits, calls, errors


def build_citation_graph(
    papers: List[PaperRef],
    items: List[Dict[str, Any]],
    options: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build nodes + in-set citation edges for the selected papers.

    Options:
        - graph_seed_k: max papers to query remote APIs for (default 8)
        - graph_ref_limit / graph_cit_limit: S2 budgets
        - graph_fetch_citations: S2 incoming cites (default True)
        - graph_use_openalex: use OpenAlex referenced_works (default True)
        - graph_use_s2: use S2 refs/cites (default True)
        - graph_use_local_raw: use raw.referenced_works / references (default True)
    """
    options = options or {}
    seed_k = max(0, int(options.get("graph_seed_k", 8)))
    ref_limit = max(0, int(options.get("graph_ref_limit", 100)))
    cit_limit = max(0, int(options.get("graph_cit_limit", 50)))
    fetch_cits = bool(options.get("graph_fetch_citations", True))
    use_oa = bool(options.get("graph_use_openalex", True))
    use_s2 = bool(options.get("graph_use_s2", True))
    use_local = bool(options.get("graph_use_local_raw", True))
    use_crossref = bool(options.get("graph_use_crossref", True))
    oa_timeout = float(options.get("graph_openalex_timeout", 20.0))
    cr_timeout = float(options.get("graph_crossref_timeout", 20.0))

    item_by_id = {
        str(it.get("paper_id")): it for it in items if it.get("paper_id")
    }
    ordered_papers = [p for p in papers if (p.paper_id or "").strip()][:seed_k]
    if not ordered_papers and items:
        nodes = [
            {
                "id": it["paper_id"],
                "title": it.get("title") or "",
                "year": it.get("year"),
                "tier": it.get("tier"),
                "score": it.get("score"),
            }
            for it in items
            if it.get("paper_id")
        ]
        return {
            "nodes": nodes,
            "edges": [],
            "stats": {"n_nodes": len(nodes), "n_edges": 0, "reason": "no_paper_refs"},
        }

    all_selected = [p for p in papers if (p.paper_id or "").strip()] or ordered_papers

    nodes: List[Dict[str, Any]] = []
    for p in all_selected:
        it = item_by_id.get(p.paper_id, {})
        nodes.append(
            {
                "id": p.paper_id,
                "title": p.title or it.get("title") or "",
                "year": p.year if p.year is not None else it.get("year"),
                "tier": it.get("tier"),
                "score": it.get("score"),
                "source": p.source,
            }
        )

    alias_index = _build_alias_index(all_selected)
    selected_ids = {p.paper_id for p in all_selected}

    edges: List[Dict[str, str]] = []
    edge_keys: Set[Tuple[str, str]] = set()
    edge_sources_used: Set[str] = set()

    def _add_edge(source: str, target: str, via: str) -> None:
        if source == target:
            return
        if source not in selected_ids or target not in selected_ids:
            return
        key = (source, target)
        if key in edge_keys:
            return
        edge_keys.add(key)
        edges.append({"source": source, "target": target, "type": "cites", "via": via})
        edge_sources_used.add(via)

    local_hits = 0
    if use_local:
        local_hits = _edges_from_local_raw(all_selected, alias_index, _add_edge)

    oa_hits = oa_calls = oa_errors = 0
    if use_oa:
        oa_hits, oa_calls, oa_errors = _edges_from_openalex(
            ordered_papers, alias_index, _add_edge, timeout=oa_timeout
        )

    cr_hits = cr_calls = cr_errors = 0
    # Crossref fills DOI→DOI edges when OA/S2 are thin or rate-limited
    if use_crossref and (not edges or options.get("graph_crossref_always")):
        cr_hits, cr_calls, cr_errors = _edges_from_crossref(
            ordered_papers, alias_index, _add_edge, timeout=cr_timeout
        )

    s2_hits = s2_calls = s2_errors = 0
    # S2 is a supplement; still run when configured unless disabled.
    # Do not abort the whole graph when S2 alone fails.
    if use_s2 and s2_configured():
        s2_hits, s2_calls, s2_errors = _edges_from_s2(
            ordered_papers,
            alias_index,
            _add_edge,
            ref_limit=ref_limit,
            cit_limit=cit_limit,
            fetch_cits=fetch_cits,
        )

    reason = ""
    if not edges:
        if (oa_errors or cr_errors or s2_errors) and not local_hits:
            reason = "all_sources_failed_or_empty"
        else:
            reason = "no_in_set_edges"

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "n_seeds_queried": len(ordered_papers),
            "sources_used": sorted(edge_sources_used),
            "local_hits": local_hits,
            "openalex": {"hits": oa_hits, "calls": oa_calls, "errors": oa_errors},
            "crossref": {"hits": cr_hits, "calls": cr_calls, "errors": cr_errors},
            "s2": {
                "hits": s2_hits,
                "calls": s2_calls,
                "errors": s2_errors,
                "configured": s2_configured(),
            },
            "n_api_calls": oa_calls + cr_calls + s2_calls,
            "n_api_errors": oa_errors + cr_errors + s2_errors,
            "reason": reason,
        },
    }
