"""Unit tests for organize citation graph (no live S2 required)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scholar_ir.organize.graph import (
    aliases_from_s2_dict,
    build_citation_graph,
    graph_dense_enough,
    paper_aliases,
    resolve_view,
    s2_query_id,
    should_build_graph,
)
from scholar_ir.organize.base import organize
from scholar_ir.search.s2_client import S2Response
from scholar_ir.types import JudgeResult, PaperRef, ScoredPaper, UnderstandingResult


def _paper(
    pid: str,
    title: str = "",
    *,
    year: int = 2024,
    s2_id: str = "",
    arxiv: str = "",
    doi: str = "",
) -> PaperRef:
    raw: Dict[str, Any] = {}
    if s2_id:
        raw["paperId"] = s2_id
    ext: Dict[str, Any] = {}
    if arxiv:
        ext["ArXiv"] = arxiv
    if doi:
        ext["DOI"] = doi
    if ext:
        raw["externalIds"] = ext
    return PaperRef(
        paper_id=pid,
        title=title or pid,
        year=year,
        source="semantic",
        raw=raw,
    )


def test_should_build_graph_by_count_not_intent() -> None:
    assert should_build_graph({}, n_papers=0) is False
    assert should_build_graph({}, n_papers=1) is False
    assert should_build_graph({}, n_papers=2) is True
    assert should_build_graph({"build_graph": False}, n_papers=10) is False
    assert should_build_graph({"build_graph": True}, n_papers=1) is True


def test_resolve_view_follows_density() -> None:
    sparse = {"edges": [], "stats": {"n_edges": 0, "n_nodes": 5}}
    dense = {"edges": [{"source": "a", "target": "b"}], "stats": {"n_edges": 1, "n_nodes": 5}}
    assert resolve_view({"view": "auto"}, sparse) == "list"
    assert resolve_view({"view": "auto"}, dense) == "graph"
    assert resolve_view({"view": "list"}, dense) == "list"
    assert resolve_view({"view": "graph"}, sparse) == "graph"
    assert graph_dense_enough(dense, {"graph_min_edges": 2}) is False
    assert graph_dense_enough(dense, {"graph_min_edges": 1}) is True


def test_s2_query_id_prefers_arxiv_and_hash() -> None:
    p1 = _paper("2106.09685", arxiv="2106.09685")
    assert s2_query_id(p1) == "ARXIV:2106.09685"
    p2 = _paper("abc", s2_id="bc8d8df9a3b0b814570a1b16fc724373e79c9cf6")
    assert s2_query_id(p2) == "bc8d8df9a3b0b814570a1b16fc724373e79c9cf6"


def test_paper_aliases_cross_source() -> None:
    p = _paper(
        "2106.09685",
        s2_id="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        arxiv="2106.09685",
        doi="10.1234/lora",
    )
    aliases = paper_aliases(p)
    assert "2106.09685" in aliases
    assert "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" in aliases
    assert "10.1234/lora" in aliases


def test_build_citation_graph_in_set_edges() -> None:
    """A cites B within selected set → one edge A→B (via S2 when OA off)."""
    a = _paper("2106.09685", "LoRA", arxiv="2106.09685", s2_id="hash_a_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    b = _paper("2308.13111", "QLoRA", arxiv="2308.13111", s2_id="hash_b_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    c = _paper("9999.99999", "Unrelated", arxiv="9999.99999")

    def fake_refs(paper_id: str, **kwargs: Any) -> Tuple[S2Response, List[Dict[str, Any]]]:
        ok = S2Response(ok=True, status_code=200, data={"data": []})
        if "2106.09685" in paper_id or "hash_a" in paper_id:
            return ok, [
                {
                    "paperId": "hash_b_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    "externalIds": {"ArXiv": "2308.13111"},
                    "title": "QLoRA",
                    "year": 2023,
                },
                {
                    "paperId": "outside",
                    "externalIds": {"ArXiv": "1111.11111"},
                    "title": "Outside",
                },
            ]
        return ok, []

    def fake_cits(paper_id: str, **kwargs: Any) -> Tuple[S2Response, List[Dict[str, Any]]]:
        return S2Response(ok=True, status_code=200, data={"data": []}), []

    items = [
        {"paper_id": a.paper_id, "title": a.title, "tier": "highly_relevant", "score": 0.9, "year": 2021},
        {"paper_id": b.paper_id, "title": b.title, "tier": "highly_relevant", "score": 0.8, "year": 2023},
        {"paper_id": c.paper_id, "title": c.title, "tier": "partially_relevant", "score": 0.5, "year": 2020},
    ]

    with patch("scholar_ir.organize.graph.s2_configured", return_value=True), patch(
        "scholar_ir.organize.graph.get_paper_references", side_effect=fake_refs
    ), patch(
        "scholar_ir.organize.graph.get_paper_citations", side_effect=fake_cits
    ):
        graph = build_citation_graph(
            [a, b, c],
            items,
            {"graph_seed_k": 3, "graph_use_openalex": False},
        )

    assert graph["stats"]["n_nodes"] == 3
    assert len(graph["edges"]) == 1
    edge = graph["edges"][0]
    assert edge["source"] == "2106.09685"
    assert edge["target"] == "2308.13111"
    assert edge["type"] == "cites"


def test_build_citation_graph_openalex_works_when_s2_fails() -> None:
    """OpenAlex referenced_works still yield edges if S2 errors out."""
    a = _paper("2305.14314", "QLoRA", arxiv="2305.14314")
    a.raw["id"] = "https://openalex.org/W111"
    a.raw["ids"] = {"openalex": "https://openalex.org/W111", "arxiv": "https://arxiv.org/abs/2305.14314"}
    b = _paper("2106.09685", "LoRA", arxiv="2106.09685")
    b.raw["id"] = "https://openalex.org/W222"
    b.raw["ids"] = {"openalex": "https://openalex.org/W222"}

    def fake_oa(lookup_key: str, **kwargs: Any):
        if "2305.14314" in lookup_key or "W111" in lookup_key:
            return {
                "id": "https://openalex.org/W111",
                "referenced_works": [
                    "https://openalex.org/W222",
                    "https://openalex.org/W999",
                ],
            }, ""
        if "2106.09685" in lookup_key or "W222" in lookup_key:
            return {"id": "https://openalex.org/W222", "referenced_works": []}, ""
        return None, "miss"

    items = [
        {"paper_id": a.paper_id, "tier": "highly_relevant", "score": 0.9},
        {"paper_id": b.paper_id, "tier": "highly_relevant", "score": 0.8},
    ]
    with patch("scholar_ir.organize.graph.s2_configured", return_value=True), patch(
        "scholar_ir.organize.graph.get_paper_references",
        return_value=(S2Response(ok=False, status_code=429, error="rate"), []),
    ), patch(
        "scholar_ir.organize.graph.get_paper_citations",
        return_value=(S2Response(ok=False, status_code=429, error="rate"), []),
    ), patch(
        "scholar_ir.organize.graph.fetch_openalex_work", side_effect=fake_oa
    ):
        graph = build_citation_graph(
            [a, b], items, {"graph_seed_k": 2, "graph_use_s2": True}
        )

    assert len(graph["edges"]) == 1
    assert graph["edges"][0]["source"] == "2305.14314"
    assert graph["edges"][0]["target"] == "2106.09685"
    assert graph["edges"][0]["via"] == "openalex"
    assert "openalex" in graph["stats"]["sources_used"]


def test_build_citation_graph_from_local_raw() -> None:
    a = _paper("a", "A")
    a.raw["referenced_works"] = ["https://openalex.org/W2"]
    a.raw["id"] = "https://openalex.org/W1"
    b = _paper("b", "B")
    b.raw["id"] = "https://openalex.org/W2"
    items = [
        {"paper_id": "a", "tier": "highly_relevant", "score": 0.9},
        {"paper_id": "b", "tier": "highly_relevant", "score": 0.8},
    ]
    with patch("scholar_ir.organize.graph.s2_configured", return_value=False):
        graph = build_citation_graph(
            [a, b],
            items,
            {
                "graph_use_openalex": False,
                "graph_use_s2": False,
                "graph_use_crossref": False,
            },
        )
    assert graph["edges"] == [
        {"source": "a", "target": "b", "type": "cites", "via": "openalex_raw"}
    ]


def test_build_citation_graph_crossref_when_others_fail() -> None:
    a = _paper("qlora", "QLoRA", doi="10.1234/qlora.2023")
    b = _paper("lora", "LoRA", doi="10.1234/lora.2021")

    def fake_cr(doi: str, **kwargs: Any):
        if "qlora" in doi:
            return ["10.1234/lora.2021", "10.1234/other"], ""
        return [], ""

    items = [
        {"paper_id": "qlora", "tier": "highly_relevant", "score": 0.9},
        {"paper_id": "lora", "tier": "highly_relevant", "score": 0.8},
    ]
    with patch("scholar_ir.organize.graph.s2_configured", return_value=False), patch(
        "scholar_ir.organize.graph.fetch_crossref_reference_dois", side_effect=fake_cr
    ):
        graph = build_citation_graph(
            [a, b],
            items,
            {
                "graph_use_openalex": False,
                "graph_use_s2": False,
                "graph_use_crossref": True,
                "graph_crossref_always": True,
            },
        )
    assert len(graph["edges"]) == 1
    assert graph["edges"][0]["via"] == "crossref"
    assert graph["edges"][0]["source"] == "qlora"
    assert graph["edges"][0]["target"] == "lora"


def test_build_citation_graph_citations_direction() -> None:
    """B cites A via citations API → edge B→A."""
    a = _paper("2106.09685", arxiv="2106.09685")
    b = _paper("2308.13111", arxiv="2308.13111")

    def fake_refs(paper_id: str, **kwargs: Any):
        return S2Response(ok=True, status_code=200, data={}), []

    def fake_cits(paper_id: str, **kwargs: Any):
        if "2106.09685" in paper_id:
            return S2Response(ok=True, status_code=200, data={}), [
                {"paperId": "x", "externalIds": {"ArXiv": "2308.13111"}, "title": "QLoRA"}
            ]
        return S2Response(ok=True, status_code=200, data={}), []

    items = [
        {"paper_id": a.paper_id, "tier": "highly_relevant", "score": 0.9},
        {"paper_id": b.paper_id, "tier": "highly_relevant", "score": 0.8},
    ]
    with patch("scholar_ir.organize.graph.s2_configured", return_value=True), patch(
        "scholar_ir.organize.graph.get_paper_references", side_effect=fake_refs
    ), patch(
        "scholar_ir.organize.graph.get_paper_citations", side_effect=fake_cits
    ):
        graph = build_citation_graph(
            [a, b], items, {"graph_seed_k": 2, "graph_use_openalex": False}
        )

    assert graph["edges"][0]["source"] == "2308.13111"
    assert graph["edges"][0]["target"] == "2106.09685"


def test_build_citation_graph_without_s2() -> None:
    a = _paper("2106.09685")
    items = [{"paper_id": a.paper_id, "title": "LoRA", "tier": "highly_relevant", "score": 0.9}]
    with patch("scholar_ir.organize.graph.s2_configured", return_value=False):
        graph = build_citation_graph(
            [a], items, {"graph_use_openalex": False}
        )
    assert len(graph["nodes"]) == 1
    assert graph["edges"] == []


def test_organize_wires_graph_when_edges_dense() -> None:
    papers = [
        _paper("2106.09685", "LoRA", arxiv="2106.09685"),
        _paper("2308.13111", "QLoRA", arxiv="2308.13111"),
    ]
    scored = [
        ScoredPaper(
            paper=p,
            score=0.9,
            reason="test",
            features={"relevance": 0.9, "normalized": 0.9},
        )
        for p in papers
    ]
    ranking = JudgeResult(scored=scored, selected=papers, paper_ids=[p.paper_id for p in papers])
    understanding = UnderstandingResult(
        raw_question="LoRA methods",
        intent="method",  # intent must not block graph
        slots={"topic": "LoRA"},
        relevance_criteria=[],
    )

    def fake_refs(paper_id: str, **kwargs: Any):
        if "2106.09685" in paper_id:
            return S2Response(ok=True, status_code=200, data={}), [
                {"externalIds": {"ArXiv": "2308.13111"}, "paperId": "b"}
            ]
        return S2Response(ok=True, status_code=200, data={}), []

    with patch("scholar_ir.organize.graph.s2_configured", return_value=True), patch(
        "scholar_ir.organize.graph.get_paper_references", side_effect=fake_refs
    ), patch(
        "scholar_ir.organize.graph.get_paper_citations",
        return_value=(S2Response(ok=True, status_code=200, data={}), []),
    ):
        result = organize(
            understanding, ranking, {"view": "auto", "graph_use_openalex": False}
        )

    assert result.view == "graph"
    assert result.graph
    assert result.graph["stats"]["n_edges"] == 1
    payload = result.to_dict()
    assert "graph" in payload
    assert "引用关系图含 1 条边" in result.summary


def test_organize_sparse_stays_list_view() -> None:
    papers = [
        _paper("2106.09685", "LoRA", arxiv="2106.09685"),
        _paper("9999.99999", "Other", arxiv="9999.99999"),
    ]
    scored = [
        ScoredPaper(paper=p, score=0.9, reason="t", features={"relevance": 0.9})
        for p in papers
    ]
    ranking = JudgeResult(scored=scored, selected=papers, paper_ids=[p.paper_id for p in papers])
    understanding = UnderstandingResult(
        raw_question="survey", intent="survey", slots={"topic": "LoRA"}
    )
    with patch("scholar_ir.organize.graph.s2_configured", return_value=True), patch(
        "scholar_ir.organize.graph.get_paper_references",
        return_value=(S2Response(ok=True, status_code=200, data={}), []),
    ), patch(
        "scholar_ir.organize.graph.get_paper_citations",
        return_value=(S2Response(ok=True, status_code=200, data={}), []),
    ):
        result = organize(
            understanding, ranking, {"view": "auto", "graph_use_openalex": False}
        )
    assert result.graph.get("stats", {}).get("n_edges") == 0
    assert result.view == "list"  # sparse → list even for survey


def test_organize_build_graph_false_skips_fetch() -> None:
    p = _paper("2106.09685", "LoRA")
    p2 = _paper("2305.14314", "QLoRA")
    scored = [
        ScoredPaper(paper=p, score=0.9, reason="t", features={"relevance": 0.9}),
        ScoredPaper(paper=p2, score=0.8, reason="t", features={"relevance": 0.8}),
    ]
    ranking = JudgeResult(
        scored=scored, selected=[p, p2], paper_ids=[p.paper_id, p2.paper_id]
    )
    understanding = UnderstandingResult(
        raw_question="LoRA", intent="method", slots={"topic": "LoRA"}
    )
    with patch("scholar_ir.organize.graph.build_citation_graph") as mocked:
        result = organize(understanding, ranking, {"build_graph": False})
        mocked.assert_not_called()
    assert result.graph == {}
    assert result.view == "list"


if __name__ == "__main__":
    test_should_build_graph_by_count_not_intent()
    test_resolve_view_follows_density()
    test_s2_query_id_prefers_arxiv_and_hash()
    test_paper_aliases_cross_source()
    test_build_citation_graph_in_set_edges()
    test_build_citation_graph_openalex_works_when_s2_fails()
    test_build_citation_graph_from_local_raw()
    test_build_citation_graph_crossref_when_others_fail()
    test_build_citation_graph_citations_direction()
    test_build_citation_graph_without_s2()
    test_organize_wires_graph_when_edges_dense()
    test_organize_sparse_stays_list_view()
    test_organize_build_graph_false_skips_fetch()
    print("organize_graph tests passed")
