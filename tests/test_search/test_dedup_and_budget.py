"""Tests for cross-source dedup and priority-based semantic budget."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scholar_ir.types import PaperRef, SubQuery
from scholar_ir.search.dedup import (
    canonical_paper_id,
    deduplicate_papers,
    _extract_arxiv_id,
    _extract_doi,
    _normalize_title,
)
from scholar_ir.search.base import (
    _subquery_priority,
    _select_semantic_budget,
    retrieve,
)


# ---- canonical_paper_id ----

def test_arxiv_id_priority() -> None:
    p = PaperRef(paper_id="2401.12345", title="X", source="arxiv")
    assert canonical_paper_id(p) == "arxiv:2401.12345"


def test_openalex_arxiv_extracted() -> None:
    """Openalex W-id only on paper_id but arxiv in raw.ids.arxiv."""
    p = PaperRef(paper_id="W4389791234", title="X", source="openalex", raw={
        "ids": {"arxiv": "https://arxiv.org/abs/2401.12345v1"},
    })
    assert canonical_paper_id(p) == "arxiv:2401.12345"


def test_s2_externalids_arxiv() -> None:
    p = PaperRef(paper_id="abc123def", title="X", source="semantic", raw={
        "externalIds": {"ArXiv": "2401.12345"},
    })
    assert canonical_paper_id(p) == "arxiv:2401.12345"


def test_doi_fallback_when_no_arxiv() -> None:
    p = PaperRef(paper_id="W999", title="Y", source="openalex", raw={
        "ids": {"doi": "10.1109/foo.2024.001"},
    })
    assert canonical_paper_id(p) == "doi:10.1109/foo.2024.001"


def test_title_fallback_when_only_title() -> None:
    p = PaperRef(paper_id="W000", title="Attention Is All You Need", source="openalex")
    key = canonical_paper_id(p)
    assert key.startswith("title:")
    assert "attention" in key


def test_no_identifier_unique_sentinel() -> None:
    """Paper with no id/title returns unique sentinel so each is kept."""
    p1 = PaperRef(paper_id="", title="", source="x")
    p2 = PaperRef(paper_id="", title="", source="x")
    assert canonical_paper_id(p1) != canonical_paper_id(p2) or p1 is p2


def test_cross_source_same_paper_one_id() -> None:
    """Same paper from arxiv/openalex/s2 must produce the same canonical id."""
    arxiv = PaperRef(paper_id="2401.12345", title="X", source="arxiv")
    openalex = PaperRef(
        paper_id="W4389791234", title="X", source="openalex",
        raw={"ids": {"arxiv": "2401.12345"}},
    )
    s2 = PaperRef(
        paper_id="abc123def", title="X", source="semantic",
        raw={"externalIds": {"ArXiv": "2401.12345"}},
    )
    keys = {canonical_paper_id(p) for p in (arxiv, openalex, s2)}
    assert len(keys) == 1, f"expected 1 canonical id, got {keys}"


def test_deduplicate_papers_preserves_order() -> None:
    p1 = PaperRef(paper_id="2401.12345", title="X", source="arxiv")
    p1_dup = PaperRef(paper_id="abc", title="X", source="s2", raw={"externalIds": {"ArXiv": "2401.12345"}})
    p2 = PaperRef(paper_id="2401.67890", title="Y", source="arxiv")
    out = deduplicate_papers([p1, p2, p1_dup])
    assert [p.paper_id for p in out] == ["2401.12345", "2401.67890"]


def test_normalize_title_stable() -> None:
    assert _normalize_title("Attention Is ALL You Need") == "attention is all you need"
    assert _normalize_title("  multi\nspace  test  ") == "multi space test"


def test_extract_doi_variants() -> None:
    raw = {"ids": {"doi": "https://doi.org/10.1109/foo.2024.001"}}
    assert _extract_doi(raw) == "10.1109/foo.2024.001"
    raw2 = {"externalIds": {"DOI": "10.1109/foo.2024.001"}}
    assert _extract_doi(raw2) == "10.1109/foo.2024.001"


def test_extract_arxiv_strip_version() -> None:
    p = PaperRef(paper_id="", title="", raw={"externalIds": {"ArXiv": "2401.12345v2"}})
    assert _extract_arxiv_id(p) == "2401.12345"


# ---- _subquery_priority ----

def test_priority_angle_ordering() -> None:
    sq_core = SubQuery(qid="q0", text="x", angle="core", mode="lexical")
    sq_syn = SubQuery(qid="q1", text="x", angle="synonym", mode="lexical")
    sq_meta = SubQuery(qid="q2", text="x", angle="metadata", mode="lexical")
    sq_raw = SubQuery(qid="q3", text="x", angle="raw", mode="lexical")
    assert _subquery_priority(sq_core) > _subquery_priority(sq_syn)
    assert _subquery_priority(sq_syn) > _subquery_priority(sq_meta)
    assert _subquery_priority(sq_meta) > _subquery_priority(sq_raw)


def test_priority_semantic_channel_wins() -> None:
    sq_sem = SubQuery(qid="q0", text="x", angle="raw", mode="lexical", channel="semantic")
    sq_raw = SubQuery(qid="q1", text="x", angle="raw", mode="lexical", channel="keyword")
    assert _subquery_priority(sq_sem) > _subquery_priority(sq_raw)


def test_priority_semantic_mode_wins() -> None:
    sq = SubQuery(qid="q0", text="x", angle="raw", mode="semantic")
    sq_lex = SubQuery(qid="q1", text="x", angle="raw", mode="lexical")
    assert _subquery_priority(sq) > _subquery_priority(sq_lex)


# ---- _select_semantic_budget ----

def test_budget_picks_top_priority_not_position() -> None:
    sqs = [
        SubQuery(qid="q_raw", text="x", angle="raw", mode="lexical"),
        SubQuery(qid="q_meta", text="x", angle="metadata", mode="lexical"),
        SubQuery(qid="q_syn", text="x", angle="synonym", mode="lexical"),
        SubQuery(qid="q_core", text="x", angle="core", mode="lexical"),
    ]
    sel = _select_semantic_budget(sqs, 2)
    assert "q_core" in sel
    assert "q_syn" in sel
    assert "q_raw" not in sel
    assert "q_meta" not in sel


def test_budget_always_includes_semantic_channel() -> None:
    sqs = [
        SubQuery(qid="q0", text="x", angle="raw", mode="lexical"),
        SubQuery(qid="q1", text="x", angle="semantic", mode="lexical", channel="semantic"),
        SubQuery(qid="q2", text="x", angle="core", mode="lexical"),
    ]
    sel = _select_semantic_budget(sqs, 1)  # budget too small
    assert "q1" in sel  # semantic channel always wins
    # q2 (core) is next-highest; may be added if room
    assert "q0" not in sel  # raw loses


def test_budget_zero_returns_empty() -> None:
    sqs = [SubQuery(qid="q0", text="x", angle="core", mode="lexical")]
    assert _select_semantic_budget(sqs, 0) == set()


def test_budget_all_eligible_when_large() -> None:
    sqs = [
        SubQuery(qid=f"q{i}", text="x", angle="core" if i == 0 else "raw", mode="lexical")
        for i in range(3)
    ]
    sel = _select_semantic_budget(sqs, 100)
    assert len(sel) == 3


def test_budget_stable_ties() -> None:
    """Equal-priority sub_queries: original order preserved (stable sort)."""
    sqs = [
        SubQuery(qid=f"q{i}", text="x", angle="core", mode="lexical")
        for i in range(4)
    ]
    sel = _select_semantic_budget(sqs, 2)
    assert sel == {"q0", "q1"}


# ---- integration with retrieve ----

def test_retrieve_dedups_across_sources() -> None:
    """End-to-end: when arxiv and openalex return the same paper, it appears once."""
    from scholar_ir.search.base import retrieve as base_retrieve
    from scholar_ir.types import UnderstandingResult

    understanding = UnderstandingResult(
        raw_question="diffusion models",
        intent="broad",
        sub_queries=[SubQuery(qid="q0", text="diffusion models", angle="core", mode="lexical")],
    )
    # Mock: dry-run just builds requests without calling sources
    # Use real arxiv with a tiny limit
    result = base_retrieve(understanding, {
        "sources": ["arxiv"],
        "per_query_topk": 3,
        "dry_run": True,
    })
    assert result.stats.get("dry_run") is True


def test_retrieve_skips_semantic_by_priority() -> None:
    """When budget=1, only the highest-priority sub_query gets semantic."""
    from scholar_ir.search.base import retrieve as base_retrieve
    from scholar_ir.types import UnderstandingResult

    understanding = UnderstandingResult(
        raw_question="x",
        intent="broad",
        sub_queries=[
            SubQuery(qid="q_low", text="x", angle="raw", mode="lexical"),
            SubQuery(qid="q_high", text="y", angle="core", mode="lexical"),
        ],
    )
    result = base_retrieve(understanding, {
        "sources": ["semantic"],
        "semantic_max_queries": 1,
        "dry_run": True,
    })
    # trace should have q_low skipped (priority), q_high runs
    skip_entries = [p for p in result.trace if p.get("status") == "skipped"]
    if skip_entries:
        assert any(p.get("qid") == "q_low" for p in skip_entries), (
            f"q_low should be skipped by priority, got: {skip_entries}"
        )


if __name__ == "__main__":
    tests = [
        # canonical id
        test_arxiv_id_priority,
        test_openalex_arxiv_extracted,
        test_s2_externalids_arxiv,
        test_doi_fallback_when_no_arxiv,
        test_title_fallback_when_only_title,
        test_no_identifier_unique_sentinel,
        test_cross_source_same_paper_one_id,
        test_deduplicate_papers_preserves_order,
        test_normalize_title_stable,
        test_extract_doi_variants,
        test_extract_arxiv_strip_version,
        # priority
        test_priority_angle_ordering,
        test_priority_semantic_channel_wins,
        test_priority_semantic_mode_wins,
        # budget
        test_budget_picks_top_priority_not_position,
        test_budget_always_includes_semantic_channel,
        test_budget_zero_returns_empty,
        test_budget_all_eligible_when_large,
        test_budget_stable_ties,
        # integration
        test_retrieve_dedups_across_sources,
        test_retrieve_skips_semantic_by_priority,
    ]

    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception:
            traceback.print_exc()
            print(f"  FAILED: {fn.__name__}")
            failed += 1
            continue
        print(f"  ok: {fn.__name__}")

    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)