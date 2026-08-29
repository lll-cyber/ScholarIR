#!/usr/bin/env python3
"""Test adapt_arxiv (+ optional live arXiv call).

Usage:
  python3 scripts/demo_retrieval_arxiv.py
  python3 scripts/demo_retrieval_arxiv.py --live
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholar_ir.search.adapt.arxiv_api import (
    adapt_arxiv,
    fetch_arxiv_by_ids,
    search_arxiv_detail,
)
from scholar_ir.types import SubQuery
from scholar_ir.query_understanding import understand


def test_adapt_only() -> None:
    print("=" * 60)
    print("1) adapt_arxiv (offline)")
    sq = SubQuery(
        qid="q0",
        text="retrieval augmented generation survey",
        filters={"year_from": 2020},
    )
    req = adapt_arxiv(sq, {"year_from": 2020}, limit=5)
    print("text_used:", req.text_used)
    print("params:", json.dumps(req.to_dict(), ensure_ascii=False, indent=2))
    assert "all:retrieval" in req.query
    assert "submittedDate:" in req.query
    print("adapt_arxiv OK")


def test_live() -> None:
    print("=" * 60)
    print("2) live search_arxiv")
    sq = SubQuery(qid="q0", text="federated domain generalization", filters={})
    detail = search_arxiv_detail(sq, {}, limit=3)
    print(f"status={detail.status_code} n_hits={len(detail.papers)} err={detail.error!r}")
    for p in detail.papers:
        print(f"  - {p.paper_id}: {p.title[:80]} ({p.year})")
    assert detail.status_code == 200 and detail.papers

    print("3) fetch_arxiv_by_ids")
    refs = fetch_arxiv_by_ids(["2009.02040", "1706.03762"])
    print("fetched:", [r.paper_id for r in refs])
    assert any(r.paper_id == "2009.02040" for r in refs)


def test_pipeline(live: bool) -> None:
    print("=" * 60)
    print("4) understand → retrieve", "(live)" if live else "(dry_run)")
    from scholar_ir.search import retrieve

    q = "Could you provide a survey on retrieval-augmented generation since 2020?"
    u = understand(q, {"use_llm": False, "max_subqueries": 2})
    r = retrieve(
        u,
        {"sources": ["arxiv"], "per_query_topk": 3, "dry_run": not live},
    )
    print("stats:", r.stats)
    for t in r.trace:
        print(" ", {k: t[k] for k in t if k != "params"})
    for p in r.candidates[:5]:
        print(f"  - {p.paper_id}: {p.title[:70]}")


def main():
    live = "--live" in sys.argv
    test_adapt_only()
    if live:
        test_live()
        test_pipeline(True)
    else:
        test_pipeline(False)
    print("=" * 60)
    print("done. entry: scripts/demo_retrieval_arxiv.py")


if __name__ == "__main__":
    main()
