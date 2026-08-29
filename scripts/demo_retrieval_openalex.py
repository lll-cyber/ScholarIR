#!/usr/bin/env python3
"""Test adapt_openalex (+ optional live OpenAlex call).

Usage:
  python3 scripts/demo_retrieval_openalex.py
  python3 scripts/demo_retrieval_openalex.py --live
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholar_ir.search.adapt.openalex import (
    adapt_openalex,
    search_openalex_detail,
)
from scholar_ir.types import SubQuery
from scholar_ir.query_understanding import understand


def test_adapt_only() -> None:
    print("=" * 60)
    print("1) adapt_openalex (offline)")
    sq = SubQuery(
        qid="q0",
        text="retrieval augmented generation survey",
        channel="keyword",
        filters={"year_from": 2020},
    )
    slots = {"topic": "retrieval augmented generation", "year_from": 2020}
    req = adapt_openalex(sq, slots, limit=5)
    print("text_used:", req.text_used)
    print("filter_parts:", req.filter_parts)
    print("params:", json.dumps(req.params, ensure_ascii=False, indent=2))
    assert "title_and_abstract.search:" in req.params["filter"]
    assert "publication_year:>2019" in req.params["filter"]
    print("adapt_openalex OK")


def test_live_search() -> None:
    print("=" * 60)
    print("2) live search_openalex")
    sq = SubQuery(qid="q0", text="federated domain generalization", filters={})
    detail = search_openalex_detail(sq, {}, limit=3)
    print(f"http={detail.status_code} n_hits={len(detail.papers)}")
    if detail.error:
        print(f"error: {detail.error[:200]}")
    for p in detail.papers[:3]:
        print(f"  - {p.paper_id}: {p.title[:80]} ({p.year})")
    assert detail.status_code == 200, "OpenAlex should work without API key"
    assert detail.papers, "expected at least one hit"


def test_pipeline(live: bool) -> None:
    print("=" * 60)
    print("3) understand → retrieve", "(live)" if live else "(dry_run)")
    from scholar_ir.search import retrieve

    q = "Could you provide a survey on retrieval-augmented generation since 2020?"
    u = understand(q, {"use_llm": False, "max_subqueries": 2})
    print("intent:", u.intent)
    print("slots:", u.slots)
    r = retrieve(
        u,
        {
            "sources": ["openalex"],
            "per_query_topk": 3,
            "dry_run": not live,
        },
    )
    print("stats:", r.stats)
    for t in r.trace:
        slim = {k: t[k] for k in t if k != "params"}
        print(" ", slim)
        print("   params=", t.get("params"))
    print(f"candidates={len(r.candidates)}")
    for p in r.candidates[:5]:
        print(f"  - {p.paper_id}: {p.title[:70]}")


def main():
    live = "--live" in sys.argv
    test_adapt_only()
    if live:
        test_live_search()
        test_pipeline(live=True)
    else:
        test_pipeline(live=False)
    print("=" * 60)
    print("done. entry: scripts/demo_retrieval_openalex.py")


if __name__ == "__main__":
    main()
