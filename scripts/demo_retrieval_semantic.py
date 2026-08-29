#!/usr/bin/env python3
"""Test adapt_semantic (+ optional live S2 call).

Usage:
  # 只测 adapt 是否把 slots/text 编成 API params（不联网）
  python3 scripts/demo_retrieval_semantic.py

  # 真正打 Semantic Scholar（建议 export S2_API_KEY=...）
  python3 scripts/demo_retrieval_semantic.py --live

  # 走完整 pipeline：understand → retrieve
  python3 scripts/demo_retrieval_semantic.py --live --pipeline
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholar_ir.search.adapt.semantic import (
    adapt_semantic,
    search_semantic_detail,
)
from scholar_ir.types import SubQuery
from scholar_ir.query_understanding import understand


def test_adapt_only() -> None:
    print("=" * 60)
    print("1) adapt_semantic (offline)")
    sq = SubQuery(
        qid="q0",
        text="retrieval augmented generation survey",
        channel="keyword",
        filters={"year_from": 2020},
    )
    slots = {"topic": "retrieval augmented generation", "year_from": 2020, "method": None}
    req = adapt_semantic(sq, slots, limit=5)
    print("text_used:", req.text_used)
    print("params:", json.dumps(req.params, ensure_ascii=False, indent=2))
    print("headers:", "x-api-key=***" if req.headers.get("x-api-key") else "(no key)")
    assert req.params["query"]
    assert req.params["year"] == "2020-"
    assert req.params["limit"] == 5
    print("adapt_semantic OK")


def test_live_search() -> None:
    print("=" * 60)
    print("2) live search_semantic")
    sq = SubQuery(qid="q0", text="federated domain generalization", filters={})
    detail = search_semantic_detail(sq, {}, limit=3)
    print(f"http={detail.status_code} n_hits={len(detail.papers)}")
    if detail.error:
        print(f"error: {detail.error[:160]}")
    for p in detail.papers[:3]:
        print(f"  - {p.paper_id}: {p.title[:80]} ({p.year})")
    if not detail.papers:
        print("[warn] empty — adapt OK if http=429; set S2_API_KEY for higher quota.")


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
            "sources": ["semantic"],
            "per_query_topk": 3,
            "dry_run": not live,
        },
    )
    print("stats:", r.stats)
    print("trace:")
    for t in r.trace:
        print(" ", {k: t[k] for k in t if k != "params"}, "params=", t.get("params"))
    print(f"candidates={len(r.candidates)}")
    for p in r.candidates[:5]:
        print(f"  - {p.paper_id}: {p.title[:70]}")


def main():
    live = "--live" in sys.argv
    pipe = "--pipeline" in sys.argv
    test_adapt_only()
    if live:
        test_live_search()
    if pipe or live:
        test_pipeline(live=live)
    elif not live:
        test_pipeline(live=False)
    print("=" * 60)
    print("done. entry: scripts/demo_retrieval_semantic.py")


if __name__ == "__main__":
    main()
