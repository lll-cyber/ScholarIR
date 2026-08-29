#!/usr/bin/env python3
"""Semantic Scholar API tests (requires S2_API_KEY in ScholarIR/.env).

Run:
  PYTHONPATH=src python3 tests/test_api/test_s2_api.py
  PYTHONPATH=src python3 -m pytest tests/test_api/test_s2_api.py -v -s

Rate limit: S2_RATE_LIMIT_RPS=1 (default) — consecutive live calls wait ~1s.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scholar_ir.config import S2_RATE_LIMIT_RPS  # noqa: E402
from scholar_ir.search.s2_client import (  # noqa: E402
    get_paper,
    get_rate_limiter,
    paper_search,
    reset_rate_limiter,
    s2_configured,
    s2_get,
    s2_headers,
)
from scholar_ir.search.adapt.semantic import (  # noqa: E402
    adapt_semantic,
    search_semantic_detail,
)
from scholar_ir.types import SubQuery  # noqa: E402


def test_configured() -> None:
    assert s2_configured(), "set S2_API_KEY in ScholarIR/.env"
    hdr = s2_headers()
    assert "x-api-key" in hdr
    print(f"[ok] S2_API_KEY present, rate_limit_rps={S2_RATE_LIMIT_RPS}")


def test_adapt_semantic() -> None:
    sq = SubQuery(
        qid="q0",
        text="retrieval augmented generation",
        filters={"year_from": 2020},
    )
    req = adapt_semantic(sq, {"year_from": 2020}, limit=3)
    assert req.params["query"]
    assert req.params["year"] == "2020-"
    assert req.headers.get("x-api-key")
    print("[ok] adapt_semantic:", req.params)


def test_paper_search_live() -> None:
    resp, papers = paper_search("attention is all you need", limit=2)
    print(
        f"[live] paper_search http={resp.status_code} n={len(papers)} "
        f"waited={resp.waited_s:.2f}s retries={resp.retries}"
    )
    assert resp.ok, resp.error
    assert papers
    p0 = papers[0]
    assert p0.get("title")
    print(f"  hit: {p0.get('paperId')} | {p0.get('title', '')[:80]}")


def test_search_semantic_detail_live() -> None:
    sq = SubQuery(qid="q0", text="federated domain generalization", filters={})
    detail = search_semantic_detail(sq, {}, limit=2)
    print(
        f"[live] search_semantic_detail http={detail.status_code} "
        f"n={len(detail.papers)} waited={detail.waited_s:.2f}s"
    )
    assert detail.status_code == 200
    assert detail.papers


def test_rate_limit_spacing() -> None:
    """Process-wide limiter spacing (offline — live calls may retry on 429)."""
    reset_rate_limiter(1.0)
    lim = get_rate_limiter()
    t0 = time.monotonic()
    lim.wait()
    w2 = lim.wait()
    elapsed = time.monotonic() - t0
    print(f"[ok] global limiter spacing elapsed={elapsed:.2f}s wait2={w2:.2f}s")
    assert w2 >= 0.85


def test_get_paper_by_arxiv() -> None:
    resp, paper = get_paper("ArXiv:1706.03762", rate_limit=True)
    print(f"[live] get_paper http={resp.status_code} title={(paper or {}).get('title', '')[:60]}")
    assert resp.ok
    assert paper and paper.get("title")


def main() -> None:
    print("S2 API test  rate_limit_rps=", S2_RATE_LIMIT_RPS)
    print("limiter interval_s=", get_rate_limiter().min_interval_s)
    if not s2_configured():
        print("[skip] S2_API_KEY not set — copy configs/env.example → .env")
        sys.exit(0)
    test_configured()
    test_adapt_semantic()
    # Rate spacing first (before other live calls consume the limiter window)
    test_rate_limit_spacing()
    test_paper_search_live()
    test_search_semantic_detail_live()
    test_get_paper_by_arxiv()
    print("all passed")


if __name__ == "__main__":
    main()
