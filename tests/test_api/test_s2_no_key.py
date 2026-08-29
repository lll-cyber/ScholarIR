#!/usr/bin/env python3
"""Semantic Scholar API — unauthenticated (no S2_API_KEY) smoke tests.

Run:
  PYTHONPATH=src python3 tests/test_api/test_s2_no_key.py
  # or
  PYTHONPATH=src python3 -m pytest tests/test_api/test_s2_no_key.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# Force no-key mode for this process (do not inherit a shell key).
os.environ.pop("S2_API_KEY", None)

from scholar_ir.search.adapt.semantic import (  # noqa: E402
    S2_SEARCH_URL,
    adapt_semantic,
    search_semantic_detail,
)
from scholar_ir.types import SubQuery  # noqa: E402


def test_adapt_has_no_api_key_header() -> None:
    sq = SubQuery(qid="q0", text="retrieval augmented generation", filters={})
    req = adapt_semantic(sq, {}, limit=3, api_key="")
    assert "x-api-key" not in {k.lower() for k in req.headers}
    assert req.params["query"]
    assert req.params["limit"] == 3
    print("[ok] adapt_semantic: no x-api-key, params=", req.params)


def test_live_search_unauthenticated() -> None:
    """Unauth calls are allowed; 200 / 429 / transient timeout are all informative."""
    sq = SubQuery(qid="q0", text="attention is all you need", filters={})
    detail = search_semantic_detail(sq, {}, limit=2, api_key="", timeout=30.0)
    print(
        f"[live] url={S2_SEARCH_URL} http={detail.status_code} "
        f"n_hits={len(detail.papers)} err={detail.error[:160]!r}"
    )
    assert "x-api-key" not in {k.lower() for k in detail.request.headers}

    if detail.status_code == 200:
        assert detail.papers, "200 should return at least one paper"
        p = detail.papers[0]
        assert p.title
        print(f"[ok] hit: {p.paper_id} | {p.title[:80]}")
        return

    if detail.status_code == 429:
        print("[ok] unauthenticated path reachable (HTTP 429 rate limit)")
        return

    # Network flaky / blocked: still proves we called without a key.
    if detail.status_code is None and detail.error:
        err_l = detail.error.lower()
        if "timed out" in err_l or "timeout" in err_l or "connection" in err_l:
            print("[warn] network timeout/error without key — adapt path OK")
            return

    raise AssertionError(
        f"unexpected status={detail.status_code} error={detail.error!r}"
    )


def main() -> None:
    print("S2 no-key mode tests")
    print("S2_API_KEY in env:", repr(os.environ.get("S2_API_KEY")))
    test_adapt_has_no_api_key_header()
    test_live_search_unauthenticated()
    print("all passed")


if __name__ == "__main__":
    main()
