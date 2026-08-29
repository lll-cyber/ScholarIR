"""Offline unit tests for S2 rate limiter (no network)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scholar_ir.search.s2_client import RateLimiter, reset_rate_limiter


def test_rate_limiter_interval() -> None:
    lim = RateLimiter(1.0)
    t0 = time.monotonic()
    lim.wait()
    slept1 = lim.wait()
    elapsed = time.monotonic() - t0
    assert slept1 >= 0.85, f"second wait should sleep ~1s, got {slept1:.2f}s"
    assert elapsed >= 0.85
    print(f"[ok] RateLimiter 1s interval elapsed={elapsed:.2f}s slept2={slept1:.2f}s")


def test_reset_global_limiter() -> None:
    reset_rate_limiter(0.2)
    from scholar_ir.search.s2_client import get_rate_limiter

    assert abs(get_rate_limiter().min_interval_s - 0.2) < 0.01
    reset_rate_limiter(1.0)
    print("[ok] reset_rate_limiter")


if __name__ == "__main__":
    test_rate_limiter_interval()
    test_reset_global_limiter()
    print("s2 rate limit unit tests passed")
