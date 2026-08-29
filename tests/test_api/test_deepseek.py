#!/usr/bin/env python3
"""Smoke test DeepSeek v4-flash API (Understanding JSON extract).

Run:
  export DEEPSEEK_API_KEY=sk-...
  PYTHONPATH=src python3 tests/test_api/test_deepseek.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scholar_ir.llm.deepseek_client import chat_completion, deepseek_configured
from scholar_ir.query_understanding.llm_extract import try_llm_extract


def test_configured() -> None:
    assert deepseek_configured(), "set DEEPSEEK_API_KEY to run this test"
    print("[ok] DEEPSEEK_API_KEY present")


def test_chat_ping() -> None:
    text = chat_completion(
        [{"role": "user", "content": "Reply with exactly: pong"}],
        max_tokens=16,
    )
    print("[live] chat:", repr(text))
    assert text and "pong" in text.lower()


def test_understanding_extract() -> None:
    q = "Could you provide a survey on retrieval-augmented generation since 2020?"
    out = try_llm_extract(q)
    print("[live] try_llm_extract:", json.dumps(out, ensure_ascii=False, indent=2) if out else None)
    assert out is not None
    intent, slots = out
    assert intent in {"survey", "broad", "method"}
    assert isinstance(slots, dict)
    assert slots.get("topic") or (slots.get("terms") or [])
    print("[ok] intent=", intent, "terms=", slots.get("terms"))


def main() -> None:
    print("DeepSeek API test  model=", os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    if not deepseek_configured():
        print("[skip] DEEPSEEK_API_KEY not set")
        sys.exit(0)
    test_configured()
    test_chat_ping()
    test_understanding_extract()
    print("all passed")


if __name__ == "__main__":
    main()
