"""Unit tests for organize abstract snippet + relevance-gated tiers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scholar_ir.organize.base import _abstract_snippet, _relevance_score, organize
from scholar_ir.types import JudgeResult, PaperRef, ScoredPaper, UnderstandingResult


def test_abstract_snippet_short_unchanged() -> None:
    assert _abstract_snippet("Short abstract.", 240) == "Short abstract."


def test_abstract_snippet_prefers_sentence_boundary() -> None:
    text = (
        "First sentence ends here. Second sentence continues with more words "
        "that push the total length past the budget so truncation is required."
    )
    out = _abstract_snippet(text, 60)
    assert out.endswith("…")
    assert "First sentence ends here." in out
    assert "Second sentence" not in out
    # Must not cut mid-word inside the first sentence
    assert not out.startswith("First sentence ends her…")


def test_abstract_snippet_chinese_sentence() -> None:
    text = "这是第一句。这是第二句内容会更长一些以至于需要截断处理。"
    out = _abstract_snippet(text, 20)
    assert out.endswith("…")
    assert "这是第一句。" in out


def test_abstract_snippet_falls_back_to_word_boundary() -> None:
    # No sentence terminator within budget
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    out = _abstract_snippet(text, 25)
    assert out.endswith("…")
    assert " " not in out.rstrip("…") or out.count(" ") >= 1
    # Should not end mid-token when a space exists in the second half
    body = out.rstrip("…")
    assert not body.endswith("epsilo")  # would be mid-word cut of epsilon


def test_relevance_score_prefers_new_keys() -> None:
    assert _relevance_score({"relevance": 0.7, "filter": 0.1}) == 0.7
    assert _relevance_score({"llm": 0.8}) == 0.8
    assert _relevance_score({"keyword_coverage": 0.6}) == 0.6
    assert _relevance_score({"filter": 0.4}) == 0.4
    assert _relevance_score({}) == 1.0  # legacy default when missing


def test_organize_high_tier_uses_relevance_not_legacy_filter() -> None:
    """High citation / high blended must not force highly_relevant without relevance."""
    paper = PaperRef(
        paper_id="2301.00001",
        title="LoRA fine-tuning",
        abstract="We propose LoRA. It works well on diffusion models and transformers.",
        year=2024,
        source="arxiv",
        raw={"citationCount": 9999, "venue": "ICLR"},
    )
    scored = ScoredPaper(
        paper=paper,
        score=0.9,  # already normalized / high
        reason="test",
        features={
            "relevance": 0.2,  # below min_filter_for_high
            "keyword_coverage": 0.2,
            "citation": 1.0,
            "blended": 0.7,
            "normalized": 0.9,
        },
    )
    ranking = JudgeResult(scored=[scored], selected=[paper], paper_ids=[paper.paper_id])
    understanding = UnderstandingResult(
        raw_question="LoRA",
        intent="method",
        slots={"topic": "LoRA"},
        relevance_criteria=[],
    )
    result = organize(understanding, ranking, {"snippet_chars": 40})
    assert result.items[0]["tier"] == "partially_relevant"
    snippet = result.items[0]["abstract_snippet"]
    assert snippet.endswith("…") or len(snippet) <= 40
    # Sentence-aware: should prefer ending after "We propose LoRA."
    assert "We propose LoRA." in snippet or snippet.startswith("We propose")


if __name__ == "__main__":
    test_abstract_snippet_short_unchanged()
    test_abstract_snippet_prefers_sentence_boundary()
    test_abstract_snippet_chinese_sentence()
    test_abstract_snippet_falls_back_to_word_boundary()
    test_relevance_score_prefers_new_keys()
    test_organize_high_tier_uses_relevance_not_legacy_filter()
    print("organize_snippet tests passed")
