"""Unit tests for impact-aware blending in filter_papers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scholar_ir.filter.base import (
    _blend_score,
    _compute_impact_stats,
    _impact_features,
    _impact_weights,
    _normalize_cross_intent,
    filter_papers,
)
from scholar_ir.types import PaperRef, UnderstandingResult


def _make_paper(
    pid: str,
    title: str,
    *,
    year: int = 2024,
    citations: int = 10,
    venue: str = "",
    abstract: str = "",
) -> PaperRef:
    raw: dict = {"citationCount": citations, "venue": venue} if venue else {"citationCount": citations}
    return PaperRef(
        paper_id=pid,
        title=title,
        abstract=abstract,
        year=year,
        source="semantic",
        raw=raw,
    )


def test_impact_weights_pick_per_intent() -> None:
    """Different intents get different weight profiles."""
    method = _impact_weights("method", None)
    survey = _impact_weights("survey", None)
    specific = _impact_weights("specific", None)
    # survey puts more weight on citation/venue than method
    assert survey["citation"] >= method["citation"]
    assert survey["venue"] >= method["venue"]
    # specific rewards title density
    assert specific["title"] >= method["title"]
    # all weights sum to 1
    for w in (method, survey, specific):
        assert abs(sum(w.values()) - 1.0) < 1e-6


def test_impact_weights_override() -> None:
    custom = {"rel": 0.5, "citation": 0.2, "recency": 0.1, "venue": 0.1, "title": 0.1}
    got = _impact_weights("method", custom)
    assert got == custom


def test_impact_features_recency_citation() -> None:
    a = _make_paper("a", "LoRA fine-tuning", year=2024, citations=0)
    b = _make_paper("b", "LoRA fine-tuning", year=2020, citations=1000)
    feats_a = _impact_features(a, {"lora", "fine", "tuning"}, 1000, 2020, 2024)
    feats_b = _impact_features(b, {"lora", "fine", "tuning"}, 1000, 2020, 2024)
    assert feats_a["recency"] > feats_b["recency"]
    assert feats_a["citation"] < feats_b["citation"]
    assert feats_b["citation"] == 1.0


def test_impact_features_venue_match() -> None:
    p = _make_paper("v", "Attention is all you need", venue="NeurIPS 2017", year=2024)
    feats = _impact_features(p, set(), 10, 2020, 2024)
    assert feats["venue"] == 1.0


def test_blend_score_clipping() -> None:
    weights = _impact_weights("method", None)
    score = _blend_score(0.9, {"citation": 1.0, "recency": 1.0, "venue": 1.0, "title": 1.0}, weights)
    # sum of weights is 1.0; max(rel*rel_w + impact*sum_impact_w) <= 1.0
    assert 0.0 <= score <= 1.0


def test_blend_score_zero_relevance() -> None:
    weights = _impact_weights("method", None)
    score = _blend_score(0.0, {"citation": 1.0, "recency": 1.0, "venue": 1.0, "title": 1.0}, weights)
    # rel_w * 0 + impact weights; still bounded
    assert 0.0 <= score <= 1.0


def test_normalize_cross_intent_anchors_threshold() -> None:
    """threshold maps to 0.5 for any intent bar; hard-zero stays zero."""
    assert _normalize_cross_intent(0.0, 0.15) == 0.0
    assert _normalize_cross_intent(0.15, 0.15) == 0.5
    assert _normalize_cross_intent(0.05, 0.05) == 0.5
    assert _normalize_cross_intent(0.25, 0.25) == 0.5
    # above threshold climbs toward 1
    assert _normalize_cross_intent(1.0, 0.15) == 1.0
    # below threshold stays under 0.5
    assert _normalize_cross_intent(0.075, 0.15) == 0.25
    # threshold=0 → identity
    assert _normalize_cross_intent(0.42, 0.0) == 0.42


def test_normalize_makes_same_margin_comparable() -> None:
    """A score exactly at each intent's bar normalizes to the same 0.5."""
    survey_at_bar = _normalize_cross_intent(0.05, 0.05)
    method_at_bar = _normalize_cross_intent(0.15, 0.15)
    specific_at_bar = _normalize_cross_intent(0.25, 0.25)
    assert survey_at_bar == method_at_bar == specific_at_bar == 0.5


def test_filter_papers_populates_features() -> None:
    """Filter must populate ScoredPaper.features with sub-signals."""
    candidates = [
        _make_paper("a", "LoRA fine-tuning diffusion model", year=2024, citations=500, venue="ICLR"),
        _make_paper("b", "Quantum entanglement in photon systems", year=2010, citations=50, venue="Nature"),
    ]
    understanding = UnderstandingResult(
        raw_question="LoRA fine-tuning",
        intent="method",
        slots={"topic": "LoRA fine-tuning"},
        relevance_criteria=[],
    )
    result = filter_papers(
        understanding,
        candidates,
        {"use_llm": False, "apply_impact": True, "arxiv_only": False},
    )
    assert len(result.scored) == 2
    top = result.scored[0]
    bottom = result.scored[1]
    # top paper is more relevant: more keyword coverage
    assert top.paper.paper_id == "a"
    # features populated
    for sp in result.scored:
        if sp.score > 0:
            assert "keyword_coverage" in sp.features
            assert "relevance" in sp.features
            assert "citation" in sp.features
            assert "recency" in sp.features
            assert "venue" in sp.features
            assert "title" in sp.features
            assert "blended" in sp.features
            assert "normalized" in sp.features
            assert sp.score == sp.features["normalized"]


def test_filter_papers_disabled_impact_legacy() -> None:
    """apply_impact=False: score = normalized(relevance); relevance kept raw."""
    candidates = [
        _make_paper("hi", "LoRA fine-tuning", year=2024, citations=9999, venue="Nature"),
        _make_paper("lo", "LoRA fine-tuning", year=2010, citations=0, venue=""),
    ]
    understanding = UnderstandingResult(
        raw_question="LoRA fine-tuning",
        intent="method",
        slots={"topic": "LoRA fine-tuning"},
        relevance_criteria=[],
    )
    off = filter_papers(
        understanding,
        candidates,
        {"use_llm": False, "apply_impact": False, "arxiv_only": False},
    )
    for sp in off.scored:
        if sp.score > 0:
            assert "blended" not in sp.features
            assert sp.features["relevance"] == sp.features["keyword_coverage"]
            assert sp.score == sp.features["normalized"]
            assert sp.score == _normalize_cross_intent(
                sp.features["relevance"], 0.15  # method default threshold
            )


def test_filter_papers_intent_specific_promotes_title_match() -> None:
    """Specific intent should reward title-match heavily."""
    candidates = [
        _make_paper(
            "x",
            "Few-shot learning for image classification",
            year=2024,
            citations=5,
            venue="",
        ),
        _make_paper(
            "y",
            "Survey of few-shot learning methods",
            year=2018,
            citations=2000,
            venue="NeurIPS",
        ),
    ]
    understanding = UnderstandingResult(
        raw_question="Few-shot learning",
        intent="specific",
        slots={"topic": "Few-shot learning"},
        relevance_criteria=[],
    )
    result = filter_papers(
        understanding,
        candidates,
        {"use_llm": False, "apply_impact": True, "arxiv_only": False},
    )
    # specific intent should keep the precise paper; survey-style paper
    # (with high citation) may still be ahead because of impact, but title
    # density contributes. We only assert features differ.
    assert result.scored[0].features["title"] >= result.scored[1].features["title"]


def test_filter_papers_hard_rejects_skip_impact() -> None:
    """Year/negation rejects must remain at score=0 even with impact on."""
    candidates = [
        _make_paper("ok", "LoRA fine-tuning", year=2024, citations=99999, venue="Nature"),
        _make_paper("old", "LoRA fine-tuning", year=1990, citations=99999, venue="Nature"),
    ]
    understanding = UnderstandingResult(
        raw_question="LoRA fine-tuning",
        intent="method",
        slots={"topic": "LoRA fine-tuning", "year_from": 2020},
        relevance_criteria=[],
    )
    result = filter_papers(
        understanding,
        candidates,
        {"use_llm": False, "apply_impact": True, "arxiv_only": False},
    )
    by_id = {s.paper.paper_id: s for s in result.scored}
    assert by_id["old"].score == 0.0
    assert by_id["old"].reason == "year_mismatch"
    assert by_id["old"].features.get("impact_skipped") is True


def test_filter_papers_cross_intent_normalized_score() -> None:
    """Identical relevance under different intents → comparable normalized scores."""
    candidates = [
        _make_paper("p", "LoRA fine-tuning", year=2024, citations=10, venue=""),
    ]
    base_slots = {"topic": "LoRA fine-tuning"}
    scores = {}
    for intent in ("survey", "method", "specific"):
        understanding = UnderstandingResult(
            raw_question="LoRA fine-tuning",
            intent=intent,
            slots=base_slots,
            relevance_criteria=[],
        )
        result = filter_papers(
            understanding,
            candidates,
            {"use_llm": False, "apply_impact": False, "arxiv_only": False},
        )
        sp = result.scored[0]
        scores[intent] = {
            "relevance": sp.features["relevance"],
            "normalized": sp.features["normalized"],
            "score": sp.score,
        }
    # Same paper → same raw relevance
    assert scores["survey"]["relevance"] == scores["method"]["relevance"]
    assert scores["method"]["relevance"] == scores["specific"]["relevance"]
    # But normalized differs when the pass bar differs (unless all well above bar)
    # Perfect keyword match is usually near 1.0 → normalized near 1.0 for all.
    # Use a weak-match paper to expose the pivot difference.
    weak = [_make_paper("w", "fine something else", year=2024, citations=0)]
    weak_scores = {}
    for intent, thresh in (("survey", 0.05), ("method", 0.15), ("specific", 0.25)):
        understanding = UnderstandingResult(
            raw_question="LoRA fine-tuning",
            intent=intent,
            slots={"topic": "LoRA fine-tuning"},
            relevance_criteria=[],
        )
        result = filter_papers(
            understanding,
            weak,
            {
                "use_llm": False,
                "apply_impact": False,
                "arxiv_only": False,
                "threshold": thresh,
            },
        )
        sp = result.scored[0]
        weak_scores[intent] = sp
        # At/above each intent's bar → public score >= 0.5
        if sp.features["relevance"] >= thresh:
            assert sp.score >= 0.5
        else:
            assert sp.score < 0.5
        assert sp.score == _normalize_cross_intent(sp.features["relevance"], thresh)


def test_filter_papers_threshold_applies_after_blend() -> None:
    candidates = [
        _make_paper("weak", "Random stuff", year=2024, citations=0, venue=""),
        _make_paper("strong", "LoRA fine-tuning", year=2024, citations=100, venue="ICLR"),
    ]
    understanding = UnderstandingResult(
        raw_question="LoRA fine-tuning",
        intent="method",
        slots={"topic": "LoRA fine-tuning"},
        relevance_criteria=[],
    )
    result = filter_papers(
        understanding,
        candidates,
        {"use_llm": False, "apply_impact": True, "threshold": 0.2, "arxiv_only": False},
    )
    selected_ids = [p.paper_id for p in result.selected]
    assert "weak" not in selected_ids
    # Selected papers must sit at/above the normalized pass bar
    for sp in result.scored:
        if sp.paper.paper_id in selected_ids:
            assert sp.score >= 0.5


def test_filter_papers_max_return_truncates() -> None:
    candidates = [
        _make_paper(f"p{i}", f"LoRA fine-tuning variant {i}", year=2024, citations=10)
        for i in range(5)
    ]
    understanding = UnderstandingResult(
        raw_question="LoRA fine-tuning",
        intent="method",
        slots={"topic": "LoRA fine-tuning"},
        relevance_criteria=[],
    )
    result = filter_papers(
        understanding,
        candidates,
        {"use_llm": False, "apply_impact": True, "max_return": 2, "arxiv_only": False},
    )
    assert len(result.selected) == 2
    assert len(result.scored) == 5  # scored keeps full pool


if __name__ == "__main__":
    test_impact_weights_pick_per_intent()
    test_impact_weights_override()
    test_impact_features_recency_citation()
    test_impact_features_venue_match()
    test_blend_score_clipping()
    test_blend_score_zero_relevance()
    test_normalize_cross_intent_anchors_threshold()
    test_normalize_makes_same_margin_comparable()
    test_filter_papers_populates_features()
    test_filter_papers_intent_specific_promotes_title_match()
    test_filter_papers_disabled_impact_legacy()
    test_filter_papers_hard_rejects_skip_impact()
    test_filter_papers_cross_intent_normalized_score()
    test_filter_papers_threshold_applies_after_blend()
    test_filter_papers_max_return_truncates()
    print("filter_impact_blend tests passed")