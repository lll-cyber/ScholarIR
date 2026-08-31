"""Unit tests for natural-language selection reasons in organize."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scholar_ir.organize.base import (
    build_match_reasons,
    build_selection_reason,
    extract_llm_reason_text,
    organize,
)
from scholar_ir.types import JudgeResult, PaperRef, ScoredPaper, UnderstandingResult


def test_extract_llm_reason_from_debug_string() -> None:
    raw = (
        "llm:0.9;Proposes a deep unfolding network for low-light image enhancement "
        "based on Retinex.;impact[cit=1.0,rec=0.3,ven=0.0,ttl=0.0]"
    )
    assert "deep unfolding" in extract_llm_reason_text(raw)
    assert "impact[" not in extract_llm_reason_text(raw)


def test_extract_expanded_llm_reason() -> None:
    raw = "expanded_llm:0.8;Cites seed LoRA paper on PEFT.;impact[cit=0.1,rec=0.5,ven=0.0,ttl=0.0]"
    assert extract_llm_reason_text(raw) == "Cites seed LoRA paper on PEFT."


def test_extract_keyword_only_returns_empty() -> None:
    assert extract_llm_reason_text("keyword_coverage:0.42") == ""


def test_selection_reason_prefers_llm_text() -> None:
    paper = PaperRef(paper_id="2106.09685", title="LoRA", year=2021)
    scored = ScoredPaper(
        paper=paper,
        score=0.9,
        reason=(
            "llm:0.9;Proposes low-rank adaptation for large language models.;"
            "impact[cit=1.0,rec=0.5,ven=1.0,ttl=1.0]"
        ),
        features={"relevance": 0.9, "llm": 0.9, "citation": 1.0, "normalized": 0.9},
    )
    understanding = UnderstandingResult(
        raw_question="LoRA fine-tuning",
        intent="method",
        slots={"topic": "LoRA"},
    )
    text = build_selection_reason(scored, understanding, "highly_relevant")
    assert text.startswith("入选「高度相关」：")
    assert "low-rank adaptation" in text


def test_selection_reason_heuristic_without_llm() -> None:
    paper = PaperRef(paper_id="x", title="Something", year=2024)
    scored = ScoredPaper(
        paper=paper,
        score=0.6,
        reason="keyword_coverage:0.55",
        features={"keyword_coverage": 0.55, "relevance": 0.55, "title": 0.8},
    )
    understanding = UnderstandingResult(
        raw_question="hybrid architectures",
        intent="survey",
        slots={"topic": "hybrid architectures", "method": "reconstruction"},
    )
    text = build_selection_reason(scored, understanding, "partially_relevant")
    assert "入选「部分相关」" in text
    assert "hybrid architectures" in text
    assert "reconstruction" in text


def test_match_reasons_nl_then_evidence() -> None:
    paper = PaperRef(paper_id="x", title="LoRA", year=2021)
    scored = ScoredPaper(
        paper=paper,
        score=0.9,
        reason="llm:0.9;Matches LoRA fine-tuning query.;impact[cit=0.2,rec=0.1,ven=0.0,ttl=0.5]",
        features={
            "llm": 0.9,
            "relevance": 0.9,
            "title": 0.5,
            "citation": 0.2,
            "normalized": 0.95,
            "blended": 0.8,
        },
    )
    understanding = UnderstandingResult(
        raw_question="LoRA", intent="method", slots={"topic": "LoRA"}
    )
    reasons = build_match_reasons(scored, understanding, "highly_relevant")
    assert reasons[0].startswith("入选「高度相关」：")
    assert "Matches LoRA" in reasons[0]
    # evidence tags follow; normalized/blended skipped
    evidence = reasons[1:]
    assert evidence
    assert all("跨意图归一" not in e and "综合得分" not in e for e in evidence)


def test_organize_item_has_selection_reason() -> None:
    paper = PaperRef(
        paper_id="2106.09685",
        title="LoRA",
        abstract="We propose LoRA. It adapts large language models efficiently.",
        year=2021,
        raw={},
    )
    scored = ScoredPaper(
        paper=paper,
        score=0.9,
        reason="llm:0.95;Core LoRA paper for parameter-efficient fine-tuning.;impact[cit=1.0,rec=0.2,ven=1.0,ttl=1.0]",
        features={"relevance": 0.95, "llm": 0.95, "citation": 1.0, "normalized": 0.9},
    )
    ranking = JudgeResult(scored=[scored], selected=[paper], paper_ids=[paper.paper_id])
    understanding = UnderstandingResult(
        raw_question="LoRA", intent="method", slots={"topic": "LoRA"}
    )
    with patch("scholar_ir.organize.graph.build_citation_graph", return_value={}):
        result = organize(understanding, ranking, {"build_graph": False})
    item = result.items[0]
    assert "selection_reason" in item
    assert "Core LoRA paper" in item["selection_reason"]
    assert item["match_reasons"][0] == item["selection_reason"]


if __name__ == "__main__":
    test_extract_llm_reason_from_debug_string()
    test_extract_expanded_llm_reason()
    test_extract_keyword_only_returns_empty()
    test_selection_reason_prefers_llm_text()
    test_selection_reason_heuristic_without_llm()
    test_match_reasons_nl_then_evidence()
    test_organize_item_has_selection_reason()
    print("organize_reasons tests passed")
