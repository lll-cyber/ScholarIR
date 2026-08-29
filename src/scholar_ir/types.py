"""Shared data types for ScholarIR (see 技术路线/核心模块IO约定.md)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PaperRef:
    paper_id: str
    title: str = ""
    abstract: str = ""
    year: Optional[int] = None
    source: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubQuery:
    qid: str
    text: str
    channel: str = "keyword"  # keyword | semantic | metadata
    filters: Dict[str, Any] = field(default_factory=dict)
    # transform: how text was produced (not research facet)
    angle: str = "core"  # core|synonym|abbrev|entity|metadata|raw|conceptual
    mode: str = "lexical"  # lexical | decomposition | semantic
    modifiers: List[str] = field(default_factory=list)  # e.g. ["survey"]
    angle_source: str = ""  # trace: slots.terms[0].synonyms[0]


@dataclass
class UnderstandingResult:
    raw_question: str
    intent: str = ""
    slots: Dict[str, Any] = field(default_factory=dict)
    relevance_criteria: List[Dict[str, Any]] = field(default_factory=list)
    sub_queries: List[SubQuery] = field(default_factory=list)
    # Stage-1 flow steps for logging / eval (not sent to retrieval)
    trace: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RetrievalResult:
    candidates: List[PaperRef] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredPaper:
    paper: PaperRef
    score: float
    reason: str = ""


@dataclass
class JudgeResult:
    scored: List[ScoredPaper] = field(default_factory=list)
    selected: List[PaperRef] = field(default_factory=list)
    paper_ids: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    understanding: UnderstandingResult
    retrieval: RetrievalResult
    judge: JudgeResult
    paper_ids: List[str]
    metrics_local: Dict[str, Any] = field(default_factory=dict)
