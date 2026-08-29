"""ScholarIR: academic paper-set retrieval agent.

Pipeline stages:
  (1) query_understanding  查询理解与分解
  (2) search + filter      自主搜索与候选过滤
  (3) ranking              论文综合排序
  (4) organize             搜索结果归纳整理
"""

from .pipeline import run
from .types import (
    JudgeResult,
    PaperRef,
    PipelineResult,
    RetrievalResult,
    SubQuery,
    UnderstandingResult,
)

__all__ = [
    "run",
    "PaperRef",
    "SubQuery",
    "UnderstandingResult",
    "RetrievalResult",
    "JudgeResult",
    "PipelineResult",
]

__version__ = "0.2.0"
