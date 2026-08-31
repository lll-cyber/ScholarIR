"""Stage (2) 自主搜索：多源检索（arxiv / openalex / semantic）+ 迭代校准。

与 `filter` 共同构成「搜索策略」阶段。
"""

from .base import paper_dict_to_ref, retrieve
from .iterate import retrieve_iterative
from . import s2_client
from .adapt import (
    adapt_arxiv,
    adapt_openalex,
    adapt_semantic,
    fetch_arxiv_by_ids,
    search_arxiv,
    search_openalex,
    search_semantic,
)

__all__ = [
    "retrieve",
    "retrieve_iterative",
    "paper_dict_to_ref",
    "s2_client",
    "adapt_semantic",
    "search_semantic",
    "adapt_openalex",
    "search_openalex",
    "adapt_arxiv",
    "search_arxiv",
    "fetch_arxiv_by_ids",
]
