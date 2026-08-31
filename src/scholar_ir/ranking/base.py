"""Stage (3) ranking — sort + truncate only.

Filter now produces the blended + cross-intent-normalized final score
(relevance + impact sub-signals merged in filter_papers; ``.score`` is the
normalized value with pass bar ≈ 0.5). This stage is responsible for:

  - Dropping papers below the configured threshold.
  - Applying the arxiv_only truncation policy (so non-arXiv ids do not crowd
    out arXiv papers when paired with PaSa gold).
  - Truncating to max_return.

The legacy `weights`/`use_embedding`/`threshold` options are kept for backward
compatibility but no longer alter the score (filter is the single source of
truth). Passing `threshold=0.0` and `max_return` here is the recommended call.
"""

from __future__ import annotations

from typing import Any, Dict

from scholar_ir.eval import is_arxiv_id
from scholar_ir.types import JudgeResult


def rank(
    understanding,  # UnderstandingResult — kept for signature parity
    filter_result: JudgeResult,
    options: Dict[str, Any] | None = None,
) -> JudgeResult:
    """Apply threshold + max_return + arxiv_only to filter's ranked pool.

    Args:
        understanding: Query understanding result (signature parity; unused).
        filter_result: Filter output (each ScoredPaper already holds the
            blended final score in `.score`).
        options:
            - threshold: minimum normalized score to keep (default 0.0; filter
              has already applied its intent-aware pass bar at ≈0.5).
              Prefer ``threshold=0.5`` if re-applying the filter bar here.
            - max_return: maximum papers to return (default 20).
            - arxiv_only: if True, drop non-arXiv ids before truncation.
            - weights, use_embedding: legacy, ignored.

    Returns:
        JudgeResult with sorted scored/selected/paper_ids.
    """
    _ = understanding  # unused; kept for pipeline signature parity
    options = options or {}
    max_return = int(options.get("max_return", 20))
    threshold = float(options.get("threshold", 0.0))
    arxiv_only = bool(options.get("arxiv_only", False))
    # Legacy options: accepted but no longer alter the score.
    _ = options.get("weights")
    _ = options.get("use_embedding", True)

    pool = list(filter_result.scored)
    pool.sort(key=lambda s: s.score, reverse=True)

    survivors = [s.paper for s in pool if s.score >= threshold]
    if arxiv_only:
        survivors = [
            p for p in survivors if p.paper_id and is_arxiv_id(p.paper_id)
        ]
    selected = survivors[:max_return]

    return JudgeResult(
        scored=pool,
        selected=selected,
        paper_ids=[p.paper_id for p in selected if p.paper_id],
    )