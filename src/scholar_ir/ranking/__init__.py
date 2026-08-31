"""Stage (3) paper ranking — sort + truncate only.

Final scores come from filter (relevance + impact + cross-intent normalize).
This package does not re-weight or run embeddings.

Naming note: the stage is still called ``ranking`` for pipeline / 赛题对齐;
behaviorally it is a **select/truncate** layer (threshold + arxiv_only + max_return).
"""

from .base import rank

__all__ = ["rank"]
