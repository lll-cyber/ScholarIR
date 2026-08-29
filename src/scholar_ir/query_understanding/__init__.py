"""Stage (1) 查询理解与分解.

解析意图与槽位、术语扩写，组装可检索子查询。
"""

from .base import understand
from .slots import (
    INTENTS,
    LIGHT_SLOT_KEYS,
    empty_slots,
    normalize_round1_output,
    slots_to_filters,
)

__all__ = [
    "understand",
    "INTENTS",
    "LIGHT_SLOT_KEYS",
    "empty_slots",
    "normalize_round1_output",
    "slots_to_filters",
]
