"""SPAR vendor package. Import submodules explicitly to avoid heavy deps at import time.

Examples:
    from scholar_ir.vendor_spar.utils import keep_letters
    from scholar_ir.vendor_spar import instruction
    from scholar_ir.vendor_spar import api_web  # needs biopython, arxiv, ...
"""

from .utils import fetch_string, get_md5, keep_letters

__all__ = [
    "fetch_string",
    "get_md5",
    "keep_letters",
]
