"""Sudachi-Khmer: Dictionary generation workflow and KCC-aware word segmentation plugin."""

from sudachi_khmer.normalizer import (
    count_kccs,
    is_kcc_boundary,
     is_kcc_dependent_start,
    iter_kccs,
    normalize_khmer,
)
from sudachi_khmer.tokenizer import KhmerMorpheme, KhmerTokenizer, SplitMode

__version__ = "0.1.0"

__all__ = [
    "KhmerMorpheme",
    "KhmerTokenizer",
    "SplitMode",
    "count_kccs",
    "is_kcc_boundary",
    "is_kcc_dependent_start",
    "iter_kccs",
    "normalize_khmer",
]
