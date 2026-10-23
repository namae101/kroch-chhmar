"""Kroch-Chhmar: Dictionary generation workflow and KCC-aware word segmentation plugin."""

from kroch_chhmar.normalizer import (
    count_kccs,
    is_kcc_boundary,
     is_kcc_dependent_start,
    iter_kccs,
    normalize_khmer,
)
from kroch_chhmar.tokenizer import KhmerMorpheme, KhmerTokenizer, SplitMode

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
