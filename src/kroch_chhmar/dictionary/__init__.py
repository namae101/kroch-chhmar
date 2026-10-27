"""Dictionary generation, POS mapping, cost estimation, and multi-granular splitting."""

from kroch_chhmar.dictionary.compound_splitter import CompoundSplitter, SplitResolution
from kroch_chhmar.dictionary.cost_estimator import CostEstimator, LexiconStats
from kroch_chhmar.dictionary.csv_writer import (
    SudachiCSVEntry,
    build_system_dic,
    build_user_dic,
    summarize_split_topology,
    validate_split_references,
    write_sudachi_csv,
)
from kroch_chhmar.dictionary.pos_mapper import UD_POS_TAGS, map_to_ud_pos, pos_to_id, pos_to_sudachi_tuple

__all__ = [
    "CompoundSplitter",
    "CostEstimator",
    "LexiconStats",
    "SplitResolution",
    "SudachiCSVEntry",
    "UD_POS_TAGS",
    "build_system_dic",
    "build_user_dic",
    "map_to_ud_pos",
    "pos_to_id",
    "pos_to_sudachi_tuple",
    "summarize_split_topology",
    "validate_split_references",
    "write_sudachi_csv",
]
