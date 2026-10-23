"""Dictionary generation, POS mapping, cost estimation, and multi-granular splitting."""

from sudachi_khmer.dictionary.compound_splitter import CompoundSplitter, SplitResolution
from sudachi_khmer.dictionary.cost_estimator import CostEstimator, LexiconStats
from sudachi_khmer.dictionary.csv_writer import (
    SudachiCSVEntry,
    build_system_dic,
    build_user_dic,
    write_sudachi_csv,
)
from sudachi_khmer.dictionary.pos_mapper import UD_POS_TAGS, map_to_ud_pos, pos_to_id, pos_to_sudachi_tuple

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
    "write_sudachi_csv",
]
