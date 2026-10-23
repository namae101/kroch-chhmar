"""Universal Dependencies (UD) POS tagset mapping and Sudachi matrix ID assignment."""

from __future__ import annotations

# Standard 17 Universal Dependencies POS tags + 1 special state for dangling dependent OOVs
UD_POS_TAGS: tuple[str, ...] = (
    "NOUN",   # 0
    "PROPN",  # 1
    "VERB",   # 2
    "ADJ",    # 3
    "ADV",    # 4
    "PRON",   # 5
    "DET",    # 6
    "ADP",    # 7
    "NUM",    # 8
    "CCONJ",  # 9
    "SCONJ",  # 10
    "PART",   # 11
    "AUX",    # 12
    "INTJ",   # 13
    "PUNCT",  # 14
    "SYM",    # 15
    "X",      # 16
)

# Special ID 17 reserved in unk.def for KHMER_DEP (dangling dependent characters)
KHMER_DEP_POS_ID: int = 17
TOTAL_MATRIX_STATES: int = 18

_POS_TO_ID: dict[str, int] = {tag: idx for idx, tag in enumerate(UD_POS_TAGS)}

# Common aliases from PAN Localization, Wiktionary, Chuon Nath, and khmer-nltk
_RAW_POS_ALIASES: dict[str, str] = {
    # Nouns
    "N": "NOUN",
    "NN": "NOUN",
    "NNS": "NOUN",
    "NC": "NOUN",
    "NOUN": "NOUN",
    "CLF": "NOUN",
    "CLASSIFIER": "NOUN",
    "នាម": "NOUN",
    # Proper nouns
    "NP": "PROPN",
    "NNP": "PROPN",
    "PROPN": "PROPN",
    # Verbs
    "V": "VERB",
    "VB": "VERB",
    "VBD": "VERB",
    "VERB": "VERB",
    "កិរិយា": "VERB",
    # Adjectives
    "ADJ": "ADJ",
    "JJ": "ADJ",
    "VA": "ADJ",
    "គុណនាម": "ADJ",
    # Adverbs
    "ADV": "ADV",
    "RB": "ADV",
    "គុណកិរិយា": "ADV",
    # Pronouns
    "PRO": "PRON",
    "PRON": "PRON",
    "PRP": "PRON",
    "សព្វនាម": "PRON",
    # Determiners
    "DET": "DET",
    "DT": "DET",
    # Adpositions / Prepositions
    "PREP": "ADP",
    "IN": "ADP",
    "ADP": "ADP",
    "ធ្នាក់": "ADP",
    # Numerals
    "NUM": "NUM",
    "CD": "NUM",
    "M": "NUM",
    "លេខ": "NUM",
    # Conjunctions
    "CONJ": "CCONJ",
    "CC": "CCONJ",
    "CCONJ": "CCONJ",
    "SCONJ": "SCONJ",
    "ឈ្នាប់": "CCONJ",
    # Particles
    "PART": "PART",
    "PRT": "PART",
    "RP": "PART",
    # Auxiliaries
    "AUX": "AUX",
    "MD": "AUX",
    # Interjections
    "INTJ": "INTJ",
    "UH": "INTJ",
    "ឧទានសព្ទ": "INTJ",
    # Punctuation & Symbols
    "PUNCT": "PUNCT",
    "PUNC": "PUNCT",
    "SYM": "SYM",
    # Other / Unknown
    "X": "X",
    "UNK": "X",
    "FW": "X",
    "ETC": "X",
}


def map_to_ud_pos(raw_pos: str | None, surface: str = "") -> str:
    """
    Map a raw POS label (or infer from `surface` if missing) to a canonical UD POS tag.
    """
    if raw_pos:
        cleaned = raw_pos.strip().upper()
        if cleaned in _RAW_POS_ALIASES:
            return _RAW_POS_ALIASES[cleaned]
        if cleaned in _POS_TO_ID:
            return cleaned

    # Heuristics if POS tag is unspecified
    if surface:
        if all(0x17E0 <= ord(c) <= 0x17E9 or c.isdigit() for c in surface):
            return "NUM"
        if all(0x17D4 <= ord(c) <= 0x17DC or c in ".,!?;:()[]{}\"'-/«»" for c in surface):
            return "PUNCT"

    return "NOUN"


def pos_to_id(ud_pos: str) -> int:
    """Return the `matrix.def` state ID (0..16) for a UD POS tag."""
    tag = map_to_ud_pos(ud_pos)
    return _POS_TO_ID.get(tag, _POS_TO_ID["X"])


def pos_to_sudachi_tuple(ud_pos: str) -> tuple[str, str, str, str, str, str]:
    """Return the 6-element POS tuple required by Sudachi's CSV format."""
    tag = map_to_ud_pos(ud_pos)
    return (tag, "*", "*", "*", "*", "*")
