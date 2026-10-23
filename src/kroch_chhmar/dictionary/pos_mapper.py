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

# Common aliases from PAN Localization, khPOS, NICT ALT NOVA, RAC Khmer Dict 2022, Wiktionary, Chuon Nath, and khmer-nltk
_RAW_POS_ALIASES: dict[str, str] = {
    # Nouns (including khPOS AB=abbreviation, RPN=relational noun, M=classifier, RAC ន., ប.)
    "N": "NOUN",
    "NN": "NOUN",
    "NNS": "NOUN",
    "NC": "NOUN",
    "NOUN": "NOUN",
    "AB": "NOUN",
    "RPN": "NOUN",
    "M": "NOUN",
    "CLF": "NOUN",
    "CLASSIFIER": "NOUN",
    "នាម": "NOUN",
    "ន.": "NOUN",
    "ន": "NOUN",
    "ប.": "NOUN",
    "ប": "NOUN",
    # Proper nouns (including khPOS PN)
    "NP": "PROPN",
    "NNP": "PROPN",
    "PN": "PROPN",
    "PROPN": "PROPN",
    "អសា.": "PROPN",
    # Verbs (including khPOS VCOM=compound verb, RAC កិ.)
    "V": "VERB",
    "VB": "VERB",
    "VBD": "VERB",
    "VCOM": "VERB",
    "VERB": "VERB",
    "កិរិយា": "VERB",
    "កិ.": "VERB",
    "កិ": "VERB",
    # Adjectives (including khPOS VB_JJ, NOVA 'a', RAC គុ.)
    "A": "ADJ",
    "ADJ": "ADJ",
    "JJ": "ADJ",
    "VB_JJ": "ADJ",
    "VA": "ADJ",
    "គុណនាម": "ADJ",
    "គុ.": "ADJ",
    "គុ": "ADJ",
    # Adverbs (including RAC គុ.កិ. / កិ.វិ.)
    "ADV": "ADV",
    "RB": "ADV",
    "គុណកិរិយា": "ADV",
    "គុ.កិ.": "ADV",
    "គុ.កិ": "ADV",
    "កិ.វិ.": "ADV",
    # Pronouns (including NOVA 'n-', RAC សព្វ.)
    "N-": "PRON",
    "PRO": "PRON",
    "PRON": "PRON",
    "PRP": "PRON",
    "សព្វនាម": "PRON",
    "សព្វ.": "PRON",
    # Determiners (including NOVA 'a-')
    "A-": "DET",
    "DET": "DET",
    "DT": "DET",
    # Adpositions / Prepositions (including NOVA 'o', RAC ធ.)
    "O": "ADP",
    "PREP": "ADP",
    "IN": "ADP",
    "ADP": "ADP",
    "ធ្នាក់": "ADP",
    "ធ.": "ADP",
    "ធ": "ADP",
    # Numerals (including NOVA '1', khPOS CD, RAC សំ.)
    "1": "NUM",
    "NUM": "NUM",
    "CD": "NUM",
    "លេខ": "NUM",
    "សំ.": "NUM",
    # Conjunctions (including RAC ឈ., និ.)
    "CONJ": "CCONJ",
    "CC": "CCONJ",
    "CCONJ": "CCONJ",
    "SCONJ": "SCONJ",
    "ឈ្នាប់": "CCONJ",
    "ឈ.": "CCONJ",
    "និ.": "CCONJ",
    # Particles (including khPOS PA, DBL=reduplication ៗ, QT=question particle, NOVA 'o-')
    "O-": "PART",
    "PA": "PART",
    "QT": "PART",
    "DBL": "PART",
    "PART": "PART",
    "PRT": "PART",
    "RP": "PART",
    # Auxiliaries (including NOVA 'v-')
    "V-": "AUX",
    "AUX": "AUX",
    "MD": "AUX",
    # Interjections (including NOVA '+', RAC ឧ.)
    "+": "INTJ",
    "INTJ": "INTJ",
    "UH": "INTJ",
    "ឧទានសព្ទ": "INTJ",
    "ឧ.": "INTJ",
    "ឧ": "INTJ",
    # Punctuation & Symbols (including khPOS KAN=។ and NOVA '.')
    ".": "PUNCT",
    "KAN": "PUNCT",
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
    Supports multi-tag strings like `'គុ.កិ., គុ.'` or `'គុ. ឬ ន.'` by taking the primary tag.
    """
    if raw_pos:
        cleaned = raw_pos.strip().upper()
        if cleaned in _RAW_POS_ALIASES:
            return _RAW_POS_ALIASES[cleaned]
        if cleaned in _POS_TO_ID:
            return cleaned
        # Handle multi-POS strings separated by comma, slash, or 'ឬ' (or)
        for sep in (",", "/", "ឬ", "និង", ";"):
            if sep in cleaned:
                first_part = cleaned.split(sep, 1)[0].strip()
                if first_part in _RAW_POS_ALIASES:
                    return _RAW_POS_ALIASES[first_part]
                if first_part in _POS_TO_ID:
                    return first_part

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
