"""Importers for local lexicons (CSV/TSV/wordlist), segmented text, CoNLL-U, and seed data."""

from __future__ import annotations

import csv
from pathlib import Path

from sudachi_khmer.dictionary.cost_estimator import CostEstimator
from sudachi_khmer.normalizer import normalize_khmer


def ingest_lexicon_tsv(
    path: str | Path,
    estimator: CostEstimator,
    delimiter: str | None = None,
) -> int:
    """
    Ingest a lexicon file (TSV, CSV, or plain 1-word-per-line wordlist).
    Supported columns:
      - 1 column: `surface`
      - 2 columns: `surface`, `pos` (or `surface`, `freq` if 2nd column is numeric)
      - 3+ columns: `surface`, `pos`, `freq`, `[reading]`
    Returns the number of entries ingested.
    """
    p = Path(path)
    if not p.exists():
        return 0

    if delimiter is None:
        delimiter = "\t" if p.suffix.lower() in (".tsv", ".txt") else ","

    count = 0
    with p.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if delimiter in line:
                parts = [col.strip() for col in next(csv.reader([line], delimiter=delimiter))]
            else:
                parts = [line]

            surface = normalize_khmer(parts[0])
            if not surface:
                continue

            pos: str | None = None
            freq: int = 0
            reading: str | None = None

            if len(parts) == 2:
                if parts[1].isdigit():
                    freq = int(parts[1])
                else:
                    pos = parts[1]
            elif len(parts) >= 3:
                pos = parts[1] if parts[1] and parts[1] != "*" else None
                if parts[2].isdigit():
                    freq = int(parts[2])
                if len(parts) >= 4 and parts[3] and parts[3] != "*":
                    reading = parts[3]

            estimator.add_lexicon_entry(surface=surface, pos=pos, freq=freq, reading=reading)
            count += 1

    return count


def ingest_segmented_text_file(
    path: str | Path,
    estimator: CostEstimator,
) -> int:
    """
    Ingest a pre-segmented Khmer corpus where words are separated by spaces,
    Zero-Width Spaces (`\\u200b`), or `|` delimiters, with optional `word/POS` tags.
    Returns the number of sentences processed.
    """
    p = Path(path)
    if not p.exists():
        return 0

    sentences = 0
    with p.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            # Replace ZWSP and pipe delimiters with spaces
            unified = line.replace("\u200b", " ").replace("|", " ")
            raw_tokens = [tok.strip() for tok in unified.split() if tok.strip()]
            if not raw_tokens:
                continue

            parsed_tokens: list[tuple[str, str | None]] = []
            for tok in raw_tokens:
                if "/" in tok and len(tok) > 1 and not tok.startswith("/"):
                    surf, pos = tok.rsplit("/", 1)
                    parsed_tokens.append((surf, pos))
                else:
                    parsed_tokens.append((tok, None))

            estimator.add_segmented_sentence(parsed_tokens)
            sentences += 1

    return sentences


def ingest_conllu_file(
    path: str | Path,
    estimator: CostEstimator,
) -> int:
    """
    Ingest a Universal Dependencies CoNLL-U file (`.conllu`).
    Extracts `(FORM, UPOS)` sequences per sentence.
    Returns the number of sentences ingested.
    """
    p = Path(path)
    if not p.exists():
        return 0

    sentences = 0
    current_sentence: list[tuple[str, str | None]] = []

    with p.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                if current_sentence:
                    estimator.add_segmented_sentence(current_sentence)
                    sentences += 1
                    current_sentence = []
                continue
            if line.startswith("#"):
                continue

            cols = line.split("\t")
            if len(cols) < 4:
                continue
            token_id = cols[0]
            # Skip multi-word token range lines (e.g. 1-2) or empty nodes (e.g. 1.1)
            if "-" in token_id or "." in token_id:
                continue

            form = cols[1].strip()
            upos = cols[3].strip()
            if form:
                current_sentence.append((form, upos if upos != "_" else None))

    if current_sentence:
        estimator.add_segmented_sentence(current_sentence)
        sentences += 1

    return sentences


def load_seed_dataset(
    estimator: CostEstimator,
    seed_dir: str | Path | None = None,
) -> tuple[int, int]:
    """
    Load the built-in Khmer seed lexicon and segmented corpus from `data/seed/`.
    Returns `(num_lexicon_entries, num_corpus_sentences)`.
    """
    if seed_dir is None:
        # Resolve relative to repository root or package
        repo_seed = Path(__file__).resolve().parents[3] / "data" / "seed"
        seed_dir = repo_seed

    sdir = Path(seed_dir)
    lex_count = 0
    sent_count = 0

    for tsv_file in sorted(sdir.glob("*.tsv")):
        lex_count += ingest_lexicon_tsv(tsv_file, estimator)
    for txt_file in sorted(sdir.glob("*.txt")):
        sent_count += ingest_segmented_text_file(txt_file, estimator)
    for conllu_file in sorted(sdir.glob("*.conllu")):
        sent_count += ingest_conllu_file(conllu_file, estimator)

    return lex_count, sent_count
