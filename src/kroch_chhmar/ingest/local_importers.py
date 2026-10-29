"""Importers for local lexicons (CSV/TSV/wordlist), segmented text, khPOS, ALT NOVA, CoNLL-U, and seed data."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from pathlib import Path

from kroch_chhmar.dictionary.compound_splitter import CompoundSplitter, SplitResolution
from kroch_chhmar.dictionary.cost_estimator import CostEstimator
from kroch_chhmar.dictionary.pos_mapper import map_to_ud_pos
from kroch_chhmar.normalizer import normalize_khmer

_ZWSP = "\u200b"


# ---------------------------------------------------------------------------
# Holdout filtering (prevents exact test sentences from leaking into training)
# ---------------------------------------------------------------------------


def _strip_pos_suffix(token: str) -> str:
    """Strip a trailing ``/POS`` annotation from a gold token."""
    tok = token.strip()
    if "/" in tok and len(tok) > 1 and not tok.startswith("/"):
        tok = tok.rsplit("/", 1)[0]
    return tok.strip()


def sentence_surface_key(tokens: Iterable[str | tuple[str, str | None]]) -> str:
    """
    Build a normalization-independent key for a segmented sentence.

    Compound markers (``_``/``~``), bound-suffix markers (``^``), POS suffixes
    (``/NN``), zero-width spaces, and Unicode encoding variants are all removed
    so that the same underlying text compares equal across corpora and gold
    files.
    """
    parts: list[str] = []
    for item in tokens:
        raw_surface = item[0] if isinstance(item, tuple) else item
        surface = _strip_pos_suffix(str(raw_surface)).replace("^", "")
        for piece in re.split(r"[_~]+", surface):
            normalized = normalize_khmer(piece)
            if normalized:
                parts.append(normalized)
    return "".join(parts)


class HoldoutFilter:
    """
    Exact-sentence holdout filter.

    khPOS's upstream ``train.all2`` file starts with the entire ``CLOSE-TEST``
    split, so simply skipping the ``khpos_close_test.pos`` file is not enough:
    the same sentences are already present in the training file. This filter
    removes any sentence whose normalized surface text matches a held-out
    sentence *before* its tokens, compound overrides, or lexicon sub-parts are
    committed to the estimator.
    """

    def __init__(self, keys: Iterable[str] | None = None) -> None:
        self.keys: set[str] = set(keys or ())
        self.loaded_sentences = 0
        self.skipped = 0

    @classmethod
    def from_files(cls, paths: Iterable[str | Path]) -> HoldoutFilter:
        """Load held-out sentences from space/ZWSP/pipe-delimited gold files."""
        holdout = cls()
        for raw_path in paths:
            path = Path(raw_path)
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    tokens = [
                        tok
                        for tok in line.replace(_ZWSP, " ").replace("|", " ").split()
                        if tok.strip()
                    ]
                    if tokens:
                        holdout.add_sentence(tokens)
                        holdout.loaded_sentences += 1
        return holdout

    def add_sentence(self, tokens: Iterable[str | tuple[str, str | None]]) -> None:
        key = sentence_surface_key(tokens)
        if key:
            self.keys.add(key)

    def is_held_out(self, tokens: Iterable[str | tuple[str, str | None]]) -> bool:
        key = sentence_surface_key(tokens)
        return bool(key) and key in self.keys

    def record_skip(self) -> None:
        self.skipped += 1

    def __len__(self) -> int:
        return len(self.keys)


# ---------------------------------------------------------------------------
# RAC 2022 pronunciation cleaning
# ---------------------------------------------------------------------------


def clean_rac_pronunciation(raw: str | None) -> str | None:
    """
    Extract a clean pronunciation from RAC 2022 ``t_pron`` values.

    RAC readings frequently look like ``[ក័ន-ដៈ-កៈ-ស័ញ-ញ៉ា] ឬ [កន់-ដក-ស័ញ-ញ៉ា]``
    or carry trailing notes such as ``] (ជើង ត),``. The previous implementation
    only stripped outer brackets, leaving annotation fragments in the compiled
    dictionary's ``reading_form`` column.
    """
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None

    bracket = re.search(r"\[([^\]]+)\]", text)
    if bracket:
        text = bracket.group(1)
    else:
        text = text.split("]", 1)[0]
        text = text.split("(", 1)[0]

    text = text.split(" ឬ ", 1)[0]
    text = re.sub(r"\s+", " ", text).strip().strip("[]").rstrip(",;").strip()
    return text or None


# ---------------------------------------------------------------------------
# Lexicon importers
# ---------------------------------------------------------------------------


def ingest_lexicon_tsv(
    path: str | Path,
    estimator: CostEstimator,
    delimiter: str | None = None,
    default_pos: str | None = None,
    use_frequencies: bool = False,
) -> int:
    """
    Ingest a lexicon file (TSV, CSV, or plain 1-word-per-line wordlist).
    Supported columns:
      - 1 column: `surface`
      - 2 columns: `surface`, `pos` (or `surface`, `freq` if 2nd column is numeric, as in `khmerlbdict`)
      - 3+ columns: `surface`, `pos`, `freq`, `[reading]`
    Returns the number of entries ingested.

    `use_frequencies=True` keeps raw wordlist counts (e.g. `khmerlbdict`) as
    emission-cost evidence. This is intended for the permissive, corpus-free
    dictionary profile; the default full profile deliberately zeroes those
    uncalibrated counts so they cannot dilute sentence-corpus NLL probabilities.
    """
    p = Path(path)
    if not p.exists():
        return 0

    if delimiter is None:
        delimiter = "\t" if p.suffix.lower() in (".tsv", ".txt") else ","

    # Infer default POS for proper noun lists (names, places, villages) from khmerlbdict
    stem = p.stem.lower()
    if default_pos is None and any(k in stem for k in ("names", "places", "villages")):
        default_pos = "PROPN"

    # Uncalibrated raw counts in `khmerlbdict_*` (e.g. 6.4M counts in seafreq) should not be
    # added directly into sentence-corpus NLL probabilities unless the caller explicitly
    # requests a wordlist-only (permissive) build.
    ignore_numeric_freq = stem.startswith("khmerlbdict_") and not use_frequencies

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

            pos: str | None = default_pos
            freq: int = 0
            reading: str | None = None

            if len(parts) == 2:
                if parts[1].isdigit():
                    freq = 0 if ignore_numeric_freq else int(parts[1])
                else:
                    pos = parts[1]
            elif len(parts) >= 3:
                pos = parts[1] if parts[1] and parts[1] != "*" else default_pos
                if parts[2].isdigit():
                    freq = 0 if ignore_numeric_freq else int(parts[2])
                if len(parts) >= 4 and parts[3] and parts[3] != "*":
                    reading = parts[3]

            estimator.add_lexicon_entry(surface=surface, pos=pos, freq=freq, reading=reading)
            count += 1

    return count


def ingest_rac_dictionary_csv(
    path: str | Path,
    estimator: CostEstimator,
    splitter: CompoundSplitter | None = None,
) -> int:
    """
    Ingest the Royal Academy of Cambodia (RAC) 2022 Khmer Dictionary CSV
    (`RAC-Khmer-Dict-2022.csv` from `seanghay/khmer-dictionary-44k`).
    Extracts headwords (`t_main`), subwords/compounds (`t_subword`), POS tags (`t_pos`),
    and pronunciation readings (`t_pron`), and links `t_subword` compounds to their `t_main`
    base words for Mode B/C multi-granular splitting.
    """
    p = Path(path)
    if not p.exists():
        return 0

    count = 0
    last_main = ""
    with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_main = (row.get("t_main") or "").strip()
            raw_sub = (row.get("t_subword") or "").strip()
            raw_pos = (row.get("t_pos") or "").strip()
            raw_pron = clean_rac_pronunciation(row.get("t_pron"))

            if raw_main:
                last_main = normalize_khmer(raw_main)

            target_raw = raw_sub if raw_sub else raw_main
            surface = normalize_khmer(target_raw)
            if not surface:
                continue

            ud_pos = map_to_ud_pos(raw_pos, surface) if raw_pos else None
            reading = normalize_khmer(raw_pron) if raw_pron else None

            estimator.add_lexicon_entry(
                surface=surface,
                pos=ud_pos,
                freq=0,
                reading=reading,
                is_canonical=True,
            )
            count += 1

            # If this is a subword derived from `last_main` (e.g. t_main='កក', t_subword='កកស្ទះ'),
            # record last_main in base_vocab so CompoundSplitter can split `last_main + remainder`
            if raw_sub and last_main and splitter is not None:
                splitter.base_vocab.add(last_main)
                if (
                    surface.startswith(last_main)
                    and len(surface) > len(last_main)
                    and surface not in splitter.overrides
                ):
                    rem = surface[len(last_main) :]
                    if rem in splitter.base_vocab:
                        splitter.overrides[surface] = SplitResolution(
                            surface=surface,
                            split_type="B",
                            split_a=[last_main, rem],
                            override_pos=ud_pos,
                        )
            elif not raw_sub and last_main and splitter is not None:
                splitter.base_vocab.add(last_main)

    return count


# ---------------------------------------------------------------------------
# Corpus importers
# ---------------------------------------------------------------------------


def ingest_segmented_text_file(
    path: str | Path,
    estimator: CostEstimator,
    splitter: CompoundSplitter | None = None,
    holdout: HoldoutFilter | None = None,
) -> int:
    """
    Ingest a pre-segmented Khmer corpus where words are separated by spaces,
    Zero-Width Spaces (`\\u200b`), or `|` delimiters, with optional `word/POS` tags.
    Also supports `ye-kyaw-thu/khPOS` compound markers (`_` and `~`), automatically
    extracting multi-granular Mode B/C compound splits when `splitter` is provided!

    If `holdout` is provided, sentences whose normalized surface text matches a
    held-out sentence are skipped **before** their sub-parts or compound
    overrides are added to the estimator/splitter.
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
            unified = line.replace(_ZWSP, " ").replace("|", " ")
            raw_tokens = [tok.strip() for tok in unified.split() if tok.strip()]
            if not raw_tokens:
                continue

            parsed_tokens: list[tuple[str, str | None]] = []
            compound_actions: list[tuple[str, str | None, list[str]]] = []

            for tok in raw_tokens:
                if "/" in tok and len(tok) > 1 and not tok.startswith("/"):
                    raw_surf, raw_pos = tok.rsplit("/", 1)
                else:
                    raw_surf, raw_pos = tok, None

                # Strip khPOS bound derivational suffix marker '^' so bound stems
                # (e.g., `សេដ្ឋ^កិច្ច` -> `សេដ្ឋកិច្ច`, `កសិ^កម្ម` -> `កសិកម្ម`) stay intact
                # as single lexical words instead of polluting base_vocab with bound stems (`កសិ`, `សេដ្ឋ`).
                if "^" in raw_surf and len(raw_surf) > 1:
                    raw_surf = raw_surf.replace("^", "")

                # Handle khPOS compound sub-token separators ('_' and '~')
                if ("_" in raw_surf or "~" in raw_surf) and len(raw_surf) > 1:
                    sub_parts = [
                        normalize_khmer(sp)
                        for sp in re.split(r"[_~]+", raw_surf)
                        if normalize_khmer(sp)
                    ]
                    if len(sub_parts) >= 2 and all(
                        any(0x1780 <= ord(c) <= 0x17FF for c in sp) for sp in sub_parts
                    ):
                        full_surf = normalize_khmer("".join(sub_parts))
                        ud_pos = map_to_ud_pos(raw_pos, full_surf)
                        parsed_tokens.append((full_surf, ud_pos))
                        compound_actions.append((full_surf, ud_pos, sub_parts))
                        continue

                parsed_tokens.append((raw_surf, raw_pos))

            if not parsed_tokens:
                continue

            # Holdout check happens before any lexicon/compound side effects.
            if holdout is not None and holdout.is_held_out(parsed_tokens):
                holdout.record_skip()
                continue

            # Register sub-parts in base vocabulary & estimator
            for full_surf, action_pos, sub_parts in compound_actions:
                for sp in sub_parts:
                    estimator.add_lexicon_entry(sp, pos=None, freq=1)
                    if splitter is not None:
                        splitter.base_vocab.add(sp)

                if splitter is not None and full_surf not in splitter.overrides:
                    mode = "B" if len(sub_parts) == 2 else "C"
                    splitter.overrides[full_surf] = SplitResolution(
                        surface=full_surf,
                        split_type=mode,
                        split_a=sub_parts,
                        split_b=sub_parts if mode == "C" else [],
                        override_pos=action_pos,
                    )

            estimator.add_segmented_sentence(parsed_tokens)
            sentences += 1

    return sentences


def ingest_alt_nova_files(
    tok_path: str | Path,
    tag_path: str | Path,
    estimator: CostEstimator,
    splitter: CompoundSplitter | None = None,
    holdout: HoldoutFilter | None = None,
) -> int:
    """
    Ingest the NICT/NIPTICT Khmer ALT NOVA parallel `.km-tok.nova` and `.km-tag.nova` files.
    Parses both leaf words (`Mode A`) and bracketed compounds (`n[n v]n` -> `Mode B/C`)
    along with their POS tags.
    Returns the number of sentences processed.
    """
    tok_p = Path(tok_path)
    tag_p = Path(tag_path)
    if not tok_p.exists() or not tag_p.exists():
        return 0

    tok_lines = tok_p.read_text(encoding="utf-8", errors="replace").splitlines()
    tag_lines = tag_p.read_text(encoding="utf-8", errors="replace").splitlines()

    sentences = 0
    for tok_line, tag_line in zip(tok_lines, tag_lines, strict=False):
        if "\t" not in tok_line or "\t" not in tag_line:
            continue
        _, tok_str = tok_line.split("\t", 1)
        _, tag_str = tag_line.split("\t", 1)

        words = tok_str.split()
        # Match either bracketed compound `outer[inner1 inner2]outer` or single tag
        chunks = re.findall(r"\S+\[[^\]]+\]\S+|\S+", tag_str)

        sentence_pairs: list[tuple[str, str | None]] = []
        leaf_entries: set[tuple[str, str]] = set()
        single_words: set[str] = set()
        word_idx = 0

        for chunk in chunks:
            if "[" in chunk:
                outer_tag = chunk.split("[", 1)[0]
                inner_match = re.search(r"\[([^\]]+)\]", chunk)
                inner_tags = inner_match.group(1).split() if inner_match else []
                k = len(inner_tags)
                if word_idx + k > len(words) or k == 0:
                    break

                sub_words = [normalize_khmer(w) for w in words[word_idx : word_idx + k]]
                word_idx += k

                # Register individual leaf words with their inner POS tags
                for sw, itag in zip(sub_words, inner_tags, strict=False):
                    if sw:
                        leaf_entries.add((sw, map_to_ud_pos(itag, sw)))

                # If the bracketed phrase matches an established lexical compound in `splitter.overrides`
                # (e.g. from khPOS or compounds.yaml), count it as a compound token in the sentence;
                # otherwise emit the leaf words so syntactic NPs do not greedily merge independent words.
                clean_subs = [sw for sw in sub_words if sw]
                compound_surf = normalize_khmer("".join(clean_subs))
                if (
                    len(clean_subs) >= 2
                    and splitter is not None
                    and compound_surf in splitter.overrides
                ):
                    ud_outer = map_to_ud_pos(outer_tag, compound_surf)
                    sentence_pairs.append((compound_surf, ud_outer))
                else:
                    for sw, itag in zip(sub_words, inner_tags, strict=False):
                        if sw:
                            sentence_pairs.append((sw, map_to_ud_pos(itag, sw)))
            else:
                if word_idx >= len(words):
                    break
                w = normalize_khmer(words[word_idx])
                word_idx += 1
                if w:
                    sentence_pairs.append((w, map_to_ud_pos(chunk, w)))
                    single_words.add(w)

        if not sentence_pairs:
            continue

        if holdout is not None and holdout.is_held_out(sentence_pairs):
            holdout.record_skip()
            continue

        # Commit leaf vocabulary only after the holdout check.
        if splitter is not None:
            for word in sorted(single_words):
                splitter.base_vocab.add(word)
        for sw, tag in sorted(leaf_entries):
            estimator.add_lexicon_entry(sw, pos=tag, freq=0)
            if splitter is not None:
                splitter.base_vocab.add(sw)

        estimator.add_segmented_sentence(sentence_pairs)
        sentences += 1

    return sentences


def ingest_conllu_file(
    path: str | Path,
    estimator: CostEstimator,
    holdout: HoldoutFilter | None = None,
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

    def _commit() -> None:
        nonlocal sentences
        if not current_sentence:
            return
        if holdout is not None and holdout.is_held_out(current_sentence):
            holdout.record_skip()
            return
        estimator.add_segmented_sentence(current_sentence)
        sentences += 1

    with p.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                _commit()
                current_sentence = []
                continue
            if line.startswith("#"):
                continue

            cols = line.split("\t")
            if len(cols) < 4:
                continue
            token_id = cols[0]
            if "-" in token_id or "." in token_id:
                continue

            form = cols[1].strip()
            upos = cols[3].strip()
            if form:
                current_sentence.append((form, upos if upos != "_" else None))

    _commit()
    return sentences


def load_seed_dataset(
    estimator: CostEstimator,
    seed_dir: str | Path | None = None,
    splitter: CompoundSplitter | None = None,
    holdout: HoldoutFilter | None = None,
) -> tuple[int, int]:
    """
    Load the built-in Khmer seed lexicon and segmented corpus.
    Returns `(num_lexicon_entries, num_corpus_sentences)`.

    Seed files are resolved from the packaged ``resources/seed`` directory first
    (so the CLI works from an installed wheel), then from the repository's
    ``data/seed`` directory.
    """
    if seed_dir is None:
        package_seed = Path(__file__).resolve().parents[1] / "resources" / "seed"
        repo_seed = Path(__file__).resolve().parents[3] / "data" / "seed"
        seed_dir = package_seed if package_seed.exists() else repo_seed

    sdir = Path(seed_dir)
    lex_count = 0
    sent_count = 0

    for tsv_file in sorted(sdir.glob("*.tsv")):
        lex_count += ingest_lexicon_tsv(tsv_file, estimator)
    for txt_file in sorted(sdir.glob("*.txt")):
        sent_count += ingest_segmented_text_file(
            txt_file, estimator, splitter=splitter, holdout=holdout
        )
    for conllu_file in sorted(sdir.glob("*.conllu")):
        sent_count += ingest_conllu_file(conllu_file, estimator, holdout=holdout)

    return lex_count, sent_count
