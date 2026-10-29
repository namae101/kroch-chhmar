"""Exact and near-duplicate sentence/lexicon overlap auditing.

The khPOS leak showed that a corpus can contain the same sentence in its
training and test splits. This module provides a reusable audit for the other
upstream corpora: normalized exact keys plus an optional MinHash/LSH estimate
of near-duplicate Jaccard similarity.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from kroch_chhmar.evaluate import load_gold_sentences
from kroch_chhmar.ingest.local_importers import sentence_surface_key
from kroch_chhmar.normalizer import normalize_khmer

_ZWSP = "\u200b"
_MERSENNE = (1 << 61) - 1


@dataclass
class OverlapExample:
    """One exact or near-duplicate pair (or duplicate group in internal mode)."""

    left: str
    right: str
    jaccard: float
    kind: str  # "exact" or "near"


@dataclass
class OverlapReport:
    """Summary of exact and near-duplicate overlap between two item collections."""

    mode: str
    left_items: int
    right_items: int
    left_unique: int
    right_unique: int
    exact_pairs: int
    near_pairs: int
    left_duplicate_groups: int
    right_duplicate_groups: int
    left_duplicate_items: int
    right_duplicate_items: int
    threshold: float
    near_enabled: bool
    examples: list[OverlapExample] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_nova_tokens(path: Path) -> list[str]:
    """Read the token column of an ALT NOVA ``*.km-tok.nova`` file."""
    items: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            if "\t" in line:
                line = line.split("\t", 1)[1]
            key = sentence_surface_key(line.split())
            if key:
                items.append(key)
    return items


def _read_conllu_tokens(path: Path) -> list[str]:
    items: list[str] = []
    sentence: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                if sentence:
                    key = sentence_surface_key(sentence)
                    if key:
                        items.append(key)
                    sentence = []
                continue
            if line.startswith("#"):
                continue
            columns = line.split("\t")
            if len(columns) < 2 or "-" in columns[0] or "." in columns[0]:
                continue
            sentence.append(columns[1])
    if sentence:
        key = sentence_surface_key(sentence)
        if key:
            items.append(key)
    return items


def _read_lexicon(path: Path) -> list[str]:
    """
    Read unique headwords from a TSV/CSV lexicon (RAC's ``t_main``/``t_subword`` aware).

    Lexicon rows are deduplicated because RAC repeats ``t_main`` for every
    subword row; those repeats describe dictionary structure, not duplicate
    sentences. Cross-lexicon overlap is still measured on the unique headwords.
    """
    items: list[str] = []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            dict_reader = csv.DictReader(handle)
            fields = dict_reader.fieldnames or []
            if "t_main" in fields:
                for row in dict_reader:
                    for column in ("t_main", "t_subword"):
                        surface = normalize_khmer((row.get(column) or "").strip())
                        if surface:
                            items.append(surface)
                return list(dict.fromkeys(items))
            handle.seek(0)
            row_reader = csv.reader(handle)
            for csv_row in row_reader:
                if not csv_row or csv_row[0].startswith("#"):
                    continue
                surface = normalize_khmer(csv_row[0].strip())
                if surface and surface.lower() not in {"surface", "word", "headword"}:
                    items.append(surface)
        return list(dict.fromkeys(items))

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            first = line.split("\t", 1)[0].strip()
            surface = normalize_khmer(first)
            if surface and surface.lower() not in {"surface", "word", "headword"}:
                items.append(surface)
    return list(dict.fromkeys(items))


def load_audit_items(path: str | Path, kind: str = "auto") -> list[str]:
    """
    Load normalized comparison keys from a corpus or lexicon file.

    ``kind`` is ``auto``, ``sentences``, or ``lexicon``. Auto-detection uses
    the extension: ``.nova``/``.conllu``/``.pos``/``.txt`` are sentences, while
    ``.tsv``/``.csv`` are lexicons. For sentence files, whitespace, ZWSP, pipe
    separators, POS suffixes, and compound markers are normalized away so the
    same underlying text compares equal.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audit input not found: {p}")

    resolved_kind = kind.strip().lower()
    if resolved_kind == "auto":
        if p.suffix.lower() == ".nova":
            resolved_kind = "sentences"
        elif p.suffix.lower() == ".conllu":
            resolved_kind = "sentences"
        elif p.suffix.lower() in {".tsv", ".csv"}:
            resolved_kind = "lexicon"
        else:
            resolved_kind = "sentences"

    if resolved_kind == "lexicon":
        return _read_lexicon(p)
    if resolved_kind != "sentences":
        raise ValueError(f"Unknown audit input kind: {kind!r} (use auto, sentences, or lexicon)")

    if p.suffix.lower() == ".nova":
        return _read_nova_tokens(p)
    if p.suffix.lower() == ".conllu":
        return _read_conllu_tokens(p)
    return [
        key
        for tokens in load_gold_sentences(p)
        if (key := sentence_surface_key(tokens))
    ]


# ---------------------------------------------------------------------------
# MinHash / LSH
# ---------------------------------------------------------------------------


def _shingles(text: str, n: int = 4) -> set[str]:
    if len(text) <= n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _permutations(count: int) -> list[tuple[int, int]]:
    """Return deterministic affine permutation coefficients for MinHash."""
    result: list[tuple[int, int]] = []
    for seed in range(count):
        digest = hashlib.blake2b(f"kroch-minhash-{seed}".encode(), digest_size=16).digest()
        a = int.from_bytes(digest[:8], "big") % _MERSENNE or 1
        b = int.from_bytes(digest[8:], "big") % _MERSENNE
        result.append((a, b))
    return result


_PERMUTATIONS = _permutations(64)
_BANDS = 16
_ROWS = 4


def _signature(text: str) -> tuple[int, ...]:
    """Compute a 64-value MinHash signature over character shingles."""
    shingles = _shingles(text)
    if not shingles:
        return tuple([_MERSENNE] * len(_PERMUTATIONS))
    hashes = [
        int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")
        for s in shingles
    ]
    return tuple(
        min((a * value + b) % _MERSENNE for value in hashes)
        for a, b in _PERMUTATIONS
    )


def _band_keys(signature: Sequence[int]) -> Iterable[tuple[int, tuple[int, ...]]]:
    for band in range(_BANDS):
        start = band * _ROWS
        yield band, tuple(signature[start : start + _ROWS])


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    if not left:
        return 0.0
    return sum(a == b for a, b in zip(left, right, strict=True)) / len(left)


def _counter_stats(counter: Counter[str]) -> tuple[int, int]:
    groups = sum(1 for count in counter.values() if count > 1)
    items = sum(count - 1 for count in counter.values() if count > 1)
    return groups, items


def _internal_duplicate_examples(
    counter: Counter[str], limit: int
) -> list[OverlapExample]:
    examples: list[OverlapExample] = []
    for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        if count > 1:
            examples.append(
                OverlapExample(
                    left=key,
                    right=f"{count} occurrences",
                    jaccard=1.0,
                    kind="exact",
                )
            )
        if len(examples) >= limit:
            break
    return examples


# ---------------------------------------------------------------------------
# Auditing
# ---------------------------------------------------------------------------


def _audit_internal(
    items: Sequence[str],
    *,
    threshold: float,
    near: bool,
    max_examples: int,
) -> OverlapReport:
    counter = Counter(items)
    groups, duplicate_items = _counter_stats(counter)
    examples = _internal_duplicate_examples(counter, max_examples)

    near_pairs = 0
    if near and len(counter) > 1:
        signatures = {key: _signature(key) for key in counter}
        buckets: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
        for key, signature in signatures.items():
            for band, band_key in _band_keys(signature):
                buckets[(band, band_key)].append(key)

        seen: set[tuple[str, str]] = set()
        near_examples: list[OverlapExample] = []
        for bucket in buckets.values():
            for i, left_key in enumerate(bucket):
                for right_key in bucket[i + 1 :]:
                    pair: tuple[str, str] = (
                        (left_key, right_key)
                        if left_key <= right_key
                        else (right_key, left_key)
                    )
                    if pair in seen:
                        continue
                    seen.add(pair)
                    score = _jaccard(signatures[left_key], signatures[right_key])
                    if score >= threshold:
                        near_pairs += 1
                        if len(near_examples) < max_examples:
                            near_examples.append(
                                OverlapExample(
                                    left=left_key,
                                    right=right_key,
                                    jaccard=score,
                                    kind="near",
                                )
                            )
        examples.extend(near_examples[: max(0, max_examples - len(examples))])

    return OverlapReport(
        mode="internal",
        left_items=len(items),
        right_items=len(items),
        left_unique=len(counter),
        right_unique=len(counter),
        exact_pairs=sum(count * (count - 1) // 2 for count in counter.values()),
        near_pairs=near_pairs,
        left_duplicate_groups=groups,
        right_duplicate_groups=groups,
        left_duplicate_items=duplicate_items,
        right_duplicate_items=duplicate_items,
        threshold=threshold,
        near_enabled=near,
        examples=examples,
    )


def audit_overlap(
    left: Sequence[str],
    right: Sequence[str] | None = None,
    *,
    threshold: float = 0.8,
    near: bool = True,
    max_examples: int = 10,
) -> OverlapReport:
    """
    Compare two collections of normalized keys for exact and near-duplicate overlap.

    If ``right`` is omitted, audit ``left`` against itself and report internal
    duplicate groups and near-duplicate pairs. Exact pairs are counted with
    multiplicity; near-duplicate candidates are generated with a 64-value
    MinHash signature and 16x4 banded LSH, then verified by signature Jaccard.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if right is None:
        return _audit_internal(
            left, threshold=threshold, near=near, max_examples=max_examples
        )

    left_counter = Counter(left)
    right_counter = Counter(right)
    left_groups, left_duplicates = _counter_stats(left_counter)
    right_groups, right_duplicates = _counter_stats(right_counter)

    common = set(left_counter) & set(right_counter)
    exact_pairs = sum(left_counter[key] * right_counter[key] for key in common)
    exact_examples = [
        OverlapExample(
            left=key,
            right=key,
            jaccard=1.0,
            kind="exact",
        )
        for key in sorted(common, key=lambda key: (-left_counter[key], key))[:max_examples]
    ]

    near_pairs = 0
    near_examples: list[OverlapExample] = []
    if near:
        left_keys = [key for key in left_counter if key not in common]
        right_keys = [key for key in right_counter if key not in common]
        if left_keys and right_keys:
            left_signatures = {key: _signature(key) for key in left_keys}
            right_signatures = {key: _signature(key) for key in right_keys}

            buckets: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
            for key, signature in right_signatures.items():
                for band, band_key in _band_keys(signature):
                    buckets[(band, band_key)].append(key)

            for left_key, left_sig in left_signatures.items():
                candidates: set[str] = set()
                for band, band_key in _band_keys(left_sig):
                    candidates.update(buckets.get((band, band_key), ()))
                for right_key in candidates:
                    score = _jaccard(left_sig, right_signatures[right_key])
                    if score < threshold:
                        continue
                    near_pairs += left_counter[left_key] * right_counter[right_key]
                    if len(near_examples) < max_examples:
                        near_examples.append(
                            OverlapExample(
                                left=left_key,
                                right=right_key,
                                jaccard=score,
                                kind="near",
                            )
                        )

    examples = exact_examples + near_examples
    return OverlapReport(
        mode="cross",
        left_items=len(left),
        right_items=len(right),
        left_unique=len(left_counter),
        right_unique=len(right_counter),
        exact_pairs=exact_pairs,
        near_pairs=near_pairs,
        left_duplicate_groups=left_groups,
        right_duplicate_groups=right_groups,
        left_duplicate_items=left_duplicates,
        right_duplicate_items=right_duplicates,
        threshold=threshold,
        near_enabled=near,
        examples=examples[:max_examples],
    )
