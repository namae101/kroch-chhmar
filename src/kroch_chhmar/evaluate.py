"""Word segmentation evaluation metrics (Precision, Recall, F1, OOV rate, KCC violation rate)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kroch_chhmar.dictionary.pos_mapper import map_to_ud_pos
from kroch_chhmar.normalizer import get_kcc_boundaries, normalize_khmer
from kroch_chhmar.tokenizer import KhmerTokenizer, SplitMode


@dataclass
class EvaluationReport:
    """Summary metrics for Khmer word segmentation evaluation."""

    num_sentences: int
    gold_tokens: int
    pred_tokens: int
    precision: float
    recall: float
    f1: float
    oov_rate: float
    kcc_violation_rate: float
    # Additional reproducibility-oriented metrics
    scored_sentences: int = 0
    exact_matches: int = 0
    exact_match_rate: float = 0.0
    boundary_precision: float = 0.0
    boundary_recall: float = 0.0
    boundary_f1: float = 0.0
    khmer_oov_tokens: int = 0
    khmer_oov_rate: float = 0.0


@dataclass
class POSReport:
    """POS-tagging metrics on tokens whose spans match the gold segmentation."""

    num_sentences: int
    gold_tokens: int
    aligned_tokens: int
    coverage: float
    accuracy: float
    macro_f1: float
    per_tag: dict[str, dict[str, float]]


def _tokens_to_spans(tokens: list[str]) -> set[tuple[int, int]]:
    """Convert a sequence of tokens into character offset spans `(start, end)` on their concatenation."""
    spans: set[tuple[int, int]] = set()
    offset = 0
    for tok in tokens:
        length = len(tok)
        if length > 0:
            spans.add((offset, offset + length))
            offset += length
    return spans


def _boundaries(tokens: list[str]) -> set[int]:
    """Return the set of internal character boundaries (excluding 0 and the final offset)."""
    boundaries: set[int] = set()
    offset = 0
    for tok in tokens:
        offset += len(tok)
        boundaries.add(offset)
    boundaries.discard(0)
    boundaries.discard(offset)
    return boundaries


def _contains_khmer_letters(surface: str) -> bool:
    """True if `surface` contains Khmer consonants/vowels/signs (not just digits or punctuation)."""
    return any(0x1780 <= ord(char) <= 0x17D3 or ord(char) == 0x17DD for char in surface)


def _clean_gold_token(raw_tok: str, split_compounds: bool = False) -> list[str]:
    """
    Strip optional `/POS` suffix and normalize `khPOS` compound/derivation separators (`_`, `~`, and `^`).
    - `^` marks a bound derivational suffix (e.g. `សេដ្ឋ^កិច្ច` -> `សេដ្ឋកិច្ច`, `ពាណិជ្ជ^កម្ម` -> `ពាណិជ្ជកម្ម`),
      which is always joined into a single word in real Khmer text.
    - If `split_compounds=False` (default), `_` and `~` are also joined into a single compound token.
    - If `split_compounds=True`, `_` and `~` are split into sub-tokens (while `^` stays joined on the stem).
    """
    if "/" in raw_tok and len(raw_tok) > 1 and not raw_tok.startswith("/"):
        surf = raw_tok.rsplit("/", 1)[0].strip()
    else:
        surf = raw_tok.strip()

    if not surf:
        return []

    # Strip `^` derivational marker so `ពាណិជ្ជ^កម្ម` -> `ពាណិជ្ជកម្ម` and `ឯក_រាជ្យ^ភាព` -> `ឯក_រាជ្យភាព`
    if "^" in surf and len(surf) > 1:
        surf = surf.replace("^", "")

    if ("_" in surf or "~" in surf) and len(surf) > 1:
        import re

        parts = [normalize_khmer(p) for p in re.split(r"[_~]+", surf) if normalize_khmer(p)]
        if split_compounds:
            return parts
        joined = normalize_khmer("".join(parts))
        return [joined] if joined else []

    norm = normalize_khmer(surf)
    return [norm] if norm else []


def _clean_gold_token_with_pos(
    raw_tok: str, split_compounds: bool = False
) -> list[tuple[str, str]]:
    """Return `(normalized_surface, UD POS)` pairs for a gold token."""
    if "/" in raw_tok and len(raw_tok) > 1 and not raw_tok.startswith("/"):
        surf, pos_text = raw_tok.rsplit("/", 1)
        raw_pos: str | None = pos_text.strip() or None
    else:
        surf, raw_pos = raw_tok, None

    surf = surf.strip()
    if not surf:
        return []
    if "^" in surf and len(surf) > 1:
        surf = surf.replace("^", "")

    if ("_" in surf or "~" in surf) and len(surf) > 1:
        import re

        parts = [normalize_khmer(p) for p in re.split(r"[_~]+", surf) if normalize_khmer(p)]
        if split_compounds:
            return [(part, map_to_ud_pos(raw_pos, part)) for part in parts]
        joined = normalize_khmer("".join(parts))
        return [(joined, map_to_ud_pos(raw_pos, joined))] if joined else []

    norm = normalize_khmer(surf)
    return [(norm, map_to_ud_pos(raw_pos, norm))] if norm else []


def evaluate_segmentation(
    tokenizer: KhmerTokenizer,
    gold_sentences: list[list[str]],
    mode: SplitMode | str = SplitMode.C,
    split_compounds: bool = False,
) -> EvaluationReport:
    """
    Evaluate `tokenizer` on a list of gold-segmented sentences.
    Computes span-level Precision, Recall, F1, OOV token rate, and KCC boundary violation rate.
    """
    total_gold_spans = 0
    total_pred_spans = 0
    total_correct_spans = 0
    total_oov_tokens = 0
    total_khmer_oov_tokens = 0
    total_kcc_violations = 0
    scored_sentences = 0
    exact_matches = 0
    boundary_tp = 0
    boundary_fp = 0
    boundary_fn = 0

    for raw_gold_tokens in gold_sentences:
        norm_gold: list[str] = []
        for tok in raw_gold_tokens:
            norm_gold.extend(_clean_gold_token(tok, split_compounds=split_compounds))
        if not norm_gold:
            continue

        unsegmented_text = "".join(norm_gold)
        valid_kcc_bounds = get_kcc_boundaries(unsegmented_text)

        morphemes = tokenizer.tokenize(unsegmented_text, mode=mode, skip_whitespace=True)
        pred_tokens = [m.surface for m in morphemes]

        gold_spans = _tokens_to_spans(norm_gold)
        pred_spans = _tokens_to_spans(pred_tokens)

        total_gold_spans += len(gold_spans)
        total_pred_spans += len(pred_spans)
        total_correct_spans += len(gold_spans & pred_spans)

        scored_sentences += 1
        if pred_tokens == norm_gold:
            exact_matches += 1

        gold_boundaries = _boundaries(norm_gold)
        pred_boundaries = _boundaries(pred_tokens)
        boundary_tp += len(gold_boundaries & pred_boundaries)
        boundary_fp += len(pred_boundaries - gold_boundaries)
        boundary_fn += len(gold_boundaries - pred_boundaries)

        for m in morphemes:
            if m.is_oov:
                total_oov_tokens += 1
                if _contains_khmer_letters(m.surface):
                    total_khmer_oov_tokens += 1
        for start, end in pred_spans:
            if start not in valid_kcc_bounds or end not in valid_kcc_bounds:
                total_kcc_violations += 1

    precision = total_correct_spans / total_pred_spans if total_pred_spans > 0 else 0.0
    recall = total_correct_spans / total_gold_spans if total_gold_spans > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    oov_rate = total_oov_tokens / total_pred_spans if total_pred_spans > 0 else 0.0
    khmer_oov_rate = (
        total_khmer_oov_tokens / total_pred_spans if total_pred_spans > 0 else 0.0
    )
    kcc_violation_rate = (
        total_kcc_violations / total_pred_spans if total_pred_spans > 0 else 0.0
    )
    boundary_precision = (
        boundary_tp / (boundary_tp + boundary_fp)
        if (boundary_tp + boundary_fp) > 0
        else 1.0
    )
    boundary_recall = (
        boundary_tp / (boundary_tp + boundary_fn)
        if (boundary_tp + boundary_fn) > 0
        else 1.0
    )
    boundary_f1 = (
        2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)
        if (boundary_precision + boundary_recall) > 0
        else 1.0
    )

    return EvaluationReport(
        num_sentences=len(gold_sentences),
        gold_tokens=total_gold_spans,
        pred_tokens=total_pred_spans,
        precision=precision,
        recall=recall,
        f1=f1,
        oov_rate=oov_rate,
        kcc_violation_rate=kcc_violation_rate,
        scored_sentences=scored_sentences,
        exact_matches=exact_matches,
        exact_match_rate=exact_matches / scored_sentences if scored_sentences > 0 else 0.0,
        boundary_precision=boundary_precision,
        boundary_recall=boundary_recall,
        boundary_f1=boundary_f1,
        khmer_oov_tokens=total_khmer_oov_tokens,
        khmer_oov_rate=khmer_oov_rate,
    )


def evaluate_pos_tagging(
    tokenizer: KhmerTokenizer,
    gold_sentences: list[list[str]],
    mode: SplitMode | str = SplitMode.C,
    split_compounds: bool = False,
) -> POSReport:
    """
    Evaluate POS tagging on tokens whose spans match the gold segmentation.

    Boundary errors are reflected in `coverage` (aligned/gold tokens) and are
    excluded from per-tag precision/recall, so `accuracy` is a conditional
    POS accuracy on correctly segmented tokens.
    """
    gold_total = 0
    aligned_total = 0
    correct = 0
    gold_counts: dict[str, int] = {}
    pred_counts: dict[str, int] = {}
    true_positives: dict[str, int] = {}

    for raw_gold_tokens in gold_sentences:
        gold: list[tuple[str, str]] = []
        for tok in raw_gold_tokens:
            gold.extend(_clean_gold_token_with_pos(tok, split_compounds=split_compounds))
        if not gold:
            continue

        text = "".join(surface for surface, _ in gold)
        morphemes = tokenizer.tokenize(text, mode=mode, skip_whitespace=True)

        span_to_tag: dict[tuple[int, int], str] = {}
        offset = 0
        for surface, tag in gold:
            span_to_tag[(offset, offset + len(surface))] = tag
            offset += len(surface)

        gold_total += len(gold)
        for morph in morphemes:
            gold_tag = span_to_tag.get((morph.begin, morph.end))
            if gold_tag is None:
                continue
            pred_tag = morph.ud_pos
            aligned_total += 1
            gold_counts[gold_tag] = gold_counts.get(gold_tag, 0) + 1
            pred_counts[pred_tag] = pred_counts.get(pred_tag, 0) + 1
            if pred_tag == gold_tag:
                correct += 1
                true_positives[gold_tag] = true_positives.get(gold_tag, 0) + 1

    per_tag: dict[str, dict[str, float]] = {}
    f1_scores: list[float] = []
    for tag in sorted(set(gold_counts) | set(pred_counts)):
        tp = true_positives.get(tag, 0)
        support = gold_counts.get(tag, 0)
        predicted = pred_counts.get(tag, 0)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_tag[tag] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
            "predicted": float(predicted),
        }
        if support:
            f1_scores.append(f1)

    return POSReport(
        num_sentences=len(gold_sentences),
        gold_tokens=gold_total,
        aligned_tokens=aligned_total,
        coverage=aligned_total / gold_total if gold_total else 0.0,
        accuracy=correct / aligned_total if aligned_total else 0.0,
        macro_f1=sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        per_tag=per_tag,
    )


def load_gold_sentences(path: str | Path) -> list[list[str]]:
    """Load gold-segmented sentences from a space/ZWSP/pipe-delimited text file."""
    p = Path(path)
    sentences: list[list[str]] = []
    with p.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            unified = line.replace("\u200b", " ").replace("|", " ")
            toks = [t.strip() for t in unified.split() if t.strip()]
            if toks:
                sentences.append(toks)
    return sentences
