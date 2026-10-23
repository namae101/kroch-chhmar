"""Word segmentation evaluation metrics (Precision, Recall, F1, OOV rate, KCC violation rate)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    total_kcc_violations = 0

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

        for m in morphemes:
            if m.is_oov:
                total_oov_tokens += 1
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
    kcc_violation_rate = (
        total_kcc_violations / total_pred_spans if total_pred_spans > 0 else 0.0
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
