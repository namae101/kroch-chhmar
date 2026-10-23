"""Corpus statistical estimation for word emission costs and POS transition matrix (`matrix.def`)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from kroch_chhmar.dictionary.pos_mapper import (
    KHMER_DEP_POS_ID,
    TOTAL_MATRIX_STATES,
    UD_POS_TAGS,
    map_to_ud_pos,
    pos_to_id,
)
from kroch_chhmar.normalizer import count_kccs, normalize_khmer


@dataclass
class LexiconStats:
    """Aggregated unigram and POS bigram statistics from corpora and wordlists."""

    word_pos_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    pos_counts: Counter[str] = field(default_factory=Counter)
    pos_bigram_counts: Counter[tuple[int, int]] = field(default_factory=Counter)
    lexicon_words: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    untyped_words: Counter[str] = field(default_factory=Counter)
    reading_forms: dict[str, str] = field(default_factory=dict)
    raw_lexicon_rows: int = 0
    raw_sentences: int = 0
    seen_sentence_hashes: set[str] = field(default_factory=set)
    dedup_sentences: int = 0
    total_corpus_tokens: int = 0


class CostEstimator:
    """
    Estimates Sudachi word emission costs and POS connection costs (`matrix.def`)
    using smoothed negative log-likelihood (NLL).
    """

    def __init__(
        self,
        scale_factor: float = 500.0,
        smoothing_alpha: float = 0.5,
        unseen_base_penalty: int = 7200,
        unseen_kcc_penalty: int = 500,
    ) -> None:
        self.scale_factor = scale_factor
        self.smoothing_alpha = smoothing_alpha
        self.unseen_base_penalty = unseen_base_penalty
        self.unseen_kcc_penalty = unseen_kcc_penalty
        self.stats = LexiconStats()

    def add_lexicon_entry(
        self,
        surface: str,
        pos: str | None = None,
        freq: int = 0,
        reading: str | None = None,
    ) -> None:
        """
        Normalize `surface` and register/merge it in the lexicon.
        If `pos` is None (untyped wordlist), it is merged into existing POS tags for `surface`
        during `finalize_deduplication()` instead of creating a duplicate `NOUN` entry.
        """
        norm = normalize_khmer(surface.strip())
        if not norm:
            return
        self.stats.raw_lexicon_rows += 1
        if reading:
            norm_reading = normalize_khmer(reading.strip())
            if norm_reading and norm not in self.stats.reading_forms:
                self.stats.reading_forms[norm] = norm_reading

        if pos is None:
            self.stats.untyped_words[norm] += max(freq, 0)
            return

        ud_pos = map_to_ud_pos(pos, norm)
        self.stats.lexicon_words[norm].add(ud_pos)
        if freq > 0:
            self.stats.word_pos_counts[(norm, ud_pos)] += freq
            self.stats.pos_counts[ud_pos] += freq
            self.stats.total_corpus_tokens += freq

    def add_segmented_sentence(
        self,
        tokens: Iterable[str | tuple[str, str | None]],
        deduplicate_sentence: bool = True,
    ) -> bool:
        """
        Normalize and ingest a segmented sentence (either raw surface strings or `(surface, pos)` pairs)
        to update unigram and POS bigram counts.
        Returns True if the sentence was added, or False if skipped as an exact normalized duplicate.
        """
        normalized_items: list[tuple[str, str | None]] = []
        for item in tokens:
            if isinstance(item, tuple):
                raw_surf, raw_pos = item
            else:
                raw_surf, raw_pos = item, None

            norm = normalize_khmer(raw_surf.strip())
            if norm:
                normalized_items.append((norm, raw_pos))

        if not normalized_items:
            return False

        self.stats.raw_sentences += 1
        if deduplicate_sentence:
            sig = " ".join(f"{s}/{p or ''}" for s, p in normalized_items)
            if sig in self.stats.seen_sentence_hashes:
                return False
            self.stats.seen_sentence_hashes.add(sig)

        self.stats.dedup_sentences += 1
        prev_pos_id: int | None = None
        for norm, raw_pos in normalized_items:
            if raw_pos is None and norm in self.stats.lexicon_words and self.stats.lexicon_words[norm]:
                # Pick the highest-frequency known POS for this normalized word
                known_tags = sorted(
                    self.stats.lexicon_words[norm],
                    key=lambda t: (-self.stats.word_pos_counts.get((norm, t), 0), t),
                )
                ud_pos = known_tags[0]
            else:
                ud_pos = map_to_ud_pos(raw_pos, norm)

            self.stats.lexicon_words[norm].add(ud_pos)
            self.stats.word_pos_counts[(norm, ud_pos)] += 1
            self.stats.pos_counts[ud_pos] += 1
            self.stats.total_corpus_tokens += 1

            curr_pos_id = pos_to_id(ud_pos)
            if prev_pos_id is not None:
                self.stats.pos_bigram_counts[(prev_pos_id, curr_pos_id)] += 1
            prev_pos_id = curr_pos_id

        return True

    def finalize_deduplication(self) -> tuple[int, int]:
        """
        Merge untyped wordlist frequencies into existing normalized `(surface, pos)` entries,
        and prune noisy low-frequency POS tags when a word has a clear dominant POS.
        Returns `(unique_normalized_surfaces, unique_surface_pos_pairs)`.
        """
        # 1. Merge untyped words (from frequency lists like khmerlbdict) into existing POS tags
        # without adding raw unsegmented wordlist counts into total_corpus_tokens (which would
        # dilute sentence-corpus NLL probabilities).
        for norm, _freq in list(self.stats.untyped_words.items()):
            existing_tags = self.stats.lexicon_words.get(norm)
            if not existing_tags:
                ud_pos = map_to_ud_pos(None, norm)
                self.stats.lexicon_words[norm].add(ud_pos)
        self.stats.untyped_words.clear()

        # 2. Prune noisy minority POS tags from corpus tagging errors (keep tags with >= 10% share or explicit dictionary entries)
        for norm, tags in list(self.stats.lexicon_words.items()):
            if len(tags) <= 1:
                continue
            max_cnt = max(self.stats.word_pos_counts.get((norm, t), 0) for t in tags)
            if max_cnt >= 10:
                kept = {
                    t
                    for t in tags
                    if self.stats.word_pos_counts.get((norm, t), 0) == 0  # explicit dictionary POS
                    or self.stats.word_pos_counts.get((norm, t), 0) >= max(2, int(max_cnt * 0.10))
                }
                if kept:
                    self.stats.lexicon_words[norm] = kept

        unique_surfaces = len(self.stats.lexicon_words)
        unique_pairs = sum(len(tags) for tags in self.stats.lexicon_words.values())
        return unique_surfaces, unique_pairs

    def export_interim_datasets(self, interim_dir: str | Path) -> tuple[Path, Path]:
        """
        Write normalized and deduplicated lexicon (`normalized_lexicon.tsv`) and
        corpus sentences (`normalized_corpus.txt`) to `data/interim/`.
        """
        self.finalize_deduplication()
        out_dir = Path(interim_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        lex_path = out_dir / "normalized_lexicon.tsv"
        lines: list[str] = ["# surface\tpos\tfreq\treading"]
        for surf in sorted(self.stats.lexicon_words.keys()):
            reading = self.stats.reading_forms.get(surf, "*")
            for tag in sorted(self.stats.lexicon_words[surf]):
                freq = self.stats.word_pos_counts.get((surf, tag), 0)
                lines.append(f"{surf}\t{tag}\t{freq}\t{reading}")
        lex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        corp_path = out_dir / "normalized_corpus.txt"
        corp_lines = sorted(self.stats.seen_sentence_hashes)
        corp_path.write_text("\n".join(corp_lines) + "\n", encoding="utf-8")

        return lex_path, corp_path

    def estimate_word_cost(self, surface: str, ud_pos: str) -> int:
        """
        Compute the integer emission cost for `(surface, ud_pos)` in `[-32767, 32767]`.
        Lower cost = higher probability in the Viterbi lattice.
        """
        norm = normalize_khmer(surface)
        count = self.stats.word_pos_counts.get((norm, ud_pos), 0)
        vocab_size = max(len(self.stats.lexicon_words), 1000)
        total = max(self.stats.total_corpus_tokens, 1000)

        if count > 0:
            prob = (count + self.smoothing_alpha) / (
                total + self.smoothing_alpha * vocab_size
            )
            raw_cost = int(round(-self.scale_factor * math.log(prob)))
        else:
            # Unseen lexicon word (count == 0 in sentence corpora): scale penalty gently with KCC length
            # so unattested multi-word dictionary entries do not greedily override attested corpus words,
            # while still beating OOV fallback when none of the constituent words are in the corpus.
            kccs = max(count_kccs(norm), 1)
            raw_cost = min(15000, self.unseen_base_penalty + (kccs - 1) * self.unseen_kcc_penalty)

        return max(-32000, min(32000, raw_cost))

    def compute_transition_cost(self, left_id: int, right_id: int) -> int:
        """
        Compute the transition cost between `left_id` (previous token's rightId)
        and `right_id` (current token's leftId).
        """
        if left_id == KHMER_DEP_POS_ID or right_id == KHMER_DEP_POS_ID:
            # High penalty for transitions involving dangling dependent characters
            return 10000

        left_tag = UD_POS_TAGS[left_id] if 0 <= left_id < len(UD_POS_TAGS) else "X"
        left_total = self.stats.pos_counts.get(left_tag, 0)
        bigram_count = self.stats.pos_bigram_counts.get((left_id, right_id), 0)
        num_tags = len(UD_POS_TAGS)

        if left_total > 0:
            prob = (bigram_count + self.smoothing_alpha) / (
                left_total + self.smoothing_alpha * num_tags
            )
            # Center around 0..1500 so transitions modulate paths without overwhelming emission costs
            baseline_prob = 1.0 / num_tags
            cost = int(round(-220.0 * math.log(prob / baseline_prob)))
            return max(-2000, min(4000, cost))

        return 0

    def write_matrix_def(self, output_path: str | Path) -> Path:
        """
        Write Sudachi-compatible `matrix.def` with `TOTAL_MATRIX_STATES x TOTAL_MATRIX_STATES` entries.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        lines = [f"{TOTAL_MATRIX_STATES} {TOTAL_MATRIX_STATES}"]
        for left_id in range(TOTAL_MATRIX_STATES):
            for right_id in range(TOTAL_MATRIX_STATES):
                cost = self.compute_transition_cost(left_id, right_id)
                lines.append(f"{left_id} {right_id} {cost}")

        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out
