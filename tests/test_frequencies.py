"""Wordlist frequency handling for the full and permissive dictionary profiles."""

from pathlib import Path

from kroch_chhmar.dictionary.cost_estimator import CostEstimator
from kroch_chhmar.ingest.local_importers import ingest_lexicon_tsv


def test_khmerlbdict_frequencies_are_ignored_by_default(tmp_path: Path) -> None:
    lexicon = tmp_path / "khmerlbdict_sample.tsv"
    lexicon.write_text("ពាក្យ\t10\n", encoding="utf-8")

    estimator = CostEstimator()
    assert ingest_lexicon_tsv(lexicon, estimator) == 1
    estimator.finalize_deduplication()

    assert "ពាក្យ" in estimator.stats.lexicon_words
    assert estimator.stats.word_pos_counts[("ពាក្យ", "NOUN")] == 0
    assert estimator.stats.total_corpus_tokens == 0


def test_khmerlbdict_frequencies_are_propagated_when_requested(tmp_path: Path) -> None:
    lexicon = tmp_path / "khmerlbdict_sample.tsv"
    lexicon.write_text("ពាក្យ\t10\n", encoding="utf-8")

    estimator = CostEstimator()
    assert ingest_lexicon_tsv(lexicon, estimator, use_frequencies=True) == 1
    estimator.finalize_deduplication()

    assert estimator.stats.word_pos_counts[("ពាក្យ", "NOUN")] == 10
    assert estimator.stats.pos_counts["NOUN"] == 10
    assert estimator.stats.total_corpus_tokens == 10
    assert estimator.estimate_word_cost("ពាក្យ", "NOUN") < estimator.unseen_base_penalty


def test_untyped_frequency_merges_into_an_existing_pos(tmp_path: Path) -> None:
    estimator = CostEstimator()
    estimator.add_lexicon_entry("ពាក្យ", pos="VERB", freq=3)
    estimator.add_lexicon_entry("ពាក្យ", pos=None, freq=10)
    estimator.finalize_deduplication()

    assert estimator.stats.lexicon_words["ពាក្យ"] == {"VERB"}
    assert estimator.stats.word_pos_counts[("ពាក្យ", "VERB")] == 13
    assert estimator.stats.total_corpus_tokens == 13
