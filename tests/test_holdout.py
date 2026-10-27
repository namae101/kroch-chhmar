"""Holdout filtering tests (khPOS train.all2 contains the CLOSE-TEST split)."""

from pathlib import Path

from kroch_chhmar.dictionary.compound_splitter import CompoundSplitter
from kroch_chhmar.dictionary.cost_estimator import CostEstimator
from kroch_chhmar.ingest.local_importers import (
    HoldoutFilter,
    ingest_segmented_text_file,
    load_seed_dataset,
    sentence_surface_key,
)


def test_sentence_key_normalizes_annotation_markers() -> None:
    assert sentence_surface_key(["សេដ្ឋ^កិច្ច/NN", "ការ~សិក្សា/NN"]) == sentence_surface_key(
        ["សេដ្ឋកិច្ច/NN", "ការសិក្សា/NN"]
    )
    assert sentence_surface_key(["ភាសា\u200bខ្មែរ"]) == "ភាសាខ្មែរ"


def test_holdout_sentences_are_not_committed(tmp_path: Path) -> None:
    train = tmp_path / "train.pos"
    holdout_file = tmp_path / "holdout.pos"
    train.write_text(
        "ខ្ញុំ/NN ទៅ/VB ផ្សេង/VB ។/KAN\n"
        "ព្រះ_រាជាណាចក្រ/NN កម្ពុជា/NN\n"
        "ភាសា/NN ខ្មែរ/NN\n",
        encoding="utf-8",
    )
    # Same sentence as the compound line, but written without the `_` marker.
    holdout_file.write_text("ព្រះរាជាណាចក្រ កម្ពុជា\n", encoding="utf-8")

    holdout = HoldoutFilter.from_files([holdout_file])
    assert len(holdout) == 1

    estimator = CostEstimator()
    splitter = CompoundSplitter()
    processed = ingest_segmented_text_file(train, estimator, splitter=splitter, holdout=holdout)

    assert processed == 2
    assert holdout.skipped == 1
    # The compound override and its constituent sub-parts must not leak in.
    assert "ព្រះរាជាណាចក្រកម្ពុជា" not in splitter.overrides
    assert "ព្រះ" not in splitter.base_vocab
    assert "រាជាណាចក្រ" not in splitter.base_vocab
    assert len(estimator.stats.seen_sentence_hashes) == 2


def test_seed_data_is_packaged_and_loadable() -> None:
    estimator = CostEstimator()
    lex_count, sent_count = load_seed_dataset(estimator)
    assert lex_count > 0
    assert sent_count > 0
