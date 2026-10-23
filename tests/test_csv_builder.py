"""Tests for cost estimation, matrix.def generation, and 19-column Sudachi CSV formatting."""

from pathlib import Path

from sudachi_khmer.dictionary.compound_splitter import CompoundSplitter
from sudachi_khmer.dictionary.cost_estimator import CostEstimator
from sudachi_khmer.dictionary.csv_writer import generate_sudachi_entries, write_sudachi_csv
from sudachi_khmer.dictionary.pos_mapper import TOTAL_MATRIX_STATES


def test_cost_estimator_and_matrix_def(tmp_path: Path) -> None:
    estimator = CostEstimator()
    estimator.add_lexicon_entry("ខ្ញុំ", "PRON", freq=100)
    estimator.add_lexicon_entry("ស្រឡាញ់", "VERB", freq=50)
    estimator.add_lexicon_entry("កម្ពុជា", "PROPN", freq=80)
    estimator.add_segmented_sentence([("ខ្ញុំ", "PRON"), ("ស្រឡាញ់", "VERB"), ("កម្ពុជា", "PROPN")])

    # Frequent word has lower cost than rare/unseen word
    cost_freq = estimator.estimate_word_cost("ខ្ញុំ", "PRON")
    cost_unseen = estimator.estimate_word_cost("ពាក្យថ្មី", "NOUN")
    assert cost_freq < cost_unseen

    matrix_file = estimator.write_matrix_def(tmp_path / "matrix.def")
    lines = matrix_file.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == f"{TOTAL_MATRIX_STATES} {TOTAL_MATRIX_STATES}"
    assert len(lines) == 1 + TOTAL_MATRIX_STATES * TOTAL_MATRIX_STATES


def test_19_column_csv_generation(tmp_path: Path) -> None:
    estimator = CostEstimator()
    for w, p in [
        ("ការ", "NOUN"),
        ("សិក្សា", "VERB"),
        ("ការសិក្សា", "NOUN"),
    ]:
        estimator.add_lexicon_entry(w, p, freq=20)

    splitter = CompoundSplitter(base_vocab={"ការ", "សិក្សា"})
    entries = generate_sudachi_entries(estimator, splitter, is_user_dict=False)

    csv_file = write_sudachi_csv(entries, tmp_path / "lex.csv")
    rows = [line.split(",") for line in csv_file.read_text(encoding="utf-8").strip().splitlines()]
    assert all(len(r) == 19 for r in rows)

    by_surface = {r[0]: r for r in rows}
    assert "ការសិក្សា" in by_surface
    compound_row = by_surface["ការសិក្សា"]
    assert compound_row[14] == "B"
    # split_info_a (col 15) must reference 2 valid integer line indices
    parts = compound_row[15].split("/")
    assert len(parts) == 2
    assert all(p.isdigit() for p in parts)
