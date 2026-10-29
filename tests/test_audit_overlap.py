"""Tests for exact and near-duplicate corpus/lexicon overlap auditing."""

from pathlib import Path

from kroch_chhmar.audit import audit_overlap, load_audit_items


def test_load_pos_sentences_normalizes_annotation_markers(tmp_path: Path) -> None:
    gold = tmp_path / "sample.pos"
    gold.write_text(
        "ភាសា/NN ខ្មែរ/NN\n"
        "ព្រះ_រាជាណាចក្រ/NN កម្ពុជា/NN\n",
        encoding="utf-8",
    )

    assert load_audit_items(gold) == ["ភាសាខ្មែរ", "ព្រះរាជាណាចក្រកម្ពុជា"]


def test_load_nova_tokens(tmp_path: Path) -> None:
    nova = tmp_path / "sample.km-tok.nova"
    nova.write_text("1\tភាសា ខ្មែរ\n2\tកម្ពុជា\n", encoding="utf-8")

    assert load_audit_items(nova) == ["ភាសាខ្មែរ", "កម្ពុជា"]


def test_load_rac_style_csv_uses_main_and_subword(tmp_path: Path) -> None:
    lexicon = tmp_path / "RAC-Khmer-Dict-2022.csv"
    lexicon.write_text(
        "t_id,t_main,t_subword\n"
        "1,កម្ពុជា,\n"
        "2,កក,កកស្ទះ\n",
        encoding="utf-8",
    )

    assert load_audit_items(lexicon) == ["កម្ពុជា", "កក", "កកស្ទះ"]


def test_exact_overlap_counts_multiplicity() -> None:
    left = ["ភាសាខ្មែរ", "ភាសាខ្មែរ", "ខ្មែរ"]
    right = ["ភាសាខ្មែរ", "កម្ពុជា"]

    report = audit_overlap(left, right, near=False)

    assert report.exact_pairs == 2
    assert report.left_unique == 2
    assert report.left_duplicate_groups == 1
    assert report.left_duplicate_items == 1
    assert report.examples[0].kind == "exact"


def test_internal_duplicate_audit() -> None:
    items = ["ភាសាខ្មែរ", "ភាសាខ្មែរ", "កម្ពុជា"]

    report = audit_overlap(items, near=False)

    assert report.mode == "internal"
    assert report.exact_pairs == 1
    assert report.left_duplicate_groups == 1
    assert report.left_duplicate_items == 1


def test_near_duplicate_detection() -> None:
    left = ["the quick brown fox jumps over the lazy dog"]
    right = ["the quick brown fox jumps over the lazy cat"]

    report = audit_overlap(left, right, threshold=0.8)

    assert report.exact_pairs == 0
    assert report.near_pairs == 1
    assert report.examples[0].kind == "near"
    assert report.examples[0].jaccard >= 0.8


def test_near_detection_can_be_disabled() -> None:
    left = ["the quick brown fox jumps over the lazy dog"]
    right = ["the quick brown fox jumps over the lazy cat"]

    report = audit_overlap(left, right, threshold=0.8, near=False)

    assert report.near_pairs == 0
    assert report.examples == []
