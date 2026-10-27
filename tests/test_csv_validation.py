"""Tests for split-reference validation and topology reporting."""

from kroch_chhmar.dictionary.csv_writer import (
    SudachiCSVEntry,
    summarize_split_topology,
    validate_split_references,
)


def _entry(
    surface: str,
    split_type: str = "A",
    split_info_a: str = "*",
    split_info_b: str = "*",
) -> SudachiCSVEntry:
    return SudachiCSVEntry(
        surface=surface,
        left_id=0,
        right_id=0,
        cost=1000,
        display_surface=surface,
        pos1="NOUN",
        pos2="*",
        pos3="*",
        pos4="*",
        pos5="*",
        pos6="*",
        reading_form=surface,
        normalized_form=surface,
        split_type=split_type,
        split_info_a=split_info_a,
        split_info_b=split_info_b,
    )


def test_valid_split_references_and_topology() -> None:
    entries = [
        _entry("ព្រះ"),
        _entry("រាជាណាចក្រ"),
        _entry("កម្ពុជា"),
        _entry("ព្រះរាជាណាចក្រ", "B", "0/1"),
        _entry(
            "ព្រះរាជាណាចក្រកម្ពុជា",
            "C",
            split_info_a="0/1/2",
            split_info_b="3/2",
        ),
    ]
    assert validate_split_references(entries) == []
    topology = summarize_split_topology(entries)
    assert topology == {"A": 3, "B": 1, "C": 1, "c_with_b": 1, "c_without_b": 0}


def test_invalid_forward_reference_is_reported() -> None:
    entries = [
        _entry("ព្រះ"),
        _entry("រាជាណាចក្រ"),
        _entry("ព្រះរាជាណាចក្រ", "B", "0/2"),
    ]
    errors = validate_split_references(entries)
    assert any("invalid row 2" in error for error in errors)


def test_surface_mismatch_is_reported() -> None:
    entries = [
        _entry("ព្រះ"),
        _entry("រាជាណាចក្រ"),
        _entry("ភាសាខ្មែរ", "B", "0/1"),
    ]
    errors = validate_split_references(entries)
    assert any("do not concatenate" in error for error in errors)


def test_user_dictionary_ids_with_u_prefix() -> None:
    entries = [
        _entry("ការ"),
        _entry("សិក្សា"),
        _entry("ការសិក្សា", "B", "U0/U1"),
    ]
    assert validate_split_references(entries, is_user_dict=True) == []
