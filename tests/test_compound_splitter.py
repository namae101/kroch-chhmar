"""Tests for hybrid Mode A/B/C Khmer compound decomposition."""

from pathlib import Path

from kroch_chhmar.dictionary.compound_splitter import CompoundSplitter

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_curated_overrides_and_prefixes() -> None:
    base_vocab = {
        "ការ",
        "ភាព",
        "អ្នក",
        "សិក្សា",
        "អភិវឌ្ឍ",
        "ជោគជ័យ",
        "គ្រូ",
        "ព្រះ",
        "រាជាណាចក្រ",
        "កម្ពុជា",
        "សាលា",
        "រៀន",
        "បង្កើត",
    }
    splitter = CompoundSplitter(
        base_vocab=base_vocab,
        compounds_yaml_path=REPO_ROOT / "configs" / "compounds.yaml",
    )

    # Mode A override (ការងារ should not be split into ការ + ងារ)
    res_work = splitter.resolve("ការងារ")
    assert res_work.split_type == "A"
    assert res_work.split_a == []

    # Mode B prefix derivation (ការ + បង្កើត)
    res_create = splitter.resolve("ការបង្កើត")
    assert res_create.split_type == "B"
    assert res_create.split_a == ["ការ", "បង្កើត"]

    # Mode C multi-granular compound (ព្រះរាជាណាចក្រកម្ពុជា)
    res_kingdom = splitter.resolve("ព្រះរាជាណាចក្រកម្ពុជា")
    assert res_kingdom.split_type == "C"
    assert res_kingdom.split_a == ["ព្រះ", "រាជាណាចក្រ", "កម្ពុជា"]
    assert res_kingdom.split_b == ["ព្រះរាជាណាចក្រ", "កម្ពុជា"]


def test_dp_compound_decomposition_respects_kcc() -> None:
    base_vocab = {"សាលា", "បឋម", "សិក្សា"}
    splitter = CompoundSplitter(base_vocab=base_vocab)

    res = splitter.resolve("សាលាបឋមសិក្សា")
    assert res.split_type == "C"
    assert res.split_a == ["សាលា", "បឋម", "សិក្សា"]
