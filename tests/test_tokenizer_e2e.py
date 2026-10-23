"""End-to-end tests for Kroch-Chhmar dictionary building, multi-granular tokenization, user dicts, and KCC repair."""

from pathlib import Path

import pytest

from kroch_chhmar.dictionary.compound_splitter import CompoundSplitter
from kroch_chhmar.dictionary.cost_estimator import CostEstimator
from kroch_chhmar.dictionary.csv_writer import (
    build_system_dic,
    build_user_dic,
    generate_sudachi_entries,
    write_sudachi_csv,
)
from kroch_chhmar.evaluate import evaluate_segmentation, load_gold_sentences
from kroch_chhmar.ingest.local_importers import load_seed_dataset
from kroch_chhmar.tokenizer import KhmerMorpheme, KhmerTokenizer, SplitMode, enforce_kcc_boundaries

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def ensure_system_dic() -> Path:
    """Ensure `data/build/system_khmer.dic` is compiled before running e2e tokenizer tests."""
    build_dir = REPO_ROOT / "data" / "build"
    sys_dic = build_dir / "system_khmer.dic"
    if not sys_dic.exists():
        estimator = CostEstimator()
        load_seed_dataset(estimator)
        splitter = CompoundSplitter(
            base_vocab=set(estimator.stats.lexicon_words.keys()),
            compounds_yaml_path=REPO_ROOT / "configs" / "compounds.yaml",
        )
        entries = generate_sudachi_entries(estimator, splitter, is_user_dict=False)
        csv_path = write_sudachi_csv(entries, build_dir / "lex.csv")
        matrix_path = estimator.write_matrix_def(build_dir / "matrix.def")
        build_system_dic(csv_path, matrix_path, sys_dic)
    return sys_dic


def test_multigranular_tokenization_modes() -> None:
    tokenizer = KhmerTokenizer()
    text = "ព្រះរាជាណាចក្រកម្ពុជាមានការសិក្សានិងការអភិវឌ្ឍល្អ។"

    seg_c = tokenizer.segment(text, mode=SplitMode.C)
    seg_b = tokenizer.segment(text, mode=SplitMode.B)
    seg_a = tokenizer.segment(text, mode=SplitMode.A)

    assert seg_c == [
        "ព្រះរាជាណាចក្រកម្ពុជា",
        "មាន",
        "ការសិក្សា",
        "និង",
        "ការអភិវឌ្ឍ",
        "ល្អ",
        "។",
    ]
    assert seg_b == [
        "ព្រះរាជាណាចក្រ",
        "កម្ពុជា",
        "មាន",
        "ការសិក្សា",
        "និង",
        "ការអភិវឌ្ឍ",
        "ល្អ",
        "។",
    ]
    assert seg_a == [
        "ព្រះ",
        "រាជាណាចក្រ",
        "កម្ពុជា",
        "មាន",
        "ការ",
        "សិក្សា",
        "និង",
        "ការ",
        "អភិវឌ្ឍ",
        "ល្អ",
        "។",
    ]


def test_oov_and_kcc_boundary_preservation() -> None:
    tokenizer = KhmerTokenizer()
    # Include an unknown multi-cluster Khmer word "កុលាបក្រហម" between known words
    text = "ខ្ញុំទិញកុលាបក្រហមនៅផ្ទះ។"
    morphemes = tokenizer.tokenize(text, mode=SplitMode.C)

    # Verify no token starts with a dependent vowel or coeng
    for m in morphemes:
        assert not m.surface.startswith("\u17d2")

    # Also directly test enforce_kcc_boundaries when a hypothetical split occurs inside a KCC
    broken_tokens = [
        KhmerMorpheme("ស", "ស", "ស", ("X", "*", "*", "*", "*", "*"), 0, 1, True),
        KhmerMorpheme("\u17d2រី", "\u17d2រី", "\u17d2រី", ("X", "DEP", "*", "*", "*", "*"), 1, 4, True),
    ]
    repaired = enforce_kcc_boundaries(broken_tokens)
    assert len(repaired) == 1
    assert repaired[0].surface == "ស្រី"
    assert repaired[0].begin == 0
    assert repaired[0].end == 4


def test_user_dictionary_ubuild_and_tokenization(tmp_path: Path) -> None:
    estimator = CostEstimator()
    estimator.add_lexicon_entry("ក្រូចឆ្មារ", "PROPN", freq=500)
    splitter = CompoundSplitter()
    entries = generate_sudachi_entries(estimator, splitter, is_user_dict=True)

    user_csv = write_sudachi_csv(entries, tmp_path / "user.csv")
    sys_dic = REPO_ROOT / "data" / "build" / "system_khmer.dic"
    user_dic = build_user_dic(user_csv, sys_dic, tmp_path / "user_khmer.dic")

    tokenizer = KhmerTokenizer(user_dic_paths=[user_dic])
    tokens = tokenizer.tokenize("ខ្ញុំប្រើប្រាស់ក្រូចឆ្មារ។", mode=SplitMode.C)
    surfaces = [t.surface for t in tokens]
    assert "ក្រូចឆ្មារ" in surfaces


def test_evaluation_benchmark_zero_kcc_violations() -> None:
    tokenizer = KhmerTokenizer()
    gold = load_gold_sentences(REPO_ROOT / "data" / "seed" / "seed_corpus.txt")
    report = evaluate_segmentation(tokenizer, gold, mode=SplitMode.C)
    assert report.f1 >= 0.90
    assert report.oov_rate == 0.0
    assert report.kcc_violation_rate == 0.0


def test_structured_token_merging() -> None:
    tokenizer = KhmerTokenizer()

    # URLs, bare domains, emails, and ellipsis
    web_text = "សូមចូលទៅកាន់ http://abc.com ឬ https://mptc.gov.kh/news?id=12 និង info@rac.edu.kh ឬ www.moe.gov.kh..."
    web_tokens = tokenizer.segment(web_text, mode=SplitMode.C)
    assert "http://abc.com" in web_tokens
    assert "https://mptc.gov.kh/news?id=12" in web_tokens
    assert "info@rac.edu.kh" in web_tokens
    assert "www.moe.gov.kh" in web_tokens
    assert "..." in web_tokens

    # Formatted numbers, decimals, thousands, clock times, and peyyal
    num_text = "ចំនួន ២.៦២៥ នាក់ និង 1,250.50 ដុល្លារ នៅម៉ោង ០៨:៣០ ព្រឹក ក្នុង គ.ស. ១៨៦៥ ។ល។"
    num_tokens = tokenizer.segment(num_text, mode=SplitMode.C)
    assert "២.៦២៥" in num_tokens
    assert "1,250.50" in num_tokens
    assert "០៨:៣០" in num_tokens
    assert "គ.ស." in num_tokens
    assert "។ល។" in num_tokens

    # Alphanumeric hyphenated terms and social tags
    tag_text = "ជំងឺ COVID-19 និង កូវីដ-១៩ តាម #Khmer2026 @moh_kh"
    tag_tokens = tokenizer.segment(tag_text, mode=SplitMode.C)
    assert "COVID-19" in tag_tokens
    assert "កូវីដ-១៩" in tag_tokens
    assert "#Khmer2026" in tag_tokens
    assert "@moh_kh" in tag_tokens
