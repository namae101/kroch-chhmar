"""End-to-end tests for Kroch-Chhmar dictionary building, multi-granular tokenization, user dicts, and KCC repair."""

from pathlib import Path

import pytest
from sudachipy.errors import SudachiError

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


@pytest.fixture(scope="session")
def system_dic(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Compile a small, isolated seed dictionary for user-dictionary tests.

    The previous fixture wrote into `data/build/system_khmer.dic`, which could
    clobber a developer's full build and used a different Sudachi signature from
    the bundled resource dictionary (making every user dictionary incompatible).
    """
    build_dir = tmp_path_factory.mktemp("system_dic")
    sys_dic = build_dir / "system_khmer.dic"
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


def test_user_dictionary_ubuild_and_tokenization(system_dic: Path, tmp_path: Path) -> None:
    estimator = CostEstimator()
    estimator.add_lexicon_entry("ក្រូចឆ្មារ", "PROPN", freq=500)
    splitter = CompoundSplitter()
    entries = generate_sudachi_entries(estimator, splitter, is_user_dict=True)

    user_csv = write_sudachi_csv(entries, tmp_path / "user.csv")
    user_dic = build_user_dic(user_csv, system_dic, tmp_path / "user_khmer.dic")

    tokenizer = KhmerTokenizer(system_dic_path=system_dic, user_dic_paths=[user_dic])
    tokens = tokenizer.tokenize("ខ្ញុំប្រើប្រាស់ក្រូចឆ្មារ។", mode=SplitMode.C)
    surfaces = [t.surface for t in tokens]
    assert "ក្រូចឆ្មារ" in surfaces


def test_user_dictionary_signature_mismatch_is_explained(
    system_dic: Path, tmp_path: Path
) -> None:
    """A user dictionary built against one system dict must fail with actionable guidance."""
    estimator = CostEstimator()
    estimator.add_lexicon_entry("ក្រូចឆ្មារ", "PROPN", freq=500)
    entries = generate_sudachi_entries(estimator, CompoundSplitter(), is_user_dict=True)
    user_dic = build_user_dic(
        write_sudachi_csv(entries, tmp_path / "mismatch.csv"),
        system_dic,
        tmp_path / "mismatch.dic",
    )

    bundled = REPO_ROOT / "src" / "kroch_chhmar" / "resources" / "system_khmer.dic"
    with pytest.raises(SudachiError, match="not compatible"):
        KhmerTokenizer(system_dic_path=bundled, user_dic_paths=[user_dic])


def test_evaluation_benchmark_zero_kcc_violations() -> None:
    tokenizer = KhmerTokenizer()
    gold = load_gold_sentences(REPO_ROOT / "data" / "seed" / "seed_corpus.txt")
    report = evaluate_segmentation(tokenizer, gold, mode=SplitMode.C)
    assert report.f1 >= 0.90
    # The bundled permissive dictionary may leave punctuation/digits to Sudachi's
    # OOV plugins, but it must not leave Khmer words uncovered on its own seed corpus.
    assert report.khmer_oov_rate == 0.0
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


def test_coeng_da_ta_visual_twins_and_rac_normalization(
    system_dic: Path, tmp_path: Path
) -> None:
    estimator = CostEstimator()
    # Register RAC canonical Coeng DA (្ដ) and Coeng TA (្ត) words
    estimator.add_lexicon_entry("កណ្ដាល", "NOUN", freq=100, is_canonical=True)
    estimator.add_lexicon_entry("ក្ដៅ", "ADJ", freq=100, is_canonical=True)
    estimator.add_lexicon_entry("សេចក្ដីសុខ", "NOUN", freq=80, is_canonical=True)
    estimator.add_lexicon_entry("សេចក្ដី", "NOUN", freq=90, is_canonical=True)
    estimator.add_lexicon_entry("សុខ", "NOUN", freq=90, is_canonical=True)
    estimator.add_lexicon_entry("សន្តិភាព", "NOUN", freq=100, is_canonical=True)

    splitter = CompoundSplitter(base_vocab={"សេចក្ដី", "សុខ"})
    entries = generate_sudachi_entries(estimator, splitter, is_user_dict=True)

    user_csv = write_sudachi_csv(entries, tmp_path / "coeng_user.csv")
    user_dic = build_user_dic(user_csv, system_dic, tmp_path / "coeng_user.dic")

    tokenizer = KhmerTokenizer(system_dic_path=system_dic, user_dic_paths=[user_dic])

    # 1. Both ក្ដៅ (RAC Coeng DA) and ក្តៅ (common typo Coeng TA) match as single tokens
    #    and both resolve to canonical RAC normalized_form "ក្ដៅ"
    m_da = tokenizer.tokenize("ក្ដៅ", mode=SplitMode.C)
    m_ta = tokenizer.tokenize("ក្តៅ", mode=SplitMode.C)
    assert len(m_da) == 1 and not m_da[0].is_oov
    assert len(m_ta) == 1 and not m_ta[0].is_oov
    assert m_da[0].surface == "ក្ដៅ"
    assert m_ta[0].surface == "ក្តៅ"
    assert m_da[0].normalized_form == "ក្ដៅ"
    assert m_ta[0].normalized_form == "ក្ដៅ"

    # 2. Phonotactic ណ្ត -> ណ្ដ and ន្ដ -> ន្ត
    m_kandal = tokenizer.tokenize("កណ្តាល", mode=SplitMode.C)
    assert len(m_kandal) == 1 and m_kandal[0].normalized_form == "កណ្ដាល"

    m_santepheap = tokenizer.tokenize("សន្ដិភាព", mode=SplitMode.C)
    assert len(m_santepheap) == 1 and m_santepheap[0].normalized_form == "សន្តិភាព"

    # 3. Compound visual twin in Mode C and Mode A
    m_comp_c = tokenizer.tokenize("សេចក្តីសុខ", mode=SplitMode.C)
    assert len(m_comp_c) == 1
    assert m_comp_c[0].surface == "សេចក្តីសុខ"
    assert m_comp_c[0].normalized_form == "សេចក្ដីសុខ"

    m_comp_a = tokenizer.tokenize("សេចក្តីសុខ", mode=SplitMode.A)
    assert [m.surface for m in m_comp_a] == ["សេចក្តី", "សុខ"]
    assert [m.normalized_form for m in m_comp_a] == ["សេចក្ដី", "សុខ"]
