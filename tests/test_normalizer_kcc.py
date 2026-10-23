"""Tests for Khmer Unicode canonicalization (betterkhmer) and Khmer Character Cluster (KCC) boundaries."""

from kroch_chhmar.normalizer import (
    count_kccs,
    get_kcc_boundaries,
    is_kcc_boundary,
    is_kcc_dependent_start,
    iter_kccs,
    normalize_khmer,
)


def test_betterkhmer_canonical_vowel_merging() -> None:
    # Split vowel េ (U+17C1) + ា (U+17B6) after ک (ក) -> canonical ោ (U+17C4)
    unnormalized = "\u1780\u17c1\u17b6"  # ក + េ + ា
    expected = "\u1780\u17c4"            # កោ
    assert normalize_khmer(unnormalized) == expected

    # Split vowel េ (U+17C1) + ី (U+17B8) -> canonical ើ (U+17BE)
    unnormalized_oe = "\u1780\u17c1\u17b8"
    expected_oe = "\u1780\u17be"
    assert normalize_khmer(unnormalized_oe) == expected_oe


def test_betterkhmer_coeng_and_syllable_ordering() -> None:
    # Vowel placed before coeng: ក + ា + ្ + រ -> canonical ក + ្ + រ + ា (ក្រា)
    misordered = "\u1780\u17b6\u17d2\u179a"
    canonical = "\u1780\u17d2\u179a\u17b6"
    assert normalize_khmer(misordered) == canonical

    # Coeng DA (្ដ) vs Coeng TA (្ត) canonicalization
    coeng_da = "សេចក\u17d2\u178aី"
    coeng_ta = "សេចក\u17d2\u178fី"
    assert normalize_khmer(coeng_da) == normalize_khmer(coeng_ta)


def test_zero_width_space_cleanup() -> None:
    text_with_zwsp = "ភាសា\u200bខ្មែរ\ufeff"
    assert normalize_khmer(text_with_zwsp, strip_zwsp=True) == "ភាសាខ្មែរ"
    assert "\u200b" in normalize_khmer(text_with_zwsp, strip_zwsp=False)


def test_kcc_iteration_and_boundaries() -> None:
    # ខ្មែរ = 1 KCC (ខ + ្ + ម + ែ) + 1 KCC (រ) = 2 KCCs
    word = "ខ្មែរ"
    kccs = [span.text for span in iter_kccs(word)]
    assert kccs == ["ខ្មែ", "រ"]
    assert count_kccs(word) == 2

    boundaries = get_kcc_boundaries(word)
    # "ខ្មែ" has length 4, "រ" has length 1 -> valid boundaries are {0, 4, 5}
    assert boundaries == {0, 4, 5}
    assert is_kcc_boundary(word, 0)
    assert not is_kcc_boundary(word, 1)  # before coeng
    assert not is_kcc_boundary(word, 2)  # after coeng
    assert not is_kcc_boundary(word, 3)  # before vowel ែ
    assert is_kcc_boundary(word, 4)      # between ខ្មែ and រ


def test_is_kcc_dependent_start() -> None:
    assert is_kcc_dependent_start("\u17b6")  # dependent vowel ា
    assert is_kcc_dependent_start("\u17d2រ")  # starts with coeng
    assert is_kcc_dependent_start("រ", prev_ends_with_coeng=True)
    assert not is_kcc_dependent_start("រ", prev_ends_with_coeng=False)
