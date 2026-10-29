"""Independent KCC validation against UAX #29 grapheme clusters (`regex` ``\\X``).

`enforce_kcc_boundaries` is a self-check: it guarantees that the tokenizer does
not split clusters according to our own `iter_kccs()` engine. These tests use
the third-party `regex` module's Unicode grapheme-cluster implementation as an
external reference, plus a hand-audited fixture and a real khPOS corpus check.
"""

from pathlib import Path

import pytest

from kroch_chhmar.evaluate import load_gold_sentences
from kroch_chhmar.ingest.local_importers import sentence_surface_key
from kroch_chhmar.normalizer import get_kcc_boundaries, iter_kccs, normalize_khmer
from kroch_chhmar.tokenizer import KhmerTokenizer, SplitMode

regex = pytest.importorskip("regex")

REPO_ROOT = Path(__file__).resolve().parents[1]

# Hand-audited KCC splits. Each cluster is a base character plus its dependent
# coeng/vowel/sign attachments; independent vowels and digits stand alone.
HAND_AUDITED = {
    "ខ្មែរ": ["ខ្មែ", "រ"],
    "ស្រី": ["ស្រី"],
    "ក្សត្រិយ៍": ["ក្ស", "ត្រិ", "យ៍"],
    "ព្រះរាជាណាចក្រ": ["ព្រះ", "រា", "ជា", "ណា", "ច", "ក្រ"],
    "ការសិក្សា": ["កា", "រ", "សិ", "ក្សា"],
    "កណ្តាល": ["ក", "ណ្ដា", "ល"],
    "សន្តិភាព": ["ស", "ន្តិ", "ភា", "ព"],
    "២.៦២៥": ["២", ".", "៦", "២", "៥"],
}


@pytest.mark.parametrize(("text", "expected"), sorted(HAND_AUDITED.items()))
def test_hand_audited_clusters(text: str, expected: list[str]) -> None:
    normalized = normalize_khmer(text)
    ours = [span.text for span in iter_kccs(normalized)]
    reference = regex.findall(r"\X", normalized)
    assert ours == expected
    assert reference == expected


def test_reference_requires_normalization_for_malformed_coeng_order() -> None:
    malformed = "ទេ្ទ"  # vowel before coeng: not a valid KCC order
    assert [span.text for span in iter_kccs(malformed)] != regex.findall(r"\X", malformed)

    normalized = normalize_khmer(malformed)
    assert normalized == "ទ្ទេ"
    assert [span.text for span in iter_kccs(normalized)] == regex.findall(r"\X", normalized)


@pytest.mark.parametrize("mode", [SplitMode.A, SplitMode.B, SplitMode.C])
def test_tokenizer_boundaries_are_reference_grapheme_boundaries(mode: SplitMode) -> None:
    text = "ព្រះរាជាណាចក្រកម្ពុជាមានការសិក្សានិងការអភិវឌ្ឍល្អ។"
    normalized = normalize_khmer(text)
    reference = regex.findall(r"\X", normalized)

    boundaries = {0}
    offset = 0
    for cluster in reference:
        offset += len(cluster)
        boundaries.add(offset)

    tokenizer = KhmerTokenizer()
    tokens = tokenizer.tokenize(normalized, mode=mode)
    assert tokens
    for token in tokens:
        # The token span must lie on reference grapheme-cluster boundaries.
        assert token.begin in boundaries, (token.surface, token.begin, boundaries)
        assert token.end in boundaries, (token.surface, token.end, boundaries)


@pytest.mark.parametrize(
    ("name", "filename"),
    [
        ("OPEN", "khpos_open_test.pos"),
        ("CLOSE", "khpos_close_test.pos"),
        ("TRAIN", "khpos_train.pos"),
    ],
)
def test_reference_matches_khpos_corpus(name: str, filename: str) -> None:
    path = REPO_ROOT / "data" / "raw" / filename
    if not path.exists():
        pytest.skip(f"{name} corpus not downloaded")

    sentences = 0
    clusters = 0
    for tokens in load_gold_sentences(path):
        key = sentence_surface_key(tokens)
        if not key:
            continue
        sentences += 1
        ours = [span.text for span in iter_kccs(key)]
        reference = regex.findall(r"\X", key)
        assert ours == reference, f"{name} sentence {sentences}: {key!r}"
        clusters += len(reference)
    assert sentences > 0
    assert clusters > 0


def test_kcc_boundaries_are_closed_under_normalization() -> None:
    text = "ភាសាខ្មែរ"
    normalized = normalize_khmer(text)
    assert normalize_khmer(normalized) == normalized
    assert get_kcc_boundaries(normalized) == get_kcc_boundaries(normalize_khmer(normalized))
