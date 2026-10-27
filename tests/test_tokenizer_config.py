"""Tests for tokenizer path resolution, temp-config cleanup, and normalized offsets."""

from kroch_chhmar.normalizer import normalize_khmer
from kroch_chhmar.tokenizer import KhmerTokenizer


def test_temp_config_is_cleaned_up_and_paths_exposed() -> None:
    tokenizer = KhmerTokenizer()
    assert tokenizer.system_dic_path.exists()
    assert tokenizer.config_path.exists()
    assert tokenizer._temp_config_path is None


def test_offsets_refer_to_normalized_text() -> None:
    tokenizer = KhmerTokenizer()
    text = "សូម\u200bចូល"
    normalized = normalize_khmer(text)
    morphemes = tokenizer.tokenize(text)
    assert morphemes[-1].end == len(normalized)
    assert "".join(m.surface for m in morphemes) == normalized


def test_explicit_custom_config_still_loads() -> None:
    tokenizer = KhmerTokenizer()
    # Reuse the resolved dictionary but pass the bundled config explicitly.
    explicit = KhmerTokenizer(
        config_path=tokenizer.config_path,
        system_dic_path=tokenizer.system_dic_path,
    )
    assert explicit.segment("ភាសាខ្មែរ")
