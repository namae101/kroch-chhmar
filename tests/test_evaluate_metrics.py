"""Tests for exact-match, boundary-F1, and Khmer-OOV evaluation metrics."""

from kroch_chhmar.evaluate import evaluate_pos_tagging, evaluate_segmentation
from kroch_chhmar.tokenizer import KhmerMorpheme


class FakeTokenizer:
    """Minimal tokenizer stub mapping a text string to a fixed token list."""

    def __init__(
        self,
        mapping: dict[str, list[str]],
        oov: set[str] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        self.mapping = mapping
        self.oov = oov or set()
        self.tags = tags or []

    def tokenize(self, text: str, mode=None, skip_whitespace: bool = True) -> list[KhmerMorpheme]:
        morphemes: list[KhmerMorpheme] = []
        offset = 0
        for index, surface in enumerate(self.mapping[text]):
            tag = self.tags[index] if index < len(self.tags) else "NOUN"
            morphemes.append(
                KhmerMorpheme(
                    surface=surface,
                    normalized_form=surface,
                    reading_form=surface,
                    pos=(tag, "*", "*", "*", "*", "*"),
                    begin=offset,
                    end=offset + len(surface),
                    is_oov=surface in self.oov,
                )
            )
            offset += len(surface)
        return morphemes


def test_exact_match_and_boundary_metrics() -> None:
    tokenizer = FakeTokenizer({"abcd": ["ab", "cd"], "efgh": ["efgh"]})
    gold = [["ab", "cd"], ["ef", "gh"]]
    report = evaluate_segmentation(tokenizer, gold, mode="C")

    assert report.scored_sentences == 2
    assert report.exact_matches == 1
    assert report.exact_match_rate == 0.5
    assert report.boundary_precision == 1.0
    assert report.boundary_recall == 0.5
    assert report.boundary_f1 == 2 * 1.0 * 0.5 / 1.5
    assert report.precision == 2 / 3
    assert report.recall == 2 / 4


def test_khmer_oov_counting() -> None:
    tokenizer = FakeTokenizer({"ខ្ញុំ": ["ខ្ញុំ"]}, oov={"ខ្ញុំ"})
    report = evaluate_segmentation(tokenizer, [["ខ្ញុំ"]], mode="C")
    assert report.khmer_oov_tokens == 1
    assert report.khmer_oov_rate == 1.0
    assert report.kcc_violation_rate == 0.0


def test_khmer_digits_are_not_counted_as_khmer_word_oov() -> None:
    tokenizer = FakeTokenizer({"១២៣": ["១២៣"]}, oov={"១២៣"})
    report = evaluate_segmentation(tokenizer, [["១២៣"]], mode="C")
    assert report.khmer_oov_tokens == 0


def test_pos_accuracy_and_coverage() -> None:
    tokenizer = FakeTokenizer({"abcd": ["ab", "cd"]}, tags=["NOUN", "NOUN"])
    report = evaluate_pos_tagging(tokenizer, [["ab/NOUN", "cd/VERB"]], mode="C")
    assert report.gold_tokens == 2
    assert report.aligned_tokens == 2
    assert report.coverage == 1.0
    assert report.accuracy == 0.5
    assert report.per_tag["NOUN"]["precision"] == 0.5
    assert report.per_tag["NOUN"]["recall"] == 1.0
    assert report.per_tag["VERB"]["recall"] == 0.0


def test_pos_coverage_excludes_boundary_errors() -> None:
    tokenizer = FakeTokenizer({"efgh": ["efgh"]})
    report = evaluate_pos_tagging(tokenizer, [["ef/NOUN", "gh/VERB"]], mode="C")
    assert report.aligned_tokens == 0
    assert report.coverage == 0.0
