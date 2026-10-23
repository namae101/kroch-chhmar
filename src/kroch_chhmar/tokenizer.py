"""Runtime Khmer tokenizer plugin/wrapper around SudachiPy with KCC boundary enforcement."""

from __future__ import annotations

import enum
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sudachipy import Dictionary, SplitMode as SudachiSplitMode

from kroch_chhmar.normalizer import is_kcc_dependent_start, normalize_khmer

# Structured token patterns (URLs, emails, domains, formatted numbers, acronyms, repeated punctuation)
_STRUCTURED_SPAN_RE = re.compile(
    r"(?P<url>https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+|www\.[A-Za-z0-9][-A-Za-z0-9.]*\.[A-Za-z]{2,}(?:/[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]*)?)"
    r"|(?P<email>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
    r"|(?P<domain>(?<![A-Za-z0-9])[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.(?:com|org|net|edu|gov|io|co|kh|info|biz|me|tv|ai|app|dev)(?:\.kh)?(?:/[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]*)?(?![A-Za-z0-9]))"
    r"|(?P<hashtag>(?<![A-Za-z0-9_])[#@][A-Za-z0-9_]+)"
    r"|(?P<peyyal>។ល។)"
    r"|(?P<abbrev>(?:[A-Za-z]\.){2,}|(?:[\u1780-\u17dd]{1,2}\.){2,})"
    r"|(?P<alnum_hyphen>(?<![A-Za-z0-9])[A-Za-z]+-\d+(?![A-Za-z0-9])|(?<![A-Za-z0-9])[A-Za-z]{2,}-[A-Za-z]{2,}(?:-[A-Za-z]{2,})*(?![A-Za-z0-9])|កូវីដ-[\d០-៩]+)"
    r"|(?P<number>[\d០-៩]+(?:[.,:][\d០-៩]+)+)"
    r"|(?P<ellipsis>\.{2,}|…+|[?!]{2,}|-{2,})"
)

_STRUCTURED_POS_MAP: dict[str, tuple[str, str, str, str, str, str]] = {
    "url": ("SYM", "URL", "*", "*", "*", "*"),
    "email": ("SYM", "EMAIL", "*", "*", "*", "*"),
    "domain": ("PROPN", "DOMAIN", "*", "*", "*", "*"),
    "hashtag": ("SYM", "TAG", "*", "*", "*", "*"),
    "peyyal": ("PUNCT", "*", "*", "*", "*", "*"),
    "abbrev": ("PROPN", "ABBR", "*", "*", "*", "*"),
    "alnum_hyphen": ("PROPN", "*", "*", "*", "*", "*"),
    "number": ("NUM", "*", "*", "*", "*", "*"),
    "ellipsis": ("PUNCT", "*", "*", "*", "*", "*"),
}


class SplitMode(str, enum.Enum):
    """Multi-granular segmentation modes matching Sudachi (A = short, B = middle, C = long)."""

    A = "A"
    B = "B"
    C = "C"

    def to_sudachi(self) -> SudachiSplitMode:
        if self == SplitMode.A:
            return SudachiSplitMode.A
        if self == SplitMode.B:
            return SudachiSplitMode.B
        return SudachiSplitMode.C


@dataclass
class KhmerMorpheme:
    """A segmented Khmer morpheme/word produced by `KhmerTokenizer`."""

    surface: str
    normalized_form: str
    reading_form: str
    pos: tuple[str, str, str, str, str, str]
    begin: int
    end: int
    is_oov: bool

    @property
    def ud_pos(self) -> str:
        """Primary Universal Dependencies POS tag."""
        return self.pos[0] if self.pos else "X"


def enforce_kcc_boundaries(morphemes: Iterable[KhmerMorpheme]) -> list[KhmerMorpheme]:
    """
    Post-processing safeguard that enforces Khmer Character Cluster (KCC) boundaries.
    If any token starts with a dependent character (Coeng, dependent vowel, register shifter,
    diacritic sign) or follows a trailing Coeng (`\\u17d2`) at the end of the previous token,
    it is automatically merged into the preceding token.
    """
    merged: list[KhmerMorpheme] = []
    for morph in morphemes:
        if not merged:
            merged.append(morph)
            continue

        prev = merged[-1]
        prev_ends_with_coeng = prev.surface.endswith("\u17d2")

        if is_kcc_dependent_start(morph.surface, prev_ends_with_coeng=prev_ends_with_coeng):
            combined_surface = normalize_khmer(
                prev.surface + morph.surface, strip_zwsp=False
            )
            combined_norm = normalize_khmer(
                prev.normalized_form + morph.normalized_form, strip_zwsp=False
            )
            combined_reading = normalize_khmer(
                prev.reading_form + morph.reading_form, strip_zwsp=False
            )
            # Keep the base token's POS unless it was a dangling DEP tag
            effective_pos = prev.pos if prev.pos[1] != "DEP" else morph.pos
            merged[-1] = KhmerMorpheme(
                surface=combined_surface,
                normalized_form=combined_norm,
                reading_form=combined_reading,
                pos=effective_pos,
                begin=prev.begin,
                end=morph.end,
                is_oov=prev.is_oov or morph.is_oov,
            )
        else:
            merged.append(morph)

    return merged


def merge_structured_tokens(
    morphemes: list[KhmerMorpheme],
    text: str,
) -> list[KhmerMorpheme]:
    """
    Merge contiguous tokens whose character spans `(begin, end)` match structured entities
    in `text`, including:
      - URLs (`http://abc.com`, `https://mptc.gov.kh/news?id=1`, `www.example.org`)
      - Email addresses (`info@rac.edu.kh`)
      - Bare domains (`abc.com`, `moe.gov.kh`)
      - Hashtags & mentions (`#កម្ពុជា`, `@user_name`)
      - Khmer peyyal (`។ល។`) and dotted abbreviations (`គ.ស.`, `ព.ស.`, `អ.ស.ប.`, `U.S.A.`)
      - Hyphenated alphanumeric terms (`COVID-19`, `កូវីដ-១៩`, `Wi-Fi`)
      - Formatted numbers, decimals, thousands, and clock times (`២.៦២៥`, `1,250.50`, `០៨:៣០`)
      - Repeated punctuation / ellipses (`...`, `?!`, `!!`, `---`)
    """
    if len(morphemes) <= 1 or not text:
        return morphemes

    matches = list(_STRUCTURED_SPAN_RE.finditer(text))
    if not matches:
        return morphemes

    # Map character offsets to morpheme indices for exact span alignment
    begin_to_idx = {m.begin: i for i, m in enumerate(morphemes)}
    end_to_idx = {m.end: i for i, m in enumerate(morphemes)}

    merge_ranges: list[tuple[int, int, tuple[str, str, str, str, str, str]]] = []
    last_end_idx = -1

    for match in matches:
        span_start, span_end = match.span()
        # Trim trailing sentence punctuation (e.g. '.', ',', ':', ';') from URLs/domains/emails
        kind = match.lastgroup or "url"
        if kind in ("url", "domain", "email"):
            while span_end > span_start and text[span_end - 1] in ".,:;!?":
                span_end -= 1

        start_idx = begin_to_idx.get(span_start)
        end_idx = end_to_idx.get(span_end)
        if start_idx is None or end_idx is None or start_idx >= end_idx:
            continue
        if start_idx <= last_end_idx:
            continue

        # Ensure contiguous tokens with no whitespace gaps inside the span
        sub_slice = morphemes[start_idx : end_idx + 1]
        is_contiguous = all(
            sub_slice[k].end == sub_slice[k + 1].begin for k in range(len(sub_slice) - 1)
        )
        if not is_contiguous:
            continue

        pos_tuple = _STRUCTURED_POS_MAP.get(kind, ("X", "*", "*", "*", "*", "*"))
        merge_ranges.append((start_idx, end_idx, pos_tuple))
        last_end_idx = end_idx

        # If trimming trailing dots left a contiguous ellipsis (e.g. "info@rac.edu.kh..."), merge those dots too
        if span_end < match.end():
            rem_text = text[span_end : match.end()]
            if re.fullmatch(r"\.{2,}|[?!]{2,}", rem_text):
                r_start = begin_to_idx.get(span_end)
                r_end = end_to_idx.get(match.end())
                if r_start is not None and r_end is not None and r_start < r_end:
                    merge_ranges.append((r_start, r_end, _STRUCTURED_POS_MAP["ellipsis"]))
                    last_end_idx = r_end

    if not merge_ranges:
        return morphemes

    result: list[KhmerMorpheme] = []
    range_ptr = 0
    i = 0
    n = len(morphemes)

    while i < n:
        if range_ptr < len(merge_ranges) and i == merge_ranges[range_ptr][0]:
            start_i, end_i, pos_tuple = merge_ranges[range_ptr]
            group = morphemes[start_i : end_i + 1]
            span_surface = text[group[0].begin : group[-1].end]
            result.append(
                KhmerMorpheme(
                    surface=span_surface,
                    normalized_form=span_surface,
                    reading_form=span_surface,
                    pos=pos_tuple,
                    begin=group[0].begin,
                    end=group[-1].end,
                    is_oov=any(m.is_oov for m in group),
                )
            )
            i = end_i + 1
            range_ptr += 1
        else:
            result.append(morphemes[i])
            i += 1

    return result


class KhmerTokenizer:
    """
    Khmer-aware wrapper around `sudachipy.Dictionary` and `sudachipy.Tokenizer`.
    Combines:
      1. `betterkhmer` Unicode canonicalization & zero-width space handling
      2. Multi-granular Sudachi lattice segmentation (Modes A, B, C)
      3. Post-tokenization KCC boundary enforcement
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        system_dic_path: str | Path | None = None,
        user_dic_paths: list[str | Path] | None = None,
        default_mode: SplitMode | str = SplitMode.C,
        strip_zwsp: bool = True,
        enforce_kcc: bool = True,
        merge_structured: bool = True,
    ) -> None:
        self.default_mode = SplitMode(default_mode)
        self.strip_zwsp = strip_zwsp
        self.enforce_kcc = enforce_kcc
        self.merge_structured = merge_structured

        repo_root = Path(__file__).resolve().parents[2]
        default_cfg = repo_root / "configs" / "sudachi.json"
        cfg_file = Path(config_path) if config_path else default_cfg

        if not cfg_file.exists():
            raise FileNotFoundError(f"Sudachi config file not found: {cfg_file}")

        cfg_data = json.loads(cfg_file.read_text(encoding="utf-8"))
        cfg_dir = cfg_file.parent.resolve()

        # Resolve relative paths in config to absolute paths so SudachiPy works from any CWD
        if system_dic_path:
            sys_dic = Path(system_dic_path).resolve()
        else:
            sys_dic = (cfg_dir / cfg_data.get("systemDict", "../data/build/system_khmer.dic")).resolve()

        if not sys_dic.exists():
            raise FileNotFoundError(
                f"Compiled system dictionary not found at: {sys_dic}\n"
                "Please build the system dictionary first by running:\n"
                "  • Quick seed build : uv run kroch-chhmar build\n"
                "  • Full 100k build  : uv run kroch-chhmar fetch all && uv run kroch-chhmar prepare && uv run kroch-chhmar build\n"
                "Or specify an existing `.dic` file via `--system-dic /path/to/system_khmer.dic`."
            )

        cfg_data["systemDict"] = str(sys_dic)
        if "characterDefinitionFile" in cfg_data:
            cfg_data["characterDefinitionFile"] = str(
                (cfg_dir / cfg_data["characterDefinitionFile"]).resolve()
            )

        for plugin in cfg_data.get("inputTextPlugin", []):
            if "rewriteDef" in plugin:
                plugin["rewriteDef"] = str((cfg_dir / plugin["rewriteDef"]).resolve())

        for plugin in cfg_data.get("oovProviderPlugin", []):
            if "charDef" in plugin:
                plugin["charDef"] = str((cfg_dir / plugin["charDef"]).resolve())
            if "unkDef" in plugin:
                plugin["unkDef"] = str((cfg_dir / plugin["unkDef"]).resolve())

        if user_dic_paths:
            cfg_data["userDict"] = [str(Path(p).resolve()) for p in user_dic_paths]

        self._temp_cfg = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8", delete=False
        )
        json.dump(cfg_data, self._temp_cfg, ensure_ascii=False, indent=2)
        self._temp_cfg.flush()
        self._temp_cfg.close()

        self.dictionary = Dictionary(config_path=self._temp_cfg.name)
        self._tokenizer = self.dictionary.tokenizer()

    def tokenize(
        self,
        text: str,
        mode: SplitMode | str | None = None,
        skip_whitespace: bool = True,
    ) -> list[KhmerMorpheme]:
        """
        Normalize and segment `text` into `KhmerMorpheme` objects.
        """
        if not text:
            return []

        active_mode = SplitMode(mode) if mode is not None else self.default_mode
        sudachi_mode = active_mode.to_sudachi()

        normalized_input = normalize_khmer(text, strip_zwsp=self.strip_zwsp)
        raw_morphemes = self._tokenizer.tokenize(normalized_input, mode=sudachi_mode)

        converted: list[KhmerMorpheme] = []
        for m in raw_morphemes:
            surf = m.surface()
            if skip_whitespace and not surf.strip():
                continue
            pos_list = list(m.part_of_speech())
            while len(pos_list) < 6:
                pos_list.append("*")
            pos_tuple = (
                pos_list[0],
                pos_list[1],
                pos_list[2],
                pos_list[3],
                pos_list[4],
                pos_list[5],
            )
            converted.append(
                KhmerMorpheme(
                    surface=surf,
                    normalized_form=m.normalized_form(),
                    reading_form=m.reading_form(),
                    pos=pos_tuple,
                    begin=m.begin(),
                    end=m.end(),
                    is_oov=m.is_oov(),
                )
            )

        if self.enforce_kcc:
            converted = enforce_kcc_boundaries(converted)
        if self.merge_structured:
            converted = merge_structured_tokens(converted, normalized_input)
        return converted

    def segment(
        self,
        text: str,
        mode: SplitMode | str | None = None,
        skip_whitespace: bool = True,
    ) -> list[str]:
        """Convenience method returning a list of segmented word surface strings."""
        return [
            m.surface
            for m in self.tokenize(text, mode=mode, skip_whitespace=skip_whitespace)
        ]
