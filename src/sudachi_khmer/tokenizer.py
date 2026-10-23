"""Runtime Khmer tokenizer plugin/wrapper around SudachiPy with KCC boundary enforcement."""

from __future__ import annotations

import enum
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sudachipy import Dictionary, SplitMode as SudachiSplitMode

from sudachi_khmer.normalizer import is_kcc_dependent_start, normalize_khmer


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
    ) -> None:
        self.default_mode = SplitMode(default_mode)
        self.strip_zwsp = strip_zwsp
        self.enforce_kcc = enforce_kcc

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
                f"Compiled system dictionary not found at {sys_dic}. "
                "Run `sudachi-khmer build` first."
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
            return enforce_kcc_boundaries(converted)
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
