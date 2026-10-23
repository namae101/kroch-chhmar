"""19-column Sudachi CSV formatter, two-pass split ID resolver, and binary `.dic` builder."""

from __future__ import annotations

import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from sudachi_khmer.dictionary.compound_splitter import CompoundSplitter
from sudachi_khmer.dictionary.cost_estimator import CostEstimator
from sudachi_khmer.dictionary.pos_mapper import (
    UD_POS_TAGS,
    map_to_ud_pos,
    pos_to_id,
    pos_to_sudachi_tuple,
)
from sudachi_khmer.normalizer import normalize_khmer


@dataclass
class SudachiCSVEntry:
    """Represents a single 19-column Sudachi dictionary CSV row."""

    surface: str
    left_id: int
    right_id: int
    cost: int
    display_surface: str
    pos1: str
    pos2: str
    pos3: str
    pos4: str
    pos5: str
    pos6: str
    reading_form: str
    normalized_form: str
    dict_form_id: str = "*"
    split_type: str = "A"
    split_info_a: str = "*"
    split_info_b: str = "*"
    unused_col18: str = "*"
    unused_col19: str = "*"

    def to_row(self) -> list[str]:
        """Convert to a 19-element list of strings for CSV serialization."""
        return [
            self.surface,
            str(self.left_id),
            str(self.right_id),
            str(self.cost),
            self.display_surface,
            self.pos1,
            self.pos2,
            self.pos3,
            self.pos4,
            self.pos5,
            self.pos6,
            self.reading_form,
            self.normalized_form,
            self.dict_form_id,
            self.split_type,
            self.split_info_a,
            self.split_info_b,
            self.unused_col18,
            self.unused_col19,
        ]


def generate_sudachi_entries(
    estimator: CostEstimator,
    splitter: CompoundSplitter,
    is_user_dict: bool = False,
) -> list[SudachiCSVEntry]:
    """
    Generate a two-pass ordered list of 19-column Sudachi CSV entries:
      - Pass 1: Collect all Mode A base words (including any sub-units required by Mode B/C splits)
        and assign them 0-based word IDs (`0..N_A-1`).
      - Pass 2: Collect Mode B words (`N_A..N_B-1`) referencing Mode A IDs in `split_info_a`.
      - Pass 3: Collect Mode C words referencing Mode A IDs in `split_info_a` and Mode A/B IDs in `split_info_b`.
    """
    id_prefix = "U" if is_user_dict else ""
    estimator.finalize_deduplication()

    # Gather all candidate (surface, ud_pos) pairs
    raw_candidates: dict[tuple[str, str], None] = {}
    for surf, tags in sorted(estimator.stats.lexicon_words.items()):
        norm = normalize_khmer(surf)
        if not norm:
            continue
        # Only include words containing Khmer characters or standard single punctuation marks;
        # ASCII numbers (e.g. "1,000") and Latin words are handled by Sudachi's NUMERIC/ALPHA OOV plugins
        has_khmer = any(0x1780 <= ord(c) <= 0x17FF or 0x19E0 <= ord(c) <= 0x19FF for c in norm)
        is_basic_punct = norm in ("!", "?", ".", ";", ":", "(", ")", "[", "]", "«", "»", "-", "/")
        if not (has_khmer or is_basic_punct):
            continue
        for tag in sorted(tags):
            raw_candidates[(norm, map_to_ud_pos(tag, norm))] = None

    if not is_user_dict:
        # Ensure all 17 UD POS tags are registered in the system dictionary POS table
        present_pos = {pos for _, pos in raw_candidates}
        for ud_tag in UD_POS_TAGS:
            if ud_tag not in present_pos:
                raw_candidates[(f"__pos_{ud_tag.lower()}__", ud_tag)] = None

    # Gather all candidate (surface, ud_pos) pairs
    resolutions = {}
    extra_base_words: set[str] = set()
    extra_mode_b_words: set[str] = set()

    for surf, _ in list(raw_candidates.keys()):
        res = splitter.resolve(surf)
        resolutions[surf] = res
        for unit_a in res.split_a:
            extra_base_words.add(unit_a)
        for unit_b in res.split_b:
            if unit_b in splitter.overrides and splitter.overrides[unit_b].split_type == "B":
                extra_mode_b_words.add(unit_b)
                for sub_a in splitter.overrides[unit_b].split_a:
                    extra_base_words.add(sub_a)
            else:
                extra_base_words.add(unit_b)

    for base_w in sorted(extra_base_words):
        if base_w not in resolutions:
            resolutions[base_w] = splitter.resolve(base_w)
        # Ensure at least one POS entry exists for every constituent unit
        has_entry = any(s == base_w for s, _ in raw_candidates)
        if not has_entry:
            inferred_pos = map_to_ud_pos(None, base_w)
            raw_candidates[(base_w, inferred_pos)] = None

    for b_w in sorted(extra_mode_b_words):
        if b_w not in resolutions:
            resolutions[b_w] = splitter.resolve(b_w)
        has_entry = any(s == b_w for s, _ in raw_candidates)
        if not has_entry:
            override_pos = resolutions[b_w].override_pos or "NOUN"
            raw_candidates[(b_w, map_to_ud_pos(override_pos, b_w))] = None

    # Separate into Mode A, Mode B, and Mode C buckets
    mode_a_items: list[tuple[str, str]] = []
    mode_b_items: list[tuple[str, str]] = []
    mode_c_items: list[tuple[str, str]] = []

    for surf, pos in sorted(raw_candidates.keys()):
        res = resolutions[surf]
        effective_pos = map_to_ud_pos(res.override_pos, surf) if res.override_pos else pos
        # If a word is needed as a Mode A constituent by another compound, force it to Mode A
        if surf in extra_base_words or res.split_type == "A" or not res.split_a:
            mode_a_items.append((surf, effective_pos))
        elif res.split_type == "B":
            mode_b_items.append((surf, effective_pos))
        else:
            mode_c_items.append((surf, effective_pos))

    # Deduplicate while preserving order
    mode_a_items = list(dict.fromkeys(mode_a_items))
    mode_b_items = list(dict.fromkeys(mode_b_items))
    mode_c_items = list(dict.fromkeys(mode_c_items))

    entries: list[SudachiCSVEntry] = []
    surface_to_word_id: dict[str, str] = {}

    def _make_base_entry(
        surf: str,
        ud_pos: str,
        split_type: str = "A",
        split_info_a: str = "*",
        split_info_b: str = "*",
    ) -> SudachiCSVEntry:
        pid = pos_to_id(ud_pos)
        cost = estimator.estimate_word_cost(surf, ud_pos)
        pos_tuple = pos_to_sudachi_tuple(ud_pos)
        reading = estimator.stats.reading_forms.get(surf, surf)
        return SudachiCSVEntry(
            surface=surf,
            left_id=pid,
            right_id=pid,
            cost=cost,
            display_surface=surf,
            pos1=pos_tuple[0],
            pos2=pos_tuple[1],
            pos3=pos_tuple[2],
            pos4=pos_tuple[3],
            pos5=pos_tuple[4],
            pos6=pos_tuple[5],
            reading_form=reading,
            normalized_form=surf,
            dict_form_id="*",
            split_type=split_type,
            split_info_a=split_info_a,
            split_info_b=split_info_b,
        )

    # Pass 1: Mode A entries
    for surf, ud_pos in mode_a_items:
        word_idx = len(entries)
        entries.append(_make_base_entry(surf, ud_pos, split_type="A"))
        if surf not in surface_to_word_id and not surf.isdigit():
            surface_to_word_id[surf] = f"{id_prefix}{word_idx}"

    # Pass 2: Mode B entries
    for surf, ud_pos in mode_b_items:
        res = resolutions[surf]
        a_ids = [surface_to_word_id.get(u) for u in res.split_a]
        if all(a_ids) and len(a_ids) >= 2:
            info_a = "/".join(str(x) for x in a_ids)
            word_idx = len(entries)
            entries.append(
                _make_base_entry(
                    surf,
                    ud_pos,
                    split_type="B",
                    split_info_a=info_a,
                    split_info_b="*",
                )
            )
            if surf not in surface_to_word_id and not surf.isdigit():
                surface_to_word_id[surf] = f"{id_prefix}{word_idx}"
        else:
            # Fallback to Mode A if any constituent ID could not be resolved
            word_idx = len(entries)
            entries.append(_make_base_entry(surf, ud_pos, split_type="A"))
            if surf not in surface_to_word_id and not surf.isdigit():
                surface_to_word_id[surf] = f"{id_prefix}{word_idx}"

    # Pass 3: Mode C entries
    for surf, ud_pos in mode_c_items:
        res = resolutions[surf]
        a_ids = [surface_to_word_id.get(u) for u in res.split_a]
        b_units = res.split_b if res.split_b else res.split_a
        b_ids = [surface_to_word_id.get(u) for u in b_units]

        if all(a_ids) and len(a_ids) >= 2:
            info_a = "/".join(str(x) for x in a_ids)
            info_b = (
                "/".join(str(x) for x in b_ids)
                if (all(b_ids) and len(b_ids) >= 2)
                else info_a
            )
            word_idx = len(entries)
            entries.append(
                _make_base_entry(
                    surf,
                    ud_pos,
                    split_type="C",
                    split_info_a=info_a,
                    split_info_b=info_b,
                )
            )
            if surf not in surface_to_word_id and not surf.isdigit():
                surface_to_word_id[surf] = f"{id_prefix}{word_idx}"
        else:
            word_idx = len(entries)
            entries.append(_make_base_entry(surf, ud_pos, split_type="A"))
            if surf not in surface_to_word_id and not surf.isdigit():
                surface_to_word_id[surf] = f"{id_prefix}{word_idx}"

    if not is_user_dict:
        # Also include Sudachi's built-in default SimpleOovPlugin POS (名詞,普通名詞,一般,*,*,*)
        # at the end of the system dictionary POS table so `sudachipy ubuild` works out of the box
        entries.append(
            SudachiCSVEntry(
                surface="__sudachi_default_oov__",
                left_id=16,
                right_id=16,
                cost=32000,
                display_surface="__sudachi_default_oov__",
                pos1="名詞",
                pos2="普通名詞",
                pos3="一般",
                pos4="*",
                pos5="*",
                pos6="*",
                reading_form="__sudachi_default_oov__",
                normalized_form="__sudachi_default_oov__",
                dict_form_id="*",
                split_type="A",
            )
        )

    return entries


def write_sudachi_csv(
    entries: list[SudachiCSVEntry],
    output_csv_path: str | Path,
) -> Path:
    """Write 19-column `SudachiCSVEntry` rows to a UTF-8 CSV file."""
    out = Path(output_csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        for entry in entries:
            writer.writerow(entry.to_row())
    return out


def build_system_dic(
    csv_paths: list[str | Path] | str | Path,
    matrix_path: str | Path,
    output_dic_path: str | Path,
    description: str = "Sudachi-Khmer System Dictionary",
) -> Path:
    """Compile a Sudachi binary system dictionary (`system_khmer.dic`) using `sudachipy build`."""
    if isinstance(csv_paths, (str, Path)):
        csv_list = [str(csv_paths)]
    else:
        csv_list = [str(p) for p in csv_paths]

    out = Path(output_dic_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    cmd = [
        sys.executable,
        "-m",
        "sudachipy.command_line",
        "build",
        "-m",
        str(matrix_path),
        "-o",
        str(out),
        "-d",
        description,
        *csv_list,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(
            f"sudachipy build failed (exit {res.returncode}):\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
        )
    return out


def build_user_dic(
    csv_paths: list[str | Path] | str | Path,
    system_dic_path: str | Path,
    output_udic_path: str | Path,
    description: str = "Sudachi-Khmer User Dictionary",
) -> Path:
    """Compile a Sudachi binary user dictionary (`user_khmer.dic`) using `sudachipy ubuild`."""
    if isinstance(csv_paths, (str, Path)):
        csv_list = [str(csv_paths)]
    else:
        csv_list = [str(p) for p in csv_paths]

    out = Path(output_udic_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    cmd = [
        sys.executable,
        "-m",
        "sudachipy.command_line",
        "ubuild",
        "-s",
        str(system_dic_path),
        "-o",
        str(out),
        "-d",
        description,
        *csv_list,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(
            f"sudachipy ubuild failed (exit {res.returncode}):\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
        )
    return out
