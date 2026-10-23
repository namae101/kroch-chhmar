"""Hybrid rule-based + DP/Viterbi compound decomposer for Sudachi Modes A, B, and C."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sudachi_khmer.normalizer import count_kccs, get_kcc_boundaries, normalize_khmer


@dataclass
class SplitResolution:
    """Resolved Sudachi multi-granular split structure for a dictionary entry."""

    surface: str
    split_type: str = "A"  # "A", "B", or "C"
    split_a: list[str] = field(default_factory=list)
    split_b: list[str] = field(default_factory=list)
    override_pos: str | None = None


class CompoundSplitter:
    """
    Resolves Mode A (short/base), Mode B (middle/derived), and Mode C (long/compound)
    splits for Khmer dictionary entries using:
      1. Curated YAML overrides (`configs/compounds.yaml`)
      2. Derivational prefix stripping (`ការ-`, `ភាព-`, `សេចក្ដី-`, `អ្នក-`)
      3. KCC-constrained Dynamic Programming (DP) decomposition into base words.
    """

    def __init__(
        self,
        base_vocab: set[str] | None = None,
        word_costs: dict[str, int] | None = None,
        compounds_yaml_path: str | Path | None = None,
    ) -> None:
        self.base_vocab: set[str] = {normalize_khmer(w) for w in (base_vocab or set()) if w}
        self.word_costs: dict[str, int] = {
            normalize_khmer(k): v for k, v in (word_costs or {}).items()
        }
        self.prefixes: list[str] = []
        self.overrides: dict[str, SplitResolution] = {}

        if compounds_yaml_path:
            self.load_compounds_yaml(compounds_yaml_path)

    def load_compounds_yaml(self, path: str | Path) -> None:
        """Load curated prefix and compound override definitions from a YAML file."""
        p = Path(path)
        if not p.exists():
            return
        data: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

        for pref in data.get("prefixes", []):
            surf = normalize_khmer(pref.get("surface", ""))
            if surf:
                self.prefixes.append(surf)
                self.base_vocab.add(surf)

        for raw_surf, spec in data.get("overrides", {}).items():
            surf = normalize_khmer(raw_surf)
            mode = str(spec.get("mode", "A")).upper()
            split_a = [normalize_khmer(x) for x in spec.get("split_a", []) if x]
            split_b = [normalize_khmer(x) for x in spec.get("split_b", []) if x]
            pos = spec.get("pos")

            for unit in split_a:
                self.base_vocab.add(unit)

            self.overrides[surf] = SplitResolution(
                surface=surf,
                split_type=mode,
                split_a=split_a if mode in ("B", "C") else [],
                split_b=split_b if mode == "C" else [],
                override_pos=pos,
            )

    def _dp_segment_into_base_words(self, surface: str) -> list[str] | None:
        """
        Decompose `surface` into a sequence of >=2 known base words in `self.base_vocab`,
        strictly respecting Khmer Character Cluster (KCC) boundaries.
        """
        n = len(surface)
        valid_boundaries = sorted(get_kcc_boundaries(surface))
        if len(valid_boundaries) <= 2:
            return None

        # dp[i] = (best_score, backpointer_index, word)
        INF = 10**9
        dp: dict[int, tuple[int, int, str]] = {0: (0, -1, "")}

        for end in valid_boundaries:
            if end == 0:
                continue
            best_cost = INF
            best_prev = -1
            best_piece = ""

            for start in valid_boundaries:
                if start >= end or start not in dp:
                    continue
                # Disallow full-word self-match so we only find true decompositions
                if start == 0 and end == n:
                    continue

                piece = surface[start:end]
                if piece not in self.base_vocab:
                    continue
                # Avoid splitting into single-KCC noise unless it's a known prefix or >=1 KCC valid root
                if count_kccs(piece) < 1:
                    continue

                piece_cost = self.word_costs.get(piece, 4800) + 600  # small split penalty
                cand_cost = dp[start][0] + piece_cost
                if cand_cost < best_cost:
                    best_cost = cand_cost
                    best_prev = start
                    best_piece = piece

            if best_prev != -1:
                dp[end] = (best_cost, best_prev, best_piece)

        if n not in dp:
            return None

        parts: list[str] = []
        curr = n
        while curr > 0:
            _, prev, piece = dp[curr]
            parts.append(piece)
            curr = prev
        parts.reverse()
        return parts if len(parts) >= 2 else None

    def resolve(self, surface: str) -> SplitResolution:
        """
        Resolve the multi-granular split structure (`A`, `B`, or `C`) for `surface`.
        """
        norm = normalize_khmer(surface)

        # 1. Check explicit overrides first
        if norm in self.overrides:
            return self.overrides[norm]

        # Only decompose words composed of Khmer script characters (0x1780..0x17FF)
        if not all(0x1780 <= ord(c) <= 0x17FF for c in norm):
            return SplitResolution(surface=norm, split_type="A")

        num_kccs = count_kccs(norm)
        if num_kccs <= 2:
            # Short words (1-2 KCCs) default to indivisible Mode A base words
            return SplitResolution(surface=norm, split_type="A")

        # 2. Check derivational prefixes (e.g. ការ + base_word -> Mode B)
        valid_boundaries = get_kcc_boundaries(norm)
        for pref in self.prefixes:
            if norm.startswith(pref) and len(norm) > len(pref):
                split_idx = len(pref)
                if split_idx in valid_boundaries:
                    remainder = norm[split_idx:]
                    if remainder in self.base_vocab and count_kccs(remainder) >= 1:
                        return SplitResolution(
                            surface=norm,
                            split_type="B",
                            split_a=[pref, remainder],
                        )

        # 3. Try DP decomposition for longer words (>= 3 KCCs)
        parts = self._dp_segment_into_base_words(norm)
        if not parts:
            return SplitResolution(surface=norm, split_type="A")

        if len(parts) == 2 and num_kccs <= 5:
            # 2-constituent compound -> Mode B
            return SplitResolution(
                surface=norm,
                split_type="B",
                split_a=parts,
            )

        # >= 3 constituents or very long compound -> Mode C
        return SplitResolution(
            surface=norm,
            split_type="C",
            split_a=parts,
            split_b=[],
        )
