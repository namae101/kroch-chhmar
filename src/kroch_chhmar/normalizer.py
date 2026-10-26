"""Khmer Unicode canonicalization (via betterkhmer) and Khmer Character Cluster (KCC) engine."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Iterator

# ---------------------------------------------------------------------------
# BetterKhmer / SIL Global Khmer Normalizer (MIT License)
# Ensures identical-looking Khmer strings have a single canonical encoding.
# ---------------------------------------------------------------------------


class _Cats(enum.IntEnum):
    Other = 0
    Base = 1
    Robat = 2
    Coeng = 3
    Shift = 4
    Z = 5
    VPre = 6
    VB = 7
    VA = 8
    VPost = 9
    MS = 10
    MF = 11
    ZFCoeng = 12


_categories = (
    [_Cats.Base] * 35      # 1780-17A2 (Consonants)
    + [_Cats.Other] * 2    # 17A3-17A4 (Deprecated independent vowels)
    + [_Cats.Base] * 15    # 17A5-17B3 (Independent vowels)
    + [_Cats.Other] * 2    # 17B4-17B5 (Inherent vowels)
    + [_Cats.VPost]        # 17B6 (Vowel AA)
    + [_Cats.VA] * 4       # 17B7-17BA (Above vowels)
    + [_Cats.VB] * 3       # 17BB-17BD (Below vowels)
    + [_Cats.VPre] * 8     # 17BE-17C5 (Pre/compound vowels)
    + [_Cats.MS]           # 17C6 (Nikahit)
    + [_Cats.MF] * 2       # 17C7-17C8 (Reahmuk, Yuukaleapintu)
    + [_Cats.Shift] * 2    # 17C9-17CA (Muusikatoan, Triisap)
    + [_Cats.MS]           # 17CB (Bantoc)
    + [_Cats.Robat]        # 17CC (Robat)
    + [_Cats.MS] * 5       # 17CD-17D1 (Signs)
    + [_Cats.Coeng]        # 17D2 (Coeng)
    + [_Cats.MS]           # 17D3 (Bathamasat)
    + [_Cats.Other] * 9    # 17D4-17DC (Punctuation / Signs)
    + [_Cats.MS]           # 17DD (Atthacan)
)

_ZWSP = 0x200B
_ZWNJ = 0x200C
_ZWJ = 0x200D
_COENG = 0x17D2
_ROBAT = 0x17CC
_BA = 0x1794


def _charcat(o: int) -> _Cats:
    if 0x1780 <= o <= 0x17DD:
        return _categories[o - 0x1780]
    if o == _ZWNJ:
        return _Cats.Z
    if o == _ZWJ:
        return _Cats.ZFCoeng
    return _Cats.Other


def _is_base(r: int) -> bool:
    return (0x1780 <= r <= 0x17A2) or (0x17A5 <= r <= 0x17B3) or r == 0x25CC


def _is_non_ro(r: int) -> bool:
    return (0x1780 <= r <= 0x1799) or (0x179B <= r <= 0x17A2) or (0x17A5 <= r <= 0x17B3)


def _is_non_ba(r: int) -> bool:
    return (0x1780 <= r <= 0x1793) or (0x1795 <= r <= 0x17A2) or (0x17A5 <= r <= 0x17B3)


def _is_s1(r: int) -> bool:
    return (
        (0x1780 <= r <= 0x1783)
        or (0x1785 <= r <= 0x1788)
        or (0x178A <= r <= 0x178D)
        or (0x178F <= r <= 0x1792)
        or (0x1795 <= r <= 0x1797)
        or (0x179E <= r <= 0x17A0)
        or r == 0x17A2
    )


def _is_s2(r: int) -> bool:
    return (
        r in (0x1780, 0x1784, 0x178E, 0x1793, 0x1794, 0x17A1)
        or (0x1798 <= r <= 0x179D)
        or (0x17A3 <= r <= 0x17B3)
    )


def _is_vpre(r: int) -> bool:
    return 0x17C1 <= r <= 0x17C5


def _is_digit(r: int) -> bool:
    return 0x17E0 <= r <= 0x17E9


def _opt_robat(r: list[int], p: int) -> tuple[int, ...]:
    if p < len(r) and r[p] == _ROBAT:
        return (p, p + 1)
    return (p,)


def _coeng_ends(r: list[int], s: int) -> list[int]:
    n = len(r)
    res = []
    if s + 1 < n and r[s] == _COENG and _is_base(r[s + 1]):
        res.append(s + 2)
    if (
        s + 3 < n
        and r[s] == _COENG
        and _is_non_ro(r[s + 1])
        and r[s + 2] == _COENG
        and _is_base(r[s + 3])
    ):
        res.append(s + 4)
    return res


def _strong_ends(r: list[int], s: int, add) -> None:
    n = len(r)
    if s >= n:
        return
    if _is_s1(r[s]):
        for p in _opt_robat(r, s + 1):
            add(p)
            if p + 1 < n and r[p] == _COENG and _is_non_ba(r[p + 1]):
                q = p + 2
                add(q)
                if q + 1 < n and r[q] == _COENG and _is_non_ba(r[q + 1]):
                    add(q + 2)
    if _is_non_ba(r[s]):
        for p in _opt_robat(r, s + 1):
            if p + 1 < n and r[p] == _COENG and _is_s1(r[p + 1]):
                q = p + 2
                add(q)
                if q + 1 < n and r[q] == _COENG and _is_non_ba(r[q + 1]):
                    add(q + 2)
            if (
                p + 3 < n
                and r[p] == _COENG
                and _is_non_ba(r[p + 1])
                and r[p + 2] == _COENG
                and _is_s1(r[p + 3])
            ):
                add(p + 4)


def _nstrong_ends(r: list[int], s: int, add) -> None:
    n = len(r)
    if s >= n:
        return
    if _is_s2(r[s]):
        for p in _opt_robat(r, s + 1):
            add(p)
            if p + 1 < n and r[p] == _COENG and _is_s2(r[p + 1]):
                q = p + 2
                add(q)
                if q + 1 < n and r[q] == _COENG and _is_s2(r[q + 1]):
                    add(q + 2)
    if r[s] == _BA:
        for p in _opt_robat(r, s + 1):
            add(p)
            for e1 in _coeng_ends(r, p):
                add(e1)
                for e2 in _coeng_ends(r, e1):
                    add(e2)
    if _is_base(r[s]):
        for p in _opt_robat(r, s + 1):
            if (
                p + 3 < n
                and r[p] == _COENG
                and _is_non_ro(r[p + 1])
                and r[p + 2] == _COENG
                and r[p + 3] == _BA
            ):
                add(p + 4)
            if (
                p + 3 < n
                and r[p] == _COENG
                and r[p + 1] == _BA
                and r[p + 2] == _COENG
                and _is_base(r[p + 3])
            ):
                add(p + 4)


def _can_end_at(r: list[int], target: int, ends) -> bool:
    for s in range(target):
        found = [False]

        def _mark(e: int) -> None:
            if e == target:
                found[0] = True

        ends(r, s, _mark)
        if found[0]:
            return True
    return False


def _va_samyok_at(r: list[int], p: int) -> bool:
    n = len(r)
    if p >= n:
        return False
    c = r[p]
    if c == 0x17D0:
        return True
    if 0x17B7 <= c <= 0x17BA:
        return True
    if c in (0x17BE, 0x17BF, 0x17DD):
        return True
    if c == 0x17B6 and p + 1 < n and r[p + 1] == 0x17C6:
        return True
    return False


def _apply_shifter(r: list[int], ends, shifter: int) -> None:
    for k in range(len(r)):
        if r[k] != 0x17BB:
            continue
        ctx = _can_end_at(r, k, ends) or (
            k >= 1 and _is_vpre(r[k - 1]) and _can_end_at(r, k - 1, ends)
        )
        if ctx and _va_samyok_at(r, k + 1):
            r[k] = shifter


def _collapse_invis(r: list[int]) -> list[int]:
    n = len(r)
    out = []
    i = 0
    while i < n:
        g1_end = -1
        if r[i] == _ZWJ and i + 1 < n and r[i + 1] == _COENG:
            g1_end = i + 2
        elif r[i] == _COENG:
            g1_end = i + 1
        if g1_end >= 0:
            k = g1_end
            while k < n and r[k] in (_COENG, _ZWNJ, _ZWJ):
                k += 1
            if k > g1_end:
                out.extend(r[i:g1_end])
                i = k
                continue
        out.append(r[i])
        i += 1
    return out


def _pair_replace(r: list[int], a: int, b: int, *repl: int) -> list[int]:
    n = len(r)
    out = []
    i = 0
    while i < n:
        if i + 1 < n and r[i] == a and r[i + 1] == b:
            out.extend(repl)
            i += 2
            continue
        out.append(r[i])
        i += 1
    return out


def _pair_replace3(r: list[int], a: int, b: int, c: int, repl: int) -> list[int]:
    n = len(r)
    out = []
    i = 0
    while i < n:
        if i + 2 < n and r[i] == a and r[i + 1] == b and r[i + 2] == c:
            out.append(repl)
            i += 3
            continue
        out.append(r[i])
        i += 1
    return out


def _vowel_split(r: list[int], tail: int, head: int) -> list[int]:
    n = len(r)
    out = []
    i = 0
    while i < n:
        if r[i] == 0x17C1:
            if i + 2 < n and 0x17BB <= r[i + 1] <= 0x17BD and r[i + 2] == tail:
                out.append(head)
                out.append(r[i + 1])
                i += 3
                continue
            if i + 1 < n and r[i + 1] == tail:
                out.append(head)
                i += 2
                continue
        out.append(r[i])
        i += 1
    return out


def _coeng_ro(r: list[int]) -> list[int]:
    n = len(r)
    out = []
    i = 0
    while i < n:
        if (
            i + 3 < n
            and r[i] == _COENG
            and r[i + 1] == 0x179A
            and r[i + 2] == _COENG
            and 0x1780 <= r[i + 3] <= 0x17B3
        ):
            out.append(r[i + 2])
            out.append(r[i + 3])
            out.append(r[i])
            out.append(r[i + 1])
            i += 4
            continue
        out.append(r[i])
        i += 1
    return out


def _coeng_da(r: list[int]) -> list[int]:
    """
    Context-aware Khmer phonotactic canonicalization for Coeng DA (្ដ) vs Coeng TA (្ត):
      - Retroflex nasal ណ (0x178E) + Coeng TA (0x17D2 0x178F) -> ណ + Coeng DA (0x17D2 0x178A)
      - Dental nasal ន (0x1793) + Coeng DA (0x17D2 0x178A) -> ន + Coeng TA (0x17D2 0x178F)
    Other base consonants (ក, ផ, ម, ស, ប, etc.) preserve their etymological Coeng DA/TA
    so RAC 2022 standard spellings (e.g. ក្ដៅ, ផ្ដើម, ម្ដេច, សេចក្ដី) remain intact.
    """
    n = len(r)
    if n < 3:
        return r
    base = r[0]
    out = []
    i = 0
    while i < n:
        if i + 1 < n and r[i] == _COENG:
            if base == 0x178E and r[i + 1] == 0x178F:
                out.append(_COENG)
                out.append(0x178A)
                i += 2
                continue
            if base == 0x1793 and r[i + 1] == 0x178A:
                out.append(_COENG)
                out.append(0x178F)
                i += 2
                continue
        out.append(r[i])
        i += 1
    return out


def _lunar1(r: list[int]) -> list[int]:
    n = len(r)
    out = []
    i = 0
    while i < n:
        if (
            r[i] == 0x17E1
            and i + 3 < n
            and _is_digit(r[i + 1])
            and r[i + 2] == _COENG
            and r[i + 3] == 0x17D4
        ):
            v = 10 + (r[i + 1] - 0x17E0)
            if v > 15:
                out.extend(r[i : i + 4])
            else:
                out.append(0x19E0 + v)
            i += 4
            continue
        if (
            i + 2 < n
            and _is_digit(r[i])
            and r[i + 1] == _COENG
            and r[i + 2] == 0x17D4
        ):
            out.append(0x19E0 + (r[i] - 0x17E0))
            i += 3
            continue
        out.append(r[i])
        i += 1
    return out


def _lunar2(r: list[int]) -> list[int]:
    n = len(r)
    out = []
    i = 0
    while i < n:
        if r[i] == 0x17D4 and i + 1 < n and r[i + 1] == _COENG:
            if i + 3 < n and r[i + 2] == 0x17E1 and _is_digit(r[i + 3]):
                v = 10 + (r[i + 3] - 0x17E0)
                if v > 15:
                    out.extend(r[i : i + 4])
                else:
                    out.append(0x19F0 + v)
                i += 4
                continue
            if i + 2 < n and _is_digit(r[i + 2]):
                out.append(0x19F0 + (r[i + 2] - 0x17E0))
                i += 3
                continue
        out.append(r[i])
        i += 1
    return out


def _has_khmer(cps: list[int]) -> bool:
    for o in cps:
        if 0x1780 <= o <= 0x17FF:
            return True
    return False


def betterkhmer_normalize(txt: str, lang: str = "km") -> str:
    """Return the BetterKhmer canonical normalized form of `txt`."""
    cps = [ord(c) for c in txt]

    if lang == "xhm":
        out = []
        n = len(cps)
        for i, o in enumerate(cps):
            if 0x17B7 <= o <= 0x17C5 and i + 1 < n and cps[i + 1] == _COENG:
                out.append(_ZWJ)
            out.append(o)
        cps = out

    if not _has_khmer(cps):
        return "".join(chr(o) for o in cps)

    n = len(cps)
    cats = [_charcat(o) for o in cps]

    for i in range(1, n):
        if cps[i - 1] == _ZWJ or cps[i - 1] == _COENG:
            if cats[i] == _Cats.Base or cats[i] == _Cats.Coeng:
                cats[i] = cats[i - 1]

    i = 0
    res = []
    while i < n:
        if cats[i] != _Cats.Base:
            res.append(cps[i])
            i += 1
            continue
        j = i + 1
        while j < n and cats[j] > _Cats.Base:
            j += 1

        indices = sorted(range(i, j), key=lambda e: (cats[e], e))
        syl = [cps[k] for k in indices]

        syl = _collapse_invis(syl)
        syl = _pair_replace(syl, 0x17BE, 0x17B6, 0x17C4, 0x17B8)  # ើា -> ោី
        syl = _vowel_split(syl, 0x17B8, 0x17BE)                   # េ(◌)ី -> ើ(◌)
        syl = _vowel_split(syl, 0x17B6, 0x17C4)                   # េ(◌)ា -> ោ(◌)
        syl = _pair_replace(syl, 0x17BE, 0x17BB, 0x17BB, 0x17BE)  # ើុ -> ុើ
        _apply_shifter(syl, _strong_ends, 0x17CA)                 # strong -u -> ៊
        _apply_shifter(syl, _nstrong_ends, 0x17C9)                # weak   -u -> ៉
        syl = _coeng_ro(syl)
        syl = _coeng_da(syl)
        syl = _lunar1(syl)
        syl = _lunar2(syl)
        syl = _pair_replace3(syl, 0x17D4, 0x17D2, 0x17D4, 0x19F0)  # ។្។ -> ᧰

        res.extend(syl)
        i = j

    return "".join(chr(o) for o in res)


# ---------------------------------------------------------------------------
# Public Normalization & Zero-Width Cleanup
# ---------------------------------------------------------------------------


def normalize_khmer(
    text: str,
    strip_zwsp: bool = True,
    strip_joiners: bool = False,
) -> str:
    """
    Canonicalize Khmer text using BetterKhmer and clean up zero-width characters.

    Args:
        text: Input string.
        strip_zwsp: Whether to remove Zero-Width Space (\\u200b) and BOM (\\ufeff).
        strip_joiners: Whether to also strip ZWNJ (\\u200c) and ZWJ (\\u200d).
    """
    if not text:
        return ""

    # Legacy deprecated independent vowel replacements
    text = text.replace("\u17a3", "\u17a2").replace("\u17a4", "\u17a2\u17b6")
    # Remove inherent vowels (invisible deprecated codepoints U+17B4, U+17B5)
    text = text.replace("\u17b4", "").replace("\u17b5", "")

    if strip_zwsp:
        text = text.replace("\u200b", "").replace("\ufeff", "")
    if strip_joiners:
        text = text.replace("\u200c", "").replace("\u200d", "")

    return betterkhmer_normalize(text)


def fix_coeng_da_ta_phonotactics(text: str) -> str:
    """
    Apply context-aware Khmer phonotactic rules:
      - ណ + ្ត (retroflex + dental Coeng TA) -> ណ + ្ដ (retroflex + Coeng DA)
      - ន + ្ដ (dental + retroflex Coeng DA) -> ន + ្ត (dental + Coeng TA)
    """
    text = text.replace("\u178e\u17d2\u178f", "\u178e\u17d2\u178a")
    text = text.replace("\u1793\u17d2\u178a", "\u1793\u17d2\u178f")
    return text


def coeng_skeleton(text: str) -> str:
    """
    Return a canonical lookup key that treats Coeng DA (្ដ) and Coeng TA (្ត) as equivalent.
    Used to link visual twin variants (e.g. ក្តៅ <-> ក្ដៅ) to their authoritative RAC headword.
    """
    return text.replace("\u17d2\u178a", "\u17d2\u178f")


def generate_coeng_twins(surface: str) -> list[str]:
    """
    Return visual twin surface variants of `surface` where Coeng DA (`\\u17d2\\u178a`)
    and Coeng TA (`\\u17d2\\u178f`) are interchanged (excluding `surface` itself).
    """
    if "\u17d2\u178a" not in surface and "\u17d2\u178f" not in surface:
        return []
    candidates = [
        surface.replace("\u17d2\u178a", "\u17d2\u178f"),
        surface.replace("\u17d2\u178f", "\u17d2\u178a"),
        fix_coeng_da_ta_phonotactics(surface.replace("\u17d2\u178a", "\u17d2\u178f")),
        fix_coeng_da_ta_phonotactics(surface.replace("\u17d2\u178f", "\u17d2\u178a")),
    ]
    twins: list[str] = []
    for cand in candidates:
        if cand != surface and cand not in twins:
            twins.append(cand)
    return twins


# ---------------------------------------------------------------------------
# Khmer Character Cluster (KCC) Engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KCCSpan:
    """A single Khmer Character Cluster (orthographic syllable cluster) or non-Khmer token."""

    text: str
    begin: int
    end: int
    is_khmer_cluster: bool


def is_khmer_base_char(ch: str) -> bool:
    """Return True if `ch` is a Khmer base consonant or independent vowel."""
    if not ch:
        return False
    o = ord(ch[0])
    return (0x1780 <= o <= 0x17A2) or (0x17A3 <= o <= 0x17B3)


def is_kcc_dependent_char(ch: str) -> bool:
    """
    Return True if `ch` is a Khmer dependent character (Coeng, dependent vowel,
    register shifter, or diacritic sign) that cannot start a new word/KCC on its own.
    """
    if not ch:
        return False
    o = ord(ch[0])
    return (0x17B4 <= o <= 0x17D3) or o == 0x17DD or o in (_ZWNJ, _ZWJ)


def is_kcc_dependent_start(text: str, prev_ends_with_coeng: bool = False) -> bool:
    """
    Return True if `text` starts with a character that belongs to the previous KCC:
    - Any dependent vowel, sign, shifter, or Coeng (`0x17B4..0x17D3`, `0x17DD`)
    - Or any Khmer base consonant when the previous token ended with a Coeng (`\\u17d2`).
    """
    if not text:
        return False
    first = text[0]
    if is_kcc_dependent_char(first):
        return True
    if prev_ends_with_coeng and is_khmer_base_char(first):
        return True
    return False


def iter_kccs(text: str) -> Iterator[KCCSpan]:
    """
    Segment `text` into Khmer Character Clusters (KCCs) and non-Khmer characters.

    A KCC consists of:
      Base Consonant / Independent Vowel
      + (Coeng + Subscript Base)*
      + (Register Shifter / Robat / Dependent Vowels / Signs / ZWNJ / ZWJ)*
    """
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if is_khmer_base_char(ch):
            start = i
            i += 1
            while i < n:
                c = text[i]
                o = ord(c)
                if o == _COENG and i + 1 < n and is_khmer_base_char(text[i + 1]):
                    # Coeng + subscript consonant stays in the same KCC
                    i += 2
                elif is_kcc_dependent_char(c):
                    i += 1
                else:
                    break
            yield KCCSpan(text=text[start:i], begin=start, end=i, is_khmer_cluster=True)
        else:
            # Dangling dependent char or non-Khmer char
            start = i
            i += 1
            yield KCCSpan(text=text[start:i], begin=start, end=i, is_khmer_cluster=False)


def count_kccs(text: str) -> int:
    """Count the number of Khmer Character Clusters (or character units) in `text`."""
    return sum(1 for _ in iter_kccs(text))


def get_kcc_boundaries(text: str) -> set[int]:
    """Return the set of valid character offset boundaries in `text` that do not split any KCC."""
    boundaries = {0, len(text)}
    for span in iter_kccs(text):
        boundaries.add(span.begin)
        boundaries.add(span.end)
    return boundaries


def is_kcc_boundary(text: str, index: int) -> bool:
    """Return True if `index` is a valid KCC boundary within `text`."""
    if index <= 0 or index >= len(text):
        return True
    if text[index - 1] == "\u17d2":
        return False
    if is_kcc_dependent_char(text[index]):
        return False
    return True
