"""Downloaders and parsers for open-source Khmer lexicons and corpora (khPOS, ALT NOVA, khmerlbdict, etc.)."""

from __future__ import annotations

import io
import ssl
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PublicDatasetSpec:
    """Metadata and URL(s) for a public Khmer dataset."""

    name: str
    description: str
    urls: tuple[tuple[str, str], ...]  # tuple of (url, target_filename)
    kind: str  # "lexicon", "corpus", or "zip"


PUBLIC_SOURCES: dict[str, PublicDatasetSpec] = {
    "khpos": PublicDatasetSpec(
        name="khpos",
        description="ye-kyaw-thu/khPOS manually word-segmented & POS-tagged Khmer corpus (12k sentences)",
        urls=(
            (
                "https://raw.githubusercontent.com/ye-kyaw-thu/khPOS/master/corpus-draft-ver-1.0/data/after-replace/train.all2",
                "khpos_train.pos",
            ),
            (
                "https://raw.githubusercontent.com/ye-kyaw-thu/khPOS/master/corpus-draft-ver-1.0/data/OPEN-TEST",
                "khpos_open_test.pos",
            ),
            (
                "https://raw.githubusercontent.com/ye-kyaw-thu/khPOS/master/corpus-draft-ver-1.0/data/CLOSE-TEST",
                "khpos_close_test.pos",
            ),
        ),
        kind="corpus",
    ),
    "alt-km-nova": PublicDatasetSpec(
        name="alt-km-nova",
        description="NICT/NIPTICT Asian Language Treebank (ALT) Khmer NOVA tokenized & bracket-POS corpus (20k sentences)",
        urls=(
            (
                "https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/km-nova-181101.zip",
                "km-nova-181101.zip",
            ),
        ),
        kind="zip",
    ),
    "khmerlbdict": PublicDatasetSpec(
        name="khmerlbdict",
        description="SIL International Khmer LineBreaking frequency dictionaries (SEALang, Bible, villages, places, names)",
        urls=(
            (
                "https://raw.githubusercontent.com/silnrsi/khmerlbdict/master/src/seafreq.txt",
                "khmerlbdict_seafreq.tsv",
            ),
            (
                "https://raw.githubusercontent.com/silnrsi/khmerlbdict/master/src/KHSV.txt",
                "khmerlbdict_KHSV.tsv",
            ),
            (
                "https://raw.githubusercontent.com/silnrsi/khmerlbdict/master/src/KHOV.txt",
                "khmerlbdict_KHOV.tsv",
            ),
            (
                "https://raw.githubusercontent.com/silnrsi/khmerlbdict/master/src/DFD.txt",
                "khmerlbdict_DFD.tsv",
            ),
            (
                "https://raw.githubusercontent.com/silnrsi/khmerlbdict/master/src/TD.txt",
                "khmerlbdict_TD.tsv",
            ),
            (
                "https://raw.githubusercontent.com/silnrsi/khmerlbdict/master/src/HC.txt",
                "khmerlbdict_HC.tsv",
            ),
            (
                "https://raw.githubusercontent.com/silnrsi/khmerlbdict/master/src/places.txt",
                "khmerlbdict_places.tsv",
            ),
            (
                "https://raw.githubusercontent.com/silnrsi/khmerlbdict/master/src/names.txt",
                "khmerlbdict_names.tsv",
            ),
            (
                "https://raw.githubusercontent.com/silnrsi/khmerlbdict/master/src/villages.txt",
                "khmerlbdict_villages.tsv",
            ),
        ),
        kind="lexicon",
    ),
    "khmer-dictionary-44k": PublicDatasetSpec(
        name="khmer-dictionary-44k",
        description="Royal Academy of Cambodia (RAC 2022) 44k Khmer Dictionary with POS & pronunciation (seanghay/khmer-dictionary-44k)",
        urls=(
            (
                "https://huggingface.co/datasets/seanghay/khmer-dictionary-44k/resolve/main/RAC-Khmer-Dict-2022.csv",
                "RAC-Khmer-Dict-2022.csv",
            ),
        ),
        kind="lexicon",
    ),
    "khmer-words": PublicDatasetSpec(
        name="khmer-words",
        description="Open Khmer dictionary wordlist (seanghay/khmer-words)",
        urls=(
            (
                "https://raw.githubusercontent.com/seanghay/khmer-words/main/words.txt",
                "khmer_words.tsv",
            ),
        ),
        kind="lexicon",
    ),
    "khmer-nltk-dict": PublicDatasetSpec(
        name="khmer-nltk-dict",
        description="Khmer-NLTK dictionary wordlist",
        urls=(
            (
                "https://raw.githubusercontent.com/VietHoang1512/khmer-nltk/master/khmernltk/word_tokenize/dictionary.txt",
                "khmer_nltk_words.tsv",
            ),
        ),
        kind="lexicon",
    ),
}


def _download_bytes(url: str, timeout: int = 30) -> bytes:
    """Download raw bytes from `url` with fallback for legacy SSL certificates (e.g. NICT)."""
    req = urllib.request.Request(url, headers={"User-Agent": "kroch-chhmar/0.1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.URLError:
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read()


def fetch_public_dataset(
    source_name: str,
    output_dir: str | Path,
    timeout: int = 30,
) -> list[Path]:
    """
    Download a public Khmer dataset by key (`khpos`, `alt-km-nova`, `khmerlbdict`, `all`, etc.)
    or direct HTTP(S) URL into `output_dir`.
    Returns a list of saved file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    if source_name.lower() == "all":
        for key in ("khpos", "alt-km-nova", "khmerlbdict", "khmer-dictionary-44k"):
            saved_paths.extend(fetch_public_dataset(key, out_dir, timeout=timeout))
        return saved_paths

    if source_name in PUBLIC_SOURCES:
        spec = PUBLIC_SOURCES[source_name]
        for url, filename in spec.urls:
            raw_bytes = _download_bytes(url, timeout=timeout)
            if spec.kind == "zip" or filename.endswith(".zip"):
                zf = zipfile.ZipFile(io.BytesIO(raw_bytes))
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    base_name = Path(member.filename).name
                    if base_name.startswith("."):
                        continue
                    dest = out_dir / base_name
                    dest.write_bytes(zf.read(member.filename))
                    saved_paths.append(dest)
            else:
                dest = out_dir / filename
                dest.write_bytes(raw_bytes)
                saved_paths.append(dest)
        return saved_paths

    if source_name.startswith("http://") or source_name.startswith("https://"):
        raw_bytes = _download_bytes(source_name, timeout=timeout)
        fname = source_name.rstrip("/").split("/")[-1] or "downloaded_data.tsv"
        if fname.endswith(".zip"):
            zf = zipfile.ZipFile(io.BytesIO(raw_bytes))
            for member in zf.infolist():
                if not member.is_dir():
                    dest = out_dir / Path(member.filename).name
                    dest.write_bytes(zf.read(member.filename))
                    saved_paths.append(dest)
        else:
            dest = out_dir / fname
            dest.write_bytes(raw_bytes)
            saved_paths.append(dest)
        return saved_paths

    raise ValueError(
        f"Unknown public dataset '{source_name}'. Available: {list(PUBLIC_SOURCES.keys()) + ['all']} or pass an HTTP(S) URL."
    )
