"""Downloaders and parsers for open-source Khmer lexicons and corpora."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PublicDatasetSpec:
    """Metadata and URL for a public Khmer dataset."""

    name: str
    description: str
    url: str
    filename: str
    kind: str  # "lexicon" or "corpus"


PUBLIC_SOURCES: dict[str, PublicDatasetSpec] = {
    "khmer-words": PublicDatasetSpec(
        name="khmer-words",
        description="Open Khmer dictionary wordlist (seanghay/khmer-words / Chuon Nath derived)",
        url="https://raw.githubusercontent.com/seanghay/khmer-words/main/words.txt",
        filename="khmer_words.tsv",
        kind="lexicon",
    ),
    "khmer-nltk-dict": PublicDatasetSpec(
        name="khmer-nltk-dict",
        description="Khmer-NLTK dictionary wordlist",
        url="https://raw.githubusercontent.com/VietHoang1512/khmer-nltk/master/khmernltk/word_tokenize/dictionary.txt",
        filename="khmer_nltk_words.tsv",
        kind="lexicon",
    ),
}


def fetch_public_dataset(
    source_name: str,
    output_dir: str | Path,
    timeout: int = 20,
) -> Path:
    """
    Download a public Khmer dataset by key (`khmer-words`, `khmer-nltk-dict`) or direct URL
    into `output_dir`.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if source_name in PUBLIC_SOURCES:
        spec = PUBLIC_SOURCES[source_name]
        url = spec.url
        target = out_dir / spec.filename
    elif source_name.startswith("http://") or source_name.startswith("https://"):
        url = source_name
        fname = source_name.rstrip("/").split("/")[-1] or "downloaded_lexicon.tsv"
        target = out_dir / fname
    else:
        raise ValueError(
            f"Unknown public dataset '{source_name}'. Available: {list(PUBLIC_SOURCES.keys())} or pass an HTTP(S) URL."
        )

    req = urllib.request.Request(url, headers={"User-Agent": "sudachi-khmer/0.1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content = resp.read().decode("utf-8", errors="replace")

    target.write_text(content, encoding="utf-8")
    return target
