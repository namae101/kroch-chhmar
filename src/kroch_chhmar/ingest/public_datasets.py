"""Downloaders and parsers for open-source Khmer lexicons and corpora (khPOS, ALT NOVA, khmerlbdict, etc.)."""

from __future__ import annotations

import hashlib
import io
import ssl
import urllib.error
import urllib.parse
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

try:  # pragma: no cover - exercised only in installed builds
    from importlib.metadata import PackageNotFoundError, version

    try:
        _VERSION = version("kroch-chhmar")
    except PackageNotFoundError:  # source checkout
        _VERSION = "0.2.0"
except Exception:  # pragma: no cover
    _VERSION = "0.2.0"

_USER_AGENT = f"kroch-chhmar/{_VERSION}"
_INSECURE_SSL_HOSTS = {"www2.nict.go.jp"}


@dataclass(frozen=True)
class PublicDatasetSpec:
    """Metadata, URL(s), and optional SHA-256 checksums for a public Khmer dataset."""

    name: str
    description: str
    # tuple of (url, target_filename, sha256-or-None)
    urls: tuple[tuple[str, str, str | None], ...]
    kind: str  # "lexicon", "corpus", or "zip"


PUBLIC_SOURCES: dict[str, PublicDatasetSpec] = {
    "khpos": PublicDatasetSpec(
        name="khpos",
        description="ye-kyaw-thu/khPOS manually word-segmented & POS-tagged Khmer corpus (12k sentences)",
        urls=(
            (
                "https://raw.githubusercontent.com/ye-kyaw-thu/khPOS/master/corpus-draft-ver-1.0/data/after-replace/train.all2",
                "khpos_train.pos",
                "2573de07a60cb1b214bb5d00ac4f4fc3144d13da33bc56e36b7c04d9a4ea0ed7",
            ),
            (
                "https://raw.githubusercontent.com/ye-kyaw-thu/khPOS/master/corpus-draft-ver-1.0/data/OPEN-TEST",
                "khpos_open_test.pos",
                "55b960e6049e84e8f9244dabe63604cdc392090c92facce9dd43fa490faa3456",
            ),
            (
                "https://raw.githubusercontent.com/ye-kyaw-thu/khPOS/master/corpus-draft-ver-1.0/data/CLOSE-TEST",
                "khpos_close_test.pos",
                "33f6e4af597e6824c5477ca69880ea23def81ef241fe070098ab806af998b9be",
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
                "0c6457d4a3327f3dc0b381704cbad71af120e963bfa1cdb06765fa0ed0c9098a",
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
                "213ce026b8e20cdf70ef38a8ebc4cca0117ebbe65dabac4c8483d99eebb53efd",
            ),
            (
                "https://raw.githubusercontent.com/silnrsi/khmerlbdict/master/src/KHSV.txt",
                "khmerlbdict_KHSV.tsv",
                "feb51dd6dd79991422016a5150308d5245d36657289a79d8868ac296de44e61d",
            ),
            (
                "https://raw.githubusercontent.com/silnrsi/khmerlbdict/master/src/KHOV.txt",
                "khmerlbdict_KHOV.tsv",
                "4617335e48605deda12ca89b88da4c09c2ca9dc1c9993d7a64946c2655fec37f",
            ),
            (
                "https://raw.githubusercontent.com/silnrsi/khmerlbdict/master/src/DFD.txt",
                "khmerlbdict_DFD.tsv",
                "20f7e3a1693968a88c49eb9d127451f21a01c17dffb1c72cf63e6643b9a87f5f",
            ),
            (
                "https://raw.githubusercontent.com/silnrsi/khmerlbdict/master/src/TD.txt",
                "khmerlbdict_TD.tsv",
                "f5cfa6568e9e52d5439a684075f960202501bef051fc2276c5be9480cb64d92c",
            ),
            (
                "https://raw.githubusercontent.com/silnrsi/khmerlbdict/master/src/HC.txt",
                "khmerlbdict_HC.tsv",
                "beaf34514b6f0f8bbd573356f2d0dd48d22fd67fcb3c494262d983d5d9ee2c5d",
            ),
            (
                "https://raw.githubusercontent.com/silnrsi/khmerlbdict/master/src/places.txt",
                "khmerlbdict_places.tsv",
                "7a86e4f60badaaa15cba86567524575d6c09b59e760df3f290c3d4b4af8e04d0",
            ),
            (
                "https://raw.githubusercontent.com/silnrsi/khmerlbdict/master/src/names.txt",
                "khmerlbdict_names.tsv",
                "7de738d0fb0c7284e24ae88cd8000fbda2f4645a946153e18a4565e674cad6eb",
            ),
            (
                "https://raw.githubusercontent.com/silnrsi/khmerlbdict/master/src/villages.txt",
                "khmerlbdict_villages.tsv",
                "3065bb198a6d0d59517be1fee7f077bfd4810b742ae54ffc2e1fffb7321cda68",
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
                "3c6e9908b7881d36c1e43ae66258554d5c1dc07cc75bc9ea660c985cd413ee76",
            ),
        ),
        kind="lexicon",
    ),
}


def _download_bytes(
    url: str,
    timeout: int = 30,
    expected_sha256: str | None = None,
) -> bytes:
    """
    Download raw bytes from `url`.

    TLS verification is always enforced. For the legacy NICT host (whose
    certificate chain is not trusted by modern clients), an unverified retry is
    allowed only after a warning; it is never used for arbitrary hosts.
    A SHA-256 checksum is required when `expected_sha256` is provided.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
    except urllib.error.URLError as exc:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        reason = str(getattr(exc, "reason", exc)).upper()
        is_certificate_error = isinstance(getattr(exc, "reason", None), ssl.SSLError) or (
            "CERTIFICATE" in reason or "SSL" in reason
        )
        if host in _INSECURE_SSL_HOSTS and is_certificate_error:
            warnings.warn(
                f"TLS certificate verification failed for {host}; retrying with an "
                "unverified context because this legacy host is explicitly allowlisted.",
                RuntimeWarning,
                stacklevel=2,
            )
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
                data = response.read()
        else:
            raise

    if expected_sha256:
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual}. "
                "The upstream dataset may have changed; pin a new revision deliberately."
            )
    return data


def _write_atomic(destination: Path, data: bytes) -> None:
    """Write `data` to `destination` atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    temporary.write_bytes(data)
    temporary.replace(destination)


def _extract_zip_bytes(raw_bytes: bytes, output_dir: Path) -> list[Path]:
    """Extract regular files from a ZIP payload using basenames only (no path traversal)."""
    saved: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            base_name = Path(member.filename).name
            if not base_name or base_name.startswith("."):
                continue
            destination = output_dir / base_name
            _write_atomic(destination, archive.read(member))
            saved.append(destination)
    return saved


def fetch_public_dataset(
    source_name: str,
    output_dir: str | Path,
    timeout: int = 30,
) -> list[Path]:
    """
    Download a public Khmer dataset by key (`khpos`, `alt-km-nova`, `khmerlbdict`,
    `khmer-dictionary-44k`, `all`) or direct HTTP(S) URL into `output_dir`.
    Returns a list of saved file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    if source_name.lower() == "all":
        for key in ("khpos", "alt-km-nova", "khmerlbdict", "khmer-dictionary-44k"):
            saved_paths.extend(fetch_public_dataset(key, out_dir, timeout=timeout))
        return saved_paths

    key = source_name.lower()
    if key in PUBLIC_SOURCES:
        spec = PUBLIC_SOURCES[key]
        for url, filename, expected_sha256 in spec.urls:
            raw_bytes = _download_bytes(url, timeout=timeout, expected_sha256=expected_sha256)
            if spec.kind == "zip" or filename.endswith(".zip"):
                saved_paths.extend(_extract_zip_bytes(raw_bytes, out_dir))
            else:
                destination = out_dir / filename
                _write_atomic(destination, raw_bytes)
                saved_paths.append(destination)
        return saved_paths

    if source_name.startswith("http://") or source_name.startswith("https://"):
        raw_bytes = _download_bytes(source_name, timeout=timeout)
        filename = source_name.rstrip("/").split("/")[-1] or "downloaded_data.tsv"
        if filename.endswith(".zip"):
            saved_paths.extend(_extract_zip_bytes(raw_bytes, out_dir))
        else:
            destination = out_dir / filename
            _write_atomic(destination, raw_bytes)
            saved_paths.append(destination)
        return saved_paths

    available = list(PUBLIC_SOURCES.keys()) + ["all"]
    raise ValueError(
        f"Unknown public dataset '{source_name}'. Available: {available} or pass an HTTP(S) URL."
    )
