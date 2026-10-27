"""Tests for dataset checksums, safe ZIP extraction, and RAC pronunciation cleaning."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from kroch_chhmar.ingest import public_datasets
from kroch_chhmar.ingest.local_importers import clean_rac_pronunciation


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_checksum_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"khmer dataset"
    monkeypatch.setattr(
        public_datasets.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(payload)
    )
    digest = hashlib.sha256(payload).hexdigest()

    assert public_datasets._download_bytes("https://example.com/x", expected_sha256=digest) == payload
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        public_datasets._download_bytes("https://example.com/x", expected_sha256="0" * 64)


def test_zip_extraction_uses_basenames_only(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../evil.txt", "nope")
        archive.writestr("nested/good.txt", "ok")
    public_datasets._extract_zip_bytes(buffer.getvalue(), tmp_path)

    assert (tmp_path / "good.txt").read_text() == "ok"
    assert (tmp_path / "evil.txt").read_text() == "nope"
    assert not (tmp_path.parent / "evil.txt").exists()


def test_public_source_specs_are_pinned() -> None:
    assert set(public_datasets.PUBLIC_SOURCES) == {
        "khpos",
        "alt-km-nova",
        "khmerlbdict",
        "khmer-dictionary-44k",
    }
    for spec in public_datasets.PUBLIC_SOURCES.values():
        for url, filename, checksum in spec.urls:
            assert url.startswith("https://")
            assert filename
            assert checksum and len(checksum) == 64


def test_clean_rac_pronunciation() -> None:
    assert clean_rac_pronunciation("[ក]") == "ក"
    assert (
        clean_rac_pronunciation("[ក័ន-ដៈ-កៈ-ស័ញ-ញ៉ា] ឬ [កន់-ដក-ស័ញ-ញ៉ា]")
        == "ក័ន-ដៈ-កៈ-ស័ញ-ញ៉ា"
    )
    assert (
        clean_rac_pronunciation("ប្រូ-តេ-ស្តង់-និ-យំ ] (ជើង ត),")
        == "ប្រូ-តេ-ស្តង់-និ-យំ"
    )
    assert clean_rac_pronunciation("") is None
    assert clean_rac_pronunciation(None) is None
