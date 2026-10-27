"""Build reproducibility test: identical inputs must produce identical artifacts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"


def _prepare(build_dir: Path, interim_dir: Path, hash_seed: str) -> None:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = hash_seed
    env["PYTHONPATH"] = str(SRC)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kroch_chhmar.cli",
            "prepare",
            "--seed",
            "--no-raw",
            "--no-auto-holdout",
            "--build-dir",
            str(build_dir),
            "--interim-dir",
            str(interim_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


def test_seed_build_is_deterministic_across_hash_seeds(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _prepare(first, tmp_path / "interim1", "0")
    _prepare(second, tmp_path / "interim2", "12345")

    assert (first / "lex.csv").read_bytes() == (second / "lex.csv").read_bytes()
    assert (first / "matrix.def").read_bytes() == (second / "matrix.def").read_bytes()
