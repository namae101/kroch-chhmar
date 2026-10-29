# Changelog

## 0.2.0 — 2026-10-27

### Licensing & data profiles (breaking for the bundled dictionary)

- The wheel now bundles a **permissive** `system_khmer.dic` built only from MIT-licensed sources: the project seed data + SIL `khmerlbdict`. It has 35,883 entries and is redistributable; the previous 87,119-entry research dictionary used khPOS/ALT (CC BY-NC-SA 4.0) and RAC 2022 (research/non-commercial).
- Added `prepare --profile {full,permissive}`. `permissive` auto-ingests only `khmerlbdict_*.tsv`, keeps wordlist frequencies for emission costs, and never auto-ingests khPOS/ALT/RAC. `full` preserves the previous research pipeline.
- Added `scripts/rebuild_bundled.sh` and updated `LICENSE`, `README.md`, and `publishing.md` to separate the redistributable default from the local research build.
- Added `kroch-chhmar audit-overlap` for exact + normalized + MinHash/LSH near-duplicate corpus and lexicon audits.
- Added independent KCC validation against the `regex` module's UAX #29 grapheme clusters (`tests/test_kcc_reference.py`), including a hand-audited fixture and a full khPOS corpus check.

### Evaluation integrity (breaking for reported numbers)

- **Fixed CLOSE-TEST leakage.** Upstream khPOS `train.all2` contains the entire 1,000-line `CLOSE-TEST` split. `prepare` now applies an exact-sentence `HoldoutFilter` before committing token counts, compound overrides, or lexicon sub-parts (1,028 training lines skipped in the standard build).
- Added `--holdout` / `--no-auto-holdout` and a build-log holdout summary.
- Added `tests/test_data_integrity.py` (zero-overlap assertion against the real khPOS files) and `tests/test_holdout.py`.
- Re-generated all reported results. Held-out v0.2.0: **94.66% Mode C / 91.93% Mode A on OPEN-TEST** and **94.86% / 92.21% on CLOSE-TEST**. The previously reported 98.35% “aligned” result is now explicitly labelled a test-vocabulary upper bound.
- Added `Exact Sentence Match`, `Boundary Precision/Recall/F1`, and a genuine Khmer-letter OOV metric to `kroch-chhmar evaluate` (`--json` supported).
- Added `kroch-chhmar evaluate-pos` and POS diagnostics: conditional accuracy on boundary-correct tokens is ~75% (Mode C) / ~62% (Mode A), documenting the limits of the shipped lexical POS priors.

### Reproducibility

- Builds are deterministic: replaced `next(iter(tags))` with `CostEstimator.best_pos()` and added `tests/test_determinism.py` (byte-identical `lex.csv` / `matrix.def` across `PYTHONHASHSEED` values).
- Pinned SHA-256 checksums for all `fetch` sources; removed two dead upstream wordlists (`khmer-words`, `khmer-nltk-dict`).
- TLS verification is no longer disabled globally; an unverified retry is limited to the explicitly allowlisted legacy NICT host, with a warning.

### Packaging & CLI

- Packaged seed data (`resources/seed/`) and made the CLI fall back to repository configs/builds only when they exist, so the Python API and CLI agree in a checkout while installed wheels work out of the box.
- Included `configs/`, `data/seed/`, `example/app.py`, and `assets/` in the sdist.
- `prepare` now refuses to emit an 18-entry placeholder dictionary; missing raw data falls back to packaged seed/`--lexicon`/`--corpus` inputs with a warning.
- Added `--raw-dir`, `--timeout`, clearer user-dictionary signature-mismatch errors, atomic dictionary builds, and split-reference validation (`validate_split_references`).
- Fixed the temp-config leak (one JSON file per tokenizer instance) and exposed `system_dic_path` / `config_path` for diagnostics.
- Added `ruff` and `mypy` configuration, a Python 3.10–3.13 CI test matrix, and an isolated-wheel seed rebuild check.

### Data quality

- Cleaned RAC `t_pron` parsing; dictionary readings containing annotation fragments dropped from ~1,419 rows to 38 punctuation rows.
- Documented the Mode B limitation: only 2 / 14,572 Mode C entries currently use a curated B intermediate; DP-derived long compounds fall back to A.
- Escaped the FastAPI playground DOM output and added request size limits.
- Documented that token offsets refer to `normalize_khmer(text)`.

### Upgrade note

The bundled `system_khmer.dic` was rebuilt, so its Sudachi signature changed. **Rebuild any existing `user_khmer.dic` against the new system dictionary** (`kroch-chhmar ubuild ... --system-dic ...`); old user dictionaries will be rejected with a “not compatible” error.
