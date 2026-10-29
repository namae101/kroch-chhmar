# TODO / Roadmap

Status: **v0.2.0 release candidate**. Four of the five P0 items are complete: the bundled dictionary is permissively licensed, corpus overlap is audited, KCC enforcement is validated against an external UAX #29 reference, and the CI workflow is updated for this release. CI verification on a real runner and publishing remain (publishing requires PyPI credentials). Items below are the remaining research, quality, and release work.

Legend: **P0** = blocks a credible release · **P1** = needed for publication-grade claims · **P2** = quality/maintenance.

---

## P0 — Release & correctness

- [ ] **Verify CI on GitHub.** `.github/workflows/build-wheel.yml` now runs the 3.10–3.13 matrix, Ruff, mypy, wheel/sdist inspection, isolated install, and a from-wheel permissive seed rebuild. The release push is the first real-runner verification; record the Actions run URL here once it passes.
- [ ] **Tag and publish `v0.2.0`.** The release is prepared but publishing is blocked on PyPI credentials. Once a token is available: run the `publishing.md` checklist, upload to TestPyPI, install from TestPyPI in a clean environment, then publish to PyPI and create the GitHub release (the workflow attaches artifacts on `release: published`).
- [x] **Resolve the bundled-dictionary license conflict.** The wheel now ships a **permissive** 35,883-entry dictionary built only from MIT-licensed seed data + SIL `khmerlbdict`. The full research build (`prepare --profile full`, 87,119 entries, khPOS/ALT/RAC) is opt-in and local-only. `LICENSE`, `README.md`, `publishing.md`, and `user_dic.md` document the split; `scripts/rebuild_bundled.sh` reproduces the bundled artifact.
- [x] **Audit remaining corpora for train/test overlap.** Added `kroch-chhmar audit-overlap` with normalized exact keys and MinHash/LSH near-duplicate detection. Results: khPOS train→OPEN 15 exact / 90 near at 0.8, train→CLOSE 1,016 exact (holdout-filtered) / 0 near; ALT internal 4 exact groups / 0 near; ALT↔khPOS 0 exact / 0 near; RAC 38,694 unique headwords, 0 exact duplicates, 826 morphological near pairs. See `data_audit.md`.
- [x] **Independent KCC validation.** Added `tests/test_kcc_reference.py`, which compares `iter_kccs()` with the third-party `regex` module's UAX #29 `\X` grapheme clusters on a hand-audited fixture, on tokenizer spans in Modes A/B/C, and on all 14,000 normalized khPOS sentences (0 disagreements across 363,297 clusters).

## P1 — Evaluation & research

- [ ] **ALT as an out-of-domain test set.** Create a held-out ALT split (e.g. last 2,000 sentences), rebuild the dictionary without it, and report Mode A/B/C token F1, exact match, boundary F1, and OOV. This is the missing generalization number.
- [ ] **External baselines.** Evaluate `khmer-nltk` (CRF) and KhmerCut on the same khPOS OPEN/CLOSE gold with identical normalization, joining rules, and span metrics. Optionally add a BiLSTM/transformer baseline. Without this, 94.7% has no reference point.
- [ ] **Dedicated development split for hyperparameters.** `7,200` unseen base, `500` per-KCC penalty, `600` split penalty, and `220` transition scale were tuned on v0.1.0 data. Carve a dev set from training (never OPEN/CLOSE), run a sensitivity grid, and report chosen values + variance.
- [ ] **Make Mode B a real intermediate layer.** Currently only **2 / 14,572** Mode C entries reference a curated B intermediate; the rest fall back to A. Design a principled B construction (constituent bracketing from ALT or RAC subwords), populate `split_info_b`, and report the `c_with_b` ratio (target: majority of multi-constituent C entries).
- [ ] **Improve POS quality or stop advertising it.** Conditional accuracy is **75.20%** (OPEN C) / **61.69%** (OPEN A), macro F1 ~52% / ~43%. Either train a tagger (CRF/BiLSTM on khPOS+ALT) or restrict `ud_pos` to a documented lexical prior and remove POS from headline API claims.
- [ ] **Use ZWSP as a boundary hint.** ZWSP is currently stripped by normalization. Preserve it as a soft/ hard boundary signal in the lattice and measure the effect on OPEN/CLOSE.
- [ ] **Error taxonomy at scale.** Current analysis covers only the top mismatch types. Add a reproducible script that emits categorized errors (annotation inconsistency, `ការ` attachment, OOV, structured span, name) with counts per test set, and track regressions across releases.

## P2 — Quality, performance, packaging

- [ ] **Clean remaining `reading_form` artifacts.** 38 dictionary rows still contain bracket/paren/comma characters (mostly punctuation headwords such as `(`, `[`). Filter punctuation-only headwords or blank their readings.
- [ ] **Map normalized offsets back to the original string.** `begin`/`end` refer to `normalize_khmer(text)`. Provide `tokenize_with_offsets()` (or a span map) that returns original-text offsets after ZWSP removal and codepoint replacement.
- [ ] **Make `.dic` builds reproducible.** `lex.csv` and `matrix.def` are byte-identical across hash seeds, but the compiled `.dic` embeds a timestamp signature. Either make the signature deterministic or document a canonical comparison procedure in CI.
- [ ] **Throughput profiling.** Measured ~3,500–3,800 sentences/s (single process, Apple Silicon), lower than the old 6,526 sent/s claim. Profile hot paths, add a repeatable benchmark script, and publish methodology (hardware, corpus, warm-up, batch size).
- [ ] **Golden regression tests for tokenization.** Add a small fixture of input → expected tokens per mode plus KCC invariants, so dictionary/algorithm changes produce reviewable diffs instead of silent metric drift.
- [ ] **Refresh `Report.md` ablation to v0.2.0.** The ablation table is still the v0.1.0 run; re-run the uncalibrated → calibrated → structured-merger sequence on the holdout-corrected pipeline so the delta attributed to each component is valid.
- [ ] **Reconcile remaining doc numbers.** `ingestion_strategy.md` retains historical v0.1.0 entry counts (99,797 / 81,921); label them as historical or recompute. Check README/Report for any other pre-v0.2.0 figures.
- [ ] **Add `kroch-chhmar --version`.** Useful for bug reports and wheel verification; currently only `kroch_chhmar.__version__` exists.
- [ ] **CI integration tests with real data.** `tests/test_data_integrity.py` skips without `data/raw/`. Add an opt-in CI job that fetches checksum-pinned khPOS and runs the overlap assertion, or commit a small synthetic fixture that exercises the same path.
- [ ] **Dictionary growth review.** 87,119 entries / 82,428 unique surfaces; 59,043 rows are `NOUN`. Audit POS defaults and duplicate surface+POS entries before the next release.

---

## Reference: v0.2.0 metrics (do not regress)

| Metric | OPEN C | OPEN A | CLOSE C | CLOSE A |
| :--- | ---: | ---: | ---: | ---: |
| Token F1 | 94.66% | 91.93% | 94.86% | 92.21% |
| Exact sentence match | 78.40% | 68.00% | 83.00% | 70.30% |
| Boundary F1 | 97.79% | 96.90% | 97.90% | 97.03% |
| Khmer-word OOV | 11 | 11 | 9 | 9 |
| POS accuracy (aligned) | 75.20% | 61.69% | 75.86% | 62.09% |

Full 14k khPOS corpus (mostly in-sample): **95.52% Mode C / 92.08% Mode A**.

Quality gates: `pytest` (63 tests, including the `regex` KCC reference), `ruff check src tests example scripts`, `mypy src/kroch_chhmar`, deterministic `prepare`, and an isolated-wheel permissive seed rebuild. See `evaluation.md` §6 for the reproducibility checklist.

> **Metric scope:** the table above is the **full research build** (`--profile full`, local only). The wheel ships the **permissive** build: OPEN C 77.57% / A 64.40%, CLOSE C 77.15% / A 63.95%, KCC 0.00%. Do not cite the research numbers for the bundled default.
