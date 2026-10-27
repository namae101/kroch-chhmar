# Evaluation Guide & Results (`evaluation.md`, v0.2.0)

This document explains how to run the Khmer word-segmentation evaluation suite in **`kroch-chhmar`**, what each metric measures, and the results for the current (v0.2.0) held-out build.

> **v0.2.0 correction:** upstream khPOS `train.all2` *contains the entire `CLOSE-TEST` split*. Earlier versions skipped the separate `khpos_close_test.pos` file but still ingested those sentences through `train.all2`, inflating the CLOSE-TEST score. `prepare` now removes exact held-out sentences before any counts are committed. See §1.3 for the contamination experiment and corrected numbers.

---

## 1. How to Run the Evaluation

### 1.1 Fetch datasets, build the dictionary, and verify holdout integrity

```bash
uv run kroch-chhmar fetch all
uv run kroch-chhmar prepare
uv run kroch-chhmar build
```

`prepare` automatically treats `data/raw/khpos_open_test.pos` and `data/raw/khpos_close_test.pos` as holdout files. Pass extra holdout files with `--holdout path/to/gold.pos`, or disable auto-detection with `--no-auto-holdout`. The build log reports how many sentences were skipped (currently **1,028**).

### 1.2 Run the CLI evaluator

`kroch-chhmar evaluate` accepts any segmented corpus file (`-f`), a Sudachi split granularity mode (`--mode A`, `B`, or `C`), and a compound handling flag (`--join-compounds` or `--split-compounds`). Add `--json` to emit machine-readable metrics.

```bash
# Mode C against compound-level gold segmentation
uv run kroch-chhmar evaluate -f data/raw/khpos_open_test.pos --mode C --join-compounds --json

# Mode A against base-word (sub-compound) gold segmentation
uv run kroch-chhmar evaluate -f data/raw/khpos_open_test.pos --mode A --split-compounds --json

# CLOSE-TEST
uv run kroch-chhmar evaluate -f data/raw/khpos_close_test.pos --mode C --join-compounds
uv run kroch-chhmar evaluate -f data/raw/khpos_close_test.pos --mode A --split-compounds

# Built-in seed corpus
uv run kroch-chhmar evaluate -f data/seed/seed_corpus.txt --mode C
```

### 1.3 Holdout integrity and the khPOS CLOSE-TEST issue

The upstream khPOS `corpus-draft-ver-1.0/data/after-replace/train.all2` file is 12,000 lines, and **its first 1,000 lines are exactly the `CLOSE-TEST` split**:

```text
train.all2[:1000] == CLOSE-TEST     → True
CLOSE-TEST lines present in train   → 1000 / 1000
```

Because `prepare` ingests `train.all2` as `khpos_train.pos`, skipping the separate `khpos_close_test.pos` file is not sufficient. v0.2.0 therefore applies an exact-sentence `HoldoutFilter` before committing any token counts, compound overrides, or lexicon sub-parts.

To quantify the original bias, I rebuilt the dictionary under three conditions:

| Experiment | CLOSE-TEST Mode C F1 | OPEN-TEST Mode C F1 |
| :--- | ---: | ---: |
| Previous build (CLOSE-TEST inside training) | **96.25%** | 94.85% |
| Clean build (CLOSE/OPEN sentences removed) | **94.86%** | 94.69% |
| Size-matched control (1,022 random non-test lines removed) | **95.86%** | 94.58% |

Removing the test sentences costs ~1.0 F1 point beyond the effect of removing the same number of ordinary training sentences. The previously reported OPEN↔CLOSE gap of 1.45 points was therefore mostly leakage; on the corrected build it is **0.20 points** (94.66 vs 94.86) in Mode C and **0.28 points** (91.93 vs 92.21) in Mode A.

`tests/test_data_integrity.py` asserts zero exact-sentence overlap when the real khPOS files are present (`pytest` skips it otherwise).

---

## 2. Datasets & Metrics

### 2.1 Evaluation datasets

- **`CLOSE-TEST`** (1,000 sentences, 10,397 compound tokens / 12,467 base tokens): upstream this is the first 1,000 lines of `train.all2`. After v0.2.0 filtering it is a matched-distribution held-out set (same domain and annotation style as training), **not** an unseen-domain set.
- **`OPEN-TEST`** (1,000 sentences, 10,778 compound tokens / 12,931 base tokens): open-vocabulary news, rare loanwords (`ថេប្ល៊ិថ` *“tablet”*, `ខាល័រ` *“color”*), and personal names. 11 OPEN-TEST sentences also occur verbatim in `train.all2`; the holdout filter removes them.

### 2.2 What each metric measures

The evaluator concatenates the gold tokens of each sentence into a raw unsegmented string, runs `KhmerTokenizer`, and compares predicted character spans to gold spans.

1. **`Gold Tokens` / `Predicted Tokens`**: token counts under the chosen gold granularity.
2. **`Precision` / `Recall` / `F1`**: span-level token metrics.
3. **`Exact Sentence Match`**: percentage of sentences where the entire predicted token list equals the gold list. This is stricter than boundary F1.
4. **`Boundary F1`**: precision/recall over internal character boundaries (ignoring token identity). This is the standard segmentation metric when token labels may differ but boundaries agree.
5. **`OOV Rate`**: fraction of predicted tokens produced by Sudachi's OOV fallback. Most OOV tokens are Khmer/ASCII digits and punctuation.
6. **`Khmer-Word OOV`**: OOV tokens containing Khmer consonants/vowels/signs (excluding digits and punctuation). This isolates genuine lexical gaps such as unseen loanwords and typos.
7. **`KCC Boundary Violation Rate`**: fraction of predicted boundaries that split a Khmer Character Cluster. The tokenizer enforces this invariant in post-processing, so it should remain `0.00%`; it is a correctness guarantee rather than an independent quality score.

Gold annotations: `^` bound-suffix markers are always joined (`សេដ្ឋ^កិច្ច` → `សេដ្ឋកិច្ច`); `_`/`~` compounds are joined under `--join-compounds` and split under `--split-compounds`.

---

## 3. Evaluation Results

### 3.1 Held-out `OPEN-TEST` and `CLOSE-TEST`

| Test Set | Gold Granularity | Mode | Gold Tok. | Pred Tok. | Precision | Recall | **F1** | Exact Sent. | Boundary F1 | OOV | KCC |
| :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **OPEN** | Compounds joined | **C** | 10,778 | 10,669 | **95.14%** | **94.18%** | **94.66%** | 78.40% | **97.79%** | 2.75% | 0.00% |
| OPEN | Compounds joined | B | 10,778 | 11,159 | 90.16% | 93.35% | 91.73% | 75.00% | 96.21% | 2.63% | 0.00% |
| OPEN | Compounds joined | A | 10,778 | 12,886 | 67.72% | 80.97% | 73.76% | 27.70% | 89.73% | 2.27% | 0.00% |
| **OPEN** | Base words split | **A** | 12,931 | 12,886 | **92.09%** | **91.77%** | **91.93%** | 68.00% | **96.90%** | 2.27% | 0.00% |
| OPEN | Base words split | B | 12,931 | 11,159 | 81.69% | 70.50% | 75.68% | 30.20% | 90.92% | 2.63% | 0.00% |
| OPEN | Base words split | C | 12,931 | 10,669 | 80.17% | 66.14% | 72.48% | 30.40% | 89.11% | 2.75% | 0.00% |
| **CLOSE** | Compounds joined | **C** | 10,397 | 10,303 | **95.29%** | **94.43%** | **94.86%** | 83.00% | **97.90%** | 2.53% | 0.00% |
| CLOSE | Compounds joined | B | 10,397 | 10,755 | 90.47% | 93.58% | 92.00% | 78.60% | 96.36% | 2.43% | 0.00% |
| CLOSE | Compounds joined | A | 10,397 | 12,455 | 67.52% | 80.88% | 73.60% | 28.50% | 89.67% | 2.10% | 0.00% |
| **CLOSE** | Base words split | **A** | 12,467 | 12,455 | **92.25%** | **92.16%** | **92.21%** | 70.30% | **97.03%** | 2.10% | 0.00% |
| CLOSE | Base words split | B | 12,467 | 10,755 | 81.42% | 70.24% | 75.42% | 31.60% | 90.84% | 2.43% | 0.00% |
| CLOSE | Base words split | C | 12,467 | 10,303 | 80.06% | 66.17% | 72.45% | 32.00% | 89.14% | 2.53% | 0.00% |

**Khmer-word OOVs:** `OPEN-TEST` **11** tokens (`0.10%`), `CLOSE-TEST` **9** (`0.09%`). Examples include `ថេប្ល៊ិថ`, `ខាល័រ`, `យុត្តាធិការ`, `សុឆ្ឆន្ទៈ`, `ជង្គ្រោង`.

### 3.2 Full 14,000-sentence khPOS corpus

This table is **mostly in-sample**: up to 10,972 training sentences plus 1,000 CLOSE-TEST sentences now filtered from training and 1,000 OPEN-TEST sentences. Use §3.1 for held-out comparisons.

| Split | Gold Granularity | Mode | Exact Sentence Match | Token Precision | Token Recall | **Token F1** | **Boundary F1** | Speed (sent/s) |
| :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `train.all2` (12,000) | Compounds joined | **C** | 9,884 / 12,000 (**82.37%**) | 96.48% | 94.82% | **95.64%** | **98.23%** | 3,510 |
| `OPEN-TEST` (1,000) | Compounds joined | **C** | 784 / 1,000 (**78.40%**) | 95.14% | 94.18% | **94.66%** | **97.79%** | 3,555 |
| `CLOSE-TEST` (1,000) | Compounds joined | **C** | 830 / 1,000 (**83.00%**) | 95.29% | 94.43% | **94.86%** | **97.90%** | 3,699 |
| **All khPOS (14,000)** | Compounds joined | **C** | **11,498 / 14,000** (**82.13%**) | **96.30%** | **94.75%** | **95.52%** | **98.18%** | **3,558** |
| `train.all2` (12,000) | Base words split | **A** | 8,287 / 12,000 (**69.06%**) | 92.38% | 91.78% | **92.08%** | **96.98%** | 3,713 |
| `OPEN-TEST` (1,000) | Base words split | **A** | 680 / 1,000 (**68.00%**) | 92.09% | 91.77% | **91.93%** | **96.90%** | 3,841 |
| `CLOSE-TEST` (1,000) | Base words split | **A** | 703 / 1,000 (**70.30%**) | 92.25% | 92.16% | **92.21%** | **97.03%** | 3,680 |
| **All khPOS (14,000)** | Base words split | **A** | **9,670 / 14,000** (**69.07%**) | **92.35%** | **91.81%** | **92.08%** | **96.98%** | **3,710** |

### 3.3 Test-vocabulary upper bound (not a held-out result)

Earlier documentation reported a **`khPOS`-aligned dictionary** at 98.35% (OPEN, Mode C) / 94.89% (OPEN, Mode A) and 98.24% / 94.62% (CLOSE). That dictionary was extended with the test vocabulary (and, for CLOSE, test sentences). It is an **oracle/upper bound** illustrating how much headroom dictionary coverage provides; it is not comparable to §3.1 and is no longer presented as a headline result. It is retained here for historical context only:

| Configuration | Test Set | Mode | F1 | Status |
| :--- | :--- | :---: | ---: | :--- |
| `khPOS`-aligned dictionary | OPEN-TEST | C | 98.35% | test-vocabulary upper bound |
| `khPOS`-aligned dictionary | OPEN-TEST | A | 94.89% | test-vocabulary upper bound |
| `khPOS`-aligned dictionary | CLOSE-TEST | C | 98.24% | test-vocabulary upper bound |
| `khPOS`-aligned dictionary | CLOSE-TEST | A | 94.62% | test-vocabulary upper bound |

For a legitimate domain-adaptation measurement, extend the dictionary with a **separate development** vocabulary, then evaluate once on OPEN/CLOSE.

### 3.4 POS tagging on boundary-correct tokens

`kroch-chhmar evaluate-pos` compares predicted UD POS tags to the khPOS gold tags on tokens whose spans exactly match the gold segmentation. Boundary errors are reported separately as `Segmentation Coverage`; they are not counted as POS errors.

| Test Set | Mode | Gold Tokens | Aligned Tokens | Coverage | POS Accuracy | Macro F1 |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| OPEN-TEST | C | 10,778 | 10,151 | 94.18% | **75.20%** | 52.02% |
| OPEN-TEST | A | 12,931 | 11,867 | 91.77% | **61.69%** | 42.99% |
| CLOSE-TEST | C | 10,397 | 9,818 | 94.43% | **75.86%** | 54.86% |
| CLOSE-TEST | A | 12,467 | 11,490 | 92.16% | **62.09%** | 44.99% |

These numbers reflect the dictionary/heuristic POS priors described in the README: frequent corpus words are usually correct, while rare words and Mode A sub-tokens (which inherit a parent tag or a heuristic `NOUN` default) are much weaker. Macro F1 is low because several UD tags are rare. This is **not** a trained POS tagger benchmark; it is a diagnostic of the lexical priors shipped with the dictionary.

---

## 4. Discussion & Analysis

1. **Held-out quality:** Mode C reaches **94.66%** F1 on OPEN-TEST and **94.86%** on CLOSE-TEST; Mode A reaches **91.93%** and **92.21%**. Mode B is intermediate (**91.73%** / **92.00%** against compound gold; **75.68%** / **75.42%** against base-word gold). The Mode B numbers are not measured against a genuine B-level gold standard, because none exists in khPOS.
2. **Generalization is now honest but modest:** the held-out OPEN↔CLOSE gap is **0.20 points** in Mode C and **0.28 points** in Mode A. After fixing the CLOSE-TEST leak, there is no large “generalization gap” to explain; the two test sets behave almost identically.
3. **OOV is mostly non-lexical:** 2.75% of OPEN tokens are OOV, but only **11** contain Khmer letters (0.10%). The rest are digits and punctuation intentionally left to OOV handling.
4. **Dictionary extension still has clear value:** adding domain terms through `kroch-chhmar ubuild` is fast and removes lexical OOVs. The 98.35% upper bound shows the ceiling when the test vocabulary is known, not what to expect on unseen text.
5. **Mode B is not a complete intermediate layer:** only **2 of 14,572** Mode C entries use a curated Mode B intermediate. DP-derived long compounds split directly from C to A. The `prepare` log reports this ratio. A future version could construct B intermediates systematically, but arbitrary groupings would not be linguistically grounded.
6. **POS tags are lexical priors:** corpus-attested words use khPOS/ALT counts, RAC headwords use RAC labels, and untyped wordlists fall back to heuristics (59,043 of 87,119 rows are `NOUN`). No POS-accuracy benchmark is claimed.
7. **KCC guarantees hold:** 0.00% violations across all 14,000 sentences and all modes, enforced by `char.def` flags plus `enforce_kcc_boundaries`.

---

## 5. Error Analysis

Across all 14,000 khPOS sentences (current build):

- **Mode C:** **11,498 / 14,000 (82.13%)** exact sentence matches; **2,502** sentences contain at least one boundary difference.
- **Mode A:** **9,670 / 14,000 (69.07%)** exact matches; **4,330** sentences differ.

### 5.1 Mode C mismatch categories

#### A. Annotation inconsistency in khPOS (largest source)

khPOS annotators sometimes join compounds with `_`/`~` and sometimes leave the same words separated. The Viterbi decoder is deterministic, so it segments a phrase consistently every time.

**Merged by `kroch-chhmar` when khPOS left the words split:**

| Count | `khPOS` Gold (split) | `kroch-chhmar` (Mode C) | Meaning |
|---:|:---|:---|:---|
| **124×** | `ប្រទេស \| កម្ពុជា` | `ប្រទេសកម្ពុជា` | *Cambodia* |
| **96×** | `នៅ \| ពេល` | `នៅពេល` | *when / at the time* |
| **80×** | `រាជធានី \| ភ្នំពេញ` | `រាជធានីភ្នំពេញ` | *Phnom Penh Capital* |
| **64×** | `មិន \| ទាន់` | `មិនទាន់` | *not yet* |
| **61×** | `នាយក \| រដ្ឋមន្ត្រី` | `នាយករដ្ឋមន្ត្រី` | *Prime Minister* |
| **41×** | `ផ្សេង \| ទៀត` | `ផ្សេងទៀត` | *other / another* |
| **35×** | `ភាសា \| ខ្មែរ` | `ភាសាខ្មែរ` | *Khmer language* |
| **32×** | `ចេញ \| ពី` | `ចេញពី` | *out from* |
| **30×** | `រស់ \| នៅ` | `រស់នៅ` | *live / reside* |
| **29×** | `ពស់ \| កេងកង` | `ពស់កេងកង` | *Keng Kang snake* (folktale) |
| **26×** | `អង្គការ \| មិន \| មែន \| រដ្ឋាភិបាល` | `អង្គការមិនមែនរដ្ឋាភិបាល` | *non-governmental organization* |
| **24×** | `អគ្គ \| លេខាធិការ` | `អគ្គលេខាធិការ` | *Secretary-General* |

**Split by `kroch-chhmar` when khPOS joined the words:**

| Count | `khPOS` Gold (joined) | `kroch-chhmar` (Mode C) | Meaning |
|---:|:---|:---|:---|
| **198×** | `ត្រូវបាន` | `ត្រូវ \| បាន` | passive auxiliary *“was / has been”* |
| **57×** | `ឲ្យបាន` | `ឲ្យ \| បាន` | *so as to / in order to* |
| **41×** | `ពេលនេះ` | `ពេល \| នេះ` | *this time / now* |
| **34×** | `បានជា` | `បាន \| ជា` | *became / that is why* |
| **30×** | `ជាដើម` | `ជា \| ដើម` | *et cetera / as the beginning* |
| **28×** | `មើលឃើញ` | `មើល \| ឃើញ` | *to see* |
| **21×** | `ហើយនិង` | `ហើយ \| និង` | *and also* |
| **19×** | `ដំណើរការ` | `ដំណើរ \| ការ` | *operation / to operate* |
| **18×** | `ផ្លូវការ` | `ផ្លូវ \| ការ` | *official* |
| **18×** | `មនុស្សលោក` | `មនុស្ស \| លោក` | *mankind* |
| **18×** | `ទៅជា` | `ទៅ \| ជា` | *become* |
| **15×** | `ជាថ្មី` | `ជា \| ថ្មី` | *anew* |

#### B. Overlapping boundary shifts (`ការ` nominalizations)

When two valid compounds share a middle morpheme such as `ការ`, the decoder can attach `ការ` to the right word:

- `វិធានការ | កាត់បន្ថយ` → `វិធាន | ការកាត់បន្ថយ` (OPEN-TEST #123), because `ការកាត់បន្ថយ` has a slightly lower path cost.

#### C. OOV typos, rare loanwords, and names

- `ដុលារ` (typo for `ដុល្លារ` *“dollar”*) → `ដុ | លា | រ`.
- `ថេប្ល៊ិថ` (*“tablet”*) → `ថេ | ប្ល៊ិ | ថ`.
- `ខាល័រ` (*“color”*) contributes an OOV fragment because it is absent from all four dictionaries.
- `កូ | រេនា` (personal name) → `កូរេ | នា` because `កូរេ` (*“Korea”*) is a high-frequency word.

#### D. Structured token merging

`merge_structured_tokens()` prevents Sudachi's `SYMBOL` rule from splitting structured spans:

- URLs, domains, and emails: `http://abc.com`, `https://mptc.gov.kh/news?id=12`, `info@rac.edu.kh`.
- Formatted numbers and times: `២.៦២៥`, `១,៧៥`, `1,250.50`, `០៨:៣០`.
- Ellipses, peyyal, and abbreviations: `...`, `។ល។`, `គ.ស.`, `U.S.A.`.
- Hyphenated terms and tags: `COVID-19`, `កូវីដ-១៩`, `Wi-Fi`, `#Khmer2026`, `@moh_kh`.

A few khPOS sports scores and section numbers are annotated as three tokens (`១ | . | ០`, `២ | . | ១`), which the merger intentionally joins into `១.០` and `២.១`.

### 5.2 Concrete examples from the current build

These are illustrative snippets; the full per-sentence report can be regenerated with `--json`.

1. **Overlapping compound (`OPEN-TEST` #123):** Gold `វិធានការ | កាត់បន្ថយ` → Predicted `វិធាន | ការកាត់បន្ថយ`.
2. **OOV typo (`OPEN-TEST` #109):** Gold `១ | ដុលារ | ដែរ` → Predicted `១ | ដុ | លា | រ | ដែរ`.
3. **Unseen loanword (`OPEN-TEST` #76):** Gold `កុំព្យូទ័រ | ថេប្ល៊ិថ | ដែល` → Predicted `កុំព្យូទ័រ | ថេ | ប្ល៊ិ | ថ | ដែល`.
4. **Name ambiguity (`OPEN-TEST` #51):** Gold `ខ្ញុំ | ឈ្មោះ | កូ | រេនា` → Predicted `ខ្ញុំ | ឈ្មោះ | កូរេ | នា`.
5. **Structured number preserved (`OPEN-TEST` #82):** `២.៦២៥` is emitted as a single `NUM` token instead of `២ | . | ៦២៥`.

### 5.3 Mode A mismatches (vs. split gold)

Mode A keeps 2-KCC words whole unless an explicit override exists, which causes two directions of error:

**2-KCC compounds kept whole:**

| Count | Gold split | `kroch-chhmar` (Mode A) |
|---:|:---|:---|
| **436×** | `នៅ \| ក្នុង` | `នៅក្នុង` |
| **221×** | `ពួក \| គេ` | `ពួកគេ` |
| **174×** | `ភ្នំ \| ពេញ` | `ភ្នំពេញ` |
| **168×** | `ជា \| ច្រើន` | `ជាច្រើន` |
| **145×** | `ប្រជា \| ជន` | `ប្រជាជន` |
| **143×** | `កីឡា \| ករ` | `កីឡាករ` |
| **120×** | `ព្រះ \| អង្គ` | `ព្រះអង្គ` |
| **98×** | `អន្តរ \| ជាតិ` | `អន្តរជាតិ` |
| **96×** | `ម្សិល \| មិញ` | `ម្សិលមិញ` |
| **89×** | `រាជ \| ធានី` | `រាជធានី` |
| **75×** | `រស់ \| នៅ` | `រស់នៅ` |
| **75×** | `យ៉ាង \| ណា` | `យ៉ាងណា` |

**Multi-KCC words split although khPOS did not mark them with `_`:**

| Count | Gold | `kroch-chhmar` (Mode A) |
|---:|:---|:---|
| **112×** | `នគរបាល` | `នគរ \| បាល` |
| **103×** | `លីកាដូ` (LICADHO) | `លី \| កាដូ` |
| **73×** | `របាយការណ៍` | `របាយ \| ការណ៍` |
| **67×** | `ត្រូវតែ` | `ត្រូវ \| តែ` |
| **58×** | `រូបភាព` | `រូប \| ភាព` |
| **55×** | `វៀតណាម` | `វៀត \| ណាម` |
| **51×** | `មេដាយ` | `មេ \| ដាយ` |
| **47×** | `ដំណើរការ` | `ដំណើរ \| ការ` |
| **45×** | `ចំណាត់ថ្នាក់` | `ចំណាត់ \| ថ្នាក់` |
| **42×** | `ស្ថានភាព` | `ស្ថាន \| ភាព` |
| **42×** | `មូលហេតុ` | `មូល \| ហេតុ` |
| **36×** | `អូស្ត្រាលី` | `អូស្ត្រា \| លី` |

---

## 6. Reproducibility Checklist

```bash
# 1. Install and run the test suite (37 tests)
uv run --with pytest pytest -q
uv run --with ruff ruff check src tests example
uv run --with mypy mypy src/kroch_chhmar

# 2. Fetch checksum-pinned raw data
uv run kroch-chhmar fetch all

# 3. Build (filters held-out sentences, deterministic)
uv run kroch-chhmar prepare
uv run kroch-chhmar build

# 4. Reproduce a table row as JSON
uv run kroch-chhmar evaluate -f data/raw/khpos_open_test.pos --mode C --join-compounds --json
uv run kroch-chhmar evaluate -f data/raw/khpos_close_test.pos --mode C --join-compounds --json
uv run kroch-chhmar evaluate-pos -f data/raw/khpos_open_test.pos --mode C --join-compounds --json
```

- `prepare` writes the number of skipped held-out sentences (currently 1,028).
- `lex.csv` and `matrix.def` are byte-identical across `PYTHONHASHSEED` values (`tests/test_determinism.py`). The binary `.dic` embeds a build timestamp signature, so compare tokenization/test results rather than raw `.dic` hashes.
- `tests/test_data_integrity.py` verifies zero exact-sentence overlap against the real khPOS files when they are present.
- CI (`.github/workflows/build-wheel.yml`) runs Python 3.10–3.13, lint, tests, wheel/sdist builds, an isolated wheel install, and a from-wheel seed rebuild.

---

## 7. Known Limitations

- Only khPOS is benchmarked; ALT is used for training vocabulary but not as a held-out test set.
- No comparison against CRF/neural Khmer segmenters is included yet.
- Mode B has no independent B-level gold standard; its scores are projections against C or A gold.
- POS tags are lexical priors, not a trained tagger. Conditional POS accuracy is measured in §3.4, but macro F1 (~52% Mode C / ~43% Mode A) shows that rare tags remain weak.
- `CLOSE-TEST` is a matched-distribution held-out set, not a new domain. The OPEN↔CLOSE difference is small by construction.
- Hyperparameters (`7,200` unseen base, `500` per-KCC penalty, `600` split penalty, `220` transition scale) were calibrated in v0.1.0; no separate development split is shipped. For new domains, tune them on a dedicated dev corpus, then evaluate once on OPEN/CLOSE.
- Tokens are evaluated after Unicode normalization; offsets refer to normalized text.
