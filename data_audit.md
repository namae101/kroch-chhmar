# Corpus Overlap Audit (`data_audit.md`, v0.2.0)

This report records the P0 train/test overlap audit for the corpora used by the
**full research profile**. It complements the khPOS holdout fix described in
[`evaluation.md` §1.3](evaluation.md). The audit command is:

```bash
kroch-chhmar audit-overlap LEFT RIGHT [--threshold 0.8] [--near/--no-near] [--json]
```

Sentence keys are Unicode-normalized and remove ZWSP, pipe separators, POS
suffixes (`/NN`), compound markers (`_`/`~`), and bound-suffix markers (`^`).
Near-duplicate candidates use character 4-gram shingles, a 64-value MinHash
signature, 16x4 banded LSH, and signature-Jaccard verification. The threshold
below is 0.8 unless stated otherwise.

## 1. khPOS: exact leakage is fixed, near-duplicates remain

The upstream `train.all2` file contains the entire CLOSE-TEST split. The
v0.2.0 `HoldoutFilter` removes every training sentence whose normalized key
matches an OPEN/CLOSE key before any counts are committed. The build log
reports **1,028 skipped training lines**: 15 OPEN-TEST keys + 1,013 CLOSE-TEST
occurrences (CLOSE has 999 unique keys; 15 of them occur twice in training).

| Comparison | Left items | Right items | Exact pairs | Near pairs (Jaccard >= 0.8) |
| :--- | ---: | ---: | ---: | ---: |
| `khpos_train.pos` vs `khpos_open_test.pos` | 12,000 | 1,000 | **15** | **90** |
| `khpos_train.pos` vs `khpos_close_test.pos` | 12,000 | 1,000 | **1,016** | 0 |

Interpretation:

- The 1,016 CLOSE pairs are the known upstream duplication; they are removed
  by the exact holdout filter.
- The 15 OPEN pairs are also removed by the filter. An earlier README figure
  said 11; the normalized-key audit finds 15 unique keys.
- The **90 near-duplicate OPEN pairs are not removed** by the exact filter.
  They are paraphrases/light edits of training sentences, so OPEN-TEST scores
  may still have a small in-domain advantage. This is a P1 improvement
  candidate: add an opt-in near-duplicate holdout filter and report the
  resulting delta.

## 2. ALT NOVA: no cross-corpus sentence overlap

ALT is used as training vocabulary but has no official train/test split in this
project. Its 20,106 sentences contain only 4 exact duplicate groups (4 extra
items) and no near-duplicate pairs at 0.8.

| Comparison | Left items | Right items | Exact pairs | Near pairs |
| :--- | ---: | ---: | ---: | ---: |
| ALT internal | 20,106 | 20,106 | 4 groups / 4 extra items | 0 |
| `khpos_train.pos` vs ALT | 12,000 | 20,106 | 0 | 0 |
| `khpos_open_test.pos` vs ALT | 1,000 | 20,106 | 0 | 0 |
| `khpos_close_test.pos` vs ALT | 1,000 | 20,106 | 0 | 0 |

The four ALT duplicates are worth deduplicating internally, but they do not
create a khPOS/ALT train-test leak.

## 3. RAC 2022: a lexicon, not a sentence corpus

`RAC-Khmer-Dict-2022.csv` has 54,953 rows but only **38,694 unique normalized
headwords** (`t_main` + non-empty `t_subword`). The repeated `t_main` values are
structural: RAC repeats the headword for each subword row. After deduplication
there are **0 exact duplicate groups**.

A near-duplicate audit over the 38,694 headwords finds **826 pairs at 0.8**.
These are mostly derivational/morphological variants (e.g. `កតញ្ញុតា` /
`អកតញ្ញុតា`, `មត្តញ្ញុតា` / `អមត្តញ្ញុតា`), not duplicate sentences. This is
expected for a dictionary and is not a train/test contamination signal.

Lexical coverage against the corpora (exact headword matches, near detection
disabled):

| Comparison | RAC headwords | Corpus vocabulary | Shared headwords |
| :--- | ---: | ---: | ---: |
| RAC vs khPOS (train + OPEN + CLOSE) | 38,694 | 5,710 | 3,622 (9.4% of RAC) |
| RAC vs ALT | 38,694 | 25,320 | 4,301 (11.1% of RAC) |

## 4. Conclusions and follow-ups

1. **No ALT↔khPOS sentence leakage** was found at the exact or near-duplicate
   level. ALT can be used as an out-of-domain evaluation source without
   deduplicating against khPOS first.
2. **khPOS exact leakage is fixed**, but 90 OPEN-TEST sentences remain
   near-duplicates of training sentences. The exact holdout filter is
   intentionally conservative; a near-duplicate filter should be evaluated as
   a separate P1 experiment so the metric delta is visible.
3. **RAC is a lexicon**, not a sentence corpus. Its duplicate rows are
   structural and its near-duplicates are morphological variants. Use it for
   vocabulary/POS/readings, not as a held-out sentence set.
4. The audit is reproducible from `data/raw/`; that directory is gitignored, so
   this report is the committed record. Re-run the commands below after
   refreshing any upstream dataset.

## 5. Reproduction commands

```bash
# khPOS leakage
uv run kroch-chhmar audit-overlap data/raw/khpos_train.pos data/raw/khpos_open_test.pos --json
uv run kroch-chhmar audit-overlap data/raw/khpos_train.pos data/raw/khpos_close_test.pos --json

# ALT internal + cross-corpus
uv run kroch-chhmar audit-overlap data/raw/data_km.km-tok.nova --json
uv run kroch-chhmar audit-overlap data/raw/khpos_train.pos data/raw/data_km.km-tok.nova --json
uv run kroch-chhmar audit-overlap data/raw/khpos_open_test.pos data/raw/data_km.km-tok.nova --json
uv run kroch-chhmar audit-overlap data/raw/khpos_close_test.pos data/raw/data_km.km-tok.nova --json

# RAC headword structure
uv run kroch-chhmar audit-overlap data/raw/RAC-Khmer-Dict-2022.csv --json
```
