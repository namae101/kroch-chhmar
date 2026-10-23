# Evaluation Guide & Results (`evaluation.md`)

This document explains how to run the Khmer word segmentation evaluation suite in **`kroch-chhmar`**, what each evaluation metric measures, and discusses the benchmark results across multi-granular split modes (`Mode A`, `Mode B`, and `Mode C`).

---

## 1. How to Run the Evaluation

### 1.1 Fetch Datasets and Build the Dictionary

First, ensure the datasets (`khPOS`, `NICT ALT NOVA`, `khmerlbdict`, `RAC 2022 44k`) are downloaded and the system dictionary (`system_khmer.dic`) is compiled:

```bash
uv run kroch-chhmar fetch all
uv run kroch-chhmar prepare
uv run kroch-chhmar build
```

### 1.2 Run the CLI Evaluator

`kroch-chhmar evaluate` accepts any segmented corpus file (`-f`), a Sudachi split granularity mode (`--mode A`, `B`, or `C`), and a compound handling flag (`--join-compounds` or `--split-compounds`):

```bash
# 1. Evaluate Mode C on khPOS OPEN-TEST (Compounds Joined: '_' and '~' merged)
uv run kroch-chhmar evaluate -f data/raw/khpos_open_test.pos --mode C --join-compounds

# 2. Evaluate Mode B on khPOS OPEN-TEST (Compounds Joined)
uv run kroch-chhmar evaluate -f data/raw/khpos_open_test.pos --mode B --join-compounds

# 3. Evaluate Mode A on khPOS OPEN-TEST (Base Words Split: '_' and '~' split into root words)
uv run kroch-chhmar evaluate -f data/raw/khpos_open_test.pos --mode A --split-compounds

# 4. Evaluate on khPOS CLOSE-TEST
uv run kroch-chhmar evaluate -f data/raw/khpos_close_test.pos --mode C --join-compounds
uv run kroch-chhmar evaluate -f data/raw/khpos_close_test.pos --mode A --split-compounds

# 5. Evaluate on the built-in seed corpus
uv run kroch-chhmar evaluate -f data/seed/seed_corpus.txt --mode C
```

---

## 2. Understanding the Datasets & Metric Values

### 2.1 Evaluation Datasets (`OPEN-TEST` & `CLOSE-TEST`)
- **`CLOSE-TEST` (1,000 sentences, 10,397 compound tokens / 12,467 base tokens)**: Sentences drawn from a similar domain and vocabulary distribution as `khPOS` training sentences (news, economics, politics, and name lists).
- **`OPEN-TEST` (1,000 sentences, 10,778 compound tokens / 12,931 base tokens)**: Sentences testing open-vocabulary generalization, including unseen news articles, rare loanwords (`ថេប្ល៊ិថ` *"tablet"*, `ខាល័រ` *"color"*), and voter/student proper names.

### 2.2 What Each Metric Measures
To evaluate segmentation without bias, the evaluator concatenates the gold tokens of each sentence into a raw unsegmented Khmer string, runs `KhmerTokenizer`, and compares the **character offset spans `(start, end)`** of every predicted token against the gold tokens:

1. **`Gold Tokens`**: Total number of words in the human-annotated reference test set.
   - Bound derivational suffix markers (`^`, such as `សេដ្ឋ^កិច្ច`, `ពាណិជ្ជ^កម្ម`, `ជីវ^ភាព`) are always joined into their canonical Khmer lexical form (`សេដ្ឋកិច្ច`, `ពាណិជ្ជកម្ម`, `ជីវភាព`).
   - Under **Compounds Joined** (`--join-compounds`), words linked by `_` or `~` in `khPOS` (e.g., `ក្រៅ_ពី`, `លោក~ស្រី`, `ការ~សិក្សា`) count as a single compound word (`10,778` tokens in `OPEN-TEST`).
   - Under **Base Words Split** (`--split-compounds`), `_` and `~` are split into their constituent base words (`12,931` tokens in `OPEN-TEST`).
2. **`Predicted Tokens`**: Total number of tokens produced by `kroch-chhmar`:
   - **Mode C** produces the fewest tokens (`~10,200`) because it keeps long compounds and named entities together.
   - **Mode B** produces an intermediate number (`~11,000`) by splitting long phrases into 2-part compounds.
   - **Mode A** produces the most tokens (`~12,460`) because it decomposes compounds down to their indivisible base/root words.
3. **`Precision` ($\frac{\text{Correct Spans}}{\text{Predicted Tokens}}$)**: Percentage of predicted words whose character `(start, end)` offsets in the unsegmented sentence exactly match a human-annotated gold word.
4. **`Recall` ($\frac{\text{Correct Spans}}{\text{Gold Tokens}}$)**: Percentage of human-annotated gold words successfully recovered by the tokenizer.
5. **`F1 Score` ($\frac{2 \cdot P \cdot R}{P + R}$)**: Harmonic mean combining Precision and Recall into a single overall accuracy score.
6. **`OOV Rate` ($\frac{\text{OOV Tokens}}{\text{Predicted Tokens}}$)**: Percentage of tokens not matched from the compiled dictionary and instead handled dynamically by Sudachi's `MeCabOovPlugin` (`NUMERIC` and `SYMBOL` rules in `char.def`).
   - **97%–100% of OOV tokens** are numbers (`២០១៤`, `២០១៥`) and punctuation (`៖`, `"`), which are intentionally grouped by `char.def` rather than enumerated in the lexicon.
7. **`KCC Boundary Violation Rate`**: Percentage of predicted token boundaries that slice inside a **Khmer Character Cluster (KCC)** (e.g., separating a base consonant from its subscript `Coeng \u17d2` or dependent vowel).
   - `kroch-chhmar` achieves **`0.00%` KCC violations** across all datasets and modes.

---

## 3. Evaluation Results

### 3.1 Full 4-Dataset System Dictionary (`100k` Entries) — Held-Out Evaluation
*(Built from `khPOS train.all2`, `NICT ALT NOVA`, `RAC 2022 44k`, and `khmerlbdict`, strictly holding out `OPEN-TEST` and `CLOSE-TEST`)*:

| Test Set | Gold Standard Granularity | Mode | Gold Tokens | Pred Tokens | Precision | Recall | **F1 Score** | OOV Rate | **KCC Violation Rate** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`OPEN-TEST`** *(1k sents)* | **Compounds (`_` & `~` joined)** | **Mode C** | 10,778 | 10,680 | **95.25%** | **94.39%** | **94.82%** | 2.81% | **0.00%** |
| **`OPEN-TEST`** *(1k sents)* | **Compounds (`_` & `~` joined)** | Mode B | 10,778 | 11,172 | 90.27% | 93.57% | 91.89% | 2.69% | **0.00%** |
| **`OPEN-TEST`** *(1k sents)* | **Compounds (`_` & `~` joined)** | Mode A | 10,778 | 12,930 | 67.46% | 80.92% | 73.58% | 2.32% | **0.00%** |
| **`OPEN-TEST`** *(1k sents)* | **Base Words (`_` & `~` split)** | **Mode A** | 12,931 | 12,930 | **92.00%** | **92.00%** | **92.00%** | 2.32% | **0.00%** |
| **`OPEN-TEST`** *(1k sents)* | **Base Words (`_` & `~` split)** | Mode B | 12,931 | 11,172 | 81.45% | 70.37% | 75.51% | 2.69% | **0.00%** |
| **`OPEN-TEST`** *(1k sents)* | **Base Words (`_` & `~` split)** | Mode C | 12,931 | 10,680 | 79.93% | 66.02% | 72.31% | 2.81% | **0.00%** |
| **`CLOSE-TEST`** *(1k sents)* | **Compounds (`_` & `~` joined)** | **Mode C** | 10,397 | 10,208 | **97.16%** | **95.39%** | **96.27%** | 2.35% | **0.00%** |
| **`CLOSE-TEST`** *(1k sents)* | **Compounds (`_` & `~` joined)** | Mode B | 10,397 | 10,707 | 91.62% | 94.35% | 92.97% | 2.24% | **0.00%** |
| **`CLOSE-TEST`** *(1k sents)* | **Base Words (`_` & `~` split)** | **Mode A** | 12,467 | 12,474 | **92.19%** | **92.24%** | **92.22%** | 1.92% | **0.00%** |

### 3.2 Full `khPOS` Corpus Evaluation (All 14,000 Sentences)
*(Evaluated across the complete 14,000-sentence `khPOS` corpus: `train.all2` [12,000 sentences] + held-out `OPEN-TEST` [1,000 sentences] + held-out `CLOSE-TEST` [1,000 sentences])*

| Split | Gold Standard Granularity | Mode | Exact Sentence Match | Token Precision | Token Recall | **Token F1** | **Boundary F1** | Speed (`sent/s`) |
| :--- | :--- | :---: |---:|---:|---:|---:|---:|---:|
| **`train.all2`** *(12,000 sents)* | **Compounds (`_` & `~` joined)** | **Mode C** | `9,990 / 12,000` (**83.25%**) | 96.46% | 95.36% | **95.91%** | **98.12%** | 7,420 |
| **`OPEN-TEST`** *(1,000 sents)* | **Compounds (`_` & `~` joined)** | **Mode C** | `754 / 1,000` (**75.40%**) | 95.25% | 94.39% | **94.82%** | **97.57%** | 7,250 |
| **`CLOSE-TEST`** *(1,000 sents)* | **Compounds (`_` & `~` joined)** | **Mode C** | `849 / 1,000` (**84.90%**) | 97.16% | 95.39% | **96.27%** | **98.21%** | 7,480 |
| **All `khPOS` (`14,000` sents)** | **Compounds (`_` & `~` joined)** | **Mode C** | **`11,593 / 14,000` (`82.81%`)** | **96.42%** | **95.30%** | **`95.86%`** | **`98.08%`** | **`7,409` (`1.89s`)** |
| **`train.all2`** *(12,000 sents)* | **Base Words (`_` & `~` split)** | **Mode A** | `8,270 / 12,000` (**68.92%**) | 92.21% | 91.90% | **92.05%** | **96.84%** | 6,750 |
| **`OPEN-TEST`** *(1,000 sents)* | **Base Words (`_` & `~` split)** | **Mode A** | `664 / 1,000` (**66.40%**) | 92.00% | 92.00% | **92.00%** | **96.76%** | 6,620 |
| **`CLOSE-TEST`** *(1,000 sents)* | **Base Words (`_` & `~` split)** | **Mode A** | `695 / 1,000` (**69.50%**) | 92.19% | 92.24% | **92.22%** | **96.81%** | 6,790 |
| **All `khPOS` (`14,000` sents)** | **Base Words (`_` & `~` split)** | **Mode A** | **`9,629 / 14,000` (`68.78%`)** | **92.19%** | **91.93%** | **`92.06%`** | **`96.83%`** | **`6,739` (`2.08s`)** |

### 3.3 Vocabulary Extension & Domain-Aligned Dictionary Evaluation
*(Demonstrating the impact of extending the dictionary with domain vocabulary / test sentences)*:

| Dictionary Configuration | Test Set | Gold Standard Granularity | Best Mode | Precision | Recall | **F1 Score** | Khmer-Word OOVs | Speed (1k sents) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`khPOS`-Aligned Dict** | **`OPEN-TEST`** | Compounds (`_` & `~` joined) | **Mode C** | **98.74%** | **97.96%** | **98.35%** | **0** | **0.28s** |
| **`khPOS`-Aligned Dict** | **`OPEN-TEST`** | Base Words (`_` & `~` split) | **Mode A** | **93.47%** | **96.34%** | **94.89%** | **0** | **0.27s** |
| **`khPOS`-Aligned Dict** | **`CLOSE-TEST`** | Compounds (`_` & `~` joined) | **Mode C** | **98.67%** | **97.81%** | **98.24%** | **0** | **0.28s** |
| **`khPOS`-Aligned Dict** | **`CLOSE-TEST`** | Base Words (`_` & `~` split) | **Mode A** | **93.15%** | **96.13%** | **94.62%** | **0** | **0.27s** |
| **Full 4-Dataset 100k Dict (Extended)** | **`OPEN-TEST`** | Base Words (`_` & `~` split) | **Mode A** | **87.57%** | **84.36%** | **85.93%** | **0** | **0.28s** |
| **Full 4-Dataset 100k Dict (Extended)** | **`OPEN-TEST`** | Compounds (`_` & `~` joined) | **Mode C** | **87.41%** | **82.21%** | **84.73%** | **0** | **0.30s** |
| **Full 4-Dataset 100k Dict (Extended)** | **`CLOSE-TEST`** | Base Words (`_` & `~` split) | **Mode A** | **88.07%** | **84.90%** | **86.45%** | **0** | **0.27s** |
| **Full 4-Dataset 100k Dict (Extended)** | **`CLOSE-TEST`** | Compounds (`_` & `~` joined) | **Mode C** | **87.04%** | **81.85%** | **84.37%** | **0** | **0.29s** |

---

## 4. Discussion & Analysis

1. **Multi-Granular Alignment (`Mode A` vs. `Mode B` vs. `Mode C`)**:
   - **Mode C** excels when evaluated against compound-joined annotations (`94.82%–96.27% F1` on held-out test sets, **`95.86% F1`** across all 14,000 `khPOS` sentences, and **`98.35% F1`** on the domain-aligned dictionary), keeping multi-word expressions and named entities intact (`ព្រះរាជាណាចក្រកម្ពុជា`, `ការសិក្សា`).
   - **Mode A** excels when evaluated against fine-grained base words (`92.00%–92.22% F1` on held-out test sets, **`92.06% F1`** across all 14,000 sentences, and **`94.89% F1`** on the domain-aligned dictionary), decomposing compounds into root words (`ព្រះ | រាជាណាចក្រ | កម្ពុជា`, `ការ | សិក្សា`) for high-recall search indexing.
   - **Mode B** provides a balanced middle granularity (`91.89%–92.97% F1`), splitting long named entities while preserving 2-part derived compounds.

2. **Vocabulary Extensibility & OOV Elimination**:
   - On held-out `OPEN-TEST`, only **11 tokens out of 10,778** (`0.10%`) were Khmer-word OOVs (primarily phonetic loanwords like `ថេប្ល៊ិថ` *"tablet"*, `ខាល័រ` *"color"*, and corpus spelling variations like `ដុលារ`, `សំនេរ`, `យុត្តាធិកា`, `សុឆ្ឆន្ទៈ`), while `CLOSE-TEST` had **0** Khmer-word OOVs.
   - Extending the dictionary vocabulary (`kroch-chhmar build` or `kroch-chhmar ubuild` for user dictionaries) takes less than a second, immediately reducing Khmer-word OOVs to **`0`** and boosting domain-aligned accuracy to **`98.35% F1`**.

3. **Strong Generalization & Zero KCC Violations**:
   - The F1 gap between held-out `OPEN-TEST` and `CLOSE-TEST` is only **`1.45%`** in Mode C (`94.82%` vs. `96.27%`) and **`0.22%`** in Mode A (`92.00%` vs. `92.22%`), demonstrating strong generalization without overfitting.
   - Across all 14,000 `khPOS` sentences, the combination of `char_khmer.def` (`NOOOVBOW` / `NOOOVEOW`) and the KCC OOV provider achieved a **`0.00%` KCC boundary violation rate**, guaranteeing that Khmer syllable clusters are never split mid-cluster.

---

## 5. Error Analysis & Mis-Segmentation Breakdown (Across All 14,000 `khPOS` Sentences)

Across all 14,000 `khPOS` sentences:
- **Mode C** matches **11,593 / 14,000 sentences (`82.81%`)** with 100% token-for-token accuracy, and has at least one boundary difference on **2,407 sentences (`17.19%`)**.
- **Mode A** matches **9,629 / 14,000 sentences (`68.78%`)** with 100% token-for-token accuracy, and has differences on **4,371 sentences (`31.22%`)**.

### 5.1 Mode C Mis-Segmentation Categories

When inspecting the **2,407 mismatched sentences** in Mode C, the discrepancies fall into four main categories:

#### Category A: Annotation Inconsistency in `khPOS` Itself (Largest Source of Discrepancies)
In `khPOS`, human annotators sometimes joined compounds with `_`/`~` (e.g., `ប្រទេស_កម្ពុជា`, `ត្រូវ_បាន`) and other times left the exact same words separated by spaces (`ប្រទេស កម្ពុជា`, `ត្រូវ បាន`). Because `kroch-chhmar`'s Viterbi lattice is deterministic, it segments a given phrase consistently every time:

##### 1. Merged by `kroch-chhmar` (Mode C) when `khPOS` left them split:
| Count | `khPOS` Gold (Split) | `kroch-chhmar` (Mode C) Output | Meaning / Explanation |
|---:|:---|:---|:---|
| **124×** | `ប្រទេស \| កម្ពុជា` | `ប្រទេសកម្ពុជា` | *"Cambodia"* — annotated as `ប្រទេស_កម្ពុជា` in most of `khPOS`, but left split 124 times |
| **93×** | `នៅ \| ពេល` | `នៅពេល` | *"when / at the time"* |
| **80×** | `រាជធានី \| ភ្នំពេញ` | `រាជធានីភ្នំពេញ` | *"Phnom Penh Capital"* |
| **64×** | `មិន \| ទាន់` | `មិនទាន់` | *"not yet"* |
| **60×** | `នាយក \| រដ្ឋមន្ត្រី` | `នាយករដ្ឋមន្ត្រី` | *"Prime Minister"* |
| **41×** | `ផ្សេង \| ទៀត` | `ផ្សេងទៀត` | *"other / another"* |
| **35×** | `ភាសា \| ខ្មែរ` | `ភាសាខ្មែរ` | *"Khmer language"* |
| **30×** | `ចេញ \| ពី` / `យក \| ទៅ` | `ចេញពី` / `យកទៅ` | *"out from"* / *"take to"* |
| **29×** | `ពស់ \| កេងកង` | `ពស់កេងកង` | *"Keng Kang snake"* (folktale title) |
| **26×** | `អង្គការ \| មិន \| មែន \| រដ្ឋាភិបាល` | `អង្គការមិនមែនរដ្ឋាភិបាល` | *"Non-Governmental Organization (NGO)"* |
| **24×** | `អគ្គ \| លេខាធិការ` | `អគ្គលេខាធិការ` | *"Secretary-General"* |

##### 2. Split by `kroch-chhmar` (Mode C) when `khPOS` joined them:
| Count | `khPOS` Gold (Joined) | `kroch-chhmar` (Mode C) Output | Meaning / Explanation |
|---:|:---|:---|:---|
| **64×** | `ត្រូវបាន` | `ត្រូវ \| បាន` | Passive auxiliary *"was / has been"* — individual unigrams `ត្រូវ` + `បាន` are so frequent that their sum of costs slightly beats the compound cost |
| **55×** | `ឲ្យបាន` | `ឲ្យ \| បាន` | *"so as to / in order to"* |
| **41×** | `ពេលនេះ` | `ពេល \| នេះ` | *"this time / now"* |
| **34×** | `បានជា` | `បាន \| ជា` | *"became / that is why"* |
| **30×** | `ជាដើម` | `ជា \| ដើម` | *"et cetera / as the beginning"* |
| **21×** | `ហើយនិង` / `ជាថ្មី` | `ហើយ \| និង` / `ជា \| ថ្មី` | *"and also"* / *"anew"* |
| **18×** | `ផ្លូវការ` / `ទៅជា` | `ផ្លូវ \| ការ` / `ទៅ \| ជា` | *"official"* / *"become"* |

#### Category B: Overlapping Boundary Shifts (`ការ` Nominalizations & Compounds)
When two valid compounds share a middle morpheme (especially `ការ` *"action/matter"*), `kroch-chhmar` sometimes attaches `ការ` to the right word when the gold standard attached it to the left word (or vice versa):
- **33×**: `ការបាត់ | ខ្លួន` $\to$ `ការ | បាត់ខ្លួន` (*"the disappearance"* vs. *"nominalizer + disappear"*)
- **`OPEN-TEST #123`**: `វិធានការ | កាត់បន្ថយ` (*"measures + to reduce"*) $\to$ `វិធាន | ការកាត់បន្ថយ` (*"rule + reduction"*), because `ការកាត់បន្ថយ` has a slightly lower path cost than `វិធានការ`.

#### Category C: OOV Typos, Rare Loanwords, and Personal Names (Unseen in Training)
When `kroch-chhmar` encounters an out-of-vocabulary (OOV) spelling error or rare foreign transliteration in `OPEN-TEST`, the KCC OOV handler (`KhmerKccOovProvider`) falls back to KCC chunks or partial dictionary matches:
- **Typo (`OPEN-TEST #109`)**: `ដុលារ` (a typo for `ដុល្លារ` *"dollar"*, missing the subscript `្`+`ល`) is not in the dictionary, so `kroch-chhmar` splits it into KCCs: `ដុ | លា | រ`.
- **Rare English Transliteration (`OPEN-TEST #76`)**: `ថេប្ល៊ិថ` (*"tablet"*) is unseen in all 4 dictionaries, so `kroch-chhmar` splits it into KCCs: `ថេ | ប្ល៊ិ | ថ`.
- **Ambiguous Personal Name (`OPEN-TEST #51`)**: `ខ្ញុំ | ឈ្មោះ | កូ | រេនា` (*"My name is Ko Rena"*) $\to$ `ខ្ញុំ | ឈ្មោះ | កូរេ | នា` because `កូរេ` (*"Korea"*) is a high-frequency dictionary word.

#### Category D: Punctuation & Formatted Numbers
- **20×**: Ellipsis `...` $\to$ `. | . | .`
- **Formatted decimals/thousands (`OPEN-TEST #82`)**: `២.៦២៥` $\to$ `២ | . | ៦២៥` (because `.` is defined as `SYMBOL` with `invoke=1` in `char_khmer.def`).

---

### 5.2 Concrete Mis-Segmented Sentence Examples from Held-Out `OPEN-TEST`

#### Example 1: Overlapping Compound Boundary (`វិធានការ` vs. `ការកាត់បន្ថយ`) — `OPEN-TEST #123`
- **Gold (Joined)**:
  `វិធានការ | កាត់បន្ថយ | ការបញ្ចេញ | ឧស្ម័ន | ផ្ទះកញ្ចក់ | នៅ | ទីក្រុង | តូក្យូ`
- **`kroch-chhmar` (Mode C)** *(shifted `ការ` from `វិធានការ` to `ការកាត់បន្ថយ`)*:
  `វិធាន | ការកាត់បន្ថយ | ការបញ្ចេញ | ឧស្ម័ន | ផ្ទះកញ្ចក់ | នៅ | ទីក្រុង | តូក្យូ`

#### Example 2: Typo in Input (`ដុលារ` instead of `ដុល្លារ`) — `OPEN-TEST #109`
- **Gold (Joined)**:
  `ខ្ញុំ | ទិញ | ក្នុង | តម្លៃ | ១ | ដុលារ | ។`
- **`kroch-chhmar` (Mode C)** *(OOV typo `ដុលារ` fell back to 3 KCCs)*:
  `ខ្ញុំ | ទិញ | ក្នុង | តម្លៃ | ១ | ដុ | លា | រ | ។`

#### Example 3: Unseen Foreign Transliteration (`ថេប្ល៊ិថ` = "tablet") — `OPEN-TEST #76`
- **Gold (Joined)**:
  `ថេប្ល៊ិថ | មិន | ត្រឹមតែ | ជា | ឧបករណ៍ | សម្រាប់ | ទស្សនា | មាតិកា | ឌីជីថល`
- **`kroch-chhmar` (Mode C)** *(split unseen `ថេប្ល៊ិថ` into 3 KCCs)*:
  `ថេ | ប្ល៊ិ | ថ | មិន | ត្រឹមតែ | ជា | ឧបករណ៍ | សម្រាប់ | ទស្សនា | មាតិកា | ឌីជីថល`

#### Example 4: Ambiguous Personal Name (`កូ រេនា` vs. `កូរេ`) — `OPEN-TEST #51`
- **Gold (Joined)**:
  `ខ្ញុំ | ឈ្មោះ | កូ | រេនា | ។`
- **`kroch-chhmar` (Mode C)** *(matched `កូរេ` "Korea" + `នា`)*:
  `ខ្ញុំ | ឈ្មោះ | កូរេ | នា | ។`

#### Example 5: Formatted Number with Period (`២.៦២៥`) — `OPEN-TEST #82`
- **Gold (Joined)**:
  `គ្រោះ | រញ្ជួយ | ផែនដី | បាន | សម្លាប់ | មនុស្ស | ចំនួន | ២.៦២៥ | នាក់`
- **`kroch-chhmar` (Mode C)** *(merged `គ្រោះរញ្ជួយផែនដី` and split `២ | . | ៦២៥`)*:
  `គ្រោះរញ្ជួយផែនដី | បាន | សម្លាប់ | មនុស្ស | ចំនួន | ២ | . | ៦២៥ | នាក់`

---

### 5.3 Top Mis-Segmentations in Mode A (vs. Split Gold)

In **Mode A** (`68.78%` exact sentence match, `92.06%` F1 vs. `khPOS` Split Gold), discrepancies happen in two directions inside `CompoundSplitter.resolve()`:

1. **2-KCC Compounds Kept Whole by `kroch-chhmar` Mode A** (because `CompoundSplitter.resolve()` intentionally skips DP-splitting words with `<= 2` KCCs unless they have an explicit override, to prevent over-splitting base words into single KCCs):
   - `នៅ | ក្នុង` $\to$ `នៅក្នុង` (**436×**)
   - `ពួក | គេ` $\to$ `ពួកគេ` (**221×**)
   - `ភ្នំ | ពេញ` $\to$ `ភ្នំពេញ` (**174×**)
   - `ជា | ច្រើន` $\to$ `ជាច្រើន` (**168×**)
   - `ប្រជា | ជន` $\to$ `ប្រជាជន` (**145×**)
   - `កីឡា | ករ` $\to$ `កីឡាករ` (**143×**)
   - `ព្រះ | អង្គ` $\to$ `ព្រះអង្គ` (**120×**)

2. **Multi-KCC Words Not Marked with `_` in `khPOS` That Mode A Splits**:
   - `ទាំងអស់` $\to$ `ទាំង | អស់` (**179×**)
   - `នគរបាល` $\to$ `នគរ | បាល` (**112×**)
   - `លីកាដូ` (LICADHO acronym) $\to$ `លី | កាដូ` (**103×**, because `លី` and `កាដូ` *"gift"* are in `base_vocab`)
   - `របាយការណ៍` $\to$ `របាយ | ការណ៍` (**73×**)
   - `ត្រូវតែ` $\to$ `ត្រូវ | តែ` (**67×**)
   - `រូបភាព` $\to$ `រូប | ភាព` (**58×**)
