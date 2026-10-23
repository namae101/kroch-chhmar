# Kroch-Chhmar (ក្រូចឆ្មារ) 🍋‍🟩

> *Named after **ក្រូចឆ្មារ** (`kroch chhmar`, Khmer lime) — a playful Khmer citrus nod to [**Sudachi**](https://github.com/WorksApplications/sudachi.rs) (酢橘, Japanese green citrus).*

A reproducible Python workflow for generating **Sudachi-compatible Khmer binary dictionaries** (`system_khmer.dic` and `user_khmer.dic`) and a **Khmer Character Cluster (KCC)-aware word segmentation plugin/wrapper** for [`SudachiPy`](https://github.com/WorksApplications/sudachi.rs).

Full architectural details are documented in [`plan/implementation_plan.md`](plan/implementation_plan.md).

---

## Features

- **Khmer Unicode Canonicalization (`betterkhmer`)**: Regex-free syllable sorting, compound vowel merging (`េ` + `ា` $\to$ `ោ`), consonant shifter resolution, and zero-width character (`\u200b`, `\u200c`, `\u200d`) handling.
- **Khmer Character Cluster (KCC) Boundary Enforcement**:
  - `configs/char.def` uses `NOOOVBOW` / `NOOOVEOW` flags on Khmer dependent vowels, signs, and `Coeng` (`\u17d2`) so OOV candidates do not start or end mid-cluster.
  - Post-tokenization KCC repair in `KhmerTokenizer` guarantees zero mid-KCC splits even on unseen OOV text.
- **Cross-Dataset Normalization & Deduplication**: Canonicalizes all words and corpora into a single Unicode encoding, merges untyped frequency wordlists into existing POS entries, prunes corpus POS outliers, and exports deduplicated datasets to `data/interim/`.
- **Corpus NLL Cost & Matrix Estimation**: Estimates smoothed negative log-likelihood word emission costs and 18×18 Universal Dependencies (UD) POS bigram transition matrices (`matrix.def`).
- **Multi-Granular Khmer Segmentation (Modes A / B / C)**:
  - **Mode A**: Base/root words (`ព្រះ | រាជាណាចក្រ | កម្ពុជា`)
  - **Mode B**: Derived words & intermediate compounds (`ព្រះរាជាណាចក្រ | កម្ពុជា`, `ការសិក្សា`)
  - **Mode C**: Long compounds & named entities (`ព្រះរាជាណាចក្រកម្ពុជា`, `ក្រសួងអប់រំយុវជននិងកីឡា`)

---

## Datasets & Upstream Sources

`kroch-chhmar` includes automated downloaders (`kroch-chhmar fetch`) and parsers for the following open Khmer datasets:

| Dataset Key | Source / Link | Description | Upstream License / Terms |
| :--- | :--- | :--- | :--- |
| `khpos` | [`ye-kyaw-thu/khPOS`](https://github.com/ye-kyaw-thu/khPOS) | 14,000 manually word-segmented & POS-tagged Khmer sentences (`train.all2`, `OPEN-TEST`, `CLOSE-TEST`) with `_` and `~` compound markers | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| `alt-km-nova` | [NICT/NIPTICT Asian Language Treebank (ALT) Khmer NOVA](https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/km-nova-181101.zip) ([Project Page](https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/index.html)) | 20,106 parallel tokenized (`data_km.km-tok.nova`) and bracketed compound POS-tagged (`data_km.km-tag.nova`) Khmer sentences | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| `khmerlbdict` | [`silnrsi/khmerlbdict`](https://github.com/silnrsi/khmerlbdict) | SIL International Khmer LineBreaking frequency wordlists (`seafreq`, `KHSV`, `KHOV`, `DFD`, `TD`, `HC`, `places`, `names`, `villages`) | [MIT License](https://github.com/silnrsi/khmerlbdict/blob/master/LICENSE) |
| `khmer-dictionary-44k` | [`seanghay/khmer-dictionary-44k`](https://huggingface.co/datasets/seanghay/khmer-dictionary-44k) | 41,275 entries (`RAC-Khmer-Dict-2022.csv`) from the Royal Academy of Cambodia 2022 Khmer Dictionary with headwords, subwords, POS, and pronunciation | Research / Non-Commercial Use Only |
| `khmer-words` | [`seanghay/khmer-words`](https://github.com/seanghay/khmer-words) | Open Khmer dictionary wordlist | Open Source |
| `khmer-nltk-dict` | [`VietHoang1512/khmer-nltk`](https://github.com/VietHoang1512/khmer-nltk) | Khmer-NLTK dictionary wordlist | MIT License |

Also incorporates the regex-free Khmer Unicode normalizer from:
- **BetterKhmer**: [`seanghay/betterkhmer`](https://github.com/seanghay/betterkhmer) (ported from SIL Global's `khmer-normalizer`, MIT License).

---

## Quickstart

### 1. Fetch Public Khmer Datasets & Build the System Dictionary

```bash
# Download all 4 primary datasets (khpos, alt-km-nova, khmerlbdict, khmer-dictionary-44k)
uv run kroch-chhmar fetch all

# Normalize with BetterKhmer, deduplicate into data/interim/, generate lex.csv + matrix.def, and compile system_khmer.dic
uv run kroch-chhmar prepare
uv run kroch-chhmar build
```

### 2. Segment Khmer Text (Modes C, B, A)

```bash
uv run kroch-chhmar tokenize "ព្រះរាជាណាចក្រកម្ពុជាមានការសិក្សានិងការអភិវឌ្ឍល្អ។" --mode C
uv run kroch-chhmar tokenize "ព្រះរាជាណាចក្រកម្ពុជាមានការសិក្សានិងការអភិវឌ្ឍល្អ។" --mode B
uv run kroch-chhmar tokenize "ព្រះរាជាណាចក្រកម្ពុជាមានការសិក្សានិងការអភិវឌ្ឍល្អ។" --mode A
```

### 3. Build & Use a Custom User Dictionary

See **[`user_dic.md`](user_dic.md)** for a complete guide on building custom user dictionaries, setting word priorities/costs, and configuring multi-granular splits (`Mode A / B / C`).

```bash
# Compile a custom TSV/CSV/TXT wordlist into user_khmer.dic in < 0.1s
uv run kroch-chhmar ubuild path/to/custom_words.tsv -o data/build/user_khmer.dic

# Tokenize using both system_khmer.dic and user_khmer.dic
uv run kroch-chhmar tokenize "អត្ថបទភាសាខ្មែរ" --user-dic data/build/user_khmer.dic --mode C
```

### 4. Evaluate Segmentation Benchmark

See **[`evaluation.md`](evaluation.md)** for full instructions on running the evaluation suite, detailed explanations of every metric, OOV analysis, and multi-granular discussions.

```bash
# Evaluate Mode C against compound-level gold segmentation
uv run kroch-chhmar evaluate -f data/raw/khpos_open_test.pos --mode C --join-compounds

# Evaluate Mode A against base-word (sub-compound) gold segmentation
uv run kroch-chhmar evaluate -f data/raw/khpos_open_test.pos --mode A --split-compounds
```

#### Summary Evaluation Results (`OPEN-TEST` & `CLOSE-TEST`, 1,000 Sentences Each)

| Dictionary Configuration | `OPEN-TEST` F1 *(Compounds Joined)* | `OPEN-TEST` F1 *(Base Words Split)* | `CLOSE-TEST` F1 *(Compounds Joined)* | `CLOSE-TEST` F1 *(Base Words Split)* | KCC Violations | Speed (1k sents) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`khPOS`-Aligned Dict (`Mode C` / `Mode A`)** | **98.35%** *(Mode C)* | **94.89%** *(Mode A)* | **98.24%** *(Mode C)* | **94.62%** *(Mode A)* | **0.00%** | **0.28s** |
| **Full 4-Dataset 100k Dict (Vocab Extended)** | **84.79%** *(Mode C)* | **86.26%** *(Mode A)* | **84.44%** *(Mode C)* | **86.77%** *(Mode A)* | **0.00%** | **0.28s** |
| **Full 4-Dataset 100k Dict (Held-Out Test)** | **84.03%** *(Mode C)* | **85.90%** *(Mode A)* | **84.44%** *(Mode C)* | **86.68%** *(Mode A)* | **0.00%** | **0.28s** |

### 5. Python API Usage

```python
from kroch_chhmar import KhmerTokenizer, SplitMode

tokenizer = KhmerTokenizer()

# Segment into surface strings
print(tokenizer.segment("ព្រះរាជាណាចក្រកម្ពុជាមានការសិក្សា។", mode=SplitMode.C))
# -> ['ព្រះរាជាណាចក្រកម្ពុជា', 'មាន', 'ការសិក្សា', '។']

print(tokenizer.segment("ព្រះរាជាណាចក្រកម្ពុជាមានការសិក្សា។", mode=SplitMode.A))
# -> ['ព្រះ', 'រាជាណាចក្រ', 'កម្ពុជា', 'មាន', 'ការ', 'សិក្សា', '។']
```

### 6. Run Tests

```bash
uv run --with pytest pytest -v
```

---

## License

The source code of `kroch-chhmar` is licensed under the **[MIT License](LICENSE)**.

> **Dataset Licensing Notice**: Compiled dictionary artifacts (`.dic`, `.csv`, `matrix.def`) built using third-party datasets (`khPOS`, NICT `ALT Khmer NOVA`, `RAC Khmer Dictionary 2022`) inherit the respective terms of those upstream datasets (such as **CC BY-NC-SA 4.0** / **Research & Non-Commercial Use**). For commercial deployments, build the dictionary using only permissively licensed datasets (e.g., `khmerlbdict` + your own proprietary or permissive corpora). See [`LICENSE`](LICENSE) for full details.

---

## AI Attribution & Acknowledgments

- **AI-Assisted Development**: The architecture, data ingestion parsers, statistical cost estimator, multi-granular compound splitter, and KCC boundary enforcement wrapper in this repository were collaboratively designed and pair-programmed with **Antigravity** (an agentic AI coding assistant developed by Google DeepMind).
- **Upstream Authors & Institutions**: Special thanks to **Ye Kyaw Thu** (`khPOS`), **Masao Utiyama, NICT & NIPTICT** (`Khmer ALT NOVA`), **SIL International / NRSI** (`khmerlbdict` & `khmer-normalizer`), **Seanghay Yath** (`betterkhmer` & `khmer-dictionary-44k`), the **Royal Academy of Cambodia**, and **Works Applications** (`Sudachi` / `sudachi.rs`).
