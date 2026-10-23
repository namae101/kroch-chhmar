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

### 3. Build a Custom User Dictionary

```bash
uv run kroch-chhmar ubuild path/to/custom_words.tsv -o data/build/user_khmer.dic
```

### 4. Evaluate Segmentation Benchmark

```bash
uv run kroch-chhmar evaluate --mode C
```

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
