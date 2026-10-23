# Sudachi-Khmer

A reproducible Python workflow for generating **Sudachi-compatible Khmer binary dictionaries** (`system_khmer.dic` and `user_khmer.dic`) and a **Khmer Character Cluster (KCC)-aware word segmentation plugin/wrapper** for `SudachiPy`.

Full architectural details are documented in [`plan/implementation_plan.md`](plan/implementation_plan.md).

---

## Features

- **Khmer Unicode Canonicalization (`betterkhmer`)**: Regex-free syllable sorting, compound vowel merging (`េ` + `ា` $\to$ `ោ`), consonant shifter resolution, and zero-width character (`\u200b`, `\u200c`, `\u200d`) handling.
- **Khmer Character Cluster (KCC) Boundary Enforcement**:
  - `configs/char.def` uses `NOOOVBOW` / `NOOOVEOW` flags on Khmer dependent vowels, signs, and `Coeng` (`\u17d2`) so OOV candidates do not start or end mid-cluster.
  - Post-tokenization KCC repair in `KhmerTokenizer` guarantees zero mid-KCC splits even on unseen OOV text.
- **Corpus NLL Cost & Matrix Estimation**: Estimates smoothed negative log-likelihood word emission costs and 18x18 Universal Dependencies (UD) POS bigram transition matrices (`matrix.def`).
- **Multi-Granular Khmer Segmentation (Modes A / B / C)**:
  - **Mode A**: Base/root words (`ព្រះ | រាជាណាចក្រ | កម្ពុជា`)
  - **Mode B**: Derived words & intermediate compounds (`ព្រះរាជាណាចក្រ | កម្ពុជា`, `ការសិក្សា`)
  - **Mode C**: Long compounds & named entities (`ព្រះរាជាណាចក្រកម្ពុជា`, `ក្រសួងអប់រំយុវជននិងកីឡា`)

---

## Quickstart

### 1. Fetch Public Khmer Datasets & Build the System Dictionary

```bash
# Download all 4 open datasets (khPOS, alt-km-nova, khmerlbdict, khmer-dictionary-44k)
uv run sudachi-khmer fetch all

# Normalize with BetterKhmer, deduplicate into data/interim/, generate lex.csv + matrix.def, and compile system_khmer.dic
uv run sudachi-khmer prepare
uv run sudachi-khmer build
```

Supported dataset keys for `sudachi-khmer fetch`:
- `all`: Fetches all four main datasets below
- `khpos`: [`ye-kyaw-thu/khPOS`](https://github.com/ye-kyaw-thu/khPOS) (12k POS-tagged & compound-marked sentences)
- `alt-km-nova`: [NICT/NIPTICT Asian Language Treebank (ALT) Khmer NOVA](https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/km-nova-181101.zip) (20k parallel tokenized + bracket-POS sentences)
- `khmerlbdict`: [`silnrsi/khmerlbdict`](https://github.com/silnrsi/khmerlbdict) (SEALang, Bible, village/place/name frequency wordlists)
- `khmer-dictionary-44k`: [`seanghay/khmer-dictionary-44k`](https://huggingface.co/datasets/seanghay/khmer-dictionary-44k) (Royal Academy of Cambodia 2022 44k dictionary with POS and pronunciation)

### 2. Segment Khmer Text (Modes C, B, A)

```bash
uv run sudachi-khmer tokenize "ព្រះរាជាណាចក្រកម្ពុជាមានការសិក្សានិងការអភិវឌ្ឍល្អ។" --mode C
uv run sudachi-khmer tokenize "ព្រះរាជាណាចក្រកម្ពុជាមានការសិក្សានិងការអភិវឌ្ឍល្អ។" --mode B
uv run sudachi-khmer tokenize "ព្រះរាជាណាចក្រកម្ពុជាមានការសិក្សានិងការអភិវឌ្ឍល្អ។" --mode A
```

### 3. Build a Custom User Dictionary

```bash
uv run sudachi-khmer ubuild path/to/custom_words.tsv -o data/build/user_khmer.dic
```

### 4. Evaluate Segmentation Benchmark

```bash
uv run sudachi-khmer evaluate --mode C
```

### 5. Python API Usage

```python
from sudachi_khmer import KhmerTokenizer, SplitMode

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
