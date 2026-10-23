# Building & Using Custom User Dictionaries (`user_dic.md`)

One of the biggest advantages of **`kroch-chhmar`** over static neural/CRF segmenters is that you can **instantly extend its vocabulary** with your own custom words, domain terminology, brand names, slang, or proper nouns by compiling a **User Dictionary (`user_khmer.dic`)** in `< 0.1 seconds` — without re-downloading training corpora or rebuilding `system_khmer.dic`.

---

## 1. System Dictionary vs. User Dictionary

| Dictionary | File | Role | How It Is Built |
| :--- | :--- | :--- | :--- |
| **System Dictionary** | `data/build/system_khmer.dic` | Base ~100,000 Khmer vocabulary + `matrix.def` grammar transition rules | Built via `kroch-chhmar build` (or downloaded pre-built) |
| **User Dictionary** | `user_khmer.dic` *(up to 14 files)* | Custom overlay vocabulary searched **alongside** `system_khmer.dic` | Built in `< 0.1s` via `kroch-chhmar ubuild` |

---

## 2. Step-by-Step: Build a User Dictionary from a Simple Wordlist (TSV / CSV / TXT)

The easiest way to build a `user_khmer.dic` is with `kroch-chhmar ubuild`, which automatically normalizes your Khmer text with `betterkhmer`, maps POS tags to Universal Dependencies, estimates word costs, and compiles the binary `.dic`.

### Step 1: Create Your Custom Wordlist File (`custom_words.tsv`)

You can write a plain 1-word-per-line `.txt` file, or a tab/comma-separated `.tsv` / `.csv` file with up to 4 columns (`surface`, `pos`, `freq`, `reading`):

```tsv
# Format: surface<TAB>pos<TAB>freq<TAB>reading
ក្រូចឆ្មារអេអាយ	PROPN	500	ក្រូច-ឆ្មារ-អេ-អាយ
ថេប្ល៊ិថ	NOUN	300	ថេ-ប្ល៊ិថ
ខាល័រប៊ែន	NOUN	300	ខា-ល័រ-ប៊ែន
គ្លីនិកសុខភាពថ្មី	PROPN	400	*
```

- **`surface`** *(Required)*: The Khmer word or phrase you want the tokenizer to recognize.
- **`pos`** *(Optional, defaults to `NOUN`)*: Any Universal Dependencies POS tag (`NOUN`, `PROPN`, `VERB`, `ADJ`, `ADV`, `PRON`, `DET`, `ADP`, `NUM`, `CCONJ`, `SCONJ`, `PART`, `AUX`, `INTJ`, `PUNCT`, `SYM`, `X`) or Khmer POS abbreviation (`ន.`, `កិ.`, `គុ.`).
- **`freq`** *(Optional, defaults to `0`)*: Higher frequency numbers (e.g., `300`–`1000`) give the word a **lower lattice cost** so it takes priority over splitting into smaller fragments.
- **`reading`** *(Optional, defaults to `*`)*: Pronunciation or phonetic reading.

### Step 2: Compile `user_khmer.dic` Using `kroch-chhmar ubuild`

```bash
uv run kroch-chhmar ubuild custom_words.tsv \
  --system-dic data/build/system_khmer.dic \
  --output data/build/user_khmer.dic
```

*(This also generates `data/build/user_khmer.csv` in Sudachi's 19-column format so you can inspect the exact costs and IDs.)*

### Step 3: Tokenize with Your User Dictionary

#### Via CLI (`--user-dic` / `-u`):
```bash
uv run kroch-chhmar tokenize "ខ្ញុំប្រើប្រាស់ក្រូចឆ្មារអេអាយនៅលើកុំព្យូទ័រថេប្ល៊ិថ។" \
  --user-dic data/build/user_khmer.dic \
  --mode C
```

#### Via Python API (`user_dic_paths`):
```python
from kroch_chhmar import KhmerTokenizer, SplitMode

tokenizer = KhmerTokenizer(
    system_dic_path="data/build/system_khmer.dic",
    user_dic_paths=["data/build/user_khmer.dic"],
)

tokens = tokenizer.tokenize(
    "ខ្ញុំប្រើប្រាស់ក្រូចឆ្មារអេអាយនៅលើកុំព្យូទ័រថេប្ល៊ិថ។",
    mode=SplitMode.C,
)
for m in tokens:
    print(f"{m.surface:20s} | POS={m.ud_pos:6s} | Reading={m.reading_form}")
```

---

## 3. Advanced: Defining Multi-Granular Splits (`Mode A / B / C`) in a User Dictionary

If you want your custom compound word to stay **joined in `Mode C`** (`គ្លីនិកសុខភាពថ្មី`), split into **medium units in `Mode B`**, and split into **base words in `Mode A`** (`គ្លីនិក | សុខភាព | ថ្មី`):

### Option A: Include the Base Words in Your `custom_words.tsv` or `configs/compounds.yaml`
When you run `kroch-chhmar ubuild`, `CompoundSplitter` automatically checks `configs/compounds.yaml` and your wordlist. You can add an explicit entry to `configs/compounds.yaml`:

```yaml
overrides:
  "គ្លីនិកសុខភាពថ្មី":
    pos: "PROPN"
    mode: "C"
    split_a: ["គ្លីនិក", "សុខភាព", "ថ្មី"]
    split_b: ["គ្លីនិក", "សុខភាពថ្មី"]
```

Then run:
```bash
uv run kroch-chhmar ubuild custom_words.tsv \
  --compounds configs/compounds.yaml \
  -o data/build/user_khmer.dic
```

### Option B: Author Raw 19-Column Sudachi CSV Directly
If you want full low-level control over `left_id`, `right_id`, `cost`, and `U0/U1` user dictionary split references (`split_info_a`, `split_info_b`), you can write a 19-column CSV file (`raw_user.csv`) and compile it via `build_user_dic`:

```csv
គ្លីនិក,0,0,4000,គ្លីនិក,NOUN,*,*,*,*,*,គ្លីនិក,គ្លីនិក,*,A,*,*,*,*
សុខភាព,0,0,4000,សុខភាព,NOUN,*,*,*,*,*,សុខភាព,សុខភាព,*,A,*,*,*,*
ថ្មី,3,3,4000,ថ្មី,ADJ,*,*,*,*,*,ថ្មី,ថ្មី,*,A,*,*,*,*
គ្លីនិកសុខភាពថ្មី,1,1,3200,គ្លីនិកសុខភាពថ្មី,PROPN,*,*,*,*,*,គ្លីនិកសុខភាពថ្មី,គ្លីនិកសុខភាពថ្មី,*,C,U0/U1/U2,*,*,*
```

> **Note**: In Sudachi user dictionaries, line references in `split_info_a` (Column 15) and `split_info_b` (Column 16) that point to entries inside the **same user dictionary** must be prefixed with `U` (`U0`, `U1`, `U2`), which `kroch-chhmar ubuild` handles automatically.

Compile a raw 19-column CSV directly in Python:
```python
from kroch_chhmar.dictionary import build_user_dic

build_user_dic(
    csv_paths="raw_user.csv",
    system_dic_path="data/build/system_khmer.dic",
    output_udic_path="data/build/user_khmer.dic",
)
```
