# Packaging & Publishing Guide: `kroch-chhmar`

This guide explains how `kroch-chhmar` is packaged as an **Option B Pre-Compiled Dictionary Wheel** with zero-configuration out-of-the-box usage, while maintaining full flexibility for users to supply their **own custom-compiled dictionaries**.

---

## Architecture: Pre-Compiled Bundle with Custom Overrides

### 1. Zero-Config Out-of-the-Box Setup
The published wheel (`.whl`) bundles:
- `kroch_chhmar/resources/sudachi.json` (runtime configuration)
- `kroch_chhmar/resources/char.def` (KCC lattice character rules)
- `kroch_chhmar/resources/unk.def` (OOV transition definitions)
- `kroch_chhmar/resources/rewrite.def` (orthographic rewrite rules)
- `kroch_chhmar/resources/compounds.yaml` (curated compound splits)
- `kroch_chhmar/resources/system_khmer.dic` (35,883-entry permissive binary system dictionary built from MIT-licensed seed + `khmerlbdict`; **~2.8 MB** uncompressed)
- `kroch_chhmar/resources/seed/` (packaged seed lexicon/corpus so `kroch-chhmar prepare --profile permissive --seed --no-raw` works from a wheel)

The bundled dictionary is deliberately the **permissive** profile. The full
research dictionary (87,119 entries, built from khPOS/ALT/RAC) is not shipped
because those upstream datasets are non-commercial. Users who accept those
terms can build it locally with `--profile full`.

When any user runs:
```bash
pip install kroch-chhmar
```
they can immediately tokenize Khmer text with zero extra setup:
```python
from kroch_chhmar import KhmerTokenizer

# Works immediately out of the box using the bundled pre-compiled dictionary
tokenizer = KhmerTokenizer()
print(tokenizer.segment("ព្រះរាជាណាចក្រកម្ពុជាមានសន្តិភាព"))
```

---

## 2. Using Your Own Compiled Dictionary

Users are never locked into the bundled dictionary. `kroch-chhmar` provides 4 flexible ways to supply a custom dictionary:

### Method 1: Python API Parameter
Pass `system_dic_path` to use a custom system dictionary, and/or `user_dic_paths` to add domain-specific user dictionaries:

```python
from kroch_chhmar import KhmerTokenizer

# Option A: Override the system dictionary
tokenizer = KhmerTokenizer(system_dic_path="/path/to/my_custom_system.dic")

# Option B: Keep default system dictionary + add a custom user dictionary
tokenizer = KhmerTokenizer(user_dic_paths=["/path/to/my_user_dict.dic"])
```

### Method 2: Command-Line Interface (`--system-dic` & `--user-dic`)
```bash
# Tokenize with custom system dictionary
kroch-chhmar tokenize "អត្ថបទ..." --system-dic /path/to/my_custom_system.dic

# Tokenize with bundled system dictionary + custom user dictionary
kroch-chhmar tokenize "អត្ថបទ..." --user-dic /path/to/my_user_dict.dic
```

### Method 3: Environment Variable (`KROCH_CHHMAR_SYSTEM_DIC`)
Set the environment variable in your `.env` or shell profile:
```bash
export KROCH_CHHMAR_SYSTEM_DIC="/path/to/my_custom_system.dic"
```
`KhmerTokenizer` will automatically pick it up without requiring any code changes.

### Method 4: Compile a Fresh Dictionary Yourself
Users can rebuild both system and user dictionaries using the built-in CLI:

```bash
# 1. Compile custom user dictionary from a simple CSV/TSV/TXT wordlist (< 0.1s):
kroch-chhmar ubuild my_medical_terms.tsv -o my_medical_terms.dic

# 2. Re-compile a custom system dictionary from your own lex.csv & matrix.def:
kroch-chhmar build --build-dir ./my_build_dir -o ./my_build_dir/system_khmer.dic
```

---

## 3. Dictionary Resolution Priority Order

When `KhmerTokenizer()` initializes, it resolves the config and dictionary in this order:

$$\begin{aligned}
\text{Config: 1. Explicit } \texttt{config\_path} &\longrightarrow \text{2. } \texttt{KROCH\_CHHMAR\_CONFIG} \longrightarrow \text{3. Repo } \texttt{configs/} \longrightarrow \text{4. Bundled } \texttt{resources/} \\
\text{Dictionary: 1. Explicit } \texttt{system\_dic\_path} &\longrightarrow \text{2. } \texttt{KROCH\_CHHMAR\_SYSTEM\_DIC} \longrightarrow \text{3. Config target} \longrightarrow \text{4. Local repo build} \longrightarrow \text{5. Bundled resource}
\end{aligned}$$

In a source checkout, the repository config and `data/build/system_khmer.dic` are preferred so the Python API and CLI agree after `kroch-chhmar build`. Installed wheels fall back to the bundled resources.

---

## 4. How to Build & Publish to PyPI

### Step 1: Rebuild the Permissive Dictionary into `resources/`
The bundled dictionary must be the redistributable permissive build, not the
full research build. From a checkout with `data/raw/khmerlbdict_*.tsv` present:

```bash
./scripts/rebuild_bundled.sh
```

This runs `prepare --profile permissive --seed --raw`, compiles the `.dic`, and
copies it to `src/kroch_chhmar/resources/system_khmer.dic`. Do **not** copy
`data/build/system_khmer.dic` (the full research build) into the wheel.

### Step 2: Build the Wheel & Source Distribution
```bash
rm -rf dist/
uv build
```

This creates:
- `dist/kroch_chhmar-0.2.0-py3-none-any.whl` (~1.2 MB; contains the 2.9 MB permissive `system_khmer.dic`)
- `dist/kroch_chhmar-0.2.0.tar.gz` (~2.6 MB; includes `configs/`, `data/seed/`, and `example/app.py`)

### Step 3: Test Install in an Isolated Environment
Verify the built wheel runs cleanly without needing local repo files:

```bash
uv run --isolated --with dist/kroch_chhmar-0.2.0-py3-none-any.whl python -c "
from kroch_chhmar import KhmerTokenizer, SplitMode
tok = KhmerTokenizer()
print('Tokens:', tok.segment('ព្រះរាជាណាចក្រកម្ពុជា', mode=SplitMode.C))
"
```

Then verify that the wheel can rebuild a dictionary from its **packaged** seed data (this was previously broken — the command produced an 18-entry placeholder dictionary):

```bash
BUILD_DIR=$(mktemp -d)
uv run --isolated --with dist/kroch_chhmar-0.2.0-py3-none-any.whl kroch-chhmar prepare \
  --profile permissive --seed --no-raw --no-auto-holdout \
  --build-dir "$BUILD_DIR" --interim-dir "$BUILD_DIR/interim"
uv run --isolated --with dist/kroch_chhmar-0.2.0-py3-none-any.whl kroch-chhmar build \
  --no-prepare --build-dir "$BUILD_DIR"
```

The `prepare` step should report ~213 entries (seed lexicon + POS table), not 18.

> **Release checklist:** run `pytest` (including the `regex`-based KCC reference tests) and `ruff check`; run `./scripts/rebuild_bundled.sh` so `src/kroch_chhmar/resources/system_khmer.dic` is the current permissive build; confirm the sdist contains `configs/`, `data/seed/`, and `example/app.py`; and verify the isolated wheel commands above. `prepare` is deterministic (`lex.csv`/`matrix.def` byte-identical across hash seeds), so build diffs are meaningful.

### Step 4: Publish to TestPyPI
```bash
uv publish \
  --publish-url https://test.pypi.org/legacy/ \
  --token pypi-<YOUR_TESTPYPI_TOKEN>
```

### Step 5: Publish to Official PyPI
```bash
uv publish --token pypi-<YOUR_PYPI_TOKEN>
```
