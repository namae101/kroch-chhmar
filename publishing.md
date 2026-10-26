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
- `kroch_chhmar/resources/system_khmer.dic` (87,237-word binary system dictionary, **~3.3 MB** compressed)

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

When `KhmerTokenizer()` initializes, it locates `system_khmer.dic` in the following strict priority:

$$\begin{aligned}
\text{1. Explicit Argument} &\longrightarrow \texttt{system\_dic\_path} \text{ passed to } \texttt{KhmerTokenizer()} \\
\text{2. Environment Variable} &\longrightarrow \texttt{os.environ["KROCH\_CHHMAR\_SYSTEM\_DIC"]} \\
\text{3. Config Target} &\longrightarrow \texttt{cfg\_data["systemDict"]} \text{ relative to } \texttt{sudachi.json} \\
\text{4. Bundled Resource} &\longrightarrow \texttt{kroch\_chhmar/resources/system\_khmer.dic} \text{ (inside wheel)} \\
\text{5. Local Repo Build} &\longrightarrow \texttt{data/build/system_khmer.dic} \text{ (local dev workspace)}
\end{aligned}$$

---

## 4. How to Build & Publish to PyPI

### Step 1: Sync the Latest Compiled Dictionary to `resources/`
Ensure `src/kroch_chhmar/resources/` has the latest compiled `system_khmer.dic`:

```bash
cp data/build/system_khmer.dic src/kroch_chhmar/resources/
```

### Step 2: Build the Wheel & Source Distribution
```bash
rm -rf dist/
uv build
```

This creates:
- `dist/kroch_chhmar-0.1.0-py3-none-any.whl` (~3.3 MB)
- `dist/kroch_chhmar-0.1.0.tar.gz` (~3.2 MB)

### Step 3: Test Install in an Isolated Environment
Verify the built wheel runs cleanly without needing local repo files:

```bash
uv run --isolated --with dist/kroch_chhmar-0.1.0-py3-none-any.whl python -c "
from kroch_chhmar import KhmerTokenizer, SplitMode
tok = KhmerTokenizer()
print('Tokens:', tok.segment('ព្រះរាជាណាចក្រកម្ពុជា', mode=SplitMode.C))
"
```

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
