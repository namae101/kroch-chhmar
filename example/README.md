# Kroch-Chhmar (ក្រូចឆ្មារ) FastAPI Example

This folder contains a ready-to-run **FastAPI microservice and interactive web playground** for `kroch-chhmar` Khmer word segmentation, Universal Dependencies POS tagging, RAC 2022 orthographic normalization (`្ដ` / `្ត`), and Khmer Character Cluster (KCC) analysis.

---

## 1. Run the FastAPI Server

The example uses the local `data/build/system_khmer.dic` when it exists (after `uv run kroch-chhmar build`) and otherwise falls back to the bundled dictionary. Launch it with `uv`:

```bash
uv run --with fastapi --with uvicorn uvicorn example.app:app --reload --port 8000
```

* **Interactive Web Playground**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* **Swagger OpenAPI Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 2. API Endpoints

### `POST /tokenize` — Full Morphological Analysis & RAC 2022 Normalization

```bash
curl -X POST http://127.0.0.1:8000/tokenize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "ព្រះរាជាណាចក្រកម្ពុជាមានសន្ដិភាពនិងសេចក្តីសុខនៅកណ្តាលទីក្រុងនាថ្ងៃក្តៅ។",
    "mode": "C"
  }'
```

**Response**:
```json
{
  "mode": "C",
  "normalized_input": "ព្រះរាជាណាចក្រកម្ពុជាមានសន្តិភាពនិងសេចក្តីសុខនៅកណ្ដាលទីក្រុងនាថ្ងៃក្តៅ។",
  "count": 11,
  "segmented_surface": [
    "ព្រះរាជាណាចក្រកម្ពុជា", "មាន", "សន្តិភាព", "និង", "សេចក្តីសុខ",
    "នៅ", "កណ្ដាល", "ទីក្រុង", "នា", "ថ្ងៃក្តៅ", "។"
  ],
  "segmented_normalized": [
    "ព្រះរាជាណាចក្រកម្ពុជា", "មាន", "សន្តិភាព", "និង", "សេចក្ដីសុខ",
    "នៅ", "កណ្ដាល", "ទីក្រុង", "នា", "ថ្ងៃក្ដៅ", "។"
  ],
  "morphemes": [
    {
      "surface": "សេចក្តីសុខ",
      "normalized_form": "សេចក្ដីសុខ",
      "reading_form": "សេចក្ដីសុខ",
      "ud_pos": "NOUN",
      "pos": ["NOUN", "*", "*", "*", "*", "*"],
      "begin": 34,
      "end": 44,
      "is_oov": false
    }
  ]
}
```

---

### `POST /segment` — Lightweight Segmentation (`surface` or `normalized_form`)

```bash
curl -X POST http://127.0.0.1:8000/segment \
  -H "Content-Type: application/json" \
  -d '{
    "text": "កណ្តាលសន្ដិភាពក្តៅផ្តើមសេចក្តី",
    "mode": "C",
    "use_normalized": true,
    "delimiter": " | "
  }'
```

---

### `GET /compare-modes` — Side-by-Side Mode A / B / C Granularity Comparison

```bash
curl -G http://127.0.0.1:8000/compare-modes \
  --data-urlencode "text=ព្រះរាជាណាចក្រកម្ពុជាមានការសិក្សានិងការអភិវឌ្ឍល្អ។"
```

---

### `POST /normalize` — BetterKhmer Canonicalization & KCC Cluster Breakdown

```bash
curl -X POST http://127.0.0.1:8000/normalize \
  -H "Content-Type: application/json" \
  -d '{"text": "កណ្តាលភាសាខ្មែរ"}'
```
