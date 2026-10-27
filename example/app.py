"""FastAPI microservice and interactive web playground for Kroch-Chhmar Khmer Tokenizer."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from kroch_chhmar.normalizer import count_kccs, iter_kccs, normalize_khmer
from kroch_chhmar.tokenizer import KhmerTokenizer, SplitMode

_tokenizer: KhmerTokenizer | None = None


def get_tokenizer() -> KhmerTokenizer:
    """Return the singleton `KhmerTokenizer` instance."""
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = KhmerTokenizer()
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=str(exc),
            ) from exc
    return _tokenizer


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Pre-load `system_khmer.dic` on server startup."""
    try:
        get_tokenizer()
    except HTTPException:
        # Allow server to start even if dictionary hasn't been compiled yet;
        # endpoints will return a helpful 503 message.
        pass
    yield


app = FastAPI(
    title="Kroch-Chhmar (ក្រូចឆ្មារ) Khmer NLP API",
    description=(
        "Multi-granular Khmer word segmentation (Modes A/B/C), Universal Dependencies POS tagging, "
        "RAC 2022 orthographic normalization, and Khmer Character Cluster (KCC) analysis powered by SudachiPy."
    ),
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request & Response Models
# ---------------------------------------------------------------------------


class TokenizeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Input Khmer text to segment and analyze.",
        examples=["ព្រះរាជាណាចក្រកម្ពុជាមានសន្តិភាពនិងសេចក្តីសុខនៅកណ្តាលទីក្រុង។"],
    )
    mode: Literal["A", "B", "C"] = Field(
        default="C",
        description="Split granularity: A (short/base words), B (middle/derived), C (long compounds/named entities).",
    )
    skip_whitespace: bool = Field(
        default=True,
        description="Whether to omit whitespace tokens from the output.",
    )


class MorphemeOut(BaseModel):
    surface: str = Field(..., description="Exact surface substring from the normalized input.")
    normalized_form: str = Field(..., description="Authoritative RAC 2022 orthographically normalized form.")
    reading_form: str = Field(..., description="Phonetic pronunciation or reading form.")
    ud_pos: str = Field(..., description="Primary Universal Dependencies POS tag.")
    pos: tuple[str, str, str, str, str, str] = Field(..., description="Full 6-tuple Sudachi POS hierarchy.")
    begin: int = Field(..., description="Start character offset.")
    end: int = Field(..., description="End character offset.")
    is_oov: bool = Field(..., description="True if token was produced by OOV fallback rather than the lexicon.")


class TokenizeResponse(BaseModel):
    mode: Literal["A", "B", "C"]
    normalized_input: str
    count: int
    segmented_surface: list[str]
    segmented_normalized: list[str]
    morphemes: list[MorphemeOut]


class SegmentRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Input Khmer text to segment.",
        examples=["ព្រះរាជាណាចក្រកម្ពុជាមានការសិក្សានិងការអភិវឌ្ឍល្អ។"],
    )
    mode: Literal["A", "B", "C"] = Field(default="C")
    use_normalized: bool = Field(
        default=False,
        description="If True, return canonical RAC 2022 spellings (`m.normalized_form`) instead of `m.surface`.",
    )
    delimiter: str = Field(
        default=" | ",
        description="Delimiter used to join the `segmented_text` string.",
    )


class SegmentResponse(BaseModel):
    mode: Literal["A", "B", "C"]
    tokens: list[str]
    normalized_tokens: list[str]
    segmented_text: str


class MultiModeResponse(BaseModel):
    text: str
    mode_a: list[str]
    mode_b: list[str]
    mode_c: list[str]
    normalized_mode_c: list[str]


class KCCSpanOut(BaseModel):
    text: str
    begin: int
    end: int
    is_khmer_cluster: bool


class NormalizeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Input Khmer text to canonicalize and segment into Khmer Character Clusters (KCCs).",
        examples=["កណ្តាលសន្ដិភាពក្តៅផ្តើមសេចក្តី"],
    )
    strip_zwsp: bool = Field(default=True)


class NormalizeResponse(BaseModel):
    original: str
    normalized: str
    kcc_count: int
    kccs: list[KCCSpanOut]


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["System"])
def health_check() -> dict[str, object]:
    """Check service readiness and dictionary status."""
    tok = get_tokenizer()
    return {
        "status": "ok",
        "engine": "kroch-chhmar",
        "default_mode": tok.default_mode.value,
        "enforce_kcc": tok.enforce_kcc,
        "merge_structured": tok.merge_structured,
        "modes": ["A", "B", "C"],
    }


@app.post("/tokenize", response_model=TokenizeResponse, tags=["Segmentation"])
def tokenize_text(req: TokenizeRequest) -> TokenizeResponse:
    """
    Tokenize Khmer text and return full morphological details (`surface`, `normalized_form`,
    `reading_form`, `ud_pos`, character offsets, and `is_oov`).
    """
    tok = get_tokenizer()
    norm_input = normalize_khmer(req.text, strip_zwsp=tok.strip_zwsp)
    morphemes = tok.tokenize(req.text, mode=SplitMode(req.mode), skip_whitespace=req.skip_whitespace)
    return TokenizeResponse(
        mode=req.mode,
        normalized_input=norm_input,
        count=len(morphemes),
        segmented_surface=[m.surface for m in morphemes],
        segmented_normalized=[m.normalized_form for m in morphemes],
        morphemes=[
            MorphemeOut(
                surface=m.surface,
                normalized_form=m.normalized_form,
                reading_form=m.reading_form,
                ud_pos=m.ud_pos,
                pos=m.pos,
                begin=m.begin,
                end=m.end,
                is_oov=m.is_oov,
            )
            for m in morphemes
        ],
    )


@app.post("/segment", response_model=SegmentResponse, tags=["Segmentation"])
def segment_text(req: SegmentRequest) -> SegmentResponse:
    """Lightweight word segmentation endpoint returning surface and RAC-normalized token lists."""
    tok = get_tokenizer()
    morphemes = tok.tokenize(req.text, mode=SplitMode(req.mode), skip_whitespace=True)
    surfaces = [m.surface for m in morphemes]
    norms = [m.normalized_form for m in morphemes]
    chosen = norms if req.use_normalized else surfaces
    return SegmentResponse(
        mode=req.mode,
        tokens=surfaces,
        normalized_tokens=norms,
        segmented_text=req.delimiter.join(chosen),
    )


@app.get("/compare-modes", response_model=MultiModeResponse, tags=["Segmentation"])
def compare_modes(
    text: str = Query(
        "ព្រះរាជាណាចក្រកម្ពុជាមានការសិក្សានិងការអភិវឌ្ឍល្អ។",
        min_length=1,
        description="Khmer sentence to compare across Modes A, B, and C.",
    ),
) -> MultiModeResponse:
    """Compare Mode A (short), Mode B (middle), and Mode C (long) segmentation side by side."""
    tok = get_tokenizer()
    m_c = tok.tokenize(text, mode=SplitMode.C)
    return MultiModeResponse(
        text=text,
        mode_a=tok.segment(text, mode=SplitMode.A),
        mode_b=tok.segment(text, mode=SplitMode.B),
        mode_c=[m.surface for m in m_c],
        normalized_mode_c=[m.normalized_form for m in m_c],
    )


@app.post("/normalize", response_model=NormalizeResponse, tags=["Normalization & KCC"])
def normalize_and_kcc(req: NormalizeRequest) -> NormalizeResponse:
    """Canonicalize Khmer Unicode text (`BetterKhmer` + phonotactic rules) and extract KCC clusters."""
    norm = normalize_khmer(req.text, strip_zwsp=req.strip_zwsp)
    spans = list(iter_kccs(norm))
    return NormalizeResponse(
        original=req.text,
        normalized=norm,
        kcc_count=count_kccs(norm),
        kccs=[
            KCCSpanOut(
                text=s.text,
                begin=s.begin,
                end=s.end,
                is_khmer_cluster=s.is_khmer_cluster,
            )
            for s in spans
        ],
    )


# ---------------------------------------------------------------------------
# Interactive Browser Playground (GET /)
# ---------------------------------------------------------------------------

_PLAYGROUND_HTML = """<!DOCTYPE html>
<html lang="km">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Kroch-Chhmar (ក្រូចឆ្មារ) FastAPI Playground</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --accent: #22c55e;
      --text: #f8fafc;
      --muted: #94a3b8;
    }
    body {
      font-family: 'Noto Sans Khmer', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 2rem;
      line-height: 1.6;
    }
    .container { max-width: 960px; margin: 0 auto; }
    h1 { margin-top: 0; color: var(--accent); }
    .subtitle { color: var(--muted); margin-bottom: 1.5rem; }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem;
      margin-bottom: 1.25rem;
    }
    textarea {
      width: 100%;
      box-sizing: border-box;
      background: #0b1120;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.75rem;
      font-size: 1.1rem;
      font-family: inherit;
    }
    .controls {
      display: flex;
      gap: 0.75rem;
      align-items: center;
      margin-top: 0.75rem;
      flex-wrap: wrap;
    }
    select, button {
      background: #0b1120;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.55rem 1rem;
      font-size: 0.95rem;
      cursor: pointer;
    }
    button.primary {
      background: var(--accent);
      color: #052e16;
      font-weight: 700;
      border: none;
    }
    .chips { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.5rem; }
    .chip {
      background: #0b1120;
      border: 1px solid var(--border);
      padding: 0.25rem 0.65rem;
      border-radius: 6px;
      font-size: 1rem;
    }
    .chip.fixed { border-color: var(--accent); color: #86efac; }
    table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; font-size: 0.95rem; }
    th, td { text-align: left; padding: 0.5rem 0.65rem; border-bottom: 1px solid var(--border); }
    th { color: var(--muted); font-weight: 600; }
    a { color: #38bdf8; text-decoration: none; }
  </style>
</head>
<body>
  <div class="container">
    <h1>🍋‍🟩 Kroch-Chhmar (ក្រូចឆ្មារ) Playground</h1>
    <p class="subtitle">
      Sudachi-powered Khmer Word Segmentation, UD POS Tagging &amp; RAC 2022 Orthographic Normalization
      &bull; <a href="/docs">OpenAPI Docs (/docs)</a>
    </p>

    <div class="card">
      <textarea id="inputText" rows="3">ព្រះរាជាណាចក្រកម្ពុជាមានសន្ដិភាពនិងសេចក្តីសុខនៅកណ្តាលទីក្រុងនាថ្ងៃក្តៅ។</textarea>
      <div class="controls">
        <label>Mode:
          <select id="modeSelect">
            <option value="C" selected>Mode C (Long / Named Entities)</option>
            <option value="B">Mode B (Middle / Derived Compounds)</option>
            <option value="A">Mode A (Short / Base Words)</option>
          </select>
        </label>
        <button class="primary" onclick="runTokenize()">Tokenize</button>
      </div>
    </div>

    <div class="card">
      <strong>Segmented Surface (<code>m.surface</code>):</strong>
      <div id="surfaceChips" class="chips"></div>
      <div style="margin-top: 1rem;">
        <strong>RAC 2022 Normalized Spelling (<code>m.normalized_form</code>):</strong>
        <div id="normChips" class="chips"></div>
      </div>
    </div>

    <div class="card">
      <strong>Morpheme Analysis:</strong>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Surface</th>
            <th>RAC Normalized</th>
            <th>Reading</th>
            <th>UD POS</th>
            <th>Span</th>
            <th>OOV</th>
          </tr>
        </thead>
        <tbody id="morphRows"></tbody>
      </table>
    </div>
  </div>

  <script>
    function addCell(row, value, color) {
      const cell = document.createElement('td');
      cell.textContent = String(value);
      if (color) cell.style.color = color;
      row.appendChild(cell);
    }

    async function runTokenize() {
      const text = document.getElementById('inputText').value;
      const mode = document.getElementById('modeSelect').value;
      const sChips = document.getElementById('surfaceChips');
      const nChips = document.getElementById('normChips');
      const tbody = document.getElementById('morphRows');
      sChips.replaceChildren();
      nChips.replaceChildren();
      tbody.replaceChildren();

      try {
        const res = await fetch('/tokenize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, mode })
        });
        if (!res.ok) {
          const detail = await res.text();
          sChips.textContent = 'Error: ' + detail;
          return;
        }
        const data = await res.json();

        data.morphemes.forEach((m, idx) => {
          const sc = document.createElement('span');
          sc.className = 'chip';
          sc.textContent = m.surface;
          sChips.appendChild(sc);

          const nc = document.createElement('span');
          nc.className = 'chip' + (m.normalized_form !== m.surface ? ' fixed' : '');
          nc.textContent = m.normalized_form;
          nChips.appendChild(nc);

          const tr = document.createElement('tr');
          addCell(tr, idx + 1);
          addCell(tr, m.surface);
          addCell(tr, m.normalized_form, m.normalized_form !== m.surface ? '#4ade80' : null);
          addCell(tr, m.reading_form);
          const posCell = document.createElement('td');
          const code = document.createElement('code');
          code.textContent = m.ud_pos;
          posCell.appendChild(code);
          tr.appendChild(posCell);
          addCell(tr, m.begin + ':' + m.end);
          addCell(tr, m.is_oov ? 'Yes' : 'No');
          tbody.appendChild(tr);
        });
      } catch (err) {
        sChips.textContent = 'Request failed: ' + err;
      }
    }
    runTokenize();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def playground() -> HTMLResponse:
    """Render the interactive browser playground."""
    return HTMLResponse(content=_PLAYGROUND_HTML)
