"""CLI workflow for Kroch-Chhmar: fetch, prepare, build, ubuild, tokenize, and evaluate."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from kroch_chhmar.dictionary.compound_splitter import CompoundSplitter
from kroch_chhmar.dictionary.cost_estimator import CostEstimator
from kroch_chhmar.dictionary.csv_writer import (
    build_system_dic,
    build_user_dic,
    generate_sudachi_entries,
    write_sudachi_csv,
)
from kroch_chhmar.evaluate import evaluate_segmentation, load_gold_sentences
from kroch_chhmar.ingest.local_importers import (
    ingest_alt_nova_files,
    ingest_conllu_file,
    ingest_lexicon_tsv,
    ingest_rac_dictionary_csv,
    ingest_segmented_text_file,
    load_seed_dataset,
)
from kroch_chhmar.ingest.public_datasets import PUBLIC_SOURCES, fetch_public_dataset
from kroch_chhmar.tokenizer import KhmerTokenizer, SplitMode

app = typer.Typer(
    name="kroch-chhmar",
    help="Workflow for generating Sudachi-compatible Khmer dictionaries and KCC-aware word segmentation.",
    add_completion=False,
)
console = Console()

REPO_ROOT = Path(__file__).resolve().parents[2]


@app.command("fetch")
def cmd_fetch(
    source: str = typer.Argument(
        "all",
        help=f"Dataset key ({', '.join(list(PUBLIC_SOURCES.keys()) + ['all'])}) or direct URL.",
    ),
    output_dir: Path = typer.Option(
        REPO_ROOT / "data" / "raw",
        "--output-dir",
        "-o",
        help="Directory to store downloaded raw data.",
    ),
) -> None:
    """Download public Khmer wordlists or corpora (`khpos`, `alt-km-nova`, `khmerlbdict`, `all`) into `data/raw`."""
    console.print(f"[bold cyan]Fetching dataset:[/bold cyan] {source} -> {output_dir}")
    targets = fetch_public_dataset(source, output_dir)
    for t in targets:
        console.print(f"  [bold green]✓ Saved:[/bold green] {t}")


@app.command("prepare")
def cmd_prepare(
    lexicon: Optional[ list[Path] ] = typer.Option(
        None,
        "--lexicon",
        "-l",
        help="Additional lexicon TSV/CSV/wordlist files to ingest.",
    ),
    corpus: Optional[ list[Path] ] = typer.Option(
        None,
        "--corpus",
        "-c",
        help="Additional segmented text or .conllu corpus files to ingest.",
    ),
    include_seed: bool = typer.Option(
        True,
        "--seed/--no-seed",
        help="Include built-in seed lexicon and corpus (`data/seed/`).",
    ),
    include_raw: bool = typer.Option(
        True,
        "--raw/--no-raw",
        help="Automatically include any downloaded files in `data/raw/`.",
    ),
    compounds_yaml: Path = typer.Option(
        REPO_ROOT / "configs" / "compounds.yaml",
        "--compounds",
        help="Path to curated compound & prefix rules YAML.",
    ),
    build_dir: Path = typer.Option(
        REPO_ROOT / "data" / "build",
        "--build-dir",
        "-b",
        help="Output directory for `lex.csv` and `matrix.def`.",
    ),
    interim_dir: Path = typer.Option(
        REPO_ROOT / "data" / "interim",
        "--interim-dir",
        help="Output directory for normalized & deduplicated interim data.",
    ),
) -> None:
    """Normalize inputs, estimate NLL costs, decompose compounds (A/B/C), and write `lex.csv` + `matrix.def`."""
    estimator = CostEstimator()
    splitter = CompoundSplitter(compounds_yaml_path=compounds_yaml)

    if include_seed:
        l_cnt, s_cnt = load_seed_dataset(estimator, splitter=splitter)
        console.print(
            f"[green]Loaded seed dataset:[/green] {l_cnt} lexicon entries, {s_cnt} corpus sentences"
        )

    if include_raw:
        raw_dir = REPO_ROOT / "data" / "raw"
        if raw_dir.exists():
            # Check for NICT ALT NOVA parallel files first
            nova_tok = raw_dir / "data_km.km-tok.nova"
            nova_tag = raw_dir / "data_km.km-tag.nova"
            if nova_tok.exists() and nova_tag.exists():
                c = ingest_alt_nova_files(nova_tok, nova_tag, estimator, splitter=splitter)
                console.print(
                    f"[green]Loaded NICT ALT NOVA corpus ({nova_tok.name} + {nova_tag.name}):[/green] {c} sentences"
                )

            for f in sorted(raw_dir.glob("*")):
                if f.name in (
                    "data_km.km-tok.nova",
                    "data_km.km-tag.nova",
                    "README.txt",
                ):
                    continue
                if f.name == "RAC-Khmer-Dict-2022.csv":
                    c = ingest_rac_dictionary_csv(f, estimator, splitter=splitter)
                    console.print(f"[green]Loaded RAC 2022 Dictionary {f.name}:[/green] {c} entries")
                elif f.suffix.lower() in (".tsv", ".csv"):
                    c = ingest_lexicon_tsv(f, estimator)
                    console.print(f"[green]Loaded raw lexicon {f.name}:[/green] {c} entries")
                elif f.suffix.lower() == ".conllu":
                    c = ingest_conllu_file(f, estimator)
                    console.print(f"[green]Loaded raw CoNLL-U {f.name}:[/green] {c} sentences")
                elif f.suffix.lower() in (".txt", ".pos"):
                    c = ingest_segmented_text_file(f, estimator, splitter=splitter)
                    console.print(f"[green]Loaded raw corpus {f.name}:[/green] {c} sentences")

    for lex_path in lexicon or []:
        c = ingest_lexicon_tsv(lex_path, estimator)
        console.print(f"[green]Ingested lexicon {lex_path}:[/green] {c} entries")

    for corp_path in corpus or []:
        if corp_path.suffix.lower() == ".conllu":
            c = ingest_conllu_file(corp_path, estimator)
        else:
            c = ingest_segmented_text_file(corp_path, estimator, splitter=splitter)
        console.print(f"[green]Ingested corpus {corp_path}:[/green] {c} sentences")

    # Finalize normalization & deduplication and export clean interim files
    uniq_surfaces, uniq_pairs = estimator.finalize_deduplication()
    interim_lex, interim_corp = estimator.export_interim_datasets(interim_dir)
    console.print(
        f"[bold cyan]Normalized & Deduplicated Data:[/bold cyan]\n"
        f"  • Unique Khmer words      : {uniq_surfaces:,} (from {estimator.stats.raw_lexicon_rows:,} raw lexicon rows)\n"
        f"  • Unique (word, POS) pairs: {uniq_pairs:,}\n"
        f"  • Unique corpus sentences : {estimator.stats.dedup_sentences:,} (from {estimator.stats.raw_sentences:,} raw sentences)\n"
        f"  • Exported interim files  : {interim_lex.name}, {interim_corp.name}"
    )

    # Populate CompoundSplitter with all discovered base words and word costs
    splitter.base_vocab.update(estimator.stats.lexicon_words.keys())
    splitter.word_costs.update(
        {
            surf: estimator.estimate_word_cost(surf, next(iter(tags)))
            for surf, tags in estimator.stats.lexicon_words.items()
        }
    )
    # Re-load YAML overrides last so curated rules always take priority over corpus heuristics
    splitter.load_compounds_yaml(compounds_yaml)

    entries = generate_sudachi_entries(estimator, splitter, is_user_dict=False)
    csv_path = write_sudachi_csv(entries, build_dir / "lex.csv")
    matrix_path = estimator.write_matrix_def(build_dir / "matrix.def")

    mode_counts = {"A": 0, "B": 0, "C": 0}
    for e in entries:
        mode_counts[e.split_type] = mode_counts.get(e.split_type, 0) + 1

    console.print(
        f"[bold green]Prepared Sudachi artifacts:[/bold green]\n"
        f"  • Lexicon CSV : {csv_path} ({len(entries)} entries: "
        f"A={mode_counts['A']}, B={mode_counts['B']}, C={mode_counts['C']})\n"
        f"  • Matrix File : {matrix_path}"
    )


@app.command("build")
def cmd_build(
    build_dir: Path = typer.Option(
        REPO_ROOT / "data" / "build",
        "--build-dir",
        "-b",
        help="Directory containing `lex.csv` and `matrix.def`.",
    ),
    output_dic: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output binary dictionary path (defaults to `<build_dir>/system_khmer.dic`).",
    ),
    prepare_first: bool = typer.Option(
        True,
        "--prepare/--no-prepare",
        help="Automatically run `prepare` if `lex.csv` or `matrix.def` is missing.",
    ),
) -> None:
    """Compile `lex.csv` and `matrix.def` into a binary `system_khmer.dic` via `sudachipy build`."""
    csv_path = build_dir / "lex.csv"
    matrix_path = build_dir / "matrix.def"
    dic_path = output_dic or (build_dir / "system_khmer.dic")

    if (not csv_path.exists() or not matrix_path.exists()) and prepare_first:
        cmd_prepare(
            lexicon=None,
            corpus=None,
            include_seed=True,
            include_raw=True,
            compounds_yaml=REPO_ROOT / "configs" / "compounds.yaml",
            build_dir=build_dir,
            interim_dir=REPO_ROOT / "data" / "interim",
        )

    console.print(f"[bold cyan]Compiling Sudachi system dictionary:[/bold cyan] {dic_path}")
    out = build_system_dic(csv_path, matrix_path, dic_path)
    console.print(f"[bold green]Successfully compiled system dictionary:[/bold green] {out}")


@app.command("ubuild")
def cmd_ubuild(
    lexicon: Path = typer.Argument(..., help="User lexicon TSV/CSV/wordlist file."),
    system_dic: Path = typer.Option(
        REPO_ROOT / "data" / "build" / "system_khmer.dic",
        "--system-dic",
        "-s",
        help="Compiled system dictionary path.",
    ),
    output_udic: Path = typer.Option(
        REPO_ROOT / "data" / "build" / "user_khmer.dic",
        "--output",
        "-o",
        help="Output user dictionary `.dic` path.",
    ),
    compounds_yaml: Path = typer.Option(
        REPO_ROOT / "configs" / "compounds.yaml",
        "--compounds",
        help="Path to curated compound rules YAML.",
    ),
) -> None:
    """Compile a custom user lexicon into a binary `user_khmer.dic` via `sudachipy ubuild`."""
    estimator = CostEstimator()
    cnt = ingest_lexicon_tsv(lexicon, estimator)
    splitter = CompoundSplitter(
        base_vocab=set(estimator.stats.lexicon_words.keys()),
        compounds_yaml_path=compounds_yaml,
    )
    entries = generate_sudachi_entries(estimator, splitter, is_user_dict=True)
    user_csv = output_udic.with_suffix(".csv")
    write_sudachi_csv(entries, user_csv)

    out = build_user_dic(user_csv, system_dic, output_udic)
    console.print(
        f"[bold green]Compiled user dictionary ({cnt} input rows -> {len(entries)} entries):[/bold green] {out}"
    )


@app.command("tokenize")
def cmd_tokenize(
    text: str = typer.Argument(..., help="Khmer text to segment."),
    mode: SplitMode = typer.Option(
        SplitMode.C,
        "--mode",
        "-m",
        help="Split granularity mode: A (short/base), B (middle), C (long).",
    ),
    config: Path = typer.Option(
        REPO_ROOT / "configs" / "sudachi.json",
        "--config",
        "-c",
        help="Path to `sudachi.json`.",
    ),
    system_dic: Optional[Path] = typer.Option(
        None,
        "--system-dic",
        "-s",
        help="Optional override path for `system_khmer.dic`.",
    ),
) -> None:
    """Tokenize Khmer text using `KhmerTokenizer` and display morpheme details."""
    tokenizer = KhmerTokenizer(
        config_path=config,
        system_dic_path=system_dic,
        default_mode=mode,
    )
    morphemes = tokenizer.tokenize(text, mode=mode)

    table = Table(title=f"Kroch-Chhmar Segmentation (Mode {mode.value})")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Surface", style="bold cyan")
    table.add_column("Normalized", style="green")
    table.add_column("UD POS", style="magenta")
    table.add_column("Span", justify="center")
    table.add_column("OOV", justify="center")

    for idx, m in enumerate(morphemes, 1):
        table.add_row(
            str(idx),
            m.surface,
            m.normalized_form,
            m.ud_pos,
            f"{m.begin}:{m.end}",
            "[red]Yes[/red]" if m.is_oov else "No",
        )

    console.print(table)
    console.print("[bold]Segmented:[/bold] " + " | ".join(m.surface for m in morphemes))


@app.command("evaluate")
def cmd_evaluate(
    corpus_file: Path = typer.Option(
        REPO_ROOT / "data" / "seed" / "seed_corpus.txt",
        "--corpus",
        "-f",
        help="Segmented evaluation corpus file.",
    ),
    mode: SplitMode = typer.Option(
        SplitMode.C,
        "--mode",
        "-m",
        help="Split granularity mode (A, B, or C).",
    ),
    split_compounds: bool = typer.Option(
        False,
        "--split-compounds/--join-compounds",
        help="Whether to split khPOS '_' and '~' compound annotations into base words.",
    ),
    config: Path = typer.Option(
        REPO_ROOT / "configs" / "sudachi.json",
        "--config",
        "-c",
        help="Path to `sudachi.json`.",
    ),
) -> None:
    """Evaluate segmentation Precision, Recall, F1, OOV rate, and KCC violation rate."""
    tokenizer = KhmerTokenizer(config_path=config, default_mode=mode)
    gold_sentences = load_gold_sentences(corpus_file)
    report = evaluate_segmentation(
        tokenizer, gold_sentences, mode=mode, split_compounds=split_compounds
    )

    table = Table(title=f"Evaluation Report ({corpus_file.name} - Mode {mode.value})")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="cyan")

    table.add_row("Sentences", str(report.num_sentences))
    table.add_row("Gold Tokens", str(report.gold_tokens))
    table.add_row("Predicted Tokens", str(report.pred_tokens))
    table.add_row("Precision", f"{report.precision:.2%}")
    table.add_row("Recall", f"{report.recall:.2%}")
    table.add_row("F1 Score", f"{report.f1:.2%}")
    table.add_row("OOV Rate", f"{report.oov_rate:.2%}")
    table.add_row("KCC Boundary Violation Rate", f"{report.kcc_violation_rate:.2%}")

    console.print(table)


if __name__ == "__main__":
    app()
