"""CLI workflow for Kroch-Chhmar: fetch, prepare, build, ubuild, tokenize, and evaluate."""

from __future__ import annotations

import json
import urllib.error
import zipfile
from dataclasses import asdict
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sudachipy.errors import SudachiError

from kroch_chhmar.audit import OverlapReport, audit_overlap, load_audit_items
from kroch_chhmar.dictionary.compound_splitter import CompoundSplitter
from kroch_chhmar.dictionary.cost_estimator import CostEstimator
from kroch_chhmar.dictionary.csv_writer import (
    build_system_dic,
    build_user_dic,
    generate_sudachi_entries,
    summarize_split_topology,
    validate_split_references,
    write_sudachi_csv,
)
from kroch_chhmar.evaluate import (
    evaluate_pos_tagging,
    evaluate_segmentation,
    load_gold_sentences,
)
from kroch_chhmar.ingest.local_importers import (
    HoldoutFilter,
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


def _is_source_checkout() -> bool:
    return (REPO_ROOT / "pyproject.toml").exists() and (REPO_ROOT / "src").is_dir()


def _data_dir(name: str) -> Path:
    """Resolve a data directory in a source checkout, or in the current workspace for a wheel install."""
    if _is_source_checkout() or (REPO_ROOT / "data").is_dir():
        return REPO_ROOT / "data" / name
    return Path.cwd() / "data" / name


def _package_resource(name: str) -> Path:
    return Path(__file__).resolve().parent / "resources" / name


def _default_compounds_path() -> Path:
    repo = REPO_ROOT / "configs" / "compounds.yaml"
    return repo if repo.exists() else _package_resource("compounds.yaml")


def _default_sudachi_config() -> Path:
    """
    Prefer the repository config (which points at data/build) only when a local
    build exists; otherwise use the bundled config so the CLI works out of the box.
    """
    repo = REPO_ROOT / "configs" / "sudachi.json"
    local_build = _data_dir("build") / "system_khmer.dic"
    if repo.exists() and local_build.exists():
        return repo
    return _package_resource("sudachi.json")


def _seed_corpus_path() -> Path:
    repo = _data_dir("seed") / "seed_corpus.txt"
    if repo.exists():
        return repo
    return _package_resource("seed") / "seed_corpus.txt"


@app.command("fetch")
def cmd_fetch(
    source: str = typer.Argument(
        "all",
        help=f"Dataset key ({', '.join(list(PUBLIC_SOURCES.keys()) + ['all'])}) or direct URL.",
    ),
    output_dir: Path = typer.Option(
        _data_dir("raw"),
        "--output-dir",
        "-o",
        help="Directory to store downloaded raw data.",
    ),
    timeout: int = typer.Option(
        30,
        "--timeout",
        help="Per-request network timeout in seconds.",
    ),
) -> None:
    """Download public Khmer datasets into `data/raw` (with SHA-256 verification)."""
    console.print(f"[bold cyan]Fetching dataset:[/bold cyan] {source} -> {output_dir}")
    try:
        targets = fetch_public_dataset(source, output_dir, timeout=timeout)
    except (ValueError, OSError, urllib.error.URLError, zipfile.BadZipFile) as exc:
        console.print(
            Panel(
                f"[bold red]Download or verification failed:[/bold red]\n{exc}",
                title="Dataset fetch failed",
                border_style="red",
            )
        )
        raise typer.Exit(code=1) from exc

    for target in targets:
        console.print(f"  [bold green]✓ Saved:[/bold green] {target}")

    expected_raw = _data_dir("raw").resolve()
    if output_dir.resolve() != expected_raw:
        console.print(
            f"[yellow]Note:[/yellow] `prepare` reads from [bold]{expected_raw}[/bold] by "
            f"default. Re-run with `--raw-dir {output_dir}` or fetch into the default directory."
        )


@app.command("prepare")
def cmd_prepare(
    lexicon: list[Path] | None = typer.Option(
        None,
        "--lexicon",
        "-l",
        help="Additional lexicon TSV/CSV/wordlist files to ingest.",
    ),
    corpus: list[Path] | None = typer.Option(
        None,
        "--corpus",
        "-c",
        help="Additional segmented text or .conllu corpus files to ingest.",
    ),
    include_seed: bool = typer.Option(
        True,
        "--seed/--no-seed",
        help="Include built-in seed lexicon and corpus (`data/seed/` or packaged resources).",
    ),
    include_raw: bool = typer.Option(
        True,
        "--raw/--no-raw",
        help="Automatically include downloaded files in the raw directory.",
    ),
    profile: str = typer.Option(
        "full",
        "--profile",
        help=(
            "Data profile: `full` ingests all fetched corpora (khPOS, ALT, RAC, khmerlbdict); "
            "`permissive` ingests only the packaged seed and `khmerlbdict_*.tsv` (MIT-compatible)."
        ),
    ),
    raw_dir: Path = typer.Option(
        _data_dir("raw"),
        "--raw-dir",
        help="Directory containing downloaded raw corpora and lexicons (used by `--raw`).",
    ),
    holdout: list[Path] | None = typer.Option(
        None,
        "--holdout",
        "-H",
        help="Extra held-out gold files whose exact sentences must be excluded from training.",
    ),
    auto_holdout: bool = typer.Option(
        True,
        "--auto-holdout/--no-auto-holdout",
        help="Automatically treat khPOS OPEN-TEST / CLOSE-TEST files in --raw-dir as held out.",
    ),
    compounds_yaml: Path = typer.Option(
        _default_compounds_path(),
        "--compounds",
        help="Path to curated compound & prefix rules YAML.",
    ),
    build_dir: Path = typer.Option(
        _data_dir("build"),
        "--build-dir",
        "-b",
        help="Output directory for `lex.csv` and `matrix.def`.",
    ),
    interim_dir: Path = typer.Option(
        _data_dir("interim"),
        "--interim-dir",
        help="Output directory for normalized & deduplicated interim data.",
    ),
) -> None:
    """
    Normalize inputs, estimate NLL costs, decompose compounds (A/B/C), and write `lex.csv` + `matrix.def`.

    Sentences that appear verbatim in a holdout file are removed before their
    tokens, compound overrides, or lexicon sub-parts are committed. This is
    required because upstream khPOS `train.all2` contains the entire CLOSE-TEST
    split.

    `--profile permissive` produces the redistributable default dictionary from
    MIT-licensed sources only: the packaged seed data plus `khmerlbdict_*.tsv`.
    `--profile full` is the research build and additionally ingests khPOS, ALT
    NOVA, and RAC 2022, whose upstream terms are non-commercial.
    """
    profile = profile.strip().lower()
    if profile not in {"full", "permissive"}:
        console.print(
            Panel(
                f"[bold red]Unknown profile:[/bold red] {profile}\n"
                "Choose `full` (all fetched corpora) or `permissive` "
                "(seed + khmerlbdict only).",
                title="Invalid data profile",
                border_style="red",
            )
        )
        raise typer.Exit(code=2)
    permissive = profile == "permissive"

    estimator = CostEstimator()
    splitter = CompoundSplitter(compounds_yaml_path=compounds_yaml)

    holdout_paths = [Path(p) for p in (holdout or [])]
    # The permissive profile never auto-ingests khPOS/ALT, so their test files
    # are not relevant. Explicit `--holdout` files are still honored.
    if auto_holdout and not permissive:
        for name in ("khpos_open_test.pos", "khpos_close_test.pos"):
            candidate = raw_dir / name
            if candidate.exists() and candidate.resolve() not in {
                p.resolve() for p in holdout_paths
            }:
                holdout_paths.append(candidate)

    holdout_filter = HoldoutFilter.from_files(holdout_paths) if holdout_paths else None
    holdout_resolved = {p.resolve() for p in holdout_paths}

    if include_seed:
        l_cnt, s_cnt = load_seed_dataset(
            estimator, splitter=splitter, holdout=holdout_filter
        )
        console.print(
            f"[green]Loaded seed dataset:[/green] {l_cnt} lexicon entries, {s_cnt} corpus sentences"
        )

    def _is_raw_holdout(path: Path) -> bool:
        return path.resolve() in holdout_resolved

    if include_raw and not raw_dir.exists():
        console.print(
            Panel(
                f"[yellow]Raw data directory not found:[/yellow]\n{raw_dir}\n\n"
                "Building from the packaged seed dataset and any `--lexicon` / `--corpus` "
                "inputs. Run `kroch-chhmar fetch all` for the full research dictionary, or "
                "`kroch-chhmar fetch khmerlbdict` for the permissive default sources.",
                title="Raw data not found",
                border_style="yellow",
            )
        )
        include_raw = False

    if include_raw and permissive:
        # Permissive profile: only MIT-licensed wordlists. Frequencies are kept
        # because there is no sentence corpus to provide calibrated NLL counts.
        permissive_files = sorted(raw_dir.glob("khmerlbdict_*.tsv"))
        if not permissive_files:
            console.print(
                Panel(
                    f"[yellow]No `khmerlbdict_*.tsv` files found in:[/yellow]\n{raw_dir}\n\n"
                    "The permissive profile will use the packaged seed data only. Run "
                    "`kroch-chhmar fetch khmerlbdict` to add the MIT-licensed wordlists.",
                    title="Permissive raw data not found",
                    border_style="yellow",
                )
            )
        for f in permissive_files:
            c = ingest_lexicon_tsv(f, estimator, use_frequencies=True)
            console.print(
                f"[green]Loaded permissive wordlist {f.name}:[/green] {c} entries"
            )
    elif include_raw:
        # 1. Ingest segmented corpora (.pos, .txt, .conllu) first so `splitter.overrides`
        #    captures canonical compound structures before NICT ALT NOVA and RAC 2022.
        for f in sorted(raw_dir.glob("*")):
            if f.name in (
                "data_km.km-tok.nova",
                "data_km.km-tag.nova",
                "README.txt",
                "khpos_open_test.pos",
                "khpos_close_test.pos",
            ) or _is_raw_holdout(f):
                continue
            if f.suffix.lower() == ".conllu":
                c = ingest_conllu_file(f, estimator, holdout=holdout_filter)
                console.print(f"[green]Loaded raw CoNLL-U {f.name}:[/green] {c} sentences")
            elif f.suffix.lower() in (".txt", ".pos"):
                c = ingest_segmented_text_file(
                    f, estimator, splitter=splitter, holdout=holdout_filter
                )
                console.print(f"[green]Loaded raw corpus {f.name}:[/green] {c} sentences")

        # 2. Ingest NICT ALT NOVA parallel files
        nova_tok = raw_dir / "data_km.km-tok.nova"
        nova_tag = raw_dir / "data_km.km-tag.nova"
        if nova_tok.exists() and nova_tag.exists():
            c = ingest_alt_nova_files(
                nova_tok, nova_tag, estimator, splitter=splitter, holdout=holdout_filter
            )
            console.print(
                f"[green]Loaded NICT ALT NOVA corpus ({nova_tok.name} + {nova_tag.name}):[/green] {c} sentences"
            )

        # 3. Ingest RAC 2022 Dictionary and raw wordlists (.tsv, .csv)
        for f in sorted(raw_dir.glob("*")):
            if f.name in (
                "data_km.km-tok.nova",
                "data_km.km-tag.nova",
                "README.txt",
                "khpos_open_test.pos",
                "khpos_close_test.pos",
            ) or _is_raw_holdout(f):
                continue
            if f.name == "RAC-Khmer-Dict-2022.csv":
                c = ingest_rac_dictionary_csv(f, estimator, splitter=splitter)
                console.print(f"[green]Loaded RAC 2022 Dictionary {f.name}:[/green] {c} entries")
            elif f.suffix.lower() in (".tsv", ".csv"):
                c = ingest_lexicon_tsv(f, estimator)
                console.print(f"[green]Loaded raw lexicon {f.name}:[/green] {c} entries")

    for lex_path in lexicon or []:
        c = ingest_lexicon_tsv(lex_path, estimator, use_frequencies=permissive)
        console.print(f"[green]Ingested lexicon {lex_path}:[/green] {c} entries")

    for corp_path in corpus or []:
        if corp_path.suffix.lower() == ".conllu":
            c = ingest_conllu_file(corp_path, estimator, holdout=holdout_filter)
        else:
            c = ingest_segmented_text_file(
                corp_path, estimator, splitter=splitter, holdout=holdout_filter
            )
        console.print(f"[green]Ingested corpus {corp_path}:[/green] {c} sentences")

    console.print(f"[bold cyan]Data profile:[/bold cyan] {profile}")
    if permissive:
        console.print(
            "[green]Permissive profile:[/green] auto-ingested sources are limited to the "
            "packaged seed and `khmerlbdict_*.tsv`; khPOS/ALT/RAC were not ingested."
        )
    if holdout_filter is not None:
        console.print(
            f"[green]Holdout integrity:[/green] {len(holdout_filter)} unique held-out sentences "
            f"loaded; skipped from training: [bold]{holdout_filter.skipped}[/bold]"
        )
    elif not permissive:
        console.print(
            "[yellow]Warning:[/yellow] no holdout files found; evaluation may be in-sample."
        )

    if estimator.stats.raw_lexicon_rows == 0 and estimator.stats.dedup_sentences == 0:
        if permissive:
            hint = (
                "The permissive profile needs at least one of: the packaged seed "
                "(`--seed`), `khmerlbdict_*.tsv` in `--raw-dir`, or an explicit "
                "`--lexicon` / `--corpus` file."
            )
        else:
            hint = (
                "Run `kroch-chhmar fetch all` first, or pass `--lexicon` / "
                "`--corpus` files."
            )
        console.print(
            Panel(
                "[bold red]No lexicon or corpus data was ingested.[/bold red]\n\n"
                "The build would contain only POS-table placeholders and cannot tokenize "
                f"Khmer text. {hint}",
                title="Refusing to build an empty dictionary",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

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

    # Populate CompoundSplitter with word costs (base_vocab is already populated from
    # verified root/headword sources during corpus & RAC dictionary ingestion).
    # `best_pos` is deterministic, so repeated builds are byte-identical.
    if not splitter.base_vocab:
        splitter.base_vocab.update(estimator.stats.lexicon_words.keys())
    splitter.word_costs.update(
        {
            surf: estimator.estimate_word_cost(surf, estimator.best_pos(surf))
            for surf in estimator.stats.lexicon_words
        }
    )
    # Re-load YAML overrides last so curated rules always take priority over corpus heuristics
    splitter.load_compounds_yaml(compounds_yaml)

    entries = generate_sudachi_entries(estimator, splitter, is_user_dict=False)
    validation_errors = validate_split_references(entries, is_user_dict=False)
    if validation_errors:
        console.print(
            Panel(
                "[bold red]Invalid split references in generated dictionary:[/bold red]\n"
                + "\n".join(f"  • {err}" for err in validation_errors[:10]),
                title="Dictionary validation failed",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    csv_path = write_sudachi_csv(entries, build_dir / "lex.csv")
    matrix_path = estimator.write_matrix_def(build_dir / "matrix.def")

    topology = summarize_split_topology(entries)
    console.print(
        f"[bold green]Prepared Sudachi artifacts:[/bold green]\n"
        f"  • Lexicon CSV : {csv_path} ({len(entries)} entries: "
        f"A={topology['A']}, B={topology['B']}, C={topology['C']})\n"
        f"  • Matrix File : {matrix_path}"
    )
    if topology["c_with_b"]:
        console.print(
            f"  • Mode C entries with a curated Mode B intermediate: "
            f"{topology['c_with_b']}/{topology['C']}"
        )
    if topology["c_without_b"]:
        console.print(
            f"  • [yellow]Mode C entries splitting directly to Mode A "
            f"(no curated B intermediate): {topology['c_without_b']}/{topology['C']}[/yellow]"
        )


@app.command("build")
def cmd_build(
    build_dir: Path = typer.Option(
        _data_dir("build"),
        "--build-dir",
        "-b",
        help="Directory containing `lex.csv` and `matrix.def`.",
    ),
    output_dic: Path | None = typer.Option(
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
    profile: str = typer.Option(
        "full",
        "--profile",
        help=(
            "Data profile used by the automatic `prepare` step: `full` (all fetched corpora) "
            "or `permissive` (packaged seed + `khmerlbdict_*.tsv` only)."
        ),
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
            profile=profile,
            raw_dir=_data_dir("raw"),
            holdout=None,
            auto_holdout=True,
            compounds_yaml=_default_compounds_path(),
            build_dir=build_dir,
            interim_dir=_data_dir("interim"),
        )

    console.print(f"[bold cyan]Compiling Sudachi system dictionary:[/bold cyan] {dic_path}")
    out = build_system_dic(csv_path, matrix_path, dic_path)
    console.print(f"[bold green]Successfully compiled system dictionary:[/bold green] {out}")


def _print_missing_system_dic_message(missing_path: Path | str) -> None:
    """Print a user-friendly Rich panel instructing the user how to build `system_khmer.dic`."""
    console.print(
        Panel(
            f"[bold red]System dictionary (`system_khmer.dic`) was not found at:[/bold red]\n"
            f"  [yellow]{missing_path}[/yellow]\n\n"
            f"[bold]You need to build `system_khmer.dic` before running this command:[/bold]\n"
            f"  1. [cyan]Quick permissive build (seed + MIT wordlists):[/cyan]\n"
            f"     [bold green]uv run kroch-chhmar build --profile permissive[/bold green]\n\n"
            f"  2. [cyan]Full research build (all public datasets):[/cyan]\n"
            f"     [bold green]uv run kroch-chhmar fetch all[/bold green]\n"
            f"     [bold green]uv run kroch-chhmar prepare --profile full[/bold green]\n"
            f"     [bold green]uv run kroch-chhmar build[/bold green]\n\n"
            f"  3. [cyan]Or pass a custom dictionary path:[/cyan]\n"
            f"     [bold green]--system-dic /path/to/system_khmer.dic[/bold green]",
            title="Missing System Dictionary (`system_khmer.dic`)",
            border_style="red",
        )
    )


@app.command("ubuild")
def cmd_ubuild(
    lexicon: Path = typer.Argument(..., help="User lexicon TSV/CSV/wordlist file."),
    system_dic: Path = typer.Option(
        _data_dir("build") / "system_khmer.dic",
        "--system-dic",
        "-s",
        help="Compiled system dictionary path. Must match the runtime system dictionary.",
    ),
    output_udic: Path = typer.Option(
        _data_dir("build") / "user_khmer.dic",
        "--output",
        "-o",
        help="Output user dictionary `.dic` path.",
    ),
    compounds_yaml: Path = typer.Option(
        _default_compounds_path(),
        "--compounds",
        help="Path to curated compound rules YAML.",
    ),
) -> None:
    """Compile a custom user lexicon into a binary `user_khmer.dic` via `sudachipy ubuild`."""
    if not system_dic.exists():
        _print_missing_system_dic_message(system_dic)
        raise typer.Exit(code=1)

    estimator = CostEstimator()
    cnt = ingest_lexicon_tsv(lexicon, estimator)
    if cnt == 0:
        console.print(
            Panel(
                f"[bold red]No entries were ingested from:[/bold red]\n{lexicon}",
                title="Empty user lexicon",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    splitter = CompoundSplitter(
        base_vocab=set(estimator.stats.lexicon_words.keys()),
        compounds_yaml_path=compounds_yaml,
    )
    entries = generate_sudachi_entries(estimator, splitter, is_user_dict=True)
    validation_errors = validate_split_references(entries, is_user_dict=True)
    if validation_errors:
        console.print(
            Panel(
                "[bold red]Invalid split references in generated user dictionary:[/bold red]\n"
                + "\n".join(f"  • {err}" for err in validation_errors[:10]),
                title="Dictionary validation failed",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

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
        _default_sudachi_config(),
        "--config",
        "-c",
        help="Path to `sudachi.json`.",
    ),
    system_dic: Path | None = typer.Option(
        None,
        "--system-dic",
        "-s",
        help="Optional override path for `system_khmer.dic`.",
    ),
    user_dic: list[Path] | None = typer.Option(
        None,
        "--user-dic",
        "-u",
        help="One or more compiled user dictionary `.dic` files (`user_khmer.dic`).",
    ),
) -> None:
    """Tokenize Khmer text using `KhmerTokenizer` and display morpheme details."""
    try:
        tokenizer = KhmerTokenizer(
            config_path=config,
            system_dic_path=system_dic,
            user_dic_paths=user_dic,
            default_mode=mode,
        )
    except (FileNotFoundError, SudachiError) as exc:
        console.print(
            Panel(str(exc), title="Tokenizer initialization failed", border_style="red")
        )
        raise typer.Exit(code=1) from exc

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
        _seed_corpus_path(),
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
        _default_sudachi_config(),
        "--config",
        "-c",
        help="Path to `sudachi.json`.",
    ),
    system_dic: Path | None = typer.Option(
        None,
        "--system-dic",
        "-s",
        help="Optional override path for `system_khmer.dic`.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit the evaluation report as JSON (for reproducible tables and CI).",
    ),
) -> None:
    """Evaluate segmentation Precision, Recall, F1, boundary F1, exact sentence match, OOV, and KCC rates."""
    try:
        tokenizer = KhmerTokenizer(
            config_path=config,
            system_dic_path=system_dic,
            default_mode=mode,
        )
    except (FileNotFoundError, SudachiError) as exc:
        console.print(
            Panel(str(exc), title="Tokenizer initialization failed", border_style="red")
        )
        raise typer.Exit(code=1) from exc

    gold_sentences = load_gold_sentences(corpus_file)
    report = evaluate_segmentation(
        tokenizer, gold_sentences, mode=mode, split_compounds=split_compounds
    )

    if as_json:
        console.print_json(json.dumps(asdict(report), ensure_ascii=False))
        return

    table = Table(title=f"Evaluation Report ({corpus_file.name} - Mode {mode.value})")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="cyan")

    table.add_row("Sentences", str(report.num_sentences))
    table.add_row("Scored Sentences", str(report.scored_sentences))
    table.add_row("Gold Tokens", str(report.gold_tokens))
    table.add_row("Predicted Tokens", str(report.pred_tokens))
    table.add_row("Precision", f"{report.precision:.2%}")
    table.add_row("Recall", f"{report.recall:.2%}")
    table.add_row("F1 Score", f"{report.f1:.2%}")
    table.add_row("Exact Sentence Match", f"{report.exact_matches} ({report.exact_match_rate:.2%})")
    table.add_row(
        "Boundary F1",
        f"{report.boundary_f1:.2%} (P={report.boundary_precision:.2%}, R={report.boundary_recall:.2%})",
    )
    table.add_row("OOV Rate", f"{report.oov_rate:.2%}")
    table.add_row(
        "Khmer-Word OOV Rate",
        f"{report.khmer_oov_rate:.2%} ({report.khmer_oov_tokens:,} tokens)",
    )
    table.add_row("KCC Boundary Violation Rate", f"{report.kcc_violation_rate:.2%}")

    console.print(table)


@app.command("evaluate-pos")
def cmd_evaluate_pos(
    corpus_file: Path = typer.Option(
        _seed_corpus_path(),
        "--corpus",
        "-f",
        help="POS-tagged evaluation corpus file (`word/POS` tokens).",
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
        _default_sudachi_config(),
        "--config",
        "-c",
        help="Path to `sudachi.json`.",
    ),
    system_dic: Path | None = typer.Option(
        None,
        "--system-dic",
        "-s",
        help="Optional override path for `system_khmer.dic`.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit the POS report as JSON.",
    ),
) -> None:
    """Evaluate UD POS accuracy on tokens whose spans match the gold segmentation."""
    try:
        tokenizer = KhmerTokenizer(
            config_path=config,
            system_dic_path=system_dic,
            default_mode=mode,
        )
    except (FileNotFoundError, SudachiError) as exc:
        console.print(
            Panel(str(exc), title="Tokenizer initialization failed", border_style="red")
        )
        raise typer.Exit(code=1) from exc

    report = evaluate_pos_tagging(
        tokenizer,
        load_gold_sentences(corpus_file),
        mode=mode,
        split_compounds=split_compounds,
    )

    if as_json:
        console.print_json(json.dumps(asdict(report), ensure_ascii=False))
        return

    summary = Table(title=f"POS Evaluation ({corpus_file.name} - Mode {mode.value})")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right", style="cyan")
    summary.add_row("Sentences", str(report.num_sentences))
    summary.add_row("Gold Tokens", str(report.gold_tokens))
    summary.add_row("Aligned Tokens", str(report.aligned_tokens))
    summary.add_row("Segmentation Coverage", f"{report.coverage:.2%}")
    summary.add_row("POS Accuracy (aligned)", f"{report.accuracy:.2%}")
    summary.add_row("Macro F1", f"{report.macro_f1:.2%}")
    console.print(summary)

    per_tag = Table(title="Per-Tag POS Metrics (aligned tokens)")
    per_tag.add_column("Tag", style="bold")
    per_tag.add_column("Support", justify="right")
    per_tag.add_column("Predicted", justify="right")
    per_tag.add_column("Precision", justify="right")
    per_tag.add_column("Recall", justify="right")
    per_tag.add_column("F1", justify="right")
    for tag, metrics in sorted(
        report.per_tag.items(), key=lambda item: (-item[1]["support"], item[0])
    ):
        if metrics["support"] == 0 and metrics["predicted"] == 0:
            continue
        per_tag.add_row(
            tag,
            str(int(metrics["support"])),
            str(int(metrics["predicted"])),
            f"{metrics['precision']:.2%}",
            f"{metrics['recall']:.2%}",
            f"{metrics['f1']:.2%}",
        )
    console.print(per_tag)


@app.command("audit-overlap")
def cmd_audit_overlap(
    left: Path = typer.Argument(
        ...,
        help="Left corpus/lexicon file (or the only file for an internal duplicate audit).",
    ),
    right: Path | None = typer.Argument(
        None,
        help="Right corpus/lexicon file. Omit to audit `left` against itself.",
    ),
    left_kind: str = typer.Option(
        "auto",
        "--left-kind",
        help="Input kind for `left`: auto, sentences, or lexicon.",
    ),
    right_kind: str = typer.Option(
        "auto",
        "--right-kind",
        help="Input kind for `right`: auto, sentences, or lexicon.",
    ),
    threshold: float = typer.Option(
        0.8,
        "--threshold",
        min=0.0,
        max=1.0,
        help="Minimum estimated Jaccard similarity for a near-duplicate pair.",
    ),
    near: bool = typer.Option(
        True,
        "--near/--no-near",
        help="Enable MinHash/LSH near-duplicate detection (exact matching always runs).",
    ),
    max_examples: int = typer.Option(
        10,
        "--max-examples",
        min=1,
        help="Maximum number of example pairs/groups to print.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the audit report as JSON."),
) -> None:
    """
    Audit exact and near-duplicate overlap between corpora or lexicons.

    Sentence inputs are normalized (ZWSP/pipe separators, POS suffixes, and
    compound markers removed) before comparison. `.tsv`/`.csv` inputs are
    treated as lexicons; RAC's `t_main`/`t_subword` columns are recognized.
    Near-duplicate candidates use a 64-value MinHash signature with 16x4
    banded LSH and are verified by signature Jaccard.
    """
    try:
        left_items = load_audit_items(left, left_kind)
        right_items = load_audit_items(right, right_kind) if right is not None else None
    except (FileNotFoundError, ValueError) as exc:
        console.print(Panel(str(exc), title="Overlap audit failed", border_style="red"))
        raise typer.Exit(code=1) from exc

    report: OverlapReport = audit_overlap(
        left_items,
        right_items,
        threshold=threshold,
        near=near,
        max_examples=max_examples,
    )

    if as_json:
        console.print_json(json.dumps(asdict(report), ensure_ascii=False))
        return

    summary = Table(title="Overlap Audit")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right", style="cyan")
    summary.add_row("Mode", report.mode)
    summary.add_row("Left items / unique", f"{report.left_items:,} / {report.left_unique:,}")
    summary.add_row("Right items / unique", f"{report.right_items:,} / {report.right_unique:,}")
    summary.add_row("Exact pairs", f"{report.exact_pairs:,}")
    summary.add_row(
        "Near-duplicate pairs",
        f"{report.near_pairs:,} (Jaccard >= {report.threshold:.2f})"
        if report.near_enabled
        else "disabled",
    )
    summary.add_row(
        "Left duplicate groups / items",
        f"{report.left_duplicate_groups:,} / {report.left_duplicate_items:,}",
    )
    summary.add_row(
        "Right duplicate groups / items",
        f"{report.right_duplicate_groups:,} / {report.right_duplicate_items:,}",
    )
    console.print(summary)

    if report.examples:
        examples = Table(title="Examples")
        examples.add_column("Kind", style="bold")
        examples.add_column("Jaccard", justify="right")
        examples.add_column("Left")
        examples.add_column("Right")
        for example in report.examples:
            examples.add_row(
                example.kind,
                f"{example.jaccard:.3f}",
                example.left,
                example.right,
            )
        console.print(examples)

    if report.exact_pairs or report.near_pairs:
        console.print(
            Panel(
                f"[yellow]Overlap detected:[/yellow] {report.exact_pairs:,} exact pair(s), "
                f"{report.near_pairs:,} near-duplicate pair(s). Review before using these "
                "collections as independent train/test splits.",
                title="Train/test contamination risk",
                border_style="yellow",
            )
        )


if __name__ == "__main__":
    app()
