"""Command line (docs/SPEC.md 15.2): `guardrails --help`.

Exit codes: 0 = allowed / done, 1 = blocked (so `check` is usable in scripts), 2 = usage
error or a command that a later task implements.
"""

import asyncio
import contextlib
import json
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import TypeAdapter, ValidationError

from llm_guardrails import __version__
from llm_guardrails.types import ConfigError, SourceDoc

app = typer.Typer(
    help="Layered guardrails for LLM apps: check text, serve the API, run the evaluation.",
    no_args_is_help=True,
    add_completion=False,
)
corpus_app = typer.Typer(help="Attack corpus for injection layer 2.", no_args_is_help=True)
dataset_app = typer.Typer(help="Evaluation datasets.", no_args_is_help=True)
sweep_app = typer.Typer(help="Threshold sweeps on the dev split.", no_args_is_help=True)
baseline_app = typer.Typer(help="Comparison baselines.", no_args_is_help=True)
report_app = typer.Typer(help="Generated reports.", no_args_is_help=True)
for sub, name in [(corpus_app, "corpus"), (dataset_app, "dataset"), (sweep_app, "sweep"),
                  (baseline_app, "baseline"), (report_app, "report")]:  # fmt: skip
    app.add_typer(sub, name=name)

REPORTS = Path("reports")

Live = Annotated[bool, typer.Option("--live", help="Make real (paid) Bedrock calls.")]


class Suite(StrEnum):
    injection = "injection"
    pii = "pii"
    grounding = "grounding"
    schema = "schema"


class EvalSuite(StrEnum):
    injection = "injection"
    pii = "pii"
    schema = "schema"
    grounding = "grounding"
    e2e = "e2e"
    all = "all"


class Split(StrEnum):
    dev = "dev"
    test = "test"


SplitOption = Annotated[Split, typer.Option("--split", help="Dataset split.")]


def _fail(message: str, code: int = 2) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code)


def _later(task: str, what: str) -> NoReturn:
    typer.echo(f"`guardrails {what}` is implemented in Task {task}; not available yet.", err=True)
    raise typer.Exit(2)


def _settings():
    from llm_guardrails.settings import load_settings

    try:
        return load_settings()
    except ConfigError as exc:
        _fail(f"configuration: {exc}")


@app.callback()
def _main() -> None:
    # Hindi and other non-Latin text must never crash printing on a Windows console.
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


# ----------------------------------------------------------------------------- check


def _load_sources(path: Path | None) -> list[SourceDoc]:
    if path is None:
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return TypeAdapter(list[SourceDoc]).validate_python(raw)
    except (OSError, ValueError, ValidationError) as exc:
        _fail(f"--sources must be a JSON list of {{id, text}} objects ({type(exc).__name__})")


@app.command()
def check(
    text: Annotated[str, typer.Argument(help="The text to check.")],
    output: Annotated[bool, typer.Option("--output", help="Check it as a model OUTPUT.")] = False,
    sources: Annotated[
        Path | None,
        typer.Option(help="JSON file [{id, text}, ...]: context for input, sources for output."),
    ] = None,
    policy: Annotated[str, typer.Option(help="Bundled policy name.")] = "default",
    as_json: Annotated[bool, typer.Option("--json", help="Print the full JSON result.")] = False,
) -> None:
    """Run the input guards on TEXT (or the output guards, with --output)."""
    from llm_guardrails.runtime import Pipelines

    docs = _load_sources(sources)
    settings = _settings()
    try:
        pipes = Pipelines(settings)
        pipe = pipes.get(policy)
    except ConfigError as exc:
        _fail(str(exc))

    if output:
        result = asyncio.run(pipe.check_output(text, sources=docs))
        shown = result.output
    else:
        result = asyncio.run(pipe.check_input(text, docs))
        shown = result.sanitized_text

    if as_json:
        payload = {"request_id": result.request_id, "decision": result.decision,
                   "text": shown, "report": result.report.model_dump(mode="json")}  # fmt: skip
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        typer.echo(f"decision: {result.decision}")
        if shown is not None:
            typer.echo(f"text:     {shown}")
        for v in result.report.verdicts:
            reasons = ", ".join(v.reasons)
            typer.echo(f"  {v.guard:<10} {v.action.value:<7} {reasons}".rstrip())

    missing = getattr(pipes.llm, "missing", [])
    if missing:
        typer.echo(
            f"note: {len(missing)} model call(s) had no recording in replay mode, so the "
            "guard that needed them failed closed. Set GUARDRAILS_LLM_MODE=fake (no AWS) or "
            "bedrock (live) in .env.",
            err=True,
        )
    raise typer.Exit(0 if result.decision == "allow" else 1)


# ----------------------------------------------------------------------------- serve


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to bind; keep it local.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port.")] = 8000,
) -> None:
    """Run the HTTP API (see docs/SPEC.md 15.1)."""
    import uvicorn

    from llm_guardrails.api import create_app

    try:
        application = create_app(_settings())
    except ConfigError as exc:
        _fail(str(exc))
    uvicorn.run(application, host=host, port=port)


# ----------------------------------------------------------------------------- spend


@app.command()
def spend() -> None:
    """Show live Bedrock spend recorded in the ledger against the budget."""
    from llm_guardrails.spend import SpendLedger

    settings = _settings()
    ledger = SpendLedger(limit_usd=settings.live_budget_usd)
    total = ledger.total()
    calls = len(ledger.entries())
    share = 100 * total / settings.live_budget_usd if settings.live_budget_usd else 0.0
    typer.echo(f"live spend: USD {total:.4f} of USD {settings.live_budget_usd:.2f} "
               f"({share:.1f}%) across {calls} recorded call(s)")  # fmt: skip


# ----------------------------------------------------------------------------- corpus


@corpus_app.command("generate")
def corpus_generate(live: Live = False) -> None:
    """Reword the hand-written attacks and benign anchors (replays recordings by default)."""
    from llm_guardrails.eval import corpus

    corpus.main(["generate", *(["--live"] if live else [])])


@corpus_app.command("embed")
def corpus_embed(live: Live = False) -> None:
    """Embed the corpus with Titan (needs --live)."""
    from llm_guardrails.eval import corpus

    corpus.main(["embed", *(["--live"] if live else [])])


# ----------------------------------------------------------------------------- dataset

SuiteOrAll = Annotated[
    str, typer.Option("--suite", help="injection, pii, grounding, schema or all.")
]


def _suites(value: str) -> list[str]:
    from llm_guardrails.eval.datasets import SUITES

    if value == "all":
        return list(SUITES)
    if value not in SUITES:
        _fail(f"--suite must be one of {', '.join(SUITES)} or all")
    return [value]


def _generation_llm(suite: str, live: bool):
    from llm_guardrails.eval.generate import recordings_path
    from llm_guardrails.recording import RecordingLLM

    settings = _settings()
    if not live:
        return RecordingLLM(recordings_path(suite), mode="replay", model_id=settings.chat_model_id)
    from llm_guardrails.llm import BedrockLLM
    from llm_guardrails.spend import SpendLedger

    if settings.auth_method == "none":
        _fail("--live needs Bedrock credentials in .env")
    inner = BedrockLLM(settings, ledger=SpendLedger(limit_usd=settings.live_budget_usd))
    return RecordingLLM(recordings_path(suite), mode="record", inner=inner)


@dataset_app.command("generate")
def dataset_generate(
    suite: Annotated[Suite, typer.Option("--suite", help="Which dataset.")],
    live: Live = False,
) -> None:
    """Grow the seed rows into data/<suite>.jsonl (replays recordings unless --live)."""
    from llm_guardrails.eval.generate import batches_for, generate
    from llm_guardrails.types import ReplayMissing

    name = suite.value
    typer.echo(f"{name}: {len(batches_for(name))} model call(s) planned "
               f"({'live, paid' if live else 'replay, free'})")  # fmt: skip
    try:
        stats = asyncio.run(generate(name, _generation_llm(name, live)))
    except ReplayMissing as exc:
        _fail(f"{len(exc.keys)} call(s) have no recording: rerun with --live")
    typer.echo(json.dumps(dict(sorted(stats.items())), indent=2))


@dataset_app.command("embed")
def dataset_embed(live: Live = False) -> None:
    """Titan vectors for the injection rows (needed for the meaning-level leakage check)."""
    from llm_guardrails.eval.datasets import load_suite
    from llm_guardrails.eval.leakage import embed_texts

    if not live:
        _fail("embedding calls Bedrock: rerun with --live")
    from llm_guardrails.embeddings import TitanEmbedder

    rows = load_suite("injection", include_dropped=True)
    new = asyncio.run(embed_texts([r["text"] for r in rows], TitanEmbedder(_settings())))
    typer.echo(f"embedded {new} new text(s); {len(rows)} rows have vectors")


@dataset_app.command("download-external")
def dataset_download_external() -> None:
    """Download the deepset/prompt-injections TEST split (116 rows, Apache-2.0) into
    data/raw/ (git-ignored) for the external injection check."""
    from llm_guardrails.eval.run_injection import EXTERNAL_RAW, download_external

    n = download_external()
    typer.echo(f"saved {n} rows to {EXTERNAL_RAW}")


@dataset_app.command("split")
def dataset_split(suite: SuiteOrAll = "all") -> None:
    """Assign dev/test (50/50 per stratum, duplicate groups together, seeds in dev)."""
    from llm_guardrails.eval.datasets import split_suite

    for name in _suites(suite):
        typer.echo(f"{name}: {split_suite(name)}")


@dataset_app.command("validate")
def dataset_validate(suite: SuiteOrAll = "all") -> None:
    """Check every row (schema, ids, labels, checksums, example-only domains)."""
    from llm_guardrails.eval.datasets import validate_suite

    failed = False
    for name in _suites(suite):
        problems = validate_suite(name)
        typer.echo(f"{name}: {'OK' if not problems else f'{len(problems)} problem(s)'}")
        for p in problems[:30]:
            typer.echo(f"  {p}")
        failed |= bool(problems)
    raise typer.Exit(1 if failed else 0)


@dataset_app.command("leakcheck")
def dataset_leakcheck(suite: SuiteOrAll = "all") -> None:
    """Fail if a test row is too close to dev rows or to the layer-2 corpus."""
    from llm_guardrails.eval.leakage import check_suite

    failed = False
    for name in _suites(suite):
        findings = check_suite(name)
        typer.echo(f"{name}: {'OK' if not findings else f'{len(findings)} leak(s)'}")
        for f in findings[:30]:
            typer.echo(f"  {f}")
        failed |= bool(findings)
    raise typer.Exit(1 if failed else 0)


@dataset_app.command("review-sample")
def dataset_review_sample(
    suite: SuiteOrAll = "all",
    frac: Annotated[float, typer.Option(help="Share of rows to sample.")] = 0.10,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 7,
) -> None:
    """Write data/review/review_sample.csv for a human spot-check."""
    from llm_guardrails.eval.review import (
        REVIEW_DOC,
        SAMPLE_PATH,
        review_doc,
        sample_rows,
        write_sample,
    )

    rows = sample_rows(_suites(suite), frac, seed)
    write_sample(rows)
    REVIEW_DOC.write_text(review_doc("pending", len(rows)), encoding="utf-8", newline="\n")
    typer.echo(f"wrote {len(rows)} rows to {SAMPLE_PATH}")


@dataset_app.command("apply-review")
def dataset_apply_review(
    csv_path: Annotated[Path | None, typer.Option("--csv", help="Filled review CSV.")] = None,
) -> None:
    """Apply the verdicts from the review CSV and record the counts."""
    from llm_guardrails.eval.review import REVIEW_DOC, SAMPLE_PATH, apply_review, review_doc

    path = csv_path or SAMPLE_PATH
    if not path.is_file():
        _fail(f"no review file at {path}: run `guardrails dataset review-sample` first")
    outcome = apply_review(path)
    reviewed = sum(outcome.counts[k] for k in ("ok", "edited", "dropped"))
    status = "applied" if reviewed and not outcome.counts["blank"] else "partly applied"
    total = reviewed + outcome.counts["blank"]
    REVIEW_DOC.write_text(review_doc(status, total, outcome), encoding="utf-8", newline="\n")
    typer.echo(json.dumps(dict(outcome.counts)))
    for e in outcome.errors:
        typer.echo(f"  {e}", err=True)
    raise typer.Exit(1 if outcome.errors else 0)


# ----------------------------------------------------------------------------- sweep


def _sweep(kind: str, split: Split, task: str) -> None:
    if split is Split.test:
        # Tuning on the test split would make every reported number optimistic.
        _fail("sweeps run on the dev split only; the test split is for the final eval")
    _later(task, f"sweep {kind}")


def _run_eval(coro_factory) -> dict:
    """Run an eval coroutine, turning a missing recording into a clear message."""
    from llm_guardrails.types import BudgetExceeded, ReplayMissing

    try:
        return asyncio.run(coro_factory())
    except ReplayMissing as exc:
        _fail(f"{len(exc.keys)} model/embedding call(s) have no recording: rerun with --live")
    except BudgetExceeded as exc:
        _fail(str(exc))


@sweep_app.command("injection")
def sweep_injection(split: SplitOption = Split.dev, live: Live = False) -> None:
    """Choose injection thresholds on dev; writes reports/operating_point.json."""
    if split is Split.test:
        _fail("sweeps run on the dev split only; the test split is for the final eval")
    from llm_guardrails.eval import sweep

    settings = _settings()
    point = _run_eval(lambda: sweep.run(live=live, settings=settings, reports_dir=REPORTS))
    typer.echo(json.dumps({k: point[k] for k in ("t_block", "t_allow", "t_llm",
                                                 "benign_lambda", "dev")}, indent=2))  # fmt: skip


@sweep_app.command("grounding")
def sweep_grounding(split: SplitOption = Split.dev) -> None:
    """Choose min_overlap on dev."""
    _sweep("grounding", split, "20")


# ----------------------------------------------------------------------------- eval


@app.command("eval")
def eval_(
    suite: Annotated[EvalSuite, typer.Option("--suite", help="Which suite to run.")],
    live: Live = False,
    split: SplitOption = Split.test,
) -> None:
    """Run an evaluation suite and write reports/ (replay by default: offline and free)."""
    needs_operating_point = suite in (EvalSuite.injection, EvalSuite.all)
    if needs_operating_point and not (REPORTS / "operating_point.json").is_file():
        _fail("run `guardrails sweep injection` on dev first: the test eval uses the "
              "thresholds it writes to reports/operating_point.json")  # fmt: skip
    if split is Split.dev:
        _fail("reports are computed on the test split (each includes a dev reference); "
              "tune on dev with `guardrails sweep`")  # fmt: skip
    if suite in (EvalSuite.pii, EvalSuite.all):
        from llm_guardrails.eval import run_pii

        r = run_pii.run(reports_dir=REPORTS)
        typer.echo(f"pii: micro F1 {r['test']['shipped']['micro']['f1']} -> reports/pii.md")
    if suite in (EvalSuite.injection, EvalSuite.all):
        from llm_guardrails.eval import run_injection

        settings = _settings()
        r = _run_eval(lambda: run_injection.run(live=live, settings=settings,
                                                reports_dir=REPORTS))  # fmt: skip
        shipped = r["test"]["layers"]["all"]
        typer.echo(f"injection: recall {shipped['attack_recall']['rate']}, benign FPR "
                   f"{shipped['benign_fpr']['rate']} -> reports/injection.md")  # fmt: skip
    if suite not in (EvalSuite.pii, EvalSuite.injection, EvalSuite.all):
        _later("20", f"eval --suite {suite.value}")
    if suite is EvalSuite.all:
        _later("20", "eval --suite schema/grounding/e2e")


# ----------------------------------------------------------------------------- others


@baseline_app.command("bedrock")
def baseline_bedrock(live: Live = False) -> None:
    """Optional comparison with Amazon Bedrock Guardrails (needs a separate yes)."""
    _later("21", "baseline bedrock")


@report_app.command("readme")
def report_readme() -> None:
    """Regenerate the README's result tables from reports/."""
    _later("20", "report readme")


if __name__ == "__main__":
    app()
