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


@dataset_app.command("generate")
def dataset_generate(
    suite: Annotated[Suite, typer.Option("--suite", help="Which dataset.")],
    live: Live = False,
) -> None:
    """Generate an evaluation dataset."""
    _later("18", "dataset generate")


def _dataset_stub(name: str):
    # A factory, not a default argument: Typer would turn a default argument into an option.
    def stub() -> None:
        _later("18", f"dataset {name}")

    return stub


for _cmd in ["validate", "leakcheck", "split", "review-sample", "apply-review"]:
    dataset_app.command(_cmd, help=f"Dataset step: {_cmd} (Task 18).")(_dataset_stub(_cmd))


# ----------------------------------------------------------------------------- sweep


def _sweep(kind: str, split: Split, task: str) -> None:
    if split is Split.test:
        # Tuning on the test split would make every reported number optimistic.
        _fail("sweeps run on the dev split only; the test split is for the final eval")
    _later(task, f"sweep {kind}")


@sweep_app.command("injection")
def sweep_injection(split: SplitOption = Split.dev) -> None:
    """Choose injection thresholds on dev; writes reports/operating_point.json."""
    _sweep("injection", split, "19")


@sweep_app.command("grounding")
def sweep_grounding(split: SplitOption = Split.dev) -> None:
    """Choose min_overlap on dev."""
    _sweep("grounding", split, "20")


# ----------------------------------------------------------------------------- eval


@app.command("eval")
def eval_(
    suite: Annotated[EvalSuite, typer.Option("--suite", help="Which suite to run.")],
    live: Live = False,
    split: SplitOption = Split.dev,
) -> None:
    """Run an evaluation suite (replay by default: offline and free)."""
    needs_operating_point = suite in (EvalSuite.injection, EvalSuite.all)
    if (
        split is Split.test
        and needs_operating_point
        and not (REPORTS / "operating_point.json").is_file()
    ):
        _fail("run `guardrails sweep injection` on dev first: the test eval uses the "
              "thresholds it writes to reports/operating_point.json")  # fmt: skip
    _later("19" if suite in (EvalSuite.injection, EvalSuite.pii) else "20", "eval")


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
